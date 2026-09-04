# -*- coding: utf-8 -*-
"""LLM 字幕校对引擎（两遍分离架构）。

Pass 1: 局部润色 —— 小窗口(10 目标 + 前后各 2 上下文)，1:1 替换约束，
        输出 ID 集合必须与目标完全一致且文本长度 <= 1.5x，违规丢弃保原文。
Pass 2: 结构分析 —— 全片只读，LLM 只输出标注(术语表/合并建议/说话人质疑)，
        术语表由 Python 确定性试应用统计命中，LLM 永远不重写全文。
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from moss_transcribe_diarize.subtitle import SubtitleSegment

logger = logging.getLogger(__name__)

MAX_TEXT_RATIO = 1.5
WINDOW_TARGETS = 10
WINDOW_CONTEXT = 2
MAX_WORKERS = 4

PASS1_SYSTEM = """You are a subtitle proofreader for ASR (speech-to-text) output.
Fix ONLY these kinds of errors in the TARGET segments:
- spelling/typo errors (including missing apostrophes like dont -> don't)
- punctuation and capitalization
- obvious grammar slips (get's -> gets, five more minute -> five more minutes)
- duplicated stutters (I I don't -> I don't) ONLY when clearly accidental
Do NOT rewrite, rephrase, shorten, merge or expand meaning. Keep the original language.
Keep speaker labels and segment ids exactly as given.
Context segments (ctx) are read-only: never include them in output.
Return ONLY a JSON array, one object per TARGET segment, in order:
[{"id":"seg_0001","text":"..."}]
The output array must contain exactly the same ids as the TARGET segments."""

PASS2_SYSTEM = """You analyze a full ASR transcript of a video for recurring terminology errors.
This is a READ-ONLY analysis: you never rewrite the transcript.
Return JSON only, in this shape:
{"term_corrections":[{"wrong":"...","right":"..."}]}
Rules:
- term_corrections: consistent mis-transcriptions of names/terms that appear MULTIPLE times and are clearly wrong. Do NOT list terms you are unsure about, and NEVER list a correction where wrong == right.
- Be very conservative: an empty list is fine.
- Output ONLY the JSON object, nothing else. No markdown fences."""

ALIGNMENT_SYSTEM = """You compare numbered subtitle segments in the source language with their Chinese translations.
For each pair decide whether the Chinese faithfully conveys the source meaning.
Flag ONLY clear problems:
- omission: a meaningful part of the source text is missing in the Chinese
- addition: the Chinese states content that is not in the source text
- mistranslation: wrong meaning, wrong name/number/negation, or opposite meaning
- terminology: a recurring name/term is translated inconsistently with the rest
Do NOT flag style, tone, naturalness, or acceptable paraphrases. Do NOT flag untranslated
noise or bracketed effects like [Music]. When unsure, do not flag.
For each flagged pair also provide "suggested": a corrected full Chinese translation of that
pair, fixing the flagged problem, in the same subtitle style as the original Chinese.
Return JSON only: {"issues":[{"n":<pair number>,"type":"omission|addition|mistranslation|terminology","note":"<Chinese, under 20 words>","suggested":"<corrected Chinese translation>"}]}
If everything is fine return {"issues":[]}."""

CLIP_RANK_SYSTEM = """You select highlights from a long-form transcript for short video clips.
Judge semantic quality, not keyword count. Prefer self-contained excerpts with a strong opening,
clear development and payoff, emotional or informational value, and little dependency on missing context.
Avoid repetitive or substantially overlapping choices. Return JSON only in this shape:
{"selected":[{"id":"clip_001","score":92,"title":"short Chinese title","reason":"specific Chinese reason"}]}.
Use only provided ids."""


@dataclass(slots=True)
class Proofreader:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    provider: str = "openai"
    timeout: float = 300.0
    disable_thinking: bool = False

    def runtime_info(self) -> dict[str, Any]:
        return {
            "available": True,
            "backend": self.provider,
            "base_url": self.base_url,
            "model": self.model,
        }

    # ------------------------------------------------------------------ API

    def _chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if self.provider == "ollama":
            if base.endswith("/api/chat"):
                return base
            if base.endswith("/api"):
                return base + "/chat"
            return base + "/api/chat"
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to connect to LLM API: {exc.reason}") from exc

    def _chat(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.provider == "ollama":
            payload["stream"] = False
            payload["format"] = "json"
            payload.pop("temperature", None)
            payload["options"] = {"temperature": temperature}
        if self.disable_thinking and self.provider == "openai":
            payload["thinking"] = {"type": "disabled"}
        response = self._post_json(self._chat_url(), payload)
        return _chat_content(response)

    def test_connection(self) -> dict[str, Any]:
        started = time.time()
        try:
            content = self._chat("You are a connectivity probe.", "Reply with exactly: ok")
            latency = int((time.time() - started) * 1000)
            return {"ok": True, "message": f"连接成功 ({latency} ms)，模型回复: {content[:40]!r}", "latency_ms": latency}
        except Exception as exc:
            return {"ok": False, "message": str(exc), "latency_ms": int((time.time() - started) * 1000)}

    # --------------------------------------------------------------- clips

    def rank_clip_candidates(
        self, candidates: Iterable[dict[str, Any]], *, limit: int = 8
    ) -> list[dict[str, Any]]:
        """用激活的 LLM 配置为精华切片候选打分排序。"""
        items = list(candidates)
        if not items:
            return []
        user_payload = {
            "max_results": max(1, int(limit)),
            "candidates": [
                {
                    "id": item.get("id"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "duration": item.get("duration"),
                    "transcript": str(item.get("text") or "")[:1200],
                }
                for item in items
            ],
        }
        content = self._chat(CLIP_RANK_SYSTEM, json.dumps(user_payload, ensure_ascii=False), temperature=0.15)
        data = _parse_json_object(content)
        selected = data.get("selected") if isinstance(data, dict) else None
        if not isinstance(selected, list):
            raise RuntimeError("Highlight model did not return a valid selected list.")
        by_id = {str(item.get("id")): dict(item) for item in items}
        output: list[dict[str, Any]] = []
        for choice in selected:
            if not isinstance(choice, dict):
                continue
            candidate = by_id.get(str(choice.get("id") or ""))
            if candidate is None:
                continue
            try:
                candidate["score"] = max(0.0, min(100.0, float(choice.get("score") or 0.0)))
            except (TypeError, ValueError):
                candidate["score"] = 0.0
            candidate["title"] = str(choice.get("title") or candidate.get("title") or "未命名片段").strip()
            candidate["reason"] = str(choice.get("reason") or "模型精选").strip()
            candidate["selection_method"] = "model"
            output.append(candidate)
            if len(output) >= max(1, int(limit)):
                break
        if not output:
            raise RuntimeError("Highlight model did not return any valid candidate ids.")
        return output

    # --------------------------------------------------------------- Pass 1

    def _pass1_window(self, items: list[SubtitleSegment], start: int, end: int) -> dict[str, str]:
        """Run one proofreading window. Returns {seg_id: new_text} for changed segments only."""
        targets = items[start:end]
        before = items[max(0, start - WINDOW_CONTEXT) : start]
        after = items[end : min(len(items), end + WINDOW_CONTEXT)]

        def entry(segment: SubtitleSegment, role: str) -> dict[str, Any]:
            return {
                "id": segment.id,
                "speaker": segment.speaker,
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment.text,
                "role": role,
            }

        payload = {
            "ctx_before": [entry(s, "context") for s in before],
            "targets": [entry(s, "target") for s in targets],
            "ctx_after": [entry(s, "context") for s in after],
        }
        content = self._chat(PASS1_SYSTEM, json.dumps(payload, ensure_ascii=False))
        data = _parse_json_array(content)
        expected_ids = {s.id for s in targets}
        got_ids = set()
        changes: dict[str, str] = {}
        by_id = {s.id: s for s in targets}
        for item in data:
            if not isinstance(item, dict):
                continue
            seg_id = str(item.get("id") or "")
            text = str(item.get("text") or "").strip()
            if seg_id not in expected_ids or seg_id in got_ids or not text:
                continue
            got_ids.add(seg_id)
            original = by_id[seg_id]
            if text == original.text.strip():
                continue
            if len(text) > MAX_TEXT_RATIO * max(len(original.text.strip()), 1):
                continue  # too aggressive, reject
            changes[seg_id] = text
        if got_ids != expected_ids:
            # Model dropped or invented ids: keep only fully-trusted windows' safe subset.
            # Windows that failed the id contract contribute nothing.
            return {}
        return changes

    def _run_pass1(
        self,
        items: list[SubtitleSegment],
        progress_callback: Callable[[int, int], None] | None,
    ) -> tuple[dict[str, str], list[str]]:
        windows: list[tuple[int, int]] = []
        start = 0
        while start < len(items):
            end = min(len(items), start + WINDOW_TARGETS)
            windows.append((start, end))
            start = end
        fixed: dict[str, str] = {}
        rejected: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._pass1_window, items, w_start, w_end): (w_start, w_end)
                for w_start, w_end in windows
            }
            for future in as_completed(futures):
                w_start, w_end = futures[future]
                try:
                    changes = future.result()
                except Exception:
                    rejected.append(f"window[{w_start}:{w_end}]")
                    changes = {}
                fixed.update(changes)
                done += 1
                if progress_callback:
                    progress_callback(done, len(windows))
        return fixed, rejected

    # --------------------------------------------------------------- Pass 2

    def _run_pass2(self, items: list[SubtitleSegment]) -> dict[str, Any]:
        lines = [
            f"[{s.id}] ({s.speaker}) {s.start:.1f}-{s.end:.1f} {s.text}"
            for s in items
        ]
        content = self._chat(PASS2_SYSTEM, "\n".join(lines), temperature=0.1)
        data = _parse_json_object(content)
        terms = data.get("term_corrections") or []
        return {"term_corrections": [t for t in terms if isinstance(t, dict)]}

    @staticmethod
    def _apply_terms(items: list[SubtitleSegment], terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deterministically trial-apply each term mapping; keep only terms with hits."""
        results: list[dict[str, Any]] = []
        for term in terms:
            wrong = str(term.get("wrong") or "").strip()
            right = str(term.get("right") or "").strip()
            if not wrong or not right or wrong.lower() == right.lower():
                continue
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            hits = 0
            previews: list[dict[str, Any]] = []
            for segment in items:
                # 用 lambda 提供 replacement:LLM 返回的 right 含 \ 或 \g<...>
                # 时按替换模板解析会抛 re.error 或错误展开。
                new_text, count = pattern.subn(lambda _match: right, segment.text)
                if count and new_text != segment.text:
                    hits += count
                    if len(previews) < 3:
                        previews.append({"id": segment.id, "original": segment.text, "corrected": new_text})
            if hits:
                results.append({
                    "wrong": wrong,
                    "right": right,
                    "hits": hits,
                    "previews": previews,
                })
        return results

    # ------------------------------------------------------------- alignment

    def _alignment_window(
        self, pairs: list[dict[str, Any]], start: int, end: int
    ) -> list[dict[str, Any]]:
        """Run one comparison window. Returns raw issue dicts from the model."""
        lines = []
        for offset, pair in enumerate(pairs[start:end], start=1):
            lines.append(f"{offset}. SRC: {pair['source_text']}")
            lines.append(f"   ZH: {pair['translated_text']}")
        content = self._chat(ALIGNMENT_SYSTEM, "\n".join(lines), temperature=0.0)
        data = _parse_json_object(content)
        issues_raw = data.get("issues")
        if not isinstance(issues_raw, list):
            return []
        count = end - start
        seen: set[int] = set()
        output: list[dict[str, Any]] = []
        # suggested 若与原译文完全一致说明模型没给出可用修正,跳过该条。
        current_by_offset = {offset: pairs[start + offset - 1]["translated_text"] for offset in range(1, count + 1)}
        for item in issues_raw:
            if not isinstance(item, dict):
                continue
            try:
                n = int(item.get("n"))
            except (TypeError, ValueError):
                continue
            if not 1 <= n <= count or n in seen:
                continue
            issue_type = str(item.get("type") or "").strip().lower()
            if issue_type not in {"omission", "addition", "mistranslation", "terminology"}:
                continue
            note = str(item.get("note") or "").strip()[:120]
            suggested = str(item.get("suggested") or "").strip()
            current_text = current_by_offset.get(n, "")
            if not suggested or suggested == current_text:
                continue
            seen.add(n)
            output.append({"index": start + n - 1, "type": issue_type, "note": note, "suggested": suggested})
        return output

    def check_alignment(
        self,
        pairs: Iterable[dict[str, Any]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """成对比对源文与译文,返回带原始 pair 信息的标注(只读,不改文本)。"""
        items = [pair for pair in pairs if str(pair.get("source_text") or "").strip()]
        if not items:
            return []
        windows: list[tuple[int, int]] = []
        start = 0
        while start < len(items):
            end = min(len(items), start + WINDOW_TARGETS)
            windows.append((start, end))
            start = end
        issues: list[dict[str, Any]] = []
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._alignment_window, items, w_start, w_end): (w_start, w_end)
                for w_start, w_end in windows
            }
            for future in as_completed(futures):
                w_start, w_end = futures[future]
                try:
                    window_issues = future.result()
                except Exception:
                    logger.exception("Alignment window [%s:%s] failed", w_start, w_end)
                    window_issues = []
                for issue in window_issues:
                    pair = items[issue["index"]]
                    issues.append(
                        {
                            "id": pair.get("id"),
                            "index": pair.get("index"),
                            "start": pair.get("start"),
                            "end": pair.get("end"),
                            "type": issue["type"],
                            "note": issue["note"],
                            "suggested": issue["suggested"],
                            "source_text": pair.get("source_text"),
                            "translated_text": pair.get("translated_text"),
                        }
                    )
                done += 1
                if progress_callback:
                    progress_callback(done, len(windows))
        issues.sort(key=lambda item: (item.get("index") is None, item.get("index") or 0))
        return issues

    # ----------------------------------------------------------------- main

    def proofread(
        self,
        segments: Iterable[SubtitleSegment],
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        items = list(segments)
        if not items:
            return {"suggestions": [], "term_corrections": [], "reference": {}, "usage": {}, "elapsed_sec": 0.0}
        started = time.time()

        def pass1_progress(done: int, total: int) -> None:
            if progress_callback:
                progress_callback("pass1", done, total)

        fixed, rejected = self._run_pass1(items, pass1_progress)
        if progress_callback:
            progress_callback("pass2", 0, 1)
        try:
            pass2 = self._run_pass2(items)
        except Exception:
            # Pass 2 是长视频单请求,最容易超时/爆上下文;它只产术语修正,
            # 失败不应作废 Pass 1 已完成的全部修正。
            logger.exception("Proofread pass 2 failed; keeping pass 1 results only")
            pass2 = {"term_corrections": []}
        term_results = self._apply_terms(items, pass2["term_corrections"])
        if progress_callback:
            progress_callback("pass2", 1, 1)

        typo_ids = set(fixed)
        suggestions = [
            {"id": segment.id, "original": segment.text, "corrected": fixed[segment.id], "type": "typo"}
            for segment in items
            if segment.id in typo_ids
        ]
        return {
            "suggestions": suggestions,
            "term_corrections": term_results,
            "usage": {
                "typo_changes": len(fixed),
                "rejected_windows": rejected,
                "segments": len(items),
            },
            "elapsed_sec": round(time.time() - started, 1),
        }


def _chat_content(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    content = response.get("text")
    if isinstance(content, str):
        return content.strip()
    return ""


def _strip_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _parse_json_array(content: str) -> list[Any]:
    content = _strip_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        # some models wrap the array: {"corrections": [...]}
        for value in data.values():
            if isinstance(value, list):
                return value
        return []
    return data if isinstance(data, list) else []


def _parse_json_object(content: str) -> dict[str, Any]:
    content = _strip_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}
