# -*- coding: utf-8 -*-
"""LLM 字幕校对引擎（两遍分离架构）。

Pass 1: 局部润色 —— 小窗口(10 目标 + 前后各 2 上下文)，1:1 替换约束，
        输出 ID 集合必须与目标完全一致且文本长度 <= 1.5x，违规丢弃保原文。
Pass 2: 结构分析 —— 全片只读，LLM 只输出标注(术语表/合并建议/说话人质疑)，
        术语表由 Python 确定性试应用统计命中，LLM 永远不重写全文。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from moss_transcribe_diarize.subtitle import SubtitleSegment

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

PASS2_SYSTEM = """You analyze a full ASR transcript of a video for structural issues.
This is a READ-ONLY analysis: you never rewrite the transcript.
Return JSON only, in this shape:
{"term_corrections":[{"wrong":"...","right":"..."}],
 "merge_suggestions":[{"id":"seg_0001","with_next":true,"reason":"..."}],
 "speaker_questions":[{"id":"seg_0001","current":"S01","suspect":"S02","reason":"..."}]}
Rules:
- term_corrections: consistent mis-transcriptions of names/terms that appear MULTIPLE times and are clearly wrong. Do NOT list terms you are unsure about, and NEVER list a correction where wrong == right.
- merge_suggestions: ONLY the most obvious cases where one sentence was split across two adjacent same-speaker segments mid-clause. Maximum 15 suggestions, ranked by confidence. If the transcript is mostly fine, return fewer or none.
- speaker_questions: only if conversational evidence strongly suggests wrong speaker label.
- Be very conservative: empty lists are fine. Keep each reason under 15 words.
- Output ONLY the JSON object, nothing else. No markdown fences."""


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
        merges = data.get("merge_suggestions") or []
        speakers = data.get("speaker_questions") or []
        return {
            "term_corrections": [t for t in terms if isinstance(t, dict)],
            "merge_suggestions": [m for m in merges if isinstance(m, dict)][:15],
            "speaker_questions": [q for q in speakers if isinstance(q, dict)][:15],
        }

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
                new_text, count = pattern.subn(right, segment.text)
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
        pass2 = self._run_pass2(items)
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
            "reference": {
                "merge_suggestions": pass2["merge_suggestions"],
                "speaker_questions": pass2["speaker_questions"],
            },
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
