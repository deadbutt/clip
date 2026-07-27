from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from moss_transcribe_diarize.subtitle import SubtitleSegment


class _RetryableTranslationError(RuntimeError):
    pass


PROTECTED_TERMS = [
    "Twitter",
    "Twitch",
    "OBS",
    "Vidal",
]


@dataclass(slots=True)
class TextTranslator:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 600.0
    provider: str = "openai"
    protected_terms: tuple[str, ...] = tuple(PROTECTED_TERMS)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "available": True,
            "backend": "ollama" if self.provider == "ollama" else "vllm-chat",
            "base_url": self.base_url,
            "model": self.model,
            "protected_terms": list(self.protected_terms),
        }

    def translate_segments(
        self,
        segments: Iterable[SubtitleSegment],
        *,
        target_language: str = "简体中文",
        batch_size: int = 18,
        context_window: int = 2,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> list[str]:
        items = list(segments)
        if self.provider == "ollama" and batch_size == 18:
            batch_size = 6
        batch_size = max(1, int(batch_size))
        context_window = max(0, int(context_window))
        translations: list[str] = []
        for start in range(0, len(items), batch_size):
            end = min(len(items), start + batch_size)
            batch = items[start:end]
            context_before = items[max(0, start - context_window) : start]
            context_after = items[end : min(len(items), end + context_window)]
            translations.extend(
                self._translate_batch_resilient(
                    batch,
                    target_language=target_language,
                    context_before=context_before,
                    context_after=context_after,
                )
            )
            if progress_callback is not None:
                progress_callback(min(len(translations), len(items)), len(items), start, len(batch))
        return translations

    def _translate_batch_resilient(
        self,
        segments: list[SubtitleSegment],
        *,
        target_language: str,
        context_before: list[SubtitleSegment] | None = None,
        context_after: list[SubtitleSegment] | None = None,
    ) -> list[str]:
        if not segments:
            return []
        try:
            translated = self._translate_batch(
                segments,
                target_language=target_language,
                context_before=context_before,
                context_after=context_after,
            )
            if len(translated) == len(segments):
                return translated
            if len(segments) == 1 and translated:
                return [translated[0]]
        except _RetryableTranslationError:
            if len(segments) == 1:
                return [segments[0].text]
        if len(segments) == 1:
            return [segments[0].text]
        midpoint = max(1, len(segments) // 2)
        left_context_before = context_before or []
        left_context_after = segments[midpoint:] + (context_after or [])
        right_context_before = (context_before or []) + segments[:midpoint]
        right_context_after = context_after or []
        return [
            *self._translate_batch_resilient(
                segments[:midpoint],
                target_language=target_language,
                context_before=left_context_before[-2:],
                context_after=left_context_after[:2],
            ),
            *self._translate_batch_resilient(
                segments[midpoint:],
                target_language=target_language,
                context_before=right_context_before[-2:],
                context_after=right_context_after[:2],
            ),
        ]

    def _translate_batch(
        self,
        segments: list[SubtitleSegment],
        *,
        target_language: str,
        context_before: list[SubtitleSegment] | None = None,
        context_after: list[SubtitleSegment] | None = None,
    ) -> list[str]:
        protected = ", ".join(term for term in self.protected_terms if term.strip())
        protected_instruction = (
            f"Do not translate or rewrite these protected terms: {protected}. "
            if protected
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a subtitle translator. Translate spoken subtitles into natural "
                    f"{target_language}. Preserve names, memes, and streamer terms when possible. "
                    f"{protected_instruction}"
                    "Translate the full sentence, including filler words like well, okay, yeah, and anyways; "
                    "do not leave English words unless they are protected terms, names, or established platform/tool terms. "
                    "Use context_before and context_after only to understand pronouns, names, tone, and incomplete sentences. "
                    "Do not translate context_before or context_after. "
                    'Return JSON only. Preferred shape: {"translations":["..."]}. '
                    "The translations array must have the same length and order as items."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "context_before": [_segment_context_item(segment) for segment in (context_before or [])],
                        "items": [_segment_context_item(segment) for segment in segments],
                        "context_after": [_segment_context_item(segment) for segment in (context_after or [])],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        if self.provider == "ollama":
            response = self._post_json(
                self._ollama_chat_url(),
                {
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0},
                    "messages": messages,
                },
            )
        else:
            response = self._post_json(
                self._chat_url(),
                {
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": messages,
                },
            )
        content = _chat_content(response)
        try:
            return _parse_translation_array(content)
        except RuntimeError as exc:
            raise _RetryableTranslationError(str(exc)) from exc

    def rank_clip_candidates(self, candidates: Iterable[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
        items = list(candidates)
        if not items:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You select highlights from a long-form transcript for short video clips. "
                    "Judge semantic quality, not keyword count. Prefer self-contained excerpts with a strong opening, "
                    "clear development and payoff, emotional or informational value, and little dependency on missing context. "
                    "Avoid repetitive or substantially overlapping choices. Return JSON only in this shape: "
                    '{"selected":[{"id":"clip_001","score":92,"title":"short Chinese title",'
                    '"reason":"specific Chinese reason"}]}. Use only provided ids.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
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
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        if self.provider == "ollama":
            response = self._post_json(
                self._ollama_chat_url(),
                {
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.15},
                    "messages": messages,
                },
            )
        else:
            response = self._post_json(
                self._chat_url(),
                {
                    "model": self.model,
                    "temperature": 0.15,
                    "response_format": {"type": "json_object"},
                    "messages": messages,
                },
            )
        ranked = _parse_clip_ranking(_chat_content(response))
        by_id = {str(item.get("id")): dict(item) for item in items}
        output: list[dict[str, Any]] = []
        for choice in ranked:
            candidate = by_id.get(str(choice.get("id") or ""))
            if candidate is None:
                continue
            candidate["score"] = max(0.0, min(100.0, float(choice.get("score") or 0.0)))
            candidate["title"] = str(choice.get("title") or candidate.get("title") or "未命名片段").strip()
            candidate["reason"] = str(choice.get("reason") or "模型精选").strip()
            candidate["selection_method"] = "model"
            output.append(candidate)
            if len(output) >= max(1, int(limit)):
                break
        if not output:
            raise RuntimeError("Highlight model did not return any valid candidate ids.")
        return output

    def _chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _ollama_chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/api/chat"):
            return base
        if base.endswith("/api"):
            return base + "/chat"
        return base + "/api/chat"

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
            raise RuntimeError(f"Text model request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to connect to translation API: {exc.reason}") from exc


def apply_translations(
    segments: Iterable[SubtitleSegment],
    translations: Iterable[str],
    *,
    mode: str = "replace",
) -> list[SubtitleSegment]:
    mode = mode if mode in {"replace", "bilingual"} else "replace"
    output: list[SubtitleSegment] = []
    for segment, translation in zip(segments, translations):
        translation = str(translation or "").strip()
        if not translation:
            translation = segment.text
        text = translation if mode == "replace" else f"{translation}\n{segment.text}"
        output.append(
            SubtitleSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                text=text,
            )
        )
    return output


def _segment_context_item(segment: SubtitleSegment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "speaker": segment.speaker,
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "text": segment.text,
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


def _parse_translation_array(content: str) -> list[str]:
    content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
        if not match:
            raise RuntimeError("Translation response was not a JSON array.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Translation response contained malformed JSON.") from exc
    if isinstance(data, dict):
        data = data.get("translations") or data.get("translated") or data.get("items")
    if not isinstance(data, list):
        raise RuntimeError("Translation response was not a JSON array.")
    output: list[str] = []
    for item in data:
        if isinstance(item, dict):
            value = item.get("text") or item.get("translation") or item.get("translated")
            output.append(str(value or "").strip())
        else:
            output.append(str(item).strip())
    return output


def _parse_clip_ranking(content: str) -> list[dict[str, Any]]:
    content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise RuntimeError("Highlight response was not a JSON object.")
        data = json.loads(match.group(0))
    selected = data.get("selected") if isinstance(data, dict) else None
    if not isinstance(selected, list):
        raise RuntimeError("Highlight response did not contain a selected list.")
    return [item for item in selected if isinstance(item, dict)]
