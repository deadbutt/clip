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
    "TTS",
]

KNOWN_TRANSCRIPT_NOISE = {
    "i'm honey",
    "im honey",
}

FILLER_OR_NOISE_WORDS = {
    "ah",
    "eh",
    "er",
    "ha",
    "hm",
    "hmm",
    "huh",
    "la",
    "nah",
    "oh",
    "ooh",
    "uh",
    "uhh",
    "um",
    "umm",
    "whoa",
    "woo",
    "wow",
    "yeah",
    "yo",
}

TRANSLATION_ARTIFACT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"<\s*/?\s*tool[_-]?call\b",
        r"\bDisclaimer\s*:",
        r"\btranslation exercise\b",
        r"\bresponse provided\b",
        r"\bReturn JSON\b",
        r'"translations"\s*:',
        r"```",
    ]
]

_BRACKETED_EFFECT_RE = re.compile(
    r"^\s*[\[(（【{<][^\]\)）】}>]{1,64}(?:sound|sfx|music|applause|laugh|laughter|gift|tts|cheer|alert|donation|sub|resub|音效|音乐|掌声|笑声|礼物|打赏|提示)[^\]\)）】}>]*[\]\)）】}>]\s*$",
    re.IGNORECASE,
)
_SHORT_NOISE_RE = re.compile(r"^[A-Z0-9_#@./\\\-]{1,14}[.!?。！]*$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(slots=True)
class _TranslationUnit:
    id: str
    segments: list[SubtitleSegment]
    start_index: int

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.segments) - 1

    @property
    def speaker(self) -> str:
        return self.segments[0].speaker if self.segments else "S00"

    @property
    def start(self) -> float:
        return self.segments[0].start if self.segments else 0.0

    @property
    def end(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def text(self) -> str:
        return _join_subtitle_text(segment.text for segment in self.segments)


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
        semantic_units: bool = True,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> list[str]:
        items = list(segments)
        skipped = {index: item.text for index, item in enumerate(items) if translation_skip_reason(item.text)}
        if skipped:
            translatable = [item for index, item in enumerate(items) if index not in skipped]
            translated = self._translate_segments_core(
                translatable,
                target_language=target_language,
                batch_size=batch_size,
                context_window=context_window,
                semantic_units=semantic_units,
                progress_callback=None,
            )
            translated_iter = iter(translated)
            output = [
                skipped[index] if index in skipped else next(translated_iter, item.text)
                for index, item in enumerate(items)
            ]
            if progress_callback is not None:
                progress_callback(len(output), len(output), 0, len(output))
            return output
        return self._translate_segments_core(
            items,
            target_language=target_language,
            batch_size=batch_size,
            context_window=context_window,
            semantic_units=semantic_units,
            progress_callback=progress_callback,
        )

    def _translate_segments_core(
        self,
        items: list[SubtitleSegment],
        *,
        target_language: str = "简体中文",
        batch_size: int = 18,
        context_window: int = 2,
        semantic_units: bool = True,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> list[str]:
        if self.provider == "ollama" and batch_size == 18:
            batch_size = 6
        batch_size = max(1, int(batch_size))
        context_window = max(0, int(context_window))
        units = _build_translation_units(items) if semantic_units else _segments_as_units(items)
        translations: list[str] = []
        for batch in _iter_unit_batches(units, batch_size):
            if not batch:
                continue
            start = batch[0].start_index
            end = batch[-1].end_index + 1
            batch_segment_count = sum(len(unit.segments) for unit in batch)
            context_before = items[max(0, start - context_window) : start]
            context_after = items[end : min(len(items), end + context_window)]
            translated_units = self._translate_units_resilient(
                batch,
                target_language=target_language,
                context_before=context_before,
                context_after=context_after,
            )
            for unit_translations in translated_units:
                translations.extend(unit_translations)
            if progress_callback is not None:
                progress_callback(min(len(translations), len(items)), len(items), start, batch_segment_count)
        return translations

    def _translate_units_resilient(
        self,
        units: list[_TranslationUnit],
        *,
        target_language: str,
        context_before: list[SubtitleSegment] | None = None,
        context_after: list[SubtitleSegment] | None = None,
    ) -> list[list[str]]:
        if not units:
            return []
        try:
            translated = self._translate_units_batch(
                units,
                target_language=target_language,
                context_before=context_before,
                context_after=context_after,
            )
            if len(translated) == len(units) and all(
                len(unit_translations) == len(unit.segments)
                for unit, unit_translations in zip(units, translated)
            ):
                return translated
        except _RetryableTranslationError:
            pass
        if len(units) == 1:
            unit = units[0]
            return [
                self._translate_batch_resilient(
                    unit.segments,
                    target_language=target_language,
                    context_before=context_before,
                    context_after=context_after,
                )
            ]
        midpoint = max(1, len(units) // 2)
        left_context_before = context_before or []
        left_context_after = _flatten_unit_segments(units[midpoint:]) + (context_after or [])
        right_context_before = (context_before or []) + _flatten_unit_segments(units[:midpoint])
        right_context_after = context_after or []
        return [
            *self._translate_units_resilient(
                units[:midpoint],
                target_language=target_language,
                context_before=left_context_before[-2:],
                context_after=left_context_after[:2],
            ),
            *self._translate_units_resilient(
                units[midpoint:],
                target_language=target_language,
                context_before=right_context_before[-2:],
                context_after=right_context_after[:2],
            ),
        ]

    def _translate_units_batch(
        self,
        units: list[_TranslationUnit],
        *,
        target_language: str,
        context_before: list[SubtitleSegment] | None = None,
        context_after: list[SubtitleSegment] | None = None,
    ) -> list[list[str]]:
        if all(len(unit.segments) == 1 for unit in units):
            translated = self._translate_batch(
                [unit.segments[0] for unit in units],
                target_language=target_language,
                context_before=context_before,
                context_after=context_after,
            )
            return [[text] for text in translated]

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
                    "Each item may contain multiple subtitle parts that form one sentence or thought. "
                    "Use the full item text to understand grammar, pronouns, tone, and incomplete fragments, "
                    "but return a translation for every original part. "
                    "Do not merge, drop, reorder, or add subtitle parts. "
                    "Translate filler words like well, okay, yeah, and anyways when they carry spoken tone. "
                    "Use context_before and context_after only for understanding; do not translate them. "
                    'Return JSON only in this shape: {"translations":[{"id":"unit_0001",'
                    '"parts":[{"id":"seg_0001","text":"..."}]}]}. '
                    "Every parts array must have the same length and order as the input parts."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "context_before": [_segment_context_item(segment) for segment in (context_before or [])],
                        "items": [_unit_context_item(unit) for unit in units],
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
        try:
            return _parse_unit_translation_array(_chat_content(response), units)
        except RuntimeError as exc:
            raise _RetryableTranslationError(str(exc)) from exc

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
            message = f"Text model request failed with HTTP {exc.code}: {detail}"
            # 5xx/429(服务过载、批次过大、模型加载中)是瞬时故障,标记为可重试,
            # 让二分重试用更小批次自救;4xx 属配置/权限问题,快速失败并保留详情。
            if exc.code >= 500 or exc.code == 429:
                raise _RetryableTranslationError(message) from exc
            raise RuntimeError(message) from exc
        except urllib.error.URLError as exc:
            # 连接被拒(服务未启动)对重试同样无意义:保持快速失败,
            # 让任务带上明确原因结束,而不是二分 n 次后静默降级成全原文。
            raise RuntimeError(f"Failed to connect to translation API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise _RetryableTranslationError(f"Text model request timed out after {self.timeout}s") from exc


def apply_translations(
    segments: Iterable[SubtitleSegment],
    translations: Iterable[str],
    *,
    mode: str = "replace",
) -> list[SubtitleSegment]:
    mode = mode if mode in {"replace", "bilingual"} else "replace"
    output: list[SubtitleSegment] = []
    translated_items = list(translations)
    for index, segment in enumerate(segments):
        translation = translated_items[index] if index < len(translated_items) else segment.text
        translation = clean_translation_text(str(translation or ""), fallback=segment.text)
        if not translation:
            translation = segment.text
        if mode == "replace" or _same_subtitle_text(translation, segment.text):
            text = translation
        else:
            text = f"{translation}\n{segment.text}"
        output.append(
            SubtitleSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                text=text,
                items=segment.items,
            )
        )
    return output


def clean_translation_text(text: str, *, fallback: str = "") -> str:
    text = str(text or "").strip()
    fallback = str(fallback or "").strip()
    if not text:
        return fallback
    if _contains_translation_artifact(text):
        return fallback
    if _looks_like_json_fragment(text):
        return fallback
    if _is_suspicious_expansion(fallback, text):
        return fallback
    return text


def validate_translation_outputs(
    source_segments: Iterable[SubtitleSegment],
    translations: Iterable[str],
) -> list[dict[str, Any]]:
    sources = list(source_segments)
    translated = list(translations)
    issues: list[dict[str, Any]] = []
    if len(translated) != len(sources):
        issues.append(
            {
                "type": "count_mismatch",
                "expected": len(sources),
                "actual": len(translated),
                "message": "Translation count did not match source segment count.",
            }
        )
    for index, text in enumerate(translated):
        value = str(text or "").strip()
        source = sources[index] if index < len(sources) else None
        issue_base = {
            "index": index,
            "id": source.id if source is not None else "",
            "start": source.start if source is not None else None,
            "end": source.end if source is not None else None,
            "source_text": str(source.text or "")[:240] if source is not None else "",
        }
        if value and _contains_translation_artifact(value):
            issues.append({**issue_base, "type": "model_artifact", "text": value[:240]})
        elif value and _looks_like_json_fragment(value):
            issues.append({**issue_base, "type": "json_fragment", "text": value[:240]})
        elif source is not None and _is_suspicious_expansion(source.text, value):
            issues.append(
                {
                    **issue_base,
                    "type": "suspicious_expansion",
                    "source_length": len(str(source.text or "").strip()),
                    "translation_length": len(value),
                    "text": value[:240],
                }
            )
        elif not value and source is not None and str(source.text or "").strip():
            issues.append({**issue_base, "type": "empty_translation"})
    return issues


def collect_pretranslation_skips(segments: Iterable[SubtitleSegment]) -> list[dict[str, Any]]:
    skips: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        reason = translation_skip_reason(segment.text)
        if not reason:
            continue
        skips.append(
            {
                "index": index,
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "type": "pretranslation_skip",
                "reason": reason,
                "text": str(segment.text or "")[:240],
            }
        )
    return skips


def translation_skip_reason(text: str) -> str | None:
    stripped = str(text or "").strip()
    if not stripped:
        return "empty"
    normalized = _normalize_skip_text(stripped)
    if normalized in KNOWN_TRANSCRIPT_NOISE:
        return "known_transcript_noise"
    if _BRACKETED_EFFECT_RE.match(stripped):
        return "bracketed_effect"
    if len(stripped) <= 2:
        return "too_short"
    if _SHORT_NOISE_RE.match(stripped) and not re.search(r"[aeiou]{2,}", stripped, re.IGNORECASE):
        return "short_code_or_noise"
    tokens = [token.lower() for token in _TOKEN_RE.findall(stripped)]
    if _is_repeated_chant(tokens):
        return "repeated_chant"
    if _is_filler_noise(tokens):
        return "filler_noise"
    return None


def _contains_translation_artifact(text: str) -> bool:
    return any(pattern.search(text) for pattern in TRANSLATION_ARTIFACT_PATTERNS)


def _looks_like_json_fragment(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if stripped[0] in "{[" and stripped[-1] not in "。！？.!?)]}】）":
        return True
    if stripped[0] in "{[" and re.search(r'"\w+"\s*:', stripped):
        return True
    return False


def _same_subtitle_text(left: str, right: str) -> bool:
    return str(left or "").strip() == str(right or "").strip()


def _is_suspicious_expansion(source: str, translation: str) -> bool:
    source = str(source or "").strip()
    translation = str(translation or "").strip()
    if not source or not translation:
        return False
    return len(translation) > max(180, len(source) * 4)


def _normalize_skip_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower().strip(".!?。！？"))


def _is_repeated_chant(tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return False
    unique = set(tokens)
    if len(unique) <= 2:
        return True
    if len(unique) <= 3 and len(tokens) >= 6:
        return True
    half = len(tokens) // 2
    if half >= 2 and tokens[:half] == tokens[half : half * 2]:
        return True
    return False


def _is_filler_noise(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if len(tokens) > 12:
        return False
    return all(token in FILLER_OR_NOISE_WORDS for token in tokens)


def _segment_context_item(segment: SubtitleSegment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "speaker": segment.speaker,
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "text": segment.text,
    }


def _unit_context_item(unit: _TranslationUnit) -> dict[str, Any]:
    return {
        "id": unit.id,
        "speaker": unit.speaker,
        "start": round(float(unit.start), 3),
        "end": round(float(unit.end), 3),
        "text": unit.text,
        "parts": [_segment_context_item(segment) for segment in unit.segments],
    }


def _segments_as_units(segments: list[SubtitleSegment]) -> list[_TranslationUnit]:
    return [
        _TranslationUnit(id=f"unit_{index + 1:04d}", segments=[segment], start_index=index)
        for index, segment in enumerate(segments)
    ]


def _build_translation_units(
    segments: list[SubtitleSegment],
    *,
    max_gap_sec: float = 1.2,
    max_segments: int = 4,
    max_chars: int = 260,
) -> list[_TranslationUnit]:
    units: list[_TranslationUnit] = []
    current: list[SubtitleSegment] = []
    current_start = 0
    for index, segment in enumerate(segments):
        text = str(segment.text or "").strip()
        if not text:
            if current:
                units.append(_make_unit(len(units), current, current_start))
                current = []
            units.append(_make_unit(len(units), [segment], index))
            continue
        if current and not _should_merge_segment(
            current,
            segment,
            max_gap_sec=max_gap_sec,
            max_segments=max_segments,
            max_chars=max_chars,
        ):
            units.append(_make_unit(len(units), current, current_start))
            current = []
        if not current:
            current_start = index
        current.append(segment)
    if current:
        units.append(_make_unit(len(units), current, current_start))
    return units


def _make_unit(index: int, segments: list[SubtitleSegment], start_index: int) -> _TranslationUnit:
    return _TranslationUnit(id=f"unit_{index + 1:04d}", segments=list(segments), start_index=start_index)


def _should_merge_segment(
    current: list[SubtitleSegment],
    segment: SubtitleSegment,
    *,
    max_gap_sec: float,
    max_segments: int,
    max_chars: int,
) -> bool:
    previous = current[-1]
    if previous.speaker != segment.speaker:
        return False
    if float(segment.start) - float(previous.end) > max_gap_sec:
        return False
    if len(current) >= max_segments:
        return False
    combined_text = _join_subtitle_text([*(item.text for item in current), segment.text])
    if len(combined_text) > max_chars:
        return False
    previous_text = str(previous.text or "").strip()
    if not previous_text:
        return True
    return not _ends_sentence(previous_text)


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    return bool(re.search(r"[.!?。！？…]+[\"')\]}）】》]*$", stripped))


def _join_subtitle_text(parts: Iterable[str]) -> str:
    output = ""
    for raw in parts:
        text = str(raw or "").strip()
        if not text:
            continue
        if not output:
            output = text
        elif re.match(r"^[,.;:!?，。！？、；：）】》\]\)]", text):
            output += text
        elif output.endswith(("(", "[", "{", "（", "【", "《")):
            output += text
        else:
            output += " " + text
    return output


def _iter_unit_batches(units: list[_TranslationUnit], batch_size: int) -> Iterable[list[_TranslationUnit]]:
    batch_size = max(1, int(batch_size))
    batch: list[_TranslationUnit] = []
    segment_count = 0
    for unit in units:
        unit_size = max(1, len(unit.segments))
        if batch and segment_count + unit_size > batch_size:
            yield batch
            batch = []
            segment_count = 0
        batch.append(unit)
        segment_count += unit_size
    if batch:
        yield batch


def _flatten_unit_segments(units: Iterable[_TranslationUnit]) -> list[SubtitleSegment]:
    return [segment for unit in units for segment in unit.segments]


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


def _parse_json_fragment(content: str, *, object_only: bool = False) -> Any:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pattern = r"\{[\s\S]*\}" if object_only else r"(\{[\s\S]*\}|\[[\s\S]*\])"
        match = re.search(pattern, content)
        if not match:
            raise RuntimeError("Model response was not JSON.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Model response contained malformed JSON.") from exc


def _parse_translation_array(content: str) -> list[str]:
    data = _parse_json_fragment(content)
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


def _parse_unit_translation_array(content: str, units: list[_TranslationUnit]) -> list[list[str]]:
    data = _parse_json_fragment(content)
    if isinstance(data, dict):
        data = data.get("translations") or data.get("translated") or data.get("items")
    if not isinstance(data, list):
        raise RuntimeError("Unit translation response was not a JSON array.")
    if len(data) != len(units):
        raise RuntimeError("Unit translation count did not match input items.")
    return [_parse_unit_translation_item(item, unit) for item, unit in zip(data, units)]


def _parse_unit_translation_item(item: Any, unit: _TranslationUnit) -> list[str]:
    expected = len(unit.segments)
    parts = None
    if isinstance(item, dict):
        parts = item.get("parts") or item.get("translations") or item.get("items")
    elif isinstance(item, list):
        parts = item
    if expected == 1:
        if isinstance(parts, list) and parts:
            return [_extract_translation_text(parts[0])]
        return [_extract_translation_text(item)]
    if not isinstance(parts, list):
        raise RuntimeError("Multi-part translation item did not contain a parts array.")
    if len(parts) != expected:
        raise RuntimeError("Multi-part translation count did not match input parts.")
    by_id: dict[str, str] = {}
    ordered: list[str] = []
    for part in parts:
        text = _extract_translation_text(part)
        if isinstance(part, dict) and part.get("id") is not None:
            by_id[str(part.get("id"))] = text
        ordered.append(text)
    if by_id:
        return [by_id.get(segment.id, ordered[index]) for index, segment in enumerate(unit.segments)]
    return ordered


def _extract_translation_text(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("text") or item.get("translation") or item.get("translated")
        return str(value or "").strip()
    return str(item or "").strip()


def _parse_clip_ranking(content: str) -> list[dict[str, Any]]:
    data = _parse_json_fragment(content, object_only=True)
    selected = data.get("selected") if isinstance(data, dict) else None
    if not isinstance(selected, list):
        raise RuntimeError("Highlight response did not contain a selected list.")
    return [item for item in selected if isinstance(item, dict)]
