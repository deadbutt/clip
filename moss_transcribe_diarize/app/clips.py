from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from moss_transcribe_diarize.subtitle import SubtitleSegment

HOOK_PATTERNS = [
    "why",
    "how",
    "but",
    "actually",
    "problem",
    "key",
    "important",
    "secret",
    "mistake",
    "result",
    "because",
    "wait",
    "no way",
    "what happened",
    "the point is",
    "所以",
    "但是",
    "其实",
    "关键",
    "问题",
    "为什么",
    "怎么",
    "结果",
    "重点",
    "核心",
    "离谱",
    "真的假的",
    "不会吧",
    "破防",
    "绷不住",
    "等一下",
    "没想到",
    "到底",
]
ENDING_PATTERNS = [
    "so",
    "therefore",
    "because",
    "that's why",
    "in short",
    "总结",
    "所以",
    "因此",
    "这就是",
    "结论",
    "最后",
]


@dataclass(slots=True)
class ClipCandidate:
    id: str
    start: float
    end: float
    score: float
    title: str
    reason: str
    text: str
    segment_ids: list[str]
    selection_method: str = "rules"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = self.duration
        return data


def generate_clip_candidates(
    segments: Iterable[SubtitleSegment | dict],
    *,
    min_duration: float = 45.0,
    target_duration: float = 120.0,
    max_duration: float | None = 300.0,
    limit: int = 24,
    merge_expansion_limit: float | None = None,
) -> list[ClipCandidate]:
    prepared = _prepare_segments(segments)
    if not prepared:
        return []

    candidates: list[ClipCandidate] = []
    for start_index, first in enumerate(prepared):
        window: list[SubtitleSegment] = []
        for segment in prepared[start_index:]:
            if window and segment.start - window[-1].end > 12.0:
                break
            window.append(segment)
            duration = window[-1].end - first.start
            if duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                break
            candidates.append(_score_window(window, target_duration=target_duration))

    deduped = _dedupe_candidates(
        candidates,
        max_duration=max_duration,
        merge_expansion_limit=merge_expansion_limit,
    )
    deduped.sort(key=lambda item: item.score, reverse=True)
    for index, item in enumerate(deduped[:limit], start=1):
        item.id = f"clip_{index:03d}"
    return deduped[:limit]


def _prepare_segments(segments: Iterable[SubtitleSegment | dict]) -> list[SubtitleSegment]:
    prepared: list[SubtitleSegment] = []
    for index, item in enumerate(segments, start=1):
        segment = item if isinstance(item, SubtitleSegment) else SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}")
        text = str(segment.text or "").strip()
        if not text or segment.end <= segment.start:
            continue
        prepared.append(
            SubtitleSegment(
                id=segment.id or f"seg_{index:04d}",
                start=float(segment.start),
                end=float(segment.end),
                speaker=segment.speaker or "S00",
                text=text,
            )
        )
    prepared.sort(key=lambda item: (item.start, item.end))
    return prepared


def _score_window(window: list[SubtitleSegment], *, target_duration: float) -> ClipCandidate:
    text = _join_text(segment.text for segment in window)
    duration = max(0.1, window[-1].end - window[0].start)
    hook_hits = _pattern_hits(text, HOOK_PATTERNS)
    ending_hits = _pattern_hits(window[-1].text, ENDING_PATTERNS)
    question_bonus = 8.0 if re.search(r"[?？]", text) else 0.0
    density = min(26.0, len(text) / max(duration, 1.0) * 7.0)
    duration_score = max(0.0, 22.0 - abs(duration - target_duration) / max(target_duration, 1.0) * 22.0)
    punctuation_bonus = min(8.0, len(re.findall(r"[!！?？。]", text)) * 1.4)
    short_pause_bonus = min(6.0, _pause_count(window) * 1.5)
    completeness = 10.0 if _looks_complete(window[-1].text) else 3.0
    score = round(
        20.0
        + hook_hits * 4.0
        + ending_hits * 3.0
        + question_bonus
        + punctuation_bonus
        + short_pause_bonus
        + density
        + duration_score
        + completeness,
        2,
    )
    title = _title_from_text(text)
    reason = _reason(hook_hits, ending_hits, question_bonus, duration)
    return ClipCandidate(
        id="clip",
        start=round(window[0].start, 2),
        end=round(window[-1].end, 2),
        score=score,
        title=title,
        reason=reason,
        text=text[:700],
        segment_ids=[segment.id for segment in window],
    )


def _dedupe_candidates(
    candidates: list[ClipCandidate], *, max_duration: float | None = 300.0, merge_expansion_limit: float | None = None
) -> list[ClipCandidate]:
    """重叠候选(>65%)合并为取时间外沿的一条更完整片段；合并超时长上限才丢弃（上限为 None 时不限）。"""
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    output: list[ClipCandidate] = []
    while ordered:
        candidate = ordered.pop(0)
        merged_into: int | None = None
        blocked = False
        for i, kept in enumerate(output):
            intersection = max(0.0, min(candidate.end, kept.end) - max(candidate.start, kept.start))
            shorter = max(1.0, min(candidate.duration, kept.duration))
            # 滑动窗口通常只错开一小段时间；用较短片段作为基准，
            # 55% 以上重叠就视为同一候选主题，避免重复卡片堆积。
            if intersection / shorter <= 0.55:
                continue
            merged_start = min(candidate.start, kept.start)
            merged_end = max(candidate.end, kept.end)
            expansion = merged_end - merged_start - min(candidate.duration, kept.duration)
            if merge_expansion_limit is not None and (
                expansion > merge_expansion_limit or merged_end - merged_start > merge_expansion_limit * 2
            ):
                # 相似的长窗口可能只是包住了同一个高分片段；直接丢弃
                # 外围候选，避免连续滑窗链式合并成整部视频。
                blocked = True
                continue
            if max_duration is not None and merged_end - merged_start > max_duration:
                # 合并会突破时长硬限制：只有与所有重叠项都合不动时才丢弃
                blocked = True
                continue
            merged = ClipCandidate(
                id="clip",
                start=round(merged_start, 2),
                end=round(merged_end, 2),
                score=max(candidate.score, kept.score),
                title=kept.title,
                reason=kept.reason,
                text=kept.text if len(kept.text) >= len(candidate.text) else candidate.text,
                segment_ids=list(dict.fromkeys([*kept.segment_ids, *candidate.segment_ids])),
            )
            output.pop(i)
            # 合并结果放回队列头部，重新与其他已保留候选检查重叠(可能产生新的重叠)
            ordered.insert(0, merged)
            merged_into = i
            break
        if merged_into is None and not blocked:
            output.append(candidate)
    return output


def _pattern_hits(text: str, patterns: list[str]) -> int:
    lower = text.lower()
    return sum(1 for pattern in patterns if pattern.lower() in lower)


def _looks_complete(text: str) -> bool:
    text = text.strip()
    return bool(text and text[-1] in ".!?。！？")


def _pause_count(window: list[SubtitleSegment]) -> int:
    count = 0
    for left, right in zip(window, window[1:]):
        if 0.45 <= right.start - left.end <= 4.0:
            count += 1
    return count


def _title_from_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= 34:
        return text or "Untitled clip"
    return text[:34].rstrip() + "..."


def _reason(hook_hits: int, ending_hits: int, question_bonus: float, duration: float) -> str:
    parts = [f"{duration:.0f}s"]
    if hook_hits:
        parts.append("hook")
    if question_bonus:
        parts.append("question")
    if ending_hits:
        parts.append("clear ending")
    if len(parts) == 1:
        parts.append("dense transcript window")
    return ", ".join(parts)


def _join_text(parts: Iterable[str]) -> str:
    text = ""
    for part in parts:
        part = str(part or "").strip()
        if not part:
            continue
        if text and text[-1].isascii() and part[0].isascii():
            text += " "
        text += part
    return text


def rebase_segments_for_clip(
    segments: Iterable[SubtitleSegment | dict],
    *,
    start: float,
    end: float,
) -> list[SubtitleSegment]:
    """Clip subtitle events to a source range and rebase them to clip time zero."""
    start = max(0.0, float(start))
    end = max(start, float(end))
    rebased: list[SubtitleSegment] = []
    for index, segment in enumerate(_prepare_segments(segments), start=1):
        overlap_start = max(start, segment.start)
        overlap_end = min(end, segment.end)
        if overlap_end <= overlap_start:
            continue
        rebased.append(
            SubtitleSegment(
                id=f"clip_seg_{index:04d}",
                start=round(overlap_start - start, 3),
                end=round(overlap_end - start, 3),
                speaker=segment.speaker,
                text=segment.text,
            )
        )
    return rebased
