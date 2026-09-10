from __future__ import annotations

import math
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

# Rule screening starts from sliding windows. Those windows are useful for
# scoring, while the primary shortlist below removes their overlap. The
# looser ratio is kept for associating lower-scoring alternates with a parent.
MAX_CANDIDATE_OVERLAP_RATIO = 0.35
# Primary rule-screened clips are meant to be a readable shortlist.  Any
# overlap is treated as the same time range; a tiny gap is also treated as
# continuous so that two sliding windows cannot appear as a repeated pair.
PRIMARY_CANDIDATE_OVERLAP_RATIO = 0.0
PRIMARY_CANDIDATE_MIN_GAP = 3.0
PRIMARY_CLIP_SECONDS_PER_RESULT = 150.0
PRIMARY_MIN_RESULTS = 4
ADJACENT_MERGE_GAP = 4.0


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
    # Candidates returned by rule screening can be split into a primary lane
    # and a lower-confidence alternate lane.  The latter keeps useful
    # overlapping windows visible without putting them on top of the primary
    # ranges in the timeline.
    lane: str = "primary"
    parent_id: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass(slots=True)
class _ScoredWindow:
    """Cheap internal representation used while exploring sliding windows."""

    start_index: int
    end_index: int
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class _SegmentFeatures:
    text_length: int
    separator_before: int
    hook_mask: int
    ending_hits: int
    punctuation_count: int
    has_question: bool
    complete: bool


def generate_clip_candidates(
    segments: Iterable[SubtitleSegment | dict],
    *,
    min_duration: float = 45.0,
    target_duration: float = 120.0,
    max_duration: float | None = 300.0,
    limit: int = 24,
    merge_expansion_limit: float | None = None,
    include_alternates: bool = False,
    alternate_limit: int | None = None,
    adaptive_limit: bool = True,
) -> list[ClipCandidate]:
    prepared = _prepare_segments(segments)
    if not prepared:
        return []

    scored_windows: list[_ScoredWindow] = []
    segment_features = _build_segment_features(prepared)
    for start_index, first in enumerate(prepared):
        text_length = 0
        separator_count = 0
        hook_mask = 0
        punctuation_count = 0
        pause_count = 0
        has_question = False
        for end_index in range(start_index, len(prepared)):
            segment = prepared[end_index]
            if end_index > start_index and segment.start - prepared[end_index - 1].end > 12.0:
                break
            features = segment_features[end_index]
            if end_index > start_index:
                separator_count += features.separator_before
                previous = prepared[end_index - 1]
                if 0.45 <= segment.start - previous.end <= 4.0:
                    pause_count += 1
            text_length += features.text_length
            hook_mask |= features.hook_mask
            punctuation_count += features.punctuation_count
            has_question = has_question or features.has_question
            duration = segment.end - first.start
            if duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                break
            score = _score_feature_window(
                text_length + separator_count,
                duration,
                hook_mask.bit_count(),
                features.ending_hits,
                has_question,
                punctuation_count,
                pause_count,
                features.complete,
                target_duration=target_duration,
            )
            scored_windows.append(
                _ScoredWindow(
                    start_index=start_index,
                    end_index=end_index,
                    start=round(first.start, 2),
                    end=round(segment.end, 2),
                    score=score,
                )
            )

    # Suppress near-duplicate windows before the legacy boundary-merging pass.
    # Merging first can turn a chain of 120-second windows into a long outer
    # range, which is exactly the repeated-content behaviour the UI exposes as
    # two clips containing one another.
    raw_candidates = scored_windows
    primary_limit = _adaptive_primary_limit(prepared, limit) if adaptive_limit else max(1, int(limit))
    primary_windows = _suppress_overlapping_candidates(
        scored_windows,
        max_overlap_ratio=(PRIMARY_CANDIDATE_OVERLAP_RATIO if adaptive_limit else MAX_CANDIDATE_OVERLAP_RATIO),
        min_gap=PRIMARY_CANDIDATE_MIN_GAP if adaptive_limit else 0.0,
        limit=primary_limit,
    )
    candidates = [_materialize_window(item, prepared, target_duration=target_duration) for item in primary_windows]
    deduped = _dedupe_candidates(
        candidates,
        max_duration=max_duration,
        merge_expansion_limit=merge_expansion_limit,
    )
    # Once representative windows are selected, join touching windows that
    # are clearly one continuous topic.  This avoids the "cut, then cut
    # again" look while keeping the hard maximum duration intact.
    if adaptive_limit:
        deduped = _merge_adjacent_candidates(
            deduped,
            max_duration=max_duration,
            max_gap=ADJACENT_MERGE_GAP,
        )
    deduped.sort(key=lambda item: item.score, reverse=True)
    for index, item in enumerate(deduped[:primary_limit], start=1):
        item.id = f"clip_{index:03d}"
        item.lane = "primary"
        item.parent_id = None

    primary = deduped[:primary_limit]
    if not include_alternates or not primary:
        return primary

    # Keep a small, lower-scoring sample from each primary window's overlap
    # cluster.  These are intentionally not merged into the primary ranges:
    # the UI displays them on a separate lane for manual comparison.
    alternate_pool: list[ClipCandidate] = []
    primary_ids = {(item.start_index, item.end_index) for item in primary_windows}
    for candidate in raw_candidates:
        if (candidate.start_index, candidate.end_index) in primary_ids:
            continue
        overlaps = [item for item in primary if _overlap_ratio(candidate, item) > MAX_CANDIDATE_OVERLAP_RATIO]
        if not overlaps:
            continue
        parent = max(overlaps, key=lambda item: (item.score, -item.start))
        if candidate.score >= parent.score:
            continue
        alternate_pool.append(candidate)

    # The alternate lane is a comparison aid, not a second full result set.
    # Keep at most one lower-scoring window for roughly every two primary
    # clips, otherwise a 17-minute video fills the timeline with duplicates.
    alternate_cap = min(
        alternate_limit if alternate_limit is not None else primary_limit,
        max(1, math.ceil(len(primary) / 2)),
    )
    alternate_windows = _suppress_overlapping_candidates(
        alternate_pool,
        max_overlap_ratio=0.70,
        min_gap=0.0,
        limit=alternate_cap,
    )
    alternates = [_materialize_window(item, prepared, target_duration=target_duration) for item in alternate_windows]
    for index, item in enumerate(alternates, start=1):
        item.id = f"clip_alt_{index:03d}"
        item.lane = "alternate"
        overlaps = [parent for parent in primary if _overlap_ratio(item, parent) > MAX_CANDIDATE_OVERLAP_RATIO]
        parent = max(overlaps, key=lambda candidate: (candidate.score, -candidate.start)) if overlaps else None
        item.parent_id = parent.id if parent else None
    return [*primary, *alternates]


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


def _build_segment_features(segments: list[SubtitleSegment]) -> list[_SegmentFeatures]:
    """Precompute values that are reused by every sliding window."""
    hook_patterns = tuple(pattern.lower() for pattern in HOOK_PATTERNS)
    ending_patterns = tuple(pattern.lower() for pattern in ENDING_PATTERNS)
    features: list[_SegmentFeatures] = []
    for index, segment in enumerate(segments):
        lower = segment.text.lower()
        hook_mask = sum(1 << pattern_index for pattern_index, pattern in enumerate(hook_patterns) if pattern in lower)
        ending_hits = sum(1 for pattern in ending_patterns if pattern in lower)
        features.append(
            _SegmentFeatures(
                text_length=len(segment.text),
                separator_before=(
                    1
                    if index > 0 and segments[index - 1].text[-1:].isascii() and segment.text[:1].isascii()
                    else 0
                ),
                hook_mask=hook_mask,
                ending_hits=ending_hits,
                punctuation_count=len(re.findall(r"[!！?？。]", segment.text)),
                has_question=bool(re.search(r"[?？]", segment.text)),
                complete=_looks_complete(segment.text),
            )
        )
    return features


def _score_feature_window(
    text_length: int,
    duration: float,
    hook_hits: int,
    ending_hits: int,
    has_question: bool,
    punctuation_count: int,
    pause_count: int,
    complete: bool,
    *,
    target_duration: float,
) -> float:
    """Score a window from precomputed features without rebuilding its text."""
    duration = max(0.1, duration)
    question_bonus = 8.0 if has_question else 0.0
    density = min(26.0, text_length / max(duration, 1.0) * 7.0)
    duration_score = max(0.0, 22.0 - abs(duration - target_duration) / max(target_duration, 1.0) * 22.0)
    punctuation_bonus = min(8.0, punctuation_count * 1.4)
    short_pause_bonus = min(6.0, pause_count * 1.5)
    completeness = 10.0 if complete else 3.0
    return round(
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


def _materialize_window(
    window: _ScoredWindow,
    segments: list[SubtitleSegment],
    *,
    target_duration: float,
) -> ClipCandidate:
    """Build the full public candidate only after a window survives NMS."""
    candidate = _score_window(segments[window.start_index : window.end_index + 1], target_duration=target_duration)
    candidate.start = window.start
    candidate.end = window.end
    return candidate


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
    """重叠候选合并为取时间外沿的一条更完整片段。

    The generator performs representative-window suppression before calling
    this helper.  Keeping the merge behaviour here preserves the helper's
    existing semantics for manually supplied candidates and for callers that
    explicitly want boundary expansion.
    """
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
                reason=_reason_with_duration(kept.reason, merged_end - merged_start),
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


def _adaptive_primary_limit(prepared: list[SubtitleSegment], requested_limit: int) -> int:
    """Scale the readable rule shortlist with the source duration.

    ``limit`` remains a caller-provided upper bound, while the default rule
    screen gets about one representative result per 150 seconds.  This keeps
    a 17-minute source around 6–8 clips instead of returning every sliding
    window that happened to score well.
    """
    if not prepared:
        return 0
    requested = max(1, int(requested_limit))
    duration = max(0.0, prepared[-1].end)
    duration_limit = max(PRIMARY_MIN_RESULTS, math.ceil(duration / PRIMARY_CLIP_SECONDS_PER_RESULT))
    return min(requested, duration_limit)


def _suppress_overlapping_candidates(
    candidates: list[ClipCandidate],
    *,
    max_overlap_ratio: float = MAX_CANDIDATE_OVERLAP_RATIO,
    min_gap: float = 0.0,
    limit: int | None = None,
) -> list[ClipCandidate]:
    """Keep the highest-scoring representative from each overlapping window group.

    Candidate generation deliberately explores many start/end combinations.
    Non-maximum suppression is applied to those raw windows before any range
    merge, so a lower-scoring window cannot survive merely because it extends a
    higher-scoring window's boundary.  The overlap ratio uses the shorter
    candidate as its denominator; consequently a containing range is always
    rejected regardless of which one is longer.
    """
    if not candidates:
        return []
    threshold = max(0.0, min(1.0, float(max_overlap_ratio)))
    ordered = sorted(candidates, key=lambda item: (-item.score, item.start, item.end))
    selected: list[ClipCandidate] = []
    for candidate in ordered:
        duplicate = False
        for kept in selected:
            intersection = max(0.0, min(candidate.end, kept.end) - max(candidate.start, kept.start))
            shorter = max(1.0, min(candidate.duration, kept.duration))
            distance = min(abs(candidate.start - kept.end), abs(kept.start - candidate.end))
            if intersection / shorter > threshold or (intersection <= 0.0 and distance < max(0.0, min_gap)):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(candidate)
        if limit is not None and limit > 0 and len(selected) >= limit:
            break
    return selected


def _merge_adjacent_candidates(
    candidates: list[ClipCandidate], *, max_duration: float | None, max_gap: float
) -> list[ClipCandidate]:
    """Merge selected windows that touch at a natural topic boundary."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.start, item.end))
    output: list[ClipCandidate] = []
    for candidate in ordered:
        if output:
            previous = output[-1]
            gap = candidate.start - previous.end
            merged_end = max(previous.end, candidate.end)
            if gap <= max(0.0, max_gap) and (
                max_duration is None or merged_end - previous.start <= max_duration
            ):
                previous.end = round(merged_end, 2)
                previous.score = max(previous.score, candidate.score)
                previous.reason = _reason_with_duration(previous.reason, previous.duration)
                previous.text = previous.text if len(previous.text) >= len(candidate.text) else candidate.text
                previous.segment_ids = list(dict.fromkeys([*previous.segment_ids, *candidate.segment_ids]))
                continue
        output.append(candidate)
    return output


def _overlap_ratio(left: ClipCandidate, right: ClipCandidate) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    shorter = max(1.0, min(left.duration, right.duration))
    return intersection / shorter


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


def _reason_with_duration(reason: str, duration: float) -> str:
    """Refresh the leading duration after candidate ranges are merged."""
    parts = [part.strip() for part in str(reason or "").split(",") if part.strip()]
    duration_part = f"{max(0.0, duration):.0f}s"
    if parts and re.fullmatch(r"\d+(?:\.\d+)?s", parts[0]):
        parts[0] = duration_part
    else:
        parts.insert(0, duration_part)
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
