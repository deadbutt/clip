from __future__ import annotations

import re
from collections.abc import Iterable

from moss_transcribe_diarize.transcript_parser import (
    TranscriptSegment,
    parse_transcript,
)

from .models import SubtitleItem, SubtitleSegment

DEFAULT_MIN_DURATION = 1.0
DEFAULT_MAX_DURATION = 6.0
DEFAULT_MAX_CHARS = 24
DEFAULT_MERGE_GAP = 0.3

# 句子级重组：Whisper 的 segment 以停顿为准，经常把一句话切成多个片段。
# regroup_sentences 把不以句末标点结尾的片段向后合并成完整句，
# 再把过长的句子按字幕断行规则切成适合显示的行。
SENTENCE_FINAL_CHARS = "。！？!?…."
SENTENCE_TAIL_NOISE = " \t\"'”』】）)〉》>"
_SOFT_PUNCT = "，,；;：:、"
DEFAULT_SENTENCE_MERGE_GAP = 2.0
DEFAULT_MAX_SENTENCE_CHARS = 80
DEFAULT_MAX_SENTENCE_DURATION = 15.0
DEFAULT_LINE_MAX_CHARS = 32
DEFAULT_LINE_MAX_DURATION = 10.0

# 字幕断行规则（参考 Netflix Timed Text Style Guide）：行尾不悬挂
# 连词/介词/冠词/助动词等虚词，这类词应跟下一行走。
_EN_DANGLING_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for", "because",
    "if", "when", "while", "although", "though", "since", "that", "which", "who",
    "whom", "whose", "where", "why", "how", "of", "to", "in", "on", "at", "by",
    "with", "from", "into", "onto", "about", "as", "than", "is", "am", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "shall", "should", "may", "might",
    "must", "not", "it's", "its", "i", "there", "here", "then", "also", "just",
}


def drop_repeated_hallucinations(
    segments: Iterable[SubtitleSegment],
    *,
    min_words: int = 6,
) -> list[SubtitleSegment]:
    """丢弃重复出现的幻觉文本。

    whisper 在音乐段常见的幻觉模式：同一句"不存在的话"在整条音轨里
    反复冒出来（如 "It was hard, but it was hard for me to tell."
    隔几分钟再来一遍）。这里按归一化文本判重：
    - 与更早的段完全相同（>=min_words 词）-> 丢弃；
    - 是更早的段（>=min_words 词）的子串 -> 丢弃（幻觉常被截断复述）。
    短句（<min_words 词）不动：口号/歌词/短答的合法重复太常见。
    首次出现无法仅凭文本判定真伪，保留。
    """
    kept: list[SubtitleSegment] = []
    seen: list[str] = []
    for segment in segments:
        text = " ".join((segment.text or "").split())
        if text:
            normalized = _normalize_hallucination_text(text)
            if normalized and len(normalized.split()) >= min_words:
                duplicated = any(normalized == earlier or normalized in earlier for earlier in seen)
                if duplicated:
                    continue
                seen.append(normalized)
        kept.append(segment)
    return kept


def _normalize_hallucination_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def subtitle_segments_from_transcript(
    transcript: str,
    *,
    postprocess: bool = True,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[SubtitleSegment]:
    return subtitle_segments_from_transcript_segments(
        parse_transcript(transcript),
        postprocess=postprocess,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars,
        merge_gap=merge_gap,
    )


def subtitle_segments_from_transcript_segments(
    segments: Iterable[TranscriptSegment],
    *,
    postprocess: bool = True,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[SubtitleSegment]:
    subtitle_segments = [
        SubtitleSegment(
            id=f"seg_{index:04d}",
            start=float(segment.start),
            end=float(segment.end),
            speaker=segment.speaker,
            text=segment.text,
        )
        for index, segment in enumerate(segments, start=1)
    ]
    if not postprocess:
        return subtitle_segments
    return normalize_segments(
        subtitle_segments,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars,
        merge_gap=merge_gap,
        regenerate_ids=True,
    )


def regroup_sentences(
    segments: Iterable[SubtitleSegment | dict],
    *,
    merge_gap: float = DEFAULT_SENTENCE_MERGE_GAP,
    max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
    max_sentence_duration: float = DEFAULT_MAX_SENTENCE_DURATION,
    max_chars: int = DEFAULT_LINE_MAX_CHARS,
    max_duration: float = DEFAULT_LINE_MAX_DURATION,
) -> list[SubtitleSegment]:
    """把停顿切开的 Whisper 片段重组为完整句子的字幕段。

    与 ``normalize_segments`` 不同，这里不拉伸时长、不强推 min_duration，
    时间轴尽量保持转录原样，只做"合并成句 + 过长再按标点切行"。
    """
    prepared = _prepare_segments(segments)
    prepared = _merge_into_sentences(
        prepared,
        merge_gap=merge_gap,
        max_chars=max_sentence_chars,
        max_duration=max_sentence_duration,
    )
    prepared = _split_long_segments(
        prepared,
        min_duration=0.5,
        max_duration=max_duration,
        max_chars=max_chars,
    )
    prepared = _fix_dangling_line_ends(prepared)
    for index, segment in enumerate(prepared, start=1):
        segment.id = f"seg_{index:04d}"
    return prepared


def _char_weight(ch: str) -> float:
    return 1.0 if ord(ch) > 0x2E7F else 0.5


def _text_weight(text: str) -> float:
    """按显示宽度计字数：中日韩字符算 1，ASCII 算 0.5。

    同一句话英文的字符数约是中文的两倍以上，直接用 len() 会让英文句子
    永远撞上字数上限而无法合并成完整句。
    """
    return sum(_char_weight(ch) for ch in text)


def regroup_sentences_from_words(
    words: Iterable[tuple[float, float, str]],
    *,
    max_chars: int = DEFAULT_LINE_MAX_CHARS,
    max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
    max_sentence_duration: float = DEFAULT_MAX_SENTENCE_DURATION,
) -> list[SubtitleSegment]:
    """词级重组：把词流按标点重组为完整句、再切成字幕行。

    与 ``regroup_sentences``（segment 级）相比，这里不受 Whisper 原始
    segment 边界、合并间隔等约束影响，且每行的起止时间直接取自词的
    真实时间戳（segment 级只能按字符比例估算）。

    长句软切：超重时不在词中间硬切，等下一个逗号/分号再切（逗号是语义断点），
    极硬上限(1.5x)兜底。解决 "the reason" 孤立开头类长句硬切问题。
    """
    flat = [(float(s), float(e), str(t)) for s, e, t in words if str(t).strip()]
    if not flat:
        return []
    units = _sentence_units_from_words(
        flat,
        max_chars=max_sentence_chars,
        max_duration=max_sentence_duration,
    )
    segments: list[SubtitleSegment] = []
    for unit in units:
        segments.extend(_cut_lines_from_words(unit, max_chars=max_chars))
    segments = _clamp_word_segments(segments)
    segments = _merge_tiny_word_segments(segments, max_chars=max_chars)
    for index, segment in enumerate(segments, start=1):
        segment.id = f"seg_{index:04d}"
    return segments


def _merge_tiny_word_segments(
    segments: list[SubtitleSegment],
    *,
    max_chars: int,
    tiny_duration: float = 1.2,
    tiny_weight: float = 8.0,
) -> list[SubtitleSegment]:
    """合并过短的词级片段。

    半句碎片（逗号尾/无标点尾）向后并入相邻行，消除 400ms 量级的孤立
    碎片。但上一行以句末标点结尾时不吸收——regroup 阶段没有说话人信息，
    "句末短答"极可能是换人插话，揉进一行会让两人的话混在一条字幕里
    （翻译最怕的粘行）。
    """
    if len(segments) < 2:
        return segments

    def is_tiny(seg: SubtitleSegment) -> bool:
        return (seg.end - seg.start) <= tiny_duration and _text_weight(seg.text) <= tiny_weight

    merged: list[SubtitleSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if not is_tiny(seg) or _text_weight(prev.text) + _text_weight(seg.text) > max_chars:
            merged.append(seg)
            continue
        # 反粘行：上一行以句末标点结尾就一律不回吸——regroup 阶段没有
        # 说话人信息，"Yes./Easily. 连发"与"Thank you./No worries. 互答"
        # 无法区分，而后者揉进一行翻译就报废——宁可多出一条短行。
        if _ends_sentence(prev.text):
            merged.append(seg)
            continue
        merged[-1] = SubtitleSegment(
            id=prev.id,
            start=prev.start,
            end=seg.end,
            speaker=prev.speaker,
            text=_join_text(prev.text, seg.text),
            items=_concat_items(prev, seg),
        )
    return merged


def _sentence_units_from_words(
    words: list[tuple[float, float, str]],
    *,
    max_chars: int,
    max_duration: float,
) -> list[list[tuple[float, float, str]]]:
    units: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    weight = 0.0
    start_time = 0.0
    for start, end, text in words:
        if not current:
            start_time = start
        current.append((start, end, text))
        weight += _text_weight(text)
        over = weight > max_chars
        has_comma = (
            text.endswith(',') or text.endswith(';')
            or text.endswith('，') or text.endswith('；')
        )
        hard = (
            _ends_sentence(text)
            or weight > max_chars * 1.5
            or end - start_time > max_duration
        )
        # 长句软切：超重(weight>max_chars)时不立即在词中间硬切，等下一个逗号/分号再切——
        # 逗号是语义断点，切在此处比切在 "the reason" 这类词中间自然（解决长句硬切）。
        # 极硬上限(1.5x)兜底防过长；停顿断句经实测碎片化(um 后断)，已回退不用。
        if hard or (over and has_comma):
            units.append(current)
            current = []
            weight = 0.0
    if current:
        units.append(current)
    return units


def _items_from_words(words: list[tuple[float, float, str]]) -> list[SubtitleItem]:
    return [
        SubtitleItem(text=text, start=float(start), end=float(end))
        for start, end, text in words
        if str(text).strip()
    ]


def _concat_items(*segments: SubtitleSegment) -> list[SubtitleItem] | None:
    """合并多段的 items；任一段缺失就返回 None（词级真源不完整宁缺毋滥）。"""
    if any(segment.items is None for segment in segments):
        return None
    return [item for segment in segments for item in (segment.items or [])]


def _cut_lines_from_words(
    unit: list[tuple[float, float, str]],
    *,
    max_chars: int,
    hard_split_weight: float = 64.0,
    hard_split_duration: float = 8.0,
) -> list[SubtitleSegment]:
    """把一个句子 unit 格式化成字幕 segment。

    默认一个 unit → 一个 segment：文本超过单行显示宽度时在内部用 ``\\n``
    换行（多行文本共享同一组时间轴），不再切成多个独立 segment。这样下游
    翻译拿到的是完整句，避免半句被各自翻译导致语义断裂/突兀开头。

    只有 unit 整体超过硬切上限（太长或太久）才递归切成多个 segment，
    避免一句占屏过久、文本过多行。
    """
    if not unit:
        return []
    total_weight = _words_weight(unit)
    duration = unit[-1][1] - unit[0][0]
    if total_weight > hard_split_weight or duration > hard_split_duration:
        cut = _find_word_break(unit, max(total_weight / 2, max_chars))
        if 0 < cut < len(unit):
            return (
                _cut_lines_from_words(unit[:cut], max_chars=max_chars)
                + _cut_lines_from_words(unit[cut:], max_chars=max_chars)
            )
    lines = _split_words_into_display_lines(unit, max_chars=max_chars)
    text = "\n".join(_line_text(line) for line in lines if line)
    if not text:
        return []
    start = unit[0][0]
    end = max(unit[-1][1], start)
    return [SubtitleSegment(id="", start=start, end=end, speaker="S00", text=text, items=_items_from_words(unit))]


def _split_words_into_display_lines(
    words: list[tuple[float, float, str]],
    *,
    max_chars: int,
) -> list[list[tuple[float, float, str]]]:
    """把词流按单行显示宽度切成若干显示行（仅用于同一 segment 内部换行）。"""
    rest = words
    lines: list[list[tuple[float, float, str]]] = []
    while True:
        if _words_weight(rest) <= max_chars:
            lines.append(rest)
            break
        cut = _find_word_break(rest, max_chars)
        lines.append(rest[:cut])
        rest = rest[cut:]
    lines = _move_dangling_words_down(lines)
    lines = _compact_word_lines(lines, max_chars)
    return [line for line in lines if line]


def _line_text(words: list[tuple[float, float, str]]) -> str:
    return " ".join("".join(text for _, _, text in words).split())


def _words_weight(words: list[tuple[float, float, str]]) -> float:
    return sum(_text_weight(text) for _, _, text in words)


def _find_word_break(words: list[tuple[float, float, str]], max_chars: int) -> int:
    """选最佳断词位置（返回切点后第一行的词数），规则同 segment 级。"""
    total = _words_weight(words)
    target = min(float(max_chars), total / 2)
    window_min = target * 0.7
    best_rank = 4
    best_count = -1
    best_distance = 0.0
    weight = 0.0
    hard_cut = len(words)
    for index, (_, _, text) in enumerate(words):
        weight += _text_weight(text)
        if weight >= max_chars and hard_cut == len(words):
            hard_cut = index + 1
        if weight > max_chars * 1.5:
            break
        if index + 1 >= len(words):
            continue  # 末尾不作切点（切在末尾=不切，会退化为单行）
        stripped = text.strip()
        is_final = _ends_sentence(stripped)
        is_soft = stripped[-1:] in _SOFT_PUNCT
        # 标点切点（句末/逗号）不受 window_min 限制：语义断点优先于行长均衡，
        # 解决长句硬切在词中间（"the reason"/"to go back" 类问题）
        if not (is_final or is_soft) and weight < window_min:
            continue
        if is_final:
            rank = 0
        elif is_soft:
            rank = 1
        elif stripped.strip("'.-").lower() in _EN_DANGLING_WORDS:
            rank = 3
        else:
            rank = 2
        distance = abs(weight - target)
        if rank < best_rank or (rank == best_rank and distance < best_distance):
            best_rank, best_count, best_distance = rank, index + 1, distance
    return best_count if best_count > 0 else hard_cut


def _move_dangling_words_down(lines: list[list[tuple[float, float, str]]]) -> list[list[tuple[float, float, str]]]:
    """行尾悬挂虚词下沉到下一行（词级版，直接搬词对象）。"""
    for index in range(len(lines) - 1):
        for _ in range(2):
            current = lines[index]
            if not current or _ends_sentence(current[-1][2]):
                break
            word = current[-1][2].strip().strip("'.-").lower()
            if word not in _EN_DANGLING_WORDS or len(current) < 2:
                break
            lines[index + 1].insert(0, current.pop())
    return lines


def _compact_word_lines(
    lines: list[list[tuple[float, float, str]]], max_chars: int
) -> list[list[tuple[float, float, str]]]:
    compact: list[list[tuple[float, float, str]]] = []
    for line in lines:
        if not line:
            continue
        if compact:
            previous_weight = _words_weight(compact[-1])
            current_weight = _words_weight(line)
            fits = previous_weight + current_weight <= max_chars
            tiny_orphan = current_weight <= 2 and previous_weight + current_weight <= max_chars + 4
            if fits or tiny_orphan:
                compact[-1].extend(line)
                continue
        compact.append(line)
    return compact


def _segment_from_words(words: list[tuple[float, float, str]]) -> SubtitleSegment:
    start = words[0][0]
    end = max(words[-1][1], start)
    text = " ".join("".join(text for _, _, text in words).split())
    return SubtitleSegment(id="", start=start, end=end, speaker="S00", text=text, items=_items_from_words(words))


def _clamp_word_segments(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """词对齐偶尔会轻微重叠或给出零时长，收敛成单调不减的时间轴。"""
    cursor = 0.0
    for segment in segments:
        start = max(segment.start, cursor)
        end = max(segment.end, start + 0.1)
        segment.start = start
        segment.end = end
        cursor = end
    return segments


def _merge_into_sentences(
    segments: list[SubtitleSegment],
    *,
    merge_gap: float,
    max_chars: int,
    max_duration: float,
) -> list[SubtitleSegment]:
    merged: list[SubtitleSegment] = []
    for segment in segments:
        if merged:
            previous = merged[-1]
            gap = segment.start - previous.end
            combined_text = _join_text(previous.text, segment.text)
            can_merge = (
                previous.speaker == segment.speaker
                and -0.5 <= gap <= merge_gap
                and not _ends_sentence(previous.text)
                and _text_weight(combined_text) <= max_chars
                and segment.end - previous.start <= max_duration
            )
            if can_merge:
                merged[-1] = SubtitleSegment(
                    id=previous.id,
                    start=previous.start,
                    end=max(previous.end, segment.end),
                    speaker=previous.speaker,
                    text=combined_text,
                )
                continue
        merged.append(segment)
    return merged


def _ends_sentence(text: str) -> bool:
    for ch in reversed((text or "").strip()):
        if ch in SENTENCE_TAIL_NOISE:
            continue
        return ch in SENTENCE_FINAL_CHARS
    return False


def coerce_subtitle_segments(segments: Iterable[SubtitleSegment | dict]) -> list[SubtitleSegment]:
    """Convert user/API payloads to subtitle segments without timing edits."""
    coerced: list[SubtitleSegment] = []
    for index, item in enumerate(segments, start=1):
        segment = item if isinstance(item, SubtitleSegment) else SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}")
        coerced.append(
            SubtitleSegment(
                id=segment.id or f"seg_{index:04d}",
                start=float(segment.start),
                end=float(segment.end),
                speaker=segment.speaker or "S00",
                text=segment.text,
                items=segment.items,
            )
        )
    return coerced


def normalize_segments(
    segments: Iterable[SubtitleSegment | dict],
    *,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_gap: float = DEFAULT_MERGE_GAP,
    regenerate_ids: bool = False,
) -> list[SubtitleSegment]:
    prepared = _prepare_segments(segments)
    prepared = _fix_overlaps(prepared, min_duration=min_duration)
    prepared = _merge_adjacent(prepared, merge_gap=merge_gap, max_chars=max_chars)
    prepared = _split_long_segments(
        prepared,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars,
    )
    prepared = _fix_overlaps(prepared, min_duration=min_duration)
    if regenerate_ids:
        for index, segment in enumerate(prepared, start=1):
            segment.id = f"seg_{index:04d}"
    return prepared


def _prepare_segments(segments: Iterable[SubtitleSegment | dict]) -> list[SubtitleSegment]:
    prepared: list[SubtitleSegment] = []
    for index, item in enumerate(segments, start=1):
        segment = item if isinstance(item, SubtitleSegment) else SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}")
        text = segment.text.strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        prepared.append(
            SubtitleSegment(
                id=segment.id or f"seg_{index:04d}",
                start=start,
                end=end,
                speaker=segment.speaker or "S00",
                text=text,
            )
        )
    prepared.sort(key=lambda segment: (segment.start, segment.end))
    return prepared


def _fix_overlaps(segments: list[SubtitleSegment], *, min_duration: float) -> list[SubtitleSegment]:
    cursor = 0.0
    fixed: list[SubtitleSegment] = []
    for segment in segments:
        start = max(segment.start, cursor)
        end = max(segment.end, start + min_duration)
        fixed.append(
            SubtitleSegment(
                id=segment.id,
                start=start,
                end=end,
                speaker=segment.speaker,
                text=segment.text,
            )
        )
        cursor = end
    return fixed


def _merge_adjacent(segments: list[SubtitleSegment], *, merge_gap: float, max_chars: int) -> list[SubtitleSegment]:
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start - previous.end
        combined_text = _join_text(previous.text, segment.text)
        can_merge = (
            previous.speaker == segment.speaker
            and 0 <= gap <= merge_gap
            and len(combined_text) <= max_chars * 2
        )
        if can_merge:
            merged[-1] = SubtitleSegment(
                id=previous.id,
                start=previous.start,
                end=max(previous.end, segment.end),
                speaker=previous.speaker,
                text=combined_text,
            )
        else:
            merged.append(segment)
    return merged


def _split_long_segments(
    segments: list[SubtitleSegment],
    *,
    min_duration: float,
    max_duration: float,
    max_chars: int,
) -> list[SubtitleSegment]:
    output: list[SubtitleSegment] = []
    for segment in segments:
        duration = segment.end - segment.start
        if duration <= max_duration and _text_weight(segment.text) <= max_chars:
            output.append(segment)
            continue

        chunks = _split_text(segment.text, max_chars=max_chars)
        if len(chunks) <= 1:
            output.append(segment)
            continue

        total_chars = sum(max(_text_weight(chunk), 1.0) for chunk in chunks)
        cursor = segment.start
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                end = segment.end
            else:
                ratio = max(_text_weight(chunk), 1.0) / total_chars
                end = cursor + max(min_duration, duration * ratio)
                end = min(end, segment.end - min_duration * (len(chunks) - index - 1))
            output.append(
                SubtitleSegment(
                    id=f"{segment.id}_{index + 1}",
                    start=cursor,
                    end=max(end, cursor + min_duration),
                    speaker=segment.speaker,
                    text=chunk,
                )
            )
            cursor = output[-1].end
    return output


def _split_text(text: str, *, max_chars: int) -> list[str]:
    text = text.strip()
    if _text_weight(text) <= max_chars:
        return [text]

    # 先按句末标点拆成整句，一句内部再选最佳断点，
    # 避免把两个短句拼到一行后又从中间乱切。
    chunks: list[str] = []
    for unit in _split_display_sentences(text):
        if _text_weight(unit) <= max_chars:
            chunks.append(unit)
            continue
        chunks.extend(_split_overlong_unit(unit, max_chars))
    return _compact_chunks(chunks, max_chars)


def _fix_dangling_line_ends(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """行尾悬挂的虚词（because/why/and/I 等）挪到下一行开头。

    合并或切行受上限约束时，行尾可能正好停在虚词上；这种断点读起来
    像句子被拦腰截断。只要该行不是句子结尾，就把虚词下沉给下一行。
    """
    for index in range(len(segments) - 1):
        current = segments[index]
        following = segments[index + 1]
        for _ in range(2):
            if _ends_sentence(current.text):
                break
            words = current.text.split(" ")
            if len(words) < 2 or words[-1].lower().strip("'.") not in _EN_DANGLING_WORDS:
                break
            moved = words.pop()
            current.text = " ".join(words).rstrip()
            following.text = f"{moved} {following.text.lstrip()}"
    return segments


def _split_display_sentences(text: str) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        if ch in SENTENCE_FINAL_CHARS:
            units.append("".join(current).strip())
            current.clear()
    if current:
        units.append("".join(current).strip())
    return [unit for unit in units if unit]


def _split_overlong_unit(unit: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    rest = unit
    while _text_weight(rest) > max_chars:
        cut = _find_break_index(rest, max_chars)
        chunk = rest[:cut].strip()
        if chunk:
            chunks.append(chunk)
        rest = rest[cut:].lstrip()
    if rest.strip():
        chunks.append(rest.strip())
    return chunks


def _find_break_index(unit: str, max_chars: int) -> int:
    """选最佳断点，让两行长度尽量均衡（断在句子中线附近）。

    优先级：句末标点 > 逗号等次级标点 > 干净的词边界 > 悬挂虚词的词边界
    （后者还要尽量把虚词挪到下一行）。同级取离中线最近的候选。
    """
    total = _text_weight(unit)
    target = min(float(max_chars), total / 2)
    window_min = target * 0.7
    best_rank = 4
    best_index = -1
    best_distance = 0.0
    weight = 0.0
    hard_cut = len(unit)
    for index, ch in enumerate(unit):
        weight += _char_weight(ch)
        if weight >= max_chars and hard_cut == len(unit):
            hard_cut = index + 1
        if weight > max_chars:
            break
        rank = None
        if ch in SENTENCE_FINAL_CHARS:
            rank = 0
        elif ch in _SOFT_PUNCT:
            rank = 1
        elif ch == " ":
            rank = 3 if _dangling_tail(unit, index) else 2
        if rank is None or weight < window_min:
            continue
        distance = abs(weight - target)
        if rank < best_rank or (rank == best_rank and distance < best_distance):
            best_rank, best_index, best_distance = rank, index + 1, distance

    if best_index > 0:
        if best_rank == 3:
            best_index = _move_dangling_words(unit, best_index)
        return best_index
    return hard_cut


def _dangling_tail(unit: str, space_index: int) -> bool:
    """unit[:space_index] 以空格结尾时，最后一个词是否是虚词。"""
    end = space_index
    start = end
    while start > 0 and (unit[start - 1].isalnum() or unit[start - 1] == "'"):
        start -= 1
    word = unit[start:end].lower().strip("'.")
    return word in _EN_DANGLING_WORDS


def _move_dangling_words(unit: str, cut: int) -> int:
    """把行尾悬挂的虚词挪到下一行：断点回退到虚词前一个词边界。"""
    for _ in range(3):
        end = cut
        if end > 0 and unit[end - 1] == " ":
            end -= 1
        start = end
        while start > 0 and (unit[start - 1].isalnum() or unit[start - 1] == "'"):
            start -= 1
        word = unit[start:end].lower().strip("'.")
        if word not in _EN_DANGLING_WORDS:
            return cut
        previous_space = unit.rfind(" ", 0, start)
        if previous_space <= 0:
            return cut
        cut = previous_space
    return cut


def _compact_chunks(chunks: list[str], max_chars: int) -> list[str]:
    compact: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if compact:
            previous_weight = _text_weight(compact[-1])
            current_weight = _text_weight(chunk)
            # 正常合并；超小残片（如句尾被切出来的 "I"、"的"）放宽一点也要并入，
            # 否则会单独占一行字幕。
            fits = previous_weight + current_weight <= max_chars
            tiny_orphan = current_weight <= 2 and previous_weight + current_weight <= max_chars + 4
            if fits or tiny_orphan:
                compact[-1] = _join_text(compact[-1], chunk)
                continue
        compact.append(chunk)
    return compact


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and right[0].isascii():
        return f"{left} {right}"
    return f"{left}{right}"
