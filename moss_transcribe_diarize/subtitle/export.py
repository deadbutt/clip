from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import SubtitleSegment, SubtitleStyle


SPEAKER_COLORS = [
    "&H00FFFFFF",
    "&H005BE7FF",
    "&H0086F28F",
    "&H00BBA7FF",
    "&H0000D7FF",
    "&H00FFB56B",
    "&H00FF8EDB",
    "&H00D8D8D8",
]


def export_json(segments: Iterable[SubtitleSegment], *, indent: int = 2) -> str:
    return json.dumps([segment.to_dict() for segment in segments], ensure_ascii=False, indent=indent) + "\n"


def export_srt(
    segments: Iterable[SubtitleSegment],
    *,
    show_speaker: bool = True,
    speaker_names: dict[str, str] | None = None,
) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        text = _display_text(segment, show_speaker=show_speaker, speaker_names=speaker_names)
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def export_ass(
    segments: Iterable[SubtitleSegment],
    *,
    style: SubtitleStyle | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    style = style or SubtitleStyle()
    font_size = style.font_size or max(24, round(video_height * 0.045))
    segments = list(segments)
    speakers = sorted({segment.speaker for segment in segments})
    style_lines = [_ass_style_line("Default", style, font_size, style.primary_color)]
    if style.speaker_colors:
        for index, speaker in enumerate(speakers):
            color = SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
            style_lines.append(_ass_style_line(_speaker_style_name(speaker), style, font_size, color))

    dialogue_lines = [
        _ass_dialogue_line(start, end, style_name, text)
        for start, end, style_name, text in _ass_dialogue_events(segments, style)
    ]

    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {video_width}",
            f"PlayResY: {video_height}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            *style_lines,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *dialogue_lines,
            "",
        ]
    )


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    return path


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_time(seconds: float) -> str:
    centiseconds = max(0, round(float(seconds) * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_style_line(name: str, style: SubtitleStyle, font_size: int, primary_color: str) -> str:
    return (
        f"Style: {name},{style.font_name},{font_size},{primary_color},&H000000FF,{style.outline_color},"
        f"{style.back_color},0,0,0,0,100,100,0,0,1,{style.outline},{style.shadow},"
        f"{style.alignment},48,48,{style.margin_v},1"
    )


def _speaker_style_name(speaker: str) -> str:
    return f"Speaker_{''.join(ch if ch.isalnum() else '_' for ch in speaker)}"


def _ass_dialogue_line(start: float, end: float, style_name: str, text: str) -> str:
    return (
        f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},"
        f"{style_name},,0,0,0,,{text}"
    )


def _ass_dialogue_events(
    segments: list[SubtitleSegment],
    style: SubtitleStyle,
) -> list[tuple[float, float, str, str]]:
    ordered = sorted(
        [segment for segment in segments if segment.text and segment.end > segment.start],
        key=lambda segment: (segment.start, segment.end),
    )
    if not ordered:
        return []

    boundaries = sorted({time for segment in ordered for time in (segment.start, segment.end)})
    events: list[tuple[float, float, str, str]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = [segment for segment in ordered if segment.start < end and segment.end > start]
        if not active:
            continue
        style_name = _event_style_name(active, style)
        text = r"\N".join(
            _ass_escape(_display_text(segment, show_speaker=style.show_speaker, speaker_names=style.speaker_names))
            for segment in active
        )
        if events and events[-1][1] == start and events[-1][2] == style_name and events[-1][3] == text:
            events[-1] = (events[-1][0], end, style_name, text)
        else:
            events.append((start, end, style_name, text))
    return events


def _event_style_name(active: list[SubtitleSegment], style: SubtitleStyle) -> str:
    if len(active) == 1 and style.speaker_colors:
        return _speaker_style_name(active[0].speaker)
    return "Default"


def _display_text(
    segment: SubtitleSegment,
    *,
    show_speaker: bool,
    speaker_names: dict[str, str] | None = None,
) -> str:
    if not show_speaker or not segment.speaker:
        return segment.text
    speaker = (speaker_names or {}).get(segment.speaker) or segment.speaker
    return f"{speaker}: {segment.text}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")
