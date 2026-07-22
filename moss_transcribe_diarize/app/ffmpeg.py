from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FFmpegAvailability:
    ffmpeg: str | None
    ffprobe: str | None

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg and self.ffprobe)

    def to_dict(self) -> dict[str, str | bool | None]:
        return {"available": self.available, "ffmpeg": self.ffmpeg, "ffprobe": self.ffprobe}


def detect_ffmpeg() -> FFmpegAvailability:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return FFmpegAvailability(ffmpeg=ffmpeg, ffprobe=ffprobe)

    return _detect_portable_ffmpeg()


def _detect_portable_ffmpeg() -> FFmpegAvailability:
    root = Path(__file__).resolve().parents[2]
    tools_dir = root / "tools"
    candidates = [
        tools_dir / "ffmpeg" / "bin",
        tools_dir / "ffmpeg",
    ]
    if tools_dir.exists():
        for child in tools_dir.iterdir():
            if child.is_dir():
                candidates.extend([child / "bin", child])

    for candidate in candidates:
        ffmpeg = candidate / "ffmpeg.exe"
        ffprobe = candidate / "ffprobe.exe"
        if ffmpeg.exists() and ffprobe.exists():
            return FFmpegAvailability(ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    return FFmpegAvailability(ffmpeg=None, ffprobe=None)


def probe_media(path: str | Path) -> dict[str, Any]:
    tools = detect_ffmpeg()
    if not tools.ffprobe:
        raise RuntimeError("ffprobe is not available on PATH.")
    command = [
        tools.ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout or "{}")


def probe_video_size(path: str | Path, *, default: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    try:
        media = probe_media(path)
    except Exception:
        return default
    for stream in media.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width") or default[0])
            height = int(stream.get("height") or default[1])
            return width, height
    return default


def burn_ass_subtitles(
    input_media: str | Path,
    ass_path: str | Path,
    output_path: str | Path,
    *,
    style: Any | None = None,
    overwrite: bool = True,
) -> Path:
    tools = detect_ffmpeg()
    if not tools.available:
        raise RuntimeError("ffmpeg and ffprobe are required for video rendering.")

    input_media = Path(input_media).resolve()
    ass_path = Path(ass_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_args = _build_filter_args(ass_path, style=style)
    command = [
        tools.ffmpeg or "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_media),
        *filter_args,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, cwd=str(ass_path.parent), check=True, capture_output=True, text=True)
    return output_path


def _build_filter_args(ass_path: Path, *, style: Any | None = None) -> list[str]:
    filter_graph = _build_filter_graph(ass_path, style=style)
    return ["-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a?"]


def _build_filter_graph(ass_path: Path, *, style: Any | None = None) -> str:
    subtitles = f"subtitles={_escape_filter_path(ass_path.name)}"
    if style is not None and bool(getattr(style, "mask_enabled", False)):
        height = max(1, int(getattr(style, "mask_height", 120)))
        margin_v = max(0, int(getattr(style, "mask_margin_v", 0)))
        opacity = max(0.0, min(1.0, float(getattr(style, "mask_opacity", 0.82))))
        mode = str(getattr(style, "mask_mode", "blur") or "blur")
        if mode == "bar":
            return (
                "[0:v]"
                "drawbox="
                "x=0:"
                f"y=max(0\\,ih-{height + margin_v}):"
                "w=iw:"
                f"h=min({height}\\,ih-{margin_v}):"
                f"color=black@{opacity:.3f}:"
                f"t=fill,{subtitles}[v]"
            )
        blur = max(1, int(getattr(style, "mask_blur", 24)))
        region_y = f"max(0\\,ih-{height + margin_v})"
        region_h = f"min({height}\\,ih-{margin_v})"
        overlay_y = f"H-h-{margin_v}"
        return (
            f"[0:v]split=2[base][region];"
            f"[region]crop=iw:{region_h}:0:{region_y},boxblur={blur}:1[blurred];"
            f"[base][blurred]overlay=0:{overlay_y},"
            f"drawbox=x=0:y={region_y}:w=iw:h={region_h}:color=black@0.180:t=fill,"
            f"{subtitles}[v]"
        )
    return f"[0:v]{subtitles}[v]"


def _escape_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
