from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[float], None]


class FFmpegCancelled(RuntimeError):
    """调用方通过 cancel_check 请求中止渲染时抛出，由 jobs 层转成任务取消。"""


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
    portable = _detect_portable_ffmpeg()
    if portable.available:
        return portable

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return FFmpegAvailability(ffmpeg=ffmpeg, ffprobe=ffprobe)

    return FFmpegAvailability(ffmpeg=None, ffprobe=None)


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
    # 探测正常毫秒级完成;硬超时防止挂死的 ffprobe 卡住调用方
    # (save_segments 路由在 event loop 里同步等待它)。
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
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
    progress_callback: ProgressCallback | None = None,
    overwrite: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    tools = detect_ffmpeg()
    if not tools.available:
        raise RuntimeError("ffmpeg and ffprobe are required for video rendering.")

    input_media = Path(input_media).resolve()
    ass_path = Path(ass_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_args = _build_filter_args(ass_path, style=style)
    duration = probe_media_duration(input_media)
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
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_path),
    ]
    _run_ffmpeg_with_progress(
        command,
        cwd=ass_path.parent,
        duration=duration,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    return output_path


def burn_ass_subtitles_clip(
    input_media: str | Path,
    ass_path: str | Path,
    output_path: str | Path,
    *,
    start: float,
    end: float,
    style: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    overwrite: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    tools = detect_ffmpeg()
    if not tools.available:
        raise RuntimeError("ffmpeg and ffprobe are required for video rendering.")

    start = max(0.0, float(start))
    end = max(start + 0.1, float(end))
    duration = end - start
    input_media = Path(input_media).resolve()
    ass_path = Path(ass_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_audio = _has_audio_stream(input_media)
    filter_graph = _build_clip_filter_graph(ass_path, start=start, end=end, style=style, has_audio=has_audio)
    command = [
        tools.ffmpeg or "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_media),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        *(["-map", "[a]"] if has_audio else []),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_path),
    ]
    _run_ffmpeg_with_progress(
        command,
        cwd=ass_path.parent,
        duration=duration,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    return output_path


def _run_ffmpeg_with_progress(
    command: list[str],
    *,
    cwd: Path,
    duration: float | None,
    progress_callback: ProgressCallback | None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def _stop_process() -> None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    # -progress pipe:1 正常时每秒出进度行；阻塞读在 ffmpeg 静默挂死
    # （解码死循环/磁盘满写阻塞）时永远醒不来。读线程 + 队列轮询：
    # 取消检查每秒执行，并检测"长时间零输出"的真挂死（慢编码不算，
    # 进度行还在流）。启动探测与结尾 faststart 重写大文件可能长时间
    # 无输出，上限取宽松的 10 分钟。
    line_queue: "queue.Queue[str | None]" = queue.Queue()

    def _pump_stdout() -> None:
        try:
            for raw in process.stdout:
                line_queue.put(raw)
        finally:
            line_queue.put(None)  # EOF 哨兵

    reader = threading.Thread(target=_pump_stdout, daemon=True)
    reader.start()
    IDLE_TIMEOUT_SEC = 600.0
    last_output = time.monotonic()

    tail: list[str] = []
    last_ratio = 0.0
    while True:
        try:
            line = line_queue.get(timeout=1.0)
        except queue.Empty:
            if cancel_check and cancel_check():
                _stop_process()
                raise FFmpegCancelled("渲染已取消")
            if time.monotonic() - last_output > IDLE_TIMEOUT_SEC:
                _stop_process()
                raise RuntimeError(f"ffmpeg 已 {int(IDLE_TIMEOUT_SEC / 60)} 分钟无任何输出，判定挂死")
            continue
        if line is None:
            break
        last_output = time.monotonic()
        line = line.strip()
        if line:
            tail.append(line)
            tail = tail[-40:]
        if progress_callback is None or not duration:
            continue
        key, _, value = line.partition("=")
        seconds = None
        if key in {"out_time_us", "out_time_ms"}:
            try:
                seconds = float(value) / 1_000_000.0
            except ValueError:
                seconds = None
        elif key == "out_time":
            seconds = _parse_ffmpeg_time(value)
        if seconds is not None:
            last_ratio = max(last_ratio, max(0.0, min(1.0, seconds / duration)))
            progress_callback(last_ratio)
    return_code = process.wait()
    if return_code != 0:
        detail = "\n".join(tail[-12:])
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {detail}")
    if progress_callback is not None:
        progress_callback(1.0)


def probe_media_duration(path: str | Path) -> float | None:
    """Return media duration in seconds via ffprobe, or ``None`` if unavailable."""
    try:
        media = probe_media(path)
        duration = (media.get("format") or {}).get("duration")
        return float(duration) if duration else None
    except Exception:
        return None


def _has_audio_stream(path: str | Path) -> bool:
    try:
        media = probe_media(path)
    except Exception:
        return True
    return any(stream.get("codec_type") == "audio" for stream in media.get("streams", []))


def _parse_ffmpeg_time(value: str) -> float | None:
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception:
        return None


def _build_filter_args(ass_path: Path, *, style: Any | None = None) -> list[str]:
    filter_graph = _build_filter_graph(ass_path, style=style)
    return ["-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a?"]


def _build_clip_filter_graph(
    ass_path: Path,
    *,
    start: float,
    end: float,
    style: Any | None = None,
    has_audio: bool = True,
) -> str:
    duration = max(0.1, end - start)
    trimmed = f"[0:v]trim=start={start:.3f}:duration={duration:.3f},setpts=PTS-STARTPTS[clipv]"
    video_chain = _build_video_filter_graph(ass_path, style=style, input_label="[clipv]", output_label="[v]")
    if not has_audio:
        return f"{trimmed};{video_chain}"
    audio_chain = f"[0:a]atrim=start={start:.3f}:duration={duration:.3f},asetpts=PTS-STARTPTS[a]"
    return f"{trimmed};{video_chain};{audio_chain}"


def _build_filter_graph(ass_path: Path, *, style: Any | None = None) -> str:
    return _build_video_filter_graph(ass_path, style=style, input_label="[0:v]", output_label="[v]")


def _build_video_filter_graph(
    ass_path: Path,
    *,
    style: Any | None,
    input_label: str,
    output_label: str,
) -> str:
    subtitles = f"subtitles={_escape_filter_path(ass_path.name)}"
    if style is not None and bool(getattr(style, "mask_enabled", False)):
        height = max(1, int(getattr(style, "mask_height", 120)))
        margin_v = max(0, int(getattr(style, "mask_margin_v", 0)))
        opacity = max(0.0, min(1.0, float(getattr(style, "mask_opacity", 0.82))))
        mode = str(getattr(style, "mask_mode", "blur") or "blur")
        if mode == "bar":
            return (
                f"{input_label}"
                "drawbox="
                "x=0:"
                f"y=max(0\\,ih-{height + margin_v}):"
                "w=iw:"
                f"h=min({height}\\,ih-{margin_v}):"
                f"color=black@{opacity:.3f}:"
                f"t=fill,{subtitles}{output_label}"
            )
        blur = max(1, int(getattr(style, "mask_blur", 24)))
        region_y = f"max(0\\,ih-{height + margin_v})"
        region_h = f"min({height}\\,ih-{margin_v})"
        overlay_y = f"H-h-{margin_v}"
        return (
            f"{input_label}split=2[base][region];"
            f"[region]crop=iw:{region_h}:0:{region_y},boxblur={blur}:1[blurred];"
            f"[base][blurred]overlay=0:{overlay_y},"
            f"drawbox=x=0:y={region_y}:w=iw:h={region_h}:color=black@0.180:t=fill,"
            f"{subtitles}{output_label}"
        )
    return f"{input_label}{subtitles}{output_label}"


def _escape_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
