"""音频时间轴完整性检测与修复。

录屏软件在高负载下会丢音频帧，写出的容器里音频包 PTS 出现窟窿：
播放器按 PTS 播放（窟窿处静音跳过），而解码器拼接样本时窟窿消失，
导致转写时间轴整体前移、且随累计窟窿逐渐加大（字幕越来越早于语音）。

检测：取音频包 PTS 序列，中位帧距即正常帧长，相邻包 PTS 跳变超过
1.5 帧的部分即为窟窿。修复：aresample=async=1 按时间戳往窟窿里插静音，
重抽出的音频与视频播放时间轴完全对齐（已实测验证为插静音而非拉伸）。
"""

import json
import subprocess
from pathlib import Path

from .ffmpeg import detect_ffmpeg

# 修复触发阈值（秒）：累计窟窿低于该值时时间轴漂移不可感知，不值得重抽
HOLE_FIX_THRESHOLD = 0.15


def _holes_from_pts(pts: list[float]) -> list[tuple[float, float]] | None:
    """从音频包 PTS 序列计算窟窿。

    返回 [(窟窿在拼接后时间轴上的位置, 窟窿秒数)]，无窟窿返回 []，
    帧距分布过散（变长帧编码，无法可靠判定）返回 None。
    """
    if len(pts) < 8:
        return None
    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if len(deltas) < 4:
        return None
    # 中位数用排序副本取：就地 sort 会破坏时间顺序，窟窿位置就全错了
    frame = sorted(deltas)[len(deltas) // 2]
    if frame <= 0:
        return None
    # 变长帧编码（如 Vorbis）的帧距天然分散，硬判窟窿会大量误报
    tight = sum(1 for d in deltas if frame * 0.75 <= d <= frame * 1.25)
    if tight < len(deltas) * 0.98:
        return None

    holes: list[tuple[float, float]] = []
    stitched = pts[0]
    for d in deltas:
        if d > frame * 1.5:
            holes.append((stitched, d - frame))
        stitched += d
    return holes


def analyze_audio_timeline(path: str | Path) -> list[tuple[float, float]] | None:
    """分析媒体文件的音频时间轴窟窿。

    返回 [(拼接位置, 窟窿秒数)]；音频干净返回 []；无法分析（无音频流、
    ffprobe 失败、变长帧编码等）返回 None。
    """
    tools = detect_ffmpeg()
    ffprobe = tools.ffprobe
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "json",
                str(Path(path).expanduser()),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        packets = json.loads(proc.stdout).get("packets") or []
    except json.JSONDecodeError:
        return None
    pts: list[float] = []
    for packet in packets:
        raw = packet.get("pts_time")
        if raw is None:
            continue
        try:
            pts.append(float(raw))
        except (TypeError, ValueError):
            continue
    return _holes_from_pts(pts)


def total_hole_seconds(holes: list[tuple[float, float]] | None) -> float:
    if not holes:
        return 0.0
    return sum(size for _, size in holes)


def extract_timeline_fixed_audio(src: str | Path, dest: str | Path) -> bool:
    """重抽音频并在时间轴窟窿处插静音，输出 16k 单声道 wav。"""
    tools = detect_ffmpeg()
    ffmpeg = tools.ffmpeg
    if not ffmpeg:
        return False
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(Path(src).expanduser()),
                "-vn",
                "-af",
                "aresample=async=1:first_pts=0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(dest),
                "-y",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and dest.exists()
