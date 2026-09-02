from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[float, dict[str, Any]], None]
CancelCheck = Callable[[], bool]
TitleCallback = Callable[[str], None]

_PROGRESS_RE = re.compile(
    r"\[download\]\s+([\d.]+)%"
    r"(?:\s+of\s+~?\s*([\d.]+)\s*(K|M|G)?i?B)?"
    r"(?:\s+at\s+~?\s*([\d.]+)\s*(K|M|G)?i?B/s)?"
    r"(?:\s+ETA\s+(\d{1,2}:\d{2}(?::\d{2})?))?"
)

_SIZE_MULT = {"K": 1024, "M": 1024**2, "G": 1024**3}

_MERGE_RE = re.compile(r"\(Merger|Deleting\)", re.I)

_FILEPATH_RE = re.compile(r"^(?:Destination:|File is already|Merging|Deleting)\s+(.+)$", re.I)

_NETSCAPE_COOKIE_RE = re.compile(r"^(?:#\s*(?:HTTP|Netscape)|\S+\t\S+\t\S+)", re.MULTILINE)

# --print 输出自定义前缀模板，避免和 yt-dlp 的状态行/警告混淆。
_TITLE_PRINT_PREFIX = "mtd_title:"

# 兜底搜索下载产物时只认媒体后缀,防止把同目录的字幕文件误当视频 remux。
_MEDIA_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov", ".ts", ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}


@dataclass(slots=True)
class DownloadResult:
    path: Path
    title: str | None = None
    # 下载到的源字幕文件：{"lang": "en", "path": "...", "kind": "manual"|"auto"}
    subtitles: list[dict[str, str]] = field(default_factory=list)


# 平台字幕抓取范围：匹配不到就一个文件都没有，宁可多列几种。
# auto 原生轨在 YouTube 上会带 -orig 后缀（如 en-orig）。
_SUB_LANGS = "en.*,zh.*,ja.*,ko.*"
_SUB_LANG_PRIORITY = ("zh-Hans", "zh-Hant", "zh", "en", "ja", "ko")
# 源语-目标语双语种码 = 机翻轨（en-zh-Hans）；en-orig/en-US 不匹配。
_TRANSLATED_LANG_RE = re.compile(r"^[a-z]{2,3}-[a-z]{2,3}(?:[-.].*)?$")


def pick_best_subtitle(subtitles: list[dict[str, str]]) -> dict[str, str] | None:
    """优先人工字幕；同类里按常见语种顺序挑（字幕语言应贴近音频语言）。"""
    if not subtitles:
        return None

    def rank(entry: dict[str, str]) -> tuple[int, int]:
        kind_rank = 0 if entry.get("kind") == "manual" else 1
        lang = entry.get("lang", "")
        for i, prefix in enumerate(_SUB_LANG_PRIORITY):
            if lang == prefix or lang.startswith(prefix):
                return (kind_rank, i)
        return (kind_rank, len(_SUB_LANG_PRIORITY))

    return sorted(subtitles, key=rank)[0]


def find_yt_dlp() -> Path:
    candidates = [
        Path.cwd() / "tools" / "yt-dlp" / "yt-dlp.exe",
        Path(__file__).resolve().parent.parent.parent / "tools" / "yt-dlp" / "yt-dlp.exe",
        Path(sys.argv[0]).resolve().parent / "tools" / "yt-dlp" / "yt-dlp.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return Path("yt-dlp")


def _find_js_runtime(yt_dlp_path: Path) -> str | None:
    exe_dir = yt_dlp_path.resolve().parent
    for name in ("deno.exe", "deno"):
        p = exe_dir / name
        if p.is_file():
            return f"deno:{p}"
    import shutil
    if shutil.which("deno"):
        return "deno"
    for name in ("node.exe", "node"):
        p = exe_dir / name
        if p.is_file():
            return f"node:{p}"
    if shutil.which("node"):
        return "node"
    return None


def _find_ffmpeg_dir() -> Path | None:
    candidates = [
        Path.cwd() / "tools" / "ffmpeg",
        Path(__file__).resolve().parent.parent.parent / "tools" / "ffmpeg",
    ]
    for p in candidates:
        if (p / "ffmpeg.exe").is_file():
            return p
    import shutil
    if shutil.which("ffmpeg"):
        return None
    return None


def _parse_progress_line(line: str) -> dict[str, Any] | None:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    pct_str, size_val, size_unit, speed_val, speed_unit, eta = m.groups()
    try:
        pct = float(pct_str)
    except (ValueError, TypeError):
        return None
    info: dict[str, Any] = {"percent": round(pct, 1)}
    if size_val and size_unit:
        try:
            info["total_size"] = f"{float(size_val)}{size_unit}iB"
        except ValueError:
            pass
    if speed_val and speed_unit:
        try:
            info["speed"] = f"{float(speed_val)}{speed_unit}iB/s"
        except ValueError:
            pass
    if eta:
        info["eta"] = eta
    return info


def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "00:00"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def check_cookies_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"valid": False, "error": f"文件不存在: {p}"}
    if p.stat().st_size > 5 * 1024 * 1024:
        return {"valid": False, "error": "文件过大（>5MB），可能格式不对"}
    raw = p.read_bytes()
    if raw[:15] == b"SQLite format 3":
        return {"valid": False, "error": "文件是 SQLite 数据库，不是 Netscape cookies 格式"}
    if b"\x00" in raw[:1024]:
        return {"valid": False, "error": "文件包含二进制数据，不是文本格式"}
    text = raw.decode("utf-8", errors="replace")
    if not _NETSCAPE_COOKIE_RE.search(text):
        return {"valid": False, "error": "文件内容不像 Netscape cookies 格式"}
    domains = set()
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            domains.add(parts[0])
    return {
        "valid": True,
        "domain_count": len(domains),
        "domains": sorted(domains)[:20],
    }


def check_browser_cookies(browser: str) -> dict[str, Any]:
    browser = (browser or "").lower().strip()
    if browser in ("", "none", "off"):
        return {"valid": False, "error": "未指定浏览器"}
    known = {"firefox", "chrome", "chromium", "edge", "brave", "opera", "vivaldi", "safari"}
    if browser not in known:
        return {"valid": False, "error": f"不支持的浏览器: {browser}"}
    if browser == "firefox":
        profiles_dir = Path.home() / "AppData" / "Roaming" / "Mozilla" / "Firefox" / "Profiles"
        if not profiles_dir.is_dir():
            return {"valid": False, "error": f"未找到 Firefox 配置目录: {profiles_dir}"}
        cookie_files = list(profiles_dir.glob("*/cookies.sqlite"))
        if not cookie_files:
            return {"valid": False, "error": "Firefox 已安装但未找到 cookies.sqlite，请先在 Firefox 中登录视频网站"}
        return {"valid": True, "profile": str(cookie_files[0].parent.name)}
    if browser in ("chrome", "chromium", "edge", "brave", "opera", "vivaldi"):
        win_browser_map = {
            "chrome": "Google\\Chrome",
            "chromium": "Chromium",
            "edge": "Microsoft\\Edge",
            "brave": "BraveSoftware\\Brave-Browser",
            "opera": "Opera Software\\Opera Stable",
            "vivaldi": "Vivaldi\\User Data",
        }
        base = Path.home() / "AppData" / "Local" / win_browser_map.get(browser, "")
        cookie_path = base / "User Data" / "Default" / "Network" / "Cookies"
        if browser == "edge":
            cookie_path = base / "User Data" / "Default" / "Network" / "Cookies"
        if not cookie_path.exists():
            return {"valid": False, "error": f"未找到 {browser} 的 cookies 文件（新版浏览器可能使用了 App-Bound Encryption，yt-dlp 可能无法读取）"}
        return {"valid": True, "warning": "新版浏览器可能使用 App-Bound Encryption，cookies 可能无法读取，建议使用 Firefox 或 cookies.txt"}
    return {"valid": False, "error": f"不支持: {browser}"}


def download_with_yt_dlp(
    url: str,
    output_dir: str | Path,
    *,
    cookies_browser: str | None = "firefox",
    cookies_file: str | Path | None = None,
    format_selector: str = "bv*[height<=1080]+ba/b[height<=1080]",
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    title_callback: TitleCallback | None = None,
    extra_args: list[str] | None = None,
) -> DownloadResult:
    yt_dlp = find_yt_dlp()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "input.%(ext)s")

    cmd = [
        str(yt_dlp),
        "--newline",
        "--no-playlist",
        "--continue",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "-f",
        format_selector,
        "-o",
        output_template,
        "--no-write-info-json",
        "--no-write-thumbnail",
        # 只抓人工上传的平台字幕（CC）；auto-CC 质量通常不如 whisper，不走它。
        "--write-subs",
        "--sub-langs",
        _SUB_LANGS,
        "--convert-subs",
        "srt",
        # --print 默认等价于 --simulate 且隐式激活 quiet（关闭全部进度输出），
        # 必须显式 --no-simulate 恢复真实下载、显式 --progress 恢复进度行。
        "--no-simulate",
        "--print",
        f"{_TITLE_PRINT_PREFIX}%(title)s",
        "--progress",
    ]

    js_runtime = _find_js_runtime(yt_dlp)
    if js_runtime:
        cmd += ["--js-runtimes", js_runtime]

    ffmpeg_dir = _find_ffmpeg_dir()
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", str(ffmpeg_dir)]

    if cookies_file:
        cmd += ["--cookies", str(cookies_file)]
    elif cookies_browser and cookies_browser.lower() not in ("", "none", "off"):
        cmd += ["--cookies-from-browser", cookies_browser.lower()]

    if extra_args:
        cmd += extra_args

    cmd.append(url)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        # PyInstaller 打包的 yt-dlp 在管道 stdout 下按块缓冲，进度行会长期积压
        # 导致上层永远拿到 0%；PYTHONUNBUFFERED 强制逐行写出。
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            creationflags=creationflags,
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"无法启动 yt-dlp，路径: {yt_dlp} (exists={yt_dlp.is_file()}). "
            f"请重启服务以加载新路径。"
        ) from e

    downloaded_file: Path | None = None
    last_progress: dict[str, Any] = {}
    error_lines: list[str] = []
    video_title: str | None = None

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        if not line:
            continue

        # 取消检查必须在任何 continue 分支之前:下载主阶段几乎全是
        # [download] xx% 进度行,放后面会让取消在整个下载期间失效。
        if cancel_check and cancel_check():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError("下载已取消")

        if line.startswith(_TITLE_PRINT_PREFIX):
            raw_title = line[len(_TITLE_PRINT_PREFIX):].strip()
            if raw_title and not video_title:
                video_title = raw_title
                if title_callback:
                    title_callback(raw_title)
            continue

        if line.startswith("Destination:"):
            m = re.match(r"Destination:\s+(.+)", line)
            if m:
                downloaded_file = Path(m.group(1).strip())
            continue

        if line.startswith("[Merger] Merging formats into") or line.startswith("Merging formats into"):
            m = re.search(r'"(.+?)"', line)
            if m:
                downloaded_file = Path(m.group(1))
            continue

        if line.startswith("[download]") and "%" in line:
            info = _parse_progress_line(line)
            if info:
                last_progress = info
                ratio = max(0.0, min(1.0, info["percent"] / 100.0))
                if progress_callback:
                    progress_callback(ratio, info)
                continue

        if line.startswith("ERROR:") or line.startswith("WARNING:"):
            error_lines.append(line)

    ret = proc.wait()

    if ret != 0:
        error_msg = "; ".join(error_lines[-5:]) if error_lines else f"yt-dlp 退出码 {ret}"
        raise RuntimeError(error_msg)

    if not downloaded_file or not downloaded_file.is_file():
        downloaded_file = _pick_media_file(out_dir)
        if downloaded_file is None:
            raise FileNotFoundError(f"下载完成但未找到输出文件，请检查 {out_dir}")

    if downloaded_file.suffix.lower() != ".mkv":
        downloaded_file = _remux_to_mkv(downloaded_file)

    subtitles = _collect_subtitle_files(out_dir)
    return DownloadResult(path=downloaded_file, title=video_title, subtitles=subtitles)


def _pick_media_file(out_dir: Path) -> Path | None:
    """按 mtime 挑最新的媒体文件作为下载产物;字幕文件永不入选。"""
    candidates = sorted(
        (p for p in out_dir.glob("input.*") if p.suffix.lower() in _MEDIA_SUFFIXES),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _collect_subtitle_files(out_dir: Path) -> list[dict[str, str]]:
    """扫描 `input.<lang>.srt`，标记 manual（无 -orig/-auto 后缀）或 auto。"""
    entries: list[dict[str, str]] = []
    for srt in sorted(out_dir.glob("input.*.srt")):
        lang = srt.stem[len("input."):]
        if not lang:
            continue
        # YouTube 的机翻轨形如 en-zh-Hans（源语-目标语）；排除，避免误当文稿。
        if _TRANSLATED_LANG_RE.match(lang):
            continue
        kind = "auto" if ("-orig" in lang or "-auto" in lang) else "manual"
        entries.append({"lang": lang, "path": str(srt), "kind": kind})
    return entries


def _remux_to_mkv(src: Path) -> Path:
    dst = src.with_suffix(".mkv")
    if dst.exists():
        dst.unlink()
    ffmpeg_dir = _find_ffmpeg_dir()
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe" if ffmpeg_dir else "ffmpeg"
    cmd = [str(ffmpeg_exe), "-i", str(src), "-c", "copy", "-y", str(dst)]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"无法启动 ffmpeg，路径: {ffmpeg_exe} (exists={Path(ffmpeg_exe).is_file()}). "
            f"请重启服务以加载新路径。"
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 转封装失败: {proc.stderr[-500:] if proc.stderr else '未知错误'}")
    src.unlink()
    return dst
