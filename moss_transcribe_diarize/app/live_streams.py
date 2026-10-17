"""直播源解析、观看和录制会话管理。"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .ffmpeg import detect_ffmpeg, probe_media_duration
from .live_resolver import LiveResolver

logger = logging.getLogger(__name__)

# 平台解析通过独立的 OSRP/Streamlink 适配器，录制由本项目管理 FFmpeg。
# 清单用于前端提示和平台识别，不声称每个平台在本机均已验证。
SUPPORTED_PLATFORM_HOSTS = {
    "哔哩哔哩": ("live.bilibili.com",), "抖音": ("live.douyin.com",),
    "斗鱼": ("douyu.com",), "虎牙": ("huya.com",), "快手": ("live.kuaishou.com",),
    "Twitch": ("twitch.tv",), "YouTube Live": ("youtube.com", "youtu.be"),
    "AcFun": ("live.acfun.cn",), "战旗": ("zhanqi.tv",), "CC直播": ("cc.163.com",),
    "一直播": ("yizhibo.com",), "OPENREC": ("openrec.tv",), "企鹅电竞": ("egame.qq.com",),
    "浪Live": ("lang.live",), "花椒": ("huajiao.com",), "猫耳": ("missevan.com",),
    "克拉克拉": ("hongdoufm.com",), "YY": ("yy.com",), "微博直播": ("weibo.com",),
    "SOOP": ("sooplive.com",), "小红书": ("xiaohongshu.com",),
}


@dataclass
class LiveSession:
    id: str
    url: str
    platform: str
    title: str = ""
    status: str = "idle"
    stream_url: str = ""
    stream_headers: dict[str, str] = field(default_factory=dict, repr=False)
    quality: str = "best"
    recording_path: str = ""
    started_at: float | None = None
    stopped_at: float | None = None
    error: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    worker: threading.Thread | None = field(default=None, repr=False)
    log_handle: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "url": self.url, "platform": self.platform, "title": self.title,
            "status": self.status, "stream_url": self.stream_url, "quality": self.quality,
            "recording_path": self.recording_path, "started_at": self.started_at,
            "stopped_at": self.stopped_at, "error": self.error,
        }


class LiveStreamManager:
    def __init__(self, videos_dir: str | Path = "videos") -> None:
        self.videos_dir = Path(videos_dir).resolve()
        self.live_dir = self.videos_dir / "live"
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.resolver = LiveResolver(self.videos_dir.parent)
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.RLock()
        self.state_path = self.videos_dir.parent / "config" / "live-sessions.json"
        self._load()

    def _load(self):
        if not self.state_path.is_file():
            return
        try:
            for data in json.loads(self.state_path.read_text(encoding="utf-8")):
                session = LiveSession(**{k: v for k, v in data.items() if k in {"id", "url", "platform", "title", "quality", "recording_path", "started_at", "stopped_at"}})
                session.status = "completed" if session.recording_path else "idle"
                self._sessions[session.id] = session
        except (OSError, ValueError, TypeError):
            logger.warning("直播列表读取失败", exc_info=True)

    def _save(self):
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = [{k: v for k, v in item.to_dict().items() if k not in {"stream_url", "status", "error"}} for item in self._sessions.values()]
            temporary = self.state_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.state_path)

    @staticmethod
    def _platform(url: str) -> str:
        lower = (urlparse(url).hostname or "").lower()
        for name, hosts in SUPPORTED_PLATFORM_HOSTS.items():
            if any(lower == host or lower.endswith("." + host) for host in hosts):
                return name
        return "其它平台"

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [item.to_dict() for item in self._sessions.values()]

    def add(self, url: str, quality: str = "best") -> LiveSession:
        url = str(url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("直播链接必须以 http:// 或 https:// 开头")
        session = LiveSession(uuid.uuid4().hex[:12], url, self._platform(url), quality=quality or "best")
        with self._lock:
            self._sessions[session.id] = session
        self.resolve(session.id)
        return session

    def get(self, session_id: str) -> LiveSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def resolve(self, session_id: str) -> LiveSession:
        session = self.get(session_id)
        with session.lock:
            if session.process is not None:
                raise RuntimeError("录制中或文件正在保存，请先停止录制并等待保存完成")
            session.stream_url = ""
            session.stream_headers = {}
            try:
                resolved = self.resolver.resolve(session.url, session.quality)
                session.stream_url = str(resolved.get("stream_url") or "")
                session.stream_headers = {str(k): str(v) for k, v in (resolved.get("headers") or {}).items()}
                session.title = str(resolved.get("title") or resolved.get("anchor") or session.title)
                session.status = "online" if resolved.get("online") and session.stream_url else "offline"
                session.error = None
            except Exception as exc:
                session.status = "error"
                session.error = str(exc)
                logger.warning("直播解析失败 (%s): %s", session.platform, exc)
        self._save()
        return session

    def start_recording(self, session_id: str) -> LiveSession:
        session = self.get(session_id)
        if session.process is None:
            self.resolve(session_id)
        with session.lock:
            if session.process and session.process.poll() is None:
                return session
            if session.process is not None:
                raise RuntimeError("上一次录制正在保存，请稍后再开始")
            if not session.stream_url:
                raise RuntimeError(session.error or "当前没有可用直播流")
            ffmpeg = detect_ffmpeg().ffmpeg
            if not ffmpeg:
                raise RuntimeError("未找到 FFmpeg")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            output = self.live_dir / f"{session.platform}-{session.id}-{stamp}.mkv"
            started = time.time()
            log_path = output.with_suffix(".log")
            log_handle = log_path.open("ab")
            command = [ffmpeg, "-hide_banner", "-loglevel", "warning"]
            if session.stream_headers:
                header_blob = "".join(f"{key}: {value}\r\n" for key, value in session.stream_headers.items())
                command += ["-headers", header_blob]
            command += ["-i", session.stream_url, "-c", "copy", "-f", "matroska", str(output)]
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=log_handle, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError) as exc:
                log_handle.close()
                output.unlink(missing_ok=True)
                raise RuntimeError(f"FFmpeg 启动失败：{exc}") from exc
            session.recording_path = str(output)
            session.started_at = started
            session.stopped_at = None
            session.error = None
            session.log_handle = log_handle
            session.process = process
            session.status = "recording"
            session.worker = threading.Thread(target=self._watch_recording, args=(session, session.process), daemon=True)
            session.worker.start()
        return session

    def _watch_recording(self, session: LiveSession, process) -> None:
        process.wait()
        with session.lock:
            session.stopped_at = time.time()
            expected_stop = session.status == "stopping"
            output = Path(session.recording_path) if session.recording_path else None
            valid_output = bool(output and output.is_file() and output.stat().st_size > 0 and probe_media_duration(output))
            session.status = "completed" if valid_output and (expected_stop or process.returncode == 0) else "error"
            if not valid_output:
                session.error = "录制没有生成有效媒体文件，请检查直播是否中断或登录态"
            elif process.returncode and not expected_stop:
                session.error = f"录制意外停止（退出码 {process.returncode}），请检查连接后重试"
            if process.stdin:
                process.stdin.close()
            if session.log_handle:
                session.log_handle.close()
            session.process = None
        self._save()

    def stop_recording(self, session_id: str) -> LiveSession:
        session = self.get(session_id)
        with session.lock:
            process = session.process
            if process and process.poll() is None:
                session.status = "stopping"
                try:
                    process.stdin.write(b"q\n")
                    process.stdin.flush()
                except (OSError, ValueError):
                    pass
        if process:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                session.error = "录制进程未正常响应停止，已强制结束；请检查录制文件完整性"
            if session.worker:
                session.worker.join(timeout=5)
        return session

    def remove(self, session_id: str) -> None:
        self.get(session_id)
        self.stop_recording(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)
        self._save()

    def shutdown(self) -> None:
        for session in self.list():
            try:
                self.stop_recording(session["id"])
            except Exception:
                logger.exception("关闭直播录制失败")
        self.resolver.close()
        self._save()

