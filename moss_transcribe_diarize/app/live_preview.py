"""站内直播播放器：短时滚动 HLS 缓存，与视频录制完全独立。"""
from __future__ import annotations

import re
import subprocess
import tempfile
import threading
from pathlib import Path

from .ffmpeg import detect_ffmpeg


class LivePreviewService:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.session_id = None
        self.process = None
        self.temporary = None
        self.log = None

    def start(self, session_id, url, headers=None):
        with self.lock:
            if self.session_id == session_id and self.process and self.process.poll() is None:
                return self.snapshot(session_id)
            self.close()
            executable = detect_ffmpeg().ffmpeg
            if not executable:
                raise RuntimeError("未找到 FFmpeg")
            self.directory.mkdir(parents=True, exist_ok=True)
            self.temporary = tempfile.TemporaryDirectory(prefix="preview-", dir=self.directory,
                                                         ignore_cleanup_errors=True)
            folder = Path(self.temporary.name)
            self.log = (folder / "preview.log").open("ab")
            command = [executable, "-hide_banner", "-loglevel", "warning", "-nostdin"]
            if headers:
                command += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
            # 浏览器统一接收 H264/AAC；固定 2 秒关键帧，避免等待源直播的长 GOP。
            command += ["-i", url, "-map", "0:v:0?", "-map", "0:a:0?",
                        "-vf", "scale=w='min(1280,iw)':h=-2", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "24", "-pix_fmt", "yuv420p", "-force_key_frames", "expr:gte(t,n_forced*2)",
                        "-c:a", "aac", "-b:a", "128k", "-f", "hls", "-hls_time", "2",
                        "-hls_list_size", "5", "-hls_delete_threshold", "2",
                        "-hls_flags", "delete_segments+temp_file+omit_endlist",
                        "-hls_segment_filename", str(folder / "segment-%06d.ts"), str(folder / "stream.m3u8")]
            try:
                self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=self.log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.session_id = session_id
            except OSError as exc:
                self.close()
                raise RuntimeError(f"播放器启动失败：{exc}") from exc
            return self.snapshot(session_id)

    def snapshot(self, session_id):
        with self.lock:
            if session_id != self.session_id or not self.process:
                return {"state": "idle"}
            if self.process.poll() is not None:
                return {"state": "error", "error": "直播播放已中断，请停止后重试或打开原站直播间"}
            ready = (Path(self.temporary.name) / "stream.m3u8").is_file()
            return {"state": "ready" if ready else "buffering",
                    "url": f"/api/live/sessions/{session_id}/preview/stream.m3u8"}

    def media_path(self, session_id, filename):
        if not re.fullmatch(r"stream\.m3u8|segment-\d+\.ts", filename):
            raise FileNotFoundError(filename)
        with self.lock:
            if self.session_id != session_id or not self.temporary:
                raise FileNotFoundError(filename)
            path = Path(self.temporary.name) / filename
            if not path.is_file():
                raise FileNotFoundError(filename)
            return path

    def stop(self, session_id):
        with self.lock:
            if self.session_id == session_id:
                self.close()
        return {"state": "idle"}

    def close(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
            self.session_id = None
            if self.log:
                self.log.close()
                self.log = None
            if self.temporary:
                self.temporary.cleanup()
                self.temporary = None
