"""直播音频采集、有限积压和实时字幕草稿；不依赖直播录制进程。"""
from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .ffmpeg import detect_ffmpeg
from .index_translator import IndexTranslator

logger = logging.getLogger(__name__)
RATE = 16000
WINDOW_SECONDS = 6
OVERLAP_SECONDS = 2


@dataclass
class SubtitleSession:
    id: str
    directory: Path
    translate: bool = True
    language: str | None = None
    state: str = "starting"
    error: str | None = None
    warning: str | None = None
    received_seconds: float = 0.0
    processed_seconds: float = 0.0
    dropped_windows: int = 0
    translation_skipped: int = 0
    next_id: int = 0
    rows: deque = field(default_factory=lambda: deque(maxlen=300))
    audio_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=3))
    translation_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=16))
    stop: threading.Event = field(default_factory=threading.Event)
    capture_done: threading.Event = field(default_factory=threading.Event)
    asr_done: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    process: object = None
    threads: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def draft_path(self):
        return self.directory / "subtitles.jsonl"

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "error": self.error, "warning": self.warning,
                    "language": self.language, "translate": self.translate,
                    "received_seconds": round(self.received_seconds, 2),
                    "processed_seconds": round(self.processed_seconds, 2),
                    "backlog_seconds": round(max(0, self.received_seconds - self.processed_seconds), 2),
                    "dropped_windows": self.dropped_windows, "translation_skipped": self.translation_skipped,
                    "audio_queue": self.audio_queue.qsize(), "translation_queue": self.translation_queue.qsize(),
                    "draft_path": str(self.draft_path), "rows": [dict(row) for row in self.rows]}

    def save(self, row):
        # 追加事件日志，翻译结果沿用原文的 id，可在停止后恢复为最终双语草稿。
        with self.lock:
            with self.draft_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def confirmed_words(words, offset, watermark, cutoff):
    """按词中心时间分配重叠窗口，避免重复发布；末尾 1 秒留到下个窗口。"""
    return [(offset + start, offset + end, text) for start, end, text in words
            if watermark < offset + (start + end) / 2 <= cutoff]


class LiveSubtitleService:
    def __init__(self, runner, directory: Path, translator=None):
        self.runner = runner
        self.directory = Path(directory)
        self.translator = translator or IndexTranslator()
        self.sessions: dict[str, SubtitleSession] = {}
        self.lock = threading.RLock()

    def snapshot(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
        return session.snapshot() if session else {"state": "idle", "rows": []}

    def start(self, session_id, url, headers=None, *, translate=True, language=None):
        if language not in {None, "en", "ja", "zh", "ko", "de", "fr", "es", "ru"}:
            raise ValueError("不支持的识别语言")
        with self.lock:
            # 一块显卡同时承担 Whisper 和 Index，先明确限制一个实时频道。
            if any(any(t.is_alive() for t in s.threads) for s in self.sessions.values()):
                raise RuntimeError("已有实时字幕正在运行或停止，请先停止该频道")
            if translate and not self.translator.runtime_info()["available"]:
                raise RuntimeError("未找到 Index 翻译模型或 llama-server，可先关闭实时翻译仅识别原文")
            ffmpeg = detect_ffmpeg().ffmpeg
            if not ffmpeg:
                raise RuntimeError("未找到 FFmpeg")
            directory = self.directory / session_id / uuid.uuid4().hex[:12]
            directory.mkdir(parents=True, exist_ok=False)
            session = SubtitleSession(session_id, directory, translate=translate, language=language)
            command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin"]
            if headers:
                command += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
            command += ["-i", url, "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
            try:
                with (directory / "capture.log").open("ab") as log:
                    session.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as exc:
                raise RuntimeError(f"直播音频采集启动失败：{exc}") from exc
            self.sessions[session_id] = session
            session.threads = [threading.Thread(target=fn, args=(session,), daemon=True)
                               for fn in (self._capture, self._recognize, self._translate)]
            for thread in session.threads:
                thread.start()
            return session.snapshot()

    def _put_audio(self, session, item):
        try:
            session.audio_queue.put_nowait(item)
        except queue.Full:
            try:
                session.audio_queue.get_nowait()
            except queue.Empty:
                pass
            with session.lock:
                session.dropped_windows += 1
                session.warning = "识别落后于直播，已跳过最旧的音频窗口；录制视频仍可用于完整转录"
            session.audio_queue.put_nowait(item)

    def _capture(self, session):
        total = 0
        tail = b""
        try:
            while not session.stop.is_set():
                data = bytearray()
                while len(data) < RATE * 2 * WINDOW_SECONDS and not session.stop.is_set():
                    chunk = session.process.stdout.read(RATE * 2 * WINDOW_SECONDS - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                data = bytes(data[:len(data) // 2 * 2])
                final = len(data) < RATE * 2 * WINDOW_SECONDS
                if not data and not tail:
                    break
                offset = (total - len(tail) // 2) / RATE
                total += len(data) // 2
                audio = tail + data
                with session.lock:
                    session.received_seconds = total / RATE
                self._put_audio(session, (offset, audio, total / RATE, final))
                tail = audio[-RATE * 2 * OVERLAP_SECONDS:]
                if final:
                    break
            if not session.stop.is_set():
                code = session.process.wait(timeout=10)
                if code or total == 0:
                    raise RuntimeError("直播音频采集结束但未取得有效音频，请检查直播状态或登录态")
        except Exception as exc:
            with session.lock:
                session.error = str(exc)
            logger.warning("直播音频采集失败 session=%s: %s", session.id, exc)
        finally:
            if session.process.poll() is None:
                session.process.kill()
            session.process.wait()
            session.process.stdout.close()
            session.capture_done.set()

    def _recognize(self, session):
        import numpy as np

        watermark = -1.0
        try:
            while not session.stop.is_set():
                try:
                    offset, audio, end, final = session.audio_queue.get(timeout=.25)
                except queue.Empty:
                    if session.capture_done.is_set():
                        break
                    continue
                with session.lock:
                    session.state = "recognizing"
                result = self.runner.transcribe_live_window(
                    np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768,
                    language=session.language, stop_event=session.stop,
                )
                if session.stop.is_set():
                    break
                cutoff = end if final else max(offset, end - 1)
                words = confirmed_words(result["words"], offset, watermark, cutoff)
                watermark = cutoff
                text = "".join(w[2] for w in words).strip()
                with session.lock:
                    session.language = result.get("language") or session.language
                    session.processed_seconds = end
                    session.state = "listening"
                    context = [row["source"] for row in session.rows][-2:]
                    if text:
                        session.next_id += 1
                        row = {"id": session.next_id, "start": round(words[0][0], 3),
                               "end": round(words[-1][1], 3), "source": text, "translation": "",
                               "translation_state": "pending" if session.translate else "disabled"}
                        session.rows.append(row)
                        session.save(row)
                        if session.translate:
                            try:
                                session.translation_queue.put_nowait((row, context))
                            except queue.Full:
                                row["translation_state"] = "skipped"
                                session.translation_skipped += 1
                                session.warning = "翻译队列已满，该段原文保留但跳过翻译"
                                session.save(row)
        except Exception as exc:
            with session.lock:
                session.error = str(exc)
            session.stop.set()
            if session.process.poll() is None:
                session.process.kill()
            logger.exception("直播识别失败 session=%s", session.id)
        finally:
            session.asr_done.set()

    def _translate(self, session):
        try:
            while True:
                try:
                    row, context = session.translation_queue.get(timeout=.25)
                except queue.Empty:
                    if session.asr_done.is_set():
                        break
                    continue
                if session.stop.is_set():
                    with session.lock:
                        row["translation_state"] = "cancelled"
                        session.save(row)
                    continue
                try:
                    translated = self.translator.translate_live(row["source"], context)
                    with session.lock:
                        row["translation"] = translated
                        row["translation_state"] = "done"
                        session.save(row)
                except Exception as exc:
                    with session.lock:
                        row["translation_state"] = "failed"
                        row["translation_error"] = str(exc)
                        session.warning = "部分翻译失败，原文已保留，可查看该条错误后重试会话"
                        session.save(row)
                    logger.warning("直播翻译失败 session=%s: %s", session.id, exc)
        finally:
            # 模型仅在实时翻译会话期间占用显存，停止后释放本服务启动的进程。
            self.translator.close()
            with session.lock:
                session.state = "error" if session.error else "stopped"

    def stop(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
        if session:
            session.stop.set()
            with session.lock:
                if any(t.is_alive() for t in session.threads):
                    session.state = "stopping"
            if session.process.poll() is None:
                session.process.terminate()
            # 推理正在等待共享模型时不能强制释放模型；前端继续显示停止中。
            deadline = time.monotonic() + 3
            for thread in session.threads:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        return self.snapshot(session_id)

    def close(self):
        for session_id in list(self.sessions):
            self.stop(session_id)
        if not any(any(t.is_alive() for t in s.threads) for s in self.sessions.values()):
            self.translator.close()
