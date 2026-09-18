"""HY-MT2 本地翻译引擎: 按需拉起 llama.cpp 的 llama-server, 走纯 MT 直译路径。

与 ``LocalMtTranslator`` (CTranslate2 的 OPUS-MT) 的区别:
- 模型是 1.8B 的纯翻译模型, 不吐 JSON, 所以不能走 ``TextTranslator`` 的
  JSON 提示词路径 (那条路解析失败会二分降级并静默返回原文)。
- llama-server 是常驻进程, 但显存只有 8GB 且要和 Whisper/Demucs/pyannote 抢,
  所以这里按需启动 (实测冷启动约 1.7s), 用完不主动杀, 由进程退出时清理。
- 采样必须 temperature=0: 模型内置的 0.7/0.8/20 会把专有名词改写
  (实测 GeoGuessr 会被翻成《猜猜我是谁》)。
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from moss_transcribe_diarize.subtitle import SubtitleSegment

from .text_translator import translation_skip_reason

logger = logging.getLogger(__name__)

DEFAULT_SERVER_DIR = Path("tools/llama-server")
DEFAULT_MODEL_PATH = Path("models/Hy-MT2-1.8B-Q4_K_M.gguf")
DEFAULT_PORT = 8090
DEFAULT_PROMPT_TEMPLATE = "Translate the following segment into {target}, without additional explanation: {text}"

# 官方提示词只给了中文的写法, 其余目标语言按英文名回退。
_TARGET_NAMES = {
    "简体中文": "Chinese",
    "繁體中文": "Traditional Chinese",
    "中文": "Chinese",
    "英语": "English",
    "英文": "English",
    "日语": "Japanese",
    "日文": "Japanese",
    "韩语": "Korean",
}


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _detect_server_executable(server_dir: Path | str | None, explicit: str | Path | None = None) -> Path | None:
    """定位 llama-server 可执行文件。查找顺序与 ``ffmpeg.detect_ffmpeg`` 一致: 显式路径 > tools 目录 > PATH。"""
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None

    root = Path(__file__).resolve().parents[2]
    tools_dir = root / "tools"
    dirs = [Path(server_dir).expanduser()] if server_dir else []
    dirs.extend([tools_dir / "llama-server" / "bin", tools_dir / "llama-server"])
    if tools_dir.exists():
        for child in tools_dir.iterdir():
            if child.is_dir():
                dirs.extend([child / "bin", child])

    for directory in dirs:
        if not directory.is_dir():
            continue
        for name in ("llama-server.exe", "llama-server"):
            candidate = directory / name
            if candidate.is_file():
                return candidate

    found = shutil.which("llama-server")
    return Path(found) if found else None


@dataclass(slots=True)
class HyMtTranslator:
    model_path: str | Path = DEFAULT_MODEL_PATH
    server_dir: str | Path | None = DEFAULT_SERVER_DIR
    server_exe: str | Path | None = None
    port: int = DEFAULT_PORT
    context_size: int = 16384
    parallel_slots: int = 8
    batch_size: int = 2048
    ubatch_size: int = 512
    gpu_layers: int = 99
    concurrency: int = 16
    timeout: float = 300.0
    startup_timeout: float = 90.0
    model: str = "HY-MT2-1.8B"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _startup_error: str | None = field(default=None, init=False, repr=False)
    failure_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # 必须绝对化: 子进程的 cwd 是 exe 所在目录, 相对路径会解析到 tools/ 下面。
        self.model_path = Path(self.model_path).resolve()
        self.concurrency = max(1, int(self.concurrency))
        atexit.register(self.close)

    # ---------------------------------------------------------------- 状态

    @property
    def executable(self) -> Path | None:
        return _detect_server_executable(self.server_dir, self.server_exe)

    def runtime_info(self) -> dict[str, Any]:
        """只做静态检查, 不拉起服务 —— 供 /api/runtime 每次轮询调用。"""
        exe = self.executable
        model_ok = self.model_path.is_file()
        available = exe is not None and model_ok
        info: dict[str, Any] = {
            "available": available,
            "backend": "hy-mt",
            "model": self.model,
            "model_path": str(self.model_path),
            "model_found": model_ok,
            "server_exe": str(exe) if exe else None,
            "server_found": exe is not None,
            "port": self.port,
            "running": self._is_running(),
            "concurrency": self.concurrency,
        }
        if not model_ok:
            info["reason"] = f"未找到模型文件：{self.model_path}"
        elif exe is None:
            info["reason"] = "未找到 llama-server，请把 llama.cpp 解压到 tools/llama-server/"
        return info

    def _is_running(self) -> bool:
        """只看子进程句柄, 不发探活请求 —— ``runtime_info`` 会被前端高频轮询。"""
        return self._process is not None and self._process.poll() is None

    def _health_ok(self) -> bool:
        try:
            with urllib.request.urlopen(self._base_url() + "/health", timeout=2.0) as resp:
                return b"ok" in resp.read()
        except Exception:  # noqa: BLE001 - 健康检查失败一律视为未就绪
            return False

    def _base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ---------------------------------------------------------------- 进程

    def _ensure_server(self) -> None:
        with self._lock:
            if self._health_ok():
                return
            exe = self.executable
            if exe is None:
                raise RuntimeError(
                    "未找到 llama-server。请把 llama.cpp 的 Windows 版解压到 "
                    f"{DEFAULT_SERVER_DIR}（需要 llama-server.exe 及同目录的 CUDA 运行库）。"
                )
            if not self.model_path.is_file():
                raise RuntimeError(f"未找到 HY-MT2 模型文件：{self.model_path}")
            if not _port_is_free(self.port) and not self._health_ok():
                raise RuntimeError(f"端口 {self.port} 已被其它程序占用，无法启动本地 HY-MT2 服务。")

            argv = [
                str(exe),
                "-m", str(self.model_path),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "-ngl", str(self.gpu_layers),
                "-c", str(self.context_size),
                "-np", str(self.parallel_slots),
                "-b", str(self.batch_size),
                "-ub", str(self.ubatch_size),
            ]
            logger.info("启动本地 HY-MT2 服务: %s", " ".join(argv))
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self._process = subprocess.Popen(
                    argv,
                    cwd=str(exe.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except OSError as exc:
                raise RuntimeError(f"无法启动 llama-server：{exc}") from exc

            deadline = time.time() + self.startup_timeout
            while time.time() < deadline:
                if self._process.poll() is not None:
                    code = self._process.returncode
                    self._process = None
                    raise RuntimeError(
                        f"llama-server 启动后立即退出（退出码 {code}）。"
                        f"请确认模型文件可读、显存够用（约需 2.4GB），"
                        f"以及 {exe.parent} 下有完整的 CUDA 运行库。"
                    )
                if self._health_ok():
                    logger.info("本地 HY-MT2 服务就绪（端口 %s）", self.port)
                    return
                time.sleep(0.15)
            self.close()
            raise RuntimeError(f"llama-server 在 {self.startup_timeout:.0f}s 内未就绪。")

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except Exception:  # noqa: BLE001 - 退出清理尽力而为
            try:
                process.kill()
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- 翻译

    def translate_segments(
        self,
        segments: Iterable[SubtitleSegment],
        *,
        target_language: str = "简体中文",
        batch_size: int = 32,
        context_window: int = 0,
        semantic_units: bool = False,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> list[str]:
        del batch_size, context_window, semantic_units
        items = list(segments)
        if not items:
            return []

        results: list[str | None] = [None] * len(items)
        pending: list[int] = []
        for index, segment in enumerate(items):
            text = str(segment.text or "").strip()
            if translation_skip_reason(text) is not None:
                results[index] = str(segment.text or "")
            else:
                pending.append(index)

        if pending:
            self._ensure_server()
            template = self.prompt_template.format(
                target=_TARGET_NAMES.get(target_language, target_language),
                text="{text}",
            )
            done = len(items) - len(pending)
            if progress_callback is not None:
                progress_callback(done, len(items), 0, 0)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(self._translate_one, template, items[i].text): i for i in pending}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:  # noqa: BLE001 - 单段失败降级为原文, 但计数并告警
                        self.failure_count += 1
                        logger.warning("HY-MT2 翻译第 %d 段失败: %s", index, exc)
                        results[index] = str(items[index].text or "")
                    done += 1
                    if progress_callback is not None:
                        progress_callback(min(done, len(items)), len(items), index, 1)

        return [value if value is not None else str(items[i].text or "") for i, value in enumerate(results)]

    def _translate_one(self, template: str, text: str) -> str:
        body = {
            "messages": [{"role": "user", "content": template.format(text=str(text or "").strip())}],
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": False,
        }
        request = urllib.request.Request(
            self._base_url() + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"请求本地 HY-MT2 服务失败：{exc}") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("本地 HY-MT2 服务返回了空结果。")
        return str(choices[0].get("message", {}).get("content") or "").strip()
