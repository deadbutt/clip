"""BiliLiveGo 全平台解析桥接，Streamlink 提供补充适配。"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


class LiveResolver:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.base_url = os.environ.get("BILILIVE_GO_URL", "http://127.0.0.1:18760").rstrip("/")
        self._process = None
        self._log = None
        self._lock = threading.Lock()
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def executable(self):
        path = self.root / "tools" / "bililive-go" / ("bililive-go.exe" if os.name == "nt" else "bililive-go")
        return path if path.is_file() else None

    def runtime_info(self):
        return {"bililive_installed": self.executable is not None, "streamlink_installed": importlib.util.find_spec("streamlink") is not None,
                "bililive_running": self._process is not None and self._process.poll() is None}

    def _request(self, path, payload=None, timeout=30):
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode() if payload is not None else None,
                                         headers={"Content-Type": "application/json"})
        try:
            with self._http.open(request, timeout=timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                result = json.loads(exc.read())
            except ValueError:
                raise RuntimeError(f"直播解析服务错误 {exc.code}") from exc
        if not result.get("success"):
            raise RuntimeError((result.get("error") or {}).get("message") or "直播解析失败")
        return result.get("data") or {}

    def _ensure_backend(self):
        with self._lock:
            try:
                self._request("/osrp/v1/info", timeout=1)
                return
            except (OSError, RuntimeError):
                pass
            if not self.executable:
                raise RuntimeError("未安装 BiliLiveGo 引擎，请放入 tools/bililive-go/")
            address = urlparse(self.base_url)
            if address.hostname != "127.0.0.1" or address.port != 18760:
                raise RuntimeError("自定义 BiliLiveGo 服务未启动")
            if self._process is None or self._process.poll() is not None:
                directory = self.root / "config" / "live-engine"
                directory.mkdir(parents=True, exist_ok=True)
                (self.root / "videos" / "live").mkdir(parents=True, exist_ok=True)
                config = directory / "config.yml"
                if not config.exists():
                    config.write_text('rpc:\n  enable: true\n  bind: 127.0.0.1:18760\ninterval: 30\n'
                                      + 'out_put_path: ' + json.dumps(str(self.root / "videos" / "live")) + '\n'
                                      + 'app_data_path: ' + json.dumps(str(directory / "data")) + '\n'
                                      + 'live_rooms: []\n', encoding="utf-8")
                if self._log:
                    self._log.close()
                self._log = (directory / "engine.log").open("ab")
                self._process = subprocess.Popen([str(self.executable), "--no-launcher", "-c", str(config)], cwd=directory,
                                                 stdout=self._log, stderr=subprocess.STDOUT,
                                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError("直播解析引擎启动失败，请查看 config/live-engine/engine.log")
                try:
                    self._request("/osrp/v1/info", timeout=1)
                    return
                except (OSError, RuntimeError):
                    time.sleep(.25)
            raise RuntimeError("直播解析引擎启动超时")

    def resolve(self, url, quality):
        errors = []
        methods = [self._streamlink, self._bililive] if urlparse(url).hostname in {"twitch.tv", "www.twitch.tv"} else [self._bililive, self._streamlink]
        for method in methods:
            try:
                return method(url, quality)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                errors.append(str(exc))
        raise RuntimeError("；".join(errors)[-1200:])

    def _bililive(self, url, quality):
        self._ensure_backend()
        data = self._request("/osrp/v1/probe", {"url": url, "include_streams": True})
        result = {"online": bool(data.get("is_live")), "title": data.get("title"), "anchor": data.get("anchor_name"), "provider": "bililive-go"}
        if not result["online"]:
            return result
        streams = [item for item in data.get("streams", []) if item.get("url")]
        if not streams:
            raise RuntimeError("直播已开播，但未取得流地址；请检查平台登录态")
        ordered = sorted(streams, key=lambda item: item.get("quality_id") or 0, reverse=quality != "worst")
        stream = next((item for item in streams if item.get("quality") == quality), ordered[0])
        # OSRP intentionally does not persist cookies or signed URLs.  A
        # browser-like referer/user-agent is still needed by several CDN
        # endpoints (notably Bilibili and Douyu), and is safe to attach to
        # the short-lived FFmpeg process.
        return {**result, "stream_url": stream["url"], "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
            "Referer": url,
        }}

    def _streamlink(self, url, quality):
        if importlib.util.find_spec("streamlink") is None:
            raise RuntimeError("Streamlink 补充引擎未安装")
        result = subprocess.run([sys.executable, "-m", "streamlink", "--json", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=40,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout or "Streamlink 解析失败")[-500:])
        data = json.loads(result.stdout)
        streams = data.get("streams") or {}
        stream = streams.get(quality)
        if not stream or not stream.get("url"):
            raise RuntimeError(f"未取得 {quality} 画质；可用画质：{', '.join(streams)}")
        headers = stream.get("headers") or (stream.get("args") or {}).get("headers") or {}
        return {"online": True, "stream_url": stream["url"], "headers": headers,
                "title": (data.get("metadata") or {}).get("title"), "provider": "streamlink"}

    def close(self):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            if self._log:
                self._log.close()
