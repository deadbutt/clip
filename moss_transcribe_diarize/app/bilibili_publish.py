"""Bilibili QR login and single-video publishing support.

Credentials are kept outside job records and protected with Windows DPAPI on
Windows. Upload progress and the returned BVID are stored in the job's publish
metadata so a restart never silently submits the same video a second time.
"""
from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.cookies import SimpleCookie
from ctypes import wintypes
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from .jobs import RUNNING_STATES

logger = logging.getLogger(__name__)

_QR_GENERATE = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate?source=main-fe-header"
_QR_POLL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
_NAV = "https://api.bilibili.com/x/web-interface/nav"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "Chrome/131.0.0.0 Safari/537.36"
)
_SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp",
}
_COOKIE_FIELDS = (
    "SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5", "sid",
    "buvid3", "buvid4", "refresh_token",
)
_AUTH_MAGIC = b"MTDBILI1"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, protect: bool) -> bytes:
    """Use the current Windows account's DPAPI key without third-party modules."""
    if os.name != "nt":
        raise RuntimeError("Windows DPAPI is unavailable on this platform")
    buffer = ctypes.create_string_buffer(data)
    source = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = _DataBlob()
    if protect:
        function = ctypes.WinDLL("crypt32", use_last_error=True).CryptProtectData
        function.restype = wintypes.BOOL
        function.argtypes = [ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
        ok = function(ctypes.byref(source), "DieShang Bilibili login", None, None, None, 0x1, ctypes.byref(target))
    else:
        function = ctypes.WinDLL("crypt32", use_last_error=True).CryptUnprotectData
        function.restype = wintypes.BOOL
        function.argtypes = [ctypes.POINTER(_DataBlob), ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
        ok = function(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(target))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        local_free(target.pbData)


class CredentialStore:
    """Persist account credentials separately from jobs and API responses."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, credentials: dict[str, str]) -> None:
        safe = {key: str(credentials[key]) for key in _COOKIE_FIELDS if credentials.get(key)}
        raw = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if os.name == "nt":
            payload = _AUTH_MAGIC + b"D" + _dpapi(raw, protect=True)
        else:
            payload = _AUTH_MAGIC + b"P" + raw
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(payload)
            if os.name != "nt":
                temporary.chmod(0o600)
            os.replace(temporary, self.path)
            if os.name != "nt":
                self.path.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> dict[str, str] | None:
        if not self.path.is_file():
            return None
        payload = self.path.read_bytes()
        if not payload.startswith(_AUTH_MAGIC) or len(payload) <= len(_AUTH_MAGIC):
            raise RuntimeError("B 站登录凭据文件格式异常，请重新扫码登录")
        encoding, raw = payload[len(_AUTH_MAGIC):len(_AUTH_MAGIC) + 1], payload[len(_AUTH_MAGIC) + 1:]
        if encoding == b"D":
            raw = _dpapi(raw, protect=False)
        elif encoding != b"P":
            raise RuntimeError("B 站登录凭据文件格式异常，请重新扫码登录")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("B 站登录凭据文件格式异常，请重新扫码登录")
        return {str(key): str(value) for key, value in data.items() if key in _COOKIE_FIELDS and value}

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def _cookie_header(credentials: dict[str, str]) -> str:
    return "; ".join(f"{key}={credentials[key]}" for key in _COOKIE_FIELDS if credentials.get(key))


def _safe_error(exc: BaseException) -> str:
    message = str(exc) or type(exc).__name__
    message = re.sub(r"(?i)(SESSDATA|bili_jct|refresh_token)=([^&;\s]+)", r"\1=[已隐藏]", message)
    return message[:500]


class BilibiliPublishService:
    """Local QR login, credential management, and one-video upload coordinator."""

    def __init__(
        self,
        config_dir: str | Path,
        jobs: Any,
        *,
        requester: Callable[[str, str, bytes | None, str], dict[str, Any]] | None = None,
        uploader: Callable[[Path, Path, dict[str, Any], dict[str, str], Callable[[float, str], None]], dict[str, Any]] | None = None,
    ):
        self.store = CredentialStore(Path(config_dir) / "bilibili.auth")
        self.jobs = jobs
        self._requester = requester or self._request_json
        self._uploader = uploader or self._upload_with_library
        self._lock = threading.RLock()
        self._login_sessions: dict[str, dict[str, Any]] = {}
        self._upload_threads: dict[str, threading.Thread] = {}
        self._credential_error = ""
        try:
            self._credentials = self.store.load()
        except Exception as exc:
            self._credentials = None
            self._credential_error = "登录凭据无法解密，请重新扫码登录"
            logger.warning("Unable to load saved Bilibili credentials: %s", type(exc).__name__)
        self._recover_interrupted_uploads()

    @staticmethod
    def _request_json(method: str, url: str, body: bytes | None = None, cookie: str = "") -> dict[str, Any]:
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.bilibili.com/",
        }
        if cookie:
            headers["Cookie"] = cookie
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(1024 * 1024 + 1)
                set_cookie_headers = response.headers.get_all("Set-Cookie") or []
            if len(payload) > 1024 * 1024:
                raise RuntimeError("B 站返回内容过大")
            result = json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"B 站请求失败（HTTP {exc.code}）") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("无法连接 B 站，请检查网络连接") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("B 站返回内容格式异常") from exc
        if not isinstance(result, dict):
            raise RuntimeError("B 站返回内容格式异常")
        # Keep the login response cookies in an internal-only field. They are
        # required by the next QR poll and are never returned by the API.
        cookies = SimpleCookie()
        for header in set_cookie_headers:
            cookies.load(header)
        if cookies:
            result["_cookies"] = {key: morsel.value for key, morsel in cookies.items()}
        return result

    @staticmethod
    def _data(result: dict[str, Any]) -> dict[str, Any]:
        data = result.get("data")
        return data if isinstance(data, dict) else result

    @staticmethod
    def _result_code(result: dict[str, Any], data: dict[str, Any] | None = None) -> int:
        """Accept both passport's wrapped and unwrapped response shapes."""
        value = result.get("code")
        if isinstance(data, dict) and data.get("code") is not None:
            value = data.get("code")
        try:
            return int(value if value is not None else 0)
        except (TypeError, ValueError):
            return -1

    def start_login(self) -> dict[str, Any]:
        try:
            import qrcode
        except ImportError as exc:
            raise RuntimeError("扫码登录依赖尚未安装，请运行 uv pip install -e \".[torch-runtime,bilibili]\"") from exc
        payload = self._requester("GET", _QR_GENERATE, None, "")
        data = self._data(payload)
        if self._result_code(payload, data) != 0 or not data.get("url") or not data.get("qrcode_key"):
            logger.warning("Bilibili QR generation failed: code=%s message=%s", self._result_code(payload, data),
                           _safe_error(RuntimeError(str(payload.get("message") or data.get("message") or "unknown"))))
            raise RuntimeError(str(payload.get("message") or "B 站二维码生成失败"))
        image = qrcode.QRCode(box_size=7, border=3, error_correction=qrcode.constants.ERROR_CORRECT_M)
        image.add_data(str(data["url"]))
        image.make(fit=True)
        rendered = image.make_image(fill_color="#111827", back_color="white")
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG")
        session_id = uuid.uuid4().hex
        with self._lock:
            now = time.time()
            self._login_sessions = {key: value for key, value in self._login_sessions.items() if value["expires_at"] > now}
            self._login_sessions[session_id] = {"key": str(data["qrcode_key"]), "cookies": dict(payload.get("_cookies") or {}),
                                                 "expires_at": now + 180, "state": "waiting"}
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"session_id": session_id, "qr_image": f"data:image/png;base64,{encoded}", "expires_in": 180}

    def poll_login(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._login_sessions.get(str(session_id))
            if session is None:
                raise KeyError("扫码会话已失效，请重新生成二维码")
            if session["expires_at"] <= time.time():
                session["state"] = "expired"
                return {"state": "expired"}
            if session["state"] == "done":
                return {"state": "done", "account": self._account_name}
            key = session["key"]
            session_cookies = dict(session.get("cookies") or {})
        query = urllib.parse.urlencode({"qrcode_key": key, "source": "main-fe-header"})
        payload = self._requester("GET", _QR_POLL + "?" + query, None, _cookie_header(session_cookies))
        data = self._data(payload)
        code = self._result_code(payload, data)
        logger.info("Bilibili QR poll: session=%s code=%s state=%s", str(session_id)[:8], code,
                    str(data.get("message") or payload.get("message") or "")[:120])
        if code == 86101:
            return {"state": "waiting"}
        if code == 86090:
            with self._lock:
                session["state"] = "scanned"
            return {"state": "scanned"}
        if code == 86038:
            with self._lock:
                session["state"] = "expired"
            return {"state": "expired"}
        if code != 0:
            message = str(payload.get("message") or data.get("message") or "扫码登录失败")[:200]
            logger.warning("Bilibili QR login rejected: session=%s code=%s message=%s", str(session_id)[:8], code, message)
            return {"state": "error", "error": message, "code": code}
        response_cookies = dict(session_cookies)
        response_cookies.update(payload.get("_cookies") or {})
        with self._lock:
            session["cookies"] = response_cookies
        login_url = str(data.get("url") or payload.get("url") or "")
        parsed = urllib.parse.urlsplit(login_url)
        raw_values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        raw_values.update(urllib.parse.parse_qs(parsed.fragment, keep_blank_values=True))
        credentials = {key: response_cookies[key] for key in _COOKIE_FIELDS if response_cookies.get(key)}
        credentials.update({key: raw_values[key][0] for key in _COOKIE_FIELDS if raw_values.get(key) and not credentials.get(key)})
        refresh_token = data.get("refresh_token") or payload.get("refresh_token")
        if refresh_token:
            credentials["refresh_token"] = str(refresh_token)
        if not credentials.get("SESSDATA") or not credentials.get("bili_jct"):
            logger.warning("Bilibili QR login missing credentials: session=%s url_fields=%s refresh_token=%s",
                           str(session_id)[:8], sorted(raw_values), bool(refresh_token))
            return {"state": "error", "error": "B 站未返回完整登录凭据，请重新扫码"}
        self.store.save(credentials)
        with self._lock:
            self._credentials = credentials
            self._credential_error = ""
            session["state"] = "done"
        account = self._account_name
        logger.info("Bilibili QR login completed: session=%s", str(session_id)[:8])
        return {"state": "done", "account": account}

    @property
    def _account_name(self) -> str:
        credentials = self._credentials or {}
        try:
            payload = self._requester("GET", _NAV, None, _cookie_header(credentials))
            data = self._data(payload)
            if self._result_code(payload, data) == 0 and data.get("isLogin"):
                return str(data.get("uname") or "已登录")
        except Exception:
            logger.debug("Unable to read Bilibili account name", exc_info=True)
        return "已登录账号"

    def auth_status(self) -> dict[str, Any]:
        if not self._credentials:
            return {"logged_in": False, "account": "", **({"error": self._credential_error} if self._credential_error else {})}
        try:
            payload = self._requester("GET", _NAV, None, _cookie_header(self._credentials))
            data = self._data(payload)
            if self._result_code(payload, data) == 0 and data.get("isLogin"):
                return {"logged_in": True, "account": str(data.get("uname") or "已登录")}
            return {"logged_in": False, "account": "", "error": "B 站登录已过期，请重新扫码"}
        except Exception as exc:
            return {"logged_in": True, "account": "", "error": _safe_error(exc)}

    def logout(self) -> dict[str, bool]:
        self.store.clear()
        with self._lock:
            self._credentials = None
            self._credential_error = ""
            self._login_sessions.clear()
        return {"ok": True}

    def shutdown(self, timeout: float = 5.0) -> None:
        """Give an active upload a short chance to persist its final state."""
        deadline = time.time() + max(0.0, float(timeout))
        for thread in list(self._upload_threads.values()):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if thread.is_alive():
                thread.join(remaining)

    @staticmethod
    def categories() -> list[dict[str, Any]]:
        try:
            source = resources.files("bilibili_api").joinpath("data/video_uploader_meta_pre.json").read_text(encoding="utf-8")
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
            raise RuntimeError("投稿依赖尚未安装，请运行 uv pip install -e \".[torch-runtime,bilibili]\"") from exc
        data = json.loads(source)
        categories: list[dict[str, Any]] = []
        for parent in (data.get("tid_list", []) if isinstance(data, dict) else []):
            parent_name = str(parent.get("name") or "")
            for node in parent.get("children") or []:
                if node.get("show") is False:
                    continue
                try:
                    tid = int(node.get("id", 0))
                except (TypeError, ValueError):
                    continue
                name = str(node.get("name") or "")
                if tid > 0 and name:
                    categories.append({"tid": tid, "name": f"{parent_name} · {name}" if parent_name else name})
        if not categories:
            raise RuntimeError("未能读取 B 站投稿分区列表")
        return categories

    def submit(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("任务仍在处理中，完成后才能投稿")
        if not self._credentials:
            raise RuntimeError("请先扫码登录 B 站账号")
        current = self.jobs.publish_draft(job_id)
        if current.get("status") == "uploading":
            raise RuntimeError("这个任务的视频投稿正在进行中")
        if current.get("status") == "interrupted" and not payload.get("confirm_uncertain_retry"):
            raise RuntimeError("上次投稿被服务中断，请先到 B 站创作中心确认是否已经收到稿件；确认没有重复稿件后再勾选确认重试")
        if current.get("status") == "submitted" and not payload.get("confirm_duplicate_submission"):
            raise RuntimeError("这条任务已经提交过 B 站稿件。若要再次投稿，请先确认这是有意发布的另一条稿件")
        phase = str(current.get("phase") or "")
        if current.get("status") == "failed" and phase == "submitting" and not payload.get("confirm_uncertain_retry"):
            raise RuntimeError("上次在提交稿件时连接中断，可能已投稿成功。请先到 B 站创作中心核对；确认没有重复稿件后，再勾选确认并重试")
        title = str(payload.get("title", current.get("title") or "")).strip()
        if not title or len(title) > 80:
            raise ValueError("标题不能为空，且不能超过 80 个字符")
        description = str(payload.get("description", current.get("description") or ""))
        if len(description) > 2000:
            raise ValueError("简介不能超过 2000 个字符")
        tags = payload.get("tags", current.get("tags") or [])
        if isinstance(tags, str):
            tags = [part.strip() for part in re.split(r"[,，\s]+", tags) if part.strip()]
        if not isinstance(tags, list):
            raise ValueError("标签格式不正确")
        tags = list(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
        if not tags or len(tags) > 10 or any(len(tag) > 20 for tag in tags):
            raise ValueError("B 站投稿需要 1–10 个标签，每个标签不超过 20 个字符")
        try:
            tid = int(payload.get("tid", current.get("tid") or 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("请选择有效的 B 站分区") from exc
        valid_tids = {item["tid"] for item in self.categories()}
        if tid not in valid_tids:
            raise ValueError("请选择有效的 B 站投稿分区")
        copyright_type = str(payload.get("copyright", current.get("copyright") or "original"))
        if copyright_type not in {"original", "repost"}:
            raise ValueError("请选择原创或转载")
        source = str(payload.get("source", current.get("source") or "")).strip()
        if copyright_type == "repost" and not source:
            raise ValueError("转载投稿需要填写来源")
        if len(source) > 200:
            raise ValueError("转载来源不能超过 200 个字符")
        dynamic = str(payload.get("dynamic", current.get("dynamic") or ""))
        if len(dynamic) > 233:
            raise ValueError("动态文案不能超过 233 个字符")
        variant = str(payload.get("upload_variant", current.get("upload_variant") or "original"))
        if variant not in {"original", "processed"}:
            raise ValueError("投稿视频来源不正确")
        media_path = Path(job.input_path).resolve()
        if variant == "processed":
            media_path = Path(job.output_path).resolve()
        if not media_path.is_file():
            raise ValueError("投稿视频文件不存在，请确认任务媒体或字幕成片仍在磁盘上")
        if media_path.suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError("B 站投稿需要视频文件，当前任务的原始媒体不是受支持的视频格式")
        cover_path = Path(job.thumbnail_path or "")
        if not cover_path.is_file():
            self.jobs._ensure_job_thumbnail(job)
            cover_path = Path(job.thumbnail_path or "")
        if not cover_path.is_file():
            raise ValueError("未找到视频封面，请先为这段视频生成封面")
        draft = self.jobs.update_publish_draft(job_id, {
            "title": title, "description": description, "tags": tags,
            "tid": tid, "copyright": copyright_type, "source": source,
            "dynamic": dynamic, "upload_variant": variant,
        })
        with self._lock:
            existing = self._upload_threads.get(job_id)
            if existing and existing.is_alive():
                raise RuntimeError("这个任务的视频投稿正在进行中")
            if any(thread.is_alive() for other_id, thread in self._upload_threads.items() if other_id != job_id):
                raise RuntimeError("已有另一个 B 站投稿正在进行，请等它完成后再提交下一条")
            self.jobs.update_publish_status(job_id, {
                "status": "uploading", "phase": "preparing", "progress": 0.0,
                "error": "", "bvid": "", "aid": None, "url": "",
                "started_at": time.time(), "finished_at": None,
            })
            thread = threading.Thread(
                target=self._upload_worker,
                args=(job_id, media_path, cover_path, draft, dict(self._credentials)),
                name=f"mtd-bilibili-{job_id[:8]}", daemon=True,
            )
            self._upload_threads[job_id] = thread
            thread.start()
        return self.jobs.publish_draft(job_id)

    def _upload_worker(self, job_id: str, media_path: Path, cover_path: Path,
                       draft: dict[str, Any], credentials: dict[str, str]) -> None:
        last_progress = {"value": 0.0, "saved_at": 0.0}

        def update(progress: float, phase: str) -> None:
            progress = max(0.0, min(0.99, float(progress)))
            now = time.time()
            if phase == last_progress.get("phase") and progress - last_progress["value"] < 0.02 and now - last_progress["saved_at"] < 1.0:
                return
            last_progress.update(value=progress, phase=phase, saved_at=now)
            self.jobs.update_publish_status(job_id, {"status": "uploading", "phase": phase, "progress": progress})

        try:
            result = self._uploader(media_path, cover_path, draft, credentials, update)
            if not isinstance(result, dict) or not result.get("bvid"):
                raise RuntimeError("B 站未返回稿件编号，请到创作中心核对投稿状态")
            finalize_publish_result(self.jobs, job_id, result)
        except Exception as exc:
            logger.warning("Bilibili upload failed for job %s: %s", job_id, type(exc).__name__)
            self.jobs.update_publish_status(job_id, {
                "status": "failed", "phase": last_progress.get("phase", "preparing"),
                "progress": last_progress["value"], "error": _safe_error(exc), "finished_at": time.time(),
            })

    @staticmethod
    def _upload_with_library(media_path: Path, cover_path: Path, draft: dict[str, Any],
                             cookies: dict[str, str], update: Callable[[float, str], None]) -> dict[str, Any]:
        try:
            import httpx  # noqa: F401 - selected as bilibili-api's async transport below
            from bilibili_api import video_uploader as uploader_module
            from bilibili_api.clients.HTTPXClient import HTTPXClient
            from bilibili_api.utils.network import Api, Credential, get_client, register_client
            from bilibili_api.utils.picture import Picture
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("投稿依赖尚未安装，请运行 uv pip install -e \".[torch-runtime,bilibili]\"") from exc

        # Keep the third-party package from logging request details that may
        # include account identifiers or signed upload URLs.
        logging.getLogger("bilibili-api-request").disabled = True
        logging.getLogger("httpx").disabled = True
        logging.getLogger("httpcore").disabled = True
        credential = Credential(
            sessdata=cookies.get("SESSDATA"), bili_jct=cookies.get("bili_jct"),
            dedeuserid=cookies.get("DedeUserID"), buvid3=cookies.get("buvid3"),
            buvid4=cookies.get("buvid4"), ac_time_value=cookies.get("refresh_token"),
        )
        register_client("httpx", HTTPXClient, {"timeout": 120.0})
        with Image.open(cover_path) as source_image:
            image = source_image.convert("RGB")
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        picture = Picture(content=image_buffer.getvalue(), imageType="png", url="memory://cover")
        page = uploader_module.VideoUploaderPage(str(media_path), str(draft["title"]), "")
        meta = uploader_module.VideoMeta(
            tid=int(draft["tid"]), title=str(draft["title"]), desc=str(draft.get("description") or ""),
            cover=picture, tags=list(draft["tags"]), original=draft.get("copyright") == "original",
            source=str(draft.get("source") or "") or None, dynamic=str(draft.get("dynamic") or "") or None,
            subtitle={"open": 0, "lan": ""},
        )

        class SafeUploader(uploader_module.VideoUploader):
            async def _upload_cover(self) -> str:
                from bilibili_api.video_uploader import VideoUploaderEvents

                self.dispatch(VideoUploaderEvents.PRE_COVER.value, None)
                try:
                    encoded = base64.b64encode(self.cover.content).decode("ascii")
                    result = await Api(**uploader_module._API["cover_up"], credential=self.credential) \
                        .update_data(cover=f"data:image/png;base64,{encoded}").result
                    url = str(result["url"])
                    self.dispatch(VideoUploaderEvents.AFTER_COVER.value, {"url": url})
                    return url
                except Exception as exc:
                    self.dispatch(VideoUploaderEvents.COVER_FAILED.value, {"err": exc})
                    raise

        uploader = SafeUploader([page], meta, credential, cover=picture)
        uploaded: set[int] = set()

        def after_chunk(info: dict[str, Any]) -> None:
            total = max(1, int(info.get("total_chunk_count") or 1))
            uploaded.add(int(info.get("chunk_number") or 0))
            update(0.05 + 0.86 * len(uploaded) / total, "uploading")

        from bilibili_api.video_uploader import VideoUploaderEvents

        @uploader.on(VideoUploaderEvents.PREUPLOAD.value)
        def on_preupload(_data):
            update(0.03, "uploading")

        @uploader.on(VideoUploaderEvents.AFTER_CHUNK.value)
        def on_after_chunk(data):
            after_chunk(data)

        @uploader.on(VideoUploaderEvents.PRE_SUBMIT.value)
        def on_pre_submit(_data):
            update(0.96, "submitting")

        async def run() -> dict[str, Any]:
            try:
                result = await uploader.start()
                if not isinstance(result, dict) or int(result.get("code", 0)) != 0:
                    raise RuntimeError(str((result or {}).get("message") or "B 站未能确认投稿结果"))
                data = result.get("data") if isinstance(result.get("data"), dict) else result
                bvid = str(data.get("bvid") or "")
                if not bvid:
                    raise RuntimeError("B 站没有返回稿件编号；请到创作中心核对投稿状态后再重试")
                update(1.0, "complete")
                return {"bvid": bvid, "aid": data.get("aid"), "url": f"https://www.bilibili.com/video/{bvid}"}
            finally:
                try:
                    await get_client().close()
                except Exception:
                    logger.debug("Unable to close Bilibili HTTP session", exc_info=True)

        result = asyncio.run(run())
        return result

    def _recover_interrupted_uploads(self) -> None:
        for job in self.jobs.list_jobs():
            info = dict(job.publish_info or {})
            if info.get("status") != "uploading":
                continue
            phase = str(info.get("phase") or "preparing")
            self.jobs.update_publish_status(job.id, {
                "status": "interrupted", "phase": phase,
                "error": "服务重启导致投稿状态中断。请先到 B 站创作中心核对稿件，确认后再决定是否重试。",
                "finished_at": time.time(),
                "retry_requires_confirmation": phase == "submitting",
            })


def finalize_publish_result(jobs: Any, job_id: str, result: dict[str, Any]) -> None:
    """Store the returned BVID after the upload callback has completed."""
    jobs.update_publish_status(job_id, {
        "status": "submitted", "phase": "complete", "progress": 1.0,
        "error": "", "bvid": result["bvid"], "aid": result.get("aid"),
        "url": result["url"], "finished_at": time.time(),
    })

