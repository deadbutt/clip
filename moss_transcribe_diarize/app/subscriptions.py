"""Persistent creator subscriptions and polling for YouTube and Bilibili."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)
_CHANNEL_ID = re.compile(r"UC[A-Za-z0-9_-]{22}")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_BVID = re.compile(r"BV[A-Za-z0-9]{10}")
_WBI_ORDER = (
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
)


def _firefox_cookie_header(url: str) -> str:
    """Read only matching Firefox cookies into a request header.

    Firefox stores cookie values in the local SQLite profile database. Copying
    the database first avoids locking the user's live profile while Firefox is
    open. The values never leave this process.
    """
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    if os.name == "nt":
        profiles_dir = Path(os.environ.get("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"
    else:
        profiles_dir = Path.home() / ".mozilla" / "firefox"
    if not profiles_dir.is_dir():
        return ""
    cookie_suffix = "." + ".".join(host.split(".")[-2:]) if "." in host else host
    rows: list[tuple[str, str]] = []
    for cookie_db in profiles_dir.glob("*/cookies.sqlite"):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as handle:
                temporary = Path(handle.name)
            shutil.copy2(cookie_db, temporary)
            db = sqlite3.connect(f"file:{temporary}?mode=ro", uri=True)
            try:
                rows.extend(db.execute(
                    "SELECT name, value FROM moz_cookies WHERE host LIKE ? OR host LIKE ?",
                    (host, "%" + cookie_suffix),
                ).fetchall())
            finally:
                db.close()
        except (OSError, sqlite3.Error):
            continue
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    return "; ".join(f"{name}={value}" for name, value in rows if name and value)


def _http_text(url: str, *, referer: str = "", cookies_browser: str = "none") -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, application/atom+xml, text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    if cookies_browser == "firefox":
        cookie_header = _firefox_cookie_header(url)
        if cookie_header:
            headers["Cookie"] = cookie_header
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as response:
            body = response.read(4 * 1024 * 1024 + 1)
            if len(body) > 4 * 1024 * 1024:
                raise RuntimeError("平台返回内容过大，请稍后重试")
            return body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"平台请求失败（HTTP {exc.code}），请稍后重试") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("无法连接平台，请检查网络连接后重试") from exc


def _published_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def sign_wbi(params: dict[str, Any], key: str, timestamp: int) -> str:
    values = {**params, "wts": timestamp}
    query = urllib.parse.urlencode(sorted(
        (k, re.sub(r"[!'()*]", "", str(v))) for k, v in values.items()
    ))
    return query + "&w_rid=" + hashlib.md5((query + key).encode()).hexdigest()


class PlatformFetcher:
    def __init__(self, get_text: Callable[..., str] = _http_text):
        self.get_text = get_text
        self._wbi_key = ""
        self._wbi_expiry = 0.0

    def resolve(self, platform: str, source: str, cookies_browser: str = "none") -> dict[str, str]:
        source = str(source).strip()
        if len(source) > 512:
            raise ValueError("频道地址过长")
        if platform == "bilibili":
            uid = source
            if not source.isdigit():
                parsed = urllib.parse.urlsplit(source)
                if (parsed.scheme != "https" or parsed.hostname != "space.bilibili.com"
                        or parsed.port not in (None, 443) or parsed.username or parsed.password):
                    raise ValueError("请填写 B 站 UID 或 https://space.bilibili.com/UID 主页链接")
                uid = parsed.path.strip("/").split("/")[0]
            if not re.fullmatch(r"[1-9]\d{0,19}", uid):
                raise ValueError("B 站 UID 应为正整数")
            return {"platform": platform, "channel_id": uid, "url": f"https://space.bilibili.com/{uid}"}
        if platform != "youtube":
            raise ValueError("目前支持 YouTube 和 B 站")
        if _CHANNEL_ID.fullmatch(source):
            channel_id = source
        else:
            if source.startswith("@"):
                source = "https://www.youtube.com/" + source
            parsed = urllib.parse.urlsplit(source)
            if parsed.scheme != "https" or parsed.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
                raise ValueError("请填写 YouTube 频道主页、@账号 或 UC 开头的频道 ID")
            if parsed.port not in (None, 443) or parsed.username or parsed.password:
                raise ValueError("频道地址格式不正确")
            path = parsed.path.rstrip("/")
            match = re.fullmatch(r"/channel/(UC[A-Za-z0-9_-]{22})(?:/(?:videos|shorts|streams|featured))?", path)
            if match:
                channel_id = match.group(1)
            else:
                channel_path = re.match(r"^(/@[^/?#]+|/(?:c|user)/[^/?#]+)(?:/|$)", path)
                if not channel_path:
                    raise ValueError("需要频道主页，不能使用单个视频或播放列表链接")
                page = self._get_text("https://www.youtube.com" + channel_path.group(1), cookies_browser=cookies_browser)
                match = re.search(r'"(?:channelId|externalId)"\s*:\s*"(UC[A-Za-z0-9_-]{22})"', page)
                if not match:
                    match = re.search(r'(?:/channel/|channel_id=)(UC[A-Za-z0-9_-]{22})', page)
                if not match:
                    raise ValueError("未能解析频道 ID，请直接填写 /channel/UC… 链接或频道 ID")
                channel_id = match.group(1)
        return {"platform": platform, "channel_id": channel_id, "url": f"https://www.youtube.com/channel/{channel_id}"}

    def _get_text(self, url: str, *, referer: str = "", cookies_browser: str = "none") -> str:
        try:
            return self.get_text(url, referer=referer, cookies_browser=cookies_browser)
        except TypeError:
            # Keep lightweight test fetchers and third-party integrations
            # compatible with the original two-argument callback.
            return self.get_text(url, referer=referer)

    def fetch(self, subscription: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        cookies_browser = str(subscription.get("cookies_browser") or "none")
        if subscription["platform"] == "youtube":
            return self._youtube(subscription["channel_id"], cookies_browser)
        return self._bilibili(subscription["channel_id"], cookies_browser)

    def _youtube(self, channel_id: str, cookies_browser: str = "none") -> tuple[str, list[dict[str, Any]]]:
        xml = self._get_text("https://www.youtube.com/feeds/videos.xml?" + urllib.parse.urlencode({"channel_id": channel_id}), cookies_browser=cookies_browser)
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            raise RuntimeError("YouTube 更新列表格式异常，请稍后重试") from exc
        ns = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise RuntimeError("未取得 YouTube 频道更新列表")
        name = root.findtext("a:author/a:name", default="", namespaces=ns) or root.findtext("a:title", default=channel_id, namespaces=ns)
        videos = []
        for entry in root.findall("a:entry", ns):
            video_id = entry.findtext("yt:videoId", default="", namespaces=ns)
            if not _VIDEO_ID.fullmatch(video_id):
                continue
            videos.append({
                "video_id": video_id, "title": entry.findtext("a:title", default=video_id, namespaces=ns),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                "published_at": _published_timestamp(entry.findtext("a:published", default="", namespaces=ns)),
                "duration": "",
            })
        return name, videos

    def _bilibili(self, uid: str, cookies_browser: str = "none") -> tuple[str, list[dict[str, Any]]]:
        referer = f"https://space.bilibili.com/{uid}/video"
        if not self._wbi_key or time.time() >= self._wbi_expiry:
            nav = json.loads(self._get_text("https://api.bilibili.com/x/web-interface/nav", referer=referer, cookies_browser=cookies_browser))
            images = (nav.get("data") or {}).get("wbi_img") or {}
            keys = "".join(Path(urllib.parse.urlsplit(images.get(k, "")).path).stem for k in ("img_url", "sub_url"))
            if len(keys) < 64:
                raise RuntimeError("未取得 B 站投稿列表签名，请稍后重试")
            self._wbi_key = "".join(keys[i] for i in _WBI_ORDER)[:32]
            self._wbi_expiry = time.time() + 3600
        query = sign_wbi({"mid": uid, "pn": 1, "ps": 30, "order": "pubdate", "platform": "web", "web_location": "1550101"},
                         self._wbi_key, int(time.time()))
        payload = json.loads(self._get_text("https://api.bilibili.com/x/space/wbi/arc/search?" + query, referer=referer, cookies_browser=cookies_browser))
        if payload.get("code") != 0:
            self._wbi_expiry = 0
            raise RuntimeError(f"B 站暂时无法读取投稿列表（{payload.get('code')}：{payload.get('message', '平台限制')}），稍后重试")
        rows = ((payload.get("data") or {}).get("list") or {}).get("vlist")
        if not isinstance(rows, list):
            raise RuntimeError("B 站投稿列表格式异常")
        videos = []
        name = ""
        for row in rows:
            bvid = str(row.get("bvid") or "")
            if not _BVID.fullmatch(bvid):
                continue
            name = str(row.get("author") or name)
            thumbnail = str(row.get("pic") or "")
            if thumbnail.startswith("//"):
                thumbnail = "https:" + thumbnail
            if thumbnail.startswith("http://"):
                thumbnail = "https://" + thumbnail[7:]
            if not thumbnail.startswith("https://"):
                thumbnail = ""
            videos.append({"video_id": bvid, "title": html.unescape(str(row.get("title") or bvid)),
                           "url": f"https://www.bilibili.com/video/{bvid}", "thumbnail": thumbnail,
                           "published_at": float(row.get("created") or 0),
                           "duration": str(row.get("length") or "")})
        return name or f"UP 主 {uid}", videos


class SubscriptionManager:
    """One background poller; atomic local state and durable job deduplication."""

    def __init__(self, config_dir: str | Path, jobs: Any, *, fetcher: Any = None, clock: Callable[[], float] = time.time):
        self.path = Path(config_dir) / "subscriptions.json"
        self.jobs = jobs
        self.fetcher = fetcher or PlatformFetcher()
        self.clock = clock
        self._lock = threading.RLock()
        self._check_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._checking: str | None = None
        self._data: dict[str, Any] = {"version": 1, "subscriptions": {}, "videos": {}}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != 1 or not isinstance(data.get("subscriptions"), dict) or not isinstance(data.get("videos"), dict):
                raise RuntimeError("订阅配置格式异常，请检查 config/subscriptions.json")
            self._data = data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._lock:
            # Refresh once on every app launch so a service that was closed for
            # a while does not wait for the previous interval to expire.
            changed = False
            for subscription in self._data["subscriptions"].values():
                if subscription.get("enabled") and subscription.get("next_check", 0) > 0:
                    subscription["next_check"] = 0
                    changed = True
            if changed:
                self._save()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mtd-subscriptions", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                due = [s["id"] for s in self._data["subscriptions"].values() if s["enabled"] and s["next_check"] <= self.clock()]
            for subscription_id in due:
                if self._stop.is_set():
                    return
                if self._check_lock.locked():
                    break
                try:
                    self.check(subscription_id)
                except Exception:
                    logger.exception("Subscription polling failed")
            self._wake.wait(30)
            self._wake.clear()

    @staticmethod
    def _settings(options: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
        values = {"name": "", "interval_minutes": 10, "mode": "notify", "transcribe": False, "enabled": True,
                  "cookies_browser": "firefox", **(previous or {}), **options}
        try:
            interval = int(values["interval_minutes"])
        except (ValueError, TypeError) as exc:
            raise ValueError("检查间隔应为 5–1440 分钟的整数") from exc
        if not 5 <= interval <= 1440:
            raise ValueError("检查间隔应为 5–1440 分钟")
        if values["mode"] not in {"notify", "auto"}:
            raise ValueError("更新操作应为提醒或自动下载")
        if values["cookies_browser"] not in {"none", "firefox", "chrome", "edge", "brave"}:
            raise ValueError("不支持的下载浏览器登录态")
        if not isinstance(values["enabled"], bool) or not isinstance(values["transcribe"], bool):
            raise ValueError("启用和转录设置应为布尔值")
        return {"name": str(values["name"] or "").strip()[:120], "interval_minutes": interval, "mode": values["mode"],
                "transcribe": values["transcribe"], "enabled": values["enabled"], "cookies_browser": values["cookies_browser"]}

    def add(self, options: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings(options)
        identity = self.fetcher.resolve(str(options.get("platform") or ""), str(options.get("source") or ""),
                                        str(settings.get("cookies_browser") or "none"))
        with self._lock:
            if any(s["platform"] == identity["platform"] and s["channel_id"] == identity["channel_id"]
                   for s in self._data["subscriptions"].values()):
                raise ValueError("已经关注这个频道")
            subscription_id = uuid.uuid4().hex[:12]
            record = {**identity, **settings, "id": subscription_id, "channel_name": identity["channel_id"],
                      "created_at": self.clock(), "initialized": False, "baseline_at": 0, "last_checked": None,
                      "next_check": 0, "last_error": "", "failures": 0}
            self._data["subscriptions"][subscription_id] = record
            self._save()
        # A failed initial fetch stays uninitialized: it can never auto-download
        # an old backlog when network access recovers.
        if settings["enabled"]:
            self.check(subscription_id)
        self._wake.set()
        return self.get(subscription_id)

    def get(self, subscription_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._data["subscriptions"][subscription_id])

    def update(self, subscription_id: str, options: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self._data["subscriptions"][subscription_id]
            record.update(self._settings(options, record))
            record["next_check"] = self.clock()
            self._save()
            result = dict(record)
        self._wake.set()
        return result

    def delete(self, subscription_id: str) -> None:
        with self._lock:
            del self._data["subscriptions"][subscription_id]
            self._data["videos"] = {k: v for k, v in self._data["videos"].items() if v["subscription_id"] != subscription_id}
            self._save()

    def check(self, subscription_id: str) -> dict[str, Any]:
        with self._lock:
            subscription = dict(self._data["subscriptions"][subscription_id])
            if not subscription["enabled"]:
                return {"new_count": 0, "paused": True}
            if not self._check_lock.acquire(blocking=False):
                self._data["subscriptions"][subscription_id]["next_check"] = 0
                self._wake.set()
                return {"new_count": 0, "scheduled": True}
            self._checking = subscription_id
        new_count = 0
        try:
            channel_name, videos = self.fetcher.fetch(subscription)
            with self._lock:
                record = self._data["subscriptions"].get(subscription_id)
                if not record or not record["enabled"] or self._stop.is_set():
                    return {"new_count": 0, "paused": True}
                now = self.clock()
                # The first successful follow creates a small local snapshot so
                # a closed app still shows the five latest cards on next launch.
                if not record["initialized"]:
                    videos = sorted(videos, key=lambda v: float(v.get("published_at") or 0), reverse=True)[:5]
                for video in videos:
                    key = record["platform"] + ":" + video["video_id"]
                    if key in self._data["videos"]:
                        # Refresh presentation metadata (especially B 站's
                        # duration) without resetting unread/download state.
                        stored = self._data["videos"][key]
                        for field in ("title", "thumbnail", "published_at", "duration"):
                            if video.get(field):
                                stored[field] = video[field]
                        continue
                    published = float(video.get("published_at") or 0)
                    is_new = record["initialized"] and (published == 0 or published >= record["baseline_at"])
                    self._data["videos"][key] = {
                        **video, "id": key, "subscription_id": subscription_id, "platform": record["platform"],
                        "first_seen": now, "is_new": bool(is_new), "unread": bool(is_new),
                        "state": "pending" if is_new else "history", "job_id": None, "error": "",
                    }
                    new_count += int(is_new)
                record.update(channel_name=channel_name, last_checked=now, next_check=now + record["interval_minutes"] * 60,
                              last_error="", failures=0)
                if not record["initialized"]:
                    record.update(initialized=True, baseline_at=now)
                self._save()
                if record["mode"] == "auto":
                    for video in list(self._data["videos"].values()):
                        if video["subscription_id"] == subscription_id and video["state"] == "pending":
                            try:
                                self._download_locked(video, record, automatic=True)
                            except Exception as exc:
                                video["error"] = str(exc)
                                self._save()
            return {"new_count": new_count}
        except Exception as exc:
            with self._lock:
                record = self._data["subscriptions"].get(subscription_id)
                if record:
                    record["last_error"] = str(exc)
                    record["failures"] += 1
                    record["next_check"] = self.clock() + min(86400, record["interval_minutes"] * 60 * 2 ** min(record["failures"] - 1, 4))
                    self._save()
            return {"new_count": 0, "error": str(exc)}
        finally:
            with self._lock:
                self._checking = None
            self._check_lock.release()
            self._wake.set()

    def check_all(self) -> dict[str, Any]:
        with self._lock:
            ids = [s["id"] for s in self._data["subscriptions"].values() if s["enabled"]]
        return {"results": [self.check(i) for i in ids]}

    def _download_locked(self, video: dict[str, Any], subscription: dict[str, Any], *, automatic: bool = False) -> dict[str, Any]:
        # The job carries the stable platform/video identity before it enters the
        # queue. Recover even if the process died before saving our job_id.
        job = next((j for j in self.jobs.list_jobs() if j.subscription_video_key == video["id"]), None)
        if job is not None:
            if job.status in {"failed", "cancelled"} and not automatic:
                self.jobs._cancelled_jobs.discard(job.id)
                job.error_kind = None
                job.retry_count += 1
                self.jobs._set_status(job, "queued", 0.0, error=None)
                self.jobs.enqueue(job.id)
        else:
            job = self.jobs.create_job_for_url(
                video["url"], cookies_browser=subscription["cookies_browser"],
                download_only=not subscription["transcribe"], subscription_id=subscription["id"],
                subscription_video_key=video["id"], media_name=video["title"],
            )
        video.update(state="enqueued", job_id=job.id, error="")
        if not automatic:
            video["unread"] = False
        self._save()
        return job.to_dict()

    def download(self, video_id: str) -> dict[str, Any]:
        with self._lock:
            video = self._data["videos"][video_id]
            subscription = self._data["subscriptions"][video["subscription_id"]]
            return self._download_locked(video, subscription)

    def mark_read(self) -> None:
        with self._lock:
            for video in self._data["videos"].values():
                video["unread"] = False
            self._save()

    def dismiss(self, video_id: str) -> None:
        with self._lock:
            video = self._data["videos"][video_id]
            if video["state"] != "enqueued":
                video["state"] = "ignored"
            video["unread"] = False
            self._save()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            jobs = {j.id: j for j in self.jobs.list_jobs()}
            unread_by_subscription: dict[str, int] = {}
            for video in self._data["videos"].values():
                if video["unread"]:
                    key = video["subscription_id"]
                    unread_by_subscription[key] = unread_by_subscription.get(key, 0) + 1
            subscriptions = [dict(s, checking=s["id"] == self._checking,
                                  unread_count=unread_by_subscription.get(s["id"], 0))
                             for s in self._data["subscriptions"].values()]
            videos = []
            for v in sorted(self._data["videos"].values(), key=lambda v: (v["unread"], v["first_seen"], v["published_at"]), reverse=True)[:100]:
                job = jobs.get(v["job_id"])
                videos.append({**v, "job_status": job.status if job else None, "job_error": job.error if job else None})
            return {"subscriptions": subscriptions, "videos": videos,
                    "unread_count": sum(bool(v["unread"]) for v in self._data["videos"].values()),
                    "running": bool(self._thread and self._thread.is_alive()), "checking": self._checking}
