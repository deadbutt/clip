from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from moss_transcribe_diarize.app.downloader import DownloadResult
from moss_transcribe_diarize.app.jobs import JobManager
from moss_transcribe_diarize.app.subscriptions import PlatformFetcher, SubscriptionManager, _firefox_cookie_header, sign_wbi

CHANNEL = "UC" + "a" * 22


def video(video_id, published=300):
    return {"video_id": video_id, "title": f"Title {video_id}", "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": "", "published_at": published}


class Feed(PlatformFetcher):
    def __init__(self):
        super().__init__()
        self.videos = [video("old", 100)]
        self.error = None

    def fetch(self, subscription):
        if self.error:
            raise RuntimeError(self.error)
        return "Creator", self.videos


class FakeJobs:
    def __init__(self):
        self.jobs = []
        self.created = []
        self.enqueued = []
        self._cancelled_jobs = set()

    def list_jobs(self):
        return self.jobs

    def create_job_for_url(self, url, **kwargs):
        self.created.append((url, kwargs))
        job = SimpleNamespace(id=str(len(self.jobs)), status="queued", error=None, retry_count=0, error_kind=None, **kwargs)
        job.to_dict = lambda: {k: v for k, v in vars(job).items() if k != "to_dict"}
        self.jobs.append(job)
        return job

    def _set_status(self, job, status, progress, **kwargs):
        job.status = status
        job.error = kwargs.get("error")

    def enqueue(self, job_id):
        self.enqueued.append(job_id)


def make_service(tmp_path, **settings):
    jobs = FakeJobs()
    feed = Feed()
    service = SubscriptionManager(tmp_path, jobs, fetcher=feed, clock=lambda: 200)
    sub = service.add({"platform": "youtube", "source": CHANNEL, **settings})
    return service, feed, jobs, sub


def test_first_follow_is_baseline_even_when_auto_download_is_enabled(tmp_path):
    service, _, jobs, sub = make_service(tmp_path, mode="auto", transcribe=True)
    assert sub["initialized"]
    assert not jobs.created
    assert service.snapshot()["videos"][0]["state"] == "history"
    assert service.snapshot()["unread_count"] == 0


def test_first_follow_keeps_only_five_latest_cards(tmp_path):
    jobs = FakeJobs()
    feed = Feed()
    feed.videos = [video(f"old-{i}", 100 + i) for i in range(8)]
    service = SubscriptionManager(tmp_path, jobs, fetcher=feed, clock=lambda: 200)
    service.add({"platform": "youtube", "source": CHANNEL})
    snapshot = service.snapshot()
    assert len(snapshot["videos"]) == 5
    assert {entry["video_id"] for entry in snapshot["videos"]} == {f"old-{i}" for i in range(3, 8)}
    assert snapshot["unread_count"] == 0


def test_notification_confirmation_dedup_and_restart(tmp_path):
    service, feed, jobs, sub = make_service(tmp_path)
    feed.videos.append(video("new"))
    assert service.check(sub["id"])["new_count"] == 1
    assert service.check(sub["id"])["new_count"] == 0
    assert service.snapshot()["unread_count"] == 1
    assert not jobs.created
    first = service.download("youtube:new")
    second = service.download("youtube:new")
    assert first["id"] == second["id"]
    assert len(jobs.created) == 1
    assert jobs.created[0][1]["download_only"] is True
    restarted = SubscriptionManager(tmp_path, jobs, fetcher=feed, clock=lambda: 400)
    restarted.check(sub["id"])
    restarted.download("youtube:new")
    assert len(jobs.created) == 1
    assert restarted.snapshot()["unread_count"] == 0


def test_channel_unread_counts_include_updates_outside_the_video_preview_limit(tmp_path):
    service, feed, _, sub = make_service(tmp_path)
    feed.videos.extend(video(f"new-{i}") for i in range(105))
    service.check(sub["id"])
    snapshot = service.snapshot()
    assert len(snapshot["videos"]) == 100
    assert snapshot["subscriptions"][0]["unread_count"] == 105
    assert snapshot["unread_count"] == 105
    service.dismiss("youtube:new-0")
    assert service.snapshot()["subscriptions"][0]["unread_count"] == 104
    service.mark_read()
    assert service.snapshot()["subscriptions"][0]["unread_count"] == 0


def test_auto_download_new_videos_only_and_transcription_opt_in(tmp_path):
    service, feed, jobs, sub = make_service(tmp_path, mode="auto", transcribe=True, cookies_browser="firefox")
    feed.videos.extend([video("older-but-not-listed", 90), video("new")])
    service.check(sub["id"])
    service.check(sub["id"])
    assert len(jobs.created) == 1
    assert jobs.created[0][0].endswith("v=new")
    assert jobs.created[0][1]["download_only"] is False
    assert jobs.created[0][1]["cookies_browser"] == "firefox"


def test_failed_initial_fetch_cannot_trigger_a_historical_download(tmp_path):
    feed = Feed()
    feed.error = "network unavailable"
    jobs = FakeJobs()
    service = SubscriptionManager(tmp_path, jobs, fetcher=feed, clock=lambda: 200)
    sub = service.add({"platform": "youtube", "source": CHANNEL, "mode": "auto"})
    assert sub["last_error"] == feed.error
    assert not sub["initialized"]
    feed.error = None
    service.check(sub["id"])
    assert service.get(sub["id"])["initialized"]
    assert not jobs.created


def test_reconcile_crash_after_job_creation_and_retry_same_job(tmp_path):
    service, feed, jobs, sub = make_service(tmp_path)
    feed.videos.append(video("new"))
    service.check(sub["id"])
    job = service.download("youtube:new")
    # Simulate the crash window: persisted job exists but subscriptions lost its ID.
    service._data["videos"]["youtube:new"].update(job_id=None, state="pending")
    service._save()
    restarted = SubscriptionManager(tmp_path, jobs, fetcher=feed)
    assert restarted.download("youtube:new")["id"] == job["id"]
    assert len(jobs.created) == 1
    jobs.jobs[0].status = "failed"
    jobs.jobs[0].error = "download failed"
    restarted.download("youtube:new")
    assert jobs.jobs[0].status == "queued"
    assert jobs.enqueued == [job["id"]]
    assert len(jobs.created) == 1


def test_pausing_or_deleting_during_fetch_does_not_enqueue(tmp_path):
    service, feed, jobs, sub = make_service(tmp_path, mode="auto")
    entered, finish = threading.Event(), threading.Event()

    def delayed_fetch(_):
        entered.set()
        assert finish.wait(3)
        return "Creator", [video("new")]

    feed.fetch = delayed_fetch
    worker = threading.Thread(target=lambda: service.check(sub["id"]))
    worker.start()
    assert entered.wait(3)
    service.update(sub["id"], {"enabled": False})
    finish.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert not jobs.created
    service.delete(sub["id"])
    assert not service.snapshot()["subscriptions"]


def test_concurrent_confirmations_create_one_job(tmp_path):
    service, feed, jobs, sub = make_service(tmp_path)
    feed.videos.append(video("new"))
    service.check(sub["id"])
    workers = [threading.Thread(target=lambda: service.download("youtube:new")) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)
    assert len(jobs.created) == 1


def test_ignored_updates_are_not_later_auto_downloaded(tmp_path):
    service, feed, jobs, sub = make_service(tmp_path)
    feed.videos.append(video("new"))
    service.check(sub["id"])
    service.dismiss("youtube:new")
    service.update(sub["id"], {"mode": "auto"})
    service.check(sub["id"])
    assert not jobs.created
    assert service.snapshot()["unread_count"] == 0


@pytest.mark.parametrize("settings", [{"interval_minutes": 1}, {"mode": "upload"}, {"transcribe": "false"}, {"cookies_browser": "invalid"}])
def test_invalid_settings_do_not_create_a_subscription(tmp_path, settings):
    with pytest.raises(ValueError):
        make_service(tmp_path, **settings)
    assert not (tmp_path / "subscriptions.json").exists()


def test_duplicate_subscription_is_rejected(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    with pytest.raises(ValueError, match="已经关注"):
        service.add({"platform": "youtube", "source": "https://www.youtube.com/channel/" + CHANNEL + "/videos"})


def test_youtube_handle_resolution_and_feed_parsing():
    rss = '''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
      <author><name>日本語 &amp; Creator</name></author><entry><yt:videoId>abcdefghijk</yt:videoId>
      <title>動画 &amp; Title</title><published>2026-10-07T04:00:00Z</published></entry></feed>'''
    fetcher = PlatformFetcher(lambda url, **kw: rss if "feeds/videos" in url else json.dumps({"channelId": CHANNEL}))
    channel = fetcher.resolve("youtube", "https://www.youtube.com/@creator/videos")
    name, videos = fetcher.fetch(channel)
    assert channel["channel_id"] == CHANNEL
    assert name == "日本語 & Creator"
    assert videos[0]["title"] == "動画 & Title"
    assert videos[0]["url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert videos[0]["published_at"] > 0


def test_youtube_handle_resolution_passes_browser_cookie_choice():
    seen = []

    def get(url, **kwargs):
        seen.append(kwargs.get("cookies_browser"))
        return json.dumps({"channelId": CHANNEL})

    fetcher = PlatformFetcher(get)
    identity = fetcher.resolve("youtube", "https://www.youtube.com/@creator", "firefox")
    assert identity["channel_id"] == CHANNEL
    assert seen == ["firefox"]


@pytest.mark.parametrize("platform,source", [
    ("youtube", "https://youtube.com.evil.test/@creator"), ("youtube", "http://localhost/channel/" + CHANNEL),
    ("youtube", "https://www.youtube.com/watch?v=abcdefghijk"),
    ("bilibili", "https://space.bilibili.com.evil.test/2"), ("bilibili", "0"), ("douyin", "user"),
])
def test_only_creator_addresses_of_supported_platforms_are_accepted(platform, source):
    with pytest.raises(ValueError):
        PlatformFetcher().resolve(platform, source)


def test_bilibili_signed_feed_parsing_and_api_restrictions():
    requests = []

    def get(url, **kwargs):
        requests.append(url)
        if "/nav" in url:
            return json.dumps({"data": {"wbi_img": {"img_url": "https://i.test/" + "a" * 32 + ".png", "sub_url": "https://i.test/" + "b" * 32 + ".png"}}})
        return json.dumps({"code": 0, "data": {"list": {"vlist": [
            {"bvid": "BV1xx411c7mD", "title": "UP &amp; 新视频", "author": "UP 主", "pic": "//i.test/cover.jpg", "created": 100, "length": "12:34"}
        ]}}})

    fetcher = PlatformFetcher(get)
    name, videos = fetcher.fetch(fetcher.resolve("bilibili", "https://space.bilibili.com/2/video"))
    assert name == "UP 主"
    assert videos[0]["title"] == "UP & 新视频"
    assert videos[0]["thumbnail"] == "https://i.test/cover.jpg"
    assert videos[0]["duration"] == "12:34"
    assert "w_rid=" in requests[-1] and "mid=2" in requests[-1]
    fetcher.get_text = lambda *a, **kw: '{"code":-352,"message":"平台限制"}'
    with pytest.raises(RuntimeError, match="-352"):
        fetcher.fetch({"platform": "bilibili", "channel_id": "2"})


def test_firefox_cookie_header_reads_matching_domain_without_logging_values(tmp_path, monkeypatch):
    import sqlite3

    profiles = tmp_path / "Mozilla" / "Firefox" / "Profiles" / "fixture.default"
    profiles.mkdir(parents=True)
    db = profiles / "cookies.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE moz_cookies (host TEXT, name TEXT, value TEXT)")
        connection.executemany("INSERT INTO moz_cookies VALUES (?, ?, ?)", [
            (".bilibili.com", "SESSDATA", "secret-cookie"),
            (".example.com", "ignored", "should-not-leak"),
        ])
    monkeypatch.setenv("APPDATA", str(tmp_path))
    header = _firefox_cookie_header("https://api.bilibili.com/x/web-interface/nav")
    assert header == "SESSDATA=secret-cookie"


def test_wbi_signing_removes_reserved_characters():
    query = sign_wbi({"mid": "2", "keyword": "a!'()*b"}, "secret", 100)
    assert "keyword=ab" in query and "wts=100" in query and "w_rid=" in query


def test_download_only_never_loads_or_transcribes_and_survives_restart(tmp_path, monkeypatch):
    runner = SimpleNamespace(model_path="fake", transcribe=lambda *a, **kw: pytest.fail("download-only called transcription"))
    manager = JobManager(tmp_path / "runs", runner, prompt="", max_length=1, max_new_tokens=1)
    manager.pause_queue()

    def download(url, output_dir, **kwargs):
        assert kwargs["cookies_browser"] == "none"
        path = Path(output_dir) / "input.mkv"
        path.write_bytes(b"video")
        return DownloadResult(path, title="Downloaded title")

    monkeypatch.setattr("moss_transcribe_diarize.app.downloader.download_with_yt_dlp", download)
    try:
        job = manager.create_job_for_url("https://www.youtube.com/watch?v=abcdefghijk", download_only=True,
                                         cookies_browser="none", subscription_video_key="youtube:abcdefghijk")
        manager._process_job(job)
        assert job.status == "downloaded"
        assert manager.download_path(job.id, "media").read_bytes() == b"video"
    finally:
        manager.shutdown()
    restarted = JobManager(tmp_path / "runs", runner, prompt="", max_length=1, max_new_tokens=1)
    try:
        restored = restarted.get_job(job.id)
        assert restored.download_only and restored.status == "downloaded"
        assert restored.subscription_video_key == "youtube:abcdefghijk"
    finally:
        restarted.shutdown()


def test_subscription_api_create_check_confirm_update_and_delete(tmp_path):
    from fastapi.testclient import TestClient

    from moss_transcribe_diarize.app.server import create_app

    app = create_app(model_path="fake", runs_dir=tmp_path / "runs")
    app.state.manager.pause_queue()
    app.state.subscriptions.fetcher = Feed()
    app.state.subscriptions.clock = lambda: 200
    # No network or processing in this API test; lifecycle is tested separately.
    client = TestClient(app)
    try:
        created = client.post("/api/subscriptions", json={"platform": "youtube", "source": CHANNEL})
        assert created.status_code == 200
        sub_id = created.json()["id"]
        app.state.subscriptions.fetcher.videos.append(video("new"))
        assert client.post(f"/api/subscriptions/{sub_id}/check").json()["new_count"] == 1
        assert client.get("/api/subscriptions").json()["unread_count"] == 1
        result = client.post("/api/subscription-videos/youtube%3Anew/download")
        assert result.status_code == 200 and result.json()["download_only"]
        assert client.post("/api/subscription-videos/youtube%3Anew/download").json()["id"] == result.json()["id"]
        assert client.put(f"/api/subscriptions/{sub_id}", json={"enabled": False}).status_code == 200
        assert client.post(f"/api/subscriptions/{sub_id}/check").json()["paused"]
        assert client.post("/api/subscriptions/read").status_code == 200
        assert client.delete(f"/api/subscriptions/{sub_id}").status_code == 200
        assert client.get("/api/subscriptions").json()["subscriptions"] == []
    finally:
        app.state.manager.shutdown()


def test_subscription_poller_follows_app_lifecycle(tmp_path):
    from fastapi.testclient import TestClient

    from moss_transcribe_diarize.app.server import create_app

    app = create_app(model_path="fake", runs_dir=tmp_path / "runs")
    with TestClient(app) as client:
        assert client.get("/api/subscriptions").json()["running"]
    assert not app.state.subscriptions.snapshot()["running"]
