from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from moss_transcribe_diarize.app.live_resolver import LiveResolver
from moss_transcribe_diarize.app.live_streams import SUPPORTED_PLATFORM_HOSTS, LiveStreamManager
from moss_transcribe_diarize.app.server import create_app


@pytest.mark.parametrize("platform,hosts", SUPPORTED_PLATFORM_HOSTS.items())
def test_platform_hosts_match_exact_domain(platform, hosts):
    for host in hosts:
        assert LiveStreamManager._platform("https://" + host + "/123") == platform
        assert LiveStreamManager._platform("https://" + host + ".example.org/123") == "其它平台"


def test_session_persists_without_signed_stream_url(tmp_path):
    manager = LiveStreamManager(tmp_path / "videos")
    manager.resolver.resolve = Mock(return_value={"online": True, "stream_url": "https://cdn.example/live?token=secret", "title": "频道"})
    session = manager.add("https://live.bilibili.com/1")
    assert session.status == "online"
    assert "secret" not in manager.state_path.read_text(encoding="utf-8")
    restored = LiveStreamManager(tmp_path / "videos").get(session.id)
    assert restored.title == "频道"
    assert not restored.stream_url
    assert restored.status == "idle"


def test_failed_resolution_does_not_keep_expired_url(tmp_path):
    manager = LiveStreamManager(tmp_path / "videos")
    manager.resolver.resolve = Mock(return_value={"online": True, "stream_url": "https://cdn.example/live"})
    session = manager.add("https://www.twitch.tv/example")
    manager.resolver.resolve.side_effect = RuntimeError("登录态已过期")
    manager.resolve(session.id)
    assert session.status == "error"
    assert not session.stream_url
    assert "登录态" in session.error


def test_bililive_uses_probe_for_every_platform_and_quality(tmp_path):
    resolver = LiveResolver(tmp_path)
    resolver._ensure_backend = Mock()
    resolver._request = Mock(return_value={"is_live": True, "title": "直播", "streams": [
        {"quality": "720p", "quality_id": 720, "url": "https://cdn.example/720"},
        {"quality": "1080p", "quality_id": 1080, "url": "https://cdn.example/1080"},
    ]})
    assert resolver._bililive("https://www.yy.com/1", "best")["stream_url"].endswith("1080")
    assert resolver._bililive("https://www.yy.com/1", "worst")["stream_url"].endswith("720")
    assert resolver._bililive("https://www.yy.com/1", "720p")["stream_url"].endswith("720")
    resolver._request.assert_called_with("/osrp/v1/probe", {"url": "https://www.yy.com/1", "include_streams": True})


def test_twitch_uses_streamlink_before_legacy_adapter(tmp_path):
    resolver = LiveResolver(tmp_path)
    resolver._streamlink = Mock(return_value={"online": True})
    resolver._bililive = Mock()
    resolver.resolve("https://www.twitch.tv/channel", "best")
    resolver._streamlink.assert_called_once()
    resolver._bililive.assert_not_called()


def test_live_api_isolated_state_and_invalid_requests(tmp_path):
    app = create_app(model_path="fake-model", runs_dir=tmp_path / "runs")
    manager = app.state.live_streams
    manager.resolver.resolve = Mock(return_value={"online": False, "title": "离线频道"})
    with TestClient(app) as client:
        assert client.post("/api/live/sessions", json=[]).status_code == 400
        assert client.post("/api/live/sessions", json={"url": "file:///etc/passwd"}).status_code == 400
        assert client.post("/api/live/sessions/missing/record").status_code == 404
        result = client.post("/api/live/sessions", json={"url": "https://live.douyin.com/123"})
        assert result.status_code == 200
        session_id = result.json()["id"]
        assert result.json()["status"] == "offline"
        assert client.post(f"/api/live/recordings/{session_id}/transcribe").status_code == 409
        assert client.delete(f"/api/live/sessions/{session_id}").status_code == 200
        assert client.get("/api/live/sessions").json()["sessions"] == []
    assert json.loads(manager.state_path.read_text()) == []
