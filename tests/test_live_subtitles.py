from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from moss_transcribe_diarize.app.ffmpeg import detect_ffmpeg
from moss_transcribe_diarize.app.index_translator import IndexTranslator
from moss_transcribe_diarize.app.live_subtitles import LiveSubtitleService, SubtitleSession, confirmed_words
from moss_transcribe_diarize.app.server import create_app


def test_overlap_words_are_published_once_and_final_tail_is_flushed():
    words = [(0.2, 0.5, " first"), (4.9, 5.3, " boundary"), (5.6, 5.9, " tail")]
    first = confirmed_words(words, 0, -1, 5)
    second = confirmed_words([(0.9, 1.3, " boundary"), (1.6, 1.9, " tail")], 4, 5, 6)
    assert [w[2] for w in first + second] == [" first", " boundary", " tail"]


def test_index_context_leak_retries_only_current_source():
    translator = IndexTranslator()
    translator._ensure_server = Mock()
    translator._request_translation = Mock(side_effect=["背景（仅供理解）：以前的话。当前字幕：你好", "你好。"])
    assert translator.translate_live("Hello.", ["Previous line."]) == "你好。"
    first, second = [call.args[0] for call in translator._request_translation.call_args_list]
    assert "Previous line." in first
    assert "Previous line." not in second


def test_index_repeated_bad_output_is_reported_instead_of_published():
    translator = IndexTranslator()
    translator._ensure_server = Mock()
    translator._request_translation = Mock(return_value="当前字幕：你好")
    with pytest.raises(RuntimeError, match="原文已保留"):
        translator.translate_live("Hello.", ["previous"])


def test_backlog_discards_oldest_window_with_visible_count(tmp_path):
    service = LiveSubtitleService(Mock(), tmp_path, translator=Mock())
    session = SubtitleSession("test", tmp_path)
    for i in range(6):
        service._put_audio(session, i)
    assert session.dropped_windows == 3
    assert [session.audio_queue.get_nowait() for _ in range(3)] == [3, 4, 5]
    assert session.warning


def test_real_ffmpeg_capture_delivers_audio_and_writes_bilingual_draft(tmp_path):
    ffmpeg = detect_ffmpeg().ffmpeg
    if not ffmpeg:
        pytest.skip("FFmpeg not installed")
    audio = tmp_path / "voice.wav"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                    "-t", "7", str(audio)], check=True, timeout=10)
    runner = Mock()
    runner.transcribe_live_window.return_value = {"language": "en", "words": [(0.5, 1.5, " Hello world.")]}
    translator = Mock()
    translator.runtime_info.return_value = {"available": True}
    translator.translate_live.return_value = "你好，世界。"
    service = LiveSubtitleService(runner, tmp_path / "drafts", translator=translator)
    try:
        service.start("test", str(audio))
        session = service.sessions["test"]
        for thread in session.threads:
            thread.join(timeout=15)
        assert not any(thread.is_alive() for thread in session.threads)
        state = session.snapshot()
        assert state["state"] == "stopped"
        assert state["received_seconds"] == 7
        assert state["rows"][0]["translation"] == "你好，世界。"
        events = [json.loads(line) for line in session.draft_path.read_text(encoding="utf-8").splitlines()]
        assert events[0]["translation_state"] == "pending"
        assert events[-1]["translation_state"] == "done"
        assert runner.transcribe_live_window.call_count == 2
        assert "你好" not in str(translator.translate_live.call_args_list)
    finally:
        service.close()


def test_subtitle_api_uses_current_stream_and_removal_stops_capture(tmp_path):
    app = create_app(model_path="fake-model", runs_dir=tmp_path / "runs")
    manager = app.state.live_streams
    manager.resolver.resolve = Mock(return_value={"online": True, "stream_url": "https://cdn.example/stream",
                                                  "headers": {"Referer": "https://live.bilibili.com/1"}})
    with TestClient(app) as client:
        channel = client.post("/api/live/sessions", json={"url": "https://live.bilibili.com/1"}).json()
        session_id = channel["id"]
        service = app.state.live_subtitles
        service.start = Mock(return_value={"state": "starting"})
        service.stop = Mock(return_value={"state": "stopped"})
        path = f"/api/live/sessions/{session_id}/subtitles"
        assert client.post(path + "/start", json={"translate": "false"}).status_code == 400
        result = client.post(path + "/start", json={"translate": False, "language": "ja"})
        assert result.status_code == 200
        assert service.start.call_args.kwargs == {"translate": False, "language": "ja"}
        assert service.start.call_args.args[2]["Referer"].startswith("https://live.bilibili.com")
        assert client.get(path + "/draft").status_code == 404
        assert client.delete(f"/api/live/sessions/{session_id}").status_code == 200
        service.stop.assert_called_with(session_id)
