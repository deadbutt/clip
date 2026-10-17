from __future__ import annotations

import subprocess

import pytest

from moss_transcribe_diarize.app.ffmpeg import detect_ffmpeg
from moss_transcribe_diarize.app.live_preview import LivePreviewService


def test_real_ffmpeg_preview_generates_browser_media_and_cleans_cache(tmp_path):
    ffmpeg = detect_ffmpeg().ffmpeg
    if not ffmpeg:
        pytest.skip("FFmpeg not installed")
    source = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "4",
                    "-c:v", "libx264", "-c:a", "aac", str(source)], check=True, timeout=20)
    service = LivePreviewService(tmp_path / "cache")
    try:
        assert service.start("room", str(source))["state"] == "buffering"
        service.process.wait(timeout=20)
        assert service.process.returncode == 0
        playlist = service.media_path("room", "stream.m3u8")
        assert "segment-" in playlist.read_text()
        assert service.media_path("room", "segment-000000.ts").stat().st_size > 0
        with pytest.raises(FileNotFoundError):
            service.media_path("other-room", "stream.m3u8")
        with pytest.raises(FileNotFoundError):
            service.media_path("room", "../preview.log")
        assert not (tmp_path / "videos").exists()
    finally:
        service.close()
    assert list((tmp_path / "cache").iterdir()) == []
    assert service.snapshot("room")["state"] == "idle"
