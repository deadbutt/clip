from __future__ import annotations

import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.ffmpeg import (
    FFmpegAvailability,
    detect_ffmpeg,
    probe_video_size,
)


class FFmpegTest(unittest.TestCase):
    def test_detect_ffmpeg_reports_missing_tools(self):
        with patch("shutil.which", return_value=None), patch(
            "moss_transcribe_diarize.app.ffmpeg._detect_portable_ffmpeg",
            return_value=FFmpegAvailability(ffmpeg=None, ffprobe=None),
        ):
            availability = detect_ffmpeg()

        self.assertFalse(availability.available)
        self.assertEqual(availability.to_dict()["available"], False)

    def test_detect_ffmpeg_prefers_portable_copy(self):
        portable = FFmpegAvailability(ffmpeg=r"X:\tools\ffmpeg.exe", ffprobe=r"X:\tools\ffprobe.exe")
        with patch("shutil.which", return_value=r"C:\bin\ffmpeg.exe"), patch(
            "moss_transcribe_diarize.app.ffmpeg._detect_portable_ffmpeg",
            return_value=portable,
        ):
            availability = detect_ffmpeg()

        self.assertEqual(availability, portable)

    def test_probe_video_size_returns_default_without_ffprobe(self):
        with patch("moss_transcribe_diarize.app.ffmpeg.detect_ffmpeg") as mocked:
            mocked.return_value.ffprobe = None
            self.assertEqual(probe_video_size("missing.mp4"), (1920, 1080))


if __name__ == "__main__":
    unittest.main()
