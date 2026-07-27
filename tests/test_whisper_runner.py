import sys
import types
import os
import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.whisper_runner import WhisperRunner


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    duration = 3.0


class FakeWhisperModel:
    def __init__(self, model, *, device, compute_type):
        self.model = model
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path, **kwargs):
        return iter([FakeSegment(0.0, 1.25, " hello "), FakeSegment(1.5, 3.0, "world")]), FakeInfo()


class WhisperRunnerTest(unittest.TestCase):
    def test_ensure_ffmpeg_on_path_prepends_portable_directory(self):
        from moss_transcribe_diarize.app import whisper_runner as module

        fake_tools = types.SimpleNamespace(
            ffmpeg=r"D:\MOSS-Transcribe-Diarize\tools\ffmpeg\bin\ffmpeg.exe",
            ffprobe=r"D:\MOSS-Transcribe-Diarize\tools\ffmpeg\bin\ffprobe.exe",
        )
        expected = str(module.Path(fake_tools.ffmpeg).resolve().parent)
        environ = {"PATH": r"C:\Windows\System32"}
        with patch.object(module, "detect_ffmpeg", return_value=fake_tools), patch.object(module.os, "environ", environ):
            module._ensure_ffmpeg_on_path()

        self.assertEqual(environ["PATH"].split(os.pathsep)[0], expected)

    def test_transcribe_returns_moss_compatible_transcript(self):
        module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            runner = WhisperRunner("small", device="cpu", dtype="int8")
            status = []
            result = runner.transcribe(
                "sample.mp4",
                status_callback=lambda state, progress, tokens=None: status.append((state, progress, tokens)),
            )

        self.assertEqual(result.text, "[0.00][S00]hello[1.25][1.50][S00]world[3.00]")
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(result.model, "small")
        self.assertEqual(status[-1], ("transcribing", 0.85, 2))


if __name__ == "__main__":
    unittest.main()
