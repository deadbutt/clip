import sys
import types
import os
import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.whisper_runner import WhisperRunner, _RepeatedPhraseGuard


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    duration = 3.0


class FakeWhisperModel:
    last_kwargs = None

    def __init__(self, model, *, device, compute_type):
        self.model = model
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path, **kwargs):
        FakeWhisperModel.last_kwargs = kwargs
        return iter([FakeSegment(0.0, 1.25, " hello "), FakeSegment(1.5, 3.0, "world")]), FakeInfo()


class FakeLoopWhisperModel(FakeWhisperModel):
    def transcribe(self, path, **kwargs):
        FakeLoopWhisperModel.last_kwargs = kwargs
        segments = [FakeSegment(float(i), float(i + 1), "Annie's foot.") for i in range(12)]
        return iter(segments), FakeInfo()


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
        self.assertIs(FakeWhisperModel.last_kwargs["condition_on_previous_text"], False)
        self.assertEqual(FakeWhisperModel.last_kwargs["beam_size"], 5)
        self.assertEqual(FakeWhisperModel.last_kwargs["no_repeat_ngram_size"], 3)

    def test_repeated_phrase_guard_skips_looped_short_hallucinations(self):
        guard = _RepeatedPhraseGuard(max_consecutive=3, max_total=24)
        decisions = [guard.should_skip("Annie's foot.") for _ in range(6)]

        self.assertEqual(decisions, [False, False, False, True, True, True])

    def test_transcribe_filters_repeated_short_hallucination_loop(self):
        module = types.SimpleNamespace(WhisperModel=FakeLoopWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            runner = WhisperRunner("small", device="cpu", dtype="int8", beam_size=3)
            result = runner.transcribe("sample.mp4")

        self.assertEqual(result.generated_tokens, 12)
        self.assertEqual(result.text.count("Annie's foot."), 3)


if __name__ == "__main__":
    unittest.main()
