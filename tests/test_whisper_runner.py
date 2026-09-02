import os
import sys
import types
import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.whisper_runner import (
    WhisperRunner,
    _RepeatedPhraseGuard,
    _is_likely_hallucination,
)


class FakeWord:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


class FakeSegment:
    def __init__(self, start, end, text, words=None, avg_logprob=0.0, no_speech_prob=0.0):
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


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
        return (
            iter(
                [
                    FakeSegment(0.0, 1.25, " hello ", words=[FakeWord(0.0, 0.4, " hello"), FakeWord(0.4, 1.25, " world")]),
                    FakeSegment(1.5, 3.0, "world"),
                ]
            ),
            FakeInfo(),
        )


class FakeLoopWhisperModel(FakeWhisperModel):
    def transcribe(self, path, **kwargs):
        FakeLoopWhisperModel.last_kwargs = kwargs
        segments = [FakeSegment(float(i), float(i + 1), "Annie's foot.") for i in range(12)]
        return iter(segments), FakeInfo()


class FakeHallucinationWhisperModel(FakeWhisperModel):
    """返回包含幻觉段和正常段混合的模拟模型。"""

    def transcribe(self, path, **kwargs):
        FakeHallucinationWhisperModel.last_kwargs = kwargs
        segments = [
            FakeSegment(0.0, 2.0, "Hello world", words=[FakeWord(0.0, 1.0, "Hello"), FakeWord(1.0, 2.0, "world")]),
            FakeSegment(2.0, 46.4, "Satsang with Mooji Oh, oh,"),
            FakeSegment(46.4, 48.0, "ah..."),
            FakeSegment(48.0, 50.0, "This is real speech."),
        ]
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
        self.assertIs(FakeWhisperModel.last_kwargs["condition_on_previous_text"], True)
        self.assertEqual(FakeWhisperModel.last_kwargs["beam_size"], 5)
        self.assertEqual(FakeWhisperModel.last_kwargs["no_repeat_ngram_size"], 3)
        self.assertIs(FakeWhisperModel.last_kwargs["word_timestamps"], True)
        # 带 words 的 segment 收集词时间戳，不带的（FakeLoop 等）正常跳过
        self.assertEqual(result.words, [(0.0, 0.4, " hello"), (0.4, 1.25, " world")])

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

    def test_hallucination_detection_patterns(self):
        # 无条件幻觉文本
        self.assertTrue(_is_likely_hallucination("ah..."))
        self.assertTrue(_is_likely_hallucination("Satsang with Mooji Oh, oh,"))
        self.assertTrue(_is_likely_hallucination("um"))

        # 条件幻觉文本：thank you / bye 需配合低置信度/高静音概率
        self.assertTrue(_is_likely_hallucination("Thank you.", avg_logprob=-0.9))
        self.assertTrue(_is_likely_hallucination("Bye.", no_speech_prob=0.7))
        self.assertFalse(_is_likely_hallucination("Thank you."))  # 无置信度信息时不丢弃

        # 下划线幻觉
        self.assertTrue(_is_likely_hallucination("_______"))

        # prompt 回显（输出与 initial_prompt 高度重叠）
        self.assertTrue(_is_likely_hallucination("the quick brown fox", prompt="the quick brown fox"))

        # 低置信度 + 长文本
        self.assertTrue(_is_likely_hallucination("a" * 100, avg_logprob=-1.5))
        self.assertFalse(_is_likely_hallucination("short", avg_logprob=-1.5))

        # 高静音概率
        self.assertTrue(_is_likely_hallucination("anything", no_speech_prob=0.9))
        self.assertFalse(_is_likely_hallucination("anything", no_speech_prob=0.5))

        # 正常文本不应被误杀
        self.assertFalse(_is_likely_hallucination("Hello world, this is a test."))
        self.assertFalse(_is_likely_hallucination("Thank you for your time, that was very helpful."))
        self.assertFalse(_is_likely_hallucination("Ah, I see what you mean."))

    def test_transcribe_filters_hallucination_segments(self):
        module = types.SimpleNamespace(WhisperModel=FakeHallucinationWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            runner = WhisperRunner("small", device="cpu", dtype="int8")
            result = runner.transcribe("sample.mp4")

        self.assertEqual(result.generated_tokens, 4)
        self.assertIn("Hello world", result.text)
        self.assertIn("This is real speech.", result.text)
        self.assertNotIn("Satsang with Mooji", result.text)
        self.assertNotIn("ah...", result.text)


if __name__ == "__main__":
    unittest.main()
