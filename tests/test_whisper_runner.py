import os
import sys
import types
import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.whisper_runner import (
    WhisperRunner,
    _is_likely_hallucination,
    _RepeatedPhraseGuard,
    _whisper_initial_prompt,
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


class FakePunctuationWhisperModel(FakeWhisperModel):
    def transcribe(self, path, **kwargs):
        FakePunctuationWhisperModel.last_kwargs = kwargs
        return iter([
            FakeSegment(0.0, 1.0, "..."),
            FakeSegment(1.0, 2.0, "Real speech."),
        ]), FakeInfo()


class WhisperRunnerTest(unittest.TestCase):
    def test_auto_language_has_no_english_example(self):
        from moss_transcribe_diarize.defaults import DEFAULT_PROMPT

        self.assertIsNone(_whisper_initial_prompt(DEFAULT_PROMPT, None))
        self.assertIsNone(_whisper_initial_prompt(DEFAULT_PROMPT, "ja"))
        self.assertEqual(_whisper_initial_prompt("固有名詞", None), "固有名詞")

    def test_multi_sample_detection_overrides_a_misleading_intro(self):
        import numpy as np

        runner = WhisperRunner("small")
        runner._model = types.SimpleNamespace(detect_language=unittest.mock.Mock(side_effect=[
            ("en", 0.99, [("en", 0.99), ("ja", 0.01)]),
            *[("ja", 0.95, [("ja", 0.95), ("en", 0.05)])] * 4,
        ]))
        with patch("moss_transcribe_diarize.app.whisper_runner._probe_duration", return_value=180):
            with patch("moss_transcribe_diarize.app.whisper_runner._decode_audio_sample", return_value=np.ones(16000, dtype=np.float32)) as decode:
                runner._detect_recording_language("recording.mkv")
        self.assertEqual(runner._detected_language, "ja")
        self.assertAlmostEqual(runner._language_probability, 0.762)
        self.assertEqual(len(runner._language_samples), 5)
        self.assertEqual(decode.call_args_list[-1].args, ("recording.mkv", 147.0, 30.0))

    def test_different_kana_sentences_are_not_treated_as_repetition(self):
        guard = _RepeatedPhraseGuard()
        self.assertEqual([guard.should_skip(t) for t in ["愛している", "愛されたい", "愛していた", "愛していこう"]], [False] * 4)

    def test_english_only_model_does_not_call_multilingual_detection(self):
        detect = unittest.mock.Mock(side_effect=RuntimeError("English-only model"))
        runner = WhisperRunner("small.en")
        runner._model = types.SimpleNamespace(detect_language=detect, model=types.SimpleNamespace(is_multilingual=False))
        runner._detect_recording_language("recording.wav")
        detect.assert_not_called()
        self.assertEqual(runner._detected_language, "en")

    def test_explicit_language_skips_sampling(self):
        runner = WhisperRunner("small", device="cpu", dtype="int8", language="ja")
        with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisperModel)}):
            with patch.object(runner, "_detect_recording_language") as detect:
                runner.transcribe("sample.wav")
        detect.assert_not_called()
        self.assertEqual(FakeWhisperModel.last_kwargs["language"], "ja")

    def test_detected_language_is_reused_for_retries_but_not_next_job(self):
        calls = []

        class DetectingModel(FakeWhisperModel):
            def transcribe(self, path, **kwargs):
                calls.append(kwargs["language"])
                return iter([]), types.SimpleNamespace(duration=0, language="ja", language_probability=0.98)

        runner = WhisperRunner("small", device="cpu", dtype="int8")
        with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=DetectingModel)}):
            with patch.object(runner, "_looks_sparse", return_value=True), patch.object(runner, "_recover_gaps", return_value=None):
                first = runner.transcribe("first.wav")
                runner.transcribe("second.wav")
        self.assertEqual(calls, [None, "ja", None, "ja"])
        self.assertEqual(first.language, "ja")
        self.assertEqual(first.language_probability, 0.98)

    def test_gap_quality_metrics_are_kept_with_recovered_words(self):
        runner = WhisperRunner("small", device="cpu", dtype="int8")
        recovered = (
            ["[4.00][S00]Recovered speech[5.00]"],
            [(4.0, 5.0, "Recovered speech")],
            [{"start": 4.0, "end": 5.0, "avg_logprob": -1.2, "no_speech_prob": 0.1, "compression_ratio": 1.0}],
        )
        with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisperModel)}):
            with patch.object(runner, "_recover_gaps", return_value=recovered):
                result = runner.transcribe("sample.wav")
        self.assertIn("Recovered speech", result.text)
        self.assertEqual(result.words[-1], (4.0, 5.0, "Recovered speech"))
        self.assertEqual(result.segment_metrics[-1]["avg_logprob"], -1.2)

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

    def test_transcribe_drops_punctuation_only_segments(self):
        module = types.SimpleNamespace(WhisperModel=FakePunctuationWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            result = WhisperRunner("small", device="cpu", dtype="int8").transcribe("sample.mp4")

        self.assertNotIn("...", result.text)
        self.assertIn("Real speech.", result.text)


if __name__ == "__main__":
    unittest.main()
