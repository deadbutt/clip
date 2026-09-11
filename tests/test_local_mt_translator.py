import sys
import types
import unittest
from unittest.mock import patch

from moss_transcribe_diarize.app.local_mt_translator import LocalMtTranslator
from moss_transcribe_diarize.subtitle import SubtitleSegment


class _Result:
    hypotheses = [["translated"]]


class _FakeTranslator:
    def __init__(self, device: str, compute_type: str):
        self.device = device
        self.compute_type = compute_type

    def translate_batch(self, _tokens, **_kwargs):
        if self.device == "auto":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return [_Result()]


class _FakeSourceSentencePiece:
    def encode(self, _text, out_type=str):
        return ["hello"]


class _FakeTargetSentencePiece:
    def decode(self, _tokens):
        return "你好"


class LocalMtTranslatorTest(unittest.TestCase):
    def test_auto_translation_falls_back_to_cpu_when_cuda_library_is_missing(self):
        fake_ctranslate2 = types.SimpleNamespace(
            Translator=lambda _model_dir, device, compute_type: _FakeTranslator(device, compute_type)
        )
        translator = LocalMtTranslator(device="auto", compute_type="auto")
        translator._translator = _FakeTranslator("auto", "auto")
        translator._source_sp = _FakeSourceSentencePiece()
        translator._target_sp = _FakeTargetSentencePiece()

        with patch.dict(sys.modules, {"ctranslate2": fake_ctranslate2}):
            result = translator.translate_segments(
                [SubtitleSegment(id="seg_1", start=0, end=1, speaker="S01", text="Hello world.")],
                batch_size=1,
            )

        self.assertEqual(result, ["你好"])
        self.assertEqual(translator.device, "cpu")
        self.assertEqual(translator.compute_type, "int8")


if __name__ == "__main__":
    unittest.main()
