import json
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.hy_mt_translator import HyMtTranslator, _detect_server_executable, _port_is_free
from moss_transcribe_diarize.subtitle import SubtitleSegment


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _chat_payload(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


class HyMtTranslatorTest(unittest.TestCase):
    def test_skipped_segments_bypass_the_server_entirely(self):
        translator = HyMtTranslator()
        segments = [
            SubtitleSegment(id="seg_1", start=0, end=1, speaker="S01", text="Hello world."),
            SubtitleSegment(id="seg_2", start=1, end=2, speaker="S01", text="Oh."),
        ]

        def fake_urlopen(request, timeout=None):
            body = json.loads(request.data.decode("utf-8"))
            self.assertNotIn("Oh.", body["messages"][0]["content"])
            return _FakeResponse(_chat_payload("你好，世界。"))

        with patch.object(HyMtTranslator, "_ensure_server"), patch("urllib.request.urlopen", fake_urlopen):
            result = translator.translate_segments(segments, batch_size=2)

        self.assertEqual(result, ["你好，世界。", "Oh."])

    def test_prompt_uses_official_plain_text_style_and_greedy_decoding(self):
        translator = HyMtTranslator()
        seen: dict = {}

        def fake_urlopen(request, timeout=None):
            seen.update(json.loads(request.data.decode("utf-8")))
            return _FakeResponse(_chat_payload("你好。"))

        with patch.object(HyMtTranslator, "_ensure_server"), patch("urllib.request.urlopen", fake_urlopen):
            translator.translate_segments(
                [SubtitleSegment(id="seg_1", start=0, end=1, speaker="S01", text="Hello.")],
                batch_size=1,
            )

        self.assertEqual(seen["temperature"], 0.0)
        self.assertNotIn("top_p", seen)
        self.assertEqual(
            seen["messages"][0]["content"],
            "Translate the following segment into Chinese, without additional explanation: Hello.",
        )

    def test_failed_segment_falls_back_to_source_and_is_counted(self):
        translator = HyMtTranslator()
        segments = [SubtitleSegment(id="seg_1", start=0, end=1, speaker="S01", text="Hello world.")]

        def fake_urlopen(request, timeout=None):
            raise OSError("connection refused")

        with patch.object(HyMtTranslator, "_ensure_server"), patch("urllib.request.urlopen", fake_urlopen):
            result = translator.translate_segments(segments, batch_size=1)

        self.assertEqual(result, ["Hello world."])
        self.assertEqual(translator.failure_count, 1)

    def test_runtime_info_reports_missing_model_without_starting_server(self):
        translator = HyMtTranslator(model_path="definitely/missing.gguf")
        with patch.object(HyMtTranslator, "_health_ok", side_effect=AssertionError("must not probe")):
            info = translator.runtime_info()

        self.assertFalse(info["available"])
        self.assertFalse(info["model_found"])
        self.assertIn("未找到模型文件", info["reason"])

    def test_translate_raises_when_server_executable_is_missing(self):
        translator = HyMtTranslator(model_path=__file__)
        with patch("moss_transcribe_diarize.app.hy_mt_translator._detect_server_executable", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                translator._ensure_server()

        self.assertIn("llama-server", str(ctx.exception))


class ServerDetectionTest(unittest.TestCase):
    def test_explicit_executable_wins(self):
        found = _detect_server_executable(None, explicit=__file__)
        self.assertEqual(found, Path(__file__))

    def test_missing_explicit_executable_returns_none(self):
        self.assertIsNone(_detect_server_executable(None, explicit="definitely/missing.exe"))

    def test_port_probe_detects_a_bound_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            self.assertFalse(_port_is_free(port))


if __name__ == "__main__":
    unittest.main()
