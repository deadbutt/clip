import json
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.hy_mt_translator import (
    HyMtTranslator,
    _detect_server_executable,
    _port_is_free,
    _with_subtitle_context,
)
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

    def test_protected_terms_use_identity_mappings_and_keep_braces_literal(self):
        translator = HyMtTranslator()
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen.update(json.loads(request.data.decode("utf-8")))
            return _FakeResponse(_chat_payload("Neuro 使用 {tool}。"))

        with patch.object(HyMtTranslator, "_ensure_server"), patch("urllib.request.urlopen", fake_urlopen):
            result = translator.translate_segments(
                [SubtitleSegment(id="seg_1", start=0, end=1, speaker="S01", text="Neuro uses {tool}.")],
                protected_terms=("Neuro", "{tool}", "Neuro"),
            )
        prompt = seen["messages"][0]["content"]
        self.assertEqual(prompt.count("Neuro translates to Neuro"), 1)
        self.assertIn("{tool} translates to {tool}", prompt)
        self.assertIn("Neuro uses {tool}.", prompt)
        self.assertEqual(result, ["Neuro 使用 {tool}。"])

    def test_runtime_info_reports_missing_model_without_starting_server(self):
        translator = HyMtTranslator(model_path="definitely/missing.gguf")
        with patch.object(HyMtTranslator, "_health_ok", side_effect=AssertionError("must not probe")):
            info = translator.runtime_info()

        self.assertFalse(info["available"])
        self.assertFalse(info["model_found"])
        self.assertIn("未找到模型文件", info["reason"])

    def test_context_leak_retries_only_target_without_context(self):
        translator = HyMtTranslator(concurrency=1)
        items = [
            SubtitleSegment(id="one", start=0, end=1, speaker="S01", text="Neuro controls the car."),
            SubtitleSegment(id="two", start=2, end=3, speaker="S02", text="That sounds cool."),
        ]
        calls = []

        def translate(_self, template, text):
            calls.append((template, text))
            if text == items[1].text and "[Background Information]" in template:
                return "根据背景信息，Neuro 控制汽车。"
            return "Neuro 控制汽车。" if text == items[0].text else "听起来很酷。"

        with patch.object(HyMtTranslator, "_ensure_server"), patch.object(HyMtTranslator, "_translate_one", translate):
            output = translator.translate_segments(items, context_window=2, protected_terms=("Neuro",))
        self.assertEqual(output, ["Neuro 控制汽车。", "听起来很酷。"])
        self.assertEqual(translator.context_retry_count, 1)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("[Background Information]", calls[-1][0])
        self.assertIn("Neuro translates to Neuro", calls[-1][0])
        self.assertEqual(calls[-1][1], "That sounds cool.")

    def test_context_failure_does_not_drop_or_reorder_segments(self):
        translator = HyMtTranslator()
        items = [
            SubtitleSegment(id=str(i), start=i * 2, end=i * 2 + 1, speaker="S01", text=f"Source {i}.")
            for i in range(3)
        ]

        def translate(_self, template, text):
            if text == "Source 1.":
                raise RuntimeError("request failed")
            return "译文 " + text

        with patch.object(HyMtTranslator, "_ensure_server"), patch.object(HyMtTranslator, "_translate_one", translate):
            result = translator.translate_segments(items, context_window=2)
        self.assertEqual(result, ["译文 Source 0.", "Source 1.", "译文 Source 2."])
        self.assertEqual(translator.failure_count, 1)

    def test_default_context_can_be_disabled_per_request(self):
        translator = HyMtTranslator(context_window=2)
        items = [
            SubtitleSegment(id="one", start=0, end=1, speaker="S01", text="Previous sentence."),
            SubtitleSegment(id="two", start=2, end=3, speaker="S02", text="Current sentence."),
        ]
        prompts = []

        def translate(_self, template, text):
            prompts.append(template)
            return "译文。"

        with patch.object(HyMtTranslator, "_ensure_server"), patch.object(HyMtTranslator, "_translate_one", translate):
            translator.translate_segments(items)
            self.assertTrue(all("[Background Information]" in prompt for prompt in prompts))
            prompts.clear()
            translator.translate_segments(items, context_window=0)
            self.assertTrue(all("[Background Information]" not in prompt for prompt in prompts))

    def test_failed_context_retry_preserves_original(self):
        translator = HyMtTranslator()
        items = [
            SubtitleSegment(id=str(i), start=i * 2, end=i * 2 + 1, speaker="S01", text=f"Source {i}.")
            for i in range(2)
        ]
        with (
            patch.object(HyMtTranslator, "_ensure_server"),
            patch.object(HyMtTranslator, "_translate_one", return_value="根据背景信息，这不是目标句。"),
        ):
            output = translator.translate_segments(items)
        self.assertEqual(output, [s.text for s in items])
        self.assertEqual(translator.failure_count, 2)

    def test_translate_raises_when_server_executable_is_missing(self):
        translator = HyMtTranslator(model_path=__file__)
        with (
            patch.object(HyMtTranslator, "_health_ok", return_value=False),
            patch("moss_transcribe_diarize.app.hy_mt_translator._detect_server_executable", return_value=None),
        ):
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


class SubtitleContextTest(unittest.TestCase):
    def make_segment(self, index, text, start=None):
        start = index * 2 if start is None else start
        return SubtitleSegment(id=str(index), start=start, end=start + 1, speaker="S01", text=text)

    def test_context_preserves_order_source_and_literal_braces(self):
        items = [self.make_segment(i, text) for i, text in enumerate(
            ["First line.", "Use {tool}.", "This is the source.", "Following line.", "Last line."]
        )]
        prompt = _with_subtitle_context("Translate: {text}", items, 2, 2).format(text=items[2].text)
        self.assertLess(prompt.index("First line."), prompt.index("Use {tool}."))
        self.assertLess(prompt.index("Following line."), prompt.index("Last line."))
        self.assertEqual(prompt.count(items[2].text), 1)
        self.assertTrue(prompt.endswith("Translate: This is the source."))

    def test_zero_context_and_distant_subtitles_use_original_prompt(self):
        items = [self.make_segment(0, "Distant background.", start=0), self.make_segment(1, "Target.", start=30)]
        self.assertEqual(_with_subtitle_context("{text}", items, 1, 0), "{text}")
        self.assertEqual(_with_subtitle_context("{text}", items, 1, 2), "{text}")

    def test_context_is_bounded_without_truncating_the_target(self):
        items = [self.make_segment(i, "Long background " * 100) for i in range(3)]
        items[1].text = "Actual source " * 100
        prompt = _with_subtitle_context("{text}", items, 1, 2, max_chars=100).format(text=items[1].text)
        self.assertTrue(prompt.endswith(items[1].text))
        self.assertLess(prompt.count("Long background"), 8)

    def test_context_uses_original_neighbors_even_when_noise_is_skipped(self):
        items = [self.make_segment(i, text) for i, text in enumerate(["Too far.", "Oh.", "Target.", "Nearby."])]
        prompt = _with_subtitle_context("{text}", items, 2, 1).format(text=items[2].text)
        self.assertNotIn("Too far.", prompt)
        self.assertNotIn("Oh.", prompt)
        self.assertIn("Nearby.", prompt)


if __name__ == "__main__":
    unittest.main()
