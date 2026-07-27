import unittest

from moss_transcribe_diarize.app.text_translator import (
    _chat_content,
    _parse_clip_ranking,
    _parse_translation_array,
    apply_translations,
    TextTranslator,
    _RetryableTranslationError,
)
from moss_transcribe_diarize.subtitle import SubtitleSegment


class SplittingTranslator(TextTranslator):
    def _translate_batch(self, segments, *, target_language, context_before=None, context_after=None):
        if len(segments) > 1:
            return [f"bad-{index}" for index in range(len(segments) // 2)]
        return [f"{target_language}: {segments[0].text}"]


class MalformedJsonTranslator(TextTranslator):
    def _translate_batch(self, segments, *, target_language, context_before=None, context_after=None):
        raise _RetryableTranslationError("bad json")


class ContextRecordingTranslator(TextTranslator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts = []

    def _translate_batch(self, segments, *, target_language, context_before=None, context_after=None):
        self.contexts.append(
            {
                "items": [segment.text for segment in segments],
                "before": [segment.text for segment in (context_before or [])],
                "after": [segment.text for segment in (context_after or [])],
            }
        )
        return [f"{target_language}: {segment.text}" for segment in segments]


class TextTranslatorTest(unittest.TestCase):
    def test_chat_content_from_ollama_response(self):
        response = {"message": {"role": "assistant", "content": '{"translations":["hello"]}'}}

        self.assertEqual(_chat_content(response), '{"translations":["hello"]}')

    def test_parse_translation_array_from_ollama_json_shape(self):
        content = '{"translations":["hello","world"]}'

        self.assertEqual(_parse_translation_array(content), ["hello", "world"])

    def test_parse_translation_array_from_object_items(self):
        content = '{"translations":[{"id":"seg_1","text":"hello"},{"id":"seg_2","translation":"world"}]}'

        self.assertEqual(_parse_translation_array(content), ["hello", "world"])

    def test_translate_segments_splits_mismatched_batches(self):
        segments = [
            SubtitleSegment(id=f"seg_{index}", start=index, end=index + 1, speaker="S00", text=f"text {index}")
            for index in range(6)
        ]

        translator = SplittingTranslator(base_url="http://unused", model="unused", provider="ollama")

        translated = translator.translate_segments(segments, target_language="English")

        self.assertEqual(len(translated), 6)
        self.assertEqual(translated[0], "English: text 0")
        self.assertEqual(translated[-1], "English: text 5")

    def test_translate_segments_falls_back_on_single_malformed_json(self):
        segment = SubtitleSegment(id="seg_1", start=0, end=1, speaker="S00", text="keep me")
        translator = MalformedJsonTranslator(base_url="http://unused", model="unused", provider="ollama")

        self.assertEqual(translator.translate_segments([segment], target_language="Chinese"), ["keep me"])

    def test_translate_segments_reports_top_level_progress(self):
        segments = [
            SubtitleSegment(id=f"seg_{index}", start=index, end=index + 1, speaker="S00", text=f"text {index}")
            for index in range(5)
        ]
        translator = ContextRecordingTranslator(base_url="http://unused", model="unused")
        progress = []

        translator.translate_segments(
            segments,
            target_language="Chinese",
            batch_size=2,
            progress_callback=lambda done, total, start, size: progress.append((done, total, start, size)),
        )

        self.assertEqual(progress, [(2, 5, 0, 2), (4, 5, 2, 2), (5, 5, 4, 1)])

    def test_translate_segments_supplies_surrounding_context(self):
        segments = [
            SubtitleSegment(id=f"seg_{index}", start=index, end=index + 1, speaker="S00", text=f"text {index}")
            for index in range(5)
        ]
        translator = ContextRecordingTranslator(base_url="http://unused", model="unused")

        translator.translate_segments(segments, target_language="Chinese", batch_size=2, context_window=1)

        self.assertEqual(translator.contexts[1]["items"], ["text 2", "text 3"])
        self.assertEqual(translator.contexts[1]["before"], ["text 1"])
        self.assertEqual(translator.contexts[1]["after"], ["text 4"])

    def test_parse_translation_array_from_wrapped_model_output(self):
        self.assertEqual(_parse_translation_array('结果如下：\\n["你好", "世界"]'), ["你好", "世界"])

    def test_apply_bilingual_translation_preserves_timing_and_speaker(self):
        segment = SubtitleSegment(id="seg_0001", start=1.0, end=2.0, speaker="S04", text="hello")

        translated = apply_translations([segment], ["你好"], mode="bilingual")

        self.assertEqual(translated[0].speaker, "S04")
        self.assertEqual(translated[0].text, "你好\nhello")

    def test_parse_clip_ranking_from_wrapped_json(self):
        ranked = _parse_clip_ranking('结果：\n{"selected":[{"id":"clip_002","score":91,"title":"标题","reason":"完整"}]}')

        self.assertEqual(ranked[0]["id"], "clip_002")
        self.assertEqual(ranked[0]["score"], 91)


if __name__ == "__main__":
    unittest.main()
