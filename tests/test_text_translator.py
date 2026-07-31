import unittest

from moss_transcribe_diarize.app.text_translator import (
    _chat_content,
    _parse_clip_ranking,
    _parse_translation_array,
    _parse_unit_translation_array,
    _build_translation_units,
    apply_translations,
    clean_translation_text,
    collect_pretranslation_skips,
    TextTranslator,
    _RetryableTranslationError,
    translation_skip_reason,
    validate_translation_outputs,
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


class UnitRecordingTranslator(TextTranslator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unit_batches = []

    def _translate_units_batch(self, units, *, target_language, context_before=None, context_after=None):
        self.unit_batches.append(
            {
                "items": [[segment.text for segment in unit.segments] for unit in units],
                "before": [segment.text for segment in (context_before or [])],
                "after": [segment.text for segment in (context_after or [])],
            }
        )
        return [[f"{target_language}: {segment.text}" for segment in unit.segments] for unit in units]


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
            SubtitleSegment(id=f"seg_{index}", start=index, end=index + 1, speaker="S00", text=f"text {index}.")
            for index in range(6)
        ]

        translator = SplittingTranslator(base_url="http://unused", model="unused", provider="ollama")

        translated = translator.translate_segments(segments, target_language="English")

        self.assertEqual(len(translated), 6)
        self.assertEqual(translated[0], "English: text 0.")
        self.assertEqual(translated[-1], "English: text 5.")

    def test_translate_segments_falls_back_on_single_malformed_json(self):
        segment = SubtitleSegment(id="seg_1", start=0, end=1, speaker="S00", text="keep me")
        translator = MalformedJsonTranslator(base_url="http://unused", model="unused", provider="ollama")

        self.assertEqual(translator.translate_segments([segment], target_language="Chinese"), ["keep me"])

    def test_translate_segments_reports_top_level_progress(self):
        segments = [
            SubtitleSegment(id=f"seg_{index}", start=index, end=index + 1, speaker="S00", text=f"text {index}.")
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
            SubtitleSegment(id=f"seg_{index}", start=index, end=index + 1, speaker="S00", text=f"text {index}.")
            for index in range(5)
        ]
        translator = ContextRecordingTranslator(base_url="http://unused", model="unused")

        translator.translate_segments(segments, target_language="Chinese", batch_size=2, context_window=1)

        self.assertEqual(translator.contexts[1]["items"], ["text 2.", "text 3."])
        self.assertEqual(translator.contexts[1]["before"], ["text 1."])
        self.assertEqual(translator.contexts[1]["after"], ["text 4."])

    def test_translate_segments_groups_incomplete_sentence_parts(self):
        segments = [
            SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="I think"),
            SubtitleSegment(id="seg_2", start=1.1, end=2.0, speaker="S00", text="we should go."),
            SubtitleSegment(id="seg_3", start=4.0, end=5.0, speaker="S00", text="New topic."),
        ]
        translator = UnitRecordingTranslator(base_url="http://unused", model="unused")

        translated = translator.translate_segments(segments, target_language="Chinese", batch_size=3)

        self.assertEqual(translated, ["Chinese: I think", "Chinese: we should go.", "Chinese: New topic."])
        self.assertEqual(translator.unit_batches[0]["items"], [["I think", "we should go."], ["New topic."]])

    def test_parse_unit_translation_array_maps_parts_by_id(self):
        segments = [
            SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="I think"),
            SubtitleSegment(id="seg_2", start=1.1, end=2.0, speaker="S00", text="we should go."),
        ]
        unit = _build_translation_units(segments)[0]

        parsed = _parse_unit_translation_array(
            '{"translations":[{"id":"unit_0001","parts":[{"id":"seg_2","text":"我们该走。"},{"id":"seg_1","text":"我觉得"}]}]}',
            [unit],
        )

        self.assertEqual(parsed, [["我觉得", "我们该走。"]])

    def test_parse_unit_translation_array_accepts_single_part_shape(self):
        segment = SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="Done.")
        unit = _build_translation_units([segment])[0]

        parsed = _parse_unit_translation_array(
            '{"translations":[{"id":"unit_0001","parts":[{"id":"seg_1","text":"完成。"}]}]}',
            [unit],
        )

        self.assertEqual(parsed, [["完成。"]])

    def test_parse_translation_array_from_wrapped_model_output(self):
        self.assertEqual(_parse_translation_array('结果如下：\\n["你好", "世界"]'), ["你好", "世界"])

    def test_apply_bilingual_translation_preserves_timing_and_speaker(self):
        segment = SubtitleSegment(id="seg_0001", start=1.0, end=2.0, speaker="S04", text="hello")

        translated = apply_translations([segment], ["你好"], mode="bilingual")

        self.assertEqual(translated[0].speaker, "S04")
        self.assertEqual(translated[0].text, "你好\nhello")

    def test_apply_translation_preserves_segment_count_when_model_drops_items(self):
        segments = [
            SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="hello"),
            SubtitleSegment(id="seg_2", start=1.0, end=2.0, speaker="S00", text="world"),
        ]

        translated = apply_translations(segments, ["你好"], mode="replace")

        self.assertEqual([segment.text for segment in translated], ["你好", "world"])

    def test_apply_bilingual_translation_does_not_duplicate_passthrough_text(self):
        segment = SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="TTS")

        translated = apply_translations([segment], ["TTS"], mode="bilingual")

        self.assertEqual(translated[0].text, "TTS")

    def test_clean_translation_text_rejects_model_artifacts(self):
        self.assertEqual(clean_translation_text("<tool_call>\n{}", fallback="hello"), "hello")

    def test_clean_translation_text_rejects_suspicious_expansion(self):
        source = "Ani, Tart, Ani, Tart."
        expanded = "阿尼、塔尔特、" * 80

        self.assertEqual(clean_translation_text(expanded, fallback=source), source)

    def test_validate_translation_outputs_reports_artifacts_and_count_mismatch(self):
        segments = [
            SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="hello"),
            SubtitleSegment(id="seg_2", start=1.0, end=2.0, speaker="S00", text="world"),
        ]

        issues = validate_translation_outputs(segments, ['{"translations":["bad"]}'])

        self.assertEqual(issues[0]["type"], "count_mismatch")
        self.assertEqual(issues[1]["type"], "model_artifact")

    def test_validate_translation_outputs_reports_suspicious_expansion(self):
        segment = SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="Ani, Tart.")

        issues = validate_translation_outputs([segment], ["阿尼、塔尔特、" * 80])

        self.assertEqual(issues[0]["type"], "suspicious_expansion")

    def test_translation_skip_reason_detects_noise_and_repeated_chants(self):
        self.assertEqual(translation_skip_reason("[gift sound]"), "bracketed_effect")
        self.assertEqual(translation_skip_reason("WQ."), "short_code_or_noise")
        self.assertEqual(translation_skip_reason("Ani, Tart, Ani, Tart, Ani, Tart."), "repeated_chant")
        self.assertEqual(translation_skip_reason("uh uh uh"), "filler_noise")
        self.assertEqual(translation_skip_reason("I'm honey."), "known_transcript_noise")

    def test_collect_pretranslation_skips_includes_timing_and_reason(self):
        segments = [
            SubtitleSegment(id="seg_1", start=0.0, end=1.0, speaker="S00", text="[gift sound]"),
            SubtitleSegment(id="seg_2", start=1.0, end=2.0, speaker="S00", text="Welcome back."),
        ]

        skips = collect_pretranslation_skips(segments)

        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0]["id"], "seg_1")
        self.assertEqual(skips[0]["reason"], "bracketed_effect")

    def test_parse_clip_ranking_from_wrapped_json(self):
        ranked = _parse_clip_ranking('结果：\n{"selected":[{"id":"clip_002","score":91,"title":"标题","reason":"完整"}]}')

        self.assertEqual(ranked[0]["id"], "clip_002")
        self.assertEqual(ranked[0]["score"], 91)


if __name__ == "__main__":
    unittest.main()
