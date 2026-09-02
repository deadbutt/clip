from __future__ import annotations

import json
import unittest

from moss_transcribe_diarize.subtitle import SubtitleSegment, SubtitleStyle, export_ass, export_json, export_srt
from moss_transcribe_diarize.subtitle.export import format_ass_time, format_srt_time


class SubtitleExportTest(unittest.TestCase):
    def test_time_formatters(self):
        self.assertEqual(format_srt_time(3661.234), "01:01:01,234")
        self.assertEqual(format_ass_time(3661.23), "1:01:01.23")

    def test_export_srt(self):
        text = export_srt([SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")])

        self.assertIn("00:00:00,500 --> 00:00:02,000", text)
        self.assertIn("S01: hello", text)

    def test_export_srt_with_speaker_names(self):
        text = export_srt(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            speaker_names={"S01": "Alice"},
        )

        self.assertIn("Alice: hello", text)

    def test_export_ass_defaults_to_single_style(self):
        text = export_ass(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            style=SubtitleStyle(font_size=42),
            video_width=1280,
            video_height=720,
        )

        self.assertIn("PlayResX: 1280", text)
        self.assertNotIn("Speaker_S01", text)
        self.assertIn("Dialogue: 0,0:00:00.50,0:00:02.00,Default", text)
        self.assertIn("hello", text)

    def test_export_ass_with_speaker_colors(self):
        # 单说话人时按说话人配色没有意义,回落到 Default(统一颜色)。
        single = export_ass(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            style=SubtitleStyle(font_size=42, show_speaker=False, speaker_colors=True),
            video_width=1280,
            video_height=720,
        )
        self.assertNotIn("Speaker_S01", single)
        self.assertIn("Dialogue: 0,0:00:00.50,0:00:02.00,Default", single)

        # 多说话人时生成 per-speaker 样式并按说话人引用。
        multi = export_ass(
            [
                SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello"),
                SubtitleSegment("seg_0002", 2.0, 3.5, "S02", "world"),
            ],
            style=SubtitleStyle(font_size=42, show_speaker=False, speaker_colors=True),
            video_width=1280,
            video_height=720,
        )
        self.assertIn("Style: Speaker_S01,Noto Sans CJK SC,42", multi)
        self.assertIn("Style: Speaker_S02,Noto Sans CJK SC,42", multi)
        self.assertIn("Dialogue: 0,0:00:00.50,0:00:02.00,Speaker_S01", multi)
        self.assertIn("Dialogue: 0,0:00:02.00,0:00:03.50,Speaker_S02", multi)

    def test_export_ass_with_speaker_names(self):
        text = export_ass(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            style=SubtitleStyle(font_size=42, show_speaker=True, speaker_names={"S01": "Alice"}),
            video_width=1280,
            video_height=720,
        )

        self.assertIn("Alice: hello", text)

    def test_export_ass_merges_overlapping_segments_into_multiline_events(self):
        text = export_ass(
            [
                SubtitleSegment("seg_0001", 1.0, 5.0, "S01", "first"),
                SubtitleSegment("seg_0002", 3.0, 4.0, "S02", "second"),
            ],
            style=SubtitleStyle(show_speaker=False, speaker_colors=False),
        )

        self.assertIn("Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,first", text)
        self.assertIn("Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,first\\Nsecond", text)
        self.assertIn("Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,first", text)

    def test_export_json(self):
        data = json.loads(export_json([SubtitleSegment("seg_0001", 0, 1, "S01", "hello")]))

        self.assertEqual(data[0]["id"], "seg_0001")
        self.assertEqual(data[0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
