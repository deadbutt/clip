from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from moss_transcribe_diarize.app.downloader import _collect_subtitle_files, pick_best_subtitle
from moss_transcribe_diarize.app.jobs import JobManager, JobRecord
from moss_transcribe_diarize.subtitle import SubtitleSegment, clean_source_captions, parse_srt


def seg(start: float, end: float, text: str, index: int = 1) -> SubtitleSegment:
    return SubtitleSegment(id=f"seg_{index:04d}", start=start, end=end, speaker="S00", text=text)


class CollectSubtitleFilesTest(unittest.TestCase):
    def test_skips_machine_translated_tracks(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("input.en.srt", "input.en-orig.srt", "input.en-US.srt", "input.en-zh-Hans.srt", "input.zh-Hans.srt"):
                (Path(tmp) / name).write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
            entries = _collect_subtitle_files(Path(tmp))
            langs = sorted(e["lang"] for e in entries)
            self.assertEqual(langs, ["en", "en-US", "en-orig", "zh-Hans"])
            kinds = {e["lang"]: e["kind"] for e in entries}
            self.assertEqual(kinds["en"], "manual")
            self.assertEqual(kinds["en-orig"], "auto")

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_collect_subtitle_files(Path(tmp)), [])


class PickBestSubtitleTest(unittest.TestCase):
    def test_prefers_manual_over_auto(self):
        subs = [
            {"lang": "en", "path": "a.srt", "kind": "auto"},
            {"lang": "ja", "path": "b.srt", "kind": "manual"},
        ]
        self.assertEqual(pick_best_subtitle(subs)["path"], "b.srt")

    def test_lang_priority_within_same_kind(self):
        subs = [
            {"lang": "en", "path": "en.srt", "kind": "auto"},
            {"lang": "zh-Hans", "path": "zh.srt", "kind": "auto"},
        ]
        self.assertEqual(pick_best_subtitle(subs)["path"], "zh.srt")

    def test_empty(self):
        self.assertIsNone(pick_best_subtitle([]))


class CleanSourceCaptionsTest(unittest.TestCase):
    def test_strips_tags_and_entities(self):
        out = clean_source_captions([seg(0.0, 1.0, "<c>Tom &amp; Jerry</c>")])
        self.assertEqual(out[0].text, "Tom & Jerry")

    def test_merges_rolling_duplicate(self):
        out = clean_source_captions(
            [seg(0.0, 1.0, "hello world"), seg(0.4, 1.5, "hello world")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].end, 1.5)

    def test_merges_prefix_completion(self):
        out = clean_source_captions(
            [seg(0.0, 1.0, "hello how are"), seg(0.2, 1.8, "hello how are you")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "hello how are you")
        self.assertEqual(out[0].end, 1.8)

    def test_keeps_separated_repeats(self):
        out = clean_source_captions(
            [seg(0.0, 1.0, "la la la"), seg(5.0, 6.0, "la la la")]
        )
        self.assertEqual(len(out), 2)

    def test_drops_music_markers_and_empty(self):
        out = clean_source_captions(
            [seg(0.0, 1.0, "♪♪"), seg(1.0, 2.0, "<i>  </i>"), seg(2.0, 3.0, "real text")]
        )
        self.assertEqual([s.text for s in out], ["real text"])

    def test_renumbers_ids(self):
        out = clean_source_captions(
            [seg(0.0, 1.0, "a"), seg(1.0, 2.0, "♪"), seg(2.0, 3.0, "b")]
        )
        self.assertEqual([s.id for s in out], ["seg_0001", "seg_0002"])


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:02,000
<c>Welcome</c>

2
00:00:01,500 --> 00:00:02,500
Welcome back

3
00:00:02,500 --> 00:00:03,000
♪

4
00:00:03,000 --> 00:00:04,000
today&apos;s plan
"""


class LoadSourceCaptionsTest(unittest.TestCase):
    def test_loads_and_cleans_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt_path = Path(tmp) / "input.en.srt"
            srt_path.write_text(SAMPLE_SRT, encoding="utf-8")
            job = JobRecord(
                id="t1",
                status="queued",
                media_name="x",
                input_path="x",
                job_dir=tmp,
                inference_prompt="p",
                max_length=1,
                max_new_tokens=1,
                decoding="greedy",
                temperature=None,
                source="url",
                source_url="https://example.com/v",
                source_subtitles=[{"lang": "en", "path": str(srt_path), "kind": "manual"}],
            )

            class Stub:
                def _save_job(self, job):
                    pass

            segments = JobManager._load_source_captions(Stub(), job)
            self.assertIsNotNone(segments)
            assert segments is not None
            # 滚动重复合并掉一条、音乐行剔除 → 剩 2 条
            self.assertEqual([s.text for s in segments], ["Welcome back", "today's plan"])
            self.assertEqual(job.transcript_source, "captions:manual:en")

    def test_force_transcribe_skips_captions(self):
        job = JobRecord(
            id="t2",
            status="queued",
            media_name="x",
            input_path="x",
            job_dir=".",
            inference_prompt="p",
            max_length=1,
            max_new_tokens=1,
            decoding="greedy",
            temperature=None,
            source="url",
            source_subtitles=[{"lang": "en", "path": "nope.srt", "kind": "manual"}],
            force_transcribe=True,
        )

        class Stub:
            def _save_job(self, job):
                pass

        self.assertIsNone(JobManager._load_source_captions(Stub(), job))

    def test_auto_only_falls_back_to_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt_path = Path(tmp) / "input.en-orig.srt"
            srt_path.write_text(SAMPLE_SRT, encoding="utf-8")
            job = JobRecord(
                id="t3",
                status="queued",
                media_name="x",
                input_path="x",
                job_dir=tmp,
                inference_prompt="p",
                max_length=1,
                max_new_tokens=1,
                decoding="greedy",
                temperature=None,
                source="url",
                source_subtitles=[{"lang": "en-orig", "path": str(srt_path), "kind": "auto"}],
            )

            class Stub:
                def _save_job(self, job):
                    pass

            self.assertIsNone(JobManager._load_source_captions(Stub(), job))
            self.assertIsNone(job.transcript_source)


if __name__ == "__main__":
    unittest.main()
