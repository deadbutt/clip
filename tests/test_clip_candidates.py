import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.clips import (
    ClipCandidate,
    _dedupe_candidates,
    generate_clip_candidates,
    rebase_segments_for_clip,
)
from moss_transcribe_diarize.subtitle import SubtitleSegment


class _StubRunner:
    model_path = "fake-model"


class ClipCandidateTest(unittest.TestCase):
    def test_generate_candidates_prefers_complete_hooky_windows(self):
        segments = [
            SubtitleSegment(
                id=f"seg_{index:04d}",
                start=index * 10.0,
                end=index * 10.0 + 8.0,
                speaker="S00",
                text="why this matters is actually important, so here is the result.",
            )
            for index in range(20)
        ]

        clips = generate_clip_candidates(
            segments,
            min_duration=45.0,
            target_duration=120.0,
            max_duration=180.0,
            limit=3,
        )

        # 重叠候选合并后条数会比硬性去重少,但每条仍满足时长约束
        self.assertGreaterEqual(len(clips), 1)
        self.assertLessEqual(len(clips), 3)
        self.assertEqual(clips[0].id, "clip_001")
        self.assertGreaterEqual(clips[0].duration, 45.0)
        self.assertLessEqual(clips[0].duration, 180.0)
        self.assertIn("hook", clips[0].reason)

    def test_generated_candidates_do_not_contain_each_other(self):
        # Sliding windows are intentionally dense; the returned shortlist must
        # keep only representative windows instead of nesting one clip inside
        # another.
        segments = [
            SubtitleSegment(
                id=f"seg_{index:04d}",
                start=index * 10.0,
                end=index * 10.0 + 8.0,
                speaker="S00",
                text="why this matters is actually important, so here is the result.",
            )
            for index in range(40)
        ]

        clips = generate_clip_candidates(
            segments,
            min_duration=45.0,
            target_duration=120.0,
            max_duration=300.0,
            limit=8,
            merge_expansion_limit=90.0,
        )

        for index, left in enumerate(clips):
            for right in clips[index + 1 :]:
                intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
                shorter = max(1.0, min(left.duration, right.duration))
                self.assertLessEqual(intersection / shorter, 0.35)

        primary = [clip for clip in clips if clip.lane == "primary"]
        alternates = [clip for clip in clips if clip.lane == "alternate"]
        self.assertLessEqual(len(primary), 3)
        self.assertLessEqual(len(alternates), max(1, (len(primary) + 1) // 2))
        for index, left in enumerate(primary):
            for right in primary[index + 1 :]:
                intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
                self.assertEqual(intersection, 0.0)

    def test_rule_screening_can_return_lower_scored_alternates(self):
        segments = [
            SubtitleSegment(
                id=f"seg_{index:04d}",
                start=index * 10.0,
                end=index * 10.0 + 8.0,
                speaker="S00",
                text=(
                    "why this matters is actually important, so here is the result."
                    if index < 20
                    else "ordinary sentence."
                ),
            )
            for index in range(40)
        ]

        clips = generate_clip_candidates(
            segments,
            min_duration=45.0,
            target_duration=120.0,
            max_duration=300.0,
            limit=8,
            merge_expansion_limit=90.0,
            include_alternates=True,
        )
        primary = {clip.id: clip for clip in clips if clip.lane == "primary"}
        alternates = [clip for clip in clips if clip.lane == "alternate"]

        self.assertTrue(alternates)
        for alternate in alternates:
            self.assertIn(alternate.parent_id, primary)
            self.assertLess(alternate.score, primary[alternate.parent_id].score)

    def test_rebase_segments_uses_clip_local_timeline(self):
        segments = [
            SubtitleSegment(id="a", start=98.0, end=102.0, speaker="S00", text="before"),
            SubtitleSegment(id="b", start=105.0, end=110.0, speaker="S00", text="inside"),
            SubtitleSegment(id="c", start=119.0, end=125.0, speaker="S00", text="after"),
        ]

        clipped = rebase_segments_for_clip(segments, start=100.0, end=120.0)

        self.assertEqual([(item.start, item.end) for item in clipped], [(0.0, 2.0), (5.0, 10.0), (19.0, 20.0)])
        self.assertEqual([item.text for item in clipped], ["before", "inside", "after"])


class DedupeCandidatesTest(unittest.TestCase):
    @staticmethod
    def _candidate(cid: str, start: float, end: float, score: float = 50.0) -> ClipCandidate:
        return ClipCandidate(
            id=cid,
            start=start,
            end=end,
            score=score,
            title="t",
            reason="r",
            text="x" * 10,
            segment_ids=[cid],
        )

    def test_overlapping_candidates_merge_to_outer_bounds(self):
        a = self._candidate("a", 0.0, 100.0, score=90.0)
        b = self._candidate("b", 10.0, 130.0, score=60.0)  # 重叠 90/100 = 90%

        merged = _dedupe_candidates([a, b], max_duration=180.0)

        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 130.0))
        self.assertEqual(merged[0].score, 90.0)
        self.assertEqual(merged[0].segment_ids, ["a", "b"])

    def test_merge_rejected_when_exceeding_max_duration(self):
        # 重叠 80% 但合并后 0~200 超过 max_duration=180: 退回丢弃低分条
        a = self._candidate("a", 0.0, 100.0, score=90.0)
        b = self._candidate("b", 20.0, 200.0, score=60.0)

        merged = _dedupe_candidates([a, b], max_duration=180.0)

        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 100.0))

    def test_near_duplicate_windows_are_deduped_even_when_union_exceeds_cap(self):
        a = self._candidate("a", 0.0, 180.0, score=90.0)
        b = self._candidate("b", 78.0, 256.0, score=60.0)

        merged = _dedupe_candidates([a, b], max_duration=180.0)

        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 180.0))

    def test_disjoint_candidates_are_both_kept(self):
        a = self._candidate("a", 0.0, 100.0)
        b = self._candidate("b", 500.0, 600.0)

        merged = _dedupe_candidates([a, b], max_duration=180.0)

        self.assertEqual(len(merged), 2)

    def test_chain_merge_rechecks_against_kept_items(self):
        # a+b 合并成 0~130 后与 c(90~175) 重叠 80%,必须链式合并而不是留下重叠对
        a = self._candidate("a", 0.0, 100.0, score=90.0)
        b = self._candidate("b", 30.0, 130.0, score=70.0)  # 与 a 重叠 70%
        c = self._candidate("c", 30.0, 175.0, score=50.0)  # 与合并结果 0~130 重叠 100/130=77%

        merged = _dedupe_candidates([a, b, c], max_duration=180.0)

        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 175.0))

    def test_unlimited_max_duration_allows_long_merge(self):
        # max_duration=None 时不设上限,原本超 180 的合并(0~200)也能完成
        a = self._candidate("a", 0.0, 100.0, score=90.0)
        b = self._candidate("b", 20.0, 200.0, score=60.0)  # 重叠 80%

        merged = _dedupe_candidates([a, b], max_duration=None)

        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 200.0))

    def test_generate_candidates_without_cap_allows_long_windows(self):
        segments = [
            SubtitleSegment(
                id=f"seg_{index:04d}",
                start=index * 10.0,
                end=index * 10.0 + 8.0,
                speaker="S00",
                text="why this matters is actually important, so here is the result.",
            )
            for index in range(40)
        ]

        clips = generate_clip_candidates(
            segments,
            min_duration=45.0,
            target_duration=120.0,
            max_duration=None,
            limit=3,
        )

        self.assertGreaterEqual(len(clips), 1)
        # 不设上限时窗口可以超过原 180 硬限(这正是放开的目的)
        self.assertGreaterEqual(clips[0].duration, 45.0)
        self.assertLessEqual(clips[0].duration, 400.0)


class ClipCandidateCacheTest(unittest.TestCase):
    def test_rule_candidates_are_cached_until_segments_change(self):
        from moss_transcribe_diarize.app.jobs import JobManager, JobRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = JobManager(root, _StubRunner(), prompt="p", max_length=1024, max_new_tokens=8)
            job_dir = root / "job1"
            job_dir.mkdir()
            segments_path = job_dir / "segments.json"
            payload = [
                {
                    "id": "seg_0001",
                    "start": 0.0,
                    "end": 60.0,
                    "speaker": "S00",
                    "text": "why this matters and here is the result.",
                },
                {
                    "id": "seg_0002",
                    "start": 61.0,
                    "end": 122.0,
                    "speaker": "S00",
                    "text": "This is a complete second sentence.",
                },
            ]
            segments_path.write_text(json.dumps(payload), encoding="utf-8")
            manager._jobs["job1"] = JobRecord(
                id="job1",
                status="waiting_review",
                media_name="sample.wav",
                input_path=str(root / "input.wav"),
                job_dir=str(job_dir),
                inference_prompt="p",
                max_length=1024,
                max_new_tokens=8,
                decoding="greedy",
                temperature=None,
            )

            with patch("moss_transcribe_diarize.app.clip_service.generate_clip_candidates", wraps=generate_clip_candidates) as scorer:
                first = manager.list_clip_candidates("job1", include_alternates=True)
                second = manager.list_clip_candidates("job1", include_alternates=True)
                self.assertEqual(first, second)
                self.assertEqual(scorer.call_count, 1)

                payload[1]["text"] += " Why?"
                segments_path.write_text(json.dumps(payload), encoding="utf-8")
                manager.list_clip_candidates("job1", include_alternates=True)
                self.assertEqual(scorer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
