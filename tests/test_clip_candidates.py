import unittest

from moss_transcribe_diarize.app.clips import (
    ClipCandidate,
    _dedupe_candidates,
    generate_clip_candidates,
    rebase_segments_for_clip,
)
from moss_transcribe_diarize.subtitle import SubtitleSegment


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


if __name__ == "__main__":
    unittest.main()
