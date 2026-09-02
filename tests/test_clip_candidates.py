import unittest

from moss_transcribe_diarize.app.clips import (
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

        self.assertEqual(len(clips), 3)
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


if __name__ == "__main__":
    unittest.main()
