import unittest

import numpy as np

from moss_transcribe_diarize.app.speaker_labeler import (
    _apply_cluster_labels,
    _assign_turns_to_segments,
    _cluster_features,
)
from moss_transcribe_diarize.subtitle import SubtitleSegment


class SpeakerLabelerTest(unittest.TestCase):
    def test_cluster_features_separates_obvious_groups(self):
        features = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.1],
                [5.0, 5.1, 4.9],
                [5.2, 5.0, 5.1],
            ],
            dtype=np.float32,
        )

        labels = _cluster_features(features, max_speakers=2)

        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_target_speaker_count_forces_cluster_count(self):
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [2.0, 2.0],
                [2.1, 2.0],
                [4.0, 4.0],
                [4.1, 4.0],
                [6.0, 6.0],
                [6.1, 6.0],
            ],
            dtype=np.float32,
        )

        labels = _cluster_features(features, max_speakers=4, target_speakers=4)

        self.assertEqual(len(set(labels)), 4)

    def test_apply_cluster_labels_names_by_first_appearance_and_smooths_short_island(self):
        segments = [
            SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S00", text="a"),
            SubtitleSegment(id="seg_0002", start=2.0, end=2.8, speaker="S00", text="b"),
            SubtitleSegment(id="seg_0003", start=2.8, end=5.0, speaker="S00", text="c"),
            SubtitleSegment(id="seg_0004", start=5.0, end=7.0, speaker="S00", text="d"),
        ]

        labeled = _apply_cluster_labels(segments, [0, 1, 2, 3], [3, 9, 3, 9])

        self.assertEqual([segment.speaker for segment in labeled], ["S01", "S01", "S01", "S02"])

    def test_assign_turns_to_segments_uses_largest_overlap_and_stable_names(self):
        segments = [
            SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="", text="a"),
            SubtitleSegment(id="seg_0002", start=2.0, end=4.0, speaker="", text="b"),
            SubtitleSegment(id="seg_0003", start=4.0, end=6.0, speaker="", text="c"),
        ]
        turns = [
            (0.0, 1.8, "SPEAKER_12"),
            (1.8, 4.2, "SPEAKER_07"),
            (4.2, 6.0, "SPEAKER_12"),
        ]

        labeled = _assign_turns_to_segments(segments, turns)

        self.assertEqual([segment.speaker for segment in labeled], ["S01", "S02", "S01"])


if __name__ == "__main__":
    unittest.main()
