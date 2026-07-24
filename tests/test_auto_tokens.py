from __future__ import annotations

import unittest

from moss_transcribe_diarize.app.jobs import recommend_max_new_tokens


class RecommendMaxNewTokensTest(unittest.TestCase):
    def test_none_or_nonpositive_duration_returns_none(self):
        self.assertIsNone(recommend_max_new_tokens(None))
        self.assertIsNone(recommend_max_new_tokens(0))
        self.assertIsNone(recommend_max_new_tokens(-5))

    def test_short_audio_hits_minimum_floor(self):
        # 60s -> raw 840 tokens, below the 2048 floor.
        self.assertEqual(recommend_max_new_tokens(60), 2048)

    def test_nine_minute_audio(self):
        # 540s -> 540 * 14 = 7560 -> rounded up to 7680.
        self.assertEqual(recommend_max_new_tokens(540), 7680)

    def test_thirty_minute_audio(self):
        # 1800s -> 1800 * 14 = 25200 -> rounded up to 25600.
        self.assertEqual(recommend_max_new_tokens(1800), 25600)

    def test_very_long_audio_clamped_by_context(self):
        # 100min -> raw would be 84480, but prompt estimate (~75000) leaves
        # only ~55560 of context, which clamps the recommendation below 65536.
        self.assertEqual(recommend_max_new_tokens(6000), 55560)

    def test_never_exceeds_hard_cap(self):
        # With a huge max_length so context is not the limiter, still cap at 65536.
        self.assertEqual(recommend_max_new_tokens(6000, max_length=1_000_000), 65536)

    def test_respects_smaller_max_length(self):
        # Tight context window should clamp the recommendation.
        self.assertLessEqual(recommend_max_new_tokens(1800, max_length=40000), 40000)


if __name__ == "__main__":
    unittest.main()
