import unittest
from pathlib import Path

from moss_transcribe_diarize.app.ffmpeg import _build_clip_filter_graph


class FfmpegClipTest(unittest.TestCase):
    def test_clip_is_retimed_before_local_subtitles_are_applied(self):
        graph = _build_clip_filter_graph(
            Path("clip.ass"),
            start=100.0,
            end=120.0,
            has_audio=True,
        )

        trim_position = graph.index("trim=start=100.000")
        reset_position = graph.index("setpts=PTS-STARTPTS")
        subtitle_position = graph.index("subtitles=clip.ass")
        self.assertLess(trim_position, reset_position)
        self.assertLess(reset_position, subtitle_position)
        self.assertIn("atrim=start=100.000:duration=20.000", graph)


if __name__ == "__main__":
    unittest.main()
