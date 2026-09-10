import unittest
from pathlib import Path

from moss_transcribe_diarize.app.ffmpeg import FFmpegProcessError, _build_clip_filter_graph


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

    def test_process_error_keeps_diagnostics_out_of_user_message(self):
        error = FFmpegProcessError(1, "ffmpeg version x\nlibass verbose internals")

        self.assertIn("媒体处理失败", str(error))
        self.assertNotIn("libass", str(error))
        self.assertIn("libass", error.detail)


if __name__ == "__main__":
    unittest.main()
