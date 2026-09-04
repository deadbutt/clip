"""音频时间轴窟窿检测（录屏丢帧）的单元测试。"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app import audio_timeline

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


class _FakeTools:
    ffmpeg = "ffmpeg"
    ffprobe = "ffprobe"


class _FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _ffprobe_stub(pts):
    """构造 stub：ffprobe 输出给定的 pts_time 包序列。"""
    stdout = json.dumps({"packets": [{"pts_time": f"{p:.6f}"} for p in pts]})
    return lambda cmd, **kwargs: _FakeProc(stdout=stdout)


def _clean_pts(count: int, frame: float = 0.02) -> list[float]:
    return [i * frame for i in range(count)]


def _holey_pts() -> list[float]:
    """300 帧 20ms 音频，两处 PTS 跳变（0.5s / 1.0s 窟窿）。"""
    pts: list[float] = []
    t = 0.0
    for i in range(300):
        pts.append(t)
        t += 0.02
        if i == 100:
            t += 0.5
        if i == 200:
            t += 1.0
    return pts


class HolesFromPtsTest(unittest.TestCase):
    def _analyze(self, pts):
        with (
            patch.object(audio_timeline, "detect_ffmpeg", return_value=_FakeTools()),
            patch.object(audio_timeline.subprocess, "run", _ffprobe_stub(pts)),
        ):
            return audio_timeline.analyze_audio_timeline("fake.mp4")

    def test_clean_audio_has_no_holes(self):
        self.assertEqual(self._analyze(_clean_pts(500)), [])

    def test_detects_holes_with_positions_and_sizes(self):
        holes = self._analyze(_holey_pts())
        self.assertEqual(len(holes), 2)
        (pos1, size1), (pos2, size2) = holes
        # 窟窿 1：跳变发生在拼接轴 2.0s 处，跳变 0.52s，净窟窿 0.5s
        self.assertAlmostEqual(pos1, 2.0, places=4)
        self.assertAlmostEqual(size1, 0.5, places=4)
        self.assertAlmostEqual(pos2, 4.5, places=4)
        self.assertAlmostEqual(size2, 1.0, places=4)

    def test_variable_frame_encoding_returns_none(self):
        # 变长帧编码（如 Vorbis）帧距天然分散，不能硬判窟窿
        pts: list[float] = []
        t = 0.0
        for _ in range(300):
            pts.append(t)
            t += 0.01 if len(pts) % 2 else 0.03
        self.assertIsNone(self._analyze(pts))

    def test_too_few_packets_returns_none(self):
        self.assertIsNone(self._analyze(_clean_pts(5)))

    def test_ffprobe_failure_returns_none(self):
        with (
            patch.object(audio_timeline, "detect_ffmpeg", return_value=_FakeTools()),
            patch.object(
                audio_timeline.subprocess, "run", lambda cmd, **kw: _FakeProc(returncode=1)
            ),
        ):
            self.assertIsNone(audio_timeline.analyze_audio_timeline("fake.mp4"))

    def test_missing_ffprobe_returns_none(self):
        missing = _FakeTools()
        missing.ffprobe = None
        with patch.object(audio_timeline, "detect_ffmpeg", return_value=missing):
            self.assertIsNone(audio_timeline.analyze_audio_timeline("fake.mp4"))

    def test_total_hole_seconds(self):
        self.assertEqual(audio_timeline.total_hole_seconds(None), 0.0)
        self.assertEqual(audio_timeline.total_hole_seconds([]), 0.0)
        holes = [(1.0, 0.5), (5.0, 1.25)]
        self.assertEqual(audio_timeline.total_hole_seconds(holes), 1.75)


class ExtractFixedAudioTest(unittest.TestCase):
    def test_success_writes_dest(self):
        def fake_run(cmd, **kwargs):
            # 模拟 ffmpeg 真正落盘
            Path(cmd[cmd.index("-y") - 1]).write_bytes(b"RIFF")
            return _FakeProc()

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "audio_fixed.wav"
            with (
                patch.object(audio_timeline, "detect_ffmpeg", return_value=_FakeTools()),
                patch.object(audio_timeline.subprocess, "run", fake_run),
            ):
                self.assertTrue(
                    audio_timeline.extract_timeline_fixed_audio("in.mp4", dest)
                )
            self.assertTrue(dest.exists())

    def test_ffmpeg_failure_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "audio_fixed.wav"
            with (
                patch.object(audio_timeline, "detect_ffmpeg", return_value=_FakeTools()),
                patch.object(
                    audio_timeline.subprocess, "run", lambda cmd, **kw: _FakeProc(returncode=1)
                ),
            ):
                self.assertFalse(
                    audio_timeline.extract_timeline_fixed_audio("in.mp4", dest)
                )


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class PrepareAudioTimelineTest(unittest.TestCase):
    """JobManager._prepare_audio_timeline 的阈值与回退逻辑。"""

    def _make_manager(self, tmpdir):
        from moss_transcribe_diarize.app.server import create_app

        app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
        return app.state.manager

    def test_holes_below_threshold_pass_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            job, _ = mgr.create_job_for_upload("sample.wav")
            small_holes = [(1.0, 0.05)]
            with (
                patch(
                    "moss_transcribe_diarize.app.jobs.analyze_audio_timeline",
                    return_value=small_holes,
                ),
                patch("moss_transcribe_diarize.app.jobs.extract_timeline_fixed_audio") as extract,
            ):
                result = mgr._prepare_audio_timeline(job)
            self.assertEqual(result, str(job.input_path))
            extract.assert_not_called()

    def test_clean_audio_passes_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            job, _ = mgr.create_job_for_upload("sample.wav")
            with (
                patch(
                    "moss_transcribe_diarize.app.jobs.analyze_audio_timeline", return_value=[]
                ),
                patch("moss_transcribe_diarize.app.jobs.extract_timeline_fixed_audio") as extract,
            ):
                result = mgr._prepare_audio_timeline(job)
            self.assertEqual(result, str(job.input_path))
            extract.assert_not_called()

    def test_holes_above_threshold_use_fixed_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            job, _ = mgr.create_job_for_upload("sample.wav")
            holes = [(2.0, 0.5), (4.5, 0.98)]

            def fake_extract(src, dest):
                Path(dest).write_bytes(b"RIFF")
                return True

            with (
                patch(
                    "moss_transcribe_diarize.app.jobs.analyze_audio_timeline", return_value=holes
                ),
                patch(
                    "moss_transcribe_diarize.app.jobs.extract_timeline_fixed_audio",
                    side_effect=fake_extract,
                ),
            ):
                result = mgr._prepare_audio_timeline(job)
            self.assertEqual(result, str(Path(job.job_dir) / "audio_fixed.wav"))

    def test_extraction_failure_falls_back_to_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            job, _ = mgr.create_job_for_upload("sample.wav")
            with (
                patch(
                    "moss_transcribe_diarize.app.jobs.analyze_audio_timeline",
                    return_value=[(2.0, 0.5)],
                ),
                patch(
                    "moss_transcribe_diarize.app.jobs.extract_timeline_fixed_audio",
                    return_value=False,
                ),
            ):
                result = mgr._prepare_audio_timeline(job)
            self.assertEqual(result, str(job.input_path))


if __name__ == "__main__":
    unittest.main()
