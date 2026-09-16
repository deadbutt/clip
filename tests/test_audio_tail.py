from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.jobs import JobManager, JobRecord, _adaptive_tail_endings
from moss_transcribe_diarize.subtitle import SubtitleSegment


def _segments(*pairs: tuple[float, float]) -> list[SubtitleSegment]:
    return [SubtitleSegment(id=f"s{index}", start=start, end=end, speaker="S00", text="x") for index, (start, end) in enumerate(pairs)]


def test_adaptive_tail_never_shortens_fixed_fallback() -> None:
    segments = _segments((0.0, 0.10))
    # Energy ends immediately at the ASR boundary; the established 0.5 s tail remains.
    with patch("moss_transcribe_diarize.app.jobs._rms_frames_from_audio", return_value=[0.10] * 5 + [0.0] * 30):
        result = _adaptive_tail_endings(Path("missing.mp4"), segments)
    assert result is not None
    assert result["s0"] == 0.60


def test_adaptive_tail_extends_continuous_energy_and_caps_at_next_segment() -> None:
    segments = _segments((0.0, 0.10), (0.35, 0.60))
    with patch("moss_transcribe_diarize.app.jobs._rms_frames_from_audio", return_value=[0.10] * 30):
        result = _adaptive_tail_endings(Path("missing.mp4"), segments)
    assert result is not None
    assert result["s0"] == 0.35


def test_adaptive_tail_does_not_extend_quiet_audio_beyond_fallback() -> None:
    segments = _segments((0.0, 0.10))
    with patch("moss_transcribe_diarize.app.jobs._rms_frames_from_audio", return_value=[0.01] * 40):
        result = _adaptive_tail_endings(Path("missing.mp4"), segments)
    assert result is not None
    assert result["s0"] == 0.60


class _Runner:
    model_path = "test"


def test_media_audio_analysis_is_reused_when_segments_change(tmp_path: Path) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"audio")
    manager = JobManager(tmp_path / "runs", _Runner(), prompt="", max_length=64, max_new_tokens=8)
    job_dir = tmp_path / "runs" / "job1"
    job_dir.mkdir(parents=True)
    job = JobRecord(
        id="job1",
        status="waiting_review",
        progress=0.95,
        media_name="input.wav",
        input_path=str(media),
        job_dir=str(job_dir),
        inference_prompt="",
        max_length=64,
        max_new_tokens=8,
        decoding="greedy",
        temperature=None,
    )
    manager._jobs[job.id] = job
    try:
        with patch("moss_transcribe_diarize.app.jobs._rms_frames_from_audio", return_value=[0.1] * 100) as decode:
            manager._adaptive_end_map(job, _segments((0.0, 0.5), (0.8, 1.2)))
            manager._tail_padding_cache.pop(job.id, None)
            manager._adaptive_end_map(job, _segments((0.0, 0.4), (0.4, 0.8), (0.8, 1.2)))
        assert decode.call_count == 1
        with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1280, 720)) as probe:
            assert manager._video_size_for_job(job) == (1280, 720)
            assert manager._video_size_for_job(job) == (1280, 720)
        assert probe.call_count == 1
    finally:
        manager.shutdown()
