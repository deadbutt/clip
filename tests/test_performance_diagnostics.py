import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from moss_transcribe_diarize.app import jobs


def make_record(tmp_path):
    return jobs.JobRecord(
        id="diagnostics", status="waiting_review", media_name="sample",
        input_path=str(tmp_path / "missing.mp4"), job_dir=str(tmp_path),
        inference_prompt="", max_length=8192, max_new_tokens=4096,
        decoding="greedy", temperature=None,
    )


def test_failed_phase_preserves_diagnostics_and_original_exception(tmp_path):
    manager = jobs.JobManager.__new__(jobs.JobManager)
    manager.job_log = Mock()
    record = make_record(tmp_path)
    cuda = SimpleNamespace(is_initialized=lambda: True, memory_allocated=lambda: 3 * 1048576)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=7 * 1048576))
    fake_psutil = SimpleNamespace(Process=lambda: process, Error=RuntimeError)
    with patch.object(jobs, "psutil", fake_psutil), patch.dict(jobs.sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
        with pytest.raises(ValueError, match="original failure"):
            with manager._measure_phase(record, "whisper"):
                raise ValueError("original failure")
    assert record.phase_memory_mb["whisper"] == 7
    assert record.phase_gpu_memory_mb["whisper"] == 3
    restored = jobs.JobRecord.from_dict(record.to_dict())
    assert restored.phase_memory_mb == record.phase_memory_mb
    assert restored.phase_gpu_memory_mb == record.phase_gpu_memory_mb
    assert not any(t.name in {"mtd-memory-whisper", "mtd-gpu-memory-whisper"} for t in jobs.threading.enumerate())


def test_diagnostics_do_not_initialize_cuda_or_require_psutil(tmp_path):
    manager = jobs.JobManager.__new__(jobs.JobManager)
    manager.job_log = Mock()
    record = make_record(tmp_path)
    allocate = Mock(side_effect=AssertionError("uninitialized CUDA must not be sampled"))
    cuda = SimpleNamespace(is_initialized=lambda: False, memory_allocated=allocate)
    with patch.object(jobs, "psutil", None), patch.dict(jobs.sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
        with manager._measure_phase(record, "audio_timeline"):
            pass
    allocate.assert_not_called()
    assert "audio_timeline" in record.phase_times
    assert record.phase_memory_mb == record.phase_gpu_memory_mb == {}


@pytest.mark.parametrize("failure, status", [(RuntimeError("test failure"), "failed"), (jobs.JobCancelled(), "cancelled")])
def test_pipeline_failure_cleans_repaired_audio_but_keeps_checkpoint(tmp_path, failure, status):
    from moss_transcribe_diarize.app.server import create_app

    manager = create_app(model_path="fake-model", runs_dir=tmp_path / "runs", max_new_tokens=8).state.manager
    record, _ = manager.create_job_for_upload("sample.mp4")
    repaired = jobs.Path(record.job_dir) / "audio_fixed.wav"
    repaired.write_bytes(b"temporary audio")
    record.raw_words_path.write_text("[]", encoding="utf-8")
    with (
        patch.object(manager, "_ensure_job_thumbnail"),
        patch.object(manager, "_prepare_audio_timeline", return_value=str(repaired)),
        patch.object(manager, "_load_source_captions", side_effect=failure),
    ):
        manager._process_job(record)
    assert record.status == status
    assert not repaired.exists()
    assert record.raw_words_path.exists()
    assert json.loads(record.job_path.read_text(encoding="utf-8"))["diagnostics"]["temporary_bytes"] == 0
    manager._queue.put(None)
    manager._worker.join(timeout=2)
    manager._close_job_logger(record.id)


def test_audio_extraction_timeout_removes_partial_file(tmp_path):
    from moss_transcribe_diarize.app import speaker_labeler

    def timeout(*args, **kwargs):
        (tmp_path / "pyannote_16k.wav").write_bytes(b"partial")
        raise jobs.subprocess.TimeoutExpired("ffmpeg", 600)

    with (
        patch.object(speaker_labeler, "detect_ffmpeg", return_value=SimpleNamespace(ffmpeg="ffmpeg")),
        patch.object(speaker_labeler.subprocess, "run", side_effect=timeout),
    ):
        with pytest.raises(jobs.subprocess.TimeoutExpired):
            speaker_labeler._extract_audio_file("sample.mp4", work_dir=tmp_path, stem="pyannote_16k")
    assert not (tmp_path / "pyannote_16k.wav").exists()


def test_restart_cleanup_keeps_checkpoints_media_and_recent_audio(tmp_path):
    import os
    import time

    manager = jobs.JobManager.__new__(jobs.JobManager)
    manager.runs_dir = tmp_path / "runs"
    manager.videos_dir = tmp_path / "videos"
    work = manager.runs_dir / "old-job"
    work.mkdir(parents=True)
    manager.videos_dir.mkdir()
    for name in ("audio_fixed.wav", "pyannote_16k.wav", "raw_words.json", "raw_metrics.json", "subtitle.srt"):
        path = work / name
        path.write_bytes(b"data")
        os.utime(path, (time.time() - 7200, time.time() - 7200))
    media = manager.videos_dir / "user-video.mp4"
    media.write_bytes(b"original")
    recent = work / "vocals_16k.wav"
    recent.write_bytes(b"recent")
    assert manager._cleanup_orphaned_temp_files() == 2
    assert recent.exists() and media.exists()
    assert (work / "raw_words.json").exists() and (work / "raw_metrics.json").exists()
    assert (work / "subtitle.srt").exists()
