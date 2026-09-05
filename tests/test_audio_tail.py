from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.jobs import _adaptive_tail_endings
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
