from __future__ import annotations

import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from moss_transcribe_diarize.subtitle import SubtitleSegment

from .ffmpeg import detect_ffmpeg


@dataclass(slots=True)
class SpeakerLabelingInfo:
    enabled: bool
    applied: bool
    method: str
    speakers: int
    segments: int
    reason: str = ""
    fallback: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "method": self.method,
            "speakers": self.speakers,
            "segments": self.segments,
            "reason": self.reason,
            "fallback": self.fallback,
        }


def label_speakers(
    media_path: str | Path,
    segments: Iterable[SubtitleSegment],
    *,
    work_dir: str | Path,
    enabled: bool = True,
    max_speakers: int = 4,
    target_speakers: int | None = None,
    backend: str = "auto",
    hf_token: str | None = None,
    pyannote_model: str = "pyannote/speaker-diarization-3.1",
    device: str = "auto",
) -> tuple[list[SubtitleSegment], SpeakerLabelingInfo]:
    prepared = [segment for segment in segments if segment.text and segment.end > segment.start]
    backend = (backend or "auto").lower()
    if backend in {"off", "none", "disabled"}:
        enabled = False
    if not enabled:
        return prepared, SpeakerLabelingInfo(False, False, backend, 0, len(prepared), "disabled")
    if len(prepared) < 4:
        return prepared, SpeakerLabelingInfo(True, False, backend, 1, len(prepared), "too few segments")
    existing = {segment.speaker for segment in prepared if segment.speaker}
    if len(existing) > 1:
        return prepared, SpeakerLabelingInfo(True, False, backend, len(existing), len(prepared), "speaker labels already exist")

    if backend in {"auto", "pyannote"}:
        try:
            return _label_speakers_pyannote(
                media_path,
                prepared,
                work_dir=work_dir,
                target_speakers=target_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                model_name=pyannote_model,
                device=device,
            )
        except Exception as exc:
            if backend == "pyannote":
                return prepared, SpeakerLabelingInfo(True, False, "pyannote", 1, len(prepared), str(exc))
            cluster_segments, cluster_info = _label_speakers_cluster(
                media_path,
                prepared,
                work_dir=work_dir,
                max_speakers=max_speakers,
                target_speakers=target_speakers,
            )
            cluster_info.fallback = f"pyannote failed: {exc}"
            return cluster_segments, cluster_info

    if backend != "cluster":
        return prepared, SpeakerLabelingInfo(True, False, backend, 1, len(prepared), f"unsupported backend: {backend}")

    return _label_speakers_cluster(
        media_path,
        prepared,
        work_dir=work_dir,
        max_speakers=max_speakers,
        target_speakers=target_speakers,
    )


def _label_speakers_cluster(
    media_path: str | Path,
    prepared: list[SubtitleSegment],
    *,
    work_dir: str | Path,
    max_speakers: int,
    target_speakers: int | None,
) -> tuple[list[SubtitleSegment], SpeakerLabelingInfo]:
    try:
        sample_rate, audio = _extract_audio(media_path, work_dir=work_dir)
        features, usable_indices = _segment_features(audio, sample_rate, prepared)
        if len(usable_indices) < 4:
            return prepared, SpeakerLabelingInfo(True, False, "audio_cluster", 1, len(prepared), "not enough voiced audio")
        labels = _cluster_features(features, max_speakers=max_speakers, target_speakers=target_speakers)
        if len(set(labels)) <= 1:
            return _with_single_speaker(prepared), SpeakerLabelingInfo(True, True, "audio_cluster", 1, len(prepared), "single speaker estimated")

        output = _apply_cluster_labels(prepared, usable_indices, labels)
        speaker_count = len({segment.speaker for segment in output})
        return output, SpeakerLabelingInfo(True, True, "audio_cluster", speaker_count, len(output))
    except Exception as exc:
        return prepared, SpeakerLabelingInfo(True, False, "audio_cluster", 1, len(prepared), str(exc))


def _label_speakers_pyannote(
    media_path: str | Path,
    prepared: list[SubtitleSegment],
    *,
    work_dir: str | Path,
    target_speakers: int | None,
    max_speakers: int,
    hf_token: str | None,
    model_name: str,
    device: str,
) -> tuple[list[SubtitleSegment], SpeakerLabelingInfo]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is not installed. Install the diarization extra first.") from exc

    audio_path = _extract_audio_file(media_path, work_dir=work_dir, stem="pyannote_16k")
    try:
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        resolved_device = _resolve_torch_device(device)
        if resolved_device is not None:
            pipeline.to(resolved_device)
        options: dict[str, int] = {}
        if target_speakers and target_speakers > 0:
            options["num_speakers"] = int(target_speakers)
        elif max_speakers and max_speakers > 1:
            options["max_speakers"] = int(max_speakers)
        diarization = pipeline(str(audio_path), **options)
        turns = _pyannote_turns(diarization)
        if not turns:
            return prepared, SpeakerLabelingInfo(True, False, "pyannote", 1, len(prepared), "no speaker turns returned")
        output = _assign_turns_to_segments(prepared, turns)
        speaker_count = len({segment.speaker for segment in output})
        return output, SpeakerLabelingInfo(True, True, "pyannote", speaker_count, len(output))
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass


def _extract_audio(media_path: str | Path, *, work_dir: str | Path) -> tuple[int, np.ndarray]:
    wav_path = _extract_audio_file(media_path, work_dir=work_dir, stem="speaker_labeler_16k")
    try:
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            frames = handle.readframes(handle.getnframes())
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass
    if width != 2:
        raise RuntimeError(f"unsupported wav sample width: {width}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def _extract_audio_file(media_path: str | Path, *, work_dir: str | Path, stem: str) -> Path:
    tools = detect_ffmpeg()
    if not tools.ffmpeg:
        raise RuntimeError("ffmpeg is not available")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"{stem}.wav"
    command = [
        tools.ffmpeg,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-8:])
        raise RuntimeError(f"audio extraction failed: {tail}")
    return wav_path


def _segment_features(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SubtitleSegment],
) -> tuple[np.ndarray, list[int]]:
    features: list[np.ndarray] = []
    indices: list[int] = []
    for index, segment in enumerate(segments):
        start = max(0, int((segment.start + 0.04) * sample_rate))
        end = min(len(audio), int((segment.end - 0.04) * sample_rate))
        if end - start < int(0.28 * sample_rate):
            continue
        feature = _audio_feature(audio[start:end], sample_rate)
        if feature is None:
            continue
        features.append(feature)
        indices.append(index)
    if not features:
        return np.empty((0, 0), dtype=np.float32), []
    return np.vstack(features).astype(np.float32), indices


def _audio_feature(chunk: np.ndarray, sample_rate: int) -> np.ndarray | None:
    chunk = chunk.astype(np.float32, copy=False)
    chunk = chunk - float(np.mean(chunk))
    if float(np.sqrt(np.mean(chunk * chunk))) < 0.0025:
        return None

    frame_size = max(256, int(sample_rate * 0.025))
    hop = max(128, int(sample_rate * 0.010))
    if len(chunk) < frame_size:
        return None
    frame_count = 1 + (len(chunk) - frame_size) // hop
    frames = np.lib.stride_tricks.as_strided(
        chunk,
        shape=(frame_count, frame_size),
        strides=(chunk.strides[0] * hop, chunk.strides[0]),
        writeable=False,
    ).copy()
    window = np.hanning(frame_size).astype(np.float32)
    frames *= window
    energy = np.mean(frames * frames, axis=1)
    if float(np.max(energy)) <= 0:
        return None
    voiced = energy >= np.quantile(energy, 0.45)
    frames = frames[voiced]
    energy = energy[voiced]
    if len(frames) < 3:
        return None

    n_fft = 512
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    spectrum += 1e-10
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    total_power = np.sum(spectrum, axis=1)
    centroid = np.sum(spectrum * freqs[None, :], axis=1) / total_power
    bandwidth = np.sqrt(np.sum(spectrum * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / total_power)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    bands = [
        (80, 250),
        (250, 700),
        (700, 1600),
        (1600, 3400),
        (3400, 7600),
    ]
    band_ratios = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_ratios.append(np.mean(np.sum(spectrum[:, mask], axis=1) / total_power))

    f0_values = [_estimate_pitch(frame, sample_rate) for frame in frames[:80]]
    voiced_f0 = np.array([value for value in f0_values if value > 0], dtype=np.float32)
    if len(voiced_f0):
        log_f0 = float(np.mean(np.log(voiced_f0)))
        f0_spread = float(np.std(np.log(voiced_f0)))
        voiced_ratio = float(len(voiced_f0) / len(f0_values))
    else:
        log_f0 = 0.0
        f0_spread = 0.0
        voiced_ratio = 0.0

    values = [
        float(np.mean(np.log1p(energy))),
        float(np.std(np.log1p(energy))),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.std(bandwidth)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        log_f0,
        f0_spread,
        voiced_ratio,
        *[float(value) for value in band_ratios],
    ]
    return np.array(values, dtype=np.float32)


def _estimate_pitch(frame: np.ndarray, sample_rate: int) -> float:
    frame = frame.astype(np.float32, copy=False)
    frame = frame - float(np.mean(frame))
    if float(np.sqrt(np.mean(frame * frame))) < 0.003:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if corr[0] <= 0:
        return 0.0
    min_lag = max(1, int(sample_rate / 450))
    max_lag = min(len(corr) - 1, int(sample_rate / 75))
    if max_lag <= min_lag:
        return 0.0
    region = corr[min_lag:max_lag]
    lag = int(np.argmax(region)) + min_lag
    confidence = float(corr[lag] / corr[0])
    if confidence < 0.25:
        return 0.0
    return float(sample_rate / lag)


def _cluster_features(features: np.ndarray, *, max_speakers: int, target_speakers: int | None = None) -> list[int]:
    if len(features) < 4:
        return [0 for _ in range(len(features))]
    normalized = _standardize(features)
    if target_speakers is not None and target_speakers > 1:
        k = min(int(target_speakers), len(features), 8)
        return _kmeans(normalized, k)
    best_labels = [0 for _ in range(len(features))]
    best_score = -1.0
    upper = min(max(1, int(max_speakers)), len(features), 6)
    for k in range(2, upper + 1):
        labels = _kmeans(normalized, k)
        score = _silhouette_score(normalized, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_score < 0.12:
        return [0 for _ in range(len(features))]
    return best_labels


def _resolve_torch_device(device: str):
    device = (device or "auto").lower()
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return None
    try:
        import torch

        return torch.device(device)
    except Exception:
        return None


def _pyannote_turns(diarization) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(getattr(turn, "start", 0.0))
        end = float(getattr(turn, "end", start))
        if end > start:
            turns.append((start, end, str(speaker)))
    turns.sort(key=lambda item: (item[0], item[1], item[2]))
    return turns


def _assign_turns_to_segments(
    segments: list[SubtitleSegment],
    turns: list[tuple[float, float, str]],
) -> list[SubtitleSegment]:
    speaker_names: dict[str, str] = {}
    output: list[SubtitleSegment] = []
    previous_source = ""
    for segment in segments:
        source_speaker = _best_turn_speaker(segment, turns) or previous_source or (turns[0][2] if turns else "SPEAKER_00")
        previous_source = source_speaker
        if source_speaker not in speaker_names:
            speaker_names[source_speaker] = f"S{len(speaker_names) + 1:02d}"
        output.append(
            SubtitleSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=speaker_names[source_speaker],
                text=segment.text,
            )
        )
    return output


def _best_turn_speaker(segment: SubtitleSegment, turns: list[tuple[float, float, str]]) -> str:
    scores: dict[str, float] = {}
    midpoint = (segment.start + segment.end) / 2.0
    nearest: tuple[float, str] | None = None
    for start, end, speaker in turns:
        overlap = max(0.0, min(segment.end, end) - max(segment.start, start))
        if overlap > 0:
            scores[speaker] = scores.get(speaker, 0.0) + overlap
        distance = 0.0 if start <= midpoint <= end else min(abs(midpoint - start), abs(midpoint - end))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, speaker)
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]
    return nearest[1] if nearest is not None else ""


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / np.where(std < 1e-6, 1.0, std)


def _kmeans(features: np.ndarray, k: int, *, iterations: int = 60) -> list[int]:
    centers = _initial_centers(features, k)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = np.mean(members, axis=0)
    return [int(label) for label in labels]


def _initial_centers(features: np.ndarray, k: int) -> np.ndarray:
    centers = [features[0]]
    while len(centers) < k:
        distances = np.min(
            np.linalg.norm(features[:, None, :] - np.vstack(centers)[None, :, :], axis=2),
            axis=1,
        )
        centers.append(features[int(np.argmax(distances))])
    return np.vstack(centers).astype(np.float32)


def _silhouette_score(features: np.ndarray, labels: list[int]) -> float:
    labels_array = np.array(labels)
    clusters = sorted(set(labels))
    if len(clusters) <= 1:
        return -1.0
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    scores = []
    for index, label in enumerate(labels_array):
        same = labels_array == label
        same[index] = False
        a = float(np.mean(distances[index, same])) if np.any(same) else 0.0
        b = math.inf
        for other in clusters:
            if other == label:
                continue
            other_mask = labels_array == other
            if np.any(other_mask):
                b = min(b, float(np.mean(distances[index, other_mask])))
        if not math.isfinite(b):
            continue
        scores.append((b - a) / max(a, b, 1e-6))
    return float(np.mean(scores)) if scores else -1.0


def _apply_cluster_labels(
    segments: list[SubtitleSegment],
    usable_indices: list[int],
    labels: list[int],
) -> list[SubtitleSegment]:
    assigned: list[int | None] = [None for _ in segments]
    for index, label in zip(usable_indices, labels):
        assigned[index] = int(label)
    for index, label in enumerate(assigned):
        if label is not None:
            continue
        assigned[index] = _nearest_label(index, assigned)
    assigned = _smooth_short_islands(segments, assigned)
    label_names: dict[int, str] = {}
    next_id = 1
    output: list[SubtitleSegment] = []
    for segment, label in zip(segments, assigned):
        label = int(label or 0)
        if label not in label_names:
            label_names[label] = f"S{next_id:02d}"
            next_id += 1
        output.append(
            SubtitleSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=label_names[label],
                text=segment.text,
            )
        )
    return output


def _nearest_label(index: int, assigned: list[int | None]) -> int:
    for distance in range(1, len(assigned) + 1):
        left = index - distance
        right = index + distance
        if left >= 0 and assigned[left] is not None:
            return int(assigned[left] or 0)
        if right < len(assigned) and assigned[right] is not None:
            return int(assigned[right] or 0)
    return 0


def _smooth_short_islands(segments: list[SubtitleSegment], labels: list[int | None]) -> list[int | None]:
    smoothed = list(labels)
    for index in range(1, len(smoothed) - 1):
        if smoothed[index - 1] == smoothed[index + 1] != smoothed[index]:
            duration = segments[index].end - segments[index].start
            if duration < 1.3:
                smoothed[index] = smoothed[index - 1]
    return smoothed


def _with_single_speaker(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            id=segment.id,
            start=segment.start,
            end=segment.end,
            speaker="S01",
            text=segment.text,
        )
        for segment in segments
    ]
