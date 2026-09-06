"""Clip candidate selection and export operations for :class:`JobManager`.

Keeping these operations in a mixin gives the job pipeline a smaller public
surface while preserving the existing manager API.  The mixin deliberately
uses ``self`` for storage, logging, and cancellation hooks so the split does
not introduce a second job state machine.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.subtitle import SubtitleSegment, SubtitleStyle, export_ass, export_srt, write_text

from .clips import generate_clip_candidates, rebase_segments_for_clip
from .ffmpeg import burn_ass_subtitles_clip, detect_ffmpeg, probe_video_size

_CLIP_CANDIDATES_CACHE_MAX = 24
_CLIP_TAIL_PADDING = 0.50
_CLIP_EXPORT_CACHE_VERSION = 1


def _padded_export_segments(
    segments: list[SubtitleSegment],
    padding: float = _CLIP_TAIL_PADDING,
    end_overrides: dict[str, float] | None = None,
) -> list[SubtitleSegment]:
    """Apply sentence-tail buffers while keeping segment source times untouched."""
    out: list[SubtitleSegment] = []
    for index, segment in enumerate(segments):
        next_start = segments[index + 1].start if index + 1 < len(segments) else segment.end + padding
        requested_end = (end_overrides or {}).get(segment.id, segment.end + padding)
        end = min(next_start, max(segment.end, requested_end))
        out.append(replace(segment, end=max(segment.start + 0.01, end)))
    return out


def _safe_clip_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    safe = safe.strip("._")
    return safe[:80] or f"clip_{uuid.uuid4().hex[:8]}"


def _file_signature(path: str | Any) -> dict[str, int] | None:
    """Return the cheap on-disk identity used by the clip export cache."""
    try:
        stat = Path(path).stat()
    except (OSError, TypeError, ValueError):
        return None
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _clip_export_cache_key(
    *,
    job: Any,
    clip_id: str,
    start: float,
    end: float,
    style: SubtitleStyle,
    segments_version: tuple[int, int] | None,
) -> str:
    """Build a stable key for one rendered clip.

    The key deliberately includes every input that can change the generated
    files.  It is stored in the metadata next to the output, so the cache
    survives an application restart while stale files are safely ignored.
    """
    source_path = Path(job.input_path)
    payload = {
        "version": _CLIP_EXPORT_CACHE_VERSION,
        "source_path": str(source_path.resolve()),
        "source_file": _file_signature(source_path),
        "segments_version": list(segments_version) if segments_version is not None else None,
        "clip_id": clip_id,
        "start": float(start),
        "end": float(end),
        "style": style.to_dict(),
        "tail_padding": _CLIP_TAIL_PADDING,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"v{_CLIP_EXPORT_CACHE_VERSION}-{hashlib.sha256(encoded).hexdigest()}"


def _has_complete_clip_cache(
    *,
    output_path: Any,
    srt_path: Any,
    ass_path: Any,
    metadata_path: Any,
    cache_key: str,
) -> dict[str, Any] | None:
    """Load metadata only when all four export artifacts are complete."""
    paths = (output_path, srt_path, ass_path, metadata_path)
    if not all(Path(path).is_file() for path in paths):
        return None
    try:
        # A zero-byte MP4 is a common remnant of an interrupted ffmpeg run.
        if Path(output_path).stat().st_size <= 0:
            return None
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("cache_key") != cache_key:
        return None
    return metadata


class ClipOperationsMixin:
    """Candidate lookup, caching, and clip rendering methods for JobManager."""

    def list_clip_candidates(
        self,
        job_id: str,
        *,
        min_duration: float = 45.0,
        target_duration: float = 120.0,
        max_duration: float | None = 300.0,
        limit: int = 24,
        merge_expansion_limit: float | None = None,
        selector: Any | None = None,
        include_alternates: bool = False,
    ) -> list[dict[str, Any]]:
        segment_data, segments_version = self._load_segments(job_id)
        segments = [SubtitleSegment.from_dict(item) for item in segment_data]
        cache_key = (
            job_id,
            segments_version,
            float(min_duration),
            float(target_duration),
            None if max_duration is None else float(max_duration),
            int(limit),
            None if merge_expansion_limit is None else float(merge_expansion_limit),
            bool(include_alternates),
        )
        if selector is None and segments_version is not None:
            with self._clip_candidates_cache_lock:
                cached = self._clip_candidates_cache.get(cache_key)
                if cached is not None:
                    self._clip_candidates_cache[cache_key] = self._clip_candidates_cache.pop(cache_key)
                    return [dict(item) for item in cached]
        rule_limit = max(limit, min(48, limit * 4)) if selector is not None else limit
        candidates = [
            candidate.to_dict()
            for candidate in generate_clip_candidates(
                segments,
                min_duration=min_duration,
                target_duration=target_duration,
                max_duration=max_duration,
                limit=rule_limit,
                merge_expansion_limit=merge_expansion_limit,
                include_alternates=include_alternates and selector is None,
                alternate_limit=limit if selector is None else None,
                adaptive_limit=selector is None,
            )
        ]
        if selector is not None:
            return selector.rank_clip_candidates(candidates, limit=limit)
        if segments_version is not None:
            with self._clip_candidates_cache_lock:
                self._clip_candidates_cache[cache_key] = [dict(item) for item in candidates]
                while len(self._clip_candidates_cache) > _CLIP_CANDIDATES_CACHE_MAX:
                    self._clip_candidates_cache.pop(next(iter(self._clip_candidates_cache)), None)
        return candidates

    def render_clip(
        self,
        job_id: str,
        *,
        start: float,
        end: float,
        style_payload: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        start = max(0.0, float(start))
        end = max(start + 0.25, float(end))
        style = SubtitleStyle.from_dict(style_payload)
        clip_id = _safe_clip_name(name or f"clip_{start:.2f}_{end:.2f}")
        clip_key = f"{job.id}/{clip_id}"
        with self._rendering_clips_lock:
            if clip_key in self._rendering_clips:
                raise RuntimeError("该片段正在渲染中,请等当前渲染完成后再试。")
            self._rendering_clips.add(clip_key)
        try:
            job.clips_dir.mkdir(parents=True, exist_ok=True)
            ass_path = job.clips_dir / f"{clip_id}.ass"
            srt_path = job.clips_dir / f"{clip_id}.srt"
            metadata_path = job.clips_dir / f"{clip_id}.json"
            output_path = job.clips_dir / f"{clip_id}.mp4"
            _segment_data, segments_version = self._load_segments(job_id)
            cache_key = _clip_export_cache_key(
                job=job,
                clip_id=clip_id,
                start=start,
                end=end,
                style=style,
                segments_version=segments_version,
            )
            cached_metadata = _has_complete_clip_cache(
                output_path=output_path,
                srt_path=srt_path,
                ass_path=ass_path,
                metadata_path=metadata_path,
                cache_key=cache_key,
            )
            if cached_metadata is not None:
                cached_segments = cached_metadata.get("segments")
                return {
                    "filename": output_path.name,
                    "path": str(output_path),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "segments": len(cached_segments) if isinstance(cached_segments, list) else 0,
                    "cached": True,
                    "files": {
                        "mp4": output_path.name,
                        "srt": srt_path.name,
                        "ass": ass_path.name,
                        "metadata": metadata_path.name,
                    },
                }
            if not detect_ffmpeg().available:
                raise RuntimeError("ffmpeg and ffprobe are not available on PATH.")
            segments = self._clip_segments(job, start=start, end=end)
            if not segments:
                raise RuntimeError("The selected range does not contain any subtitle segments.")
            width, height = probe_video_size(job.input_path)
            export_segments = _padded_export_segments(segments)
            write_text(ass_path, export_ass(export_segments, style=style, video_width=width, video_height=height), encoding="utf-8-sig")
            write_text(
                srt_path,
                export_srt(export_segments, show_speaker=style.show_speaker, speaker_names=style.speaker_names),
                encoding="utf-8-sig",
            )
            burn_ass_subtitles_clip(
                job.input_path,
                ass_path,
                output_path,
                start=start,
                end=end,
                style=style,
                cancel_check=lambda: job.id not in self._jobs,
            )
            # Write the cache marker only after ffmpeg succeeds.  If encoding
            # fails while an older MP4 is still present, that old file must not
            # become falsely associated with the new metadata key.
            write_text(
                metadata_path,
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "cache_version": _CLIP_EXPORT_CACHE_VERSION,
                        "source_media": job.media_name,
                        "source_start": start,
                        "source_end": end,
                        "duration": end - start,
                        "clip_timeline_start": 0.0,
                        "clip_timeline_end": end - start,
                        "segments": [segment.to_dict() for segment in segments],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        finally:
            self._rendering_clips.discard(clip_key)
        return {
            "filename": output_path.name,
            "path": str(output_path),
            "start": start,
            "end": end,
            "duration": end - start,
            "segments": len(segments),
            "cached": False,
            "files": {
                "mp4": output_path.name,
                "srt": srt_path.name,
                "ass": ass_path.name,
                "metadata": metadata_path.name,
            },
        }

    def _clip_segments(self, job: Any, *, start: float, end: float) -> list[SubtitleSegment]:
        source = [SubtitleSegment.from_dict(item) for item in self.list_segments(job.id)]
        return rebase_segments_for_clip(source, start=start, end=end)
