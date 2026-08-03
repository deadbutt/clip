from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    SubtitleStyle,
    coerce_subtitle_segments,
    export_ass,
    export_json,
    export_srt,
    parse_ass,
    parse_srt,
    subtitle_segments_from_transcript,
    write_text,
)

from .clips import generate_clip_candidates, rebase_segments_for_clip
from .ffmpeg import burn_ass_subtitles, burn_ass_subtitles_clip, detect_ffmpeg, probe_media_duration, probe_video_size
from .speaker_labeler import label_speakers
from .text_translator import apply_translations, collect_pretranslation_skips, validate_translation_outputs


RUNNING_STATES = {"queued", "loading_model", "transcribing", "postprocessing", "labeling_speakers", "translating", "rendering"}
TERMINAL_STATES = {"waiting_review", "done", "failed", "cancelled"}


class JobCancelled(RuntimeError):
    pass

# Audio consumes ~12.5 prompt tokens/sec (Whisper 30s -> 375 tokens after 4x merge).
# Transcript output is ~10 generated tokens/sec of speech; use 14 with a safety margin
# so a single pass is unlikely to hit the limit and force a costly re-run.
_AUDIO_TOKENS_PER_SECOND = 12.5
_OUTPUT_TOKENS_PER_SECOND = 14.0
_MIN_MAX_NEW_TOKENS = 2048
_MAX_MAX_NEW_TOKENS = 65536
_TOKEN_ROUNDING = 512


def recommend_max_new_tokens(
    duration_sec: float | None,
    *,
    max_length: int = 131072,
) -> int | None:
    """Suggest a non-truncating ``max_new_tokens`` for an audio of the given duration.

    Returns ``None`` when the duration is unknown, leaving any caller-provided
    value untouched. The result is clamped to a safe range and rounded so that
    ``prompt + output`` stays within ``max_length``.
    """
    if duration_sec is None or duration_sec <= 0:
        return None
    raw = duration_sec * _OUTPUT_TOKENS_PER_SECOND
    recommended = max(_MIN_MAX_NEW_TOKENS, int((raw + _TOKEN_ROUNDING - 1) // _TOKEN_ROUNDING) * _TOKEN_ROUNDING)
    # Reserve room for the audio prompt so prompt + output fits the context window.
    prompt_estimate = int(duration_sec * _AUDIO_TOKENS_PER_SECOND)
    context_ceiling = max(_MIN_MAX_NEW_TOKENS, max_length - prompt_estimate - _TOKEN_ROUNDING)
    recommended = min(recommended, context_ceiling, _MAX_MAX_NEW_TOKENS)
    return max(_MIN_MAX_NEW_TOKENS, recommended)


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    media_name: str
    input_path: str
    job_dir: str
    inference_prompt: str
    max_length: int
    max_new_tokens: int
    decoding: str
    temperature: float | None
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str | None = None
    prompt_len: int | None = None
    generated_tokens: int | None = None
    elapsed_sec: float | None = None
    subtitle_style: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    speaker_labeling: dict[str, Any] = field(default_factory=dict)
    speaker_count: int | None = None
    diarization_backend: str = "none"
    translation_info: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_transcript_path(self) -> Path:
        return Path(self.job_dir) / "raw_transcript.txt"

    @property
    def segments_path(self) -> Path:
        return Path(self.job_dir) / "segments.json"

    @property
    def source_segments_path(self) -> Path:
        return Path(self.job_dir) / "segments.source.json"

    @property
    def srt_path(self) -> Path:
        return Path(self.job_dir) / "subtitle.srt"

    @property
    def ass_path(self) -> Path:
        return Path(self.job_dir) / "subtitle.ass"

    @property
    def output_path(self) -> Path:
        return Path(self.job_dir) / "output.mp4"

    @property
    def clips_dir(self) -> Path:
        return Path(self.job_dir) / "clips"

    @property
    def job_path(self) -> Path:
        return Path(self.job_dir) / "job.json"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        possibly_truncated = (
            self.generated_tokens is not None
            and self.max_new_tokens > 0
            and self.generated_tokens >= self.max_new_tokens
        )
        data["inference"] = {
            "prompt": self.inference_prompt,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "decoding": self.decoding,
            "temperature": self.temperature,
        }
        data["usage"] = {
            "prompt_tokens": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "possibly_truncated": possibly_truncated,
            "elapsed_sec": self.elapsed_sec,
        }
        data["files"] = {
            "raw_transcript": str(self.raw_transcript_path),
            "segments": str(self.segments_path),
            "srt": str(self.srt_path),
            "ass": str(self.ass_path),
            "mp4": str(self.output_path),
            "clips": str(self.clips_dir),
        }
        data["backend"] = self.backend
        data["translation"] = {
            **self.translation_info,
            "source_available": self.source_segments_path.exists(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        inference = data.get("inference") or {}
        temperature = data.get("temperature", inference.get("temperature"))
        return cls(
            id=str(data["id"]),
            status=str(data.get("status") or "failed"),
            media_name=str(data.get("media_name") or "input.media"),
            input_path=str(data.get("input_path") or ""),
            job_dir=str(data.get("job_dir") or ""),
            inference_prompt=str(data.get("inference_prompt") or inference.get("prompt") or ""),
            max_length=int(data.get("max_length") or inference.get("max_length") or 0),
            max_new_tokens=int(data.get("max_new_tokens") or inference.get("max_new_tokens") or 0),
            decoding=str(data.get("decoding") or inference.get("decoding") or "greedy"),
            temperature=None if temperature is None else float(temperature),
            progress=float(data.get("progress") or 0.0),
            error=data.get("error"),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            model=data.get("model"),
            prompt_len=data.get("prompt_len"),
            generated_tokens=data.get("generated_tokens"),
            elapsed_sec=data.get("elapsed_sec"),
            subtitle_style=dict(data.get("subtitle_style") or {}),
            backend=str(data.get("backend") or ""),
            speaker_labeling=dict(data.get("speaker_labeling") or {}),
            speaker_count=None if data.get("speaker_count") in ("", None) else int(data.get("speaker_count")),
            diarization_backend=str(data.get("diarization_backend") or "auto"),
            translation_info=dict(data.get("translation_info") or data.get("translation") or {}),
        )


class JobManager:
    def __init__(
        self,
        runs_dir: str | Path,
        model_runner: Any,
        *,
        prompt: str,
        max_length: int,
        max_new_tokens: int,
        decoding: str = "greedy",
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str = "none",
        hf_token: str | None = None,
        pyannote_model: str = "pyannote/speaker-diarization-3.1",
        diarization_device: str = "auto",
    ):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.model_runner = model_runner
        self.prompt = prompt
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.decoding = decoding
        self.temperature = temperature
        self.speaker_count = self._resolve_speaker_count(speaker_count)
        self.diarization_backend = diarization_backend
        self.hf_token = hf_token
        self.pyannote_model = pyannote_model
        self.diarization_device = diarization_device
        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._render_lock = threading.Lock()
        self._translate_lock = threading.Lock()
        self._progress_save_times: dict[str, float] = {}
        self._cancelled_jobs: set[str] = set()
        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="mtd-job-worker", daemon=True)
        self._worker.start()

    def create_job_from_file(
        self,
        source_path: str | Path,
        media_name: str | None = None,
        *,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
    ) -> JobRecord:
        options = self._resolve_inference_options(
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
        )
        job_id = uuid.uuid4().hex[:12]
        source_path = Path(source_path)
        suffix = source_path.suffix or ".media"
        job_dir = self.runs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        shutil.copyfile(source_path, input_path)
        job = JobRecord(
            id=job_id,
            status="queued",
            progress=0.0,
            media_name=media_name or source_path.name,
            input_path=str(input_path),
            job_dir=str(job_dir),
            inference_prompt=options["prompt"],
            max_length=options["max_length"],
            max_new_tokens=options["max_new_tokens"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            model=self.model_runner.model_path,
            backend=self._runner_backend(),
            speaker_count=self._resolve_speaker_count(speaker_count),
            diarization_backend=self._resolve_diarization_backend(diarization_backend),
        )
        self._jobs[job.id] = job
        self._save_job(job)
        self._queue.put(job.id)
        return job

    def create_job_for_upload(
        self,
        filename: str,
        *,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
    ) -> tuple[JobRecord, Path]:
        options = self._resolve_inference_options(
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
        )
        job_id = uuid.uuid4().hex[:12]
        suffix = Path(filename).suffix or ".media"
        job_dir = self.runs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        job = JobRecord(
            id=job_id,
            status="queued",
            progress=0.0,
            media_name=filename,
            input_path=str(input_path),
            job_dir=str(job_dir),
            inference_prompt=options["prompt"],
            max_length=options["max_length"],
            max_new_tokens=options["max_new_tokens"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            model=self.model_runner.model_path,
            backend=self._runner_backend(),
            speaker_count=self._resolve_speaker_count(speaker_count),
            diarization_backend=self._resolve_diarization_backend(diarization_backend),
        )
        self._jobs[job.id] = job
        self._save_job(job)
        return job, input_path

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def rerun_job(
        self,
        job_id: str,
        *,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
    ) -> JobRecord:
        source = self.get_job(job_id)
        input_path = Path(source.input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        return self.create_job_from_file(
            input_path,
            media_name=source.media_name,
            prompt=source.inference_prompt if prompt is None else prompt,
            max_length=source.max_length if max_length is None else max_length,
            max_new_tokens=source.max_new_tokens if max_new_tokens is None else max_new_tokens,
            decoding=source.decoding if decoding is None else decoding,
            temperature=source.temperature if temperature is None else temperature,
            speaker_count=source.speaker_count if speaker_count is None else speaker_count,
            diarization_backend=source.diarization_backend if diarization_backend is None else diarization_backend,
        )

    def list_jobs(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown job: {job_id}") from exc

    def delete_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            self._cancelled_jobs.add(job_id)
        self._jobs.pop(job_id, None)
        shutil.rmtree(job.job_dir, ignore_errors=True)

    def list_segments(self, job_id: str) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        self._maybe_sync_segments_from_subtitle_files(job)
        if not job.segments_path.exists():
            return []
        return json.loads(job.segments_path.read_text(encoding="utf-8"))

    def update_segments(
        self,
        job_id: str,
        payload: list[dict[str, Any]],
        style_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        segments = coerce_subtitle_segments(payload)
        style = SubtitleStyle.from_dict(style_payload) if style_payload is not None else None
        if style is not None:
            job.subtitle_style = style.to_dict()
        self._write_subtitle_files(job, segments, style=style)
        if job.status == "done":
            self._set_status(job, "waiting_review", 0.95, error=None)
        else:
            self._touch(job, error=None)
        return [segment.to_dict() for segment in segments]

    def sync_segments_from_subtitle_files(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        result = self._maybe_sync_segments_from_subtitle_files(job, force=True)
        segments = self.list_segments(job_id)
        return {
            "synced": bool(result),
            "source": result["source"] if result else None,
            "count": len(segments),
            "segments": segments,
        }

    def render(self, job_id: str, style_payload: dict[str, Any] | None = None) -> JobRecord:
        job = self.get_job(job_id)
        if not detect_ffmpeg().available:
            raise RuntimeError("ffmpeg and ffprobe are not available on PATH.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")
        if job.status == "rendering":
            return job
        if job.status in RUNNING_STATES - {"rendering"}:
            raise RuntimeError("Cannot render before transcription is ready.")
        self._set_status(job, "rendering", 0.95, error=None)
        threading.Thread(
            target=self._render_job,
            args=(job.id, SubtitleStyle.from_dict(style_payload)),
            name=f"mtd-render-{job.id}",
            daemon=True,
        ).start()
        return job

    def list_clip_candidates(
        self,
        job_id: str,
        *,
        min_duration: float = 45.0,
        target_duration: float = 120.0,
        max_duration: float = 180.0,
        limit: int = 24,
        selector: Any | None = None,
    ) -> list[dict[str, Any]]:
        segments = [SubtitleSegment.from_dict(item) for item in self.list_segments(job_id)]
        rule_limit = max(limit, min(48, limit * 4)) if selector is not None else limit
        candidates = [
            candidate.to_dict()
            for candidate in generate_clip_candidates(
                segments,
                min_duration=min_duration,
                target_duration=target_duration,
                max_duration=max_duration,
                limit=rule_limit,
            )
        ]
        if selector is not None:
            return selector.rank_clip_candidates(candidates, limit=limit)
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
        if not detect_ffmpeg().available:
            raise RuntimeError("ffmpeg and ffprobe are not available on PATH.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        start = max(0.0, float(start))
        end = max(start + 0.25, float(end))
        style = SubtitleStyle.from_dict(style_payload)
        clip_id = _safe_clip_name(name or f"clip_{start:.2f}_{end:.2f}")
        job.clips_dir.mkdir(parents=True, exist_ok=True)
        ass_path = job.clips_dir / f"{clip_id}.ass"
        srt_path = job.clips_dir / f"{clip_id}.srt"
        metadata_path = job.clips_dir / f"{clip_id}.json"
        output_path = job.clips_dir / f"{clip_id}.mp4"
        segments = self._clip_segments(job, start=start, end=end)
        if not segments:
            raise RuntimeError("The selected range does not contain any subtitle segments.")
        width, height = probe_video_size(job.input_path)
        write_text(
            ass_path,
            export_ass(segments, style=style, video_width=width, video_height=height),
            encoding="utf-8-sig",
        )
        write_text(
            srt_path,
            export_srt(segments, show_speaker=style.show_speaker, speaker_names=style.speaker_names),
            encoding="utf-8-sig",
        )
        write_text(
            metadata_path,
            json.dumps(
                {
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
        burn_ass_subtitles_clip(job.input_path, ass_path, output_path, start=start, end=end, style=style)
        return {
            "filename": output_path.name,
            "path": str(output_path),
            "start": start,
            "end": end,
            "duration": end - start,
            "segments": len(segments),
            "files": {
                "mp4": output_path.name,
                "srt": srt_path.name,
                "ass": ass_path.name,
                "metadata": metadata_path.name,
            },
        }

    def translate(
        self,
        job_id: str,
        translator: Any,
        *,
        target_language: str = "简体中文",
        mode: str = "replace",
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot translate while the job is running.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        with self._translate_lock:
            self._set_status(job, "translating", max(job.progress, 0.95), error=None)
            try:
                if not job.source_segments_path.exists():
                    shutil.copyfile(job.segments_path, job.source_segments_path)
                source_payload = json.loads(job.source_segments_path.read_text(encoding="utf-8"))
                segments = [SubtitleSegment.from_dict(item) for item in source_payload]
                pretranslation_skips = collect_pretranslation_skips(segments)
                started = time.time()

                def update_translation_progress(done: int, total: int, batch_start: int, batch_count: int) -> None:
                    ratio = 1.0 if total <= 0 else max(0.0, min(1.0, done / total))
                    job.translation_info = {
                        "applied": False,
                        "in_progress": True,
                        "model": getattr(translator, "model", None),
                        "target_language": target_language,
                        "mode": mode,
                        "done": done,
                        "total": total,
                        "pretranslation_skip_count": len(pretranslation_skips),
                        "batch_start": batch_start,
                        "batch_count": batch_count,
                        "percent": round(ratio * 100, 1),
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                    self._set_status(job, "translating", 0.95 + 0.04 * ratio, error=None)

                job.translation_info = {
                    "applied": False,
                    "in_progress": True,
                    "model": getattr(translator, "model", None),
                    "target_language": target_language,
                    "mode": mode,
                    "done": 0,
                    "total": len(segments),
                    "pretranslation_skip_count": len(pretranslation_skips),
                    "percent": 0.0,
                    "elapsed_sec": 0.0,
                }
                self._touch(job, error=None)
                translate_kwargs = {
                    "target_language": target_language,
                    "progress_callback": update_translation_progress,
                }
                if batch_size is not None:
                    translate_kwargs["batch_size"] = batch_size
                translations = translator.translate_segments(segments, **translate_kwargs)
                elapsed = time.time() - started
                validation_issues = validate_translation_outputs(segments, translations)
                translated = apply_translations(segments, translations, mode=mode)
                self._write_subtitle_files(job, translated)
                job.translation_info = {
                    "applied": True,
                    "in_progress": False,
                    "model": getattr(translator, "model", None),
                    "target_language": target_language,
                    "mode": mode,
                    "done": len(translated),
                    "total": len(segments),
                    "percent": 100.0,
                    "elapsed_sec": round(elapsed, 3),
                    "pretranslation_skip_count": len(pretranslation_skips),
                    "pretranslation_skips": pretranslation_skips[:50],
                    "validation_issue_count": len(validation_issues),
                    "validation_issues": validation_issues[:20],
                }
                job.status = "waiting_review"
                job.progress = 0.95
                self._touch(job, error=None)
                return {
                    "segments": [segment.to_dict() for segment in translated],
                    "count": len(translated),
                    "target_language": target_language,
                    "mode": mode,
                    "elapsed_sec": round(elapsed, 3),
                    "pretranslation_skip_count": len(pretranslation_skips),
                    "pretranslation_skips": pretranslation_skips[:50],
                    "validation_issue_count": len(validation_issues),
                    "validation_issues": validation_issues[:20],
                }
            except Exception as exc:
                job.translation_info = {
                    **job.translation_info,
                    "in_progress": False,
                    "error": str(exc),
                }
                self._set_status(job, "waiting_review", max(job.progress, 0.95), error=f"Translation failed: {exc}")
                raise

    def restore_source_segments(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot restore subtitles while the job is running.")
        if not job.source_segments_path.exists():
            raise FileNotFoundError("No source subtitle backup is available.")
        segments = [
            SubtitleSegment.from_dict(item)
            for item in json.loads(job.source_segments_path.read_text(encoding="utf-8"))
        ]
        self._write_subtitle_files(job, segments)
        job.translation_info = {"applied": False}
        self._touch(job, error=None)
        return {"segments": [segment.to_dict() for segment in segments], "count": len(segments)}

    def clip_download_path(self, job_id: str, filename: str) -> Path:
        job = self.get_job(job_id)
        path = (job.clips_dir / Path(filename).name).resolve()
        root = job.clips_dir.resolve()
        if root not in path.parents or path.suffix.lower() not in {".mp4", ".srt", ".ass", ".json"}:
            raise FileNotFoundError(filename)
        if not path.exists():
            raise FileNotFoundError(filename)
        return path

    def download_path(self, job_id: str, kind: str) -> Path:
        job = self.get_job(job_id)
        table = {
            "json": job.segments_path,
            "segments": job.segments_path,
            "srt": job.srt_path,
            "ass": job.ass_path,
            "mp4": job.output_path,
            "transcript": job.raw_transcript_path,
        }
        if kind not in table:
            raise KeyError(f"Unsupported download kind: {kind}")
        path = table[kind]
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._process_job(self.get_job(job_id))
            finally:
                self._queue.task_done()

    def _load_existing_jobs(self) -> None:
        for path in sorted(self.runs_dir.glob("*/job.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = JobRecord.from_dict(data)
                if not job.job_dir:
                    job.job_dir = str(path.parent)
                if job.status in RUNNING_STATES:
                    job.status = "failed"
                    job.progress = 1.0
                    job.error = "Interrupted by previous server shutdown."
                    job.updated_at = time.time()
                    self._save_job(job)
                self._jobs[job.id] = job
            except Exception:
                continue

    def _process_job(self, job: JobRecord) -> None:
        try:
            def update(status: str, progress: float | None, generated_tokens: int | None = None) -> None:
                self._raise_if_cancelled(job.id)
                if status == "transcribing" and job.generated_tokens is None:
                    job.generated_tokens = 0
                if generated_tokens is not None:
                    job.generated_tokens = generated_tokens
                save = generated_tokens is None or self._should_save_live_progress(job.id)
                self._set_status(job, status, progress if progress is not None else job.progress, save=save)

            self._apply_auto_max_new_tokens(job)
            self._raise_if_cancelled(job.id)

            result = self.model_runner.transcribe(
                job.input_path,
                prompt=job.inference_prompt,
                max_length=job.max_length,
                max_new_tokens=job.max_new_tokens,
                decoding=job.decoding,
                temperature=job.temperature,
                status_callback=update,
            )
            self._raise_if_cancelled(job.id)
            job.generated_tokens = result.generated_tokens
            self._set_status(job, "postprocessing", 0.85, error=None)
            job.raw_transcript_path.write_text(result.text, encoding="utf-8")
            segments = subtitle_segments_from_transcript(result.text, postprocess=False)
            self._raise_if_cancelled(job.id)
            self._set_status(job, "labeling_speakers", 0.88, error=None)
            segments, speaker_info = label_speakers(
                job.input_path,
                segments,
                work_dir=job.job_dir,
                max_speakers=max(2, int(job.speaker_count or 0)),
                target_speakers=job.speaker_count,
                backend=job.diarization_backend,
                hf_token=self.hf_token,
                pyannote_model=self.pyannote_model,
                device=self.diarization_device,
            )
            job.speaker_labeling = speaker_info.to_dict()
            self._write_subtitle_files(job, segments)
            job.prompt_len = result.prompt_len
            job.elapsed_sec = result.elapsed_sec
            self._set_status(job, "waiting_review", 0.95, error=None)
            self._progress_save_times.pop(job.id, None)
        except JobCancelled:
            self._cancelled_jobs.discard(job.id)
            self._progress_save_times.pop(job.id, None)
        except Exception as exc:
            if job.id in self._cancelled_jobs:
                self._cancelled_jobs.discard(job.id)
                self._progress_save_times.pop(job.id, None)
                return
            self._set_status(job, "failed", 1.0, error=str(exc))
            self._progress_save_times.pop(job.id, None)

    def _apply_auto_max_new_tokens(self, job: JobRecord) -> None:
        """Raise ``max_new_tokens`` to a non-truncating floor based on audio duration.

        Only ever increases the value, so an explicit user override that is already
        large enough is left untouched. If the duration cannot be probed (e.g. a
        non-media file or ffprobe unavailable), the configured value is kept as-is.
        """
        duration = probe_media_duration(job.input_path)
        recommended = recommend_max_new_tokens(duration, max_length=job.max_length)
        if recommended is not None and recommended > job.max_new_tokens:
            job.max_new_tokens = recommended
            self._save_job(job)

    def _render_job(self, job_id: str, style: SubtitleStyle) -> None:
        job = self.get_job(job_id)
        with self._render_lock:
            try:
                self._set_status(job, "rendering", 0.95, error=None)
                segments = [SubtitleSegment.from_dict(item) for item in self.list_segments(job.id)]
                width, height = probe_video_size(job.input_path)
                write_text(job.ass_path, export_ass(segments, style=style, video_width=width, video_height=height))
                def on_render_progress(ratio: float) -> None:
                    progress = max(job.progress, 0.95 + max(0.0, min(1.0, ratio)) * 0.049)
                    self._set_status(
                        job,
                        "rendering",
                        progress,
                        error=None,
                        save=self._should_save_live_progress(job.id),
                    )

                burn_ass_subtitles(
                    job.input_path,
                    job.ass_path,
                    job.output_path,
                    style=style,
                    progress_callback=on_render_progress,
                )
                self._set_status(job, "done", 1.0, error=None)
            except Exception as exc:
                self._set_status(job, "waiting_review", 0.95, error=f"Render failed: {exc}")

    def _clip_segments(self, job: JobRecord, *, start: float, end: float) -> list[SubtitleSegment]:
        source = [SubtitleSegment.from_dict(item) for item in self.list_segments(job.id)]
        return rebase_segments_for_clip(source, start=start, end=end)

    def _write_subtitle_files(
        self,
        job: JobRecord,
        segments: list[SubtitleSegment],
        *,
        style: SubtitleStyle | None = None,
    ) -> None:
        write_text(job.segments_path, export_json(segments))
        if style is None:
            style = SubtitleStyle.from_dict(job.subtitle_style) if job.subtitle_style else SubtitleStyle(font_size=48)
        write_text(
            job.srt_path,
            export_srt(segments, show_speaker=style.show_speaker, speaker_names=style.speaker_names),
            encoding="utf-8-sig",
        )
        width, height = probe_video_size(job.input_path)
        write_text(
            job.ass_path,
            export_ass(segments, style=style, video_width=width, video_height=height),
            encoding="utf-8-sig",
        )
        write_text(job.segments_path, export_json(segments))

    def _maybe_sync_segments_from_subtitle_files(self, job: JobRecord, *, force: bool = False) -> dict[str, Any] | None:
        source_path, source_kind = self._select_subtitle_source(job, force=force)
        if source_path is None:
            return None
        if not force and job.segments_path.exists() and job.segments_path.stat().st_mtime >= source_path.stat().st_mtime:
            return None
        text = source_path.read_text(encoding="utf-8-sig" if source_kind == "srt" else "utf-8")
        segments = parse_srt(text) if source_kind == "srt" else parse_ass(text)
        if not segments and text.strip():
            return None
        style = SubtitleStyle.from_dict(job.subtitle_style) if job.subtitle_style else SubtitleStyle(font_size=48)
        self._write_subtitle_files(job, segments, style=style)
        write_text(job.segments_path, export_json(segments))
        self._touch(job, error=None)
        return {"source": source_kind, "count": len(segments)}

    def _select_subtitle_source(self, job: JobRecord, *, force: bool = False) -> tuple[Path | None, str | None]:
        if job.srt_path.exists():
            return job.srt_path, "srt"
        if job.ass_path.exists():
            return job.ass_path, "ass"
        if force and job.segments_path.exists():
            return job.segments_path, "json"
        return None, None

    def _set_status(
        self,
        job: JobRecord,
        status: str,
        progress: float,
        *,
        error: str | None = None,
        save: bool = True,
    ) -> None:
        self._raise_if_cancelled(job.id)
        job.status = status
        job.progress = max(0.0, min(1.0, progress))
        self._touch(job, error=error, save=save)

    def _resolve_inference_options(
        self,
        *,
        prompt: str | None,
        max_length: int | None,
        max_new_tokens: int | None,
        decoding: str | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        prompt_value = self.prompt if prompt is None or not prompt.strip() else prompt
        max_length_value = self.max_length if max_length is None else int(max_length)
        max_new_tokens_value = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        decoding_value = decoding or self.decoding
        if decoding_value not in {"greedy", "sample"}:
            raise ValueError("decoding must be greedy or sample.")
        if max_length_value <= 0:
            raise ValueError("max_length must be greater than 0.")
        if max_new_tokens_value <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        temperature_value = self.temperature if temperature is None else float(temperature)
        if decoding_value == "greedy":
            temperature_value = None
        else:
            if temperature_value is None:
                temperature_value = 1.0
            if temperature_value <= 0:
                raise ValueError("temperature must be greater than 0.")

        return {
            "prompt": prompt_value,
            "max_length": max_length_value,
            "max_new_tokens": max_new_tokens_value,
            "decoding": decoding_value,
            "temperature": temperature_value,
        }

    def _resolve_speaker_count(self, speaker_count: int | None) -> int | None:
        value = getattr(self, "speaker_count", None) if speaker_count is None else speaker_count
        if value in ("", None):
            return None
        value = int(value)
        if value <= 0:
            return None
        return max(1, min(value, 2))

    def _resolve_diarization_backend(self, diarization_backend: str | None) -> str:
        value = (diarization_backend or self.diarization_backend or "auto").lower()
        if value in {"auto", "pyannote", "cluster", "none", "off", "disabled"}:
            return "none" if value in {"off", "disabled"} else value
        return "auto"

    def _touch(self, job: JobRecord, *, error: str | None = None, save: bool = True) -> None:
        job.error = error
        job.updated_at = time.time()
        if save:
            self._save_job(job)

    def _save_job(self, job: JobRecord) -> None:
        job.job_path.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _should_save_live_progress(self, job_id: str) -> bool:
        now = time.time()
        last_saved = self._progress_save_times.get(job_id, 0.0)
        if now - last_saved < 0.5:
            return False
        self._progress_save_times[job_id] = now
        return True

    def _runner_backend(self) -> str:
        if hasattr(self.model_runner, "runtime_info"):
            try:
                return str(self.model_runner.runtime_info().get("backend") or "")
            except Exception:
                return ""
        return ""

    def _raise_if_cancelled(self, job_id: str) -> None:
        if job_id in self._cancelled_jobs or job_id not in self._jobs:
            raise JobCancelled(job_id)


def _safe_clip_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    safe = safe.strip("._")
    return safe[:80] or f"clip_{uuid.uuid4().hex[:8]}"
