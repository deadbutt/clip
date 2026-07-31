from __future__ import annotations

import threading
import time
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from moss_transcribe_diarize.defaults import DEFAULT_PROMPT

from .ffmpeg import detect_ffmpeg

StatusCallback = Callable[[str, float | None, int | None], None]


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    prompt_len: int
    generated_tokens: int
    elapsed_sec: float
    model: str
    audio: str
    decoding: str
    temperature: float | None
    top_p: float | None = None
    top_k: int | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_len": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "elapsed_sec": self.elapsed_sec,
            "model": self.model,
            "audio": self.audio,
            "decoding": self.decoding,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


class WhisperRunner:
    """Fast, practical transcription runner with a Whisper fallback chain."""

    def __init__(
        self,
        model_path: str | Path = "medium",
        *,
        device: str = "auto",
        dtype: str = "auto",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 3,
    ):
        self.model_path = str(model_path)
        self.device_name = device
        self.dtype_name = dtype
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._model = None
        self._engine = None
        self._openai_fp16 = False
        self._fallback_errors: list[str] = []
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def runtime_info(self) -> dict:
        return {
            "backend": "whisper",
            "path": self.model_path,
            "engine": self._engine or "unloaded",
            "device": self.device_name,
            "dtype": self.dtype_name,
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "fallback_errors": list(self._fallback_errors),
        }

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_length: int = 0,
        max_new_tokens: int = 0,
        decoding: str = "greedy",
        temperature: float | None = None,
        status_callback: StatusCallback | None = None,
        **_: object,
    ) -> TranscriptionResult:
        del max_length, max_new_tokens, decoding, temperature
        with self._lock:
            if status_callback is not None:
                status_callback("loading_model", 0.05, None)
            self._ensure_loaded()
            if status_callback is not None:
                status_callback("transcribing", 0.10, None)

            started = time.time()
            if self._engine == "faster-whisper":
                parts, segment_count = self._transcribe_faster_whisper(audio_path, prompt, status_callback)
            else:
                parts, segment_count = self._transcribe_openai_whisper(audio_path, prompt, status_callback)

            if status_callback is not None:
                status_callback("transcribing", 0.85, segment_count)
            return TranscriptionResult(
                text="".join(parts),
                prompt_len=0,
                generated_tokens=segment_count,
                elapsed_sec=time.time() - started,
                model=self.model_path,
                audio=str(Path(audio_path).expanduser()),
                decoding="beam_search",
                temperature=None,
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _ensure_ffmpeg_on_path()
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        errors: list[str] = []
        self._fallback_errors = []
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            errors.append(f"faster-whisper unavailable: {exc}")
            self._fallback_errors = list(errors)
        else:
            try:
                device, compute_type = _resolve_runtime(self.device_name, self.dtype_name)
                self._model = WhisperModel(self.model_path, device=device, compute_type=compute_type)
                self._engine = "faster-whisper"
                return
            except Exception as exc:
                errors.append(f"faster-whisper load failed: {exc}")
                self._fallback_errors = list(errors)

        try:
            import whisper as openai_whisper
        except ImportError as exc:
            errors.append(f"openai-whisper unavailable: {exc}")
        else:
            try:
                device, fp16 = _resolve_openai_runtime(self.device_name, self.dtype_name)
                self._model = openai_whisper.load_model(self.model_path, device=device)
                self._engine = "openai-whisper"
                self._openai_fp16 = fp16
                if self._fallback_errors:
                    print(
                        "[WARN] faster-whisper was not used; falling back to openai-whisper: "
                        + " | ".join(self._fallback_errors),
                        flush=True,
                    )
                return
            except Exception as exc:
                errors.append(f"openai-whisper load failed: {exc}")

        raise RuntimeError("Whisper model load failed: " + " | ".join(errors))

    def _transcribe_faster_whisper(
        self,
        audio_path: str | Path,
        prompt: str,
        status_callback: StatusCallback | None,
    ) -> tuple[list[str], int]:
        path = str(Path(audio_path).expanduser())
        kwargs = {
            "language": self.language,
            "beam_size": int(self.beam_size),
            "vad_filter": bool(self.vad_filter),
            "initial_prompt": _whisper_initial_prompt(prompt),
            "condition_on_previous_text": bool(self.condition_on_previous_text),
            "repetition_penalty": float(self.repetition_penalty),
            "no_repeat_ngram_size": int(self.no_repeat_ngram_size),
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        try:
            segments_iter, info = self._model.transcribe(path, **kwargs)
        except TypeError as exc:
            if not _looks_like_unsupported_transcribe_option(exc):
                raise
            for key in (
                "condition_on_previous_text",
                "repetition_penalty",
                "no_repeat_ngram_size",
                "compression_ratio_threshold",
                "log_prob_threshold",
                "no_speech_threshold",
            ):
                kwargs.pop(key, None)
            segments_iter, info = self._model.transcribe(path, **kwargs)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        repeat_guard = _RepeatedPhraseGuard()
        parts: list[str] = []
        segment_count = 0
        for segment in segments_iter:
            segment_count += 1
            start = max(0.0, float(segment.start))
            end = max(start, float(segment.end))
            text = str(segment.text or "").strip()
            if not text:
                continue
            if repeat_guard.should_skip(text):
                if status_callback is not None:
                    progress = _duration_progress(end, duration)
                    status_callback("transcribing", progress, segment_count)
                continue
            parts.append(f"[{start:.2f}][S00]{text}[{end:.2f}]")
            if status_callback is not None:
                progress = _duration_progress(end, duration)
                status_callback("transcribing", progress, segment_count)
        return parts, segment_count

    def _transcribe_openai_whisper(
        self,
        audio_path: str | Path,
        prompt: str,
        status_callback: StatusCallback | None,
    ) -> tuple[list[str], int]:
        result = self._model.transcribe(
            str(Path(audio_path).expanduser()),
            language=self.language,
            fp16=bool(self._openai_fp16),
            initial_prompt=_whisper_initial_prompt(prompt),
            verbose=False,
            condition_on_previous_text=bool(self.condition_on_previous_text),
            beam_size=int(self.beam_size),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        duration = float(result.get("duration") or 0.0)
        segments = result.get("segments") or []
        repeat_guard = _RepeatedPhraseGuard()
        parts: list[str] = []
        segment_count = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_count += 1
            start = max(0.0, float(segment.get("start") or 0.0))
            end = max(start, float(segment.get("end") or start))
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            if repeat_guard.should_skip(text):
                if status_callback is not None:
                    progress = _duration_progress(end, duration)
                    status_callback("transcribing", progress, segment_count)
                continue
            parts.append(f"[{start:.2f}][S00]{text}[{end:.2f}]")
            if status_callback is not None:
                progress = _duration_progress(end, duration)
                status_callback("transcribing", progress, segment_count)
        return parts, segment_count


def _resolve_runtime(device: str, dtype: str) -> tuple[str, str]:
    device = (device or "auto").lower()
    dtype = (dtype or "auto").lower()
    if dtype in {"auto", "bf16", "bfloat16"} and device == "auto":
        dtype = "default"
    elif dtype == "auto":
        dtype = "float16" if device == "cuda" else "int8"
    if dtype == "bf16":
        dtype = "float16"
    if dtype == "bfloat16":
        dtype = "float16"
    if dtype == "fp16":
        dtype = "float16"
    if dtype == "fp32":
        dtype = "float32"
    return device, dtype


def _resolve_openai_runtime(device: str, dtype: str) -> tuple[str, bool]:
    try:
        import torch
    except ImportError:
        torch = None
    device = (device or "auto").lower()
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    dtype = (dtype or "auto").lower()
    fp16 = device == "cuda" and dtype not in {"fp32", "float32"}
    return device, fp16


def _whisper_initial_prompt(prompt: str) -> str | None:
    prompt = (prompt or "").strip()
    if not prompt or prompt == DEFAULT_PROMPT:
        return None
    return prompt


class _RepeatedPhraseGuard:
    def __init__(self, *, max_consecutive: int = 3, max_total: int = 24):
        self.max_consecutive = max_consecutive
        self.max_total = max_total
        self._last = ""
        self._consecutive = 0
        self._totals: dict[str, int] = {}

    def should_skip(self, text: str) -> bool:
        normalized = _normalize_repeated_phrase(text)
        if not _is_repeated_phrase_candidate(normalized):
            self._last = ""
            self._consecutive = 0
            return False

        if normalized == self._last:
            self._consecutive += 1
        else:
            self._last = normalized
            self._consecutive = 1

        total = self._totals.get(normalized, 0) + 1
        self._totals[normalized] = total
        return self._consecutive > self.max_consecutive or total > self.max_total


def _normalize_repeated_phrase(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_repeated_phrase_candidate(normalized: str) -> bool:
    if not normalized or len(normalized) > 64:
        return False
    words = normalized.split()
    return 1 <= len(words) <= 6


def _looks_like_unsupported_transcribe_option(exc: TypeError) -> bool:
    message = str(exc).lower()
    return "unexpected keyword" in message or "got an unexpected" in message


def _ensure_ffmpeg_on_path() -> None:
    tools = detect_ffmpeg()
    ffmpeg_path = tools.ffmpeg or tools.ffprobe
    if not ffmpeg_path:
        return
    ffmpeg_dir = str(Path(ffmpeg_path).resolve().parent)
    current = os.environ.get("PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if ffmpeg_dir in parts:
        return
    os.environ["PATH"] = os.pathsep.join([ffmpeg_dir, *parts]) if parts else ffmpeg_dir


def _duration_progress(position: float, duration: float) -> float:
    if duration <= 0:
        return 0.25
    ratio = max(0.0, min(1.0, position / duration))
    return 0.10 + (0.85 - 0.10) * ratio
