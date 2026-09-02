from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from moss_transcribe_diarize.defaults import DEFAULT_PROMPT
from moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    SubtitleStyle,
    export_ass,
    export_json,
    export_srt,
    regroup_sentences,
    regroup_sentences_from_words,
    subtitle_segments_from_transcript,
    write_text,
)

from .clips import generate_clip_candidates, rebase_segments_for_clip
from .ffmpeg import burn_ass_subtitles, burn_ass_subtitles_clip, detect_ffmpeg, probe_video_size
from .speaker_labeler import SpeakerLabelingInfo, label_speakers
from .local_mt_translator import LocalMtTranslator
from .text_translator import (
    PROTECTED_TERMS,
    TextTranslator,
    apply_translations,
    collect_pretranslation_skips,
    validate_translation_outputs,
)
from .whisper_runner import WhisperRunner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate editable subtitles and optionally burn them into video.")
    parser.add_argument("input", help="Input audio or video path.")
    parser.add_argument("--segments-input", default=None, help="Use an existing segments JSON file and skip transcription.")
    parser.add_argument("--backend", choices=["whisper"], default="whisper")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--decoding", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-speaker-labeling", action="store_true", help="Skip lightweight post-transcription speaker labeling.")
    parser.add_argument("--speaker-count", type=int, default=None, help="Optional target speaker count, clamped to 1-10 (auto-detects up to 4 when omitted).")
    parser.add_argument("--diarization-backend", choices=["auto", "pyannote", "cluster", "none"], default="none")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for pyannote gated models. Defaults to HF_TOKEN env var.")
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--diarization-device", default="auto")
    parser.add_argument("--translate-base-url", default=None, help="OpenAI-compatible chat base URL for subtitle translation.")
    parser.add_argument("--translate-model", default="local", help="Translator model name or local OPUS-MT CTranslate2 directory.")
    parser.add_argument("--translate-api-key", default="EMPTY")
    parser.add_argument("--translate-timeout", type=float, default=600.0)
    parser.add_argument("--translate-provider", choices=["openai", "ollama", "opus-mt"], default="openai")
    parser.add_argument("--translate-tokenizer-dir", default="models/opus-mt-en-zh", help="Tokenizer directory for --translate-provider opus-mt.")
    parser.add_argument("--translate-device", default="auto", help="CTranslate2 device for --translate-provider opus-mt.")
    parser.add_argument("--translate-compute-type", default="auto", help="CTranslate2 compute type for --translate-provider opus-mt.")
    parser.add_argument("--translate-batch-size", type=int, default=18)
    parser.add_argument("--translate-protected-terms", default="", help="Comma-separated terms that should stay untranslated.")
    parser.add_argument("--target-language", default="简体中文")
    parser.add_argument("--translate-mode", choices=["replace", "bilingual"], default="bilingual")
    parser.add_argument("--clips", type=int, default=0, help="Generate this many highlight candidates after transcription.")
    parser.add_argument("--clip-min-duration", type=float, default=60.0)
    parser.add_argument("--clip-target-duration", type=float, default=120.0)
    parser.add_argument("--clip-max-duration", type=float, default=180.0)
    parser.add_argument("--render-clips", action="store_true", help="Render generated clip candidates to MP4.")
    parser.add_argument("--render", action="store_true", help="Burn subtitle.ass into output.mp4 with FFmpeg.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir or f"runs/cli_{time.strftime('%Y%m%d_%H%M%S')}").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.segments_input:
        source_payload = json.loads(Path(args.segments_input).expanduser().read_text(encoding="utf-8"))
        segments = [SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}") for index, item in enumerate(source_payload, start=1)]
        raw_transcript_text = "".join(f"[{segment.start:.2f}][{segment.speaker}]{segment.text}[{segment.end:.2f}]" for segment in segments)
        speaker_info = SpeakerLabelingInfo(False, False, "none", 0, len(segments), "segments input")
        transcription_summary = {
            "skipped": True,
            "segments_input": str(Path(args.segments_input).expanduser()),
        }
    else:
        runner = WhisperRunner(args.model, device=args.device, dtype=args.dtype)
        result = runner.transcribe(
            input_path,
            prompt=args.prompt,
            max_length=args.max_len,
            max_new_tokens=args.max_new_tokens,
            decoding=args.decoding,
            temperature=args.temperature,
        )
        raw_transcript_text = result.text
        if result.words:
            segments = regroup_sentences_from_words(result.words)
        else:
            segments = regroup_sentences(subtitle_segments_from_transcript(result.text, postprocess=False))
        segments, speaker_info = label_speakers(
            input_path,
            segments,
            work_dir=out_dir,
            enabled=not args.no_speaker_labeling,
            # 未指定时给 pyannote 自动检测留空间；显式指定时按指定值钳定。
            max_speakers=_resolve_speaker_count(args.speaker_count) or 4,
            target_speakers=_resolve_speaker_count(args.speaker_count),
            backend="none" if args.no_speaker_labeling else args.diarization_backend,
            hf_token=args.hf_token,
            pyannote_model=args.pyannote_model,
            device=args.diarization_device,
        )
        transcription_summary = {k: v for k, v in result.to_dict().items() if k != "text"}

    translated = False
    translation_elapsed_sec = None
    if args.translate_base_url or args.translate_provider == "opus-mt":
        write_text(out_dir / "segments_original.json", export_json(segments))
        protected_terms = tuple(term.strip() for term in args.translate_protected_terms.split(",") if term.strip())
        if args.translate_provider == "opus-mt":
            model_dir = args.translate_model if args.translate_model != "local" else "models/opus-mt-en-zh-ct2-int8"
            translator = LocalMtTranslator(
                model_dir=model_dir,
                tokenizer_dir=args.translate_tokenizer_dir,
                device=args.translate_device,
                compute_type=args.translate_compute_type,
            )
        else:
            translator = TextTranslator(
                base_url=args.translate_base_url,
                model=args.translate_model,
                api_key=args.translate_api_key,
                timeout=args.translate_timeout,
                provider=args.translate_provider,
                protected_terms=protected_terms or tuple(PROTECTED_TERMS),
            )
        translation_started = time.time()

        def print_translation_progress(done: int, total: int, batch_start: int, batch_count: int) -> None:
            percent = 100 if total <= 0 else round(done * 100 / total)
            sys.stdout.write(f"\rTranslating: {done}/{total} ({percent}%)")
            sys.stdout.flush()

        translations = translator.translate_segments(
            segments,
            target_language=args.target_language,
            batch_size=args.translate_batch_size,
            progress_callback=print_translation_progress,
        )
        translation_elapsed_sec = time.time() - translation_started
        if segments:
            sys.stdout.write("\n")
        pretranslation_skips = collect_pretranslation_skips(segments)
        validation_issues = validate_translation_outputs(segments, translations)
        segments = apply_translations(segments, translations, mode=args.translate_mode)
        translated = True

    style = SubtitleStyle(show_speaker=False, speaker_colors=False)
    raw_transcript = write_text(out_dir / "raw_transcript.txt", raw_transcript_text)
    segments_path = write_text(out_dir / "segments.json", export_json(segments))
    srt_path = write_text(out_dir / "subtitle.srt", export_srt(segments, show_speaker=style.show_speaker), encoding="utf-8-sig")
    width, height = probe_video_size(input_path)
    ass_path = write_text(out_dir / "subtitle.ass", export_ass(segments, style=style, video_width=width, video_height=height), encoding="utf-8-sig")

    output_path = None
    if args.render:
        if not detect_ffmpeg().available:
            raise SystemExit("ffmpeg and ffprobe are required for --render.")
        output_path = burn_ass_subtitles(input_path, ass_path, out_dir / "output.mp4")

    clips = []
    rendered_clips = []
    if args.clips > 0:
        clips = [
            candidate.to_dict()
            for candidate in generate_clip_candidates(
                segments,
                min_duration=args.clip_min_duration,
                target_duration=args.clip_target_duration,
                max_duration=args.clip_max_duration,
                limit=args.clips,
            )
        ]
        write_text(out_dir / "clips.json", json.dumps(clips, ensure_ascii=False, indent=2) + "\n")
        if args.render_clips:
            if not detect_ffmpeg().available:
                raise SystemExit("ffmpeg and ffprobe are required for --render-clips.")
            clips_dir = out_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            for index, clip in enumerate(clips, start=1):
                clip_segments = _clip_segments(segments, start=float(clip["start"]), end=float(clip["end"]))
                if not clip_segments:
                    continue
                clip_ass_path = clips_dir / f"clip_{index:02d}.ass"
                clip_mp4_path = clips_dir / f"clip_{index:02d}.mp4"
                write_text(
                    clip_ass_path,
                    export_ass(clip_segments, style=style, video_width=width, video_height=height),
                    encoding="utf-8-sig",
                )
                burn_ass_subtitles_clip(input_path, clip_ass_path, clip_mp4_path, start=clip["start"], end=clip["end"], style=style)
                rendered_clips.append(str(clip_mp4_path))

    summary = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "segments": len(segments),
        "translated": translated,
        "translation_elapsed_sec": translation_elapsed_sec,
        "translation_pretranslation_skip_count": len(pretranslation_skips) if translated else 0,
        "translation_pretranslation_skips": pretranslation_skips[:20] if translated else [],
        "translation_validation_issue_count": len(validation_issues) if translated else 0,
        "translation_validation_issues": validation_issues[:20] if translated else [],
        "files": {
            "raw_transcript": str(raw_transcript),
            "segments": str(segments_path),
            "srt": str(srt_path),
            "ass": str(ass_path),
            "mp4": str(output_path) if output_path else None,
            "clips": str(out_dir / "clips.json") if clips else None,
        },
        "clips": clips,
        "rendered_clips": rendered_clips,
        "transcription": transcription_summary,
        "speaker_labeling": speaker_info.to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _clip_segments(segments: list[SubtitleSegment], *, start: float, end: float) -> list[SubtitleSegment]:
    return rebase_segments_for_clip(segments, start=start, end=end)


def _resolve_speaker_count(value: int | None) -> int | None:
    if value in ("", None):
        return None
    value = int(value)
    if value <= 0:
        return None
    return max(1, min(value, 10))


if __name__ == "__main__":
    main()
