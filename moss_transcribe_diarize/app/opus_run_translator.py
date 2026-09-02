from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    SubtitleStyle,
    export_ass,
    export_json,
    export_srt,
    write_text,
)

from .local_mt_translator import LocalMtTranslator
from .text_translator import (
    apply_translations,
    collect_pretranslation_skips,
    validate_translation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a run's source segments with local OPUS-MT.")
    parser.add_argument("run_dir", help="Run directory containing segments.source.json or segments.json.")
    parser.add_argument("--model-dir", default="models/opus-mt-en-zh-ct2-int8")
    parser.add_argument("--tokenizer-dir", default="models/opus-mt-en-zh")
    parser.add_argument("--suffix", default="opus")
    parser.add_argument("--mode", choices=["replace", "bilingual"], default="bilingual")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    source_path = _source_segments_path(run_dir)
    job_path = run_dir / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else {}
    style = SubtitleStyle.from_dict(job.get("subtitle_style"))
    width, height = _read_ass_resolution(run_dir / "subtitle.ass")
    source_segments = [
        SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}")
        for index, item in enumerate(json.loads(source_path.read_text(encoding="utf-8")), start=1)
    ]
    translator = LocalMtTranslator(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
        compute_type=args.compute_type,
    )
    started = time.time()

    def progress(done: int, total: int, batch_start: int, batch_count: int) -> None:
        percent = 100 if total <= 0 else round(done * 100 / total)
        sys.stdout.write(f"\rTranslating OPUS-MT: {done}/{total} ({percent}%)")
        sys.stdout.flush()

    translations = translator.translate_segments(
        source_segments,
        batch_size=args.batch_size,
        progress_callback=progress,
    )
    if source_segments:
        sys.stdout.write("\n")
    elapsed = time.time() - started
    issues = validate_translation_outputs(source_segments, translations)
    pretranslation_skips = collect_pretranslation_skips(source_segments)
    translated_segments = apply_translations(source_segments, translations, mode=args.mode)
    suffix = args.suffix.strip(".") or "opus"
    segments_path = write_text(run_dir / f"segments.{suffix}.json", export_json(translated_segments))
    srt_path = write_text(
        run_dir / f"subtitle.{suffix}.srt",
        export_srt(translated_segments, show_speaker=style.show_speaker, speaker_names=style.speaker_names),
        encoding="utf-8-sig",
    )
    ass_path = write_text(
        run_dir / f"subtitle.{suffix}.ass",
        export_ass(translated_segments, style=style, video_width=width, video_height=height),
        encoding="utf-8-sig",
    )
    report = {
        "source": str(source_path),
        "model": translator.runtime_info(),
        "mode": args.mode,
        "segments": len(source_segments),
        "elapsed_sec": round(elapsed, 3),
        "pretranslation_skip_count": len(pretranslation_skips),
        "pretranslation_skips": pretranslation_skips[:50],
        "validation_issue_count": len(issues),
        "validation_issues": issues[:50],
        "files": {
            "segments": str(segments_path),
            "srt": str(srt_path),
            "ass": str(ass_path),
        },
    }
    report_path = write_text(
        run_dir / f"translation_report.{suffix}.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2))


def _source_segments_path(run_dir: Path) -> Path:
    for name in ("segments.source.json", "segments_original.json", "segments.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise SystemExit(f"No segments file found in {run_dir}")


def _read_ass_resolution(path: Path) -> tuple[int, int]:
    width = 1920
    height = 1080
    if not path.exists():
        return width, height
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if line.startswith("PlayResX:"):
            width = int(line.split(":", 1)[1].strip())
        elif line.startswith("PlayResY:"):
            height = int(line.split(":", 1)[1].strip())
    return width, height


if __name__ == "__main__":
    main()
