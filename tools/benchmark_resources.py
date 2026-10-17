"""Run isolated, read-only media benchmarks and retain measured evidence.

GPU numbers are device-wide driver readings, including other applications.
They cover CTranslate2/FFmpeg allocations that PyTorch cannot see.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(stage: str, source: Path, workers: int = 1) -> dict:
    with tempfile.TemporaryDirectory(prefix="mtd-benchmark-") as directory:
        if stage == "whisper":
            from moss_transcribe_diarize.app.whisper_runner import WhisperRunner

            runner = WhisperRunner("models/faster-whisper-large-v3-turbo", device="cuda", dtype="float16")
            result = runner.transcribe(str(source), max_length=8192, max_new_tokens=4096)
            return {"characters": len(result.text), "words": len(result.words or []), "runtime": runner.runtime_info()}
        if stage == "demucs":
            from moss_transcribe_diarize.app.vocal_separator import separate_vocals

            output = separate_vocals(source, directory, device="cuda")
            if output is None or not output.is_file():
                raise RuntimeError("Demucs did not produce vocals")
            return {"output_bytes": output.stat().st_size}
        if stage == "pyannote":
            from moss_transcribe_diarize.app.speaker_labeler import label_speakers
            from moss_transcribe_diarize.subtitle import SubtitleSegment

            segments = [SubtitleSegment(id=str(i), start=i * 10, end=i * 10 + 8, text="sample", speaker="") for i in range(5)]
            _, info = label_speakers(source, segments, work_dir=directory, backend="pyannote", device="cuda")
            if not info.applied:
                raise RuntimeError(info.reason)
            return info.to_dict()
        if stage == "translation":
            from moss_transcribe_diarize.app.local_mt_translator import LocalMtTranslator
            from moss_transcribe_diarize.subtitle import SubtitleSegment

            items = json.loads(source.read_text(encoding="utf-8"))
            segments = [SubtitleSegment.from_dict(item) for item in items]
            translator = LocalMtTranslator(device="cuda", compute_type="float16")
            result = translator.translate_segments(segments)
            return {"segments": len(result), "runtime": translator.runtime_info()}
        if stage == "render":
            from moss_transcribe_diarize.app.ffmpeg import burn_ass_subtitles, probe_media_duration
            from moss_transcribe_diarize.subtitle import SubtitleSegment, export_ass

            output = Path(directory) / "render.mp4"
            ass = Path(directory) / "subtitle.ass"
            duration = probe_media_duration(source)
            segments = [SubtitleSegment(id="sample", start=0, end=duration, text="Performance benchmark", speaker="")]
            ass.write_text(export_ass(segments), encoding="utf-8-sig")
            burn_ass_subtitles(source, ass, output)
            return {"output_bytes": output.stat().st_size, "duration_sec": duration, "subtitle_burn": True}
        if stage == "clips":
            from concurrent.futures import ThreadPoolExecutor

            from moss_transcribe_diarize.app.ffmpeg import burn_ass_subtitles_clip
            from moss_transcribe_diarize.subtitle import SubtitleSegment, export_ass

            def render(index: int) -> int:
                ass = Path(directory) / f"subtitle-{index}.ass"
                output = Path(directory) / f"clip-{index}.mp4"
                ass.write_text(export_ass([SubtitleSegment(id="sample", start=0, end=6, text="Performance benchmark", speaker="")]), encoding="utf-8-sig")
                burn_ass_subtitles_clip(source, ass, output, start=10 + index * 8, end=16 + index * 8)
                return output.stat().st_size

            with ThreadPoolExecutor(max_workers=workers) as executor:
                sizes = list(executor.map(render, range(4)))
            return {"clips": 4, "duration_each_sec": 6, "workers": workers, "output_bytes": sizes, "subtitle_burn": True}
        raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["whisper", "demucs", "pyannote", "translation", "render", "clips"])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workers", type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.stage, args.source, args.workers), ensure_ascii=True), flush=True)
        return
    import psutil

    if args.output is None:
        parser.error("--output is required")
    gpu_samples: list[float] = []
    stop = threading.Event()

    def sample_gpu() -> None:
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_samples.append(sum(float(line) for line in result.stdout.splitlines()))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            stop.wait(0.5)

    monitor = threading.Thread(target=sample_gpu, daemon=True)
    monitor.start()
    started = time.perf_counter()
    peak_rss = 0
    # Disk spools avoid deadlocking on dependency warnings filling stdout pipes.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), args.stage, str(args.source.resolve()), "--worker", "--workers", str(args.workers)], stdout=stdout, stderr=stderr)
        while process.poll() is None:
            try:
                root = psutil.Process(process.pid)
                rss = 0
                for child in [root, *root.children(recursive=True)]:
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak_rss = max(peak_rss, rss)
            except psutil.Error:
                pass
            time.sleep(0.1)
        stop.set()
        monitor.join(timeout=6)
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read().decode("utf-8", errors="replace")
        errors = stderr.read().decode("utf-8", errors="replace")
    report = {
        "stage": args.stage, "source": str(args.source), "elapsed_sec": round(time.perf_counter() - started, 3),
        "peak_tree_rss_mib": round(peak_rss / 1048576, 1), "gpu_scope": "all GPUs, device-wide including other processes",
        "gpu_samples": len(gpu_samples), "gpu_baseline_mib": gpu_samples[0] if gpu_samples else None,
        "gpu_peak_used_mib": max(gpu_samples) if gpu_samples else None,
        "returncode": process.returncode, "result": output[-16000:], "stderr": errors[-8000:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"stderr", "result"}}, ensure_ascii=True))
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
