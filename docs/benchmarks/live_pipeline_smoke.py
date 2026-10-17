"""用真实任务媒体测试实时管线，临时音频退出即清理，不改动 runs。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from moss_transcribe_diarize.app.ffmpeg import detect_ffmpeg  # noqa: E402
from moss_transcribe_diarize.app.live_subtitles import LiveSubtitleService  # noqa: E402
from moss_transcribe_diarize.app.whisper_runner import WhisperRunner  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    job = json.loads((ROOT / "runs/11a0ae8d36ff/job.json").read_text(encoding="utf-8"))
    runner = WhisperRunner(ROOT / "models/faster-whisper-large-v3-turbo", device="cuda", dtype="float16")
    with tempfile.TemporaryDirectory(prefix="mtd-live-smoke-") as temporary:
        directory = Path(temporary)
        audio = directory / "source.wav"
        subprocess.run([detect_ffmpeg().ffmpeg, "-v", "error", "-ss", "120", "-i", job["input_path"],
                        "-t", "12", "-ac", "1", "-ar", "16000", str(audio)], check=True, timeout=30)
        service = LiveSubtitleService(runner, directory / "drafts")
        started = time.perf_counter()
        try:
            service.start("real-run", str(audio), language="en")
            session = service.sessions["real-run"]
            for thread in session.threads:
                thread.join(timeout=120)
            if any(thread.is_alive() for thread in session.threads):
                raise RuntimeError("实时管线测试超时")
            result = session.snapshot()
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            result["source_run"] = job["id"]
            result["source_start_seconds"] = 120
            result.pop("draft_path", None)
            output = ROOT / "docs/benchmarks/live-pipeline-real-20261016.json"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["error"] or not result["rows"] or any(row["translation_state"] != "done" for row in result["rows"]):
                raise RuntimeError("真实管线未成功输出完整双语字幕")
        finally:
            service.close()


if __name__ == "__main__":
    main()
