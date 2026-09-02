from __future__ import annotations

import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.whisper_runner import TranscriptionResult

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


class FakeRunner:
    model_path = "fake-model"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("status_callback")
        if callback:
            callback("transcribing", 0.5)
        return TranscriptionResult(
            text="[0][S01]hello[1.5]",
            prompt_len=10,
            generated_tokens=5,
            elapsed_sec=0.01,
            model="fake-model",
            audio=str(audio_path),
            decoding="greedy",
            temperature=None,
        )


class BlockingRunner:
    model_path = "fake-model"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio_path, **kwargs):
        callback = kwargs.get("status_callback")
        if callback:
            callback("transcribing", 0.55, 3)
        self.started.set()
        self.release.wait(timeout=2)
        return TranscriptionResult(
            text="[0][S01]hello[1.5]",
            prompt_len=10,
            generated_tokens=5,
            elapsed_sec=0.01,
            model="fake-model",
            audio=str(audio_path),
            decoding="greedy",
            temperature=None,
        )


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class UploadWhitelistTest(unittest.TestCase):
    def test_upload_rejects_non_media_suffix(self):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            client = TestClient(app)
            resp = client.post(
                "/api/jobs",
                files={"file": ("document.pdf", b"%PDF-1.4", "application/pdf")},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("不支持的文件类型", resp.json()["detail"])

    def test_upload_accepts_media_suffix(self):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            client = TestClient(app)
            resp = client.post(
                "/api/jobs",
                files={"file": ("video.MKV", b"video-bytes", "application/octet-stream")},
            )
            self.assertEqual(resp.status_code, 200)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class CookiesTempFileTest(unittest.TestCase):
    def test_check_cookies_deletes_uploaded_temp_file(self):
        """cookies 检测接口用完临时文件必须删除,登录凭据不能残留在 runs 根目录。"""
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            client = TestClient(app)
            cookies_content = (
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t0\tCONSENT\tYES\n"
            )
            resp = client.post(
                "/api/cookies/check",
                files={"file": ("cookies.txt", cookies_content.encode("utf-8"), "text/plain")},
                data={"browser": "none"},
            )
            self.assertEqual(resp.status_code, 200)
            leftovers = list(Path(tmpdir).glob("*.cookies.txt"))
            self.assertEqual(leftovers, [], f"cookies 临时文件泄漏: {leftovers}")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class AppApiTest(unittest.TestCase):
    def test_job_lifecycle_and_missing_ffmpeg_render_error(self):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            runner = FakeRunner()
            app.state.manager.model_runner = runner
            client = TestClient(app)

            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                data={
                    "prompt": "custom prompt",
                    "max_new_tokens": "5",
                    "max_len": "456",
                    "decoding": "sample",
                    "temperature": "0.7",
                },
            )
            self.assertEqual(created.status_code, 200)
            job_id = created.json()["id"]

            job = {}
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "waiting_review")
            self.assertEqual(job["inference"]["prompt"], "custom prompt")
            self.assertEqual(job["inference"]["max_new_tokens"], 5)
            self.assertEqual(job["inference"]["max_length"], 456)
            self.assertEqual(job["inference"]["decoding"], "sample")
            self.assertEqual(job["inference"]["temperature"], 0.7)
            self.assertEqual(job["usage"]["generated_tokens"], 5)
            self.assertEqual(job["usage"]["max_new_tokens"], 5)
            self.assertTrue(job["usage"]["possibly_truncated"])
            self.assertEqual(runner.calls[-1]["prompt"], "custom prompt")
            self.assertEqual(runner.calls[-1]["max_new_tokens"], 5)
            self.assertEqual(runner.calls[-1]["max_length"], 456)
            self.assertEqual(runner.calls[-1]["decoding"], "sample")
            self.assertEqual(runner.calls[-1]["temperature"], 0.7)

            listed = client.get("/api/jobs")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["jobs"][0]["id"], job_id)

            media = client.get(f"/api/jobs/{job_id}/media")
            self.assertEqual(media.status_code, 200)

            rerun = client.post(f"/api/jobs/{job_id}/rerun", json={"max_new_tokens": 10})
            self.assertEqual(rerun.status_code, 200)
            rerun_id = rerun.json()["id"]
            self.assertNotEqual(rerun_id, job_id)
            rerun_job = {}
            for _ in range(40):
                rerun_job = client.get(f"/api/jobs/{rerun_id}").json()
                if rerun_job["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(rerun_job["status"], "waiting_review")
            self.assertEqual(rerun_job["media_name"], "sample.wav")
            self.assertEqual(rerun_job["inference"]["max_new_tokens"], 10)
            self.assertEqual(runner.calls[-1]["max_new_tokens"], 10)

            segments = client.get(f"/api/jobs/{job_id}/segments").json()["segments"]
            self.assertEqual(segments[0]["speaker"], "S01")
            segments[0]["text"] = "edited"
            updated = client.put(
                f"/api/jobs/{job_id}/segments",
                # 前端保存样式时总是提交完整 style；show_speaker 默认关闭，
                # 显式打开才会在 SRT 里输出说话人前缀。
                json={"segments": segments, "style": {"show_speaker": True, "speaker_names": {"S01": "Alice"}}},
            )
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(updated.json()["segments"][0]["text"], "edited")
            self.assertEqual(client.get(f"/api/jobs/{job_id}").json()["subtitle_style"]["speaker_names"]["S01"], "Alice")

            download = client.get(f"/api/jobs/{job_id}/download?kind=srt")
            self.assertEqual(download.status_code, 200)
            self.assertIn("edited", download.text)
            self.assertIn("Alice: edited", download.text)

            class Missing:
                available = False

            with patch("moss_transcribe_diarize.app.jobs.detect_ffmpeg", return_value=Missing()):
                render = client.post(f"/api/jobs/{job_id}/render", json={"style": {}})
            self.assertEqual(render.status_code, 503)

            deleted = client.delete(f"/api/jobs/{job_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.get(f"/api/jobs/{job_id}").status_code, 404)

    def test_auto_max_new_tokens_bumped_by_duration(self):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            runner = FakeRunner()
            app.state.manager.model_runner = runner
            client = TestClient(app)

            with patch("moss_transcribe_diarize.app.jobs.probe_media_duration", return_value=1800.0):
                created = client.post(
                    "/api/jobs",
                    files={"file": ("sample.wav", b"audio", "audio/wav")},
                    data={"max_new_tokens": "2048"},
                )
                self.assertEqual(created.status_code, 200)
                job_id = created.json()["id"]
                job = {}
                for _ in range(40):
                    job = client.get(f"/api/jobs/{job_id}").json()
                    if job["status"] == "waiting_review":
                        break
                    time.sleep(0.05)

            self.assertEqual(job["status"], "waiting_review")
            # 1800s of audio -> recommended 25600, which overrides the requested 2048.
            self.assertEqual(job["inference"]["max_new_tokens"], 25600)
            self.assertEqual(job["usage"]["max_new_tokens"], 25600)
            self.assertEqual(runner.calls[-1]["max_new_tokens"], 25600)

    def test_running_job_exposes_live_token_progress(self):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            runner = BlockingRunner()
            app.state.manager.model_runner = runner
            client = TestClient(app)

            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                data={"max_new_tokens": "5"},
            )
            self.assertEqual(created.status_code, 200)
            job_id = created.json()["id"]
            self.assertTrue(runner.started.wait(timeout=2))

            running = client.get(f"/api/jobs/{job_id}").json()
            self.assertEqual(running["status"], "transcribing")
            self.assertEqual(running["usage"]["generated_tokens"], 3)
            self.assertEqual(running["usage"]["max_new_tokens"], 5)
            self.assertAlmostEqual(running["progress"], 0.55)

            runner.release.set()
            finished = {}
            for _ in range(40):
                finished = client.get(f"/api/jobs/{job_id}").json()
                if finished["status"] == "waiting_review":
                    break
                time.sleep(0.05)
            self.assertEqual(finished["status"], "waiting_review")
            self.assertEqual(finished["usage"]["generated_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
