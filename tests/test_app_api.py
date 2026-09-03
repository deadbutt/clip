from __future__ import annotations

import asyncio
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

    def test_url_job_failure_deletes_uploaded_cookies(self):
        """/api/jobs/url 建任务失败(空 URL)时任务未入列,上传的 cookies 临时副本必须删除。"""
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            client = TestClient(app)
            resp = client.post(
                "/api/jobs/url",
                files={
                    "cookies_file": (
                        "cookies.txt",
                        b"# Netscape HTTP Cookie File\n",
                        "text/plain",
                    )
                },
                data={"url": ""},
            )
            self.assertEqual(resp.status_code, 400)
            leftovers = list(Path(tmpdir).glob("*.cookies.txt"))
            self.assertEqual(leftovers, [], f"cookies 临时文件泄漏: {leftovers}")

    def test_startup_sweeps_orphan_cookies_files(self):
        """启动清扫只删上一进程遗留的孤儿 .cookies.txt,待下载任务引用的保留。"""
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.jobs import JobRecord
        from moss_transcribe_diarize.app.server import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            orphan = runs / "orphan.cookies.txt"
            orphan.write_text("# leftover\n", encoding="utf-8")
            referenced = runs / "pending.cookies.txt"
            referenced.write_text("# pending job\n", encoding="utf-8")
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            mgr = app.state.manager
            mgr._jobs["job-pending"] = JobRecord(
                id="job-pending",
                status="downloading",
                media_name="pending",
                input_path="",
                job_dir=str(runs / "job-pending"),
                inference_prompt="",
                max_length=1000,
                max_new_tokens=8,
                decoding="greedy",
                temperature=None,
                source="url",
                source_url="https://example.com/v",
                cookies_config={"browser": "firefox", "file": str(referenced)},
            )
            with TestClient(app):
                self.assertFalse(orphan.exists(), "孤儿 cookies 临时文件未被清扫")
                self.assertTrue(referenced.exists(), "待下载任务引用的 cookies 被误删")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class RenderClipConcurrencyTest(unittest.TestCase):
    def test_render_clip_rejects_concurrent_same_clip(self):
        """同一片段并发渲染必须互斥:两个 ffmpeg 写同一输出文件会互相损坏。"""
        from moss_transcribe_diarize.app.server import create_app

        class Available:
            # CI runner 不带 ffmpeg(镜像 Tools 清单里没有),互斥检查在 ffmpeg
            # 探测之后,必须 stub 成可用才能测到互斥分支本身。
            available = True
            ffmpeg = "ffmpeg"
            ffprobe = "ffprobe"

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(model_path="fake-model", runs_dir=tmpdir, max_new_tokens=8)
            mgr = app.state.manager
            job, _ = mgr.create_job_for_upload("video.mp4")
            job.segments_path.write_text("[]", encoding="utf-8")
            # clip 名经 _safe_clip_name 安全化后点号变下划线: clip_1.00_5.00 → clip_1_00_5_00
            mgr._rendering_clips.add(f"{job.id}/clip_1_00_5_00")
            with (
                patch("moss_transcribe_diarize.app.jobs.detect_ffmpeg", return_value=Available()),
                self.assertRaises(RuntimeError) as ctx,
            ):
                mgr.render_clip(job.id, start=1.0, end=5.0)
            self.assertIn("正在渲染", str(ctx.exception))


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


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class EventsCancelSearchApiTest(unittest.TestCase):
    """SSE 事件流、取消与查找替换的 API 冒烟测试。

    注意必须用 `with TestClient(app)` 进入上下文，lifespan startup 才会
    执行（把事件循环与 SSE hub 绑进 JobManager），否则事件推不出来。
    """

    def _make_app_and_client(self, runner):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = create_app(
            model_path="fake-model",
            runs_dir=tmp.name,
            max_new_tokens=8,
        )
        app.state.manager.model_runner = runner
        return app, TestClient(app)

    def _create_done_job(self, client):
        created = client.post(
            "/api/jobs",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
        )
        self.assertEqual(created.status_code, 200)
        job_id = created.json()["id"]
        for _ in range(60):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in {
                "queued", "downloading", "loading_model", "transcribing", "postprocessing", "labeling_speakers"
            }:
                break
            time.sleep(0.05)
        return job

    def test_startup_binds_event_loop_and_hub(self):
        """lifespan startup 必须把事件循环与 SSE hub 绑进 JobManager。

        starlette TestClient 会缓冲完整响应体，无法直接测无限 SSE 流，
        这里验证等价的接线：hub 绑定成功 + hub 发布/订阅可用。
        """
        runner = FakeRunner()
        app, client = self._make_app_and_client(runner)
        with client:
            manager = app.state.manager
            self.assertIsNotNone(manager._event_loop)
            self.assertIsNotNone(manager._event_hub)

            async def scenario():
                hub = manager._event_hub
                queue = hub.subscribe("job:x")
                hub.publish("job:x", "job", {"id": "x"})
                item = await asyncio.wait_for(queue.get(), timeout=1)
                self.assertEqual(item["data"], {"id": "x"})

            asyncio.run(scenario())

    def test_cancel_rejects_finished_job(self):
        runner = FakeRunner()
        app, client = self._make_app_and_client(runner)
        with client:
            job = self._create_done_job(client)
            self.assertEqual(client.post(f"/api/jobs/{job['id']}/cancel").status_code, 409)

    def test_cancel_running_job_eventually_marks_cancelled(self):
        runner = BlockingRunner()
        app, client = self._make_app_and_client(runner)
        with client:
            created = client.post(
                "/api/jobs",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
            )
            job_id = created.json()["id"]
            self.assertTrue(runner.started.wait(timeout=2))

            cancelled = client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(cancelled.status_code, 200)
            runner.release.set()
            status = ""
            for _ in range(80):
                status = client.get(f"/api/jobs/{job_id}").json()["status"]
                if status == "cancelled":
                    break
                time.sleep(0.05)
            self.assertEqual(status, "cancelled")
            # 任务与产物保留：还能查到任务详情
            self.assertEqual(client.get(f"/api/jobs/{job_id}").status_code, 200)

    def test_search_and_replace_endpoints(self):
        runner = FakeRunner()
        app, client = self._make_app_and_client(runner)
        with client:
            job = self._create_done_job(client)
            job_id = job["id"]

            found = client.get(f"/api/jobs/{job_id}/search", params={"q": "hello"})
            self.assertEqual(found.status_code, 200)
            self.assertEqual(found.json()["total"], 1)

            replaced = client.post(
                f"/api/jobs/{job_id}/replace",
                json={"query": "hello", "replacement": "world", "mode": "literal"},
            )
            self.assertEqual(replaced.status_code, 200)
            self.assertEqual(replaced.json()["replacements"], 1)
            segments = client.get(f"/api/jobs/{job_id}/segments").json()["segments"]
            self.assertEqual([s["text"] for s in segments], ["world"])


class SegmentsCacheApiTest(unittest.TestCase):
    """GET /segments 的 ETag/304 短路与解析缓存（长视频 2s 轮询卡顿优化的回归测试）。

    行为约定：
    - segments.json 未变（stat 指纹一致）→ If-None-Match 命中 304 空响应；
    - 外部编辑 srt 触发反向同步重写 segments.json → 缓存与 ETag 都必须失效；
    - PUT 保存重写 segments.json → 旧 ETag 失配，必须返回 200 全量。
    """

    def _make_app_and_client(self):
        from fastapi.testclient import TestClient

        from moss_transcribe_diarize.app.server import create_app

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = create_app(
            model_path="fake-model",
            runs_dir=tmp.name,
            max_new_tokens=8,
        )
        app.state.manager.model_runner = FakeRunner()
        self._runs_dir = Path(tmp.name)
        return app, TestClient(app)

    def _create_done_job(self, client):
        created = client.post(
            "/api/jobs",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
        )
        self.assertEqual(created.status_code, 200)
        job_id = created.json()["id"]
        for _ in range(60):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in {
                "queued", "downloading", "loading_model", "transcribing", "postprocessing", "labeling_speakers"
            }:
                break
            time.sleep(0.05)
        return job

    def test_segments_etag_304_short_circuit(self):
        app, client = self._make_app_and_client()
        with client:
            job = self._create_done_job(client)
            url = f"/api/jobs/{job['id']}/segments"

            first = client.get(url)
            self.assertEqual(first.status_code, 200)
            etag = first.headers.get("etag")
            self.assertTrue(etag, "GET /segments 必须带 ETag（stat 指纹）")
            self.assertTrue(first.json()["segments"])

            again = client.get(url, headers={"If-None-Match": etag})
            self.assertEqual(again.status_code, 304)
            self.assertEqual(again.content, b"")

            fresh = client.get(url)
            self.assertEqual(fresh.status_code, 200)

            # 缓存命中返回共享对象（文档化约束：调用方不得就地修改返回值）
            manager = app.state.manager
            self.assertIs(manager.list_segments(job["id"]), manager.list_segments(job["id"]))

    def test_external_srt_edit_invalidates_etag_and_cache(self):
        app, client = self._make_app_and_client()
        with client:
            job = self._create_done_job(client)
            url = f"/api/jobs/{job['id']}/segments"
            etag = client.get(url).headers["etag"]

            srt_path = self._runs_dir / job["id"] / "subtitle.srt"
            self.assertTrue(srt_path.exists())
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nchanged externally\n\n",
                encoding="utf-8",
            )

            after = client.get(url, headers={"If-None-Match": etag})
            self.assertEqual(after.status_code, 200, "外部编辑 srt 后不得再 304")
            self.assertEqual([s["text"] for s in after.json()["segments"]], ["changed externally"])
            self.assertNotEqual(after.headers.get("etag"), etag)

    def test_put_segments_updates_etag(self):
        app, client = self._make_app_and_client()
        with client:
            job = self._create_done_job(client)
            url = f"/api/jobs/{job['id']}/segments"
            first = client.get(url)
            etag = first.headers["etag"]

            segments = first.json()["segments"]
            segments[0]["text"] = "edited"
            put = client.put(url, json={"segments": segments})
            self.assertEqual(put.status_code, 200)

            refetched = client.get(url, headers={"If-None-Match": etag})
            self.assertEqual(refetched.status_code, 200, "PUT 保存后旧 ETag 必须失配")
            self.assertEqual([s["text"] for s in refetched.json()["segments"]], ["edited"])
            self.assertNotEqual(refetched.headers.get("etag"), etag)


if __name__ == "__main__":
    unittest.main()
