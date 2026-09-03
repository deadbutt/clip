"""SSE 事件中心、任务取消、manager 级查找替换的回归测试。"""
from __future__ import annotations

import asyncio
import tempfile
import types
import unittest
from pathlib import Path

from moss_transcribe_diarize.app.events import EventHub
from moss_transcribe_diarize.app.jobs import RUNNING_STATES, JobManager, JobRecord
from moss_transcribe_diarize.subtitle import SubtitleSegment, export_json


class _StubRunner:
    model_path = "fake-model"


class _TestCaseBase(unittest.TestCase):
    """公共基类：统一管理临时目录与每任务日志句柄的关闭顺序。

    Windows 下 pipeline.log 的 FileHandler 未关闭会让 TemporaryDirectory
    清理报 PermissionError，所以先关句柄（后注册，先执行）再删目录。
    """

    def make_tmpdir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def make_manager(self, root: Path) -> JobManager:
        manager = JobManager(root, _StubRunner(), prompt="", max_length=1, max_new_tokens=1)

        def _close_all():
            for job_id in list(manager._job_loggers):
                manager._close_job_logger(job_id)

        self.addCleanup(_close_all)
        return manager

    def insert_job(self, manager: JobManager, job_id: str, job_dir: Path, status: str = "waiting_review") -> JobRecord:
        job = JobRecord(
            id=job_id,
            status=status,
            media_name="input.mp4",
            input_path=str(job_dir / "input.mp4"),
            job_dir=str(job_dir),
            inference_prompt="",
            max_length=1,
            max_new_tokens=1,
            decoding="greedy",
            temperature=None,
        )
        manager._jobs[job.id] = job
        return job


class EventHubTest(unittest.TestCase):
    def test_publish_delivers_to_subscribers(self):
        async def scenario():
            hub = EventHub()
            queue = hub.subscribe("job:x")
            hub.publish("job:x", "job", {"id": "x"})
            item = await asyncio.wait_for(queue.get(), timeout=1)
            self.assertEqual(item["event"], "job")
            self.assertEqual(item["data"], {"id": "x"})

        asyncio.run(scenario())

    def test_unsubscribe_stops_delivery_and_slow_consumer_drops_oldest(self):
        async def scenario():
            hub = EventHub()
            queue = hub.subscribe("ch")
            hub.unsubscribe("ch", queue)
            hub.publish("ch", "job", {"n": 1})
            with self.assertRaises(asyncio.QueueEmpty):
                queue.get_nowait()

            small = hub.subscribe("ch2")
            for i in range(600):  # 超过 _MAX_QUEUE，最旧事件应被丢弃
                hub.publish("ch2", "job", {"n": i})
            self.assertEqual(small.qsize(), 256)
            first = small.get_nowait()
            self.assertEqual(first["data"]["n"], 600 - 256)

        asyncio.run(scenario())


class CancelJobTest(_TestCaseBase):
    def test_cancel_running_job_marks_cancelled_and_keeps_artifacts(self):
        """用户取消运行中任务：落 cancelled 终态后任务与产物保留。"""
        root = self.make_tmpdir()
        job_dir = root / "job"
        job_dir.mkdir()
        manager = self.make_manager(root)
        job = self.insert_job(manager, "job", job_dir, status="transcribing")
        job.segments_path.write_text(
            export_json([SubtitleSegment(id="one", start=1.0, end=2.0, speaker="S00", text="hello")]),
            encoding="utf-8",
        )

        manager.cancel_job(job.id)
        self.assertIn(job.id, manager._cancelled_jobs)
        with self.assertRaises(BaseException):
            # 取消后任何状态写入都会被 _raise_if_cancelled 拦截
            manager._set_status(job, "transcribing", 0.5)
        manager._cancelled_jobs.discard(job.id)
        job.status = "cancelled"
        manager._touch(job, error=None)

        self.assertEqual(job.status, "cancelled")
        self.assertTrue(job.job_path.exists(), "取消后 job.json 必须保留")
        self.assertTrue(job.segments_path.exists(), "取消后字幕产物必须保留")
        self.assertNotIn(job.status, RUNNING_STATES)

    def test_cancel_non_running_job_raises(self):
        root = self.make_tmpdir()
        manager = self.make_manager(root)
        self.insert_job(manager, "job", root / "job")
        with self.assertRaises(RuntimeError):
            manager.cancel_job("job")

    def test_cancel_transcription_via_worker(self):
        """转录回调里触发取消：worker 应落 cancelled 且不写 failed。"""
        root = self.make_tmpdir()
        (root / "job").mkdir()
        manager = self.make_manager(root)
        job = self.insert_job(manager, "job", root / "job", status="queued")

        class CancelOnStartRunner:
            model_path = "fake-model"

            def transcribe(self, audio_path, status_callback=None, **kwargs):
                if status_callback is not None:
                    manager.cancel_job(job.id)
                    status_callback("transcribing", 0.1, None)
                return types.SimpleNamespace(text="", words=[], generated_tokens=0, prompt_len=0, elapsed_sec=0.0)

        manager.model_runner = CancelOnStartRunner()
        manager._process_job(job)

        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.error, None)
        self.assertNotIn(job.id, manager._cancelled_jobs)
        self.assertTrue(manager._worker.is_alive())


class ManagerSearchReplaceTest(_TestCaseBase):
    def test_search_and_replace_updates_segments_and_srt(self):
        root = self.make_tmpdir()
        job_dir = root / "job"
        job_dir.mkdir()
        manager = self.make_manager(root)
        self.insert_job(manager, "job", job_dir)
        segments = [
            SubtitleSegment(id="one", start=1.0, end=2.0, speaker="S00", text="hello NEURO"),
            SubtitleSegment(id="two", start=2.0, end=3.0, speaker="S00", text="welcome to 纽罗"),
        ]
        record = manager.get_job("job")
        record.segments_path.write_text(export_json(segments), encoding="utf-8")

        result = manager.search_segments("job", "neuro")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["matches"][0]["segment_id"], "one")

        homophone = manager.search_segments("job", "牛罗", mode="pinyin")
        self.assertEqual([m["segment_id"] for m in homophone["matches"]], ["two"])

        replaced = manager.replace_segments("job", "纽罗", "Neuro", mode="pinyin")
        self.assertEqual(replaced["replacements"], 1)
        self.assertEqual(replaced["segments"][1]["text"], "welcome to Neuro")
        srt_text = record.srt_path.read_text(encoding="utf-8-sig")
        self.assertIn("welcome to Neuro", srt_text)

        undo_like = manager.replace_segments("job", "Neuro", "", mode="literal")
        self.assertEqual(undo_like["segments"][1]["text"], "welcome to ")

    def test_replace_rejected_while_running(self):
        root = self.make_tmpdir()
        manager = self.make_manager(root)
        self.insert_job(manager, "job", root / "job", status="transcribing")
        with self.assertRaises(RuntimeError):
            manager.replace_segments("job", "a", "b")


if __name__ == "__main__":
    unittest.main()
