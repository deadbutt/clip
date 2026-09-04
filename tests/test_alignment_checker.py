# -*- coding: utf-8 -*-
"""源文-译文对照检查(LLM 只读标注)的单元测试。

覆盖:LLM 返回的 issue 解析过滤、配对构建(双语首行/跳过未翻译对)、
manager.alignment_check 的状态流转与落盘。
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.proofreader import Proofreader


def _pair(index: int, source: str, translated: str) -> dict:
    return {
        "id": f"seg_{index:04d}",
        "index": index,
        "start": index * 10.0,
        "end": index * 10.0 + 5.0,
        "source_text": source,
        "translated_text": translated,
    }


class _StubRunner:
    model_path = "fake-model"


class AlignmentCheckerTest(unittest.TestCase):
    def _patched_chat(self, replies):
        """返回 (calls, fake_chat);fake_chat 首参是实例,适配 patch 后的绑定调用。"""
        lock = threading.Lock()
        calls: list[str] = []

        def fake_chat(_self, system, user, *, temperature=0.0):
            with lock:
                calls.append(user)
            return replies[min(len(calls), len(replies)) - 1]

        return calls, fake_chat

    def test_check_alignment_filters_and_enriches_issues(self):
        reply = json.dumps(
            {
                "issues": [
                    {"n": 1, "type": "mistranslation", "note": "把拒绝翻成了同意", "suggested": "我拒绝去。"},
                    {"n": 99, "type": "omission", "note": "越界序号应丢弃", "suggested": "x"},
                    {"n": 2, "type": "weird-type", "note": "未知类型应丢弃", "suggested": "x"},
                    {"n": 2, "type": "addition", "note": "多出了字幕里没有的内容", "suggested": "你好。"},
                    {"n": 3, "type": "omission"},
                    {"n": 4, "type": "terminology", "note": "建议等于原文应丢弃", "suggested": "原样"},
                ]
            },
            ensure_ascii=False,
        )
        calls, fake_chat = self._patched_chat([reply])
        proofreader = Proofreader(base_url="http://localhost:1", model="m")
        pairs = [
            _pair(0, "I refuse to go.", "我同意去。"),
            _pair(1, "Hello.", "你好，欢迎各位来到直播间。原样"),
        ]
        with patch.object(Proofreader, "_chat", new=fake_chat):
            issues = proofreader.check_alignment(pairs)
        self.assertEqual([issue["index"] for issue in issues], [0, 1])
        self.assertEqual(issues[0]["type"], "mistranslation")
        self.assertEqual(issues[1]["type"], "addition")
        self.assertEqual(issues[0]["id"], "seg_0000")
        self.assertEqual(issues[0]["source_text"], "I refuse to go.")
        self.assertEqual(issues[0]["translated_text"], "我同意去。")
        self.assertEqual(issues[0]["suggested"], "我拒绝去。")
        self.assertEqual(issues[1]["suggested"], "你好。")

    def test_check_alignment_batches_windows(self):
        calls, fake_chat = self._patched_chat([json.dumps({"issues": []})])
        proofreader = Proofreader(base_url="http://localhost:1", model="m")
        pairs = [_pair(i, f"English {i}.", f"中文 {i}。") for i in range(25)]
        with patch.object(Proofreader, "_chat", new=fake_chat):
            issues = proofreader.check_alignment(pairs)
        self.assertEqual(issues, [])
        self.assertEqual(len(calls), 3)

    def test_check_alignment_empty_source_is_skipped(self):
        calls, fake_chat = self._patched_chat(["{}"])
        proofreader = Proofreader(base_url="http://localhost:1", model="m")
        with patch.object(Proofreader, "_chat", new=fake_chat):
            issues = proofreader.check_alignment([_pair(0, "", "空源文不送检")])
        self.assertEqual(issues, [])
        self.assertEqual(calls, [])


class AlignmentManagerTest(unittest.TestCase):
    def _write_job(self, runs: Path, job_id: str) -> Path:
        job_dir = runs / job_id
        job_dir.mkdir(parents=True)
        payload = {
            "id": job_id,
            "status": "waiting_review",
            "media_name": "video.mp4",
            "input_path": str(job_dir / "input.mp4"),
            "job_dir": str(job_dir),
            "inference_prompt": "",
            "max_length": 1024,
            "max_new_tokens": 8,
            "decoding": "greedy",
            "temperature": None,
            "progress": 0.95,
        }
        (job_dir / "job.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return job_dir

    def _write_segments(self, job_dir: Path, name: str, segments: list[dict]) -> None:
        (job_dir / name).write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")

    def _manager(self, runs: Path):
        from moss_transcribe_diarize.app.jobs import JobManager

        with patch.object(JobManager, "_process_job"):
            return JobManager(runs, _StubRunner(), prompt="p", max_length=1024, max_new_tokens=8)

    def test_alignment_pairs_builds_from_bilingual_and_skips_untranslated(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "pair-job")
            self._write_segments(
                job_dir,
                "segments.source.json",
                [
                    {"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": "S01"},
                    {"id": "seg_0002", "start": 2.0, "end": 4.0, "text": "Thanks.", "speaker": "S01"},
                    {"id": "seg_0003", "start": 4.0, "end": 6.0, "text": "[Music]", "speaker": "S01"},
                ],
            )
            self._write_segments(
                job_dir,
                "segments.json",
                [
                    {"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "你好，世界。\nHello world.", "speaker": "S01"},
                    {"id": "seg_0002", "start": 2.0, "end": 4.0, "text": "Thanks.", "speaker": "S01"},
                    {"id": "seg_0003", "start": 4.0, "end": 6.0, "text": "[Music]", "speaker": "S01"},
                ],
            )
            manager = self._manager(runs)
            job = manager.get_job("pair-job")
            pairs = manager._alignment_pairs(job)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["translated_text"], "你好，世界。")
            self.assertEqual(pairs[0]["source_text"], "Hello world.")
            self.assertEqual(pairs[0]["index"], 0)

    def test_alignment_check_writes_result_and_restores_status(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        class _FakeChecker:
            def check_alignment(self, pairs, *, progress_callback=None):
                if progress_callback:
                    progress_callback(1, 1)
                return [
                    {
                        "id": pairs[0]["id"],
                        "index": pairs[0]["index"],
                        "start": pairs[0]["start"],
                        "end": pairs[0]["end"],
                        "type": "omission",
                        "note": "漏掉了后半句",
                        "suggested": "你好，世界，很高兴见到大家。",
                        "source_text": pairs[0]["source_text"],
                        "translated_text": pairs[0]["translated_text"],
                    }
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "align-job")
            self._write_segments(
                job_dir,
                "segments.source.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "Hello world again.", "speaker": "S01"}],
            )
            self._write_segments(
                job_dir,
                "segments.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "你好。", "speaker": "S01"}],
            )
            manager = self._manager(runs)
            result = manager.alignment_check("align-job", _FakeChecker())
            self.assertEqual(result["issue_count"], 1)
            self.assertEqual(result["pair_count"], 1)
            self.assertTrue((job_dir / "alignment.json").exists())
            job = manager.get_job("align-job")
            self.assertEqual(job.status, "waiting_review")
            self.assertFalse(job.alignment_info.get("in_progress"))
            self.assertEqual(job.alignment_info.get("issue_count"), 1)
            loaded = manager.get_alignment_result("align-job")
            self.assertEqual(loaded["issues"][0]["type"], "omission")

    def test_proofread_translated_job_runs_alignment_in_one_click(self):
        """一键校对:已翻译任务在校对源稿后自动追加译文对照检查,写 alignment.json。"""

        class _FakeProofreader:
            def proofread(self, segments, *, progress_callback=None):
                return {"suggestions": [], "term_corrections": [], "usage": {}}

            def check_alignment(self, pairs, *, progress_callback=None):
                if progress_callback:
                    progress_callback(1, 1)
                return [
                    {
                        "id": pairs[0]["id"],
                        "index": pairs[0]["index"],
                        "start": pairs[0]["start"],
                        "end": pairs[0]["end"],
                        "type": "omission",
                        "note": "漏掉了后半句",
                        "source_text": pairs[0]["source_text"],
                        "translated_text": pairs[0]["translated_text"],
                    }
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "oneclick-job")
            self._write_segments(
                job_dir,
                "segments.source.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "Hello world again.", "speaker": "S01"}],
            )
            self._write_segments(
                job_dir,
                "segments.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "你好。", "speaker": "S01"}],
            )
            manager = self._manager(runs)
            result = manager.proofread("oneclick-job", _FakeProofreader())
            self.assertEqual(result["target"], "source")
            self.assertIn("alignment", result)
            self.assertEqual(result["alignment"]["issue_count"], 1)
            self.assertEqual(result["alignment"]["pair_count"], 1)
            self.assertTrue((job_dir / "alignment.json").exists())
            job = manager.get_job("oneclick-job")
            self.assertEqual(job.status, "waiting_review")
            self.assertEqual(job.proofread_info.get("alignment_count"), 1)
            self.assertEqual(job.alignment_info.get("issue_count"), 1)

    def test_proofread_untranslated_job_skips_alignment(self):
        class _FakeProofreader:
            def proofread(self, segments, *, progress_callback=None):
                return {"suggestions": [], "term_corrections": [], "usage": {}}

            def check_alignment(self, pairs, *, progress_callback=None):  # 不应被调用
                raise AssertionError("alignment must not run for untranslated jobs")

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "plain-job")
            self._write_segments(
                job_dir,
                "segments.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "Hello.", "speaker": "S01"}],
            )
            manager = self._manager(runs)
            result = manager.proofread("plain-job", _FakeProofreader())
            self.assertNotIn("alignment", result)
            self.assertEqual(manager.get_job("plain-job").proofread_info.get("alignment_count"), 0)

    def test_apply_alignment_replaces_translation_line_only(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "apply-job")
            self._write_segments(
                job_dir,
                "segments.json",
                [
                    {"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "旧译文\nHello world.", "speaker": "S01"},
                    {"id": "seg_0002", "start": 2.0, "end": 4.0, "text": "不受影响的段落", "speaker": "S01"},
                ],
            )
            (job_dir / "alignment.json").write_text(
                json.dumps(
                    {
                        "issues": [
                            {"id": "seg_0001", "type": "mistranslation", "note": "n", "suggested": "新译文"},
                        ],
                        "issue_count": 1,
                        "pair_count": 2,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manager = self._manager(runs)
            result = manager.apply_alignment("apply-job", ["seg_0001"])
            self.assertEqual(result["applied_count"], 1)
            segments = json.loads((job_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(segments[0]["text"], "新译文\nHello world.")
            self.assertEqual(segments[1]["text"], "不受影响的段落")
            remaining = json.loads((job_dir / "alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(remaining["issue_count"], 0)
            self.assertEqual(remaining["issues"], [])
            self.assertEqual(remaining["applied_ids"], ["seg_0001"])

    def test_apply_alignment_skips_unknown_ids(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "apply-job2")
            self._write_segments(
                job_dir,
                "segments.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "旧译文", "speaker": "S01"}],
            )
            (job_dir / "alignment.json").write_text(
                json.dumps({"issues": [], "issue_count": 0, "pair_count": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            manager = self._manager(runs)
            result = manager.apply_alignment("apply-job2", ["seg_9999"])
            self.assertEqual(result["applied_count"], 0)

    def test_alignment_check_requires_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            job_dir = self._write_job(runs, "plain-job")
            self._write_segments(
                job_dir,
                "segments.json",
                [{"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "Hello.", "speaker": "S01"}],
            )
            manager = self._manager(runs)
            with self.assertRaises(RuntimeError):
                manager.alignment_check("plain-job", object())

    def test_alignment_check_rejects_running_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir)
            self._write_job(runs, "running-job")
            manager = self._manager(runs)
            job = manager.get_job("running-job")
            job.status = "translating"
            with self.assertRaises(RuntimeError):
                manager.alignment_check("running-job", object())


if __name__ == "__main__":
    unittest.main()
