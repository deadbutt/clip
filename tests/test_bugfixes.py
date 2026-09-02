"""回归测试:worker 线程存活、下载兜底文件筛选、LLM key 剥离、
说话人簇时长、缺口恢复时间偏移、proofreader 容错与替换串转义、翻译重试分类。"""
from __future__ import annotations

import io
import json
import socket
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from moss_transcribe_diarize.app.llm_profiles import LlmProfileStore
from moss_transcribe_diarize.app.proofreader import Proofreader
from moss_transcribe_diarize.app.speaker_labeler import _merge_turn_clusters, _real_speaker_clusters
from moss_transcribe_diarize.app.text_translator import (
    TextTranslator,
    _RetryableTranslationError,
)
from moss_transcribe_diarize.app.whisper_runner import _shift_part_times
from moss_transcribe_diarize.subtitle import SubtitleItem, SubtitleSegment


class _StubRunner:
    model_path = "fake-model"


class WorkerThreadTest(unittest.TestCase):
    def test_worker_survives_deleted_queued_job(self):
        """删除排队中的任务后,worker 必须继续处理后续任务而不是线程死亡。"""
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sample.wav"
            source.write_bytes(b"audio")
            manager = JobManager(
                Path(tmpdir),
                _StubRunner(),
                prompt="p",
                max_length=1024,
                max_new_tokens=8,
            )
            processed: list[str] = []
            with patch.object(manager, "_process_job", side_effect=lambda job: processed.append(job.id)):
                created = manager.create_job_from_file(str(source), "sample.wav")
                job = created[0] if isinstance(created, tuple) else created
                # 队列里混入一个已被删除的任务 id(模拟 delete_job 与取队列的竞态)
                manager._queue.put("deleted-but-queued")
                deadline = time.time() + 5
                while time.time() < deadline:
                    if processed and manager._queue.empty():
                        break
                    time.sleep(0.02)

            self.assertEqual(processed, [job.id])
            self.assertTrue(manager._worker.is_alive())

    def test_worker_survives_process_job_crash(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JobManager(
                Path(tmpdir),
                _StubRunner(),
                prompt="p",
                max_length=1024,
                max_new_tokens=8,
            )

            def boom(_job):
                raise RuntimeError("transcription exploded")

            with patch.object(manager, "_process_job", side_effect=boom):
                with patch.object(manager, "get_job", return_value=object()):
                    manager._queue.put("crash-job")
                    deadline = time.time() + 5
                    while time.time() < deadline and not manager._queue.empty():
                        time.sleep(0.02)
                    manager._queue.join()

            self.assertTrue(manager._worker.is_alive())


class DownloaderFallbackTest(unittest.TestCase):
    def test_pick_media_file_skips_subtitles(self):
        from moss_transcribe_diarize.app.downloader import _pick_media_file

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            video = out_dir / "input.mp4"
            video.write_bytes(b"video")
            subtitle = out_dir / "input.en.srt"
            subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
            # 字幕通常更晚写出(mtime 更新);兜底也绝不能选中它
            old, new = time.time() - 60, time.time()
            import os

            os.utime(video, (old, old))
            os.utime(subtitle, (new, new))

            picked = _pick_media_file(out_dir)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.suffix, ".mp4")

    def test_pick_media_file_returns_none_without_media(self):
        from moss_transcribe_diarize.app.downloader import _pick_media_file

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "input.en.srt").write_text("x", encoding="utf-8")
            self.assertIsNone(_pick_media_file(out_dir))


class AlignItemsToTextTest(unittest.TestCase):
    """text 编辑后 items 的词级 diff 演化: 改字继承时间戳、加词插值、删词丢弃、重写降级 None。"""

    ITEMS = [
        SubtitleItem(text="I", start=0.0, end=0.3),
        SubtitleItem(text="heard", start=0.3, end=0.7),
        SubtitleItem(text="Nero", start=0.7, end=1.1),
        SubtitleItem(text="sleep", start=1.1, end=1.6),
        SubtitleItem(text="today", start=1.6, end=2.1),
    ]
    OLD_TEXT = "I heard Nero sleep today"

    def test_renamed_word_inherits_timestamp(self):
        from moss_transcribe_diarize.app.jobs import align_items_to_text

        out = align_items_to_text(self.OLD_TEXT, self.ITEMS, "I heard Neuro sleep today")
        self.assertEqual([i.text for i in out], ["I", "heard", "Neuro", "sleep", "today"])
        self.assertEqual(out[2].start, 0.7)
        self.assertEqual(out[2].end, 1.1)

    def test_inserted_word_interpolated_between_neighbours(self):
        from moss_transcribe_diarize.app.jobs import align_items_to_text

        out = align_items_to_text(self.OLD_TEXT, self.ITEMS, "I heard that Nero sleep today")
        words = [i.text for i in out]
        self.assertEqual(words.index("that"), 2)
        that = out[2]
        self.assertGreaterEqual(that.start, 0.7 - 1e-6)
        self.assertLessEqual(that.end, 1.1 + 1e-6)

    def test_deleted_word_dropped(self):
        from moss_transcribe_diarize.app.jobs import align_items_to_text

        out = align_items_to_text(self.OLD_TEXT, self.ITEMS, "I heard Nero today")
        self.assertEqual([i.text for i in out], ["I", "heard", "Nero", "today"])

    def test_full_rewrite_returns_none(self):
        from moss_transcribe_diarize.app.jobs import align_items_to_text

        self.assertIsNone(align_items_to_text(self.OLD_TEXT, self.ITEMS, "totally different words here now ok"))

    def test_unchanged_text_round_trips(self):
        from moss_transcribe_diarize.app.jobs import align_items_to_text

        out = align_items_to_text(self.OLD_TEXT, self.ITEMS, self.OLD_TEXT)
        self.assertEqual([(i.text, i.start, i.end) for i in out], [(i.text, i.start, i.end) for i in self.ITEMS])


class SplitPreservesEditedTextTest(unittest.TestCase):
    """拆分必须保留用户编辑过的文本,绝不从旧 items 重建回退。"""

    ITEMS = [
        SubtitleItem(text="I", start=0.0, end=0.5),
        SubtitleItem(text="heard", start=0.5, end=1.0),
        SubtitleItem(text="Neuro", start=1.0, end=1.5),
        SubtitleItem(text="sleep", start=1.5, end=2.0),
    ]

    def test_split_edited_segment_keeps_text(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        # text 已把 "Nero" 改成 "Neuro" 而 items 仍旧(对齐失败/绕过保存的兜底场景)。
        seg = SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S01",
                              text="I heard Neuro sleep", items=list(self.ITEMS))
        left, right = JobManager._split_record_at([seg], 0, 1.2, "seg_0002")
        combined = (left.text + " " + right.text).strip()
        self.assertIn("Neuro", combined)
        self.assertNotIn("Nero", combined)
        # 时间轴仍按最近词边界切(t=1.2 -> Neuro 的 1.0 边界,Neuro 归右半)。
        self.assertAlmostEqual(left.end, 1.0, places=6)
        self.assertAlmostEqual(right.start, 1.0, places=6)
        self.assertEqual(right.text, "Neuro sleep")

    def test_split_unedited_segment_rebuilds_from_items(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        seg = SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S01",
                              text="I heard Neuro sleep", items=list(self.ITEMS))
        # items 同步为编辑后的词,未再编辑 -> 精确分支。
        seg = SubtitleSegment(id=seg.id, start=seg.start, end=seg.end, speaker=seg.speaker,
                              text=seg.text, items=[
                                  SubtitleItem(text="I", start=0.0, end=0.5),
                                  SubtitleItem(text="heard", start=0.5, end=1.0),
                                  SubtitleItem(text="Neuro", start=1.0, end=1.5),
                                  SubtitleItem(text="sleep", start=1.5, end=2.0),
                              ])
        left, right = JobManager._split_record_at([seg], 0, 1.2, "seg_0002")
        self.assertEqual(left.text, "I heard")
        self.assertEqual(right.text, "Neuro sleep")


class UpdateSegmentsAlignsItemsTest(unittest.TestCase):
    """整包保存(payload 不带 items)时,被编辑 text 的段会从磁盘旧 items 演化出对齐 items。"""

    def test_backfill_aligns_items_to_edited_text(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "a.wav"
            source.write_bytes(b"audio")
            manager = JobManager(Path(tmpdir), _StubRunner(), prompt="p", max_length=64, max_new_tokens=8)
            created = manager.create_job_from_file(str(source), "a.wav")
            job = created[0] if isinstance(created, tuple) else created
            manager._write_subtitle_files(job, [
                SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S01",
                                text="I heard Nero sleep",
                                items=[
                                    SubtitleItem(text="I", start=0.0, end=0.5),
                                    SubtitleItem(text="heard", start=0.5, end=1.0),
                                    SubtitleItem(text="Nero", start=1.0, end=1.5),
                                    SubtitleItem(text="sleep", start=1.5, end=2.0),
                                ]),
            ])
            updated = manager.update_segments(job.id, [
                {"id": "seg_0001", "start": 0.0, "end": 2.0, "speaker": "S01", "text": "I heard Neuro sleep"},
            ])
            items = updated[0].get("items") or []
            self.assertEqual([i["text"] for i in items], ["I", "heard", "Neuro", "sleep"])
            self.assertEqual(items[2]["start"], 1.0)

    def test_backfill_preserves_mismatched_items_for_translated_job(self):
        """已翻译任务: 中文 text + 英文 items 固有错配,编辑中文文本时 items 必须原样保留而非清空。"""
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "a.wav"
            source.write_bytes(b"audio")
            manager = JobManager(Path(tmpdir), _StubRunner(), prompt="p", max_length=64, max_new_tokens=8)
            created = manager.create_job_from_file(str(source), "a.wav")
            job = created[0] if isinstance(created, tuple) else created
            # 磁盘: 翻译后的段(中文文本 + 英文词级 items)
            manager._write_subtitle_files(job, [
                SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S01",
                                text="你好世界",
                                items=[
                                    SubtitleItem(text="hello", start=0.0, end=1.0),
                                    SubtitleItem(text="world", start=1.0, end=2.0),
                                ]),
            ])
            updated = manager.update_segments(job.id, [
                {"id": "seg_0001", "start": 0.0, "end": 2.0, "speaker": "S01", "text": "你好世界啊"},
            ])
            items = updated[0].get("items") or []
            self.assertEqual([i["text"] for i in items], ["hello", "world"])


class ProofreadAppliesItemsAlignmentTest(unittest.TestCase):
    """校对应用替换文本时同步演化 items,后续拆分仍有词级时间轴。"""

    def test_proofread_replacement_aligns_items(self):
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "a.wav"
            source.write_bytes(b"audio")
            manager = JobManager(Path(tmpdir), _StubRunner(), prompt="p", max_length=64, max_new_tokens=8)
            created = manager.create_job_from_file(str(source), "a.wav")
            job = created[0] if isinstance(created, tuple) else created
            manager._write_subtitle_files(job, [
                SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S01",
                                text="I heard Nero sleep",
                                items=[
                                    SubtitleItem(text="I", start=0.0, end=0.5),
                                    SubtitleItem(text="heard", start=0.5, end=1.0),
                                    SubtitleItem(text="Nero", start=1.0, end=1.5),
                                    SubtitleItem(text="sleep", start=1.5, end=2.0),
                                ]),
            ])
            job.proofread_path.parent.mkdir(parents=True, exist_ok=True)
            job.proofread_path.write_text(json.dumps({
                "suggestions": [
                    {"id": "seg_0001", "original": "I heard Nero sleep", "corrected": "I heard Neuro sleep",
                     "reason": "rename"},
                ],
            }, ensure_ascii=False), encoding="utf-8")
            manager._set_status(job, "waiting_review", 1.0, error=None)
            manager.apply_proofread(job.id, ["seg_0001"], [])
            segments = manager._load_segments_records(job)
            items = segments[0].items or []
            self.assertEqual(segments[0].text, "I heard Neuro sleep")
            self.assertEqual([i.text for i in items], ["I", "heard", "Neuro", "sleep"])
            self.assertEqual(items[2].start, 1.0)

    def test_proofread_on_mismatched_items_keeps_them(self):
        """已翻译任务校对(中文 text + 英文 items): 替换中文文本时 items 原样保留。"""
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "a.wav"
            source.write_bytes(b"audio")
            manager = JobManager(Path(tmpdir), _StubRunner(), prompt="p", max_length=64, max_new_tokens=8)
            created = manager.create_job_from_file(str(source), "a.wav")
            job = created[0] if isinstance(created, tuple) else created
            manager._write_subtitle_files(job, [
                SubtitleSegment(id="seg_0001", start=0.0, end=2.0, speaker="S01",
                                text="你好世界",
                                items=[
                                    SubtitleItem(text="hello", start=0.0, end=1.0),
                                    SubtitleItem(text="world", start=1.0, end=2.0),
                                ]),
            ])
            job.proofread_path.parent.mkdir(parents=True, exist_ok=True)
            job.proofread_path.write_text(json.dumps({
                "suggestions": [
                    {"id": "seg_0001", "original": "你好世界", "corrected": "你好，世界",
                     "reason": "punct"},
                ],
            }, ensure_ascii=False), encoding="utf-8")
            manager._set_status(job, "waiting_review", 1.0, error=None)
            manager.apply_proofread(job.id, ["seg_0001"], [])
            segments = manager._load_segments_records(job)
            items = segments[0].items or []
            self.assertEqual(segments[0].text, "你好，世界")
            self.assertEqual([i.text for i in items], ["hello", "world"])


class CookiesTempCleanupTest(unittest.TestCase):
    """URL 任务下载结束后必须删除上传的 .cookies.txt 临时文件;用户自备路径不删。"""

    def _make_url_job(self, manager, tmpdir: Path, cookies_file: str) -> object:
        job = manager.create_job_for_url("https://example.com/v", cookies_file=cookies_file)
        manager._set_status(job, "queued", 0.0, error=None)
        return job

    def test_download_phase_deletes_uploaded_cookies_temp(self):
        from moss_transcribe_diarize.app.downloader import DownloadResult
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manager = JobManager(tmpdir, _StubRunner(), prompt="p", max_length=64, max_new_tokens=8)
            cookies = tmpdir / "tmpabc123.cookies.txt"
            cookies.write_text("# netscape cookies", encoding="utf-8")
            job = self._make_url_job(manager, tmpdir, str(cookies))

            media = tmpdir / "job-media" / "input.mkv"
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(b"video")

            with patch(
                "moss_transcribe_diarize.app.downloader.download_with_yt_dlp",
                side_effect=lambda *a, **k: DownloadResult(path=media),
            ):
                manager._download_phase(job)
            self.assertFalse(cookies.exists(), "下载完成后 cookies 临时文件应被删除")

    def test_download_phase_keeps_user_provided_cookies_file(self):
        from moss_transcribe_diarize.app.downloader import DownloadResult
        from moss_transcribe_diarize.app.jobs import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manager = JobManager(tmpdir, _StubRunner(), prompt="p", max_length=64, max_new_tokens=8)
            # 用户自备路径(非 .cookies.txt 后缀)不能删。
            cookies = tmpdir / "my-cookies.txt"
            cookies.write_text("# netscape cookies", encoding="utf-8")
            job = self._make_url_job(manager, tmpdir, str(cookies))

            media = tmpdir / "job-media" / "input.mkv"
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(b"video")

            with patch(
                "moss_transcribe_diarize.app.downloader.download_with_yt_dlp",
                side_effect=lambda *a, **k: DownloadResult(path=media),
            ):
                manager._download_phase(job)
            self.assertTrue(cookies.exists(), "用户自备 cookies 文件不应被删除")


class LlmProfileSecretTest(unittest.TestCase):
    def test_responses_never_carry_plaintext_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LlmProfileStore(tmpdir)
            added = store.add_profile(
                {"name": "p1", "base_url": "https://api.example.com", "api_key": "sk-secret-value", "model": "m1"}
            )

            self.assertIsNone(added["api_key"])
            self.assertEqual(added["api_key_masked"], mask_of("sk-secret-value"))

            listed = store.list_profiles()
            item = listed["profiles"][0]
            self.assertNotIn("api_key", item)
            self.assertEqual(item["api_key_masked"], mask_of("sk-secret-value"))

            updated = store.update_profile(item["id"], {"name": "renamed"})
            self.assertIsNone(updated["api_key"])

            # 后端内部流程(test connection)显式取明文仍然可用
            internal = store.list_profiles(include_secrets=True)
            self.assertEqual(internal["profiles"][0]["api_key"], "sk-secret-value")


def mask_of(key: str) -> str:
    from moss_transcribe_diarize.app.llm_profiles import mask_api_key

    return mask_api_key(key)


class MergeTurnClustersTest(unittest.TestCase):
    def test_uses_real_duration_not_absolute_end(self):
        # S01 说了 200s(0-200),S02 只说了 5s(500-505):
        # 按真实时长,合并方向必须是 S02 → S01;按绝对 end 则会反过来。
        turns = [(0.0, 200.0, "S01"), (500.0, 505.0, "S02")]
        merged = _merge_turn_clusters(turns, embeddings=None, target=1)

        self.assertEqual({speaker for _, _, speaker in merged}, {"S01"})


class RealSpeakerClustersTest(unittest.TestCase):
    """杂簇收编判定:占比门槛 + 连续长 turn 双判据。"""

    def test_short_lived_real_speaker_with_long_turn_survives(self):
        # 复刻 d5363c 案例:第三说话人后半段登场,占比 9%(<10%)
        # 但有 5.2s 连续长 turn——是真人,必须保留。
        turns = [(0.0, 200.0, "S01"), (50.0, 93.7, "S02")]
        turns += [(176.8, 178.6, "S03"), (190.0, 195.2, "S03"), (210.0, 212.4, "S03")]
        kept = _real_speaker_clusters(turns)
        self.assertEqual(set(kept), {"S01", "S02", "S03"})

    def test_fragment_ghost_cluster_still_merged(self):
        # 幽灵簇(TTS/噪声):占比 <10% 且全是 1~2s 碎片,不保留。
        turns = [(0.0, 200.0, "S01"), (50.0, 93.7, "S02")]
        turns += [(176.8, 178.2, "G"), (190.0, 191.5, "G"), (210.0, 211.3, "G")]
        kept = _real_speaker_clusters(turns)
        self.assertEqual(set(kept), {"S01", "S02"})

    def test_dominant_share_always_kept(self):
        turns = [(0.0, 60.0, "S01"), (60.0, 100.0, "S02")]
        kept = _real_speaker_clusters(turns)
        self.assertEqual(set(kept), {"S01", "S02"})


class ShiftPartTimesTest(unittest.TestCase):
    def test_shifts_head_and_tail_timestamps(self):
        self.assertEqual(_shift_part_times("[0.50][S00]hi[1.20]", 30.0), "[30.50][S00]hi[31.20]")

    def test_text_with_brackets_untouched(self):
        self.assertEqual(_shift_part_times("[2.00][S00][Music] la[3.00]", 10.0), "[12.00][S00][Music] la[13.00]")


class ProofreaderResilienceTest(unittest.TestCase):
    def test_pass2_failure_keeps_pass1_results(self):
        class _Pass2Explodes(Proofreader):
            def _run_pass1(self, items, progress_callback=None):
                return {"seg_0001": "hello!"}, 0

            def _run_pass2(self, items):
                raise RuntimeError("context length exceeded")

        reader = _Pass2Explodes(base_url="http://127.0.0.1:1", model="m")
        segments = [SubtitleSegment("seg_0001", 0.0, 1.0, "S01", "hello")]
        result = reader.proofread(segments)

        self.assertEqual(result["suggestions"][0]["corrected"], "hello!")
        self.assertEqual(result["term_corrections"], [])
        self.assertEqual(result["reference"]["merge_suggestions"], [])
        self.assertEqual(result["reference"]["speaker_questions"], [])

    def test_apply_terms_escapes_replacement_string(self):
        reader = Proofreader(base_url="http://127.0.0.1:1", model="m")
        segments = [SubtitleSegment("seg_0001", 0.0, 1.0, "S01", "go to bad now")]
        results = reader._apply_terms(segments, [{"wrong": "bad", "right": "C:\\new\\dir"}])

        self.assertEqual(results[0]["hits"], 1)
        self.assertEqual(results[0]["previews"][0]["corrected"], "go to C:\\new\\dir now")


class PostJsonRetryClassificationTest(unittest.TestCase):
    def _post_with(self, exc: Exception):
        translator = TextTranslator(base_url="http://127.0.0.1:1/v1", model="m", timeout=1)
        with patch(
            "moss_transcribe_diarize.app.text_translator.urllib.request.urlopen",
            side_effect=exc,
        ):
            return translator._post_json("http://127.0.0.1:1/v1/chat/completions", {})

    def test_http_500_is_retryable(self):
        err = urllib.error.HTTPError("http://x", 500, "boom", None, io.BytesIO(b"overloaded"))
        with self.assertRaises(_RetryableTranslationError):
            self._post_with(err)

    def test_http_401_fails_fast(self):
        err = urllib.error.HTTPError("http://x", 401, "unauthorized", None, io.BytesIO(b"bad key"))
        with self.assertRaises(RuntimeError) as ctx:
            self._post_with(err)
        self.assertNotIsInstance(ctx.exception, _RetryableTranslationError)

    def test_connection_refused_fails_fast(self):

        with self.assertRaises(RuntimeError):
            self._post_with(urllib.error.URLError(ConnectionRefusedError()))

    def test_timeout_is_retryable(self):
        with self.assertRaises(_RetryableTranslationError):
            self._post_with(socket.timeout("timed out"))


class SplitMergeSegmentsTest(unittest.TestCase):
    """拆分/合并 + 整包保存时词级 items 回填。"""

    SEGMENTS = [
        {
            "id": "seg_0001", "start": 0.0, "end": 4.0, "speaker": "S01",
            "text": "hello world foo bar",
            "items": [
                {"text": "hello", "start": 0.0, "end": 1.0},
                {"text": "world", "start": 1.0, "end": 2.0},
                {"text": "foo", "start": 2.0, "end": 3.0},
                {"text": "bar", "start": 3.0, "end": 4.0},
            ],
        },
        {
            "id": "seg_0002", "start": 4.5, "end": 6.0, "speaker": "S02",
            "text": "hi there", "items": None,
        },
    ]

    def _make_job_manager(self, tmpdir: str):
        from moss_transcribe_diarize.app.jobs import JobManager, JobRecord

        manager = JobManager(
            Path(tmpdir), _StubRunner(), prompt="p", max_length=1024, max_new_tokens=8
        )
        job_dir = Path(tmpdir) / "job1"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.wav").write_bytes(b"x")
        manager._jobs["job1"] = JobRecord(
            id="job1",
            status="waiting_review",
            progress=0.95,
            media_name="m.wav",
            input_path=str(job_dir / "input.wav"),
            job_dir=str(job_dir),
            inference_prompt="p",
            max_length=1024,
            max_new_tokens=8,
            decoding="greedy",
            temperature=None,
        )
        (job_dir / "segments.json").write_text(
            json.dumps(self.SEGMENTS, ensure_ascii=False), encoding="utf-8"
        )
        return manager

    def test_split_at_word_boundary(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_job_manager(tmpdir)
            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                out = manager.split_segment("job1", "seg_0001", 2.4)
            self.assertEqual(len(out), 3)
            left = out[0]
            right = out[1]
            self.assertEqual(left["id"], "seg_0001")
            self.assertNotEqual(right["id"], "seg_0001")
            self.assertEqual([i["text"] for i in left["items"]], ["hello", "world"])
            self.assertEqual([i["text"] for i in right["items"]], ["foo", "bar"])
            self.assertAlmostEqual(left["end"], 2.0)
            self.assertAlmostEqual(right["start"], 2.0)
            self.assertEqual(left["text"], "hello world")
            self.assertEqual(right["text"], "foo bar")

            # 非相邻合并拒绝(此时 [left, right, seg_0002],left 和 seg_0002 不相邻)
            with self.assertRaises(ValueError):
                manager.merge_segments("job1", [left["id"], "seg_0002"])

            # 合并回去:词数/时间复原
            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                merged = manager.merge_segments("job1", [left["id"], right["id"]])
            self.assertEqual(len(merged), 2)
            self.assertEqual(len(merged[0]["items"]), 4)
            self.assertAlmostEqual(merged[0]["start"], 0.0)
            self.assertAlmostEqual(merged[0]["end"], 4.0)
            self.assertEqual(merged[0]["text"], "hello world foo bar")

    def test_update_segments_backfills_items(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_job_manager(tmpdir)
            # 前端整包保存只带 5 个字段
            payload = [
                {k: s[k] for k in ("id", "start", "end", "speaker", "text")}
                for s in self.SEGMENTS
            ]
            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                out = manager.update_segments("job1", payload)
            self.assertEqual(len(out[0]["items"]), 4, "词级 items 应按 id 回填")
            self.assertIsNone(out[1]["items"], "原本没有 items 的段保持 None")
            # 回填的 items 被夹进(收窄后的)段边界
            out2 = manager.update_segments(
                "job1",
                [
                    {**payload[0], "start": 0.5, "end": 3.5},
                    payload[1],
                ],
            )
            for item in out2[0]["items"]:
                self.assertGreaterEqual(item["start"], 0.5)
                self.assertLessEqual(item["end"], 3.5)

    def test_split_without_items_falls_back_to_ratio(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_job_manager(tmpdir)
            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                out = manager.split_segment("job1", "seg_0002", 5.25)
            self.assertEqual(len(out), 3)
            left = out[1]
            right = out[2]
            self.assertAlmostEqual(left["end"], 5.25)
            self.assertAlmostEqual(right["start"], 5.25)
            self.assertAlmostEqual(right["end"], 6.0)
            self.assertIsNone(right["items"])

    def test_split_syncs_source_backup_for_translated_job(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_job_manager(tmpdir)
            job = manager.get_job("job1")
            # 模拟已翻译: 源稿备份 + 当前稿是中文
            job.source_segments_path.write_text(
                json.dumps(self.SEGMENTS, ensure_ascii=False), encoding="utf-8"
            )
            translated = [
                {
                    "id": "seg_0001", "start": 0.0, "end": 4.0, "speaker": "S01",
                    "text": "你好世界", "items": self.SEGMENTS[0]["items"],
                },
                self.SEGMENTS[1],
            ]
            (Path(tmpdir) / "job1" / "segments.json").write_text(
                json.dumps(translated, ensure_ascii=False), encoding="utf-8"
            )
            job.translation_info = {"applied": True}

            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                out = manager.split_segment("job1", "seg_0001", 2.4)
            self.assertEqual(len(out), 3)
            # 当前稿按英文词重建文本(退回原文待重译)
            self.assertEqual(out[0]["text"], "hello world")
            self.assertEqual(out[1]["text"], "foo bar")
            # 源稿同步拆分,新 id 一致
            source = json.loads(job.source_segments_path.read_text(encoding="utf-8"))
            self.assertEqual(len(source), 3)
            self.assertEqual(source[0]["id"], out[0]["id"])
            self.assertEqual(source[1]["id"], out[1]["id"])
            self.assertEqual([i["text"] for i in source[0]["items"]], ["hello", "world"])
            self.assertEqual([i["text"] for i in source[1]["items"]], ["foo", "bar"])
            # 标记结构变更
            self.assertTrue(manager.get_job("job1").translation_info.get("structure_changed"))

            # 合并: 源稿同步合并,结构复原
            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                merged = manager.merge_segments("job1", [out[0]["id"], out[1]["id"]])
            self.assertEqual(len(merged), 2)
            source = json.loads(job.source_segments_path.read_text(encoding="utf-8"))
            self.assertEqual(len(source), 2)
            self.assertEqual(len(source[0]["items"]), 4)

    def test_apply_proofread_keeps_items(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_job_manager(tmpdir)
            job = manager.get_job("job1")
            # 校对结果: 一条错字修正 + 一条术语
            (Path(tmpdir) / "job1" / "proofread.json").write_text(
                json.dumps({
                    "suggestions": [
                        {"id": "seg_0001", "original": "hello world foo bar", "corrected": "hello worlds foo bar"}
                    ],
                    "term_corrections": [{"wrong": "foo", "right": "FOO"}],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch("moss_transcribe_diarize.app.jobs.probe_video_size", return_value=(1920, 1080)):
                manager.apply_proofread("job1", ids=["seg_0001"], terms=[{"wrong": "foo", "right": "FOO"}])
            segments = json.loads(job.segments_path.read_text(encoding="utf-8"))
            self.assertEqual(segments[0]["text"], "hello worlds FOO bar")
            self.assertIsNotNone(segments[0]["items"], "应用校对后必须保留词级 items")
            self.assertEqual(len(segments[0]["items"]), 4)


if __name__ == "__main__":
    unittest.main()


class DownloaderCancelTest(unittest.TestCase):
    def test_cancel_fires_during_progress_lines(self):
        """取消检查必须在 [download] 进度行内生效,而不是等下载结束。"""

        from moss_transcribe_diarize.app import downloader

        lines = iter(
            [
                "mtd_title:Some Video",
                "[download]   1.0% of  100.00MiB at    1.00MiB/s ETA 01:39",
                "[download]   2.0% of  100.00MiB at    1.00MiB/s ETA 01:38",
                "[download]   3.0% of  100.00MiB at    1.00MiB/s ETA 01:37",
            ]
        )

        class FakeProc:
            stdout = lines
            pid = 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

        terminated = {"flag": False}

        def do_terminate(self):
            terminated["flag"] = True

        FakeProc.terminate = do_terminate

        with patch.object(downloader, "find_yt_dlp", return_value=Path("C:/fake/yt-dlp.exe")), patch.object(
            downloader.subprocess, "Popen", return_value=FakeProc()
        ), patch.object(downloader, "_find_js_runtime", return_value=None), patch.object(
            downloader, "_find_ffmpeg_dir", return_value=None
        ):
            with self.assertRaises(RuntimeError) as ctx:
                downloader.download_with_yt_dlp(
                    "https://example.com/v",
                    tempfile.mkdtemp(),
                    cookies_browser=None,
                    progress_callback=None,
                    cancel_check=lambda: True,
                )

        self.assertIn("取消", str(ctx.exception))
        self.assertTrue(terminated["flag"])

    def test_progress_lines_still_reported(self):
        from moss_transcribe_diarize.app import downloader

        lines = iter(
            [
                "mtd_title:Some Video",
                "[download]  50.0% of  100.00MiB at    1.00MiB/s ETA 00:50",
            ]
        )

        class FakeProc:
            stdout = lines

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        seen: list[float] = []
        with patch.object(downloader, "find_yt_dlp", return_value=Path("C:/fake/yt-dlp.exe")), patch.object(
            downloader.subprocess, "Popen", return_value=FakeProc()
        ), patch.object(downloader, "_find_js_runtime", return_value=None), patch.object(
            downloader, "_find_ffmpeg_dir", return_value=None
        ):
            # 取消永远为 False:验证进度行正常上报后走到兜底文件检查
            try:
                downloader.download_with_yt_dlp(
                    "https://example.com/v",
                    tempfile.mkdtemp(),
                    cookies_browser=None,
                    progress_callback=lambda ratio, info: seen.append(ratio),
                    cancel_check=lambda: False,
                )
            except (FileNotFoundError, RuntimeError):
                pass

        self.assertTrue(any(r > 0.4 for r in seen))
