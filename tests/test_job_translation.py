import tempfile
import types
import unittest
from pathlib import Path

from moss_transcribe_diarize.app.jobs import JobManager, JobRecord
from moss_transcribe_diarize.subtitle import SubtitleSegment, export_json


class FakeTranslator:
    model = "fake-qwen"

    def __init__(self):
        self.inputs = []

    def translate_segments(self, segments, *, target_language, progress_callback=None, batch_size=None):
        items = list(segments)
        self.inputs.append([item.text for item in items])
        if progress_callback is not None:
            progress_callback(len(items), len(items), 0, len(items))
        return ["你好" for _ in items]


class JobTranslationTest(unittest.TestCase):
    def test_retranslation_uses_source_backup_and_restore_recovers_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "job"
            job_dir.mkdir()
            manager = JobManager(
                root,
                types.SimpleNamespace(model_path="test"),
                prompt="",
                max_length=1,
                max_new_tokens=1,
            )
            job = JobRecord(
                id="job",
                status="waiting_review",
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
            source = [SubtitleSegment(id="one", start=1.0, end=2.0, speaker="S00", text="hello")]
            job.segments_path.write_text(export_json(source), encoding="utf-8")
            translator = FakeTranslator()

            manager.translate(job.id, translator, mode="replace")
            manager.translate(job.id, translator, mode="bilingual")
            restored = manager.restore_source_segments(job.id)

            self.assertEqual(translator.inputs, [["hello"], ["hello"]])
            self.assertTrue(job.source_segments_path.exists())
            self.assertEqual(restored["segments"][0]["text"], "hello")
            self.assertEqual(job.translation_info["applied"], False)


if __name__ == "__main__":
    unittest.main()
