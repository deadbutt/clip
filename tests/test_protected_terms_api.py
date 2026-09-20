import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from moss_transcribe_diarize.app.hy_mt_translator import HyMtTranslator
from moss_transcribe_diarize.app.server import create_app


class ProtectedTermsApiTest(unittest.TestCase):
    def test_persistence_clearing_and_hy_translation_receives_saved_terms(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder) / "runs"
            app = create_app(model_path="fake-model", runs_dir=runs)
            with TestClient(app) as client:
                response = client.put("/api/protected-terms", json={"terms": ["Neuro", "GeoGuessr", "Neuro"]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["terms"], ["Neuro", "GeoGuessr"])
                self.assertEqual(client.put("/api/protected-terms", json={"terms": "wrong"}).status_code, 400)
                job, _ = app.state.manager.create_job_for_upload("test.mp4")
                job.status = "waiting_review"
                job.segments_path.write_text(json.dumps([
                    {"id": "seg_1", "start": 0, "end": 2, "speaker": "S01", "text": "Neuro plays GeoGuessr."},
                ]), encoding="utf-8")
                with patch.object(HyMtTranslator, "translate_segments", return_value=["Neuro 玩 GeoGuessr。"]) as translate:
                    response = client.post(f"/api/jobs/{job.id}/translate", json={"engine": "hy-mt"})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(translate.call_args.kwargs["protected_terms"], ("Neuro", "GeoGuessr"))
                    response = client.post(f"/api/jobs/{job.id}/translate", json={"engine": "hy-mt", "protected_terms": ""})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(translate.call_args.kwargs["protected_terms"], ())
                self.assertEqual(client.put("/api/protected-terms", json={"terms": []}).status_code, 200)
            with TestClient(create_app(model_path="fake-model", runs_dir=runs)) as client:
                self.assertEqual(client.get("/api/protected-terms").json(), {"terms": []})


if __name__ == "__main__":
    unittest.main()
