from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from moss_transcribe_diarize.app import bilibili_publish
from moss_transcribe_diarize.app.bilibili_publish import BilibiliPublishService, CredentialStore


def fake_dpapi(data: bytes, *, protect: bool) -> bytes:
    return base64.b64encode(data) if protect else base64.b64decode(data)


class FakeJob:
    def __init__(self, root: Path):
        self.id = "job-test"
        self.status = "done"
        self.input_path = str(root / "source.mp4")
        self.job_dir = str(root / "runs" / self.id)
        self.thumbnail_path = str(root / "cover.jpg")
        self.publish_info: dict = {}
        self.output_path = Path(self.job_dir) / "output.mp4"


class FakeJobs:
    def __init__(self, root: Path):
        self.job = FakeJob(root)

    def get_job(self, job_id: str):
        if job_id != self.job.id:
            raise KeyError("Unknown job")
        return self.job

    def publish_draft(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if not job.publish_info:
            job.publish_info = {
                "title": "source", "description": "", "tags": [], "tid": 0,
                "copyright": "original", "source": "", "dynamic": "",
                "upload_variant": "original", "status": "draft",
            }
        return dict(job.publish_info)

    def update_publish_draft(self, job_id: str, payload: dict) -> dict:
        job = self.get_job(job_id)
        job.publish_info = {**self.publish_draft(job_id), **payload, "updated_at": time.time()}
        return dict(job.publish_info)

    def update_publish_status(self, job_id: str, patch: dict) -> dict:
        job = self.get_job(job_id)
        job.publish_info = {**job.publish_info, **patch}
        return dict(job.publish_info)

    def _ensure_job_thumbnail(self, _job):
        return None

    def list_jobs(self):
        return [self.job]


def make_service(tmp_path: Path, *, uploader=None, requester=None):
    root = tmp_path
    (root / "source.mp4").write_bytes(b"source")
    (root / "cover.jpg").write_bytes(b"fake image bytes")
    jobs = FakeJobs(root)
    store = CredentialStore(root / "config" / "bilibili.auth")
    with patch.object(bilibili_publish, "_dpapi", side_effect=fake_dpapi):
        store.save({"SESSDATA": "sess-secret", "bili_jct": "csrf-secret", "refresh_token": "refresh-secret"})
        service = BilibiliPublishService(root / "config", jobs, uploader=uploader, requester=requester)
    return service, jobs, store


def valid_payload(**updates):
    return {
        "title": "测试投稿", "description": "说明", "tags": ["转录"],
        "tid": 12, "copyright": "original", "source": "", "dynamic": "",
        "upload_variant": "original", **updates,
    }


def wait_for_publish(jobs: FakeJobs, status: str):
    deadline = time.time() + 3
    while time.time() < deadline:
        current = jobs.job.publish_info.get("status")
        if current == status:
            return dict(jobs.job.publish_info)
        time.sleep(0.01)
    raise AssertionError(f"投稿没有进入 {status}，当前信息：{jobs.job.publish_info}")


def test_qr_login_persists_credentials_without_returning_them(tmp_path):
    replies = [
        {"code": 0, "data": {"url": "https://passport.test/?SESSDATA=sess-secret&bili_jct=csrf-secret&DedeUserID=42", "qrcode_key": "qr-key"}},
        {"code": 0, "data": {"url": "https://passport.test/?SESSDATA=sess-secret&bili_jct=csrf-secret&DedeUserID=42", "refresh_token": "refresh-secret"}},
        {"code": 0, "data": {"isLogin": True, "uname": "测试账号"}},
    ]

    def requester(_method, _url, _body, _cookie):
        return replies.pop(0)

    jobs = FakeJobs(tmp_path)
    service = BilibiliPublishService(tmp_path / "config", jobs, requester=requester)

    class FakeQrImage:
        def add_data(self, _value):
            pass

        def make(self, fit):
            assert fit is True

        def make_image(self, **_kwargs):
            return self

        def save(self, stream, format):
            assert format == "PNG"
            stream.write(b"fake-png")

    fake_qrcode = SimpleNamespace(QRCode=lambda **_kwargs: FakeQrImage(), constants=SimpleNamespace(ERROR_CORRECT_M=1))
    with patch.dict(sys.modules, {"qrcode": fake_qrcode}), patch.object(
        bilibili_publish, "_dpapi", side_effect=fake_dpapi
    ):
        qr = service.start_login()
        result = service.poll_login(qr["session_id"])
    assert qr["qr_image"].startswith("data:image/png;base64,")
    assert "sess-secret" not in json.dumps(qr)

    assert result == {"state": "done", "account": "测试账号"}
    assert "sess-secret" not in json.dumps(jobs.job.publish_info)
    stored = (tmp_path / "config" / "bilibili.auth").read_bytes()
    assert b"sess-secret" not in stored
    assert b"refresh-secret" not in stored


def test_qr_login_wait_and_scan_states(tmp_path):
    replies = [{"code": 86101, "data": {}}, {"code": 86090, "data": {}}]
    service = BilibiliPublishService(tmp_path / "config", FakeJobs(tmp_path), requester=lambda *_: replies.pop(0))
    service._login_sessions["session"] = {"key": "key", "expires_at": time.time() + 60, "state": "waiting"}
    assert service.poll_login("session")["state"] == "waiting"
    assert service.poll_login("session")["state"] == "scanned"


def test_qr_login_accepts_unwrapped_passport_payload(tmp_path):
    replies = [
        {"code": 0, "url": "https://passport.test/?qrcode_key=qr-key", "qrcode_key": "qr-key"},
        {"code": 0, "url": "https://passport.test/?SESSDATA=sess-secret&bili_jct=csrf-secret",
         "refresh_token": "refresh-secret"},
        {"code": 0, "data": {"isLogin": True, "uname": "测试账号"}},
    ]

    def requester(_method, _url, _body, _cookie):
        return replies.pop(0)

    service = BilibiliPublishService(tmp_path / "config", FakeJobs(tmp_path), requester=requester)

    class FakeQrImage:
        def add_data(self, _value):
            pass

        def make(self, fit):
            assert fit is True

        def make_image(self, **_kwargs):
            return self

        def save(self, stream, format):
            assert format == "PNG"
            stream.write(b"fake-png")

    fake_qrcode = SimpleNamespace(QRCode=lambda **_kwargs: FakeQrImage(), constants=SimpleNamespace(ERROR_CORRECT_M=1))
    with patch.dict(sys.modules, {"qrcode": fake_qrcode}), patch.object(
        bilibili_publish, "_dpapi", side_effect=fake_dpapi
    ):
        qr = service.start_login()
        result = service.poll_login(qr["session_id"])
    assert result == {"state": "done", "account": "测试账号"}


def test_qr_poll_uses_status_code_inside_data(tmp_path):
    replies = [
        {"code": 0, "data": {"code": 86101, "message": "未扫码"}},
        {"code": 0, "data": {"code": 86090, "message": "已扫码"}},
    ]
    service = BilibiliPublishService(tmp_path / "config", FakeJobs(tmp_path), requester=lambda *_: replies.pop(0))
    service._login_sessions["session"] = {"key": "key", "cookies": {}, "expires_at": time.time() + 60, "state": "waiting"}
    assert service.poll_login("session")["state"] == "waiting"
    assert service.poll_login("session")["state"] == "scanned"


def test_successful_upload_persists_only_public_result_fields(tmp_path):
    seen = []

    def uploader(media, cover, draft, credentials, update):
        seen.append((media.name, cover.name, draft["title"], credentials["SESSDATA"]))
        update(0.5, "uploading")
        update(0.96, "submitting")
        return {"bvid": "BV1test000001", "aid": 1234, "url": "https://www.bilibili.com/video/BV1test000001"}

    service, jobs, _store = make_service(tmp_path, uploader=uploader)
    with patch.object(BilibiliPublishService, "categories", return_value=[{"tid": 12, "name": "知识 · 科普"}]):
        initial = service.submit(jobs.job.id, valid_payload())
    assert initial["status"] in {"uploading", "submitted"}
    final = wait_for_publish(jobs, "submitted")
    assert final["bvid"] == "BV1test000001"
    assert final["url"].endswith("BV1test000001")
    assert seen == [("source.mp4", "cover.jpg", "测试投稿", "sess-secret")]
    record = json.dumps(jobs.job.publish_info)
    assert "sess-secret" not in record
    assert "csrf-secret" not in record


def test_categories_are_leaf_upload_partitions(tmp_path):
    with patch.object(bilibili_publish.resources, "files") as files:
        files.return_value.joinpath.return_value.read_text.return_value = json.dumps({
            "tid_list": [
                {"id": 1, "name": "父级", "show": True, "children": [
                    {"id": 2, "name": "可投稿", "show": True},
                    {"id": 3, "name": "隐藏", "show": False},
                ]},
            ]
        })
        assert BilibiliPublishService.categories() == [{"tid": 2, "name": "父级 · 可投稿"}]


def test_uncertain_submit_failure_requires_explicit_confirmation(tmp_path):
    attempts = []

    def uploader(_media, _cover, _draft, _credentials, update):
        attempts.append(True)
        update(0.96, "submitting")
        if len(attempts) == 1:
            raise RuntimeError("connection dropped")
        return {"bvid": "BV1test000001", "aid": 1234, "url": "https://www.bilibili.com/video/BV1test000001"}

    service, jobs, _store = make_service(tmp_path, uploader=uploader)
    with patch.object(BilibiliPublishService, "categories", return_value=[{"tid": 12, "name": "知识 · 科普"}]):
        service.submit(jobs.job.id, valid_payload())
        failed = wait_for_publish(jobs, "failed")
        assert failed["phase"] == "submitting"
        with pytest.raises(RuntimeError, match="创作中心"):
            service.submit(jobs.job.id, valid_payload())
        service.submit(jobs.job.id, valid_payload(confirm_uncertain_retry=True))
    assert wait_for_publish(jobs, "submitted")["status"] == "submitted"


def test_startup_marks_active_upload_interrupted_instead_of_resubmitting(tmp_path):
    jobs = FakeJobs(tmp_path)
    jobs.job.publish_info = {"title": "保留稿件", "status": "uploading", "phase": "submitting", "progress": 0.96}
    with patch.object(bilibili_publish, "_dpapi", side_effect=fake_dpapi):
        store = CredentialStore(tmp_path / "config" / "bilibili.auth")
        store.save({"SESSDATA": "secret", "bili_jct": "csrf"})
    service = BilibiliPublishService(tmp_path / "config", jobs, requester=lambda *_: {})

    assert service is not None
    assert jobs.job.publish_info["status"] == "interrupted"
    assert jobs.job.publish_info["retry_requires_confirmation"] is True
    assert "创作中心" in jobs.job.publish_info["error"]
