import json
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

from .ffmpeg import detect_ffmpeg
from .jobs import JobManager
from .model_runner import ModelRunner
from .vllm_runner import VllmRunner


def create_app(
    *,
    model_path: str | Path,
    runs_dir: str | Path = "runs",
    device: str = "auto",
    dtype: str = "bf16",
    prompt: str = DEFAULT_PROMPT,
    max_length: int = 131072,
    max_new_tokens: int = 8192,
    decoding: str = "greedy",
    temperature: float | None = None,
    backend: str = "hf",
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_api_key: str | None = None,
    vllm_timeout: float = 600.0,
):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    except ImportError as exc:
        raise RuntimeError("Install fastapi, uvicorn, and python-multipart to run the local web app.") from exc

    app = FastAPI(title="MOSS Subtitle Studio")
    if backend == "vllm":
        if not vllm_base_url:
            raise ValueError("--vllm-base-url is required when backend='vllm'.")
        runner = VllmRunner(
            base_url=vllm_base_url,
            model=vllm_model or str(model_path),
            api_key=vllm_api_key,
            timeout=vllm_timeout,
        )
    else:
        runner = ModelRunner(model_path, device=device, dtype=dtype)
    manager = JobManager(
        runs_dir,
        runner,
        prompt=prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        decoding=decoding,
        temperature=temperature,
    )
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/favicon.svg")
    def favicon():
        return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.get("/api/runtime")
    def runtime():
        return {
            "ffmpeg": detect_ffmpeg().to_dict(),
            "model": _runner_runtime_info(manager.model_runner),
            "inference": {
                "prompt": manager.prompt,
                "max_length": manager.max_length,
                "max_new_tokens": manager.max_new_tokens,
                "decoding": manager.decoding,
                "temperature": manager.temperature,
            },
        }

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [job.to_dict() for job in manager.list_jobs()]}

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
        max_len: int | None = Form(None),
        decoding: str | None = Form(None),
        temperature: float | None = Form(None),
    ):
        try:
            job, input_path = manager.create_job_for_upload(
                file.filename or "input.media",
                prompt=prompt,
                max_length=max_len,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            with input_path.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            manager.enqueue(job.id)
            return job.to_dict()
        except Exception as exc:
            manager._set_status(job, "failed", 1.0, error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return manager.get_job(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            manager.delete_job(job_id)
            return {"ok": True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun_job(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            job = manager.rerun_job(
                job_id,
                prompt=payload.get("prompt"),
                max_length=payload.get("max_len") or payload.get("max_length"),
                max_new_tokens=payload.get("max_new_tokens"),
                decoding=payload.get("decoding"),
                temperature=payload.get("temperature"),
            )
            return job.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Media file is missing.") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/media")
    def media(job_id: str):
        try:
            job = manager.get_job(job_id)
            path = Path(job.input_path)
            if not path.exists():
                raise FileNotFoundError(str(path))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Media file is missing.") from exc
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/segments")
    def get_segments(job_id: str):
        try:
            return {"segments": manager.list_segments(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/jobs/{job_id}/segments")
    async def update_segments(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
            style = payload.get("style") if isinstance(payload, dict) else None
            if not isinstance(segments, list):
                raise ValueError("Expected a JSON list or an object with a segments list.")
            return {"segments": manager.update_segments(job_id, segments, style)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/render")
    async def render(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            job = manager.render(job_id, payload.get("style") if isinstance(payload, dict) else None)
            return job.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, kind: str):
        try:
            path = manager.download_path(job_id, kind)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"File is not ready: {kind}") from exc
        return FileResponse(path, filename=path.name)

    return app


def _read_processor_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser() / "processor_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    keys = [
        "audio_tokens_per_second",
        "audio_merge_size",
        "time_marker_every_seconds",
        "enable_time_marker",
    ]
    return {key: data[key] for key in keys if key in data}


def _runner_runtime_info(runner) -> dict[str, Any]:
    if hasattr(runner, "runtime_info"):
        info = dict(runner.runtime_info())
    else:
        info = {
            "backend": "hf",
            "path": runner.model_path,
            "device": runner.device_name,
            "dtype": runner.dtype_name,
        }
    if info.get("backend") == "hf":
        info["processor"] = _read_processor_config(info.get("path") or "")
    else:
        info.setdefault("processor", {})
    return info


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#007d77"/>
  <rect x="11" y="12" width="42" height="40" rx="8" fill="#fffdfa"/>
  <rect x="16" y="17" width="32" height="19" rx="4" fill="#1d1f22"/>
  <path d="M29 22v9.5l8.5-4.75z" fill="#c94b35"/>
  <rect x="17" y="41" width="30" height="4" rx="2" fill="#007d77"/>
  <rect x="17" y="48" width="21" height="3.5" rx="1.75" fill="#6d6a63"/>
</svg>"""


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MOSS 字幕工作台</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg" />
  <style>
    :root {
      --bg: #f7f5f0;
      --panel: #ffffff;
      --line: #d8d3c7;
      --text: #1d1f22;
      --muted: #6d6a63;
      --teal: #007d77;
      --coral: #c94b35;
      --green: #2f7d4f;
      --sidebar-width: 300px;
      --sidebar-collapsed-width: 12px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: #fffdfa;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 720; }
    main {
      height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: minmax(0, 1fr);
      overflow: hidden;
    }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    input, select, button, textarea {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
    }
    input[type="file"], input[type="number"], input[type="text"], select {
      width: 100%;
      padding: 8px;
    }
    button {
      min-height: 36px;
      padding: 8px 12px;
      cursor: pointer;
      background: #fff;
    }
    button.primary { background: var(--teal); border-color: var(--teal); color: white; }
    button.warn { background: var(--coral); border-color: var(--coral); color: white; }
    button.ghost { background: transparent; }
    button.saved { color: var(--muted); background: #f6f3ec; }
    button.small {
      min-height: 28px;
      padding: 4px 8px;
      font-size: 12px;
    }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: white;
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.ok { color: var(--green); border-color: #9cc5aa; }
    .pill.bad { color: var(--coral); border-color: #d6a193; }
    .muted { color: var(--muted); font-size: 13px; }
    .meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .error { color: var(--coral); font-size: 13px; overflow-wrap: anywhere; }
    .warning { color: #9b5a00; font-size: 12px; overflow-wrap: anywhere; }
    .save-status {
      font-size: 12px;
      color: var(--muted);
      min-height: 18px;
      margin-top: 8px;
    }
    .save-status.dirty { color: #946b00; }
    .save-status.saving { color: var(--teal); }
    .save-status.saved { color: var(--green); }
    .save-status.error { color: var(--coral); }
    .is-hidden { display: none !important; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .task-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .task-title {
      font-weight: 700;
      font-size: 18px;
      line-height: 1.25;
      overflow-wrap: anywhere;
      margin-bottom: 8px;
    }
    .task-meta {
      margin-top: 4px;
      line-height: 1.35;
    }
    .task-notice {
      margin-top: 10px;
      line-height: 1.35;
    }
    .primary-action {
      width: 100%;
      min-height: 42px;
      margin-top: 12px;
      font-weight: 650;
    }
    .secondary-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 10px;
    }
    .secondary-actions > div {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      min-width: 0;
    }
    .secondary-actions button {
      min-height: 30px;
      padding: 5px 9px;
      font-size: 13px;
    }
    .secondary-actions .save-status {
      margin-top: 0;
      min-height: auto;
      white-space: nowrap;
    }
    .progress {
      height: 8px;
      background: #ebe6dc;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar { width: 0%; height: 100%; background: var(--teal); transition: width 160ms ease; }
    .sidebar {
      position: relative;
      border-right: 1px solid var(--line);
      background: #fbfaf7;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
      overflow: visible;
    }
    .sidebar-collapsed .sidebar {
      grid-template-rows: 1fr;
      border-right: 0;
      background: #f4f1e9;
    }
    .sidebar-body {
      min-height: 0;
      overflow: auto;
      display: flex;
      flex-direction: column;
    }
    .sidebar-body .task-list { overflow: visible; }
    .sidebar-head {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .sidebar-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .sidebar-title strong { font-size: 15px; }
    .sidebar-tools {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }
    .sidebar-primary {
      width: 100%;
      min-height: 34px;
    }
    .sidebar-toggle-zone {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: 24px;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: flex-end;
    }
    .sidebar-toggle-zone::before {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      right: 0;
      width: 1px;
      background: rgba(216, 211, 199, 0.8);
    }
    .sidebar-toggle-zone:hover::before,
    .sidebar-toggle-zone:focus-within::before {
      background: var(--teal);
    }
    .sidebar-toggle {
      width: 28px;
      height: 48px;
      min-height: 48px;
      padding: 0;
      border-radius: 999px;
      border-color: #cfc8ba;
      background: #fffdfa;
      color: var(--text);
      box-shadow: 0 4px 14px rgba(24, 25, 26, 0.12);
      opacity: 0;
      pointer-events: none;
      transform: translateX(10px);
      transition: opacity 120ms ease, transform 120ms ease, border-color 120ms ease;
    }
    .sidebar-toggle::before { content: "‹"; font-size: 18px; line-height: 1; }
    .sidebar-toggle-zone:hover .sidebar-toggle,
    .sidebar-toggle:focus-visible {
      opacity: 1;
      pointer-events: auto;
      transform: translateX(14px);
    }
    .sidebar-toggle:hover,
    .sidebar-toggle:focus-visible { border-color: var(--teal); }
    .sidebar-collapsed .sidebar-head,
    .sidebar-collapsed .task-list {
      display: none;
    }
    .sidebar-collapsed .sidebar-toggle-zone {
      left: 0;
      right: auto;
      width: 30px;
      justify-content: flex-start;
    }
    .sidebar-collapsed .sidebar-toggle-zone::before {
      right: auto;
      left: 0;
      background: #cfc8ba;
    }
    .sidebar-collapsed .sidebar-toggle::before { content: "›"; }
    .sidebar-collapsed .sidebar-toggle-zone:hover .sidebar-toggle,
    .sidebar-collapsed .sidebar-toggle:focus-visible {
      transform: translateX(8px);
    }
    .task-list {
      overflow: auto;
      padding: 8px;
    }
    .task-item {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 9px;
      background: transparent;
      cursor: pointer;
      text-align: left;
      margin-bottom: 6px;
    }
    .task-item:hover { background: #fffdfa; border-color: var(--line); }
    .task-item.active { background: white; border-color: #9cc5aa; }
    .task-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .task-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 640;
    }
    .task-id { margin-top: 4px; }
    .task-foot {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 8px;
    }
    .task-progress {
      flex: 1 1 auto;
      min-width: 54px;
    }
    .content {
      height: 100%;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }
    .view {
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    .center-view {
      height: 100%;
      display: grid;
      align-items: start;
      justify-items: center;
      padding: 28px;
      overflow: auto;
    }
    .import-panel,
    .process-panel {
      width: min(680px, 100%);
    }
    .view-title {
      margin: 0 0 16px;
      font-size: 22px;
      font-weight: 760;
    }
    textarea.prompt-input {
      width: 100%;
      min-height: 112px;
      resize: vertical;
      padding: 8px;
    }
    details.advanced {
      width: 100%;
      margin: 10px 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fffdfa;
    }
    details.advanced summary {
      min-height: 38px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      cursor: pointer;
      color: var(--text);
      font-weight: 650;
      font-size: 13px;
    }
    details.advanced summary::-webkit-details-marker { display: none; }
    details.advanced summary::marker { content: ""; }
    details.advanced summary::after {
      content: "+";
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      border: 1px solid var(--line);
      color: var(--teal);
      background: white;
      flex: 0 0 auto;
      font-weight: 800;
    }
    details.advanced[open] summary::after { content: "-"; }
    .advanced-title { display: block; }
    .advanced-hint {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .advanced-body {
      padding: 0 10px 10px;
      border-top: 1px solid var(--line);
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .process-panel .progress { margin: 16px 0 8px; }
    .task-view {
      height: 100%;
      overflow: auto;
      padding: 20px 28px 28px;
    }
    .task-view-inner {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      max-width: 1280px;
      margin: 0 auto;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }
    .upload-card { display: flex; flex-direction: column; gap: 10px; }
    .jobs-card { display: flex; flex-direction: column; min-height: 0; }
    .jobs-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .jobs-head strong { font-size: 15px; }
    .jobs-tools { display: inline-flex; align-items: center; gap: 8px; }
    .jobs-card .task-list {
      overflow: visible;
      padding: 6px;
    }
    .pending-list { display: flex; flex-direction: column; gap: 8px; }
    .pending-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fffdfa;
      padding: 8px 10px;
    }
    .pending-item-head { display: flex; align-items: center; gap: 8px; }
    .pending-item-name {
      flex: 1 1 0;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 600;
      font-size: 13px;
    }
    .pending-item-summary { color: var(--muted); font-size: 12px; }
    .pending-item-body { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
    .pending-item-body.is-hidden { display: none; }
    .pending-toggle {
      min-height: 26px;
      padding: 2px 8px;
      font-size: 12px;
      color: var(--muted);
      background: transparent;
    }
    .pending-remove {
      min-height: 26px;
      padding: 2px 8px;
      font-size: 14px;
      line-height: 1;
      color: var(--muted);
      background: transparent;
    }
    .workbench-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      flex: 0 0 auto;
      min-height: 40px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fffdfa;
    }
    .workbench-bar .bar-name {
      font-weight: 700;
      font-size: 14px;
      max-width: 30ch;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .workbench-bar .bar-spacer { flex: 1 1 auto; }
    .workbench-bar .bar-progress { width: 120px; margin: 0; }
    .workbench-bar .icon-btn {
      min-height: 34px;
      width: 38px;
      padding: 0;
      font-size: 18px;
      line-height: 1;
    }
    .workbench-bar .save-status { font-size: 12px; color: var(--muted); }
    .workbench-bar .render-progress-meta { font-size: 12px; color: var(--muted); }
    .settings-modal {
      position: fixed;
      inset: 0;
      z-index: 50;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(20, 22, 26, 0.5);
    }
    .settings-modal.is-hidden { display: none; }
    .settings-modal-card {
      width: min(560px, 100%);
      max-height: min(80vh, 720px);
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
      overflow: hidden;
    }
    .settings-modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      font-weight: 700;
    }
    .settings-modal-body {
      padding: 12px 16px 16px;
      overflow: auto;
    }
    .settings-modal-body .group:last-child { border-bottom: 0; margin-bottom: 0; padding-bottom: 0; }
    .workbench {
      height: 100%;
      padding: 12px 14px 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .editor-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      flex: 1 1 0;
      min-height: 0;
    }
    .preview-column {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 10px;
      min-width: 0;
      min-height: 0;
    }
    .video-shell {
      width: 100%;
      min-height: 0;
      background: transparent;
      overflow: hidden;
      display: flex;
      align-items: flex-start;
      justify-content: center;
    }
    .video-stage {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      max-height: 48vh;
      max-width: 100%;
      background: #111;
      border-radius: 6px;
      overflow: hidden;
      flex: 0 0 auto;
    }
    video {
      width: 100%;
      height: 100%;
      background: #111;
      display: block;
      object-fit: contain;
    }
    .preview-mask-video {
      position: absolute;
      inset: 0;
      display: none;
      pointer-events: none;
      z-index: 1;
      filter: blur(18px);
      transform: scale(1.02);
      transform-origin: center;
    }
    .preview-mask-video.visible { display: block; }
    .subtitle-overlay {
      position: absolute;
      left: 50%;
      bottom: 56px;
      width: max-content;
      max-width: none;
      transform: translateX(-50%);
      display: none;
      justify-content: center;
      pointer-events: none;
      text-align: center;
      color: white;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif;
      font-size: 48px;
      font-weight: 400;
      line-height: 1.448;
      white-space: pre;
      overflow-wrap: normal;
      word-break: normal;
      -webkit-text-stroke: 3px #000;
      paint-order: stroke fill;
      text-shadow: 0 2px 3px rgba(0, 0, 0, 0.65);
      z-index: 3;
    }
    .subtitle-overlay.visible { display: flex; }
    .source-mask-overlay {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 72px;
      display: none;
      pointer-events: none;
      background: rgba(0, 0, 0, 0.82);
      z-index: 2;
    }
    .source-mask-overlay.visible { display: block; }
    .timeline-panel {
      min-width: 0;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #181b20;
      overflow: hidden;
      color: #e9edf2;
      user-select: none;
      -webkit-user-select: none;
      display: flex;
      flex-direction: column;
    }
    .timeline-panel * {
      user-select: none;
      -webkit-user-select: none;
      -webkit-user-drag: none;
    }
    .timeline-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      height: 32px;
      padding: 0 10px;
      border-bottom: 1px solid #2b3038;
      color: #b7c0cc;
      font-size: 12px;
      font-weight: 650;
      background: #20242b;
    }
    .timeline-title {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .timeline-title::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--teal);
      box-shadow: 0 0 0 3px rgba(0, 125, 119, 0.18);
    }
    .timeline-actions {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .timeline-tool {
      height: 24px;
      padding: 0 9px;
      border-radius: 5px;
      border: 1px solid #3a424e;
      background: #2a3039;
      color: #dce3ec;
      font-size: 12px;
      font-weight: 650;
    }
    .timeline-tool:hover:not(:disabled) {
      border-color: #6adbd0;
      color: white;
    }
    .timeline-tool:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .timeline-scroll {
      position: relative;
      overflow-x: auto;
      overflow-y: auto;
      flex: 1 1 0;
      min-height: 0;
      cursor: pointer;
      touch-action: none;
      background:
        linear-gradient(to bottom, #181b20 0, #181b20 31px, #111419 31px, #111419 100%),
        repeating-linear-gradient(to right, rgba(255,255,255,0.06) 0 1px, transparent 1px 80px);
      scrollbar-gutter: stable;
    }
    .timeline-track {
      position: relative;
      min-width: 100%;
      min-height: 100%;
    }
    .timeline-ruler {
      position: absolute;
      inset: 0 0 auto 0;
      height: 32px;
      border-bottom: 1px solid #2b3038;
    }
    .timeline-tick {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      background: rgba(255, 255, 255, 0.18);
    }
    .timeline-tick.major {
      background: rgba(255, 255, 255, 0.38);
    }
    .timeline-tick span {
      position: absolute;
      top: 7px;
      left: 6px;
      color: #9aa5b3;
      font-size: 11px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .timeline-lane {
      position: absolute;
      left: 0;
      right: 0;
      top: 32px;
      min-height: 140px;
      background:
        repeating-linear-gradient(to bottom, rgba(255,255,255,0.04) 0 1px, transparent 1px 44px),
        linear-gradient(to bottom, #151922, #111419);
    }
    .timeline-lane::before {
      content: "SUB";
      position: sticky;
      left: 0;
      top: 0;
      z-index: 3;
      display: inline-flex;
      align-items: center;
      height: 100%;
      width: 46px;
      padding-left: 10px;
      color: #697586;
      font-size: 11px;
      font-weight: 800;
      background: linear-gradient(to right, #151922 0, #151922 34px, rgba(21,25,34,0) 100%);
      pointer-events: none;
    }
    .timeline-segment {
      position: absolute;
      top: 45px;
      height: 36px;
      min-width: 8px;
      border: 1px solid rgba(104, 224, 209, 0.45);
      border-radius: 6px;
      background: linear-gradient(180deg, #226d69, #174c49);
      color: #eefdfb;
      cursor: grab;
      overflow: hidden;
      padding: 4px 7px;
      font-size: 11px;
      line-height: 1.2;
      text-align: left;
      touch-action: none;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }
    .timeline-segment:active { cursor: grabbing; }
    .timeline-segment::before,
    .timeline-segment::after {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      width: 8px;
      cursor: ew-resize;
      z-index: 1;
    }
    .timeline-segment::before { left: 0; }
    .timeline-segment::after { right: 0; }
    .timeline-segment.dragging {
      cursor: grabbing;
      z-index: 6;
      transition: none;
      box-shadow: 0 0 0 2px rgba(133, 255, 243, 0.55), 0 10px 24px rgba(0, 0, 0, 0.42);
    }
    .timeline-segment:hover {
      border-color: #85fff3;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.28);
    }
    .timeline-segment.active {
      background: linear-gradient(180deg, #20a69d, #007d77);
      border-color: #b1fff8;
      color: white;
      transform: translateY(-2px);
      box-shadow: 0 0 0 2px rgba(133, 255, 243, 0.22), 0 8px 20px rgba(0, 0, 0, 0.32);
    }
    .timeline-segment-speaker {
      display: block;
      margin-bottom: 2px;
      color: #bffbf5;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .timeline-segment-text {
      display: -webkit-box;
      -webkit-line-clamp: 1;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .timeline-playhead {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #ff4f3e;
      pointer-events: auto;
      transform: translateX(-1px);
      z-index: 4;
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35), 0 0 16px rgba(255, 79, 62, 0.48);
      cursor: ew-resize;
    }
    .timeline-playhead::before {
      content: "";
      position: absolute;
      top: 0;
      left: 50%;
      width: 12px;
      height: 12px;
      border-radius: 0 0 4px 4px;
      background: #ff4f3e;
      transform: translateX(-50%);
      pointer-events: auto;
    }
    .timeline-guide {
      position: absolute;
      top: 32px;
      bottom: 0;
      width: 3px;
      display: none;
      pointer-events: none;
      z-index: 12;
      background: rgba(255, 255, 255, 0.96);
      transform: translateX(-1.5px);
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.65), 0 0 18px rgba(255, 255, 255, 0.95);
    }
    .timeline-guide.visible.snapped { display: block; }
    .timeline-guide.snapped {
      background: #ffffff;
    }
    .timeline-guide::before,
    .timeline-guide::after {
      content: "";
      position: absolute;
      left: 50%;
      width: 0;
      height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      transform: translateX(calc(-50% + var(--guide-label-offset, 0px)));
    }
    .timeline-guide::before {
      top: -1px;
      border-left-width: 7px;
      border-right-width: 7px;
      border-top: 11px solid #ffffff;
      border-bottom: 0;
      transform: translateX(-50%);
      filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.8));
    }
    .timeline-guide::after {
      top: 12px;
      width: auto;
      height: 16px;
      padding: 0 5px;
      border: 1px solid rgba(255, 255, 255, 0.8);
      border-radius: 3px;
      background: #ffffff;
      color: #111419;
      font-size: 10px;
      font-weight: 800;
      line-height: 15px;
      white-space: nowrap;
      content: attr(data-label);
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.55);
    }
    .table-wrap {
      overflow: auto;
      min-width: 0;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      scrollbar-gutter: stable;
    }
    .table-column {
      display: flex;
      flex-direction: column;
      min-height: 0;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      overflow: hidden;
    }
    .table-head {
      display: flex;
      align-items: center;
      gap: 10px;
      height: 32px;
      padding: 0 10px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      background: #fffdfa;
      flex: 0 0 auto;
    }
    .table-head .table-title {
      color: var(--text);
      font-weight: 700;
    }
    .table-column .table-wrap {
      flex: 1 1 0;
      min-height: 0;
      border: 0;
      border-radius: 0;
    }
    .subtitle-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
      font-size: 14px;
      line-height: 1.35;
    }
    .subtitle-table th,
    .subtitle-table td {
      border-bottom: 1px solid #ece7dc;
      padding: 4px 8px;
      vertical-align: middle;
    }
    .subtitle-table th {
      position: sticky;
      top: 0;
      z-index: 2;
      height: 30px;
      background: #fffdfa;
      box-shadow: 0 1px 0 var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-align: left;
    }
    .subtitle-table th.time { width: 76px; }
    .subtitle-table th.speaker { width: 68px; }
    .subtitle-table th.row-actions { width: 92px; text-align: center; }
    .subtitle-table tbody tr {
      cursor: pointer;
      background: #fff;
      transition: background 120ms ease;
    }
    .subtitle-table tbody tr:nth-child(even) { background: #fcfbf8; }
    .subtitle-table tbody tr:hover { background: #f4f1ea; }
    .subtitle-table tbody tr.active { background: #e1f1ee; }
    .subtitle-table tbody tr.active td {
      border-bottom-color: #a8d0ca;
      box-shadow: inset 3px 0 0 var(--teal);
    }
    .subtitle-table tbody tr.active td:first-child { box-shadow: inset 3px 0 0 var(--teal); }
    .subtitle-table tbody tr.active td:not(:first-child) { box-shadow: none; }
    .subtitle-table input,
    .subtitle-table textarea {
      width: 100%;
      min-width: 0;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      color: var(--text);
      font: inherit;
      line-height: 1.35;
      transition: border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }
    .subtitle-table input {
      height: 30px;
      padding: 4px 5px;
      font-variant-numeric: tabular-nums;
    }
    .subtitle-table input.start,
    .subtitle-table input.end,
    .subtitle-table input.speaker {
      color: #313438;
    }
    .subtitle-table input.start,
    .subtitle-table input.end {
      text-align: right;
    }
    .subtitle-table input.speaker {
      text-align: center;
      font-weight: 600;
    }
    .subtitle-table textarea {
      display: block;
      min-height: 30px;
      max-height: 48px;
      padding: 5px 6px;
      resize: none;
      overflow: hidden;
      white-space: pre-wrap;
    }
    .subtitle-table tr.active textarea,
    .subtitle-table textarea:focus {
      max-height: 112px;
    }
    .subtitle-table input:focus,
    .subtitle-table textarea:focus {
      outline: none;
      border-color: #86bcb5;
      background: #fff;
      box-shadow: 0 0 0 2px rgba(0, 125, 119, 0.12);
    }
    .subtitle-table input::-webkit-outer-spin-button,
    .subtitle-table input::-webkit-inner-spin-button {
      margin: 0;
      -webkit-appearance: none;
    }
    .subtitle-table input[type="number"] { -moz-appearance: textfield; }
    .segment-actions {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      width: 100%;
    }
    .segment-action {
      width: 26px;
      height: 24px;
      padding: 0;
      border-radius: 5px;
      border: 1px solid #d8d0c2;
      background: #fffdfa;
      color: var(--teal);
      font-size: 15px;
      font-weight: 800;
      line-height: 1;
    }
    .segment-action.delete-row {
      color: #b24132;
    }
    .segment-action:hover {
      border-color: #86bcb5;
      background: #eef7f5;
    }
    .render-progress-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin: 10px 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .render-progress-meta strong {
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }
    .group {
      border-bottom: 1px solid var(--line);
      padding: 0 0 16px;
      margin-bottom: 16px;
    }
    .group:last-child { border-bottom: 0; }
    .speaker-map {
      display: grid;
      gap: 8px;
    }
    .speaker-map-row {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
    }
    .speaker-tag {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .downloads a {
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 6px 10px;
      margin: 0 6px 6px 0;
      border-radius: 6px;
      color: var(--teal);
      background: white;
      border: 1px solid var(--line);
      text-decoration: none;
    }
    .export-status {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    @media (max-width: 900px) {
      body { overflow: auto; }
      main { height: auto; grid-template-columns: 1fr; }
      .view, .workbench { height: auto; }
      .editor-grid { grid-template-columns: 1fr; }
      .table-column { border: 0; }
      .task-view-inner { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MOSS 字幕工作台</h1>
    <span id="runtime" class="pill">检测中</span>
  </header>
  <main>
    <section class="content">
      <section id="importView" class="view task-view">
        <div class="task-view-inner">
          <div class="panel upload-card">
            <h2 id="importTitle" class="view-title">任务管理</h2>
            <label for="file">选择媒体文件（可多选）</label>
            <input id="file" type="file" accept="audio/*,video/*,.mp4,.mov,.mkv,.wav,.mp3,.m4a" multiple />
            <div id="rerunSource" class="meta" style="margin-top:8px"></div>
            <details class="advanced">
              <summary>
                <span>
                  <span class="advanced-title">默认推理参数</span>
                  <span class="advanced-hint">新文件继承此参数，可在下方逐个修改</span>
                </span>
              </summary>
              <div class="advanced-body">
                <label for="prompt" style="margin-top:10px">推理 Prompt</label>
                <textarea id="prompt" class="prompt-input"></textarea>
                <div class="row" style="margin-top:10px">
                  <div>
                    <label for="maxNewTokens">输出 tokens</label>
                    <input id="maxNewTokens" type="number" min="1" step="1" value="8192" title="留默认或留空时，服务端会按音频时长自动抬高以避免截断" />
                  </div>
                  <div>
                    <label for="maxLen">上下文上限</label>
                    <input id="maxLen" type="number" min="1" step="1" value="131072" />
                  </div>
                </div>
                <div class="row" style="margin-top:10px">
                  <div>
                    <label for="decoding">解码</label>
                    <select id="decoding"><option value="greedy">greedy</option><option value="sample">sample</option></select>
                  </div>
                  <div>
                    <label for="temperature">温度</label>
                    <input id="temperature" type="number" min="0.01" step="0.05" value="1.0" />
                  </div>
                </div>
                <div id="modelinfo" class="meta" style="margin-top:8px"></div>
              </div>
            </details>
            <div id="pendingList" class="pending-list"></div>
            <button id="upload" class="primary">全部开始转写</button>
            <div id="importError" class="error" style="margin-top:10px"></div>
          </div>
          <div class="panel jobs-card">
            <div class="jobs-head">
              <strong>任务列表</strong>
              <div class="jobs-tools">
                <span id="jobCount" class="meta">0 个任务</span>
                <button id="refreshJobs" class="small ghost">刷新</button>
                <button id="newTask" class="small primary">新建任务</button>
              </div>
            </div>
            <div id="jobList" class="task-list"></div>
          </div>
        </div>
      </section>
      <section id="processingView" class="view center-view is-hidden">
        <div class="process-panel">
          <h2 id="processTitle" class="view-title">转写中</h2>
          <div id="processName"></div>
          <div id="processMeta" class="meta" style="margin-top:8px"></div>
          <div class="progress"><div id="processBar" class="bar"></div></div>
          <div id="processError" class="error"></div>
          <div class="actions" style="margin-top:14px">
            <button id="deleteCurrent">删除任务</button>
            <button id="openNew">新建任务</button>
          </div>
        </div>
      </section>
      <section id="workbench" class="view workbench is-hidden">
        <div class="workbench-bar">
          <button id="backToTasks" class="ghost" type="button">← 任务</button>
          <span id="selectedName" class="bar-name"></span>
          <span id="taskStatus" class="pill">待校对</span>
          <span id="taskNotice" class="task-notice is-hidden"></span>
          <span id="renderProgressMeta" class="render-progress-meta is-hidden"><span>烧录进度</span><strong id="renderProgressText">0%</strong></span>
          <div id="renderProgress" class="progress bar-progress is-hidden"><div id="renderProgressBar" class="bar"></div></div>
          <div class="bar-spacer"></div>
          <span id="saveStatus" class="save-status saved">已保存</span>
          <button id="save" class="primary is-hidden">保存修改</button>
          <button id="render" class="warn" disabled>检测 FFmpeg...</button>
          <button id="openSettings" class="ghost icon-btn" type="button" title="设置" aria-label="设置">⚙</button>
        </div>
        <div class="editor-grid">
          <div class="preview-column">
            <div class="video-shell">
              <div id="videoStage" class="video-stage">
                <video id="preview" controls></video>
                <video id="maskPreviewVideo" class="preview-mask-video" muted playsinline></video>
                <div id="sourceMaskOverlay" class="source-mask-overlay"></div>
                <div id="subtitleOverlay" class="subtitle-overlay"></div>
              </div>
            </div>
            <div class="timeline-panel">
              <div class="timeline-head">
                <span class="timeline-title">字幕轨道</span>
                <div class="timeline-actions">
                  <button id="addSegment" class="timeline-tool" type="button">添加字幕</button>
                  <button id="deleteSegment" class="timeline-tool" type="button">删除当前</button>
                  <span id="timelineMeta">0 段</span>
                </div>
              </div>
              <div id="timelineScroll" class="timeline-scroll">
                <div id="timelineTrack" class="timeline-track">
                  <div id="timelineRuler" class="timeline-ruler"></div>
                  <div id="timelineLane" class="timeline-lane"></div>
                  <div id="timelinePlayhead" class="timeline-playhead"></div>
                  <div id="timelineGuide" class="timeline-guide"></div>
                </div>
              </div>
            </div>
          </div>
          <div class="table-column">
            <div class="table-head">
              <span class="table-title">字幕</span>
            </div>
            <div class="table-wrap">
              <table class="subtitle-table">
                <thead>
                  <tr>
                    <th class="time">开始</th>
                    <th class="time">结束</th>
                    <th class="speaker">说话人</th>
                    <th>字幕</th>
                    <th class="row-actions">操作</th>
                  </tr>
                </thead>
                <tbody id="segments"></tbody>
              </table>
            </div>
          </div>
        </div>
        <div id="settingsModal" class="settings-modal is-hidden">
          <div class="settings-modal-card">
            <div class="settings-modal-head">
              <strong>设置</strong>
              <button id="closeSettings" class="ghost small" type="button" aria-label="关闭">✕</button>
            </div>
            <div class="settings-modal-body">
              <div class="group">
                <label>任务信息</label>
                <div id="taskUsage" class="meta task-meta"></div>
                <div id="taskParams" class="meta task-meta"></div>
              </div>
              <div class="group">
                <label>说话人名称</label>
                <div id="speakerMap" class="speaker-map"></div>
              </div>
              <div class="group">
                <div class="row">
                  <div>
                    <label for="fontSize">字号</label>
                    <input id="fontSize" type="number" min="18" max="96" value="48" />
                  </div>
                  <div>
                    <label for="marginV">底边距</label>
                    <input id="marginV" type="number" min="12" max="220" value="56" />
                  </div>
                </div>
                <div class="row" style="margin-top:10px">
                  <div>
                    <label for="showSpeaker">说话人</label>
                    <select id="showSpeaker"><option value="true">显示</option><option value="false">隐藏</option></select>
                  </div>
                  <div>
                    <label for="speakerColors">颜色</label>
                    <select id="speakerColors"><option value="true">按说话人</option><option value="false">统一</option></select>
                  </div>
                </div>
              </div>
              <div class="group">
                <label for="maskEnabled">原字幕遮挡</label>
                <select id="maskEnabled">
                  <option value="false">关闭</option>
                  <option value="true">开启</option>
                </select>
                <div style="margin-top:10px">
                  <label for="maskMode">遮挡方式</label>
                  <select id="maskMode">
                    <option value="blur">模糊原字幕区域</option>
                    <option value="bar">黑底遮挡</option>
                  </select>
                </div>
                <div class="row" style="margin-top:10px">
                  <div>
                    <label for="maskHeight">遮挡高度</label>
                    <input id="maskHeight" type="number" min="12" max="360" value="120" />
                  </div>
                  <div>
                    <label for="maskMarginV">离底距离</label>
                    <input id="maskMarginV" type="number" min="0" max="360" value="0" />
                  </div>
                </div>
                <div class="row" style="margin-top:10px">
                  <div>
                    <label for="maskBlur">模糊强度</label>
                    <input id="maskBlur" type="number" min="1" max="80" value="24" />
                  </div>
                  <div>
                    <label for="maskOpacity">黑底透明度</label>
                    <input id="maskOpacity" type="number" min="0" max="1" step="0.05" value="0.82" />
                  </div>
                </div>
              </div>
              <div class="group">
                <label>输出</label>
                <div class="downloads" id="downloads"></div>
                <button id="exportFolder" class="ghost" type="button" style="margin-top:8px">选择文件夹保存</button>
                <div id="exportStatus" class="export-status"></div>
              </div>
              <div class="group">
                <button id="rerun" class="ghost">重新转写</button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </section>
  </main>
<script>
const RUNNING_STATES = new Set(['queued', 'loading_model', 'transcribing', 'postprocessing', 'rendering']);
const EDIT_STATES = new Set(['waiting_review', 'rendering', 'done']);
const TERMINAL_STATES = new Set(['waiting_review', 'done', 'failed', 'cancelled']);
const fileInput = document.querySelector('#file');
const importTitleEl = document.querySelector('#importTitle');
const rerunSourceEl = document.querySelector('#rerunSource');
const promptInput = document.querySelector('#prompt');
const advancedDetails = document.querySelector('.advanced');
const maxNewTokensInput = document.querySelector('#maxNewTokens');
const maxLenInput = document.querySelector('#maxLen');
const decodingSelect = document.querySelector('#decoding');
const temperatureInput = document.querySelector('#temperature');
const uploadBtn = document.querySelector('#upload');
const newTaskBtn = document.querySelector('#newTask');
const refreshJobsBtn = document.querySelector('#refreshJobs');
const deleteCurrentBtn = document.querySelector('#deleteCurrent');
const openNewBtn = document.querySelector('#openNew');
const backToTasksBtn = document.querySelector('#backToTasks');
const openSettingsBtn = document.querySelector('#openSettings');
const closeSettingsBtn = document.querySelector('#closeSettings');
const settingsModal = document.querySelector('#settingsModal');
const pendingListEl = document.querySelector('#pendingList');
const saveBtn = document.querySelector('#save');
const renderBtn = document.querySelector('#render');
const rerunBtn = document.querySelector('#rerun');
const addSegmentBtn = document.querySelector('#addSegment');
const deleteSegmentBtn = document.querySelector('#deleteSegment');
const exportFolderBtn = document.querySelector('#exportFolder');
const saveStatusEl = document.querySelector('#saveStatus');
const importView = document.querySelector('#importView');
const processingView = document.querySelector('#processingView');
const workbench = document.querySelector('#workbench');
const runtimeEl = document.querySelector('#runtime');
const jobListEl = document.querySelector('#jobList');
const jobCountEl = document.querySelector('#jobCount');
const importErrorEl = document.querySelector('#importError');
const processTitleEl = document.querySelector('#processTitle');
const processNameEl = document.querySelector('#processName');
const processMetaEl = document.querySelector('#processMeta');
const processBarEl = document.querySelector('#processBar');
const processErrorEl = document.querySelector('#processError');
const selectedNameEl = document.querySelector('#selectedName');
const taskStatusEl = document.querySelector('#taskStatus');
const taskUsageEl = document.querySelector('#taskUsage');
const taskParamsEl = document.querySelector('#taskParams');
const taskNoticeEl = document.querySelector('#taskNotice');
const renderProgressMetaEl = document.querySelector('#renderProgressMeta');
const renderProgressTextEl = document.querySelector('#renderProgressText');
const renderProgressEl = document.querySelector('#renderProgress');
const renderProgressBarEl = document.querySelector('#renderProgressBar');
const modelInfoEl = document.querySelector('#modelinfo');
const tbody = document.querySelector('#segments');
const speakerMapEl = document.querySelector('#speakerMap');
const videoStage = document.querySelector('#videoStage');
const videoShell = document.querySelector('.video-shell');
const preview = document.querySelector('#preview');
const maskPreviewVideo = document.querySelector('#maskPreviewVideo');
const sourceMaskOverlay = document.querySelector('#sourceMaskOverlay');
const subtitleOverlay = document.querySelector('#subtitleOverlay');
const timelineScroll = document.querySelector('#timelineScroll');
const timelineTrack = document.querySelector('#timelineTrack');
const timelineRuler = document.querySelector('#timelineRuler');
const timelineLane = document.querySelector('#timelineLane');
const timelinePlayhead = document.querySelector('#timelinePlayhead');
const timelineGuide = document.querySelector('#timelineGuide');
const timelineMeta = document.querySelector('#timelineMeta');
const downloads = document.querySelector('#downloads');
const exportStatusEl = document.querySelector('#exportStatus');
let jobs = [];
let currentJob = null;
let rerunDraftJob = null;
let pendingUploads = [];
let pendingIdCounter = 0;
let pollTimer = null;
let runtimeChecked = false;
let ffmpegAvailable = false;
let activeSegmentIndex = -1;
let assPlayRes = { x: 1920, y: 1080 };
let layoutFitFrame = 0;
let editorDirty = false;
let saveStatusTimer = 0;
let speakerNameMap = {};
let timelineDragging = false;
let currentPixelsPerSecond = 12;
let segmentDragState = null;
const SEGMENT_EDGE_PX = 8;
const SEGMENT_DRAG_THRESHOLD = 3;
const SNAP_PX = 18;
const assFontLineHeightFactor = 1.448;
const speakerPalette = ['#ffffff', '#ffe75b', '#8ff286', '#ffa7bb', '#ffd700', '#6bb5ff', '#db8eff', '#d8d8d8'];
const RENDER_PROGRESS_BASE = 0.95;
const RENDER_PROGRESS_SPAN = 0.049;

function apiUrl(path) {
  const clean = String(path).replace(/^[/]+/, '');
  const basePath = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
  return new URL(clean, window.location.origin + basePath).toString();
}

function setPreviewSource(src) {
  preview.src = src;
  maskPreviewVideo.src = src;
  maskPreviewVideo.load();
}

async function refreshRuntime() {
  try {
    const res = await fetch(apiUrl('api/runtime'), { cache: 'no-store' });
    if (!res.ok) throw new Error('runtime status ' + res.status);
    const data = await res.json();
    runtimeChecked = true;
    ffmpegAvailable = !!(data.ffmpeg && data.ffmpeg.available);
    runtimeEl.textContent = ffmpegAvailable ? 'FFmpeg 可用' : 'FFmpeg 缺失';
    runtimeEl.className = 'pill ' + (ffmpegAvailable ? 'ok' : 'bad');
    updateRenderAction(currentJob);
    applyInferenceDefaults(data.inference || {});
    renderModelInfo(data.model || {});
  } catch (err) {
    runtimeChecked = true;
    ffmpegAvailable = false;
    runtimeEl.textContent = 'API 连接失败';
    runtimeEl.className = 'pill bad';
    updateRenderAction(currentJob);
    importErrorEl.textContent = '无法连接 api/runtime，请确认页面来自 mtd-subtitle-web 服务。';
  }
}

function applyInferenceDefaults(defaults) {
  if (!promptInput.value && defaults.prompt) promptInput.value = defaults.prompt;
  if (defaults.max_new_tokens) maxNewTokensInput.value = defaults.max_new_tokens;
  if (defaults.max_length) maxLenInput.value = defaults.max_length;
  if (defaults.decoding) decodingSelect.value = defaults.decoding;
  if (defaults.temperature) temperatureInput.value = defaults.temperature;
  updateDecodingControls();
}

function renderModelInfo(model) {
  const parts = [];
  if (model.path) {
    const pathParts = String(model.path).split('/');
    parts.push(pathParts.slice(-2).join('/'));
  }
  if (model.device) parts.push(model.device);
  if (model.dtype) parts.push(model.dtype);
  const processor = model.processor || {};
  if (processor.time_marker_every_seconds) parts.push('time marker ' + processor.time_marker_every_seconds + 's');
  modelInfoEl.textContent = parts.join(' · ');
}

function updateDecodingControls() {
  temperatureInput.disabled = decodingSelect.value !== 'sample';
}

function scheduleLayoutFit() {
  if (layoutFitFrame) cancelAnimationFrame(layoutFitFrame);
  layoutFitFrame = requestAnimationFrame(() => {
    layoutFitFrame = 0;
    fitVideoStageToMedia();
  });
}

function openSettings() { settingsModal.classList.remove('is-hidden'); }
function closeSettings() { settingsModal.classList.add('is-hidden'); }

function setSaveState(state, message) {
  if (saveStatusTimer) {
    clearTimeout(saveStatusTimer);
    saveStatusTimer = 0;
  }
  saveStatusEl.className = 'save-status ' + state;
  saveStatusEl.textContent = message;
  const showButton = state === 'dirty' || state === 'saving' || state === 'error';
  saveBtn.classList.toggle('is-hidden', !showButton);
  saveBtn.classList.toggle('primary', showButton);
  saveBtn.classList.toggle('saved', false);
  saveBtn.disabled = state === 'saving' || !currentJob;
  if (state === 'dirty') saveBtn.textContent = '保存修改';
  else if (state === 'saving') saveBtn.textContent = '保存中...';
  else if (state === 'error') saveBtn.textContent = '重试保存';
  else saveBtn.textContent = '保存修改';
}

function setEditorDirty(dirty) {
  editorDirty = dirty;
  if (dirty) setSaveState('dirty', '有未保存修改');
  else setSaveState('saved', '已保存');
}

function markEditorDirty() {
  if (!currentJob) return;
  setEditorDirty(true);
}

decodingSelect.addEventListener('change', updateDecodingControls);

newTaskBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
openNewBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
refreshJobsBtn.addEventListener('click', () => refreshJobs());
backToTasksBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
openSettingsBtn.addEventListener('click', openSettings);
closeSettingsBtn.addEventListener('click', closeSettings);
settingsModal.addEventListener('click', (event) => {
  if (event.target === settingsModal) closeSettings();
});
deleteCurrentBtn.addEventListener('click', async () => {
  if (currentJob) await deleteJob(currentJob.id);
});

jobListEl.addEventListener('click', async (event) => {
  const deleteButton = event.target.closest('[data-delete-id]');
  if (deleteButton) {
    event.stopPropagation();
    await deleteJob(deleteButton.dataset.deleteId);
    return;
  }
  const item = event.target.closest('[data-job-id]');
  if (item) await selectJob(item.dataset.jobId);
});

function defaultPendingParams() {
  return {
    prompt: promptInput.value,
    maxNewTokens: maxNewTokensInput.value,
    maxLen: maxLenInput.value,
    decoding: decodingSelect.value,
    temperature: temperatureInput.value
  };
}

function pendingSummary(item) {
  return 'tokens ' + (item.maxNewTokens || '8192') + ' · ' + (item.decoding || 'greedy');
}

function renderPendingList() {
  if (!pendingUploads.length) {
    pendingListEl.innerHTML = '';
    return;
  }
  pendingListEl.innerHTML = pendingUploads.map((item) => `
    <div class="pending-item" data-id="${item.id}">
      <div class="pending-item-head">
        <span class="pending-item-name">${escapeHtml(item.file.name)}</span>
        <span class="pending-item-summary">${escapeHtml(pendingSummary(item))}</span>
        <button class="pending-toggle" type="button" data-action="toggle">参数</button>
        <button class="pending-remove" type="button" data-action="remove" title="移除">✕</button>
      </div>
      <div class="pending-item-body${item.expanded ? '' : ' is-hidden'}">
        <div>
          <label>推理 Prompt</label>
          <textarea class="pending-prompt" rows="2">${escapeHtml(item.prompt)}</textarea>
        </div>
        <div class="row">
          <div>
            <label>输出 tokens</label>
            <input class="pending-tokens" type="number" min="1" step="1" value="${escapeHtml(String(item.maxNewTokens || ''))}" />
          </div>
          <div>
            <label>上下文上限</label>
            <input class="pending-maxlen" type="number" min="1" step="1" value="${escapeHtml(String(item.maxLen || ''))}" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>解码</label>
            <select class="pending-decoding"><option value="greedy"${item.decoding === 'greedy' ? ' selected' : ''}>greedy</option><option value="sample"${item.decoding === 'sample' ? ' selected' : ''}>sample</option></select>
          </div>
          <div>
            <label>温度</label>
            <input class="pending-temp" type="number" min="0.01" step="0.05" value="${escapeHtml(String(item.temperature || ''))}" />
          </div>
        </div>
      </div>
    </div>`).join('');
}

function updatePendingSummary(id) {
  const item = pendingUploads.find((p) => p.id === id);
  if (!item) return;
  const row = pendingListEl.querySelector('.pending-item[data-id="' + id + '"] .pending-item-summary');
  if (row) row.textContent = pendingSummary(item);
}

function updateUploadBtnLabel() {
  if (rerunDraftJob) {
    uploadBtn.textContent = '开始重跑';
  } else if (pendingUploads.length) {
    uploadBtn.textContent = '开始转写 ' + pendingUploads.length + ' 个任务';
  } else {
    uploadBtn.textContent = '全部开始转写';
  }
}

function addPendingFiles(fileList) {
  for (const file of fileList) {
    pendingUploads.push(Object.assign({ id: ++pendingIdCounter, file, expanded: false }, defaultPendingParams()));
  }
  renderPendingList();
  updateUploadBtnLabel();
}

fileInput.addEventListener('change', () => {
  if (rerunDraftJob) resetImportMode();
  if (fileInput.files && fileInput.files.length) {
    addPendingFiles(fileInput.files);
  }
  fileInput.value = '';
});

pendingListEl.addEventListener('click', (event) => {
  const row = event.target.closest('.pending-item');
  if (!row) return;
  const id = Number(row.dataset.id);
  const action = event.target.dataset.action;
  if (action === 'toggle') {
    const item = pendingUploads.find((p) => p.id === id);
    if (item) {
      item.expanded = !item.expanded;
      const body = row.querySelector('.pending-item-body');
      if (body) body.classList.toggle('is-hidden', !item.expanded);
    }
  } else if (action === 'remove') {
    pendingUploads = pendingUploads.filter((p) => p.id !== id);
    renderPendingList();
    updateUploadBtnLabel();
  }
});

pendingListEl.addEventListener('input', (event) => {
  const row = event.target.closest('.pending-item');
  if (!row) return;
  const id = Number(row.dataset.id);
  const item = pendingUploads.find((p) => p.id === id);
  if (!item) return;
  if (event.target.classList.contains('pending-prompt')) item.prompt = event.target.value;
  else if (event.target.classList.contains('pending-tokens')) { item.maxNewTokens = event.target.value; updatePendingSummary(id); }
  else if (event.target.classList.contains('pending-maxlen')) item.maxLen = event.target.value;
  else if (event.target.classList.contains('pending-temp')) item.temperature = event.target.value;
});

pendingListEl.addEventListener('change', (event) => {
  const row = event.target.closest('.pending-item');
  if (!row) return;
  const id = Number(row.dataset.id);
  const item = pendingUploads.find((p) => p.id === id);
  if (!item) return;
  if (event.target.classList.contains('pending-decoding')) { item.decoding = event.target.value; updatePendingSummary(id); }
});

uploadBtn.addEventListener('click', async () => {
  if (rerunDraftJob) {
    await startRerunDraft();
    return;
  }
  if (!pendingUploads.length) return;
  uploadBtn.disabled = true;
  importErrorEl.textContent = '';
  const total = pendingUploads.length;
  let created = 0;
  for (const item of pendingUploads) {
    uploadBtn.textContent = '上传中 ' + (created + 1) + '/' + total;
    const form = new FormData();
    form.append('file', item.file);
    form.append('prompt', item.prompt);
    if (item.maxNewTokens) form.append('max_new_tokens', item.maxNewTokens);
    if (item.maxLen) form.append('max_len', item.maxLen);
    form.append('decoding', item.decoding);
    if (item.temperature) form.append('temperature', item.temperature);
    try {
      const res = await fetch(apiUrl('api/jobs'), { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) {
        importErrorEl.textContent = (data && data.detail) || '上传失败';
        break;
      }
      created += 1;
    } catch (err) {
      importErrorEl.textContent = '上传失败：' + (err && err.message ? err.message : String(err));
      break;
    }
  }
  pendingUploads = [];
  renderPendingList();
  uploadBtn.disabled = false;
  updateUploadBtnLabel();
  await refreshJobs({ keepSelection: true });
});

saveBtn.addEventListener('click', async () => {
  await saveSegments();
});

addSegmentBtn.addEventListener('click', addSegmentAtPlayhead);
deleteSegmentBtn.addEventListener('click', deleteActiveSegment);
exportFolderBtn.addEventListener('click', exportCurrentJobToFolder);

renderBtn.addEventListener('click', async () => {
  if (!currentJob || !ffmpegAvailable) return;
  const saved = await saveSegments();
  if (!saved) return;
  const style = collectSubtitleStyle();
  currentJob = { ...currentJob, status: 'rendering', progress: RENDER_PROGRESS_BASE, error: null };
  renderCurrentJob(currentJob, { skipSegments: true });
  const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/render`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style })
  });
  const data = await res.json();
  if (!res.ok) {
    currentJob = { ...currentJob, status: 'waiting_review', progress: 0.95, error: data.detail || '烧录失败' };
    renderCurrentJob(currentJob, { skipSegments: true });
  }
  else {
    currentJob = data;
    renderCurrentJob(data, { skipSegments: true });
    await refreshJobs({ keepSelection: true });
  }
});

rerunBtn.addEventListener('click', () => {
  if (currentJob) showRerunDraft(currentJob);
});

preview.addEventListener('timeupdate', syncActiveSegment);
preview.addEventListener('seeked', syncActiveSegment);
preview.addEventListener('play', syncMaskPreviewPlayback);
preview.addEventListener('pause', syncMaskPreviewPlayback);
preview.addEventListener('seeking', syncMaskPreviewTime);
preview.addEventListener('seeked', syncMaskPreviewTime);
preview.addEventListener('ratechange', syncMaskPreviewPlaybackRate);
preview.addEventListener('loadedmetadata', () => {
  fitVideoStageToMedia();
  renderTimeline(collectSegments());
  syncActiveSegment();
  syncMaskPreviewPlaybackRate();
});
timelineScroll.addEventListener('pointerdown', (event) => {
  if (!currentJob || event.target.closest('.timeline-segment')) return;
  event.preventDefault();
  timelineDragging = true;
  timelineScroll.setPointerCapture(event.pointerId);
  seekTimelineFromPointer(event);
});
timelineScroll.addEventListener('pointermove', (event) => {
  if (!timelineDragging) return;
  event.preventDefault();
  seekTimelineFromPointer(event);
});
timelineScroll.addEventListener('pointerup', (event) => {
  if (!timelineDragging) return;
  event.preventDefault();
  timelineDragging = false;
  hideTimelineGuide();
  try {
    timelineScroll.releasePointerCapture(event.pointerId);
  } catch (err) {}
});
timelineScroll.addEventListener('pointercancel', () => {
  timelineDragging = false;
  hideTimelineGuide();
});
timelineScroll.addEventListener('dragstart', (event) => event.preventDefault());
window.addEventListener('resize', () => {
  scheduleLayoutFit();
  renderTimeline(collectSegments());
});
if ('ResizeObserver' in window) {
  const layoutObserver = new ResizeObserver(scheduleLayoutFit);
  for (const element of [videoShell, document.querySelector('.content'), document.querySelector('.editor-grid')]) {
    if (element) layoutObserver.observe(element);
  }
}
tbody.addEventListener('input', (event) => {
  markEditorDirty();
  renderTimeline(collectSegments());
  if (event.target.classList.contains('text')) {
    const tr = event.target.closest('tr');
    resizeSegmentTextarea(event.target, tr && tr.classList.contains('active'));
  }
  if (event.target.classList.contains('start') || event.target.classList.contains('end')) syncActiveSegment();
  else {
    if (event.target.classList.contains('speaker')) renderSpeakerMap(collectSegments());
    updateSubtitlePreview();
  }
});
tbody.addEventListener('change', markEditorDirty);
tbody.addEventListener('click', (event) => {
  const addAboveButton = event.target.closest('.add-row-above');
  const addBelowButton = event.target.closest('.add-row-below');
  const deleteButton = event.target.closest('.delete-row');
  if (!addAboveButton && !addBelowButton && !deleteButton) return;
  event.preventDefault();
  event.stopPropagation();
  const tr = event.target.closest('tr');
  if (!tr) return;
  const index = Number(tr.dataset.index);
  if (addAboveButton) addSegmentAroundIndex(index, 'above');
  else if (addBelowButton) addSegmentAroundIndex(index, 'below');
  else deleteSegmentAtIndex(index);
});
speakerMapEl.addEventListener('input', () => {
  syncSpeakerNameInputs();
  markEditorDirty();
  updateSubtitlePreview();
});
tbody.addEventListener('focusin', (event) => {
  const tr = event.target.closest('tr');
  if (!tr) return;
  setActiveSegment(Number(tr.dataset.index), false);
  resizeSegmentRow(tr, true);
  updateSubtitlePreview();
});
for (const id of ['fontSize', 'marginV', 'showSpeaker', 'speakerColors', 'maskEnabled', 'maskMode', 'maskHeight', 'maskMarginV', 'maskBlur', 'maskOpacity']) {
  document.querySelector('#' + id).addEventListener('input', () => {
    markEditorDirty();
    updateSubtitlePreview();
  });
  document.querySelector('#' + id).addEventListener('change', () => {
    markEditorDirty();
    updateSubtitlePreview();
  });
}

async function refreshJobs(options = {}) {
  const res = await fetch(apiUrl('api/jobs'), { cache: 'no-store' });
  if (!res.ok) return;
  const data = await res.json();
  jobs = data.jobs || [];
  renderJobList();
  if (currentJob) {
    const fresh = jobs.find((job) => job.id === currentJob.id);
    if (fresh) {
      const wasEditable = EDIT_STATES.has(currentJob.status);
      currentJob = fresh;
      if (options.background && wasEditable && EDIT_STATES.has(fresh.status)) {
        updateEditorChrome(fresh);
      } else {
        renderCurrentJob(fresh, { skipSegments: options.skipSegments || editorDirty });
      }
    } else {
      currentJob = null;
      showImportView();
    }
  } else if (!options.keepSelection && jobs.length && options.selectLatest) {
    await selectJob(jobs[0].id);
  }
  ensurePolling();
}

function renderJobList() {
  jobCountEl.textContent = jobs.length + ' 个任务';
  if (!jobs.length) {
    jobListEl.innerHTML = '<div class="meta" style="padding:10px">还没有任务</div>';
    return;
  }
  jobListEl.innerHTML = jobs.map((job) => {
    const active = currentJob && currentJob.id === job.id ? ' active' : '';
    const canDelete = !RUNNING_STATES.has(job.status);
    const percent = Math.round((job.progress || 0) * 100);
    const warning = truncationWarning(job);
    return `
      <div class="task-item${active}" data-job-id="${escapeHtml(job.id)}">
        <div class="task-row">
          <div class="task-name">${escapeHtml(job.media_name || 'input.media')}</div>
          <span class="${statusClass(job.status)}">${statusLabel(job.status)}</span>
        </div>
        <div class="task-id meta">${escapeHtml(job.id)}</div>
        <div class="meta">${escapeHtml(tokenUsageSummary(job))}</div>
        ${warning ? `<div class="warning">${escapeHtml(warning)}</div>` : ''}
        <div class="task-foot">
          <div class="progress task-progress"><div class="bar" style="width:${percent}%"></div></div>
          ${canDelete ? `<button class="small ghost" data-delete-id="${escapeHtml(job.id)}">删除</button>` : ''}
        </div>
      </div>`;
  }).join('');
}

async function selectJob(jobId) {
  const local = jobs.find((job) => job.id === jobId);
  currentJob = local || currentJob;
  renderJobList();
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { cache: 'no-store' });
  if (!res.ok) {
    await refreshJobs();
    return;
  }
  currentJob = await res.json();
  renderCurrentJob(currentJob);
}

function renderCurrentJob(job, options = {}) {
  renderJobList();
  if (EDIT_STATES.has(job.status)) showEditor(job, options);
  else showProcessing(job);
}

function showImportView(options = {}) {
  if (options.clearDraft !== false) resetImportMode();
  currentJob = null;
  setEditorDirty(false);
  fileInput.value = '';
  if (!options.preserveError) importErrorEl.textContent = '';
  setVisible(importView);
  renderJobList();
}

function resetImportMode() {
  rerunDraftJob = null;
  importTitleEl.textContent = '任务管理';
  rerunSourceEl.textContent = '';
  fileInput.disabled = false;
  pendingUploads = [];
  renderPendingList();
  updateUploadBtnLabel();
}

function showProcessingPlaceholder(name) {
  currentJob = null;
  processTitleEl.textContent = '创建任务';
  processNameEl.textContent = name;
  processMetaEl.textContent = '上传媒体并准备转写';
  processBarEl.style.width = '2%';
  processErrorEl.textContent = '';
  setVisible(processingView);
}

function showProcessing(job) {
  processTitleEl.textContent = job.status === 'failed' ? '任务失败' : '转写中';
  processNameEl.textContent = job.media_name || 'input.media';
  processMetaEl.textContent = jobSummary(job);
  processBarEl.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  processErrorEl.textContent = job.error || truncationWarning(job);
  deleteCurrentBtn.disabled = RUNNING_STATES.has(job.status);
  setVisible(processingView);
}

async function showEditor(job, options = {}) {
  applySubtitleStyle(job.subtitle_style || {});
  updateEditorChrome(job);
  setVisible(workbench);
  closeSettings();
  const mediaUrl = apiUrl(`api/jobs/${job.id}/media`);
  if (preview.dataset.jobId !== job.id) {
    preview.dataset.jobId = job.id;
    setPreviewSource(mediaUrl);
    resetVideoStage();
  }
  renderDownloads(job.status);
  if (!options.skipSegments) await loadSegments(job.id);
  fitVideoStageToMedia();
}

function updateEditorChrome(job) {
  selectedNameEl.textContent = job.media_name || 'input.media';
  taskStatusEl.textContent = statusLabel(job.status);
  taskStatusEl.className = statusClass(job.status);
  taskUsageEl.textContent = tokenUsageSummary(job);
  taskParamsEl.textContent = parameterSummary(job);
  updateRenderProgress(job);
  if (job.error) setTaskNotice(job.error, 'error');
  else if (truncationWarning(job)) setTaskNotice('可能截断，建议提高输出 tokens 后重新转写。', 'warning');
  else setTaskNotice('', '');
  updateRenderAction(job);
  updateRerunAction(job);
  setSaveState(editorDirty ? 'dirty' : 'saved', editorDirty ? '有未保存修改' : '已保存');
  renderDownloads(job.status);
}

function updateRenderAction(job) {
  const isRendering = job && job.status === 'rendering';
  if (!runtimeChecked) {
    renderBtn.disabled = true;
    renderBtn.textContent = '检测 FFmpeg...';
  } else {
    renderBtn.disabled = !ffmpegAvailable || isRendering;
    renderBtn.textContent = isRendering ? '烧录中...' : ffmpegAvailable ? '烧录视频' : 'FFmpeg 不可用';
  }
}

function updateRenderProgress(job) {
  const isRendering = job.status === 'rendering';
  const showProgress = isRendering || job.status === 'done';
  renderProgressMetaEl.classList.toggle('is-hidden', !showProgress);
  renderProgressEl.classList.toggle('is-hidden', !showProgress);
  const renderRatio = job.status === 'done' ? 1 : Math.max(0, Math.min(1, ((job.progress || RENDER_PROGRESS_BASE) - RENDER_PROGRESS_BASE) / RENDER_PROGRESS_SPAN));
  const percent = Math.round(renderRatio * 100);
  renderProgressBarEl.style.width = `${percent}%`;
  renderProgressTextEl.textContent = `${percent}%`;
}

function setTaskNotice(message, kind) {
  taskNoticeEl.textContent = message || '';
  taskNoticeEl.className = 'task-notice ' + (kind || '');
  taskNoticeEl.classList.toggle('is-hidden', !message);
}

function updateRerunAction(job) {
  rerunBtn.disabled = RUNNING_STATES.has(job.status);
  rerunBtn.textContent = '重新转写';
}

function showRerunDraft(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  const currentMax = Number(usage.max_new_tokens || inference.max_new_tokens || 0);
  rerunDraftJob = job;
  currentJob = null;
  importTitleEl.textContent = '重跑转写';
  fileInput.value = '';
  fileInput.disabled = true;
  rerunSourceEl.textContent = '来源媒体：' + (job.media_name || 'input.media');
  promptInput.value = inference.prompt || '';
  maxNewTokensInput.value = usage.possibly_truncated && currentMax > 0
    ? Math.max(currentMax * 2, currentMax + 512)
    : currentMax || '';
  maxLenInput.value = inference.max_length || '';
  decodingSelect.value = inference.decoding || 'greedy';
  temperatureInput.value = inference.temperature == null ? '1.0' : inference.temperature;
  updateDecodingControls();
  advancedDetails.open = true;
  uploadBtn.textContent = '开始重跑';
  importErrorEl.textContent = '';
  setVisible(importView);
  renderJobList();
}

async function startRerunDraft() {
  if (!rerunDraftJob) return;
  const source = rerunDraftJob;
  const payload = {
    prompt: promptInput.value,
    max_new_tokens: Number(maxNewTokensInput.value || 0),
    max_len: Number(maxLenInput.value || 0),
    decoding: decodingSelect.value,
  };
  if (temperatureInput.value) payload.temperature = Number(temperatureInput.value);
  uploadBtn.disabled = true;
  advancedDetails.open = false;
  importErrorEl.textContent = '';
  showProcessingPlaceholder(source.media_name || 'input.media');
  const res = await fetch(apiUrl(`api/jobs/${source.id}/rerun`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  uploadBtn.disabled = false;
  if (!res.ok) {
    importErrorEl.textContent = data.detail || '重跑失败';
    showImportView({ clearDraft: false, preserveError: true });
    return;
  }
  resetImportMode();
  currentJob = data;
  await refreshJobs({ keepSelection: true });
  await selectJob(data.id);
}

function setVisible(view) {
  importView.classList.toggle('is-hidden', view !== importView);
  processingView.classList.toggle('is-hidden', view !== processingView);
  workbench.classList.toggle('is-hidden', view !== workbench);
}

async function deleteJob(jobId) {
  const job = jobs.find((item) => item.id === jobId);
  if (job && RUNNING_STATES.has(job.status)) return;
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { method: 'DELETE' });
  if (!res.ok) return;
  if (currentJob && currentJob.id === jobId) {
    currentJob = null;
    preview.removeAttribute('src');
    maskPreviewVideo.removeAttribute('src');
    preview.removeAttribute('data-job-id');
    preview.load();
    maskPreviewVideo.load();
    tbody.innerHTML = '';
    downloads.innerHTML = '';
    setEditorDirty(false);
    showImportView();
  }
  await refreshJobs({ keepSelection: true });
}

async function saveSegments() {
  if (!currentJob) return false;
  if (!editorDirty) return true;
  setSaveState('saving', '正在保存...');
  const segments = collectSegments();
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/segments`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments, style: collectSubtitleStyle() })
    });
    const data = await res.json();
    if (!res.ok) {
      setTaskNotice(data.detail || '保存失败', 'error');
      setSaveState('error', data.detail || '保存失败');
      saveBtn.disabled = false;
      return false;
    }
    setTaskNotice('', '');
    renderSegments(data.segments);
    setEditorDirty(false);
    saveStatusEl.textContent = '已保存';
    saveStatusTimer = setTimeout(() => {
      if (!editorDirty) saveStatusEl.textContent = '已保存';
    }, 1200);
    await selectJob(currentJob.id);
    return true;
  } catch (err) {
    setTaskNotice('保存失败：' + err.message, 'error');
    setSaveState('error', '保存失败');
    saveBtn.disabled = false;
    return false;
  }
}

async function loadSegments(jobId) {
  const res = await fetch(apiUrl(`api/jobs/${jobId}/segments`));
  const data = await res.json();
  renderSegments(data.segments || []);
  setEditorDirty(false);
}

function ensurePolling() {
  const shouldPoll = jobs.some((job) => RUNNING_STATES.has(job.status));
  if (shouldPoll && !pollTimer) pollTimer = setInterval(() => refreshJobs({ keepSelection: true, background: true }), 1500);
  if (!shouldPoll && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function collectSubtitleStyle() {
  return {
    font_size: Number(document.querySelector('#fontSize').value || 48),
    margin_v: Number(document.querySelector('#marginV').value || 56),
    show_speaker: document.querySelector('#showSpeaker').value === 'true',
    speaker_colors: document.querySelector('#speakerColors').value === 'true',
    speaker_names: collectSpeakerNames(),
    mask_enabled: document.querySelector('#maskEnabled').value === 'true',
    mask_mode: document.querySelector('#maskMode').value || 'blur',
    mask_height: Number(document.querySelector('#maskHeight').value || 120),
    mask_margin_v: Number(document.querySelector('#maskMarginV').value || 0),
    mask_opacity: Number(document.querySelector('#maskOpacity').value || 0.82),
    mask_blur: Number(document.querySelector('#maskBlur').value || 24)
  };
}

function applySubtitleStyle(style) {
  if (!style || editorDirty) return;
  if (style.font_size != null) document.querySelector('#fontSize').value = style.font_size;
  if (style.margin_v != null) document.querySelector('#marginV').value = style.margin_v;
  if (style.show_speaker != null) document.querySelector('#showSpeaker').value = String(!!style.show_speaker);
  if (style.speaker_colors != null) document.querySelector('#speakerColors').value = String(!!style.speaker_colors);
  if (style.mask_enabled != null) document.querySelector('#maskEnabled').value = String(!!style.mask_enabled);
  if (style.mask_mode != null) document.querySelector('#maskMode').value = style.mask_mode === 'bar' ? 'bar' : 'blur';
  if (style.mask_height != null) document.querySelector('#maskHeight').value = style.mask_height;
  if (style.mask_margin_v != null) document.querySelector('#maskMarginV').value = style.mask_margin_v;
  if (style.mask_opacity != null) document.querySelector('#maskOpacity').value = style.mask_opacity;
  if (style.mask_blur != null) document.querySelector('#maskBlur').value = style.mask_blur;
  speakerNameMap = {};
  speakerMapEl.innerHTML = '';
  const names = style.speaker_names || {};
  for (const [speaker, name] of Object.entries(names)) {
    if (String(name).trim()) speakerNameMap[String(speaker)] = String(name).trim();
  }
}

function collectSpeakerNames() {
  const names = {};
  for (const input of speakerMapEl.querySelectorAll('input[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    const name = input.value.trim();
    if (speaker && name) names[speaker] = name;
  }
  return names;
}

function syncSpeakerNameInputs() {
  for (const input of speakerMapEl.querySelectorAll('input[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    if (!speaker) continue;
    const name = input.value.trim();
    if (name) speakerNameMap[speaker] = name;
    else delete speakerNameMap[speaker];
  }
}

function renderSpeakerMap(segments) {
  syncSpeakerNameInputs();
  const speakers = [...new Set(segments.map((segment) => segment.speaker).filter(Boolean))].sort();
  if (!speakers.length) {
    speakerMapEl.innerHTML = '<div class="meta">暂无说话人</div>';
    return;
  }
  speakerMapEl.innerHTML = speakers.map((speaker) => {
    const name = speakerNameMap[speaker] || '';
    return `
      <div class="speaker-map-row">
        <div class="speaker-tag">${escapeHtml(speaker)}</div>
        <input type="text" data-speaker="${escapeHtml(speaker)}" value="${escapeHtml(name)}" placeholder="显示名称">
      </div>`;
  }).join('');
}

function speakerDisplayName(speaker) {
  const names = collectSpeakerNames();
  return names[speaker] || speakerNameMap[speaker] || speaker;
}

function renderSegments(segments, preferredIndex = null) {
  tbody.innerHTML = '';
  activeSegmentIndex = -1;
  for (const [index, segment] of segments.entries()) {
    const tr = document.createElement('tr');
    tr.dataset.id = segment.id;
    tr.dataset.index = String(index);
    tr.innerHTML = `
      <td><input class="start" type="number" min="0" step="0.01" value="${segment.start}"></td>
      <td><input class="end" type="number" min="0" step="0.01" value="${segment.end}"></td>
      <td><input class="speaker" type="text" value="${escapeHtml(segment.speaker)}"></td>
      <td><textarea class="text" rows="1">${escapeHtml(segment.text)}</textarea></td>
      <td>
        <div class="segment-actions">
          <button class="segment-action add-row-above" type="button" title="在上方添加字幕">↑+</button>
          <button class="segment-action add-row-below" type="button" title="在下方添加字幕">↓+</button>
          <button class="segment-action delete-row" type="button" title="删除这条字幕">−</button>
        </div>
      </td>
    `;
    tr.addEventListener('click', (event) => {
      if (event.target.closest('input, textarea')) return;
      const rowIndex = Number(tr.dataset.index);
      const start = Number(tr.querySelector('.start').value);
      if (Number.isFinite(start)) preview.currentTime = Math.max(0, start);
      setActiveSegment(rowIndex, false);
      updateSubtitlePreview();
    });
    tbody.appendChild(tr);
    resizeSegmentRow(tr, false);
  }
  renderSpeakerMap(segments);
  renderTimeline(segments);
  if (preferredIndex != null && segments[preferredIndex]) {
    setActiveSegment(preferredIndex, true);
    updateSubtitlePreview(segments);
  } else {
    syncActiveSegment();
  }
}

function renderTimeline(segments) {
  const duration = timelineDuration(segments);
  const scrollWidth = timelineScroll.clientWidth || 1;
  const pixelsPerSecond = timelinePixelsPerSecond(duration, scrollWidth);
  currentPixelsPerSecond = pixelsPerSecond;
  const trackWidth = Math.max(scrollWidth, Math.ceil(duration * pixelsPerSecond));
  const layout = timelineLaneLayout(segments);
  const laneHeight = 44;
  const laneTop = 42;
  const laneCount = Math.max(1, layout.count);
  const laneAreaHeight = laneTop + laneCount * laneHeight + 14;
  timelineTrack.style.width = trackWidth + 'px';
  timelineTrack.style.height = Math.max(timelineScroll.clientHeight, 32 + laneAreaHeight) + 'px';
  timelineLane.style.height = laneAreaHeight + 'px';
  timelineMeta.textContent = segments.length + ' 段' + (duration ? ' · ' + formatTimelineTime(duration) : '') + (laneCount > 1 ? ' · ' + laneCount + ' 层' : '');
  timelineRuler.innerHTML = '';
  timelineLane.innerHTML = '';
  renderTimelineTicks(duration, pixelsPerSecond);
  for (const [index, segment] of segments.entries()) {
    const start = Math.max(0, Number(segment.start) || 0);
    const end = Math.max(start + 0.01, Number(segment.end) || start + 0.01);
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'timeline-segment';
    item.dataset.index = String(index);
    item.style.left = Math.max(0, start * pixelsPerSecond) + 'px';
    item.style.top = (laneTop + (layout.lanes.get(index) || 0) * laneHeight) + 'px';
    item.style.width = Math.max(8, (end - start) * pixelsPerSecond) + 'px';
    item.title = `${formatTimelineTime(start)} - ${formatTimelineTime(end)} ${segment.text || ''}`;
    item.innerHTML = `
      <span class="timeline-segment-speaker">${escapeHtml(segment.speaker || 'S--')}</span>
      <span class="timeline-segment-text">${escapeHtml(segment.text || '')}</span>
    `;
    item.addEventListener('pointerdown', (event) => onSegmentPointerDown(event, index, item));
    item.addEventListener('click', (event) => {
      event.preventDefault();
      if (segmentDragState && segmentDragState.moved) {
        return;
      }
      preview.currentTime = start;
      setActiveSegment(index, true);
      updateSubtitlePreview();
      updateTimelinePlayhead();
    });
    timelineLane.appendChild(item);
  }
  timelineTrack.appendChild(timelinePlayhead);
  updateTimelinePlayhead(segments);
}

function onSegmentPointerDown(event, index, segment) {
  if (event.button !== 0) return;
  event.preventDefault();
  const rect = segment.getBoundingClientRect();
  const offsetX = event.clientX - rect.left;
  let mode = 'move';
  if (rect.width > SEGMENT_EDGE_PX * 3) {
    if (offsetX <= SEGMENT_EDGE_PX) mode = 'start';
    else if (offsetX >= rect.width - SEGMENT_EDGE_PX) mode = 'end';
  }
  const segments = collectSegments();
  const seg = segments[index];
  if (!seg) return;
  const duration = timelineDuration(segments);
  const pps = currentPixelsPerSecond || timelinePixelsPerSecond(duration, timelineScroll.clientWidth || 1);
  segmentDragState = {
    index,
    mode,
    segment,
    pointerId: event.pointerId,
    startX: event.clientX,
    origStart: Math.max(0, Number(seg.start) || 0),
    origEnd: Math.max(Number(seg.start) || 0, Number(seg.end) || 0),
    duration,
    pps,
    segments,
    moved: false,
    newStart: null,
    newEnd: null
  };
  const moveHandler = (ev) => onSegmentPointerMove(ev, segmentDragState);
  const upHandler = (ev) => {
    onSegmentPointerUp(ev, segmentDragState);
    window.removeEventListener('pointermove', moveHandler);
    window.removeEventListener('pointerup', upHandler);
  };
  window.addEventListener('pointermove', moveHandler);
  window.addEventListener('pointerup', upHandler);
}

function onSegmentPointerMove(event, state) {
  if (!state) return;
  const dx = event.clientX - state.startX;
  if (!state.moved && Math.abs(dx) < SEGMENT_DRAG_THRESHOLD) return;
  if (!state.moved) {
    state.moved = true;
    state.segment.classList.add('dragging');
  }
  const deltaSec = dx / state.pps;
  let newStart = state.origStart;
  let newEnd = state.origEnd;
  if (state.mode === 'move') {
    newStart = state.origStart + deltaSec;
    newEnd = state.origEnd + deltaSec;
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (state.duration > 0 && newEnd > state.duration) {
      newStart -= (newEnd - state.duration);
      newEnd = state.duration;
    }
  } else if (state.mode === 'start') {
    newStart = Math.max(0, Math.min(state.origEnd - 0.1, state.origStart + deltaSec));
  } else {
    newEnd = Math.max(state.origStart + 0.1, state.origEnd + deltaSec);
    if (state.duration > 0) newEnd = Math.min(state.duration, newEnd);
  }
  const snap = computeSegmentSnap(state, newStart, newEnd);
  if (snap) {
    if (snap.edge === 'start') {
      const shift = snap.time - newStart;
      newStart = snap.time;
      if (state.mode === 'move') newEnd += shift;
    } else {
      const shift = snap.time - newEnd;
      newEnd = snap.time;
      if (state.mode === 'move') newStart += shift;
    }
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (state.duration > 0 && newEnd > state.duration) {
      newStart -= (newEnd - state.duration);
      newEnd = state.duration;
    }
  }
  if (snap) showTimelineGuide(snap.time, snap.label);
  else hideTimelineGuide();
  state.newStart = newStart;
  state.newEnd = newEnd;
  state.segment.style.left = Math.max(0, newStart * state.pps) + 'px';
  state.segment.style.width = Math.max(8, (newEnd - newStart) * state.pps) + 'px';
}

function computeSegmentSnap(state, newStart, newEnd) {
  const pps = state.pps || 1;
  const threshold = Math.max(0.05, SNAP_PX / pps);
  const candidates = [
    { time: 0, label: '起点' },
    { time: Number(preview.currentTime || 0), label: '播放头' }
  ];
  const segments = state.segments;
  for (let i = 0; i < segments.length; i++) {
    if (i === state.index) continue;
    candidates.push(
      { time: Number(segments[i].start) || 0, label: '头对齐' },
      { time: Number(segments[i].end) || 0, label: '尾对齐' }
    );
  }
  const edges = state.mode === 'end'
    ? [{ edge: 'end', time: newEnd }]
    : state.mode === 'start'
      ? [{ edge: 'start', time: newStart }]
      : [{ edge: 'start', time: newStart }, { edge: 'end', time: newEnd }];
  let best = null;
  for (const edge of edges) {
    for (const cand of candidates) {
      if (!Number.isFinite(cand.time)) continue;
      const d = Math.abs(edge.time - cand.time);
      if (d <= threshold && (!best || d < best.dist)) {
        best = { edge: edge.edge, time: cand.time, label: cand.label, dist: d };
      }
    }
  }
  return best;
}

function onSegmentPointerUp(event, state) {
  if (!state) return;
  state.segment.classList.remove('dragging');
  hideTimelineGuide();
  try { state.segment.releasePointerCapture(event.pointerId); } catch (err) {}
  if (state.moved && state.newStart != null && state.newEnd != null) {
    const tr = tbody.querySelector('tr[data-index="' + state.index + '"]');
    if (tr) {
      tr.querySelector('.start').value = roundTime(state.newStart);
      tr.querySelector('.end').value = roundTime(state.newEnd);
    }
    markEditorDirty();
    renderTimeline(collectSegments());
    updateSubtitlePreview();
    updateTimelinePlayhead();
  }
  const moved = state.moved;
  segmentDragState = moved ? { moved: true } : null;
  if (moved) {
    const st = segmentDragState;
    setTimeout(() => { if (segmentDragState === st) segmentDragState = null; }, 60);
  }
}

function showTimelineGuide(time, label) {
  const left = timelineTimeToX(time);
  timelineGuide.style.left = left + 'px';
  const minLabelOffset = 28;
  const maxLabelOffset = Math.max(minLabelOffset, timelineScroll.clientWidth - 44);
  const viewportLeft = left - timelineScroll.scrollLeft;
  const clampedOffset = Math.max(minLabelOffset, Math.min(maxLabelOffset, viewportLeft));
  timelineGuide.style.setProperty('--guide-label-offset', (clampedOffset - viewportLeft) + 'px');
  timelineGuide.dataset.label = label || '对齐';
  timelineGuide.classList.add('visible');
  timelineGuide.classList.add('snapped');
}

function hideTimelineGuide() {
  timelineGuide.classList.remove('visible');
  timelineGuide.classList.remove('snapped');
}

function timelineLaneLayout(segments) {
  const items = segments
    .map((segment, index) => ({
      index,
      start: Math.max(0, Number(segment.start) || 0),
      end: Math.max(Number(segment.start) || 0, Number(segment.end) || Number(segment.start) || 0)
    }))
    .sort((a, b) => a.start - b.start || a.end - b.end || a.index - b.index);
  const laneEnds = [];
  const lanes = new Map();
  for (const item of items) {
    let lane = laneEnds.findIndex((end) => end <= item.start + 0.001);
    if (lane < 0) {
      lane = laneEnds.length;
      laneEnds.push(0);
    }
    laneEnds[lane] = Math.max(item.end, item.start + 0.01);
    lanes.set(item.index, lane);
  }
  return { lanes, count: laneEnds.length };
}

function timelinePixelsPerSecond(duration, scrollWidth) {
  if (!duration || duration <= 0) return 12;
  return Math.max(8, Math.min(32, Math.max(1800, scrollWidth) / duration));
}

function renderTimelineTicks(duration, pixelsPerSecond) {
  const interval = timelineTickInterval(pixelsPerSecond);
  const end = Math.max(interval, Math.ceil((duration || interval) / interval) * interval);
  for (let time = 0; time <= end; time += interval) {
    const tick = document.createElement('div');
    tick.className = 'timeline-tick major';
    tick.style.left = Math.round(time * pixelsPerSecond) + 'px';
    const label = document.createElement('span');
    label.textContent = formatTimelineTime(time);
    tick.appendChild(label);
    timelineRuler.appendChild(tick);
    const half = time + interval / 2;
    if (half < end) {
      const minor = document.createElement('div');
      minor.className = 'timeline-tick';
      minor.style.left = Math.round(half * pixelsPerSecond) + 'px';
      timelineRuler.appendChild(minor);
    }
  }
}

function timelineTickInterval(pixelsPerSecond) {
  if (pixelsPerSecond >= 24) return 5;
  if (pixelsPerSecond >= 14) return 10;
  return 15;
}

function timelineDuration(segments) {
  const mediaDuration = Number(preview.duration || 0);
  const segmentDuration = Math.max(0, ...segments.map((segment) => Number(segment.end) || 0));
  return Math.max(mediaDuration, segmentDuration);
}

function formatTimelineTime(seconds) {
  seconds = Math.max(0, Number(seconds) || 0);
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function resizeSegmentTextarea(textarea, expanded) {
  if (!textarea) return;
  const maxHeight = expanded ? 112 : 48;
  textarea.style.height = 'auto';
  const naturalHeight = textarea.scrollHeight;
  const nextHeight = Math.max(30, Math.min(naturalHeight, maxHeight));
  textarea.style.height = nextHeight + 'px';
  textarea.style.overflowY = naturalHeight > maxHeight ? 'auto' : 'hidden';
}

function resizeSegmentRow(tr, expanded) {
  resizeSegmentTextarea(tr && tr.querySelector('textarea.text'), expanded);
}

function collectSegments() {
  return [...tbody.querySelectorAll('tr')].map((tr, index) => ({
    id: tr.dataset.id || `seg_${String(index + 1).padStart(4, '0')}`,
    start: Number(tr.querySelector('.start').value),
    end: Number(tr.querySelector('.end').value),
    speaker: tr.querySelector('.speaker').value,
    text: tr.querySelector('.text').value
  }));
}

function addSegmentAtPlayhead() {
  if (!currentJob) return;
  const segments = collectSegments();
  const start = roundTime(Math.max(0, Number(preview.currentTime || 0)));
  const next = segments.find((segment) => Number(segment.start) > start);
  const mediaEnd = Number(preview.duration || 0);
  const defaultEnd = mediaEnd > 0 ? Math.min(mediaEnd, start + 2.5) : start + 2.5;
  const end = roundTime(Math.max(start + 0.25, next ? Math.min(Number(next.start), defaultEnd) : defaultEnd));
  const currentSpeaker = segments[activeSegmentIndex] && segments[activeSegmentIndex].speaker;
  const segment = {
    id: 'seg_' + Date.now().toString(36),
    start,
    end,
    speaker: currentSpeaker || 'S01',
    text: ''
  };
  segments.push(segment);
  segments.sort((a, b) => Number(a.start) - Number(b.start));
  const index = segments.findIndex((item) => item.id === segment.id);
  preview.currentTime = start;
  renderSegments(segments, index);
  markEditorDirty();
  focusSegmentText(index);
}

function addSegmentAroundIndex(index, placement) {
  if (!currentJob) return;
  const segments = collectSegments();
  const source = segments[index];
  if (!source) {
    addSegmentAtPlayhead();
    return;
  }
  const isAbove = placement === 'above';
  const previous = segments[index - 1];
  const next = segments[index + 1];
  const anchorStart = Math.max(0, Number(source.start) || 0);
  const anchorEnd = Math.max(anchorStart, Number(source.end) || anchorStart);
  const mediaEnd = Number(preview.duration || 0);
  const segment = createBlankAdjacentSegment({
    source,
    previous,
    next,
    anchorStart,
    anchorEnd,
    mediaEnd,
    isAbove
  });
  const insertIndex = isAbove ? index : index + 1;
  segments.splice(insertIndex, 0, segment);
  preview.currentTime = segment.start;
  renderSegments(segments, insertIndex);
  markEditorDirty();
  focusSegmentText(insertIndex);
}

function createBlankAdjacentSegment({ source, previous, next, anchorStart, anchorEnd, mediaEnd, isAbove }) {
  let start;
  let end;
  if (isAbove) {
    end = anchorStart;
    const floor = previous ? Math.max(0, Number(previous.end) || 0) : 0;
    start = Math.max(floor, end - 2.5);
    if (end - start < 0.25) {
      start = Math.max(0, end - 0.25);
      if (end <= start) end = start + 0.25;
    }
  } else {
    start = anchorEnd;
    const ceiling = next ? Number(next.start) : (mediaEnd > 0 ? mediaEnd : start + 2.5);
    const defaultEnd = mediaEnd > 0 ? Math.min(mediaEnd, start + 2.5) : start + 2.5;
    end = Math.max(start + 0.25, Math.min(Number.isFinite(ceiling) ? ceiling : defaultEnd, defaultEnd));
  }
  if (mediaEnd > 0) {
    start = Math.min(start, Math.max(0, mediaEnd - 0.25));
    end = Math.min(Math.max(start + 0.25, end), mediaEnd);
  }
  start = roundTime(start);
  end = roundTime(Math.max(start + 0.25, end));
  if (mediaEnd > 0) end = roundTime(Math.min(end, mediaEnd));
  const segment = {
    id: 'seg_' + Date.now().toString(36),
    start,
    end,
    speaker: source.speaker || 'S01',
    text: ''
  };
  return segment;
}

function deleteActiveSegment() {
  if (!currentJob || activeSegmentIndex < 0) return;
  deleteSegmentAtIndex(activeSegmentIndex);
}

function deleteSegmentAtIndex(index) {
  if (!currentJob || index < 0) return;
  const segments = collectSegments();
  if (!segments[index]) return;
  segments.splice(index, 1);
  const nextIndex = Math.min(index, segments.length - 1);
  renderSegments(segments, nextIndex >= 0 ? nextIndex : null);
  markEditorDirty();
}

function focusSegmentText(index) {
  const tr = tbody.querySelector(`tr[data-index="${index}"]`);
  const textarea = tr && tr.querySelector('textarea.text');
  if (!textarea) return;
  textarea.focus();
  textarea.select();
}

function roundTime(value) {
  return Math.round(Number(value || 0) * 100) / 100;
}

function timelineTimeToX(time) {
  return Math.max(0, Number(time || 0) * (currentPixelsPerSecond || 1));
}

function timelineXToTime(x, duration) {
  const pps = currentPixelsPerSecond || 1;
  return Math.max(0, Math.min(duration, Number(x || 0) / pps));
}

function seekTimelineFromPointer(event) {
  const segments = collectSegments();
  const duration = timelineDuration(segments);
  if (!duration) return;
  const rect = timelineTrack.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const rawTime = roundTime(timelineXToTime(x, duration));
  const snap = computePlayheadSnap(rawTime, segments);
  const time = snap ? snap.time : rawTime;
  preview.currentTime = Math.max(0, Math.min(duration, time));
  if (timelineDragging && snap) showTimelineGuide(time, snap.label);
  else hideTimelineGuide();
  syncActiveSegment();
}

function computePlayheadSnap(time, segments) {
  const pps = currentPixelsPerSecond || 1;
  const threshold = Math.max(0.05, SNAP_PX / pps);
  const candidates = [{ time: 0, label: '起点' }];
  for (const segment of segments) {
    const start = Number(segment.start);
    const end = Number(segment.end);
    if (Number.isFinite(start)) candidates.push({ time: start, label: '头对齐' });
    if (Number.isFinite(end)) candidates.push({ time: end, label: '尾对齐' });
  }
  let best = null;
  for (const candidate of candidates) {
    const dist = Math.abs(time - candidate.time);
    if (dist <= threshold && (!best || dist < best.dist)) {
      best = { ...candidate, dist };
    }
  }
  return best;
}

function syncMaskPreviewTime() {
  if (!maskPreviewVideo.src) return;
  const drift = Math.abs(Number(maskPreviewVideo.currentTime || 0) - Number(preview.currentTime || 0));
  if (drift > 0.12) {
    try {
      maskPreviewVideo.currentTime = preview.currentTime || 0;
    } catch (err) {}
  }
}

function syncMaskPreviewPlaybackRate() {
  maskPreviewVideo.playbackRate = preview.playbackRate || 1;
}

function syncMaskPreviewPlayback() {
  if (!maskPreviewVideo.src) return;
  syncMaskPreviewTime();
  syncMaskPreviewPlaybackRate();
  if (preview.paused || preview.ended) {
    maskPreviewVideo.pause();
    return;
  }
  const playPromise = maskPreviewVideo.play();
  if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {});
}

function resetVideoStage() {
  assPlayRes = { x: 1920, y: 1080 };
  videoStage.style.width = '';
  videoStage.style.height = '';
  videoStage.style.aspectRatio = assPlayRes.x + ' / ' + assPlayRes.y;
  maskPreviewVideo.classList.remove('visible');
  maskPreviewVideo.pause();
}

function fitVideoStageToMedia() {
  const videoWidth = Number(preview.videoWidth || 0);
  const videoHeight = Number(preview.videoHeight || 0);
  const shell = videoStage.parentElement;
  if (!shell || videoWidth <= 0 || videoHeight <= 0) {
    resetVideoStage();
    updateSubtitlePreview();
    return;
  }
  assPlayRes = { x: videoWidth, y: videoHeight };
  const maxWidth = shell.clientWidth || videoWidth;
  const maxHeight = Math.max(180, Math.floor(window.innerHeight * 0.48));
  const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight);
  videoStage.style.width = Math.max(1, Math.floor(videoWidth * scale)) + 'px';
  videoStage.style.height = Math.max(1, Math.floor(videoHeight * scale)) + 'px';
  videoStage.style.aspectRatio = videoWidth + ' / ' + videoHeight;
  updateSubtitlePreview();
}

function assScriptScale() {
  const playResY = Number(assPlayRes.y || preview.videoHeight || 0);
  if (playResY <= 0) return 1;
  return (videoStage.clientHeight || playResY) / playResY;
}

function syncActiveSegment() {
  syncMaskPreviewTime();
  const time = Number(preview.currentTime || 0);
  const segments = collectSegments();
  const previousIndex = activeSegmentIndex;
  const index = segments.findIndex((segment, segmentIndex) => {
    const start = Number(segment.start);
    const end = Number(segment.end);
    return (
      segmentIndex === previousIndex
      && Number.isFinite(start)
      && Number.isFinite(end)
      && start <= time
      && time <= end
    );
  });
  const nextIndex = index >= 0 ? index : segments.findIndex((segment) => isSegmentVisibleAtTime(segment, time));
  setActiveSegment(nextIndex, true);
  updateTimelinePlayhead(segments);
  updateSubtitlePreview(segments);
}

function setActiveSegment(index, shouldScroll) {
  if (index === activeSegmentIndex) return;
  activeSegmentIndex = index;
  for (const tr of tbody.querySelectorAll('tr')) {
    const active = Number(tr.dataset.index) === index;
    tr.classList.toggle('active', active);
    resizeSegmentRow(tr, active);
    if (active && shouldScroll) scrollSegmentRowIntoView(tr);
  }
  for (const item of timelineLane.querySelectorAll('.timeline-segment')) {
    item.classList.toggle('active', Number(item.dataset.index) === index);
  }
}

function updateTimelinePlayhead(segments) {
  segments = segments || collectSegments();
  const duration = timelineDuration(segments);
  const trackWidth = Number.parseFloat(timelineTrack.style.width) || timelineTrack.clientWidth || timelineScroll.clientWidth || 1;
  const time = Math.max(0, Number(preview.currentTime || 0));
  const left = duration > 0 ? Math.min(trackWidth, timelineTimeToX(time)) : 0;
  timelinePlayhead.style.left = left + 'px';
  if (time > 0 && timelineScroll.clientWidth) {
    const visibleLeft = timelineScroll.scrollLeft;
    const visibleRight = visibleLeft + timelineScroll.clientWidth;
    if (left < visibleLeft + 24 || left > visibleRight - 24) {
      timelineScroll.scrollLeft = Math.max(0, left - timelineScroll.clientWidth * 0.45);
    }
  }
}

function scrollSegmentRowIntoView(tr) {
  const container = tr.closest('.table-wrap');
  if (!container) return;
  const stickyHeaderHeight = 30;
  const rowTop = tr.offsetTop;
  const rowBottom = rowTop + tr.offsetHeight;
  const viewTop = container.scrollTop + stickyHeaderHeight;
  const viewBottom = container.scrollTop + container.clientHeight;
  if (rowTop < viewTop) {
    container.scrollTop = Math.max(0, rowTop - stickyHeaderHeight - 4);
  } else if (rowBottom > viewBottom) {
    container.scrollTop = rowBottom - container.clientHeight + 8;
  }
}

function updateSubtitlePreview(segments) {
  segments = segments || collectSegments();
  updateSourceMaskPreview();
  const time = Number(preview.currentTime || 0);
  const visibleSegments = segments
    .map((segment, index) => ({ segment, index }))
    .filter((item) => isSegmentVisibleAtTime(item.segment, time) && String(item.segment.text || '').trim())
    .sort((a, b) => {
      if (a.index === activeSegmentIndex) return -1;
      if (b.index === activeSegmentIndex) return 1;
      return Number(a.segment.start) - Number(b.segment.start);
    });
  if (!visibleSegments.length) {
    subtitleOverlay.classList.remove('visible');
    subtitleOverlay.textContent = '';
    return;
  }
  const showSpeaker = document.querySelector('#showSpeaker').value === 'true';
  const useSpeakerColors = document.querySelector('#speakerColors').value === 'true';
  const fontSize = Math.max(12, Number(document.querySelector('#fontSize').value || 48));
  const marginV = Math.max(0, Number(document.querySelector('#marginV').value || 56));
  const lines = visibleSegments.map(({ segment }) => (
    showSpeaker && segment.speaker ? speakerDisplayName(segment.speaker) + ': ' + segment.text : segment.text
  ));
  const scale = assScriptScale();
  subtitleOverlay.textContent = lines.join('\\n');
  subtitleOverlay.style.fontSize = Math.max(10, fontSize * scale / assFontLineHeightFactor) + 'px';
  subtitleOverlay.style.lineHeight = String(assFontLineHeightFactor);
  subtitleOverlay.style.bottom = Math.max(0, marginV * scale) + 'px';
  subtitleOverlay.style.webkitTextStroke = subtitleTextStroke(scale);
  subtitleOverlay.style.textShadow = subtitleTextShadow(scale);
  const color = useSpeakerColors && visibleSegments.length === 1 ? speakerColor(visibleSegments[0].segment.speaker, segments) : '#ffffff';
  subtitleOverlay.style.color = color;
  subtitleOverlay.style.webkitTextFillColor = color;
  subtitleOverlay.classList.add('visible');
}

function isSegmentVisibleAtTime(segment, time) {
  const start = Number(segment.start);
  const end = Number(segment.end);
  return Number.isFinite(start) && Number.isFinite(end) && start <= time && time < end;
}

function updateSourceMaskPreview() {
  const enabled = document.querySelector('#maskEnabled').value === 'true';
  sourceMaskOverlay.classList.toggle('visible', enabled);
  maskPreviewVideo.classList.toggle('visible', false);
  if (!enabled) {
    maskPreviewVideo.pause();
    return;
  }
  const scale = assScriptScale();
  const mode = document.querySelector('#maskMode').value || 'blur';
  const height = Math.max(1, Number(document.querySelector('#maskHeight').value || 120));
  const marginV = Math.max(0, Number(document.querySelector('#maskMarginV').value || 0));
  const opacity = Math.max(0, Math.min(1, Number(document.querySelector('#maskOpacity').value || 0.82)));
  const blur = Math.max(1, Number(document.querySelector('#maskBlur').value || 24));
  const scaledHeight = Math.max(1, height * scale);
  const scaledBottom = Math.max(0, marginV * scale);
  const stageHeight = Math.max(1, videoStage.clientHeight || assPlayRes.y || 1);
  const clipTop = Math.max(0, stageHeight - scaledBottom - scaledHeight);
  const clipBottom = Math.max(0, scaledBottom);
  sourceMaskOverlay.style.height = scaledHeight + 'px';
  sourceMaskOverlay.style.bottom = scaledBottom + 'px';
  if (mode === 'bar') {
    sourceMaskOverlay.style.background = `rgba(0, 0, 0, ${opacity})`;
    maskPreviewVideo.pause();
  } else {
    maskPreviewVideo.classList.add('visible');
    maskPreviewVideo.style.clipPath = `inset(${clipTop}px 0 ${clipBottom}px 0)`;
    maskPreviewVideo.style.filter = `blur(${Math.max(1, blur * scale)}px)`;
    sourceMaskOverlay.style.background = 'rgba(0, 0, 0, 0.18)';
    syncMaskPreviewPlayback();
  }
}

function subtitleTextStroke(scale) {
  return Math.max(1, 3 * scale) + 'px #000';
}

function subtitleTextShadow(scale) {
  const shadow = Math.max(0.5, 1 * scale);
  const blur = Math.max(1, 3 * scale);
  return `0 ${shadow}px ${blur}px rgba(0, 0, 0, 0.65)`;
}

function speakerColor(speaker, segments) {
  const speakers = [];
  for (const segment of segments) {
    if (segment.speaker && !speakers.includes(segment.speaker)) speakers.push(segment.speaker);
  }
  speakers.sort();
  const index = Math.max(0, speakers.indexOf(speaker || ''));
  return speakerPalette[index % speakerPalette.length];
}

function renderDownloads(status) {
  if (!currentJob) return;
  const links = [
    ['json', 'JSON'],
    ['srt', 'SRT'],
    ['ass', 'ASS'],
    ['transcript', '原文']
  ];
  if (status === 'done') links.push(['mp4', 'MP4']);
  downloads.innerHTML = links.map(([kind, label]) =>
    `<a href="${apiUrl(`api/jobs/${currentJob.id}/download?kind=${kind}`)}" target="_blank">${label}</a>`
  ).join('');
}

async function exportCurrentJobToFolder() {
  if (!currentJob) return;
  if (!window.showDirectoryPicker) {
    exportStatusEl.textContent = '当前浏览器不支持选择文件夹，请使用上方下载链接。';
    return;
  }
  const saved = await saveSegments();
  if (!saved) return;
  const files = [
    ['json', 'segments.json'],
    ['srt', 'subtitle.srt'],
    ['ass', 'subtitle.ass'],
    ['transcript', 'raw_transcript.txt'],
  ];
  if (currentJob.status === 'done') files.push(['mp4', 'output.mp4']);
  try {
    exportFolderBtn.disabled = true;
    exportStatusEl.textContent = '请选择导出文件夹...';
    const directory = await window.showDirectoryPicker({ mode: 'readwrite' });
    let count = 0;
    for (const [kind, filename] of files) {
      exportStatusEl.textContent = `正在保存 ${filename}...`;
      const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/download?kind=${kind}`), { cache: 'no-store' });
      if (!res.ok) continue;
      const handle = await directory.getFileHandle(filename, { create: true });
      const writable = await handle.createWritable();
      await writable.write(await res.blob());
      await writable.close();
      count += 1;
    }
    exportStatusEl.textContent = `已保存 ${count} 个文件到所选文件夹。`;
  } catch (err) {
    exportStatusEl.textContent = err && err.name === 'AbortError' ? '已取消选择文件夹。' : '导出失败：' + (err.message || err);
  } finally {
    exportFolderBtn.disabled = false;
  }
}

function jobSummary(job) {
  const inference = job.inference || {};
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return tokenUsageSummary(job) + ' · max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function parameterSummary(job) {
  const inference = job.inference || {};
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return 'max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function tokenUsageSummary(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  const maxNewTokens = usage.max_new_tokens || inference.max_new_tokens || 0;
  if (usage.generated_tokens == null) return '生成 tokens ' + maxNewTokens;
  const prompt = usage.prompt_tokens == null ? '' : (' · prompt ' + usage.prompt_tokens);
  return '生成 ' + usage.generated_tokens + '/' + maxNewTokens + ' tokens' + prompt;
}

function truncationWarning(job) {
  const usage = job.usage || {};
  if (!usage.possibly_truncated) return '';
  return '可能截断：生成 token 已达到上限，请检查字幕末尾或提高输出 tokens 后重跑。';
}

function statusClass(status) {
  return 'pill ' + (status === 'failed' ? 'bad' : status === 'done' ? 'ok' : '');
}

function statusLabel(status) {
  const labels = {
    queued: '排队中',
    loading_model: '加载模型',
    transcribing: '转写中',
    postprocessing: '处理中',
    waiting_review: '待校对',
    rendering: '烧录中',
    done: '已完成',
    failed: '失败',
    cancelled: '已取消',
    idle: '空闲'
  };
  return labels[status] || status;
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

refreshRuntime();
refreshJobs();
</script>
</body>
</html>
"""
