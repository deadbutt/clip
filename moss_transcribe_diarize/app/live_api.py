"""直播工作台的 HTTP 接口。"""
import asyncio
import json
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response

from .live_overlay import overlay_launcher
from .live_streams import SUPPORTED_PLATFORM_HOSTS


def register_live_routes(app, live_streams, manager):
    @app.get("/api/live/platforms")
    def platforms():
        return {"platforms": list(SUPPORTED_PLATFORM_HOSTS), "engines": live_streams.resolver.runtime_info()}

    @app.get("/api/live/sessions")
    def sessions():
        return {"sessions": [{**item, "subtitles": app.state.live_subtitles.snapshot(item["id"]),
                              "preview": app.state.live_preview.snapshot(item["id"])}
                             for item in live_streams.list()]}

    @app.get("/api/live/sessions/{session_id}/preview/{filename}")
    def preview_media(session_id: str, filename: str):
        get(session_id)
        try:
            path = app.state.live_preview.media_path(session_id, filename)
        except FileNotFoundError as exc:
            raise HTTPException(404, "播放缓存尚未就绪或已经滚动清理") from exc
        return FileResponse(path, media_type="application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/mp2t",
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/live/sessions/{session_id}/preview/start")
    async def start_preview(session_id: str):
        session = get(session_id)
        try:
            if session.process is None:
                await asyncio.to_thread(live_streams.resolve, session_id)
            with session.lock:
                url, headers = session.stream_url, dict(session.stream_headers)
            if not url:
                raise RuntimeError(session.error or "当前没有可用直播流")
            return await asyncio.to_thread(app.state.live_preview.start, session_id, url, headers)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/live/sessions/{session_id}/preview/stop")
    async def stop_preview(session_id: str):
        get(session_id)
        return await asyncio.to_thread(app.state.live_preview.stop, session_id)

    @app.get("/api/live/sessions/{session_id}/subtitles")
    def subtitles(session_id: str):
        get(session_id)
        return app.state.live_subtitles.snapshot(session_id)

    @app.get("/api/live/sessions/{session_id}/subtitles/draft")
    def subtitle_draft(session_id: str):
        get(session_id)
        service = app.state.live_subtitles
        with service.lock:
            session = service.sessions.get(session_id)
        if session is None or not session.draft_path.is_file():
            raise HTTPException(404, "尚未生成实时字幕草稿")
        rows = {}
        with session.lock:
            for line in session.draft_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                rows[row["id"]] = row
        return Response(json.dumps(list(rows.values()), ensure_ascii=False, indent=2),
                        media_type="application/json", headers={
                            "Content-Disposition": f'attachment; filename="live-{session_id}-bilingual.json"'})

    @app.post("/api/live/sessions/{session_id}/subtitles/start")
    async def start_subtitles(session_id: str, request: Request):
        session = get(session_id)
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("translate", True), bool):
            raise HTTPException(400, "需要实时字幕设置对象，translate 必须为布尔值")
        try:
            if not session.process:
                await asyncio.to_thread(live_streams.resolve, session_id)
            with session.lock:
                url, headers = session.stream_url, dict(session.stream_headers)
            if not url:
                raise RuntimeError(session.error or "当前没有可用直播流")
            return await asyncio.to_thread(app.state.live_subtitles.start, session_id, url, headers,
                translate=payload.get("translate", True), language=payload.get("language") or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/live/sessions/{session_id}/subtitles/stop")
    async def stop_subtitles(session_id: str):
        get(session_id)
        return await asyncio.to_thread(app.state.live_subtitles.stop, session_id)

    # 桌面字幕浮窗：原生窗口（无边框 / 置顶 / 背景透明），只有它能做到真正的桌面透明。
    # 它是一个独立进程，自己轮询 /subtitles 接口，所以关掉网页也不影响它。
    @app.post("/api/live/sessions/{session_id}/overlay/start")
    async def start_overlay(session_id: str, request: Request):
        get(session_id)
        return await asyncio.to_thread(overlay_launcher.start, str(request.base_url), session_id)

    @app.post("/api/live/sessions/{session_id}/overlay/stop")
    async def stop_overlay(session_id: str):
        get(session_id)
        return await asyncio.to_thread(overlay_launcher.stop, session_id)

    @app.post("/api/live/sessions")
    async def add(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "需要直播链接设置对象")
        try:
            session = await asyncio.to_thread(live_streams.add, payload.get("url"), payload.get("quality", "best"))
            return session.to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def get(session_id):
        try:
            return live_streams.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, "直播间不存在") from exc

    @app.post("/api/live/sessions/{session_id}/{action}")
    async def action(session_id: str, action: str):
        get(session_id)
        handlers = {"resolve": live_streams.resolve, "record": live_streams.start_recording, "stop": live_streams.stop_recording}
        if action not in handlers:
            raise HTTPException(404, "未知直播操作")
        try:
            return (await asyncio.to_thread(handlers[action], session_id)).to_dict()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/live/sessions/{session_id}")
    async def delete(session_id: str):
        get(session_id)
        await asyncio.to_thread(overlay_launcher.stop, session_id)
        await asyncio.to_thread(app.state.live_preview.stop, session_id)
        await asyncio.to_thread(app.state.live_subtitles.stop, session_id)
        await asyncio.to_thread(live_streams.remove, session_id)
        return {"ok": True}

    @app.post("/api/live/recordings/{session_id}/transcribe")
    async def transcribe(session_id: str):
        session = get(session_id)
        if session.process is not None:
            raise HTTPException(409, "请先停止录制并等待文件保存完成")
        path = Path(session.recording_path)
        if not path.is_file():
            raise HTTPException(409, "录制文件尚未生成")
        job = await asyncio.to_thread(manager.create_job_from_file, path, media_name=path.name)
        # The recorder owns videos/live staging files.  Once the job has been
        # registered, JobManager has deduplicated it into the canonical
        # videos/ library, so remove only this session's staging file.
        try:
            if path.parent == live_streams.live_dir:
                path.unlink(missing_ok=True)
        except OSError:
            pass
        with session.lock:
            session.recording_path = str(job.input_path)
        live_streams._save()
        return job.to_dict()
