import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.defaults import DEFAULT_PROMPT

from .events import EventHub
from .ffmpeg import detect_ffmpeg
from .jobs import JobManager
from .whisper_runner import WhisperRunner

logger = logging.getLogger(__name__)

# 上传接口接受的音视频后缀白名单(不带点,全小写)。
ALLOWED_MEDIA_SUFFIXES = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".au",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp",
}


def _parse_protected_terms(value: Any) -> tuple[str, ...]:
    if value in ("", None):
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    return tuple(str(item).strip() for item in items if str(item).strip())


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "on", "yes")


def create_app(
    *,
    model_path: str | Path,
    runs_dir: str | Path = "runs",
    device: str = "auto",
    dtype: str = "bf16",
    language: str | None = None,
    whisper_beam_size: int = 5,
    prompt: str = DEFAULT_PROMPT,
    max_length: int = 131072,
    max_new_tokens: int = 8192,
    decoding: str = "greedy",
    temperature: float | None = None,
    translator_base_url: str | None = None,
    translator_model: str | None = None,
    translator_api_key: str | None = None,
    translator_timeout: float | None = None,
    translator_provider: str = "openai",
    translator_tokenizer_dir: str | Path | None = "models/opus-mt-en-zh",
    translator_device: str = "auto",
    translator_compute_type: str = "auto",
    translator_protected_terms: tuple[str, ...] = (),
    speaker_count: int | None = None,
    diarization_backend: str = "none",
    hf_token: str | None = None,
    pyannote_model: str = "pyannote/speaker-diarization-3.1",
    diarization_device: str = "auto",
):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Install fastapi, uvicorn, and python-multipart to run the local web app.") from exc

    app = FastAPI(title="蝶殇工作台")
    hub = EventHub()

    @app.on_event("startup")
    async def _bind_job_events() -> None:
        manager.bind_events(asyncio.get_running_loop(), hub)
        _purge_orphan_cookies_files(manager)

    runner = WhisperRunner(model_path, device=device, dtype=dtype, language=language, beam_size=whisper_beam_size)
    manager = JobManager(
        runs_dir,
        runner,
        prompt=prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        decoding=decoding,
        temperature=temperature,
        speaker_count=speaker_count,
        diarization_backend=diarization_backend,
        hf_token=hf_token,
        pyannote_model=pyannote_model,
        diarization_device=diarization_device,
    )
    app.state.manager = manager
    # 同时进行的 clip 渲染(ffmpeg 编码进程)上限;整片烧录走 worker 单线程,不占此配额。
    clip_render_semaphore = asyncio.Semaphore(2)
    from .llm_profiles import LlmProfileStore

    app.state.llm_store = LlmProfileStore(Path(runs_dir).parent / "config")
    translator = None
    translator_url = translator_base_url
    if translator_provider == "opus-mt":
        from .local_mt_translator import LocalMtTranslator

        translator = LocalMtTranslator(
            model_dir=translator_model or "models/opus-mt-en-zh-ct2-int8",
            tokenizer_dir=translator_tokenizer_dir,
            device=translator_device,
            compute_type=translator_compute_type,
        )
    elif translator_url:
        from .text_translator import PROTECTED_TERMS, TextTranslator

        translator = TextTranslator(
            base_url=translator_url,
            model=translator_model or "local",
            api_key=translator_api_key or "EMPTY",
            timeout=translator_timeout if translator_timeout is not None else 600.0,
            provider=translator_provider,
            protected_terms=translator_protected_terms or tuple(PROTECTED_TERMS),
        )
    app.state.translator = translator

    def _fail(exc: Exception, status: int | None = None) -> HTTPException:
        """未分类异常统一记完整堆栈到日志,前端仍只收 detail 字符串。

        ValueError 视为用户输入问题(400),其余视为服务端内部错误(500);
        调用方需要覆盖时显式传 status。
        """
        if status is None:
            status = 400 if isinstance(exc, ValueError) else 500
        logger.warning("API error: %s: %s", type(exc).__name__, exc, exc_info=exc)
        return HTTPException(status_code=status, detail=str(exc))

    def _purge_orphan_cookies_files(mgr) -> None:
        """启动时清扫上一进程遗留的 cookies 临时文件（浏览器登录凭据）。

        只删本服务写的 *.cookies.txt 且不被任何待下载任务引用的；
        用户自备路径的 cookies 不在 runs 目录下,天然不受影响。
        """
        try:
            runs = Path(mgr.runs_dir)
            pending = {
                Path(str((job.cookies_config or {}).get("file") or "")).resolve()
                for job in mgr._jobs.values()
            }
            for leftover in runs.glob("*.cookies.txt"):
                try:
                    if leftover.resolve() not in pending:
                        leftover.unlink(missing_ok=True)
                except OSError:
                    logger.debug("cookies temp file cleanup failed: %s", leftover, exc_info=True)
        except Exception:
            logger.debug("cookies temp file sweep failed", exc_info=True)

    static_dir = Path(__file__).parent / "static"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/static/{filename}")
    def static_file(filename: str):
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid static file name.")
        path = static_dir / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Static file not found.")
        media_types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}
        return FileResponse(
            path,
            media_type=media_types.get(path.suffix, "application/octet-stream"),
            headers={"Cache-Control": "no-store"},
        )

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
            "speaker_labeling": {
                "default_speaker_count": manager.speaker_count,
                "default_backend": manager.diarization_backend,
                "pyannote_model": manager.pyannote_model,
            },
            "translator": translator.runtime_info() if translator is not None else {"available": False},
        }

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [job.to_dict() for job in manager.list_jobs()], "queue": manager.queue_info()}

    @app.get("/api/queue")
    def queue_info():
        return manager.queue_info()

    @app.post("/api/queue/pause")
    def pause_queue():
        return manager.pause_queue()

    @app.post("/api/queue/resume")
    def resume_queue():
        return manager.resume_queue()

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
        max_len: int | None = Form(None),
        decoding: str | None = Form(None),
        temperature: float | None = Form(None),
        speaker_count: int | None = Form(None),
        diarization_backend: str | None = Form(None),
        hotwords: str | None = Form(None),
    ):
        try:
            filename = file.filename or "input.media"
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_MEDIA_SUFFIXES:
                raise ValueError(f"不支持的文件类型 '{suffix or '(无后缀)'}'，请上传常见音视频文件（mp4/mkv/mp3/wav 等）")
            job, input_path = manager.create_job_for_upload(
                filename,
                prompt=prompt,
                max_length=max_len,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
                speaker_count=speaker_count,
                diarization_backend=diarization_backend,
                hotwords=hotwords,
            )
        except Exception as exc:
            raise _fail(exc) from exc
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

    @app.post("/api/jobs/url")
    async def create_job_from_url(request: Request):
        import tempfile
        content_type = request.headers.get("content-type", "")
        cookies_file_path = None
        uploaded_tmp = None
        if "multipart/form-data" in content_type:
            params = await request.form()
            if not hasattr(params, "get"):
                params = {}
            cookies_file_upload = params.get("cookies_file")
            if cookies_file_upload and hasattr(cookies_file_upload, "read"):
                cookies_bytes = await cookies_file_upload.read()
                if cookies_bytes:
                    # .cookies.txt 后缀标记这是本服务写的临时文件,下载结束即删。
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".cookies.txt", delete=False, dir=str(manager.runs_dir)
                    )
                    tmp.write(cookies_bytes)
                    tmp.close()
                    cookies_file_path = tmp.name
                    uploaded_tmp = tmp.name
        else:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            params = payload if isinstance(payload, dict) else {}
        url = str(params.get("url") or "").strip()
        cookies_browser = params.get("cookies_browser") or "firefox"
        cookies_file_path = cookies_file_path or params.get("cookies_file")
        if cookies_browser in ("", None, "none"):
            cookies_browser = None
        try:
            if not url:
                raise ValueError("URL 不能为空")
            job = manager.create_job_for_url(
                url,
                cookies_browser=cookies_browser,
                cookies_file=cookies_file_path,
                prompt=params.get("prompt"),
                max_length=params.get("max_len") or params.get("max_length"),
                max_new_tokens=params.get("max_new_tokens"),
                decoding=params.get("decoding"),
                temperature=params.get("temperature"),
                speaker_count=params.get("speaker_count"),
                diarization_backend=params.get("diarization_backend"),
                force_transcribe=_as_bool(params.get("force_transcribe")),
                hotwords=str(params.get("hotwords") or ""),
            )
            return job.to_dict()
        except Exception as exc:
            # create_job_for_url 抛异常或 URL 校验失败时任务未入列,下载阶段的
            # finally 清理不会执行,这里兜底删除上传的 cookies 临时副本;
            # 用户自备路径的 cookies 不在删除范围。
            if uploaded_tmp:
                try:
                    Path(uploaded_tmp).unlink(missing_ok=True)
                except OSError:
                    logger.debug("cookies temp file cleanup failed: %s", uploaded_tmp, exc_info=True)
            raise _fail(exc) from exc

    @app.post("/api/cookies/check")
    async def check_cookies(request: Request):
        import tempfile

        from .downloader import check_browser_cookies, check_cookies_file

        content_type = request.headers.get("content-type", "")
        browser = "firefox"
        cookies_file_path = None
        tmp_path = None
        if "multipart/form-data" in content_type:
            form = await request.form()
            browser = str(form.get("browser") or "firefox")
            upload = form.get("file")
            if upload and hasattr(upload, "read"):
                data = await upload.read()
                if data:
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".cookies.txt", delete=False, dir=str(manager.runs_dir)
                    )
                    tmp.write(data)
                    tmp.close()
                    cookies_file_path = tmp.name
                    tmp_path = tmp.name
        else:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            browser = str(payload.get("browser") or "firefox")
            cookies_file_path = payload.get("file")

        try:
            result = {"browser": browser}
            if cookies_file_path:
                result["file_check"] = check_cookies_file(cookies_file_path)
            result["browser_check"] = check_browser_cookies(browser)
            return result
        finally:
            # 上传的 cookies 是浏览器登录凭据,检测完立即删除临时副本。
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    logger.debug("cookies temp file cleanup failed: %s", tmp_path, exc_info=True)

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

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            job = manager.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.to_dict()

    # ------------------------------------------------------------ SSE 实时事件

    def _sse_frame(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        try:
            manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        channel = f"job:{job_id}"
        queue = hub.subscribe(channel)

        async def stream():
            try:
                # 连接建立先推一份当前快照，前端不用等下一次状态变更。
                yield _sse_frame("job", manager.get_job(job_id).to_dict())
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield _sse_frame(item["event"], item["data"])
            finally:
                hub.unsubscribe(channel, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------ 查找与替换

    @app.get("/api/jobs/{job_id}/search")
    def search_job_segments(job_id: str, q: str, mode: str = "literal"):
        try:
            return manager.search_segments(job_id, q, mode=mode)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/replace")
    async def replace_job_segments(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            payload = payload if isinstance(payload, dict) else {}
            query = str(payload.get("query") or "")
            if not query.strip():
                raise ValueError("query 不能为空")
            segment_ids = payload.get("segment_ids")
            return await asyncio.to_thread(
                manager.replace_segments,
                job_id,
                query,
                str(payload.get("replacement") or ""),
                mode=str(payload.get("mode") or "literal"),
                segment_ids=segment_ids if isinstance(segment_ids, list) else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun_job(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            # rerun 会整文件复制输入视频(可达 GB 级),不能在 event loop 里同步执行。
            job = await asyncio.to_thread(
                manager.rerun_job,
                job_id,
                prompt=payload.get("prompt"),
                max_length=payload.get("max_len") or payload.get("max_length"),
                max_new_tokens=payload.get("max_new_tokens"),
                decoding=payload.get("decoding"),
                temperature=payload.get("temperature"),
                speaker_count=payload.get("speaker_count"),
                diarization_backend=payload.get("diarization_backend"),
            )
            return job.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Media file is missing.") from exc
        except Exception as exc:
            raise _fail(exc) from exc

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
    def get_segments(job_id: str, request: Request):
        try:
            # 数据与 ETag 指纹来自同一次调用;文件没变时 304 短路,
            # 长视频(几 MB segments.json)下把 2s 轮询的开销降到一次 stat + 空响应。
            segments, version = manager.list_segments_with_version(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if version is None:
            return {"segments": segments}
        etag = f'"seg-{version[0]}-{version[1]}"'
        headers = {"ETag": etag, "Cache-Control": "no-store"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return JSONResponse({"segments": segments}, headers=headers)

    @app.put("/api/jobs/{job_id}/segments")
    async def update_segments(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
            style = payload.get("style") if isinstance(payload, dict) else None
            if not isinstance(segments, list):
                raise ValueError("Expected a JSON list or an object with a segments list.")
            # 保存字幕会同步跑一次 ffprobe(现带 30s timeout),移出 event loop。
            segments_out = await asyncio.to_thread(manager.update_segments, job_id, segments, style)
            return {"segments": segments_out}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/segments/{segment_id}/split")
    async def split_segment(job_id: str, segment_id: str, request: Request):
        try:
            try:
                payload: Any = await request.json()
            except Exception:
                payload = {}
            split_time = payload.get("time") if isinstance(payload, dict) else None
            if split_time is not None:
                split_time = float(split_time)
            segments_out = await asyncio.to_thread(manager.split_segment, job_id, segment_id, split_time)
            job_info = manager.get_job(job_id)
            needs_retranslate = bool((job_info.translation_info or {}).get("structure_changed"))
            return {"segments": segments_out, "needs_retranslate": needs_retranslate}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise _fail(exc) from exc
        except Exception as exc:
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/segments/merge")
    async def merge_segments(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            ids = payload.get("ids") if isinstance(payload, dict) else payload
            if not isinstance(ids, list) or not all(isinstance(v, str) for v in ids):
                raise ValueError("Expected a JSON object with an 'ids' list of segment ids.")
            segments_out = await asyncio.to_thread(manager.merge_segments, job_id, ids)
            job_info = manager.get_job(job_id)
            needs_retranslate = bool((job_info.translation_info or {}).get("structure_changed"))
            return {"segments": segments_out, "needs_retranslate": needs_retranslate}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise _fail(exc) from exc
        except Exception as exc:
            raise _fail(exc) from exc

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
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/translate")
    async def translate(job_id: str, request: Request):
        translator = app.state.translator
        if translator is None:
            return JSONResponse({"detail": "Translation model is not configured. Start with start_ollama.bat, start_vllm.bat, --translator-provider opus-mt, or pass --translator-base-url."}, status_code=503)
        try:
            try:
                payload: Any = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            protected_terms = _parse_protected_terms(payload.get("protected_terms"))
            request_translator = translator
            if "protected_terms" in payload and hasattr(translator, "protected_terms"):
                try:
                    request_translator = replace(translator, protected_terms=protected_terms)
                except TypeError:
                    request_translator = translator
            batch_size = payload.get("batch_size")
            batch_size = None if batch_size in ("", None) else max(1, int(batch_size))
            return await asyncio.to_thread(
                manager.translate,
                job_id,
                request_translator,
                target_language=str(payload.get("target_language") or "简体中文"),
                mode=str(payload.get("mode") or "bilingual"),
                batch_size=batch_size,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception as exc:
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/translate/restore")
    def restore_translation(job_id: str):
        try:
            return manager.restore_source_segments(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    # ------------------------------------------------------------ LLM profiles

    @app.get("/api/llm/profiles")
    def list_llm_profiles():
        return app.state.llm_store.list_profiles()

    # ------------------------------------------------------------ hotwords glossary

    @app.get("/api/hotwords")
    def get_hotwords():
        return {"terms": manager.load_hotwords_glossary()}

    @app.put("/api/hotwords")
    async def update_hotwords(request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            terms = payload.get("terms")
            if not isinstance(terms, list):
                raise ValueError("terms must be a list.")
            return {"terms": manager.save_hotwords_glossary(terms)}
        except ValueError as exc:
            raise _fail(exc) from exc

    @app.post("/api/llm/profiles")
    async def create_llm_profile(request: Request):
        try:
            payload = await request.json()
            payload = payload if isinstance(payload, dict) else {}
            if not str(payload.get("name") or "").strip():
                raise ValueError("Profile name is required.")
            if not str(payload.get("base_url") or "").strip():
                raise ValueError("Base URL is required.")
            return app.state.llm_store.add_profile(payload)
        except ValueError as exc:
            raise _fail(exc) from exc

    @app.put("/api/llm/profiles/{profile_id}")
    async def update_llm_profile(profile_id: str, request: Request):
        try:
            payload = await request.json()
            payload = payload if isinstance(payload, dict) else {}
            return app.state.llm_store.update_profile(profile_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _fail(exc) from exc

    @app.delete("/api/llm/profiles/{profile_id}")
    def delete_llm_profile(profile_id: str):
        try:
            app.state.llm_store.delete_profile(profile_id)
            return app.state.llm_store.list_profiles()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/llm/profiles/{profile_id}/activate")
    def activate_llm_profile(profile_id: str):
        try:
            return app.state.llm_store.set_active(profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/llm/test")
    async def test_llm_connection(request: Request):
        from .proofreader import Proofreader

        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            profile_id = payload.get("profile_id")
            if profile_id:
                data = app.state.llm_store.list_profiles(include_secrets=True)
                profile = next(
                    (p for p in data["profiles"] if str(p.get("id")) == str(profile_id)),
                    None,
                )
                if profile is None:
                    raise KeyError(f"Profile not found: {profile_id}")
                api_key = profile.get("api_key") or ""
                if not api_key:
                    # masked key means update flow never set a real one: pull from store
                    active = app.state.llm_store.get_active()
                    if active and str(active.get("id")) == str(profile_id):
                        api_key = active.get("api_key") or ""
            else:
                profile = app.state.llm_store.get_active()
                if profile is None:
                    return JSONResponse(
                        {"ok": False, "message": "没有可用的 API 配置。请先添加并激活一个配置。"},
                        status_code=400,
                    )
                api_key = profile.get("api_key") or ""
            proofreader = Proofreader(
                base_url=str(profile.get("base_url") or ""),
                model=str(profile.get("model") or ""),
                api_key=api_key,
                provider=str(profile.get("provider") or "openai"),
                disable_thinking=bool(profile.get("disable_thinking")),
                timeout=60.0,
            )
            return await asyncio.to_thread(proofreader.test_connection)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -------------------------------------------------------------- proofread

    def _active_proofreader():
        from .proofreader import Proofreader

        profile = app.state.llm_store.get_active()
        if profile is None:
            raise RuntimeError("No LLM API profile is active. Configure one in 设置 → AI 服务.")
        return Proofreader(
            base_url=str(profile.get("base_url") or ""),
            model=str(profile.get("model") or ""),
            api_key=str(profile.get("api_key") or "EMPTY"),
            provider=str(profile.get("provider") or "openai"),
            disable_thinking=bool(profile.get("disable_thinking")),
        )

    @app.post("/api/jobs/{job_id}/proofread")
    async def run_proofread(job_id: str):
        try:
            proofreader = _active_proofreader()
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        try:
            return await asyncio.to_thread(manager.proofread, job_id, proofreader)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception as exc:
            raise _fail(exc) from exc

    @app.get("/api/jobs/{job_id}/proofread")
    def get_proofread(job_id: str):
        try:
            return manager.get_proofread_result(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/proofread/apply")
    async def apply_proofread(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            ids = payload.get("ids") or []
            terms = payload.get("terms") or []
            if not isinstance(ids, list) or not isinstance(terms, list):
                raise ValueError("ids and terms must be lists.")
            return await asyncio.to_thread(manager.apply_proofread, job_id, ids, terms)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _fail(exc) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/jobs/{job_id}/alignment")
    def get_alignment(job_id: str):
        try:
            return manager.get_alignment_result(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/alignment/apply")
    async def apply_alignment(job_id: str, request: Request):
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise HTTPException(status_code=400, detail="ids must be a list of segment ids.")
        try:
            return await asyncio.to_thread(manager.apply_alignment, job_id, ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/api/jobs/{job_id}/clips")
    def list_clips(
        job_id: str,
        min_duration: float = 45.0,
        target_duration: float = 120.0,
        max_duration: float = 180.0,
        limit: int = 24,
        strategy: str = "rules",
    ):
        try:
            if strategy not in {"rules", "model"}:
                raise ValueError("strategy must be 'rules' or 'model'.")
            selector = None
            if strategy == "model":
                try:
                    selector = _active_proofreader()
                except RuntimeError as exc:
                    return JSONResponse({"detail": str(exc)}, status_code=503)
            # max_duration <= 0 表示不设时长上限，长度交给目标秒数评分与 AI 判断
            return {
                "clips": manager.list_clip_candidates(
                    job_id,
                    min_duration=min_duration,
                    target_duration=target_duration,
                    max_duration=max_duration if max_duration > 0 else None,
                    limit=limit,
                    selector=selector,
                )
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise _fail(exc) from exc

    @app.post("/api/jobs/{job_id}/clips/render")
    async def render_clip(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            payload = payload if isinstance(payload, dict) else {}
            # 每个渲染请求起一个 ffmpeg 编码进程,不限制会把机器拖垮;
            # 超出上限的请求在信号量上排队等待,而不是直接失败。
            async with clip_render_semaphore:
                clip = await asyncio.to_thread(
                    manager.render_clip,
                    job_id,
                    start=payload.get("start"),
                    end=payload.get("end"),
                    style_payload=payload.get("style") if isinstance(payload.get("style"), dict) else None,
                    name=payload.get("name"),
                )
            clip["download_url"] = f"/api/jobs/{job_id}/clips/{clip['filename']}"
            return clip
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception as exc:
            raise _fail(exc) from exc

    @app.get("/api/jobs/{job_id}/clips/{filename}")
    def download_clip(job_id: str, filename: str):
        try:
            path = manager.clip_download_path(job_id, filename)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Clip file is not ready.") from exc
        return FileResponse(path, filename=path.name)

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


def _runner_runtime_info(runner) -> dict[str, Any]:
    info = dict(runner.runtime_info())
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
