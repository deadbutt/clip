"""桌面字幕浮窗。

为什么要有这个东西：浏览器里那个浮窗（Document Picture-in-Picture）是**浏览器画的窗口**，
网页没有任何 API 能让它透明到桌面。豆包那种"只有字浮在游戏上"的效果只能靠原生窗口。

Windows 上 tkinter 的 ``-transparentcolor`` 能做色键透明：把某一种颜色抠成全透明，
并且被抠掉的像素不接收鼠标事件，等于点在透明处会穿透到后面的窗口。
文字本身仍然完全不透明，再用一圈黑色描边保证在任何画面上都看得清。

这里只负责"显示"：识别和翻译仍然走已有的实时字幕服务，
数据来自 ``GET /api/live/sessions/<id>/subtitles``。
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

try:  # tkinter 在官方 Windows CPython 里是标准库；极简环境可能缺席
    import tkinter as tk
except Exception:  # pragma: no cover - 取决于运行环境
    tk = None  # type: ignore[assignment]

DEFAULT_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_POLL_SECONDS = 1.0
DEFAULT_WIDTH = 1000
DEFAULT_FONT_SIZE = 22

# 色键：这个颜色会被抠成全透明。
# 必须选一个"近乎黑"的颜色而不是洋红——文字抗锯齿会把字边和背景色混合，
# 用洋红就会在字周围出现明显的粉色描边；用近黑色则混成灰阶，几乎看不出来。
# 也不能用纯黑，因为描边本身是黑的，会被一起抠掉。
KEY_COLOR = "#0b0b0b"
PANEL_COLOR = "#0b1418"
SOURCE_COLOR = "#ffffff"
TRANSLATION_COLOR = "#8fe2b5"
STATUS_COLOR = "#9bb4b1"
OUTLINE_COLOR = "#000000"
GRIP_COLOR = "#1f3238"
FONT_FAMILY = "Microsoft YaHei"

# 描边：把文字在八个方向各画一遍黑色，再画本色，等于给文字描了圈边。
OUTLINE_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))

STATE_LABELS = {
    "starting": "正在启动",
    "recognizing": "正在识别",
    "listening": "等待下一段语音",
    "stopping": "正在停止",
    "stopped": "已停止",
    "idle": "未开启",
    "error": "识别失败",
}


def snapshot_url(base_url: str, session_id: str) -> str:
    return f"{base_url.rstrip('/')}/api/live/sessions/{session_id}/subtitles"


def fetch_snapshot(base_url: str, session_id: str, timeout: float = 4.0) -> dict[str, Any]:
    request = urllib.request.Request(
        snapshot_url(base_url, session_id), headers={"Cache-Control": "no-store"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_lines(snapshot: dict[str, Any], limit: int = 2) -> list[tuple[str, str]]:
    """取最近 ``limit`` 行，返回 ``[(原文, 译文), ...]``。

    先滤掉两侧都为空的空行，再取最后 ``limit`` 行：这样即使最新几行是空行，
    浮窗也会用更早的内容填满，不会突然变成一片空白。
    """
    rows = snapshot.get("rows") or []
    if not isinstance(rows, list):
        return []
    collected: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "").strip()
        translation = str(row.get("translation") or "").strip()
        if source or translation:
            collected.append((source, translation))
    return collected[-limit:] if limit > 0 else collected


def status_line(snapshot: dict[str, Any]) -> str:
    """浮窗顶部那行状态。识别正常时不写东西，避免长期占屏。"""
    state = str(snapshot.get("state") or "")
    error = snapshot.get("error")
    if error:
        return f"{STATE_LABELS.get(state, state)}：{error}"
    if state in {"stopped", "error", "idle"}:
        return STATE_LABELS.get(state, state)
    return ""


class OverlayLauncher:
    """按会话拉起/停止浮窗进程。

    网页点按钮 -> 服务端起一个独立进程；浮窗自己轮询字幕接口，
    所以它不依赖浏览器是否还开着，关掉网页浮窗照样继续。
    """

    module = "moss_transcribe_diarize.app.live_overlay"

    def __init__(self, spawn: Callable[[list[str]], Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._procs: dict[str, Any] = {}
        self._spawn = spawn

    def command(self, base_url: str, session_id: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            self.module,
            "--base-url",
            base_url.rstrip("/"),
            "--session",
            session_id,
        ]

    def _spawn_process(self, command: list[str]) -> Any:
        if self._spawn is not None:
            return self._spawn(command)
        # 不要再弹一个控制台窗口出来。
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(  # noqa: S603 - 固定命令行，参数来自本地会话 id
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )

    @staticmethod
    def _alive(proc: Any) -> bool:
        return proc is not None and getattr(proc, "poll", lambda: 0)() is None

    def running(self, session_id: str) -> bool:
        with self._lock:
            return self._alive(self._procs.get(session_id))

    def start(self, base_url: str, session_id: str) -> dict[str, Any]:
        with self._lock:
            proc = self._procs.get(session_id)
            if self._alive(proc):
                return {"running": True, "pid": getattr(proc, "pid", None)}  # 已在跑，不重复开
            proc = self._spawn_process(self.command(base_url, session_id))
            self._procs[session_id] = proc
            return {"running": True, "pid": getattr(proc, "pid", None)}

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            proc = self._procs.pop(session_id, None)
        if self._alive(proc):
            try:
                proc.terminate()
            except Exception:
                pass
        return {"running": False}

    def stop_all(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
            self._procs.clear()
        for proc in procs:
            if self._alive(proc):
                try:
                    proc.terminate()
                except Exception:
                    pass


overlay_launcher = OverlayLauncher()


class SubtitleOverlay:
    """无边框、置顶、背景透明的字幕窗。"""

    def __init__(
        self,
        root: "tk.Tk",
        base_url: str,
        session_id: str,
        *,
        font_size: int = DEFAULT_FONT_SIZE,
        width: int = DEFAULT_WIDTH,
        poll: float = DEFAULT_POLL_SECONDS,
        transparent: bool = True,
        lines_shown: int = 2,
    ) -> None:
        self.root = root
        self.base_url = base_url
        self.session_id = session_id
        self.font_size = font_size
        self.width = width
        self.poll = max(0.3, float(poll))
        self.lines_shown = max(1, int(lines_shown))
        self.pad = 14
        self.min_height = 80

        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._last_height = 0
        self._x = 120
        self._y = 120
        self._drag_origin: tuple[int, int] | None = None

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 1.0)
        except Exception:
            pass

        self.transparent = transparent and self._enable_transparency()
        background = KEY_COLOR if self.transparent else PANEL_COLOR
        self.canvas = tk.Canvas(
            root, bg=background, highlightthickness=0, bd=0, width=width, height=self.min_height
        )
        self.canvas.pack(fill="both", expand=True)

        self._build_menu()
        self._bind_events()
        self._apply_geometry(self.min_height)
        self._render([], status="正在连接字幕服务…")

        threading.Thread(target=self._poll_loop, name="mtd-overlay-poll", daemon=True).start()
        root.after(120, self._drain)

    # ---------- 窗口属性 ----------

    def _enable_transparency(self) -> bool:
        try:
            self.root.attributes("-transparentcolor", KEY_COLOR)
            return True
        except Exception:
            # 非 Windows 或该 Tk 构建不支持：退回不透明深色面板，至少能用。
            return False

    def _apply_geometry(self, height: int, *, force: bool = False) -> None:
        height = max(self.min_height, int(height))
        if height == self._last_height and not force:
            return
        self._last_height = height
        self.root.geometry(f"{self.width}x{height}+{self._x}+{self._y}")

    # ---------- 交互 ----------

    def _build_menu(self) -> None:
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="字号 +", command=lambda: self._resize_font(+2))
        self.menu.add_command(label="字号 −", command=lambda: self._resize_font(-2))
        self.menu.add_separator()
        self.menu.add_command(label="关闭浮窗", command=self.close)

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.canvas.bind("<Button-3>", self._popup_menu)
        self.canvas.bind("<MouseWheel>", lambda event: self._resize_font(2 if event.delta > 0 else -2))
        self.root.bind("<Escape>", lambda _event: self.close())
        # 方向键微调位置，透明区域不接收点击时也能挪。
        for key, delta in {"Left": (-8, 0), "Right": (8, 0), "Up": (0, -8), "Down": (0, 8)}.items():
            self.root.bind(f"<{key}>", lambda _e, d=delta: self._nudge(*d))

    def _drag_start(self, event: "tk.Event") -> None:
        self._drag_origin = (event.x_root - self._x, event.y_root - self._y)

    def _drag_move(self, event: "tk.Event") -> None:
        if self._drag_origin is None:
            return
        self._x = event.x_root - self._drag_origin[0]
        self._y = event.y_root - self._drag_origin[1]
        self.root.geometry(f"{self.width}x{self._last_height}+{self._x}+{self._y}")

    def _drag_end(self, _event: "tk.Event") -> None:
        self._drag_origin = None

    def _nudge(self, dx: int, dy: int) -> None:
        self._x += dx
        self._y += dy
        self.root.geometry(f"{self.width}x{self._last_height}+{self._x}+{self._y}")

    def _popup_menu(self, event: "tk.Event") -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _resize_font(self, delta: int) -> None:
        self.font_size = max(10, min(60, self.font_size + delta))
        self._render(self._current_lines, status=self._current_status)

    # ---------- 数据 ----------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(("data", fetch_snapshot(self.base_url, self.session_id)))
            except urllib.error.URLError as exc:
                self._queue.put(("error", f"连不上字幕服务：{exc.reason}"))
            except Exception as exc:  # pragma: no cover - 兜底
                self._queue.put(("error", f"读取字幕失败：{exc}"))
            self._stop.wait(self.poll)

    def _drain(self) -> None:
        payload: dict[str, Any] | None = None
        error: str | None = None
        changed = False
        while True:
            try:
                kind, value = self._queue.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "data":
                payload = value
            else:
                error = str(value)
        if changed:
            if error:
                self._render([], status=error)
            elif payload is not None:
                self._render(latest_lines(payload, self.lines_shown), status=status_line(payload))
        if not self._stop.is_set():
            self.root.after(150, self._drain)

    # ---------- 绘制 ----------

    def _draw_run(
        self, y: int, text: str, size: int, color: str, outline: bool
    ) -> int:
        """画一段（可换行的）文字并返回下一段的 y。"""
        font = (FONT_FAMILY, size)
        width = self.width - self.pad * 2
        for dx, dy in OUTLINE_OFFSETS if outline else ((0, 0),):
            self.canvas.create_text(
                self.pad + dx,
                y + dy,
                text=text,
                fill=OUTLINE_COLOR,
                font=font,
                anchor="nw",
                width=width,
            )
        item = self.canvas.create_text(
            self.pad, y, text=text, fill=color, font=font, anchor="nw", width=width
        )
        box = self.canvas.bbox(item)
        bottom = box[3] if box else y + size + 6
        return int(bottom)

    def _render(self, lines: list[tuple[str, str]], status: str = "") -> None:
        self._current_lines = list(lines)
        self._current_status = status

        self.canvas.delete("all")
        # 左上角的小握把：透明区域不接收点击，留着它才好拖。
        self.canvas.create_rectangle(0, 0, 20, 20, fill=GRIP_COLOR, outline="")
        for offset in (6, 10, 14):
            self.canvas.create_rectangle(offset, 9, offset + 2, 11, fill="#7fb3ad", outline="")

        y = self.pad
        if status:
            y = self._draw_run(y, status, max(10, self.font_size - 8), STATUS_COLOR, False) + 6
        outline = self.transparent  # 透明模式靠描边保证可读；不透明模式有底色就不必描
        for source, translation in lines:
            if source:
                y = self._draw_run(y, source, self.font_size, SOURCE_COLOR, outline) + 2
            if translation:
                y = self._draw_run(y, translation, self.font_size, TRANSLATION_COLOR, outline) + 2
            y += 8
        if not lines and not status:
            y = self._draw_run(y, "等待下一段语音…", self.font_size, STATUS_COLOR, outline)
        self._apply_geometry(y + self.pad)

    # ---------- 生命周期 ----------

    def close(self) -> None:
        self._stop.set()
        try:
            self.root.destroy()
        except Exception:
            pass


def run_gui(args: argparse.Namespace) -> int:
    if tk is None:
        print("当前 Python 没有 tkinter，无法显示桌面浮窗。", file=sys.stderr)
        return 2
    root = tk.Tk()
    overlay = SubtitleOverlay(
        root,
        args.base_url,
        args.session,
        font_size=args.font_size,
        width=args.width,
        poll=args.poll,
        transparent=not args.no_transparent,
        lines_shown=args.lines,
    )
    try:
        root.mainloop()
    finally:
        overlay.close()
    return 0


def run_once(args: argparse.Namespace) -> int:
    """取一次数据打印出来，不需要图形界面——用于自检和排查。"""
    try:
        snapshot = fetch_snapshot(args.base_url, args.session)
    except Exception as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 1
    status = status_line(snapshot)
    lines = latest_lines(snapshot, args.lines)
    print(f"state={snapshot.get('state')} rows={len(snapshot.get('rows') or [])}")
    if status:
        print(f"[状态] {status}")
    if not lines:
        print("[空] 还没有可显示的字幕")
    for source, translation in lines:
        print(f"[原文] {source}")
        print(f"[译文] {translation}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="桌面字幕浮窗（无边框 / 置顶 / 背景透明）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="字幕服务地址")
    parser.add_argument("--session", required=True, help="直播会话 id")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE, help="字号，默认 22")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="窗口宽度，默认 1000")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS, help="轮询间隔（秒）")
    parser.add_argument("--lines", type=int, default=2, help="显示最近几行，默认 2")
    parser.add_argument(
        "--no-transparent",
        action="store_true",
        help="不用色键透明，改成不透明深色面板（排查显示问题时用）",
    )
    parser.add_argument("--once", action="store_true", help="只取一次数据打印出来，不开窗口")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_once(args) if args.once else run_gui(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
