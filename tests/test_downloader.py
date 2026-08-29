"""yt-dlp 标题解析与 URL 任务展示名的单元测试。

不真正联网：monkeypatch downloader.subprocess.Popen 提供假输出流，
媒体文件预先落盘为 .mkv 以跳过 remux 分支。
"""
from __future__ import annotations

from pathlib import Path

import moss_transcribe_diarize.app.downloader as downloader
from moss_transcribe_diarize.app.jobs import _sanitize_display_name


class _FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.terminated = False

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def _run_download(tmp_path: Path, monkeypatch, lines: list[str], **kwargs):
    media = tmp_path / "input.mkv"
    media.write_bytes(b"\x00")

    captured: dict[str, object] = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProc([line.replace("{out}", str(tmp_path)) for line in lines])

    monkeypatch.setattr(downloader.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(downloader, "_find_js_runtime", lambda _yt: None)
    monkeypatch.setattr(downloader, "_find_ffmpeg_dir", lambda: None)

    ticks: list[tuple[float, dict]] = []
    result = downloader.download_with_yt_dlp(
        "https://example.com/watch?v=1",
        tmp_path,
        cookies_browser="none",
        progress_callback=lambda ratio, info: ticks.append((ratio, info)),
        **kwargs,
    )
    return result, captured, ticks


def test_cmd_contains_no_simulate_and_print_template(tmp_path, monkeypatch):
    _, captured, _ = _run_download(tmp_path, monkeypatch, ["mtd_title:t"])
    cmd = captured["cmd"]
    assert "--no-simulate" in cmd
    # --print 会隐式激活 quiet 关闭进度，必须显式 --progress 找回来。
    assert cmd.index("--progress") > cmd.index("--print")
    idx = cmd.index("--print")
    assert cmd[idx + 1] == f"{downloader._TITLE_PRINT_PREFIX}%(title)s"


def test_parses_printed_title_and_fires_callback_once(tmp_path, monkeypatch):
    titles: list[str] = []

    def on_title(raw: str) -> None:
        titles.append(raw)

    result, _, ticks = _run_download(
        tmp_path,
        monkeypatch,
        [
            "mtd_title:My Fake 视频 Title",
            "mtd_title:duplicate-should-be-ignored",
            "[download]  12.3% of ~ 10.00MiB at 512.00KiB/s ETA 01:20",
            "Destination: {out}" + ("\\input.mkv" if __import__("os").name == "nt" else "/input.mkv"),
            "[download] 100.0% of 10.00MiB at 1.00MiB/s ETA 00:01",
        ],
        title_callback=on_title,
    )
    assert isinstance(result, downloader.DownloadResult)
    assert result.title == "My Fake 视频 Title"
    assert result.path.name == "input.mkv"
    assert result.path.is_file()
    assert titles == ["My Fake 视频 Title"]
    # 进度照常转发：起点与收尾两个 tick 均被捕获。
    assert [round(ratio, 3) for ratio, _info in ticks][:2] == [0.123, 1.0]


def test_missing_title_falls_back_to_none(tmp_path, monkeypatch):
    result, _, _ticks = _run_download(
        tmp_path,
        monkeypatch,
        ["Destination: {out}" + ("\\input.mkv" if __import__("os").name == "nt" else "/input.mkv")],
    )
    assert result.title is None


def test_sanitize_display_name_strips_illegal_chars():
    cleaned = _sanitize_display_name(' <Bad>: "Name"? / \\ | * ')
    assert not (set(cleaned) & set('<>:"/\\|?*'))
    assert "\x00" not in _sanitize_display_name("bad\x01\x1fname")


def test_sanitize_display_name_collapses_space_and_truncates():
    assert _sanitize_display_name("a   b\t\tc") == "a b c"
    assert len(_sanitize_display_name("长" * 200)) <= 80
    assert len(_sanitize_display_name("a" * 300)) == 80
