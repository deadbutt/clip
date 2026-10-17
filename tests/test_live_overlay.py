import unittest

from moss_transcribe_diarize.app.live_overlay import (
    OverlayLauncher,
    build_parser,
    latest_lines,
    snapshot_url,
    status_line,
)


def _snapshot(*rows, state="recognizing", **extra):
    return {"state": state, "rows": list(rows), **extra}


class SnapshotUrlTest(unittest.TestCase):
    def test_trailing_slash_is_tolerated(self):
        expected = "http://127.0.0.1:7860/api/live/sessions/abc/subtitles"
        self.assertEqual(snapshot_url("http://127.0.0.1:7860", "abc"), expected)
        self.assertEqual(snapshot_url("http://127.0.0.1:7860/", "abc"), expected)


class LatestLinesTest(unittest.TestCase):
    def test_returns_the_most_recent_rows(self):
        snap = _snapshot(
            {"source": "a", "translation": "甲"},
            {"source": "b", "translation": "乙"},
            {"source": "c", "translation": "丙"},
        )
        self.assertEqual(latest_lines(snap, 2), [("b", "乙"), ("c", "丙")])

    def test_skips_rows_where_both_sides_are_blank(self):
        snap = _snapshot(
            {"source": "kept", "translation": "留"},
            {"source": "   ", "translation": ""},
            {"source": "", "translation": "只有译文"},
        )
        self.assertEqual(latest_lines(snap, 2), [("kept", "留"), ("", "只有译文")])

    def test_missing_or_broken_rows_do_not_raise(self):
        self.assertEqual(latest_lines({}), [])
        self.assertEqual(latest_lines({"rows": None}), [])
        self.assertEqual(latest_lines({"rows": ["not-a-dict", 42]}), [])
        self.assertEqual(latest_lines({"rows": [{"source": None, "translation": None}]}), [])

    def test_untranslated_row_keeps_source_only(self):
        snap = _snapshot({"source": "pending", "translation": ""})
        self.assertEqual(latest_lines(snap), [("pending", "")])


class StatusLineTest(unittest.TestCase):
    def test_running_states_stay_silent_so_the_overlay_does_not_clutter(self):
        for state in ("recognizing", "listening", "starting", "stopping"):
            with self.subTest(state=state):
                self.assertEqual(status_line({"state": state}), "")

    def test_terminal_states_are_reported(self):
        self.assertEqual(status_line({"state": "stopped"}), "已停止")
        self.assertEqual(status_line({"state": "idle"}), "未开启")
        self.assertEqual(status_line({"state": "error"}), "识别失败")

    def test_error_detail_wins_over_state(self):
        line = status_line({"state": "error", "error": "拉流失败"})
        self.assertIn("识别失败", line)
        self.assertIn("拉流失败", line)


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid
        self.terminated = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0


class OverlayLauncherTest(unittest.TestCase):
    def _launcher(self):
        calls = []

        def spawn(command):
            calls.append(command)
            return _FakeProc()

        return OverlayLauncher(spawn=spawn), calls

    def test_start_spawns_a_detached_module_process(self):
        launcher, calls = self._launcher()
        result = launcher.start("http://127.0.0.1:7860/", "sess-1")
        self.assertTrue(result["running"])
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertIn("-m", command)
        self.assertIn("moss_transcribe_diarize.app.live_overlay", command)
        self.assertIn("--session", command)
        self.assertIn("sess-1", command)
        self.assertIn("--base-url", command)
        self.assertIn("http://127.0.0.1:7860", command)

    def test_start_while_running_does_not_spawn_twice(self):
        launcher, calls = self._launcher()
        launcher.start("http://127.0.0.1:7860", "sess-1")
        launcher.start("http://127.0.0.1:7860", "sess-1")
        self.assertEqual(len(calls), 1)

    def test_crashed_process_is_replaced(self):
        procs = []

        def spawn(command):
            proc = _FakeProc()
            procs.append(proc)
            return proc

        launcher = OverlayLauncher(spawn=spawn)
        launcher.start("http://127.0.0.1:7860", "sess-1")
        procs[0].returncode = 1  # 进程挂了
        self.assertFalse(launcher.running("sess-1"))
        launcher.start("http://127.0.0.1:7860", "sess-1")
        self.assertEqual(len(procs), 2)
        self.assertTrue(launcher.running("sess-1"))

    def test_stop_terminates_and_forgets(self):
        launcher, _calls = self._launcher()
        launcher.start("http://127.0.0.1:7860", "sess-1")
        result = launcher.stop("sess-1")
        self.assertFalse(result["running"])
        self.assertFalse(launcher.running("sess-1"))
        # 再停一次不能炸
        self.assertFalse(launcher.stop("sess-1")["running"])

    def test_sessions_are_tracked_independently(self):
        launcher, calls = self._launcher()
        launcher.start("http://127.0.0.1:7860", "a")
        launcher.start("http://127.0.0.1:7860", "b")
        self.assertEqual(len(calls), 2)
        launcher.stop("a")
        self.assertFalse(launcher.running("a"))
        self.assertTrue(launcher.running("b"))


class ParserTest(unittest.TestCase):
    def test_session_is_required(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_defaults(self):
        args = build_parser().parse_args(["--session", "abc"])
        self.assertEqual(args.base_url, "http://127.0.0.1:7860")
        self.assertEqual(args.font_size, 22)
        self.assertEqual(args.lines, 2)
        self.assertFalse(args.once)
        self.assertFalse(args.no_transparent)


if __name__ == "__main__":
    unittest.main()
