import importlib
import threading
import types
import unittest
from unittest import mock
import numpy as np


def _install_dependency_stubs() -> None:
    if "riva.client" not in importlib.sys.modules:
        riva_module = types.ModuleType("riva")
        client_module = types.ModuleType("riva.client")

        class DummyAuth:
            def __init__(self, *args, **kwargs):
                pass

        class DummyASRService:
            def __init__(self, *args, **kwargs):
                pass

        class DummyRecognitionConfig:
            def __init__(self, *args, **kwargs):
                pass

        class DummyAudioEncoding:
            LINEAR_PCM = "LINEAR_PCM"

        client_module.Auth = DummyAuth
        client_module.ASRService = DummyASRService
        client_module.RecognitionConfig = DummyRecognitionConfig
        client_module.AudioEncoding = DummyAudioEncoding
        riva_module.client = client_module
        importlib.sys.modules["riva"] = riva_module
        importlib.sys.modules["riva.client"] = client_module

    if "sounddevice" not in importlib.sys.modules:
        sd_module = types.ModuleType("sounddevice")

        class DummyInputStream:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        sd_module.InputStream = DummyInputStream
        importlib.sys.modules["sounddevice"] = sd_module

    if "dotenv" not in importlib.sys.modules:
        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.load_dotenv = lambda: None
        importlib.sys.modules["dotenv"] = dotenv_module

    if "pynput.keyboard" not in importlib.sys.modules:
        pynput_module = types.ModuleType("pynput")
        keyboard_module = types.ModuleType("pynput.keyboard")

        class DummyController:
            def type(self, text):
                pass

        class DummyListener:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def join(self, timeout=None):
                pass

            def suppress_event(self):
                pass

        class DummyKey:
            esc = "esc"
            ctrl = "ctrl"
            ctrl_l = "ctrl_l"
            ctrl_r = "ctrl_r"
            shift = "shift"
            shift_l = "shift_l"
            shift_r = "shift_r"
            left = "left"
            right = "right"

        keyboard_module.Controller = DummyController
        keyboard_module.Listener = DummyListener
        keyboard_module.Key = DummyKey
        keyboard_module.KeyCode = object
        pynput_module.keyboard = keyboard_module
        importlib.sys.modules["pynput"] = pynput_module
        importlib.sys.modules["pynput.keyboard"] = keyboard_module


_install_dependency_stubs()
ptt_whisper = importlib.import_module("whispertocode.app")
cli_module = importlib.import_module("whispertocode.cli")
overlay_module = importlib.import_module("whispertocode.overlay")


def _make_app() -> "ptt_whisper.HoldToTalkRiva":
    app = object.__new__(ptt_whisper.HoldToTalkRiva)
    app._lock = threading.Lock()
    app._stop_event = threading.Event()
    app._output_mode = ptt_whisper.OUTPUT_MODE_RAW
    app._keyboard = mock.Mock()
    app._peak_level = 0.05
    app._min_level = 0.01
    return app


class ModesAndFallbackTests(unittest.TestCase):
    def test_parse_args_default_mode_is_raw(self):
        args = cli_module.parse_args([])
        self.assertEqual(args.mode, ptt_whisper.OUTPUT_MODE_RAW)

    def test_parse_args_accepts_smart_mode(self):
        args = cli_module.parse_args(["--mode", ptt_whisper.OUTPUT_MODE_SMART])
        self.assertEqual(args.mode, ptt_whisper.OUTPUT_MODE_SMART)

    def test_parse_args_defaults_to_tray_without_debug_console(self):
        args = cli_module.parse_args([])
        self.assertFalse(args.no_tray)
        self.assertFalse(args.debug_console)

    def test_parse_args_can_disable_tray_and_enable_debug_console(self):
        args = cli_module.parse_args(["--no-tray", "--debug-console"])
        self.assertTrue(args.no_tray)
        self.assertTrue(args.debug_console)

    def test_local_special_keys_switch_modes(self):
        app = _make_app()
        with mock.patch("builtins.print"):
            handled_right = app._handle_local_special_key("M")
            handled_left = app._handle_local_special_key("K")
        self.assertTrue(handled_right)
        self.assertTrue(handled_left)
        self.assertEqual(app._get_output_mode(), ptt_whisper.OUTPUT_MODE_RAW)

    def test_local_special_key_ignores_unknown_key(self):
        app = _make_app()
        handled = app._handle_local_special_key("H")
        self.assertFalse(handled)
        self.assertEqual(app._get_output_mode(), ptt_whisper.OUTPUT_MODE_RAW)

    def test_local_escape_requests_shutdown(self):
        app = _make_app()
        app.request_shutdown = mock.Mock()
        handled = app._handle_local_console_char("\x1b")
        self.assertTrue(handled)
        app.request_shutdown.assert_called_once_with("Esc")

    def test_startup_banner_windows_mentions_local_hotkeys(self):
        app = _make_app()
        app.hold_delay_sec = 0.5
        with mock.patch("whispertocode.app.os.name", "nt"):
            lines = app._startup_banner_lines()
        self.assertIn("Current mode: RAW", lines)
        self.assertTrue(
            any("Local hotkeys:" in line and "enabled in this console window" in line for line in lines)
        )

    def test_startup_banner_mentions_shift_for_recording(self):
        app = _make_app()
        app.hold_delay_sec = 0.5
        app._tray_enabled = False
        with mock.patch("whispertocode.app.os.name", "nt"):
            lines = app._startup_banner_lines()
        self.assertIn("Hold Shift for at least 0.5s to record, release Shift to transcribe and type.", lines)

    def test_startup_banner_non_windows_reports_unavailable_hotkeys(self):
        app = _make_app()
        app.hold_delay_sec = 0.5
        app._tray_enabled = False
        with mock.patch("whispertocode.app.os.name", "posix"):
            lines = app._startup_banner_lines()
        self.assertTrue(any("Local hotkeys: unavailable on this OS" in line for line in lines))

    def test_shift_press_starts_hold_timer(self):
        app = _make_app()
        app.hold_delay_sec = 0.5
        app._ctrl_count = 0
        app._press_token = 0
        app._hold_timer = None
        timer = mock.Mock()
        with mock.patch("whispertocode.app.threading.Timer", return_value=timer):
            app._on_press(ptt_whisper.keyboard.Key.shift)
        self.assertEqual(app._ctrl_count, 1)
        self.assertEqual(app._press_token, 1)
        self.assertIs(app._hold_timer, timer)
        timer.start.assert_called_once()

    def test_startup_banner_in_tray_mode_mentions_tray_controls(self):
        app = _make_app()
        app.hold_delay_sec = 0.5
        app._tray_enabled = True
        lines = app._startup_banner_lines()
        self.assertTrue(any("Tray controls:" in line for line in lines))

    def test_tray_console_toggle_delegates_to_console_visibility_handler(self):
        app = _make_app()
        app._set_console_visibility = mock.Mock()
        app._handle_tray_show_console(None, None)
        app._set_console_visibility.assert_called_once_with(True, "tray")
        app._set_console_visibility.reset_mock()
        app._handle_tray_hide_console(None, None)
        app._set_console_visibility.assert_called_once_with(False, "tray")

    def test_set_output_mode_updates_overlay_mode(self):
        app = _make_app()
        app._overlay_controller = mock.Mock()
        with mock.patch("builtins.print"):
            app._set_output_mode(ptt_whisper.OUTPUT_MODE_SMART, "test")
        app._overlay_controller.set_mode.assert_called_once_with(ptt_whisper.OUTPUT_MODE_SMART)

    def test_audio_callback_updates_overlay_level_while_recording(self):
        app = _make_app()
        app._recording = True
        app._chunks = []
        app._overlay_controller = mock.Mock()
        frame = np.array([[0.5], [-0.5], [0.25], [-0.25]], dtype=np.float32)
        app._audio_callback(frame, frames=4, time_info=None, status=None)
        self.assertEqual(len(app._chunks), 1)
        app._overlay_controller.update_level.assert_called_once()
        level_value = app._overlay_controller.update_level.call_args.args[0]
        self.assertGreater(level_value, 0.0)

    def test_audio_callback_keeps_headroom_for_loud_voice(self):
        app = _make_app()
        app._recording = True
        app._chunks = []
        app._overlay_controller = mock.Mock()

        loud_frame = np.array([[0.6], [-0.6], [0.6], [-0.6]], dtype=np.float32)
        medium_frame = np.array([[0.3], [-0.3], [0.3], [-0.3]], dtype=np.float32)

        app._audio_callback(loud_frame, frames=4, time_info=None, status=None)
        first_level = app._overlay_controller.update_level.call_args.args[0]
        app._audio_callback(medium_frame, frames=4, time_info=None, status=None)
        second_level = app._overlay_controller.update_level.call_args.args[0]

        self.assertLess(first_level, 1.0)
        self.assertGreater(first_level, 0.0)
        self.assertGreater(second_level, 0.0)
        self.assertLess(second_level, first_level)

    def test_start_recording_shows_overlay(self):
        app = _make_app()
        app._overlay_controller = mock.Mock()
        app._recording = False
        app._transcribing = False
        app._chunks = []
        app.sample_rate = 16000
        stream = mock.Mock()
        with (
            mock.patch("whispertocode.app.sd.InputStream", return_value=stream),
            mock.patch("builtins.print"),
        ):
            app._start_recording()
        app._overlay_controller.show_recording.assert_called_once_with(ptt_whisper.OUTPUT_MODE_RAW)

    def test_stop_recording_hides_overlay(self):
        app = _make_app()
        app._overlay_controller = mock.Mock()
        app._recording = True
        app._chunks = []
        app._stream = None
        with mock.patch("builtins.print"):
            app._stop_recording()
        app._overlay_controller.hide.assert_called_once()

    def test_start_overlay_initialization_failure_raises_runtime_error(self):
        app = _make_app()
        app._overlay_controller = None
        app._create_overlay_controller = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            app._start_overlay()

    def test_start_overlay_initialization_success_stores_controller(self):
        app = _make_app()
        app._overlay_controller = None
        controller = mock.Mock()
        app._create_overlay_controller = mock.Mock(return_value=controller)
        app._start_overlay()
        controller.start.assert_called_once()
        self.assertIs(app._overlay_controller, controller)

    def test_set_console_visibility_allocates_console_if_missing(self):
        app = _make_app()
        app._debug_console = False
        app._refresh_tray_menu = mock.Mock()
        app._redirect_stdio_to_console = mock.Mock()
        app._console_visible = False

        state = {"hwnd": 0}

        def _get_console_window():
            return state["hwnd"]

        def _alloc_console():
            state["hwnd"] = 101
            return 1

        kernel32 = types.SimpleNamespace(
            GetConsoleWindow=mock.Mock(side_effect=_get_console_window),
            AllocConsole=mock.Mock(side_effect=_alloc_console),
        )
        user32 = types.SimpleNamespace(ShowWindow=mock.Mock(return_value=1))
        fake_ctypes = types.SimpleNamespace(
            windll=types.SimpleNamespace(kernel32=kernel32, user32=user32)
        )

        with (
            mock.patch("whispertocode.app.os.name", "nt"),
            mock.patch("whispertocode.tray_support.os.name", "nt"),
            mock.patch.dict(importlib.sys.modules, {"ctypes": fake_ctypes}),
            mock.patch("builtins.print"),
        ):
            result = app._set_console_visibility(True, "tray")

        self.assertTrue(result)
        kernel32.AllocConsole.assert_called_once()
        app._redirect_stdio_to_console.assert_called_once()
        user32.ShowWindow.assert_called_once_with(101, ptt_whisper.WINDOWS_SW_SHOW)

    def test_smart_failure_without_output_falls_back_to_raw(self):
        app = _make_app()
        app._rewrite_text_streaming = mock.Mock(
            return_value=(False, RuntimeError("nemotron timeout"))
        )
        with mock.patch("builtins.print"):
            app._type_output_text("raw text", ptt_whisper.OUTPUT_MODE_SMART)
        app._keyboard.type.assert_called_once_with("raw text")

    def test_smart_failure_after_partial_output_does_not_append_raw(self):
        app = _make_app()
        app._rewrite_text_streaming = mock.Mock(
            return_value=(True, RuntimeError("stream interrupted"))
        )
        with mock.patch("builtins.print"):
            app._type_output_text("raw text", ptt_whisper.OUTPUT_MODE_SMART)
        app._keyboard.type.assert_not_called()

    def test_main_returns_error_when_run_raises_runtime_error(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode=ptt_whisper.OUTPUT_MODE_RAW,
            no_tray=False,
            debug_console=False,
        )
        app = mock.Mock()
        app.run.side_effect = RuntimeError("overlay failed")
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.HoldToTalkRiva", return_value=app),
            mock.patch("whispertocode.cli.signal.signal"),
            mock.patch("builtins.print"),
        ):
            code = cli_module.main()
        self.assertEqual(code, 1)


class OverlayPlacementTests(unittest.TestCase):
    def test_capsule_places_bottom_center(self):
        geometry = types.SimpleNamespace(
            x=lambda: 100,
            y=lambda: 50,
            width=lambda: 800,
            height=lambda: 600,
        )
        screen = types.SimpleNamespace(availableGeometry=lambda: geometry)
        qt_gui = types.SimpleNamespace(
            QGuiApplication=types.SimpleNamespace(primaryScreen=lambda: screen)
        )
        widget = mock.Mock()
        widget.width.return_value = 150
        widget.height.return_value = 100

        overlay = object.__new__(overlay_module._CapsuleOverlayWidget)
        overlay._qt_gui = qt_gui
        overlay._widget = widget
        overlay._target_opacity = 1.0
        overlay._current_opacity = 1.0

        overlay._place_bottom_center()

        widget.move.assert_called_once_with(425, 530)


if __name__ == "__main__":
    unittest.main()
