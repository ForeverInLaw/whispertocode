import importlib
import threading
import types
import unittest
from unittest import mock


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
ptt_whisper = importlib.import_module("ptt_whisper")


def _make_app() -> "ptt_whisper.HoldToTalkRiva":
    app = object.__new__(ptt_whisper.HoldToTalkRiva)
    app._lock = threading.Lock()
    app._stop_event = threading.Event()
    app._output_mode = ptt_whisper.OUTPUT_MODE_RAW
    app._keyboard = mock.Mock()
    return app


class ModesAndFallbackTests(unittest.TestCase):
    def test_parse_args_default_mode_is_raw(self):
        args = ptt_whisper.parse_args([])
        self.assertEqual(args.mode, ptt_whisper.OUTPUT_MODE_RAW)

    def test_parse_args_accepts_smart_mode(self):
        args = ptt_whisper.parse_args(["--mode", ptt_whisper.OUTPUT_MODE_SMART])
        self.assertEqual(args.mode, ptt_whisper.OUTPUT_MODE_SMART)

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
        with mock.patch("ptt_whisper.os.name", "nt"):
            lines = app._startup_banner_lines()
        self.assertIn("Current mode: RAW", lines)
        self.assertTrue(
            any("Local hotkeys:" in line and "enabled in this console window" in line for line in lines)
        )

    def test_startup_banner_non_windows_reports_unavailable_hotkeys(self):
        app = _make_app()
        app.hold_delay_sec = 0.5
        with mock.patch("ptt_whisper.os.name", "posix"):
            lines = app._startup_banner_lines()
        self.assertTrue(any("Local hotkeys: unavailable on this OS" in line for line in lines))

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


if __name__ == "__main__":
    unittest.main()
