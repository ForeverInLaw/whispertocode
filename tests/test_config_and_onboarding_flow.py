import importlib
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


def _install_dependency_stubs() -> None:
    if "riva.client" not in importlib.sys.modules:
        riva_module = types.ModuleType("riva")
        client_module = types.ModuleType("riva.client")
        client_module.Auth = object
        client_module.ASRService = object
        client_module.RecognitionConfig = object
        client_module.AudioEncoding = types.SimpleNamespace(LINEAR_PCM="LINEAR_PCM")
        riva_module.client = client_module
        importlib.sys.modules["riva"] = riva_module
        importlib.sys.modules["riva.client"] = client_module

    if "sounddevice" not in importlib.sys.modules:
        sd_module = types.ModuleType("sounddevice")
        sd_module.InputStream = object
        importlib.sys.modules["sounddevice"] = sd_module

    if "dotenv" not in importlib.sys.modules:
        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.load_dotenv = lambda: None
        importlib.sys.modules["dotenv"] = dotenv_module

    if "pynput.keyboard" not in importlib.sys.modules:
        pynput_module = types.ModuleType("pynput")
        keyboard_module = types.ModuleType("pynput.keyboard")
        keyboard_module.Controller = object
        keyboard_module.Listener = object
        keyboard_module.Key = types.SimpleNamespace(
            esc="esc",
            ctrl="ctrl",
            ctrl_l="ctrl_l",
            ctrl_r="ctrl_r",
            left="left",
            right="right",
            shift="shift",
            shift_l="shift_l",
            shift_r="shift_r",
        )
        keyboard_module.KeyCode = object
        pynput_module.keyboard = keyboard_module
        importlib.sys.modules["pynput"] = pynput_module
        importlib.sys.modules["pynput.keyboard"] = keyboard_module


_install_dependency_stubs()
cli_module = importlib.import_module("whispertocode.cli")
config_store = importlib.import_module("whispertocode.config_store")
onboarding_module = importlib.import_module("whispertocode.onboarding")


class ConfigAndOnboardingFlowTests(unittest.TestCase):
    def test_parse_args_accepts_onboarding_flag(self):
        args = cli_module.parse_args(["--onboarding"])
        self.assertTrue(args.onboarding)

    def test_main_runs_onboarding_when_key_missing(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="raw",
            no_tray=False,
            debug_console=False,
            onboarding=False,
        )
        configured = config_store.AppSettings(nvidia_api_key="new-key")
        resolved = config_store.AppSettings()
        app = mock.Mock()
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: False)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=resolved),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={}),
            mock.patch("whispertocode.cli.run_onboarding", return_value=configured) as onboarding_mock,
            mock.patch("whispertocode.cli.save_config_json") as save_mock,
            mock.patch("whispertocode.cli.HoldToTalkRiva", return_value=app),
            mock.patch("whispertocode.cli.signal.signal"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 0)
        # No config file yet: this really is a first run, so the dialog opens as
        # the setup wizard rather than as the Settings dialog.
        onboarding_mock.assert_called_once_with(
            resolved,
            mode=onboarding_module.MODE_SETUP,
        )
        save_mock.assert_called_once_with(configured)
        app.run.assert_called_once()

    def test_main_returns_error_when_onboarding_canceled(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="raw",
            no_tray=False,
            debug_console=False,
            onboarding=False,
        )
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: False)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=config_store.AppSettings()),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={}),
            mock.patch("whispertocode.cli.run_onboarding", return_value=None),
            mock.patch("whispertocode.cli.HoldToTalkRiva") as app_ctor_mock,
            mock.patch("builtins.print"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 1)
        app_ctor_mock.assert_not_called()

    def test_main_auto_migrates_env_setup_to_json_when_missing_config(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="raw",
            no_tray=False,
            debug_console=False,
            onboarding=False,
        )
        resolved = config_store.AppSettings(nvidia_api_key="from-env")
        app = mock.Mock()
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: False)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=resolved),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={"NVIDIA_API_KEY": "from-env"}),
            mock.patch("whispertocode.cli.run_onboarding") as onboarding_mock,
            mock.patch("whispertocode.cli.save_config_json") as save_mock,
            mock.patch("whispertocode.cli.HoldToTalkRiva", return_value=app),
            mock.patch("whispertocode.cli.signal.signal"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 0)
        onboarding_mock.assert_not_called()
        save_mock.assert_called_once_with(resolved)

    def test_local_backend_starts_without_api_key_in_raw_mode(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="raw",
            backend="local",
            no_tray=False,
            debug_console=False,
            onboarding=False,
        )
        app = mock.Mock()
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: True)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=config_store.AppSettings()),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={}),
            mock.patch("whispertocode.cli.run_onboarding") as onboarding_mock,
            mock.patch("whispertocode.cli.HoldToTalkRiva", return_value=app) as app_ctor_mock,
            mock.patch("whispertocode.cli.signal.signal"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 0)
        onboarding_mock.assert_not_called()
        self.assertEqual(app_ctor_mock.call_args.kwargs["settings"].stt_backend, "local")

    def test_local_backend_still_demands_key_for_smart_mode(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="smart",
            backend="local",
            no_tray=False,
            debug_console=False,
            onboarding=False,
        )
        resolved = config_store.AppSettings()
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: True)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=resolved),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={}),
            mock.patch("whispertocode.cli.run_onboarding", return_value=None) as onboarding_mock,
            mock.patch("whispertocode.cli.HoldToTalkRiva"),
            mock.patch("builtins.print"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 1)
        # A config already exists, so the missing key is one value to fix on an
        # installed app, not a fresh install to walk through.
        onboarding_mock.assert_called_once_with(
            resolved,
            mode=onboarding_module.MODE_SETTINGS,
        )

    def test_onboarding_flag_on_existing_config_opens_as_settings(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="raw",
            no_tray=False,
            debug_console=False,
            onboarding=True,
        )
        resolved = config_store.AppSettings(nvidia_api_key="already-set")
        app = mock.Mock()
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: True)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=resolved),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={}),
            mock.patch("whispertocode.cli.run_onboarding", return_value=resolved) as onboarding_mock,
            mock.patch("whispertocode.cli.save_config_json"),
            mock.patch("whispertocode.cli.HoldToTalkRiva", return_value=app),
            mock.patch("whispertocode.cli.signal.signal"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 0)
        onboarding_mock.assert_called_once_with(
            resolved,
            mode=onboarding_module.MODE_SETTINGS,
        )

    def test_onboarding_flag_without_config_still_opens_as_setup(self):
        args = types.SimpleNamespace(
            sample_rate=16000,
            language="auto",
            hold_delay=0.5,
            mode="raw",
            no_tray=False,
            debug_console=False,
            onboarding=True,
        )
        resolved = config_store.AppSettings()
        app = mock.Mock()
        with (
            mock.patch("whispertocode.cli.parse_args", return_value=args),
            mock.patch("whispertocode.cli.get_config_path", return_value=types.SimpleNamespace(exists=lambda: False)),
            mock.patch("whispertocode.cli.resolve_settings", return_value=resolved),
            mock.patch("whispertocode.cli.load_config_json", return_value={}),
            mock.patch("whispertocode.cli.load_env_fallback", return_value={}),
            mock.patch("whispertocode.cli.run_onboarding", return_value=resolved) as onboarding_mock,
            mock.patch("whispertocode.cli.save_config_json"),
            mock.patch("whispertocode.cli.HoldToTalkRiva", return_value=app),
            mock.patch("whispertocode.cli.signal.signal"),
        ):
            code = cli_module.main()

        self.assertEqual(code, 0)
        onboarding_mock.assert_called_once_with(
            resolved,
            mode=onboarding_module.MODE_SETUP,
        )

    def test_has_custom_advanced_settings_ignores_first_page_fields(self):
        defaults = config_store.AppSettings()
        self.assertFalse(onboarding_module.has_custom_advanced_settings(defaults))
        # The API key, the backend and the local-model fields all live on the
        # first page, so none of them counts as a customisation to reveal.
        for field, value in (
            ("nvidia_api_key", "nvapi-xyz"),
            ("stt_backend", "local"),
            ("local_model", "small-q5_1"),
            ("local_model_dir", "D:/models"),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    onboarding_module.has_custom_advanced_settings(
                        replace(defaults, **{field: value})
                    )
                )
        for field, value in (
            ("riva_server", "grpc.example.com:443"),
            ("riva_function_id", "fn-custom"),
            ("nemotron_base_url", "https://example.invalid/v1"),
            ("nemotron_model", "vendor/other"),
            ("nemotron_temperature", 0.4),
            ("nemotron_enable_thinking", False),
        ):
            with self.subTest(field=field):
                self.assertTrue(
                    onboarding_module.has_custom_advanced_settings(
                        replace(defaults, **{field: value})
                    )
                )

    def test_requires_nvidia_key_matrix(self):
        self.assertTrue(config_store.requires_nvidia_key("riva", "raw"))
        self.assertTrue(config_store.requires_nvidia_key("riva", "smart"))
        self.assertFalse(config_store.requires_nvidia_key("local", "raw"))
        self.assertTrue(config_store.requires_nvidia_key("local", "smart"))

    def test_normalize_stt_backend_falls_back_to_riva(self):
        self.assertEqual(config_store.normalize_stt_backend("LOCAL"), "local")
        self.assertEqual(config_store.normalize_stt_backend("nonsense"), "riva")
        self.assertEqual(config_store.normalize_stt_backend(""), "riva")

    def test_removed_api_key_is_not_resurrected_from_env(self):
        resolved = config_store.resolve_settings(
            {"nvidia_api_key": ""},
            {"NVIDIA_API_KEY": "from-env"},
        )
        self.assertEqual(resolved.nvidia_api_key, "")

    def test_absent_api_key_still_falls_back_to_env(self):
        resolved = config_store.resolve_settings({}, {"NVIDIA_API_KEY": "from-env"})
        self.assertEqual(resolved.nvidia_api_key, "from-env")

    def test_empty_config_value_still_falls_back_for_other_fields(self):
        resolved = config_store.resolve_settings(
            {"riva_server": "", "nemotron_model": ""},
            {"RIVA_SERVER": "env-server"},
        )
        self.assertEqual(resolved.riva_server, "env-server")
        self.assertEqual(resolved.nemotron_model, config_store.DEFAULT_NEMOTRON_MODEL)

    def test_resolve_settings_prefers_config_over_env(self):
        config_json = {
            "nvidia_api_key": "from-config",
            "nemotron_model": "config-model",
            "nemotron_enable_thinking": False,
        }
        env_map = {
            "NVIDIA_API_KEY": "from-env",
            "NEMOTRON_MODEL": "env-model",
            "NEMOTRON_ENABLE_THINKING": "true",
        }
        resolved = config_store.resolve_settings(config_json, env_map)
        self.assertEqual(resolved.nvidia_api_key, "from-config")
        self.assertEqual(resolved.nemotron_model, "config-model")
        self.assertFalse(resolved.nemotron_enable_thinking)

    def test_get_config_dir_windows_uses_appdata(self):
        with (
            mock.patch("whispertocode.config_store.os.name", "nt"),
            mock.patch("whispertocode.config_store.os.getenv", side_effect=lambda key: "C:/Users/test/AppData/Roaming" if key == "APPDATA" else None),
        ):
            path = config_store.get_config_dir()
        self.assertEqual(path, Path("C:/Users/test/AppData/Roaming/WhisperToCode"))

    def test_run_onboarding_with_qt_keeps_qt_app_alive_after_close(self):
        fake_app = mock.Mock()
        qt_widgets = types.SimpleNamespace(
            QApplication=types.SimpleNamespace(instance=lambda: fake_app),
            QDialog=types.SimpleNamespace(Accepted=1),
        )
        wizard_instance = mock.Mock()
        wizard_instance.exec.return_value = 0
        with mock.patch("whispertocode.onboarding._OnboardingWizard", return_value=wizard_instance):
            result = onboarding_module.run_onboarding_with_qt(
                qt_core=types.SimpleNamespace(),
                qt_gui=types.SimpleNamespace(),
                qt_widgets=qt_widgets,
                initial=config_store.AppSettings(),
            )
        self.assertIsNone(result)
        fake_app.setQuitOnLastWindowClosed.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
