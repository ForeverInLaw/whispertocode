import importlib
import threading
import types
import unittest
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
        )
        keyboard_module.KeyCode = object
        pynput_module.keyboard = keyboard_module
        importlib.sys.modules["pynput"] = pynput_module
        importlib.sys.modules["pynput.keyboard"] = keyboard_module


_install_dependency_stubs()
ptt_whisper = importlib.import_module("whispertocode.app")


def _make_app() -> "ptt_whisper.HoldToTalkRiva":
    app = object.__new__(ptt_whisper.HoldToTalkRiva)
    app._lock = threading.Lock()
    app._keyboard = mock.Mock()
    app._nemotron_model = "nvidia/nemotron-3-nano-30b-a3b"
    app._nemotron_temperature = 1.0
    app._nemotron_top_p = 1.0
    app._nemotron_max_tokens = 16384
    app._nemotron_reasoning_budget = 16384
    app._nemotron_enable_thinking = True
    app._reasoning_print_limit = 400
    return app


class PromptAndStreamTests(unittest.TestCase):
    def test_prompt_requires_same_language_and_light_edit(self):
        app = _make_app()
        messages = app._build_smart_messages("пример")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "<transcript>\nпример\n</transcript>")
        self.assertIn("exact original language", messages[0]["content"])
        self.assertIn("do not add any new information", messages[0]["content"].lower())
        self.assertIn("return only the final corrected text", messages[0]["content"].lower())

    def test_reasoning_budget_is_capped(self):
        app = _make_app()
        app._nemotron_reasoning_budget = 999999

        completion_stream = []
        completions = mock.Mock()
        completions.create.return_value = completion_stream
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        app._get_nemotron_client = mock.Mock(return_value=fake_client)

        with mock.patch("builtins.print"):
            app._rewrite_text_streaming("raw input")

        call_kwargs = completions.create.call_args.kwargs
        self.assertEqual(
            call_kwargs["extra_body"]["reasoning_budget"],
            ptt_whisper.NEMOTRON_REASONING_BUDGET_MAX,
        )

    def test_stream_types_only_content_and_prints_reasoning(self):
        app = _make_app()

        chunk_1 = types.SimpleNamespace(choices=[])
        chunk_2 = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(reasoning_content="think ", content=None)
                )
            ]
        )
        chunk_3 = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(reasoning_content=None, content="hello ")
                )
            ]
        )
        chunk_4 = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(reasoning_content=None, content="world")
                )
            ]
        )

        completion_stream = [chunk_1, chunk_2, chunk_3, chunk_4]

        completions = mock.Mock()
        completions.create.return_value = completion_stream
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        app._get_nemotron_client = mock.Mock(return_value=fake_client)

        with mock.patch("builtins.print") as print_mock:
            typed_any, error = app._rewrite_text_streaming("raw input")

        self.assertTrue(typed_any)
        self.assertIsNone(error)
        typed_text = "".join(call.args[0] for call in app._keyboard.type.call_args_list)
        self.assertEqual(typed_text, "hello world")
        self.assertTrue(any(call.args and call.args[0] == "think " for call in print_mock.call_args_list))

    def test_stream_does_not_truncate_reasoning_output(self):
        app = _make_app()
        app._reasoning_print_limit = 1

        chunk = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(reasoning_content="very long reasoning", content=None)
                )
            ]
        )
        completions = mock.Mock()
        completions.create.return_value = [chunk]
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        app._get_nemotron_client = mock.Mock(return_value=fake_client)

        with mock.patch("builtins.print") as print_mock:
            typed_any, error = app._rewrite_text_streaming("raw input")

        self.assertFalse(typed_any)
        self.assertIsNone(error)
        printed_values = [call.args[0] for call in print_mock.call_args_list if call.args]
        self.assertIn("very long reasoning", printed_values)
        self.assertNotIn("[reasoning truncated]", printed_values)


if __name__ == "__main__":
    unittest.main()
