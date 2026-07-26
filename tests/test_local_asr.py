import importlib
import types
import unittest

import numpy as np

local_asr = importlib.import_module("whispertocode.local_asr")


class _FakeModel:
    def __init__(self, texts):
        self._texts = texts
        self.last_language = "unset"

    def transcribe(self, audio, language=None):
        self.last_language = language
        assert audio.dtype == np.float32
        return [types.SimpleNamespace(text=text) for text in self._texts]


class LocalAsrTests(unittest.TestCase):
    def test_rejects_non_16k_audio(self):
        with self.assertRaises(RuntimeError) as ctx:
            local_asr.transcribe(
                _FakeModel(["hi"]),
                audio=np.zeros(10, dtype=np.float32),
                sample_rate=48000,
                language="auto",
            )
        self.assertIn("16000", str(ctx.exception))

    def test_drops_non_speech_markers(self):
        model = _FakeModel([" [BLANK_AUDIO]", " hello ", "(upbeat music)", " world"])
        text, took = local_asr.transcribe(
            model,
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            language="auto",
        )
        self.assertEqual(text, "hello world")
        self.assertGreaterEqual(took, 0.0)

    def test_auto_language_maps_to_empty_string(self):
        model = _FakeModel(["hi"])
        local_asr.transcribe(
            model,
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            language="auto",
        )
        self.assertEqual(model.last_language, "")

    def test_explicit_language_is_passed_through(self):
        model = _FakeModel(["привет"])
        local_asr.transcribe(
            model,
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            language="ru",
        )
        self.assertEqual(model.last_language, "ru")

    def test_float64_audio_is_converted(self):
        model = _FakeModel(["ok"])
        text, _ = local_asr.transcribe(
            model,
            audio=np.zeros(16000, dtype=np.float64),
            sample_rate=16000,
            language="en",
        )
        self.assertEqual(text, "ok")


if __name__ == "__main__":
    unittest.main()
