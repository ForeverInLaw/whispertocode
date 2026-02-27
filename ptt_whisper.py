#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import riva.client
import sounddevice as sd
from dotenv import load_dotenv
from pynput import keyboard

OUTPUT_MODE_RAW = "raw"
OUTPUT_MODE_SMART = "smart"
NEMOTRON_REASONING_BUDGET_DEFAULT = 4096
NEMOTRON_REASONING_BUDGET_MAX = 4096
NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT = 600
NEMOTRON_REASONING_PRINT_LIMIT_MAX = 4000


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
        return "".join(parts)
    return ""


class HoldToTalkRiva:
    def __init__(
        self,
        sample_rate: int,
        language: str,
        hold_delay_sec: float,
        output_mode: str,
    ) -> None:
        load_dotenv()

        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set. Put it in .env file.")
        self._api_key = api_key

        self.server = "grpc.nvcf.nvidia.com:443"
        self.function_id = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
        self.sample_rate = sample_rate
        self.language = "multi" if language == "auto" else language
        self.hold_delay_sec = hold_delay_sec
        self._output_mode = self._normalize_output_mode(output_mode)

        metadata = [
            ["function-id", self.function_id],
            ["authorization", f"Bearer {api_key}"],
        ]

        print(f"Connecting to Riva at {self.server}...")
        self.auth = riva.client.Auth(
            uri=self.server,
            use_ssl=True,
            metadata_args=metadata,
        )
        self.asr_service = riva.client.ASRService(self.auth)

        self._lock = threading.Lock()
        self._recording = False
        self._transcribing = False
        self._ctrl_count = 0
        self._press_token = 0
        self._hold_timer: Optional[threading.Timer] = None
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._stop_event = threading.Event()

        self._keyboard = keyboard.Controller()
        self._local_hotkeys_enabled = os.name == "nt"
        self._local_hotkeys_thread: Optional[threading.Thread] = None
        self._nemotron_client = None
        self._nemotron_base_url = (
            os.getenv("NEMOTRON_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
            or "https://integrate.api.nvidia.com/v1"
        )
        self._nemotron_model = (
            os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b").strip()
            or "nvidia/nemotron-3-nano-30b-a3b"
        )
        self._nemotron_temperature = self._read_float_env("NEMOTRON_TEMPERATURE", 1.0)
        self._nemotron_top_p = self._read_float_env("NEMOTRON_TOP_P", 1.0)
        self._nemotron_max_tokens = self._read_int_env("NEMOTRON_MAX_TOKENS", 16384)
        raw_reasoning_budget = self._read_int_env(
            "NEMOTRON_REASONING_BUDGET", NEMOTRON_REASONING_BUDGET_DEFAULT
        )
        self._nemotron_reasoning_budget = max(
            0,
            min(raw_reasoning_budget, NEMOTRON_REASONING_BUDGET_MAX),
        )
        if raw_reasoning_budget != self._nemotron_reasoning_budget:
            print(
                (
                    "NEMOTRON_REASONING_BUDGET was capped to "
                    f"{self._nemotron_reasoning_budget}."
                ),
                file=sys.stderr,
            )
        raw_reasoning_print_limit = self._read_int_env(
            "NEMOTRON_REASONING_PRINT_LIMIT",
            NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT,
        )
        self._reasoning_print_limit = max(
            0,
            min(raw_reasoning_print_limit, NEMOTRON_REASONING_PRINT_LIMIT_MAX),
        )
        self._nemotron_enable_thinking = _parse_bool_env("NEMOTRON_ENABLE_THINKING", True)

    @staticmethod
    def _normalize_output_mode(mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized == OUTPUT_MODE_SMART:
            return OUTPUT_MODE_SMART
        return OUTPUT_MODE_RAW

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw.strip())
        except ValueError:
            print(f"Invalid {name}='{raw}', using default {default}.", file=sys.stderr)
            return default

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw.strip())
        except ValueError:
            print(f"Invalid {name}='{raw}', using default {default}.", file=sys.stderr)
            return default

    def _get_output_mode(self) -> str:
        with self._lock:
            return self._output_mode

    def _set_output_mode(self, mode: str, source: str = "") -> None:
        normalized = self._normalize_output_mode(mode)
        with self._lock:
            if self._output_mode == normalized:
                return
            self._output_mode = normalized
        source_suffix = f" ({source})" if source else ""
        print(f"Mode: {normalized.upper()}{source_suffix}")

    def _handle_local_special_key(self, key_code: str) -> bool:
        normalized = (key_code or "").upper()
        if normalized == "K":
            self._set_output_mode(OUTPUT_MODE_RAW, "Left Arrow")
            return True
        if normalized == "M":
            self._set_output_mode(OUTPUT_MODE_SMART, "Right Arrow")
            return True
        return False

    def _handle_local_console_char(self, char: str) -> bool:
        if char == "\x1b":
            self.request_shutdown("Esc")
            return True
        return False

    def _local_hotkeys_loop(self) -> None:
        try:
            import msvcrt
        except Exception as exc:
            print(f"Local hotkeys disabled: {exc}", file=sys.stderr)
            return

        while not self._stop_event.is_set():
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.03)
                    continue
                char = msvcrt.getwch()
            except Exception as exc:
                print(f"Local hotkeys read failed: {exc}", file=sys.stderr)
                return

            if char in ("\x00", "\xe0"):
                try:
                    special_code = msvcrt.getwch()
                except Exception as exc:
                    print(f"Local hotkeys read failed: {exc}", file=sys.stderr)
                    return
                self._handle_local_special_key(special_code)
                continue

            self._handle_local_console_char(char)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        with self._lock:
            if self._recording:
                self._chunks.append(indata.copy())

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording or self._transcribing:
                return
            self._chunks = []
            self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            print("Recording... (hold Ctrl)")
        except Exception as exc:
            with self._lock:
                self._recording = False
            print(f"Failed to start recording: {exc}", file=sys.stderr)

    def _stop_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            chunks = self._chunks
            self._chunks = []

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if not chunks:
            print("No audio captured.")
            return

        audio = np.concatenate(chunks, axis=0).squeeze()
        if audio.ndim == 0:
            print("No audio captured.")
            return

        duration_sec = len(audio) / float(self.sample_rate)
        if duration_sec < 0.15:
            print("Too short, skipped.")
            return

        worker = threading.Thread(
            target=self._transcribe_and_type, args=(audio,), daemon=True
        )
        worker.start()

    def _transcribe_and_type(self, audio: np.ndarray) -> None:
        with self._lock:
            if self._transcribing:
                print("Still transcribing previous input, skipped.")
                return
            self._transcribing = True

        try:
            # Riva expects LINEAR_PCM int16 bytes.
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16)
            audio_bytes = pcm16.tobytes()

            config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
                language_code=self.language,
                enable_automatic_punctuation=True,
                verbatim_transcripts=True,
                max_alternatives=1,
            )

            start = time.time()
            response = self.asr_service.offline_recognize(audio_bytes, config)
            took = time.time() - start

            text_parts: List[str] = []
            for result in response.results:
                if result.alternatives:
                    text_parts.append(result.alternatives[0].transcript)
            text = "".join(text_parts).strip()

            if not text:
                print("No speech recognized.")
                return

            mode_snapshot = self._get_output_mode()
            print(f"Recognized ({self.language}, {took:.2f}s, {mode_snapshot.upper()}): {text}")
            self._type_output_text(text, mode_snapshot)
        except Exception as exc:
            print(f"Transcription failed: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._transcribing = False

    def _type_output_text(self, text: str, mode_snapshot: str) -> None:
        if mode_snapshot != OUTPUT_MODE_SMART:
            self._keyboard.type(text)
            return

        typed_any, error = self._rewrite_text_streaming(text)
        if error is not None:
            print(f"SMART rewrite failed: {error}", file=sys.stderr)
            if not typed_any:
                self._keyboard.type(text)
            return

        if not typed_any:
            print("SMART rewrite returned empty output, using RAW text.", file=sys.stderr)
            self._keyboard.type(text)

    def _build_smart_messages(self, raw_text: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert editor for speech-to-text transcripts.\n"
                    "Rewrite the input so it reads like clean, naturally typed text.\n"
                    "Rules:\n"
                    "1) Keep the same language as the input. Do not translate.\n"
                    "2) Preserve meaning, facts, names, numbers, links, and technical terms.\n"
                    "3) Remove dictation artifacts, filler words, repetitions, and false starts.\n"
                    "4) Fix punctuation, capitalization, sentence boundaries, and paragraph breaks.\n"
                    "5) Improve readability with light-to-moderate rewriting, but do not add new information.\n"
                    "6) Return only the rewritten final text."
                ),
            },
            {"role": "user", "content": raw_text},
        ]

    def _get_nemotron_client(self):
        if self._nemotron_client is None:
            from openai import OpenAI

            self._nemotron_client = OpenAI(
                base_url=self._nemotron_base_url,
                api_key=self._api_key,
            )
        return self._nemotron_client

    def _rewrite_text_streaming(self, raw_text: str) -> Tuple[bool, Optional[Exception]]:
        typed_any = False
        reasoning_printed = False
        reasoning_printed_chars = 0
        reasoning_truncated = False
        reasoning_budget = max(
            0,
            min(self._nemotron_reasoning_budget, NEMOTRON_REASONING_BUDGET_MAX),
        )
        try:
            client = self._get_nemotron_client()
            completion = client.chat.completions.create(
                model=self._nemotron_model,
                messages=self._build_smart_messages(raw_text),
                temperature=self._nemotron_temperature,
                top_p=self._nemotron_top_p,
                max_tokens=self._nemotron_max_tokens,
                extra_body={
                    "reasoning_budget": reasoning_budget,
                    "chat_template_kwargs": {
                        "enable_thinking": self._nemotron_enable_thinking,
                    },
                },
                stream=True,
            )

            for chunk in completion:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                reasoning_text = _coerce_stream_text(
                    getattr(delta, "reasoning_content", None)
                )
                if reasoning_text:
                    if self._reasoning_print_limit > reasoning_printed_chars:
                        remaining = self._reasoning_print_limit - reasoning_printed_chars
                        reasoning_chunk = reasoning_text[:remaining]
                        if reasoning_chunk:
                            print(reasoning_chunk, end="", flush=True)
                            reasoning_printed = True
                            reasoning_printed_chars += len(reasoning_chunk)
                        if len(reasoning_text) > len(reasoning_chunk):
                            reasoning_truncated = True
                    else:
                        reasoning_truncated = True

                content_text = _coerce_stream_text(getattr(delta, "content", None))
                if content_text:
                    for char in content_text:
                        self._keyboard.type(char)
                    typed_any = True

            return typed_any, None
        except Exception as exc:
            return typed_any, exc
        finally:
            if reasoning_printed:
                print()
            if reasoning_truncated:
                print("[reasoning truncated]", file=sys.stderr)

    @staticmethod
    def _is_ctrl_key(key: keyboard.KeyCode) -> bool:
        return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)

    def _on_press(self, key) -> Optional[bool]:
        if self._is_ctrl_key(key):
            timer_to_start: Optional[threading.Timer] = None
            with self._lock:
                self._ctrl_count += 1
                if self._ctrl_count == 1:
                    self._press_token += 1
                    token = self._press_token
                    timer_to_start = threading.Timer(
                        self.hold_delay_sec, self._start_recording_if_valid, args=(token,)
                    )
                    timer_to_start.daemon = True
                    self._hold_timer = timer_to_start
            if timer_to_start is not None:
                timer_to_start.start()
        return None

    def _on_release(self, key) -> Optional[bool]:
        if self._is_ctrl_key(key):
            timer_to_cancel: Optional[threading.Timer] = None
            should_stop = False
            with self._lock:
                self._ctrl_count = max(0, self._ctrl_count - 1)
                if self._ctrl_count == 0:
                    self._press_token += 1
                    timer_to_cancel = self._hold_timer
                    self._hold_timer = None
                    should_stop = self._recording
            if timer_to_cancel is not None:
                timer_to_cancel.cancel()
            if should_stop:
                self._stop_recording()
        return None

    def _start_recording_if_valid(self, token: int) -> None:
        with self._lock:
            if token != self._press_token:
                return
            if self._ctrl_count == 0:
                return
            if self._recording or self._transcribing:
                return
        self._start_recording()

    def request_shutdown(self, reason: str = "shutdown") -> None:
        timer_to_cancel: Optional[threading.Timer] = None
        with self._lock:
            if self._stop_event.is_set():
                return
            self._stop_event.set()
            timer_to_cancel = self._hold_timer
            self._hold_timer = None
        if timer_to_cancel is not None:
            timer_to_cancel.cancel()
        self._stop_recording()
        print(f"Exit requested ({reason}).")

    def _startup_banner_lines(self) -> List[str]:
        lines = [
            (
                f"Hold Ctrl for at least {self.hold_delay_sec:.1f}s to record, "
                "release Ctrl to transcribe and type."
            ),
            f"Current mode: {self._get_output_mode().upper()}",
        ]
        if os.name == "nt":
            lines.append(
                "Local hotkeys: Left=RAW, Right=SMART, Esc=exit (enabled in this console window)."
            )
        else:
            lines.append("Local hotkeys: unavailable on this OS; use Ctrl+C to exit.")
        lines.append("Tip: on macOS, allow Accessibility access for keyboard control.")
        return lines

    def run(self) -> None:
        for line in self._startup_banner_lines():
            print(line)
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release, suppress=False)
        if self._local_hotkeys_enabled:
            self._local_hotkeys_thread = threading.Thread(
                target=self._local_hotkeys_loop,
                daemon=True,
            )
            self._local_hotkeys_thread.start()
        try:
            listener.start()
            while not self._stop_event.is_set():
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.request_shutdown("Ctrl+C")
        finally:
            if not self._stop_event.is_set():
                self._stop_event.set()
            listener.stop()
            listener.join(timeout=1.0)
            local_hotkeys_thread = self._local_hotkeys_thread
            self._local_hotkeys_thread = None
            if local_hotkeys_thread is not None:
                local_hotkeys_thread.join(timeout=0.2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push-to-talk speech-to-text with NVIDIA Riva Whisper. "
            "Hold Ctrl to capture audio, release Ctrl to type text."
        )
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate",
    )
    parser.add_argument(
        "--language",
        default="auto",
        choices=["auto", "ru", "en", "pl", "de", "es"],
        help="Recognition language (auto for mixed multilingual input)",
    )
    parser.add_argument(
        "--hold-delay",
        type=float,
        default=0.5,
        help="How long Ctrl must be held before recording starts (seconds)",
    )
    parser.add_argument(
        "--mode",
        default=OUTPUT_MODE_RAW,
        choices=[OUTPUT_MODE_RAW, OUTPUT_MODE_SMART],
        help="Output mode: raw STT text or smart rewritten text",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    app = HoldToTalkRiva(
        sample_rate=args.sample_rate,
        language=args.language,
        hold_delay_sec=args.hold_delay,
        output_mode=args.mode,
    )
    signal.signal(signal.SIGINT, lambda sig, frame: app.request_shutdown("Ctrl+C"))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda sig, frame: app.request_shutdown("SIGTERM"))
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
