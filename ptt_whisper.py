#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import threading
import time
from typing import List, Optional

import numpy as np
import riva.client
import sounddevice as sd
from dotenv import load_dotenv
from pynput import keyboard


class HoldToTalkRiva:
    def __init__(
        self,
        sample_rate: int,
        language: str,
        hold_delay_sec: float,
    ) -> None:
        load_dotenv()

        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set. Put it in .env file.")

        self.server = "grpc.nvcf.nvidia.com:443"
        self.function_id = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
        self.sample_rate = sample_rate
        self.language = "multi" if language == "auto" else language
        self.hold_delay_sec = hold_delay_sec

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

            print(f"Recognized ({self.language}, {took:.2f}s): {text}")

            # Type text directly into the currently focused window.
            self._keyboard.type(text)
        except Exception as exc:
            print(f"Transcription failed: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._transcribing = False

    @staticmethod
    def _is_ctrl_key(key: keyboard.KeyCode) -> bool:
        return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)

    def _on_press(self, key) -> Optional[bool]:
        if key == keyboard.Key.esc:
            self.request_shutdown("Esc")
            return False

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

    def run(self) -> None:
        print(
            f"Hold Ctrl for at least {self.hold_delay_sec:.1f}s to record, "
            "release Ctrl to transcribe and type."
        )
        print("Press Esc to quit.")
        print("Tip: on macOS, allow Accessibility access for keyboard control.")
        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        try:
            listener.start()
            while not self._stop_event.is_set():
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.request_shutdown("Ctrl+C")
        finally:
            listener.stop()
            listener.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = HoldToTalkRiva(
        sample_rate=args.sample_rate,
        language=args.language,
        hold_delay_sec=args.hold_delay,
    )
    signal.signal(signal.SIGINT, lambda sig, frame: app.request_shutdown("Ctrl+C"))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda sig, frame: app.request_shutdown("SIGTERM"))
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
