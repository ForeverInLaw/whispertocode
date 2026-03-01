import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import riva.client
import sounddevice as sd
from dotenv import load_dotenv
from pynput import keyboard

from .constants import (
    NEMOTRON_REASONING_BUDGET_DEFAULT,
    NEMOTRON_REASONING_BUDGET_MAX,
    NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT,
    NEMOTRON_REASONING_PRINT_LIMIT_MAX,
    OUTPUT_MODE_RAW,
    OUTPUT_MODE_SMART,
    OVERLAY_FPS,
    OVERLAY_HEIGHT,
    OVERLAY_WIDTH,
    WINDOWS_SW_HIDE,
    WINDOWS_SW_SHOW,
)
from .audio_support import audio_callback, start_recording, stop_recording
from .hotkeys_support import (
    is_shift_key,
    on_press,
    on_release,
    request_shutdown,
    start_recording_if_valid,
)
from .overlay import QtCapsuleOverlayController
from .riva_asr import recognize_audio
from .runtime_support import run_app, startup_banner_lines
from .smart import build_smart_messages, ensure_nemotron_client, rewrite_text_streaming
from .tray_support import (
    build_tray_icon_image,
    build_tray_menu,
    handle_tray_exit,
    handle_tray_hide_console,
    handle_tray_set_mode_raw,
    handle_tray_set_mode_smart,
    handle_tray_show_console,
    has_console_window,
    ensure_console_window,
    is_console_visible,
    local_hotkeys_loop,
    notify_tray_unavailable,
    redirect_stdio_to_console,
    set_console_visibility,
    start_tray,
    stop_tray,
    tray_title,
)
from .utils import _parse_bool_env

class HoldToTalkRiva:
    def __init__(
        self,
        sample_rate: int,
        language: str,
        hold_delay_sec: float,
        output_mode: str,
        enable_tray: bool,
        debug_console: bool,
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
        self._tray_enabled = enable_tray
        self._debug_console = debug_console

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
        self._peak_level = 0.05
        self._min_level = 0.01

        self._keyboard = keyboard.Controller()
        self._local_hotkeys_enabled = os.name == "nt" and not self._tray_enabled
        self._local_hotkeys_thread: Optional[threading.Thread] = None
        self._tray_icon = None
        self._tray_available = False
        self._overlay_controller = None
        self._console_visible = self._has_console_window()
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
        self._refresh_tray_menu()
        self._set_overlay_mode(normalized)

    def _refresh_tray_menu(self) -> None:
        tray_icon = getattr(self, "_tray_icon", None)
        if tray_icon is None:
            return
        try:
            tray_icon.title = self._tray_title()
            tray_icon.update_menu()
        except Exception:
            # Tray refresh failures are non-fatal for STT flow.
            pass

    def _create_overlay_controller(self):
        return QtCapsuleOverlayController(
            width=OVERLAY_WIDTH,
            height=OVERLAY_HEIGHT,
            fps=OVERLAY_FPS,
        )

    def _start_overlay(self) -> None:
        if getattr(self, "_overlay_controller", None) is not None:
            return
        try:
            overlay_controller = self._create_overlay_controller()
            overlay_controller.start()
            overlay_controller.set_mode(self._get_output_mode())
            self._overlay_controller = overlay_controller
        except Exception as exc:
            raise RuntimeError(f"Overlay initialization failed: {exc}") from exc

    def _stop_overlay(self) -> None:
        overlay_controller = getattr(self, "_overlay_controller", None)
        if overlay_controller is None:
            return
        try:
            overlay_controller.shutdown()
        finally:
            self._overlay_controller = None

    def _show_overlay_recording(self) -> None:
        overlay_controller = getattr(self, "_overlay_controller", None)
        if overlay_controller is None:
            return
        overlay_controller.show_recording(self._get_output_mode())

    def _hide_overlay(self) -> None:
        overlay_controller = getattr(self, "_overlay_controller", None)
        if overlay_controller is None:
            return
        overlay_controller.hide()

    def _set_overlay_mode(self, mode: str) -> None:
        overlay_controller = getattr(self, "_overlay_controller", None)
        if overlay_controller is None:
            return
        overlay_controller.set_mode(mode)

    def _update_overlay_level(self, level: float) -> None:
        overlay_controller = getattr(self, "_overlay_controller", None)
        if overlay_controller is None:
            return
        overlay_controller.update_level(level)

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

    def _is_console_visible(self) -> bool:
        return is_console_visible(self)

    def _has_console_window(self) -> bool:
        return has_console_window()

    @staticmethod
    def _redirect_stdio_to_console() -> None:
        redirect_stdio_to_console()

    def _ensure_console_window(self) -> bool:
        return ensure_console_window(self._redirect_stdio_to_console)

    def _set_console_visibility(self, visible: bool, source: str = "") -> bool:
        return set_console_visibility(self, visible, source)

    def _handle_tray_set_mode_raw(self, icon, item) -> None:
        handle_tray_set_mode_raw(self, icon, item)

    def _handle_tray_set_mode_smart(self, icon, item) -> None:
        handle_tray_set_mode_smart(self, icon, item)

    def _handle_tray_show_console(self, icon, item) -> None:
        handle_tray_show_console(self, icon, item)

    def _handle_tray_hide_console(self, icon, item) -> None:
        handle_tray_hide_console(self, icon, item)

    def _handle_tray_exit(self, icon, item) -> None:
        handle_tray_exit(self, icon, item)

    def _tray_title(self) -> str:
        return tray_title(self)

    def _build_tray_icon_image(self, image_module, draw_module):
        return build_tray_icon_image(image_module, draw_module)

    def _build_tray_menu(self, pystray):
        return build_tray_menu(self, pystray)

    def _notify_tray_unavailable(self, message: str) -> None:
        notify_tray_unavailable(message)

    def _start_tray(self) -> None:
        start_tray(self)

    def _stop_tray(self) -> None:
        stop_tray(self)

    def _local_hotkeys_loop(self) -> None:
        local_hotkeys_loop(self)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        audio_callback(self, indata, frames, time_info, status)

    def _start_recording(self) -> None:
        start_recording(self, sd)

    def _stop_recording(self) -> None:
        stop_recording(self)

    def _transcribe_and_type(self, audio: np.ndarray) -> None:
        with self._lock:
            if self._transcribing:
                print("Still transcribing previous input, skipped.")
                return
            self._transcribing = True

        try:
            text, took = recognize_audio(
                self.asr_service,
                audio=audio,
                sample_rate=self.sample_rate,
                language=self.language,
            )

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
        return build_smart_messages(raw_text)

    def _get_nemotron_client(self):
        self._nemotron_client = ensure_nemotron_client(
            current_client=self._nemotron_client,
            base_url=self._nemotron_base_url,
            api_key=self._api_key,
        )
        return self._nemotron_client

    def _rewrite_text_streaming(self, raw_text: str) -> Tuple[bool, Optional[Exception]]:
        reasoning_budget = max(
            0,
            min(self._nemotron_reasoning_budget, NEMOTRON_REASONING_BUDGET_MAX),
        )
        return rewrite_text_streaming(
            raw_text=raw_text,
            get_client=self._get_nemotron_client,
            model=self._nemotron_model,
            messages=self._build_smart_messages(raw_text),
            temperature=self._nemotron_temperature,
            top_p=self._nemotron_top_p,
            max_tokens=self._nemotron_max_tokens,
            reasoning_budget=reasoning_budget,
            enable_thinking=self._nemotron_enable_thinking,
            reasoning_print_limit=self._reasoning_print_limit,
            type_char=self._keyboard.type,
        )

    @staticmethod
    def _is_shift_key(key: keyboard.KeyCode) -> bool:
        return is_shift_key(key, keyboard)

    def _on_press(self, key) -> Optional[bool]:
        return on_press(self, key, keyboard, threading)

    def _on_release(self, key) -> Optional[bool]:
        return on_release(self, key, keyboard)

    def _start_recording_if_valid(self, token: int) -> None:
        start_recording_if_valid(self, token)

    def request_shutdown(self, reason: str = "shutdown") -> None:
        request_shutdown(self, reason)

    def _startup_banner_lines(self) -> List[str]:
        return startup_banner_lines(self, os)

    def run(self) -> None:
        run_app(
            self,
            keyboard_module=keyboard,
            threading_module=threading,
            time_module=time,
            os_module=os,
        )

