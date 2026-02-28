#!/usr/bin/env python3
import argparse
import math
import os
import queue
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
WINDOWS_SW_HIDE = 0
WINDOWS_SW_SHOW = 5
OVERLAY_WIDTH = 160
OVERLAY_HEIGHT = 48
OVERLAY_FPS = 60


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


class _CapsuleOverlayWidget:
    def __init__(self, qt_core, qt_gui, qt_widgets, width: int, height: int) -> None:
        self._qt_core = qt_core
        self._qt_gui = qt_gui
        self._widget = qt_widgets.QWidget()
        self._widget.resize(width, height)
        self._widget.setObjectName("riva-ptt-overlay")

        flags = (
            qt_core.Qt.WindowStaysOnTopHint
            | qt_core.Qt.FramelessWindowHint
            | qt_core.Qt.Tool
        )
        if hasattr(qt_core.Qt, "WindowDoesNotAcceptFocus"):
            flags |= qt_core.Qt.WindowDoesNotAcceptFocus
        self._widget.setWindowFlags(flags)
        self._widget.setAttribute(qt_core.Qt.WA_TranslucentBackground, True)
        self._widget.setAttribute(qt_core.Qt.WA_ShowWithoutActivating, True)
        self._widget.setAttribute(qt_core.Qt.WA_TransparentForMouseEvents, True)
        self._widget.setFocusPolicy(qt_core.Qt.NoFocus)

        self._mode = OUTPUT_MODE_RAW.upper()
        self._target_level = 0.0
        self._display_level = 0.0
        self._phases = [idx * 0.4 for idx in range(40)]
        self._last_tick = time.monotonic()
        self._paint_hook = self._build_paint_hook()
        self._widget.paintEvent = self._paint_hook  # type: ignore[assignment]
        self._place_top_center()

    def _build_paint_hook(self):
        qt_gui = self._qt_gui

        def _paint(_event) -> None:
            painter = qt_gui.QPainter(self._widget)
            painter.setRenderHint(qt_gui.QPainter.Antialiasing, True)

            rect = self._widget.rect()
            capsule_rect = rect.adjusted(2, 2, -2, -2)
            
            # Premium Apple-like Aesthetic: deep opaque background
            painter.setPen(qt_gui.QPen(qt_gui.QColor(255, 255, 255, 25), 1))
            painter.setBrush(qt_gui.QColor(18, 18, 20, 255))
            
            # Perfect pill shape (radius is exactly half the height)
            radius = capsule_rect.height() / 2.0
            painter.drawRoundedRect(capsule_rect, radius, radius)

            bar_count = 20
            bar_gap = 4
            horizontal_padding = 24
            vertical_padding = 12
            
            available_width = self._widget.width() - (horizontal_padding * 2)
            bar_width = max(2.0, (available_width - (bar_count - 1) * bar_gap) / bar_count)
            base_x = horizontal_padding
            center_y = self._widget.height() / 2.0
            max_bar_height = self._widget.height() - (vertical_padding * 2)

            now = time.monotonic()
            
            # Smooth dynamic sensitivity multiplier
            sensitive_level = min(1.0, self._display_level * 4.0)
            
            # White bars with premium opacity
            color = qt_gui.QColor(255, 255, 255, 230)
            painter.setPen(self._qt_core.Qt.NoPen)
            painter.setBrush(color)

            for idx in range(bar_count):
                # Smooth sine wave pulse + random jitter from display level
                pulse = 0.3 + 0.7 * abs(math.sin(now * 3.5 + self._phases[idx]))
                bar_level = max(0.05, min(1.0, sensitive_level * pulse))
                
                # Minimum height to show tiny dots when silent
                bar_h = max(bar_width, max_bar_height * bar_level)
                
                x = base_x + idx * (bar_width + bar_gap)
                y = center_y - (bar_h / 2.0)
                
                # Draw bar with perfectly rounded ends (capsule within a capsule)
                bar_rect = self._qt_core.QRectF(x, y, bar_width, bar_h)
                painter.drawRoundedRect(bar_rect, bar_width / 2.0, bar_width / 2.0)

            painter.end()

        return _paint

    def _place_top_center(self) -> None:
        screen = self._qt_gui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        x = geometry.x() + int((geometry.width() - self._widget.width()) / 2)
        y = geometry.y() + 20
        self._widget.move(x, y)

    def show_recording(self, mode: str) -> None:
        self.set_mode(mode)
        self._place_top_center()
        self._widget.show()
        self._widget.raise_()
        self._widget.update()

    def hide(self) -> None:
        self._widget.hide()

    def close(self) -> None:
        self._widget.close()

    def set_mode(self, mode: str) -> None:
        normalized = (mode or OUTPUT_MODE_RAW).strip().upper()
        self._mode = OUTPUT_MODE_SMART.upper() if normalized == OUTPUT_MODE_SMART.upper() else OUTPUT_MODE_RAW.upper()
        if self._widget.isVisible():
            self._widget.update()

    def set_level(self, level: float) -> None:
        clipped = max(0.0, min(float(level), 1.0))
        self._target_level = clipped

    def animate_step(self) -> None:
        now = time.monotonic()
        dt = max(0.001, now - self._last_tick)
        self._last_tick = now
        up_speed = min(0.95, 8.0 * dt)
        down_speed = min(0.95, 4.5 * dt)
        if self._target_level > self._display_level:
            self._display_level += (self._target_level - self._display_level) * up_speed
        else:
            self._display_level += (self._target_level - self._display_level) * down_speed
        self._display_level = max(0.0, min(self._display_level, 1.0))
        if self._widget.isVisible():
            self._widget.update()


class QtCapsuleOverlayController:
    def __init__(self, width: int = OVERLAY_WIDTH, height: int = OVERLAY_HEIGHT, fps: int = OVERLAY_FPS) -> None:
        self._width = width
        self._height = height
        self._fps = max(10, fps)
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._startup_error: Optional[Exception] = None

    def start(self, timeout_sec: float = 3.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._startup_error = None
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run_ui_loop, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout_sec):
            raise RuntimeError("Overlay startup timeout.")
        if self._startup_error is not None:
            raise RuntimeError(self._startup_error)

    def show_recording(self, mode: str) -> None:
        self._queue.put(("show", mode))

    def set_mode(self, mode: str) -> None:
        self._queue.put(("mode", mode))

    def update_level(self, level: float) -> None:
        self._queue.put(("level", level))

    def hide(self) -> None:
        self._queue.put(("hide", None))

    def shutdown(self) -> None:
        self._queue.put(("shutdown", None))
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def _run_ui_loop(self) -> None:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()
            return

        try:
            app = QtWidgets.QApplication([])
            app.setQuitOnLastWindowClosed(False)
            overlay = _CapsuleOverlayWidget(
                qt_core=QtCore,
                qt_gui=QtGui,
                qt_widgets=QtWidgets,
                width=self._width,
                height=self._height,
            )
            self._ready_event.set()

            timer = QtCore.QTimer()
            timer.setInterval(int(1000 / self._fps))

            def _tick() -> None:
                while True:
                    try:
                        cmd, value = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "show":
                        overlay.show_recording(str(value))
                    elif cmd == "mode":
                        overlay.set_mode(str(value))
                    elif cmd == "level":
                        overlay.set_level(float(value))
                    elif cmd == "hide":
                        overlay.hide()
                    elif cmd == "shutdown":
                        timer.stop()
                        overlay.close()
                        app.quit()
                        return
                overlay.animate_step()

            timer.timeout.connect(_tick)
            timer.start()
            app.exec()
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()


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
        with self._lock:
            return bool(getattr(self, "_console_visible", False))

    def _has_console_window(self) -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes

            return bool(ctypes.windll.kernel32.GetConsoleWindow())
        except Exception:
            return False

    @staticmethod
    def _redirect_stdio_to_console() -> None:
        try:
            sys.stdout = open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
            sys.stderr = open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
        except Exception:
            # Debug console is optional; keep running even if stream rebinding fails.
            pass

    def _ensure_console_window(self) -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes

            console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if console_hwnd:
                return True
            allocated = ctypes.windll.kernel32.AllocConsole()
            if not allocated:
                return False
            self._redirect_stdio_to_console()
            return bool(ctypes.windll.kernel32.GetConsoleWindow())
        except Exception as exc:
            print(f"Failed to allocate debug console: {exc}", file=sys.stderr)
            return False

    def _set_console_visibility(self, visible: bool, source: str = "") -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes

            if visible and not self._ensure_console_window():
                return False
            console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if not console_hwnd:
                return False
            show_command = WINDOWS_SW_SHOW if visible else WINDOWS_SW_HIDE
            ctypes.windll.user32.ShowWindow(console_hwnd, show_command)
            with self._lock:
                self._console_visible = visible
            source_suffix = f" ({source})" if source else ""
            if visible or getattr(self, "_debug_console", False):
                state = "shown" if visible else "hidden"
                print(f"Debug console {state}{source_suffix}.")
            self._refresh_tray_menu()
            return True
        except Exception as exc:
            print(f"Failed to update console visibility: {exc}", file=sys.stderr)
            return False

    def _handle_tray_set_mode_raw(self, icon, item) -> None:
        self._set_output_mode(OUTPUT_MODE_RAW, "tray")

    def _handle_tray_set_mode_smart(self, icon, item) -> None:
        self._set_output_mode(OUTPUT_MODE_SMART, "tray")

    def _handle_tray_show_console(self, icon, item) -> None:
        self._set_console_visibility(True, "tray")

    def _handle_tray_hide_console(self, icon, item) -> None:
        self._set_console_visibility(False, "tray")

    def _handle_tray_exit(self, icon, item) -> None:
        self.request_shutdown("Tray")

    def _tray_title(self) -> str:
        return f"Riva PTT ({self._get_output_mode().upper()})"

    def _build_tray_icon_image(self, image_module, draw_module):
        image = image_module.new("RGBA", (64, 64), (16, 18, 25, 255))
        drawer = draw_module.Draw(image)
        drawer.ellipse((8, 8, 56, 56), fill=(0, 168, 255, 255))
        drawer.rectangle((28, 18, 36, 40), fill=(255, 255, 255, 255))
        drawer.ellipse((24, 38, 40, 54), fill=(255, 255, 255, 255))
        return image

    def _build_tray_menu(self, pystray):
        menu_item = pystray.MenuItem
        menu_items = [
            menu_item("Riva Push-to-Talk", None, enabled=False),
            menu_item(
                "RAW mode",
                self._handle_tray_set_mode_raw,
                checked=lambda item: self._get_output_mode() == OUTPUT_MODE_RAW,
            ),
            menu_item(
                "SMART mode",
                self._handle_tray_set_mode_smart,
                checked=lambda item: self._get_output_mode() == OUTPUT_MODE_SMART,
            ),
        ]
        if os.name == "nt":
            menu_items.extend(
                [
                    pystray.Menu.SEPARATOR,
                    menu_item(
                        "Show debug console",
                        self._handle_tray_show_console,
                        visible=lambda item: not self._is_console_visible(),
                    ),
                    menu_item(
                        "Hide debug console",
                        self._handle_tray_hide_console,
                        visible=lambda item: self._is_console_visible(),
                    ),
                ]
            )
        menu_items.extend(
            [
                pystray.Menu.SEPARATOR,
                menu_item("Exit", self._handle_tray_exit),
            ]
        )
        return pystray.Menu(*menu_items)

    def _notify_tray_unavailable(self, message: str) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "Riva PTT", 0x00001030)
        except Exception:
            pass

    def _start_tray(self) -> None:
        if not getattr(self, "_tray_enabled", False):
            return
        try:
            import pystray
            from PIL import Image, ImageDraw
        except Exception as exc:
            print(f"Tray disabled: {exc}", file=sys.stderr)
            self._tray_enabled = False
            self._local_hotkeys_enabled = os.name == "nt"
            if os.name == "nt" and not getattr(self, "_debug_console", False):
                self._notify_tray_unavailable(
                    f"System tray is unavailable.\nReason: {exc}\n\nFalling back to console mode."
                )
            return
        tray_icon = pystray.Icon(
            "riva-ptt",
            self._build_tray_icon_image(Image, ImageDraw),
            self._tray_title(),
            self._build_tray_menu(pystray),
        )
        self._tray_icon = tray_icon
        try:
            tray_icon.run_detached()
            self._tray_available = True
        except Exception as exc:
            print(f"Tray failed to start: {exc}", file=sys.stderr)
            self._tray_icon = None
            self._tray_enabled = False
            self._tray_available = False
            self._local_hotkeys_enabled = os.name == "nt"
            if os.name == "nt" and not getattr(self, "_debug_console", False):
                self._notify_tray_unavailable(
                    f"System tray failed to start.\nReason: {exc}\n\nFalling back to console mode."
                )

    def _stop_tray(self) -> None:
        tray_icon = getattr(self, "_tray_icon", None)
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        self._tray_icon = None
        self._tray_available = False

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
        level_value: Optional[float] = None
        with self._lock:
            if self._recording:
                self._chunks.append(indata.copy())
                frame = np.asarray(indata, dtype=np.float32)
                if frame.size > 0:
                    if frame.ndim > 1:
                        frame = frame[:, 0]
                    level_value = float(np.sqrt(np.mean(np.square(np.clip(frame, -1.0, 1.0)))))
        if level_value is not None:
            self._update_overlay_level(level_value)

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
            self._show_overlay_recording()
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
        self._hide_overlay()

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
        if getattr(self, "_tray_enabled", False):
            lines.append("Tray controls: switch RAW/SMART mode, show debug console, and exit.")
        elif os.name == "nt":
            lines.append(
                "Local hotkeys: Left=RAW, Right=SMART, Esc=exit (enabled in this console window)."
            )
        else:
            lines.append("Local hotkeys: unavailable on this OS; use Ctrl+C to exit.")
        lines.append("Tip: on macOS, allow Accessibility access for keyboard control.")
        return lines

    def run(self) -> None:
        self._start_tray()
        self._start_overlay()
        if os.name == "nt":
            if getattr(self, "_tray_enabled", False):
                if getattr(self, "_debug_console", False):
                    self._set_console_visibility(True, "startup")
                else:
                    self._set_console_visibility(False, "startup")
            else:
                self._set_console_visibility(True, "startup")
        if getattr(self, "_debug_console", False) or not getattr(self, "_tray_enabled", False):
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
            self._stop_overlay()
            self._stop_tray()


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
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Disable system tray controls and keep local console hotkeys.",
    )
    parser.add_argument(
        "--debug-console",
        action="store_true",
        help="Keep console visible for debugging in tray mode (Windows).",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    app = HoldToTalkRiva(
        sample_rate=args.sample_rate,
        language=args.language,
        hold_delay_sec=args.hold_delay,
        output_mode=args.mode,
        enable_tray=not args.no_tray,
        debug_console=args.debug_console,
    )
    signal.signal(signal.SIGINT, lambda sig, frame: app.request_shutdown("Ctrl+C"))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda sig, frame: app.request_shutdown("SIGTERM"))
    try:
        app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
