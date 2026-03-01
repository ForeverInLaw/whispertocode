import math
import queue
import threading
import time
from typing import Any, Optional, Tuple

from .constants import OVERLAY_FPS, OVERLAY_HEIGHT, OVERLAY_WIDTH, OUTPUT_MODE_RAW, OUTPUT_MODE_SMART

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

        self._target_opacity = 0.0
        self._current_opacity = 0.0
        self._base_x = 0
        self._base_y = 0
        self._current_y = 0.0

        self._paint_hook = self._build_paint_hook()
        self._widget.paintEvent = self._paint_hook  # type: ignore[assignment]
        self._place_bottom_center()

    @staticmethod
    def _bar_position_gain(index: int, count: int) -> float:
        if count <= 1:
            return 1.0
        center = (count - 1) / 2.0
        distance = abs(index - center) / max(center, 1.0)
        distance = max(0.0, min(1.0, distance))
        tail = 1.0 - distance
        smooth_tail = tail * tail * (3.0 - 2.0 * tail)
        return 0.35 + (0.65 * smooth_tail)

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
            
            # Keep responsiveness for speech while preserving headroom.
            sensitive_level = min(1.0, self._display_level * 1.35)
            
            # White bars with premium opacity
            color = qt_gui.QColor(255, 255, 255, 230)
            painter.setPen(self._qt_core.Qt.NoPen)
            painter.setBrush(color)

            for idx in range(bar_count):
                # Smooth sine wave pulse + random jitter from display level
                pulse = 0.3 + 0.7 * abs(math.sin(now * 3.5 + self._phases[idx]))
                position_gain = self._bar_position_gain(idx, bar_count)
                bar_level = max(0.05, min(1.0, sensitive_level * pulse * position_gain))
                
                # Minimum height to show tiny dots when silent
                bar_h = max(bar_width, max_bar_height * bar_level)
                
                x = base_x + idx * (bar_width + bar_gap)
                y = center_y - (bar_h / 2.0)
                
                # Draw bar with perfectly rounded ends (capsule within a capsule)
                bar_rect = self._qt_core.QRectF(x, y, bar_width, bar_h)
                painter.drawRoundedRect(bar_rect, bar_width / 2.0, bar_width / 2.0)

            painter.end()

        return _paint

    def _place_bottom_center(self) -> None:
        screen = self._qt_gui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        self._base_x = geometry.x() + int((geometry.width() - self._widget.width()) / 2)
        self._base_y = geometry.y() + geometry.height() - self._widget.height() - 20
        if self._target_opacity > 0 and abs(self._current_opacity - self._target_opacity) < 0.01:
            self._current_y = float(self._base_y)
            self._widget.move(self._base_x, int(self._current_y))

    def show_recording(self, mode: str) -> None:
        self.set_mode(mode)
        self._place_bottom_center()
        if not self._widget.isVisible() or self._current_opacity <= 0.01:
            self._current_opacity = 0.0
            self._current_y = float(self._base_y + 10)
            self._widget.setWindowOpacity(0.0)
            self._widget.move(self._base_x, int(self._current_y))

        self._target_opacity = 1.0
        self._widget.show()
        self._widget.raise_()
        self._widget.update()

    def hide(self) -> None:
        self._target_opacity = 0.0

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

        if hasattr(self, "_target_opacity"):
            anim_speed = min(1.0, 15.0 * dt)
            if abs(self._target_opacity - self._current_opacity) > 0.001:
                self._current_opacity += (self._target_opacity - self._current_opacity) * anim_speed
                if abs(self._target_opacity - self._current_opacity) < 0.001:
                    self._current_opacity = self._target_opacity
                self._widget.setWindowOpacity(self._current_opacity)

                target_y = float(self._base_y) if self._target_opacity > 0.5 else float(self._base_y + 10)
                self._current_y += (target_y - self._current_y) * anim_speed
                if self._current_opacity > 0.0:
                    self._widget.move(self._base_x, int(self._current_y))

                if self._current_opacity <= 0.0 and self._widget.isVisible():
                    self._widget.hide()

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

    def run_onboarding_dialog(self, initial_settings):
        response_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        done_event = threading.Event()
        self._queue.put(("onboarding", (initial_settings, response_queue, done_event)))
        done_event.wait()
        result = response_queue.get()
        if isinstance(result, Exception):
            raise result
        return result

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
                    elif cmd == "onboarding":
                        initial_settings, response_queue, done_event = value
                        try:
                            from .onboarding import run_onboarding_with_qt

                            result = run_onboarding_with_qt(
                                QtCore,
                                QtGui,
                                QtWidgets,
                                initial_settings,
                            )
                            response_queue.put(result)
                        except Exception as exc:
                            response_queue.put(exc)
                        finally:
                            done_event.set()
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

