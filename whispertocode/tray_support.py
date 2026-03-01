import os
import sys
import time

from .constants import OUTPUT_MODE_RAW, OUTPUT_MODE_SMART, WINDOWS_SW_HIDE, WINDOWS_SW_SHOW


def is_console_visible(app) -> bool:
    with app._lock:
        return bool(getattr(app, "_console_visible", False))


def has_console_window() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return False


def redirect_stdio_to_console() -> None:
    try:
        sys.stdout = open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
        sys.stderr = open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_console_window(redirect_stdio=redirect_stdio_to_console) -> bool:
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
        redirect_stdio()
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception as exc:
        print(f"Failed to allocate debug console: {exc}", file=sys.stderr)
        return False


def set_console_visibility(app, visible: bool, source: str = "") -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        if visible and not app._ensure_console_window():
            return False
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not console_hwnd:
            return False
        show_command = WINDOWS_SW_SHOW if visible else WINDOWS_SW_HIDE
        ctypes.windll.user32.ShowWindow(console_hwnd, show_command)
        with app._lock:
            app._console_visible = visible
        source_suffix = f" ({source})" if source else ""
        if visible or getattr(app, "_debug_console", False):
            state = "shown" if visible else "hidden"
            print(f"Debug console {state}{source_suffix}.")
        app._refresh_tray_menu()
        return True
    except Exception as exc:
        print(f"Failed to update console visibility: {exc}", file=sys.stderr)
        return False


def handle_tray_set_mode_raw(app, _icon, _item) -> None:
    app._set_output_mode(OUTPUT_MODE_RAW, "tray")


def handle_tray_set_mode_smart(app, _icon, _item) -> None:
    app._set_output_mode(OUTPUT_MODE_SMART, "tray")


def handle_tray_show_console(app, _icon, _item) -> None:
    app._set_console_visibility(True, "tray")


def handle_tray_hide_console(app, _icon, _item) -> None:
    app._set_console_visibility(False, "tray")


def handle_tray_exit(app, _icon, _item) -> None:
    app.request_shutdown("Tray")


def tray_title(app) -> str:
    return f"WhisperToCode ({app._get_output_mode().upper()})"


def build_tray_icon_image(image_module, draw_module):
    image = image_module.new("RGBA", (64, 64), (16, 18, 25, 255))
    drawer = draw_module.Draw(image)
    drawer.ellipse((8, 8, 56, 56), fill=(0, 168, 255, 255))
    drawer.rectangle((28, 18, 36, 40), fill=(255, 255, 255, 255))
    drawer.ellipse((24, 38, 40, 54), fill=(255, 255, 255, 255))
    return image


def build_tray_menu(app, pystray):
    menu_item = pystray.MenuItem
    menu_items = [
        menu_item("WhisperToCode", None, enabled=False),
        menu_item(
            "RAW mode",
            app._handle_tray_set_mode_raw,
            checked=lambda _item: app._get_output_mode() == OUTPUT_MODE_RAW,
        ),
        menu_item(
            "SMART mode",
            app._handle_tray_set_mode_smart,
            checked=lambda _item: app._get_output_mode() == OUTPUT_MODE_SMART,
        ),
    ]
    if os.name == "nt":
        menu_items.extend(
            [
                pystray.Menu.SEPARATOR,
                menu_item(
                    "Show debug console",
                    app._handle_tray_show_console,
                    visible=lambda _item: not app._is_console_visible(),
                ),
                menu_item(
                    "Hide debug console",
                    app._handle_tray_hide_console,
                    visible=lambda _item: app._is_console_visible(),
                ),
            ]
        )
    menu_items.extend(
        [
            pystray.Menu.SEPARATOR,
            menu_item("Exit", app._handle_tray_exit),
        ]
    )
    return pystray.Menu(*menu_items)


def notify_tray_unavailable(message: str) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "WhisperToCode", 0x00001030)
    except Exception:
        pass


def start_tray(app) -> None:
    if not getattr(app, "_tray_enabled", False):
        return
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"Tray disabled: {exc}", file=sys.stderr)
        app._tray_enabled = False
        app._local_hotkeys_enabled = os.name == "nt"
        if os.name == "nt" and not getattr(app, "_debug_console", False):
            notify_tray_unavailable(
                f"System tray is unavailable.\nReason: {exc}\n\nFalling back to console mode."
            )
        return
    tray_icon = pystray.Icon(
        "riva-ptt",
        build_tray_icon_image(Image, ImageDraw),
        tray_title(app),
        build_tray_menu(app, pystray),
    )
    app._tray_icon = tray_icon
    try:
        tray_icon.run_detached()
        app._tray_available = True
    except Exception as exc:
        print(f"Tray failed to start: {exc}", file=sys.stderr)
        app._tray_icon = None
        app._tray_enabled = False
        app._tray_available = False
        app._local_hotkeys_enabled = os.name == "nt"
        if os.name == "nt" and not getattr(app, "_debug_console", False):
            notify_tray_unavailable(
                f"System tray failed to start.\nReason: {exc}\n\nFalling back to console mode."
            )


def stop_tray(app) -> None:
    tray_icon = getattr(app, "_tray_icon", None)
    if tray_icon is not None:
        try:
            tray_icon.stop()
        except Exception:
            pass
    app._tray_icon = None
    app._tray_available = False


def local_hotkeys_loop(app) -> None:
    try:
        import msvcrt
    except Exception as exc:
        print(f"Local hotkeys disabled: {exc}", file=sys.stderr)
        return

    while not app._stop_event.is_set():
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
            app._handle_local_special_key(special_code)
            continue

        app._handle_local_console_char(char)
