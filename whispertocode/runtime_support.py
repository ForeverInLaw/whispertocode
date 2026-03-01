def startup_banner_lines(app, os_module) -> list[str]:
    lines = [
        (
            f"Hold Shift for at least {app.hold_delay_sec:.1f}s to record, "
            "release Shift to transcribe and type."
        ),
        f"Current mode: {app._get_output_mode().upper()}",
    ]
    if getattr(app, "_tray_enabled", False):
        lines.append("Tray controls: switch RAW/SMART mode, show debug console, and exit.")
    elif os_module.name == "nt":
        lines.append(
            "Local hotkeys: Left=RAW, Right=SMART, Esc=exit (enabled in this console window)."
        )
    else:
        lines.append("Local hotkeys: unavailable on this OS; use Ctrl+C to exit.")
    lines.append("Tip: on macOS, allow Accessibility access for keyboard control.")
    return lines


def run_app(app, *, keyboard_module, threading_module, time_module, os_module) -> None:
    app._start_tray()
    app._start_overlay()
    if os_module.name == "nt":
        if getattr(app, "_tray_enabled", False):
            if getattr(app, "_debug_console", False):
                app._set_console_visibility(True, "startup")
            else:
                app._set_console_visibility(False, "startup")
        else:
            app._set_console_visibility(True, "startup")
    if getattr(app, "_debug_console", False) or not getattr(app, "_tray_enabled", False):
        for line in app._startup_banner_lines():
            print(line)
    listener = keyboard_module.Listener(
        on_press=app._on_press, on_release=app._on_release, suppress=False
    )
    if app._local_hotkeys_enabled:
        app._local_hotkeys_thread = threading_module.Thread(
            target=app._local_hotkeys_loop,
            daemon=True,
        )
        app._local_hotkeys_thread.start()
    try:
        listener.start()
        while not app._stop_event.is_set():
            time_module.sleep(0.05)
    except KeyboardInterrupt:
        app.request_shutdown("Ctrl+C")
    finally:
        if not app._stop_event.is_set():
            app._stop_event.set()
        listener.stop()
        listener.join(timeout=1.0)
        local_hotkeys_thread = app._local_hotkeys_thread
        app._local_hotkeys_thread = None
        if local_hotkeys_thread is not None:
            local_hotkeys_thread.join(timeout=0.2)
        app._stop_overlay()
        app._stop_tray()
