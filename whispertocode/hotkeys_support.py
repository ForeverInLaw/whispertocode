from typing import Optional


SHIFT_VK_CODES = {16, 160, 161}


def _is_key_debug_enabled(app) -> bool:
    return bool(getattr(app, "_debug_keys", False))


def _describe_key(key) -> str:
    parts = [f"repr={key!r}"]
    for attr in ("vk", "scan", "char", "name"):
        value = getattr(key, attr, None)
        if value is not None:
            parts.append(f"{attr}={value!r}")
    return ", ".join(parts)


def _log_key_event(app, phase: str, key, is_shift: bool, note: str) -> None:
    if not _is_key_debug_enabled(app):
        return
    with app._lock:
        ctrl_count = app._ctrl_count
        press_token = app._press_token
        hold_timer_set = app._hold_timer is not None
        recording = app._recording
        transcribing = app._transcribing
    print(
        (
            f"[keys] {phase} {note}: is_shift={is_shift}, "
            f"ctrl_count={ctrl_count}, press_token={press_token}, "
            f"hold_timer_set={hold_timer_set}, recording={recording}, "
            f"transcribing={transcribing}, {_describe_key(key)}"
        )
    )


def is_shift_key(key, keyboard_module) -> bool:
    if key in (
        keyboard_module.Key.shift,
        keyboard_module.Key.shift_l,
        keyboard_module.Key.shift_r,
    ):
        return True
    # Some external keyboards report Shift as KeyCode(vk=160/161) on Windows.
    return getattr(key, "vk", None) in SHIFT_VK_CODES


def on_press(app, key, keyboard_module, threading_module) -> Optional[bool]:
    shift_pressed = is_shift_key(key, keyboard_module)
    _log_key_event(app, "press", key, shift_pressed, "received")
    if shift_pressed:
        timer_to_start = None
        repeated_press = False
        with app._lock:
            if app._ctrl_count == 0:
                app._ctrl_count = 1
                app._press_token += 1
                token = app._press_token
                timer_to_start = threading_module.Timer(
                    app.hold_delay_sec, app._start_recording_if_valid, args=(token,)
                )
                timer_to_start.daemon = True
                app._hold_timer = timer_to_start
            else:
                repeated_press = True
        if timer_to_start is not None:
            timer_to_start.start()
        if repeated_press and _is_key_debug_enabled(app):
            print("[keys] press ignored: repeated Shift press while already held")
        _log_key_event(app, "press", key, shift_pressed, "handled")
    return None


def on_release(app, key, keyboard_module) -> Optional[bool]:
    shift_released = is_shift_key(key, keyboard_module)
    _log_key_event(app, "release", key, shift_released, "received")
    if shift_released:
        timer_to_cancel = None
        should_stop = False
        with app._lock:
            if app._ctrl_count > 0:
                app._ctrl_count = 0
                app._press_token += 1
                timer_to_cancel = app._hold_timer
                app._hold_timer = None
                should_stop = app._recording
        if timer_to_cancel is not None:
            timer_to_cancel.cancel()
        if should_stop:
            app._stop_recording()
        _log_key_event(app, "release", key, shift_released, "handled")
    return None


def start_recording_if_valid(app, token: int) -> None:
    reason = "ok"
    with app._lock:
        if token != app._press_token:
            reason = "token-mismatch"
        elif app._ctrl_count == 0:
            reason = "ctrl-count-zero"
        elif app._recording or app._transcribing:
            reason = "already-busy"
        if reason != "ok":
            if _is_key_debug_enabled(app):
                print(
                    (
                        f"[keys] timer skipped: reason={reason}, token={token}, "
                        f"press_token={app._press_token}, ctrl_count={app._ctrl_count}, "
                        f"recording={app._recording}, transcribing={app._transcribing}"
                    )
                )
            return
    if _is_key_debug_enabled(app):
        print(f"[keys] timer fired: token={token}, starting recording")
    app._start_recording()


def request_shutdown(app, reason: str = "shutdown") -> None:
    timer_to_cancel = None
    with app._lock:
        if app._stop_event.is_set():
            return
        app._stop_event.set()
        timer_to_cancel = app._hold_timer
        app._hold_timer = None
    if timer_to_cancel is not None:
        timer_to_cancel.cancel()
    app._stop_recording()
    print(f"Exit requested ({reason}).")
