from typing import Optional


def is_shift_key(key, keyboard_module) -> bool:
    return key in (
        keyboard_module.Key.shift,
        keyboard_module.Key.shift_l,
        keyboard_module.Key.shift_r,
    )


def on_press(app, key, keyboard_module, threading_module) -> Optional[bool]:
    if is_shift_key(key, keyboard_module):
        timer_to_start = None
        with app._lock:
            app._ctrl_count += 1
            if app._ctrl_count == 1:
                app._press_token += 1
                token = app._press_token
                timer_to_start = threading_module.Timer(
                    app.hold_delay_sec, app._start_recording_if_valid, args=(token,)
                )
                timer_to_start.daemon = True
                app._hold_timer = timer_to_start
        if timer_to_start is not None:
            timer_to_start.start()
    return None


def on_release(app, key, keyboard_module) -> Optional[bool]:
    if is_shift_key(key, keyboard_module):
        timer_to_cancel = None
        should_stop = False
        with app._lock:
            app._ctrl_count = max(0, app._ctrl_count - 1)
            if app._ctrl_count == 0:
                app._press_token += 1
                timer_to_cancel = app._hold_timer
                app._hold_timer = None
                should_stop = app._recording
        if timer_to_cancel is not None:
            timer_to_cancel.cancel()
        if should_stop:
            app._stop_recording()
    return None


def start_recording_if_valid(app, token: int) -> None:
    with app._lock:
        if token != app._press_token:
            return
        if app._ctrl_count == 0:
            return
        if app._recording or app._transcribing:
            return
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
