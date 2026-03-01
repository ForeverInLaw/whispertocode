import sys
import threading
from typing import Optional

import numpy as np


def audio_callback(app, indata, _frames, _time_info, status) -> None:
    if status:
        print(f"Audio warning: {status}", file=sys.stderr)
    level_value: Optional[float] = None
    with app._lock:
        if app._recording:
            app._chunks.append(indata.copy())
            frame = np.asarray(indata, dtype=np.float32)
            if frame.size > 0:
                if frame.ndim > 1:
                    frame = frame[:, 0]
                raw_level = float(np.sqrt(np.mean(np.square(np.clip(frame, -1.0, 1.0)))))

                if not hasattr(app, "_level_ema"):
                    app._level_ema = max(app._min_level, raw_level)

                if raw_level > app._peak_level:
                    app._peak_level = raw_level
                else:
                    app._peak_level = max(app._min_level, app._peak_level * 0.997)

                # Adaptive gain for quiet/loud microphones without filtering out real activity.
                if raw_level >= app._level_ema:
                    app._level_ema += (raw_level - app._level_ema) * 0.22
                else:
                    app._level_ema += (raw_level - app._level_ema) * 0.08

                reference_level = max(app._min_level, app._level_ema * 1.35)
                normalized_level = max(0.0, (raw_level / reference_level) * 1.2)
                level_value = normalized_level / (1.0 + normalized_level)

    if level_value is not None:
        app._update_overlay_level(level_value)


def start_recording(app, sd_module) -> None:
    with app._lock:
        if app._recording or app._transcribing:
            return
        app._chunks = []
        app._recording = True

    try:
        app._stream = sd_module.InputStream(
            samplerate=app.sample_rate,
            channels=1,
            dtype="float32",
            callback=app._audio_callback,
        )
        app._stream.start()
        app._show_overlay_recording()
        print("Recording... (hold Shift)")
    except Exception as exc:
        with app._lock:
            app._recording = False
        print(f"Failed to start recording: {exc}", file=sys.stderr)


def stop_recording(app) -> None:
    with app._lock:
        if not app._recording:
            return
        app._recording = False
        chunks = app._chunks
        app._chunks = []
    app._hide_overlay()

    if app._stream is not None:
        try:
            app._stream.stop()
            app._stream.close()
        except Exception:
            pass
        app._stream = None

    if not chunks:
        print("No audio captured.")
        return

    audio = np.concatenate(chunks, axis=0).squeeze()
    if audio.ndim == 0:
        print("No audio captured.")
        return

    duration_sec = len(audio) / float(app.sample_rate)
    if duration_sec < 0.15:
        print("Too short, skipped.")
        return

    worker = threading.Thread(
        target=app._transcribe_and_type, args=(audio,), daemon=True
    )
    worker.start()
