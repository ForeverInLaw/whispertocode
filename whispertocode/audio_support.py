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

                if raw_level > app._peak_level:
                    app._peak_level = raw_level
                else:
                    # Slow peak decay keeps short-term dynamics without pinning voice to 100%.
                    app._peak_level = max(app._min_level, app._peak_level * 0.999)

                if app._peak_level > app._min_level:
                    reference_level = max(app._min_level, app._peak_level * 1.8)
                    level_value = min(1.0, max(0.0, raw_level / reference_level))
                else:
                    level_value = 0.0

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
