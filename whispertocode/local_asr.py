import time
from pathlib import Path
from typing import Tuple

import numpy as np

from .config_store import get_config_dir

# whisper.cpp only accepts 16 kHz mono float32.
LOCAL_SAMPLE_RATE = 16000


def _is_speech(text: str) -> bool:
    """whisper.cpp annotates non-speech as [BLANK_AUDIO], (upbeat music), [Music]."""
    stripped = text.strip()
    if not stripped:
        return False
    return not (stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")))


def default_model_dir() -> Path:
    return get_config_dir() / "models"


def load_model(model: str, model_dir: str = ""):
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:
        raise RuntimeError(
            "Local speech backend needs pywhispercpp: pip install pywhispercpp. "
            "Or start with --backend riva to use the cloud backend."
        ) from exc

    target_dir = Path(model_dir).expanduser() if model_dir else default_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    return Model(
        model or "tiny",
        models_dir=str(target_dir),
        # Off: this is per-transcription progress spam, not the model download.
        # The download has its own tqdm bar inside pywhispercpp.
        print_progress=False,
        print_realtime=False,
        redirect_whispercpp_logs_to=None,
    )


def transcribe(
    model,
    *,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
) -> Tuple[str, float]:
    if sample_rate != LOCAL_SAMPLE_RATE:
        raise RuntimeError(
            f"Local backend needs {LOCAL_SAMPLE_RATE} Hz audio, got {sample_rate}. "
            f"Restart with --sample-rate {LOCAL_SAMPLE_RATE}."
        )

    samples = np.ascontiguousarray(audio, dtype=np.float32)
    # Empty language string is whisper.cpp's auto-detect; the literal "auto"
    # reaches detection too but logs `unknown language 'auto'` on the way.
    language_code = "" if language in ("", "auto", None) else language

    start = time.time()
    segments = model.transcribe(samples, language=language_code)
    took = time.time() - start

    spoken = [segment.text.strip() for segment in segments if _is_speech(segment.text)]
    return " ".join(spoken).strip(), took
