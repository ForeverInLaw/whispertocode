import time
from typing import Tuple

import numpy as np
import riva.client


def recognize_audio(
    asr_service: riva.client.ASRService,
    *,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
) -> Tuple[str, float]:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    audio_bytes = pcm16.tobytes()

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        audio_channel_count=1,
        language_code=language,
        enable_automatic_punctuation=True,
        verbatim_transcripts=True,
        max_alternatives=1,
    )

    start = time.time()
    response = asr_service.offline_recognize(audio_bytes, config)
    took = time.time() - start

    text_parts = []
    for result in response.results:  # type: ignore[union-attr]
        if result.alternatives:
            text_parts.append(result.alternatives[0].transcript)
    return "".join(text_parts).strip(), took
