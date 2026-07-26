OUTPUT_MODE_RAW = "raw"
OUTPUT_MODE_SMART = "smart"

STT_BACKEND_RIVA = "riva"
STT_BACKEND_LOCAL = "local"

# whisper.cpp ggml models, smallest first, paired with their download size so the
# UI can state the cost before the user commits to one.
# Quantization suffix is not uniform upstream: tiny/base/small use q5_1, larger use q5_0.
LOCAL_MODELS = (
    ("tiny-q5_1", "32 MB"),
    ("base-q5_1", "60 MB"),
    ("tiny", "78 MB"),
    ("base", "148 MB"),
    ("small-q5_1", "190 MB"),
    ("small", "488 MB"),
    ("medium-q5_0", "539 MB"),
    ("large-v3-turbo-q5_0", "574 MB"),
    ("large-v3-turbo", "1.6 GB"),
)

LOCAL_MODEL_CHOICES = tuple(name for name, _size in LOCAL_MODELS)

# Whisper's 21 highest-resource languages, plus "auto" for detection.
LANGUAGE_CHOICES = (
    "auto",
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "uk",
)

NEMOTRON_REASONING_BUDGET_DEFAULT = 4096
NEMOTRON_REASONING_BUDGET_MAX = 4096

WINDOWS_SW_HIDE = 0
WINDOWS_SW_SHOW = 5

OVERLAY_WIDTH = 160
OVERLAY_HEIGHT = 48
OVERLAY_FPS = 60
