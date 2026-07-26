OUTPUT_MODE_RAW = "raw"
OUTPUT_MODE_SMART = "smart"

STT_BACKEND_RIVA = "riva"
STT_BACKEND_LOCAL = "local"

# whisper.cpp ggml model names, smallest first, with download size on first use.
# Quantization suffix is not uniform upstream: tiny/base/small use q5_1, larger use q5_0.
LOCAL_MODEL_CHOICES = (
    "tiny-q5_1",
    "tiny",
    "base-q5_1",
    "base",
    "small-q5_1",
    "small",
    "medium-q5_0",
    "large-v3-turbo-q5_0",
    "large-v3-turbo",
)

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
NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT = 600
NEMOTRON_REASONING_PRINT_LIMIT_MAX = 4000

WINDOWS_SW_HIDE = 0
WINDOWS_SW_SHOW = 5

OVERLAY_WIDTH = 160
OVERLAY_HEIGHT = 48
OVERLAY_FPS = 60
