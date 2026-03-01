import argparse
import signal
import sys
from typing import List, Optional

from .app import HoldToTalkRiva
from .constants import OUTPUT_MODE_RAW, OUTPUT_MODE_SMART

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WhisperToCode speech-to-text with NVIDIA Riva Whisper. "
            "Hold Shift to capture audio, release Shift to type text."
        )
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate",
    )
    parser.add_argument(
        "--language",
        default="auto",
        choices=["auto", "ru", "en", "pl", "de", "es"],
        help="Recognition language (auto for mixed multilingual input)",
    )
    parser.add_argument(
        "--hold-delay",
        type=float,
        default=0.5,
        help="How long Shift must be held before recording starts (seconds)",
    )
    parser.add_argument(
        "--mode",
        default=OUTPUT_MODE_RAW,
        choices=[OUTPUT_MODE_RAW, OUTPUT_MODE_SMART],
        help="Output mode: raw STT text or smart rewritten text",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Disable system tray controls and keep local console hotkeys.",
    )
    parser.add_argument(
        "--debug-console",
        action="store_true",
        help="Keep console visible for debugging in tray mode (Windows).",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    app = HoldToTalkRiva(
        sample_rate=args.sample_rate,
        language=args.language,
        hold_delay_sec=args.hold_delay,
        output_mode=args.mode,
        enable_tray=not args.no_tray,
        debug_console=args.debug_console,
    )
    signal.signal(signal.SIGINT, lambda sig, frame: app.request_shutdown("Ctrl+C"))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda sig, frame: app.request_shutdown("SIGTERM"))
    try:
        app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
