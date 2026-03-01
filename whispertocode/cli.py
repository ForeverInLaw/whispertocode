import argparse
import signal
import sys
from typing import List, Optional

from dotenv import load_dotenv

from .app import HoldToTalkRiva
from .config_store import (
    get_config_path,
    load_config_json,
    load_env_fallback,
    resolve_settings,
    save_config_json,
)
from .constants import OUTPUT_MODE_RAW, OUTPUT_MODE_SMART
from .onboarding import run_onboarding

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
    parser.add_argument(
        "--onboarding",
        action="store_true",
        help="Run UI onboarding wizard before start.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        load_dotenv()
        config_path = get_config_path()
        config_exists = config_path.exists()
        resolved = resolve_settings(load_config_json(), load_env_fallback())

        force_onboarding = bool(getattr(args, "onboarding", False))
        if force_onboarding or not resolved.nvidia_api_key:
            onboarding_result = run_onboarding(resolved)
            if onboarding_result is None:
                print("Onboarding canceled.", file=sys.stderr)
                return 1
            save_config_json(onboarding_result)
            resolved = onboarding_result
        elif not config_exists and resolved.nvidia_api_key:
            # Auto-migrate env-based setup to persistent config.
            save_config_json(resolved)

        app = HoldToTalkRiva(
            sample_rate=args.sample_rate,
            language=args.language,
            hold_delay_sec=args.hold_delay,
            output_mode=args.mode,
            enable_tray=not args.no_tray,
            debug_console=args.debug_console,
            settings=resolved,
        )
        signal.signal(signal.SIGINT, lambda sig, frame: app.request_shutdown("Ctrl+C"))
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda sig, frame: app.request_shutdown("SIGTERM"))
        app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
