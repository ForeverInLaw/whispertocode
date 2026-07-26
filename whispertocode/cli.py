import argparse
import signal
import sys
from dataclasses import replace
from typing import List, Optional

from dotenv import load_dotenv

from .app import HoldToTalkRiva
from .config_store import (
    get_config_path,
    load_config_json,
    load_env_fallback,
    requires_nvidia_key,
    resolve_settings,
    save_config_json,
)
from .constants import (
    LANGUAGE_CHOICES,
    OUTPUT_MODE_RAW,
    OUTPUT_MODE_SMART,
    STT_BACKEND_LOCAL,
    STT_BACKEND_RIVA,
)
from .onboarding import MODE_SETTINGS, MODE_SETUP, run_onboarding

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
        choices=list(LANGUAGE_CHOICES),
        help="Recognition language (auto detects the spoken language)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=[STT_BACKEND_RIVA, STT_BACKEND_LOCAL],
        help=(
            "Speech-to-text backend for this run: riva (cloud) or local "
            "(whisper.cpp). Defaults to the configured backend."
        ),
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

        backend_override = getattr(args, "backend", None)
        effective_backend = backend_override or resolved.stt_backend
        needs_key = requires_nvidia_key(effective_backend, args.mode)

        force_onboarding = bool(getattr(args, "onboarding", False))
        if force_onboarding or (needs_key and not resolved.nvidia_api_key):
            # An existing config file is what makes this not a first run, and it
            # answers for both doors: --onboarding on a configured install is
            # the CLI's Settings item, and a missing key beside a config that
            # already exists is one value to fix, not an install to complete.
            mode = MODE_SETTINGS if config_exists else MODE_SETUP
            onboarding_result = run_onboarding(resolved, mode=mode)
            if onboarding_result is None:
                print("Onboarding canceled.", file=sys.stderr)
                return 1
            save_config_json(onboarding_result)
            resolved = onboarding_result
        elif not config_exists and resolved.nvidia_api_key:
            # Auto-migrate env-based setup to persistent config.
            save_config_json(resolved)

        if backend_override:
            resolved = replace(resolved, stt_backend=backend_override)

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
