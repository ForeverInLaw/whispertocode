#!/usr/bin/env python3
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-file binary with PyInstaller.")
    parser.add_argument(
        "--name",
        default="riva-ptt",
        help="Base executable name passed to PyInstaller.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help=(
            "Optional explicit artifact suffix. "
            "Example: linux-gnu, linux-musl, macos, windows."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    dist_dir = root / "dist"
    build_dir = root / "build"

    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    if dist_dir.exists():
        shutil.rmtree(dist_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        args.name,
        "--collect-submodules",
        "riva",
        "--collect-submodules",
        "grpc",
        "--collect-submodules",
        "google.protobuf",
        "--collect-submodules",
        "pynput",
        "--collect-submodules",
        "openai",
        "--collect-data",
        "sounddevice",
        "--collect-data",
        "certifi",
        "ptt_whisper.py",
    ]

    print("Building binary with PyInstaller...")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)

    is_windows = os.name == "nt"
    binary_name = f"{args.name}.exe" if is_windows else args.name
    binary_path = dist_dir / binary_name

    if not binary_path.exists():
        print("Build finished but binary was not found in dist/.", file=sys.stderr)
        return 1

    os_tag = args.artifact_tag
    if not os_tag:
        os_tag = {
            "Windows": "windows",
            "Linux": "linux",
            "Darwin": "macos",
        }.get(platform.system(), "unknown")

    tagged_name = f"{args.name}-{os_tag}{'.exe' if is_windows else ''}"
    tagged_path = dist_dir / tagged_name
    shutil.copy2(binary_path, tagged_path)

    print(f"Built binary: {binary_path}")
    print(f"Tagged copy:  {tagged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
