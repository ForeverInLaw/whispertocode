# WhisperToCode (Shift hold)

Cross-platform speech-to-text tool for Windows/Linux/macOS using NVIDIA Riva Whisper (`whisper-large-v3`):
- hold `Shift` for at least `0.5s` -> microphone recording starts
- release `Shift` -> audio is transcribed by Riva and typed into the currently focused input
- default mode is `RAW`; optional `SMART` mode rewrites STT output via NVIDIA Nemotron
- app runs in system tray by default (mode switching and exit are available from tray icon)
- during active recording a floating `150x100` overlay capsule shows realtime input EQ + current mode
- languages: Russian, English, Polish, German, Spanish (`--language auto`, `ru`, `en`, `pl`, `de`, `es`)

## Requirements

- Python 3.10+
- Working microphone
- NVIDIA API key (entered in first-run onboarding UI)

## Install

```bash
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If `sounddevice` install fails on Linux, install PortAudio first (`portaudio19-dev` or equivalent package for your distro).

## First-run setup

On first launch, if no key is configured, the app opens a UI onboarding wizard:
- Step 1: enter `NVIDIA_API_KEY`
- Step 2+: optionally customize Riva/Nemotron endpoints and model settings
- Settings are saved to a JSON config file in OS app config directory.

Config file location:
- Windows: `%APPDATA%\WhisperToCode\config.json`
- macOS: `~/Library/Application Support/WhisperToCode/config.json`
- Linux: `${XDG_CONFIG_HOME:-~/.config}/whispertocode/config.json`

Environment variables are still supported as fallback/migration source, but `.env` next to binary is no longer required.

## Run

```bash
python -m whispertocode
```

Useful options:

```bash
python -m whispertocode --language auto
python -m whispertocode --language ru
python -m whispertocode --language en
python -m whispertocode --language pl
python -m whispertocode --language de
python -m whispertocode --language es
python -m whispertocode --hold-delay 0.7
python -m whispertocode --mode raw
python -m whispertocode --mode smart
python -m whispertocode --no-tray
python -m whispertocode --debug-console
python -m whispertocode --onboarding
```

## Modes

- `RAW` (default): types recognized text directly.
- `SMART`: sends recognized text to Nemotron and streams rewritten text for better readability.
- SMART keeps the source language and applies light editing only.
- SMART fallback (no streamed output yet): app logs error and types RAW text.
- SMART fallback (partial streamed output already typed): app keeps partial text and logs error.

## Build Binaries

Local build for current OS:

```bash
pip install -r requirements.txt -r requirements-build.txt
python build_binary.py
```

Output:
- `dist/riva-ptt` (or `dist/riva-ptt.exe` on Windows)
- `dist/riva-ptt-linux|macos|windows[.exe]`

Build CI binaries (workflow):
- use GitHub Actions workflow: `.github/workflows/build-binaries.yml`
- Native multi-arch artifacts:
- `windows-x64`
- `macos-arm64`
- `linux-x64`
- `linux-arm64`
- Linux distro artifacts (`x64`):
- `ubuntu-x64`
- `debian-x64`
- `kali-x64`
- `arch-x64`
- `arch-x64` currently uses the `linux-x64` binary as an alias artifact for faster releases
- push a tag like `v1.0.0` to auto-publish all artifacts to GitHub Release

## Controls

- `Shift` (hold >= 0.5s): record
- `Shift` (release): transcribe and type text
- Tray icon menu (default):
  - switch mode (`RAW` / `SMART`)
  - open `Settings...` onboarding to edit key/endpoints/models
  - show/hide debug console (Windows)
  - exit app
- Overlay capsule:
  - appears only when recording actually starts (after hold delay)
  - hides immediately when recording stops
  - always-on-top, bottom-center, click-through
- `--no-tray`: fallback to console controls (`Left`/`Right` switch mode, `Esc` exits on Windows)
- `Ctrl+C`: exit (console/no-tray mode)

## Notes

- On macOS, grant Accessibility permissions to the terminal/Python app to allow global keyboard listening/typing.
- Typing happens in the currently focused window (chat, terminal, editor, etc.).
- Riva endpoint and function id for `whisper-large-v3` are preconfigured in code.
- Nemotron reasoning stream is printed to console; only final content stream is typed.
