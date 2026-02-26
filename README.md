# Riva Push-to-Talk (Ctrl hold)

Cross-platform speech-to-text tool for Windows/Linux/macOS using NVIDIA Riva Whisper (`whisper-large-v3`):
- hold `Ctrl` for at least `0.5s` -> microphone recording starts
- release `Ctrl` -> audio is transcribed by Riva and typed into the currently focused input
- languages: Russian, English, Polish, German, Spanish (`--language auto`, `ru`, `en`, `pl`, `de`, `es`)

## Requirements

- Python 3.10+
- Working microphone
- NVIDIA API key (`NVIDIA_API_KEY`)

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

Create `.env` in project root:

```env
NVIDIA_API_KEY=your_api_key_here
```

If you run the built binary, place `.env` next to the binary file
(`riva-ptt`, `riva-ptt.exe`, or release artifact variants).

## Run

```bash
python ptt_whisper.py
```

Useful options:

```bash
python ptt_whisper.py --language auto
python ptt_whisper.py --language ru
python ptt_whisper.py --language en
python ptt_whisper.py --language pl
python ptt_whisper.py --language de
python ptt_whisper.py --language es
python ptt_whisper.py --hold-delay 0.7
```

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

- `Ctrl` (hold >= 0.5s): record
- `Ctrl` (release): transcribe and type text
- `Esc` or `Ctrl+C`: exit

## Notes

- On macOS, grant Accessibility permissions to the terminal/Python app to allow global keyboard listening/typing.
- Typing happens in the currently focused window (chat, terminal, editor, etc.).
- Riva endpoint and function id for `whisper-large-v3` are preconfigured in code.
