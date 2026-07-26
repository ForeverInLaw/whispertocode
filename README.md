# WhisperToCode (Shift hold)

Cross-platform speech-to-text tool for Windows/Linux/macOS with two interchangeable backends:
- `riva` (default): NVIDIA Riva Whisper `whisper-large-v3` in the cloud
- `local`: whisper.cpp on your own machine, no network, no API key

Behavior:
- hold `Shift` for at least `0.5s` -> microphone recording starts
- release `Shift` -> audio is transcribed and typed into the currently focused input
- default mode is `RAW`; optional `SMART` mode rewrites STT output via NVIDIA Nemotron
- app runs in system tray by default (mode switching and exit are available from tray icon)
- during active recording a floating `150x100` overlay capsule shows realtime input EQ + current mode
- languages: auto-detection or one of 21 manual choices (`--language auto`, `en`, `ru`, `de`, `es`, `fr`, `pl`, `uk`, ...)

## Requirements

- Python 3.10+ (verified on 3.12 and 3.14; CI builds on 3.13)
- Working microphone
- NVIDIA API key — required for the `riva` backend and for `SMART` rewrite mode in any backend.
  `--backend local --mode raw` runs with no key at all.

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

On first launch, if a key is needed but not configured, the app opens a setup wizard:
- **Speech backend** — cloud or local, the local model, and the API key. Two pages by default:
  this one and a review of what will be written.
- **Customize endpoints and models** — optional, ticked from this page. Adds the Riva endpoint
  page (cloud backend only) and the cleanup-model page.
- Settings are saved to a JSON config file in the OS app config directory.

The same dialog reopens from the tray as **Settings**. There it is titled Settings rather than
Setup, and the customize box starts ticked whenever your stored config already differs from the
defaults, so your own values are on the route instead of hidden behind it.

Config file location:
- Windows: `%APPDATA%\WhisperToCode\config.json`
- macOS: `~/Library/Application Support/WhisperToCode/config.json`
- Linux: `${XDG_CONFIG_HOME:-~/.config}/whispertocode/config.json`

Environment variables are still supported as fallback/migration source, but `.env` next to binary is no longer required.

One exception to that fallback: an API key removed through the wizard stays removed. An empty
`nvidia_api_key` in `config.json` is treated as a deliberate choice and wins over `NVIDIA_API_KEY`
in the environment or in a `.env`, which would otherwise put the deleted key straight back.

## Tests

```bash
PYTHONPATH=. python tests/test_modes_and_fallback.py     # and the other test_*.py files
```

The wizard has a separate check, because it needs a real PySide6 QApplication that the rest of
the suite stubs out:

```bash
PYTHONPATH=. QT_QPA_PLATFORM=offscreen python tests/wizard_offscreen_check.py < /dev/null
```

## Run

```bash
python -m whispertocode
```

Useful options:

```bash
python -m whispertocode --language auto
python -m whispertocode --language ru
python -m whispertocode --backend local
python -m whispertocode --backend riva
python -m whispertocode --hold-delay 0.7
python -m whispertocode --mode raw
python -m whispertocode --mode smart
python -m whispertocode --no-tray
python -m whispertocode --debug-console
python -m whispertocode --onboarding
```

`--backend` overrides the configured backend for one run without rewriting `config.json`.

## Local backend (whisper.cpp)

Runs entirely offline via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`pywhispercpp`). No API key, no network after the model download.

- The ggml model is downloaded on first transcription, not at startup, and cached in `<config dir>/models`.
  Override the location with `local_model_dir` in `config.json` or `LOCAL_MODEL_DIR`.
- Apple Silicon gets Metal GPU acceleration with no configuration — the published macOS arm64 wheel
  bundles `libggml-metal` and whisper.cpp defaults to `use_gpu=true`. Everywhere else runs on CPU.
- Intel Mac has no prebuilt wheel; `pip install` there falls back to a source build needing Xcode CLT + cmake.
- Language: `--language auto` detects from the first 30s window. Pinning the language is materially
  more accurate for short push-to-talk clips.
- The backend requires 16 kHz audio, which is the default `--sample-rate`.

Model sizes (download on first use):

| Model | Size | Notes |
|---|---:|---|
| `tiny-q5_1` | 32 MB | fastest start |
| `tiny` | 78 MB | default |
| `base-q5_1` | 60 MB | better than `tiny`, still smaller |
| `small-q5_1` | 190 MB | best quality per MB |
| `small` | 488 MB | reliable multilingual dictation |
| `large-v3-turbo-q5_0` | 574 MB | near-`large-v3` quality; worth it on Metal |

Quantization suffixes are not uniform upstream: `tiny`/`base`/`small` use `q5_1`, `medium`/`large` use `q5_0`.

## Modes

- `RAW` (default): types recognized text directly.
- `SMART`: sends recognized text to Nemotron and streams rewritten text for better readability.
- `SMART` needs `NVIDIA_API_KEY` even on the `local` backend — the rewrite is a cloud call. Without a
  key the app refuses to switch to `SMART` and stays in `RAW`.
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
- `macos-x64` (Intel — best effort: pywhispercpp has no x86_64 macOS wheel, so this
  job compiles whisper.cpp from source and is allowed to fail without blocking a release)
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
- Config/env keys for the backend: `stt_backend` / `STT_BACKEND` (`riva` | `local`),
  `local_model` / `LOCAL_MODEL`, `local_model_dir` / `LOCAL_MODEL_DIR`.
- Nemotron reasoning stream is printed to console; only final content stream is typed.
