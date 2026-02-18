# ClipGenius

100% local viral clip generator. Transforms YouTube or local videos into vertical shorts (9:16) optimized for TikTok, Instagram Reels, and YouTube Shorts.

All analysis runs on your machine — no data ever leaves your computer.

---

## Features

| Module | Description |
|--------|-------------|
| **AI Analysis** | Viral moment detection via Phi-4-mini (offline, llama.cpp) |
| **Smart Cropping** | MediaPipe face tracking + blur-fill for vertical format |
| **Animated Subtitles** | Local Whisper transcription, TikTok-style rendering |
| **Hook Optimizer** | Automatic optimization of the first 3 seconds |
| **Presets** | Podcast, Gaming, Vlog, Tutorial — automatic content type detection |
| **Native Interface** | Flask + PyWebView, no external browser required |

---

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.10+
- FFmpeg

```bash
# Install FFmpeg (macOS)
brew install ffmpeg
```

---

## Installation

```bash
git clone https://github.com/Zeffut/ClipGenius.git
cd ClipGenius/App
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Phi-4-mini model is downloaded automatically on first launch.

---

## Usage

```bash
cd App
python app.py
```

The interface opens in a native window. Workflow:

1. **Source** — Paste a YouTube URL or select a local file
2. **Preset** — Automatic selection based on content (or manual override)
3. **Settings** — Adjust duration, number of clips, minimum score
4. **Generation** — Clips are created and saved to `~/Downloads`

---

## Presets

| Preset | Duration | Min Score | Best For |
|--------|----------|-----------|----------|
| Podcast | 30–90s | 0.75 | Interviews, discussions, debates |
| Gaming | 15–60s | 0.80 | Clutch plays, epic moments, highlights |
| Vlog | 15–60s | 0.70 | Storytelling, challenges, routines |
| Tutorial | 45–120s | 0.70 | How-to, tips, demonstrations |

---

## Project Structure

```
App/
  app.py                  Entry point (PyWebView)
  web_app.py              Flask + SocketIO server
  src/
    ai_analyzer.py        Local AI analysis (Phi-4-mini)
    local_llm.py          llama.cpp interface
    viral_detector.py     Audio/video detection (fallback)
    smart_cropper.py      Cropping + MediaPipe tracking
    clip_generator.py     Generation pipeline
    subtitles.py          Whisper transcription
    enriched_subtitles.py Animated subtitles
    hook_optimizer.py     Hook optimization
    adaptive_duration.py  Platform-specific durations
    presets.py            Preset configuration
  web/templates/
    index.html            Full interface (SPA)
  tests/                  Test suite
```

---

## Tests

```bash
cd App

# Full test suite
python tests/test_pipeline.py

# Quick tests (imports only)
python tests/test_pipeline.py --quick

# Individual tests
python tests/test_import.py
python tests/test_local_llm.py
python tests/test_subtitles.py
```

---

## Troubleshooting

**FFmpeg not found** — Verify with `ffmpeg -version`, reinstall via `brew install ffmpeg`.

**No clips generated** — Lower the minimum score (0.70 recommended) or the minimum duration.

**Port in use** — Free port 5001: `lsof -ti:5001 | xargs kill -9`, then relaunch.

---

## Tech Stack

Python · MoviePy · OpenCV · MediaPipe · Whisper (mlx-whisper) · FFmpeg · Flask · SocketIO · PyWebView · Click · Rich · llama.cpp (Phi-4-mini)

---

## License

MIT

---
