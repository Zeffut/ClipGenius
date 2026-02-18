# ClipGenius Project Structure

## Directory Tree

```
ClipGenius/
├── App/                              # Main application
│   ├── app.py                           # Entry point (PyWebView wrapper)
│   ├── web_app.py                       # Flask server + API routes
│   ├── requirements.txt                 # Python dependencies
│   ├── .env.example                     # API configuration template
│   │
│   ├── build_mac.py                     # macOS .app build script
│   ├── build_mac.sh                     # .dmg build shell script
│   │
│   ├── src/                             # Python modules (~14.5k lines)
│   │   ├── __init__.py
│   │   ├── downloader.py                  # yt-dlp YouTube downloader
│   │   ├── ai_analyzer.py                 # Local AI analysis (Phi-4-mini)
│   │   ├── local_llm.py                   # Phi-4-mini local offline
│   │   ├── viral_detector.py              # Audio/video fallback
│   │   ├── smart_cropper.py               # MediaPipe face detection
│   │   ├── clip_generator.py              # Main pipeline
│   │   ├── subtitles.py                   # Whisper transcription
│   │   ├── enriched_subtitles.py          # TikTok-style animated subtitles
│   │   ├── hook_optimizer.py              # 3-second hook optimizer
│   │   ├── adaptive_duration.py           # Adaptive durations
│   │   ├── advanced_audio_analyzer.py     # Emotions/events
│   │   ├── audio_overlay.py               # Background music
│   │   ├── thumbnail_generator.py         # Thumbnails
│   │   ├── auto_config.py                 # Auto-configuration
│   │   └── presets.py                     # Preset configurations
│   │
│   ├── web/                             # Web interface
│   │   └── templates/
│   │       └── index.html                 # Full UI
│   │
│   ├── tests/                           # Unit tests
│   │   ├── test_import.py
│   │   ├── test_local_llm.py
│   │   ├── test_pipeline.py
│   │   ├── test_visual.py
│   │   ├── test_server.py
│   │   └── test_subtitles.py
│   │
│   ├── assets/                          # Static resources
│   │   └── fonts/                         # Subtitle fonts
│   │
│   ├── models/                          # Local LLM models (gitignored)
│   ├── downloads/                       # YouTube videos (temporary)
│   ├── uploads/                         # Uploaded files (temporary)
│   └── output/                          # Generated clips
│
├── Site/                             # Website (coming soon)
│
├── README.md                         # User documentation
├── AGENTS.md                         # AI agent documentation
├── STRUCTURE.md                      # This file
├── CHANGELOG.md                      # Version history
├── .gitignore                        # Git-excluded files
└── .gitattributes                    # Git configuration
```

## Statistics

- **Total lines of code**: ~14,500 lines Python + 4,500 lines HTML/JS
- **Source modules**: 16 Python files
- **Project size**: 6.3 GB (including 4.5 GB of LLM models)
- **Tests**: 6 test files

## Entry Points

### End User
```bash
cd App
python app.py
```
Launches the native application with PyWebView.

### Developer — Tests
```bash
cd App
python tests/test_pipeline.py                    # Full test suite
python tests/test_pipeline.py --quick            # Quick tests (imports)
python tests/test_pipeline.py --with-video video.mp4  # Tests with video
python tests/test_import.py                      # Import tests
python tests/test_local_llm.py                   # Local LLM tests
python tests/test_server.py                      # Flask server tests
python tests/test_subtitles.py                   # Subtitle tests
```

### Developer — Build
```bash
cd App
./build_mac.sh              # Full build (.app + .dmg)
python build_mac.py         # Build .app only
```

## Generation Pipeline

```
1. downloader.py           → Downloads YouTube video
2. ai_analyzer.py          → Analyzes content (or local_llm.py)
3. viral_detector.py       → Audio/video fallback
4. hook_optimizer.py       → Optimizes 3s hook
5. smart_cropper.py        → Smart cropping
6. adaptive_duration.py    → Adjusts durations
7. clip_generator.py       → Generates video clips
8. subtitles.py            → Whisper transcription
9. enriched_subtitles.py   → Animated subtitles
10. audio_overlay.py       → Background music
11. thumbnail_generator.py → Thumbnails
```

## Key Dependencies

**Video Processing:**
- `moviepy>=2.0.0` — Video manipulation
- `opencv-python>=4.10.0` — Image processing
- `mediapipe>=0.10.8` — Face detection

**AI & Audio:**
- `openai-whisper>=20231117` — Transcription (local)
- `llama-cpp-python>=0.2.0` — Phi-4-mini local

**Interface:**
- `flask>=3.0.0` — Web backend
- `pywebview>=5.0.0` — Native window
- `yt-dlp>=2023.12.30` — YouTube download

**Utilities:**
- `click>=8.1.0` — CLI (legacy)
- `rich>=13.0.0` — Console formatting
- `python-dotenv>=1.0.0` — Environment variables

## Available Presets

### Podcast (30–90s, score 0.75)
- Clean subtitles, no emojis
- Detection: interview, discussion, talk

### Gaming (15–60s, score 0.80)
- Neon subtitles, dynamic
- 350+ keywords (Minecraft, Fortnite, etc.)

### Vlog (15–60s, score 0.70)
- Subtitles + emojis, warm tone
- Detection: routine, challenge, lifestyle

### Tutorial (45–120s, score 0.70)
- Clean subtitles, professional
- 270+ keywords (how to, tutorial, code, etc.)

## Security & Privacy

- **100% local**: No data sent to the cloud
- **No tracking**: Zero telemetry
- **Open source**: Auditable code
- **Temporary files**: Auto-cleaned after generation

## Performance

- **Background processing**: 2.3x faster (macOS optimizations)
- **App Nap disabled**: Continuous processing
- **Caffeinate**: Prevents sleep mode
- **CPU priority boost**: High-priority process
- **Smart caching**: Transcription reuse

## Code Conventions

- **Language**: Python 3.9+
- **Docstrings**: French
- **Variables**: English snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Classes**: PascalCase
- **Type hints**: Required
- **Formatting**: 4 spaces, ~100 chars/line

## Git Workflow

```
# Automatically ignored (.gitignore)
- __pycache__/
- .venv/
- App/output/
- App/downloads/
- App/uploads/
- *.mp4, *.mp3, *.wav
- App/models/*.gguf
- .env
- .DS_Store
```

## Support

- **Issues**: GitHub Issues
- **Documentation**: README.md
- **AI Agents**: AGENTS.md
- **Structure**: This file (STRUCTURE.md)

---

**Last updated**: January 25, 2026
**Version**: beta
**Maintainer**: ClipGenius Team
