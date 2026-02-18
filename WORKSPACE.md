# Workspace Organization — ClipGenius

## Final Structure (January 25, 2026)

```
ClipGenius/
├── Root Files
│   ├── app.py                  # PyWebView wrapper → launches web_app.py
│   ├── web_app.py              # Main Flask server (50k lines)
│   ├── requirements.txt        # Python dependencies
│   ├── README.md               # User documentation
│   ├── AGENTS.md               # AI agent instructions
│   ├── STRUCTURE.md            # Detailed architecture
│   ├── WORKSPACE.md            # This file (organization)
│   ├── CHANGELOG.md            # Version history
│   ├── build_mac.py            # macOS .app build script
│   └── build_mac.sh            # .dmg build script
│
├── src/                        # Source code (16 Python modules)
│   ├── downloader.py           # YouTube downloader
│   ├── ai_analyzer.py          # Local AI analysis (Phi-4-mini)
│   ├── local_llm.py            # Local LLM (Phi-4-mini)
│   ├── viral_detector.py       # Audio/video fallback
│   ├── smart_cropper.py        # MediaPipe face detection
│   ├── clip_generator.py       # Main pipeline
│   ├── subtitles.py            # Whisper transcription
│   ├── enriched_subtitles.py   # TikTok-style animated subtitles
│   ├── hook_optimizer.py       # 3-second hook optimizer
│   ├── adaptive_duration.py    # Adaptive durations
│   ├── advanced_audio_analyzer.py  # Emotions/events
│   ├── audio_overlay.py        # Background music
│   ├── thumbnail_generator.py  # Thumbnails
│   ├── auto_config.py          # Auto-configuration
│   ├── presets.py              # Preset configurations
│   ├── ass_to_moviepy.py       # ASS conversion
│   └── tiktok_captions.py      # TikTok style
│
├── tests/                      # Unit tests (6 files)
│   ├── test_import.py          # Import tests
│   ├── test_local_llm.py       # Local LLM tests
│   ├── test_pipeline.py        # Full pipeline tests
│   ├── test_visual.py          # Visual effects tests
│   ├── test_server.py          # Flask server tests (moved)
│   └── test_subtitles.py       # Subtitle tests (moved)
│
├── web/                        # Web interface
│   └── templates/
│       └── index.html          # Full UI (4.5k lines)
│
├── assets/                     # Static resources
│   └── fonts/                  # Subtitle fonts
│
├── models/                     # LLM models (4.5 GB)
│   ├── Phi-3-mini-4k-instruct-q4.gguf
│   └── Phi-4-mini-instruct.Q4_K_M.gguf
│
├── downloads/                  # YouTube videos (temporary) — CLEANED
│   └── .gitkeep
│
├── uploads/                    # Uploaded files (temporary)
│   └── .gitkeep
│
└── output/                     # Generated clips — CLEANED
    └── .gitkeep
```

---

## Cleanup Performed

### 1. Files Moved
- `test_server.py` → `tests/test_server.py`
- `test_subtitles.py` → `tests/test_subtitles.py`

### 2. Files Deleted
- All `__pycache__/` directories (root + src/)
- All `.DS_Store` files (recursive)
- Videos in `downloads/` (921 MB freed)
- Clips in `output/` (609 MB freed)
- Residual `.sanitized_*.mp4` file (274 MB freed)

**Total freed**: ~1.8 GB

### 3. Improved .gitignore
- Added `**/.DS_Store` for recursive ignoring

### 4. Updated Documentation
- `STRUCTURE.md`: Added 2 new tests
- Date updated: January 25, 2026

---

## Final Statistics

```bash
# Project structure
$ du -sh src/ tests/ web/
442K    src/
76K     tests/
20K     web/

# Total size (excluding models)
$ du -sh --exclude=models --exclude=.venv .
~1.2GB (of which 538KB is source code)

# LLM models
$ du -sh models/
4.5GB   models/
```

---

## .gitignore Files

### Automatically Ignored
```
__pycache__/          # Python cache
.venv/                # Virtual environment
*.DS_Store            # macOS
output/               # Generated clips
downloads/            # YouTube videos
uploads/              # Temporary uploads
*.mp4, *.mp3, *.wav   # Media files
models/*.gguf         # LLM models
.env                  # Secrets
```

### Tracked in Git
```
output/.gitkeep       # Keeps the directory
downloads/.gitkeep    # Keeps the directory
uploads/.gitkeep      # Keeps the directory
models/.gitkeep       # Keeps the directory
```

---

## Entry Points

### Production
```bash
python app.py                  # Launch native app (PyWebView)
python web_app.py              # Launch Flask server only
```

### Tests
```bash
python tests/test_pipeline.py              # Full test suite
python tests/test_pipeline.py --quick      # Quick tests
python tests/test_server.py                # Server tests
python tests/test_subtitles.py             # Subtitle tests
```

### Build
```bash
./build_mac.sh                 # Build .app + .dmg
python build_mac.py            # Build .app only
```

---

## Naming Conventions

### Files
- `snake_case.py` — Python modules
- `PascalCase` — Classes
- `UPPER_SNAKE_CASE` — Constants
- `test_*.py` — Tests (in `tests/`)

### Architecture
```
app.py            ← Wrapper (PyWebView)
   ↓ import
web_app.py        ← Flask server (routes, SSE)
   ↓ import
src/*.py          ← Business logic (16 modules)
```

---

## Git Workflow

### Before Committing
```bash
# Check untracked files
git status

# Verify temporary directories are empty
ls output/ downloads/ uploads/

# Verify no __pycache__ present
find . -name "__pycache__" -type d
```

### Clean Commit
```bash
# All tracked source files
git add src/ tests/ web/

# Root files
git add app.py web_app.py requirements.txt

# Documentation
git add README.md AGENTS.md STRUCTURE.md WORKSPACE.md

# Config
git add .gitignore .gitattributes .env.example

# Build
git add build_mac.py build_mac.sh

# .gitkeep for empty directories
git add output/.gitkeep downloads/.gitkeep uploads/.gitkeep
```

---

## Regular Maintenance

### Weekly Cleanup
```bash
# Delete downloaded videos
rm -rf downloads/*
touch downloads/.gitkeep

# Delete generated clips
rm -rf output/*
touch output/.gitkeep

# Delete uploads
rm -rf uploads/*
touch uploads/.gitkeep

# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Clean .DS_Store
find . -name ".DS_Store" -delete
```

### Structure Verification
```bash
# Display tree
tree -L 2 -I ".venv|__pycache__|*.pyc|.DS_Store"

# Check sizes
du -sh src/ tests/ web/ models/
du -sh output/ downloads/ uploads/
```

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: README.md
- **Architecture**: STRUCTURE.md
- **Organization**: WORKSPACE.md (this file)
- **AI Agents**: AGENTS.md

---

**Last organized**: January 25, 2026
**Status**: Workspace clean and organized
**Ready to commit**: Yes
