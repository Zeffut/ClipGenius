# ClipGenius Changelog

All notable changes to this project will be documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [beta] - 2026-01-22

### Major Overhaul — Native Web Interface

#### Added
- **Native web interface** with PyWebView (replaces CLI)
- **4 smart presets**: Podcast, Gaming, Vlog, Tutorial
- **Automatic preset detection** based on ~670 keywords
- **Tutorial preset** with 270 keywords (code, design, DIY, music, etc.)
- **Modern UI** with purple/blue gradient design
- **UI lock** during processing (prevents user errors)
- **EventSource streaming** for real-time progress tracking
- **Automatic reconnection** to ongoing jobs
- **YouTube URL validation** with 500ms debounce
- **macOS optimizations**: App Nap, caffeinate, CPU priority (2.3x faster)
- **Local file support** via drag & drop
- **Mini video player** with preview before generation
- **localStorage state management** for persistence

#### Changed
- **Single-line presets** instead of 2–3 rows (saves vertical space)
- **Simplified filters**: Removed video quality, viral score, Whisper model
- **Optimal values hardcoded**: 1080p, score 0.75, Whisper turbo
- **Improved preset detection**: Favors specific presets over generic vlog
- **Massive vocabulary**: Gaming 350+ words, Vlog 200+, Podcast 170+
- **Complete README.md** with updated documentation

#### Removed
- **main.py** (obsolete CLI, replaced by web interface)
- **test_simple.html** (temporary test file)
- **FILTERS_REDESIGN.md** (temporary working doc)
- **PERFORMANCE_OPTIMIZATION.md** (temporary working doc)
- **Cache files**: `__pycache__`, `.cache`, `.DS_Store`
- **Temporary videos** in downloads/ and uploads/
- **Performance notification popup** (macOS optimizations suffice)

**Note**: `web_app.py` is KEPT — it is the Flask server required for the application to run!

#### Cleanup & Organization
- **Optimized repo structure**: Well-organized files
- **Complete documentation**: README, AGENTS, STRUCTURE, CHANGELOG
- **.gitattributes** for EOL and binary handling
- **.gitkeep** for empty temporary directories
- **Cleaned directories**: output/, downloads/, uploads/ emptied

### Bug Fixes
- **Consistent input height**: Select dropdowns aligned with other inputs
- **"Next" button disabled** during YouTube URL validation
- **Heuristic preset detection**: Tutorial detected as priority
- **Background performance**: Video processing not slowed by UI

### Statistics
- **~14,500 lines** of Python code
- **~4,500 lines** of HTML/CSS/JavaScript
- **16 modules** in src/
- **4 presets** with smart detection
- **~670 keywords** total for detection

---

## [1.0.0] - 2024-12-14

### Initial Release — CLI

#### Added
- **CLI with Click** for clip generation
- **Local AI analysis** with Phi-4-mini (100% offline)
- **Audio/video fallback** when LLM is unavailable
- **Smart cropping** with MediaPipe
- **Whisper subtitles** with automatic transcription
- **Enriched subtitles** TikTok-style with emojis
- **Hook optimizer** for the first 3 seconds
- **Adaptive durations** per platform
- **Audio overlay** with background music
- **Thumbnail generator** for thumbnails
- **Smart cropper** with blur-fill
- **Presets**: Podcast, Gaming, Vlog

#### Initial Structure
```
ClipGenius/
├── main.py              # Main CLI
├── src/                 # 16 Python modules
├── tests/               # 4 tests
├── requirements.txt
└── README.md
```

---

## Version Format

**MAJOR.MINOR.PATCH**

- **MAJOR**: Breaking changes incompatible with previous versions
- **MINOR**: New backwards-compatible features
- **PATCH**: Backwards-compatible bug fixes

### Change Types

- **Added**: New features
- **Changed**: Changes to existing features
- **Deprecated**: Features to be removed in a future release
- **Removed**: Features removed
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

---

**Maintained by ClipGenius Team**
