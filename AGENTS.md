# AGENTS.md

This file provides guidance to coding agents working on the ClipGenius repository.

## Project Overview

ClipGenius beta is a viral clip generator that transforms YouTube videos or local files into vertical 9:16 clips optimized for TikTok, Instagram Reels, and YouTube Shorts. It uses AI analysis (GPT-4o-mini or local Phi-4-mini), Whisper transcription, MediaPipe face detection, and advanced audio/video processing.

**Tech Stack:** Python 3.9+, MoviePy, OpenCV, MediaPipe, Whisper, FFmpeg, Click, Rich

## Build & Run Commands

### Installation
```bash
cd App

# Install dependencies (requires Python 3.9+, FFmpeg in PATH)
pip install -r requirements.txt

# Install with virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
cd App

# Launch the application (PyWebView native window)
python app.py

# Or run Flask server only (for development)
python web_app.py
```

### Testing

```bash
cd App

# Run all tests
python tests/test_pipeline.py

# Quick test (imports only)
python tests/test_pipeline.py --quick

# Test with specific video file
python tests/test_pipeline.py --with-video path/to/video.mp4

# Run individual test files
python tests/test_import.py
python tests/test_local_llm.py
python tests/test_visual.py
```

**Note:** This project uses custom test scripts (not pytest/unittest). Tests are located in:
- `App/tests/test_pipeline.py` - Main pipeline integration tests
- `App/tests/test_*.py` - Individual component tests

### Development Server
```bash
cd App

# Flask dev server (for web interface testing)
python web_app.py
```

## Code Style & Conventions

### Language & Documentation
- **Primary Language:** Python 3.9+
- **Docstrings/Comments:** French (e.g., "Analyse les moments viraux")
- **Variable Names:** English, snake_case (e.g., `video_duration`, `min_score`)

### Naming Conventions
```python
# Functions and variables: snake_case
def analyze_viral_moments(video_path: str) -> List[ViralMoment]:
    min_duration = 30.0
    
# Classes: PascalCase
class SmartCropper:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CLIP_DURATION = 90.0
PLATFORM_SPECS = {...}

# Private methods/attributes: leading underscore
def _internal_helper(self):
    self._cache = {}
```

### Type Hints & Imports
```python
# Always use type hints
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass

def process_clips(
    moments: List[ViralMoment],
    config: ClipConfig
) -> Optional[List[str]]:
    pass

# Dataclasses for data structures
@dataclass
class ViralMoment:
    start_time: float
    end_time: float
    score: float
    reason: str
```

### Import Organization
```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import List, Optional

# 2. Third-party libraries
import click
import numpy as np
from rich.console import Console
from moviepy import VideoFileClip

# 3. Local modules (relative imports in src/)
from .viral_detector import ViralMomentDetector
from .smart_cropper import SmartCropper
```

### Formatting & Structure
- **Line Length:** No strict limit, but prefer <100 chars for readability
- **Indentation:** 4 spaces
- **String Quotes:** Single quotes `'` for strings, double quotes `"` for user-facing messages
- **CLI Framework:** Use `click` decorators for CLI arguments
- **Console Output:** Use `rich.console.Console()` for formatted output

### Error Handling & Resource Management
```python
# Always use try/finally for video resources
from moviepy import VideoFileClip

video = None
try:
    video = VideoFileClip(video_path)
    # Process video...
finally:
    if video:
        video.close()
    gc.collect()  # Force garbage collection for MoviePy

# Handle missing API keys gracefully with fallbacks
if not openai_key:
    console.print("[yellow]⚠ OpenAI key missing, using fallback[/yellow]")
    # Fallback to audio/video energy analysis
```

### Configuration & Dataclasses
```python
from dataclasses import dataclass

@dataclass
class ClipConfig:
    """Configuration for clip generation"""
    min_clip_duration: float = 30.0
    max_clip_duration: float = 90.0
    output_width: int = 1080
    output_height: int = 1920
    smart_crop: bool = True
```

## Architecture & Module Organization

### Directory Structure
```
ClipGenius/
├── App/                      # Application principale
│   ├── app.py                    # Entry point (PyWebView)
│   ├── web_app.py                # Flask server
│   ├── build_mac.py              # macOS build script
│   ├── requirements.txt          # Dependencies
│   ├── src/                      # Source modules
│   │   ├── downloader.py             # yt-dlp video downloader
│   │   ├── viral_detector.py         # Audio/video energy analysis (fallback)
│   │   ├── ai_analyzer.py            # GPT-4o-mini/Phi-4-mini content analysis
│   │   ├── local_llm.py              # Local Phi-4-mini via llama.cpp
│   │   ├── smart_cropper.py          # MediaPipe face detection + blur-fill
│   │   ├── clip_generator.py         # Main video processing pipeline
│   │   ├── subtitles.py              # Whisper transcription
│   │   ├── enriched_subtitles.py     # TikTok-style animated captions
│   │   ├── hook_optimizer.py         # First 3 seconds optimization
│   │   ├── advanced_audio_analyzer.py # Emotion/event detection
│   │   ├── adaptive_duration.py      # Platform-specific duration
│   │   ├── thumbnail_generator.py    # Thumbnail creation
│   │   ├── audio_overlay.py          # Background music
│   │   ├── auto_config.py            # Intelligent auto-configuration
│   │   └── presets.py                # Visual/subtitle presets
│   ├── tests/                    # Test files
│   ├── web/                      # Frontend templates
│   ├── models/                   # AI models (gitignored)
│   ├── downloads/                # Downloaded videos (gitignored)
│   ├── uploads/                  # Uploaded files (gitignored)
│   └── output/                   # Generated clips (gitignored)
├── Site/                     # Website (coming soon)
├── AGENTS.md                 # This file
├── README.md                 # Documentation
├── CHANGELOG.md              # Version history
└── STRUCTURE.md              # Technical details
```

### Key Design Patterns

1. **Dataclass-based Configuration:** Use `@dataclass` for all config objects
2. **Fallback Mechanisms:** AI analysis → Audio energy peaks if no API key
3. **Resource Cleanup:** Always use `try/finally` for VideoFileClip objects
4. **Progress Display:** Use `rich.progress.Progress` for long operations
5. **Automatic Clip Count:** Only moments above `min_score` threshold (no quota filling)

### Pipeline Flow
```
Input → VideoDownloader → AIViralAnalyzer (or fallback) → HookOptimizer 
  → SmartCropper → AdaptiveDurationManager → ClipGenerator 
  → SubtitleGenerator → EnrichedSubtitleProcessor → Output
```

## Working with the Codebase

### Adding New Features

1. **Create module in `src/`** with French docstrings
2. **Use dataclasses** for configurations
3. **Add CLI option** in `main.py` using `@click.option`
4. **Handle errors gracefully** with fallbacks
5. **Add tests** in `tests/` or `test_pipeline.py`
6. **Update CLAUDE.md** if changing pipeline flow

### Common Tasks

**Add a new preset:**
```python
# Edit src/presets.py
PRESETS["my_preset"] = ClipGeniusPreset(
    name="my_preset",
    description="My custom preset",
    # ... configuration
)
```

**Modify video effects:**
```python
# Edit src/clip_generator.py
# Effects: zoom, blur-fill, color grading, sharpening, vignette, Ken Burns
```

**Change subtitle styling:**
```python
# Edit src/enriched_subtitles.py (word classification, colors)
# Or src/subtitles.py (pycaps templates)
```

## Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...              # Optional, for AI analysis
PYCAPS_OPENAI_API_KEY=sk-...       # Optional, for emoji generation
```

## Dependencies & External Tools

**Required:**
- FFmpeg (must be in PATH)
- Python 3.9+

**Key Python Packages:**
- `moviepy>=2.0.0` - Video processing
- `openai-whisper>=20231117` - Transcription
- `mediapipe>=0.10.8` - Face detection
- `yt-dlp>=2023.12.30` - YouTube downloading
- `click>=8.1.0` - CLI framework
- `rich>=13.0.0` - Console formatting

**Optional:**
- `openai>=1.0.0` - GPT-4o-mini analysis (fallback available)
- `llama-cpp-python>=0.2.0` - Local Phi-4-mini (100% offline)

## Common Pitfalls & Solutions

1. **MoviePy file locks:** Always use `try/finally` and `gc.collect()`
2. **AAC audio errors:** Use `sanitize_audio()` in `clip_generator.py`
3. **MediaPipe warnings:** Already suppressed via `os.environ['GLOG_minloglevel']`
4. **Windows encoding:** UTF-8 reconfiguration in `main.py:28-30`
5. **Missing OpenAI key:** Automatic fallback to audio/video energy analysis

## Testing Guidelines

- Test files use custom framework (not pytest)
- Run `python tests/test_pipeline.py` for full suite
- Tests check: imports, hook analysis, subtitle enrichment, adaptive duration
- Use `--quick` flag for fast import-only tests
- Mock external dependencies when possible

## Performance Considerations

- GPU recommended for Whisper (faster transcription)
- VideoToolbox acceleration on macOS (automatic)
- Use `--no-smart-crop` to skip MediaPipe (faster, less accurate)
- Use presets like `fast` for quick iterations
- Temp files cleaned automatically unless `--keep-source`

---

**For detailed pipeline documentation, see CLAUDE.md**
