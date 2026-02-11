# AGENTS.md

Guidance for AI coding agents working on ClipGenius - a viral clip generator.

## Project Overview

ClipGenius transforms YouTube/local videos into vertical 9:16 clips for TikTok, Instagram Reels, and YouTube Shorts. 100% offline - uses Phi-4-mini for AI analysis, Whisper for transcription, MediaPipe for face detection.

**Stack:** Python 3.10+, MoviePy, OpenCV, MediaPipe, Whisper, FFmpeg, Click, Rich, Flask

## Build & Run Commands

```bash
cd App

# Install (requires Python 3.10+, FFmpeg in PATH)
pip install -r requirements.txt

# Run application (PyWebView native window)
python app.py

# Run Flask dev server only
python web_app.py
```

### Testing

```bash
cd App

# Run all tests
python tests/test_pipeline.py

# Quick test (imports only)
python tests/test_pipeline.py --quick

# Run single test file
python tests/test_import.py
python tests/test_local_llm.py
python tests/test_subtitles.py
python tests/test_visual.py
python tests/test_server.py
```

**Note:** Uses custom test scripts (not pytest). Tests return exit code 0 on success, 1 on failure.

## Code Style

### Language Convention
- **Code/Variables:** English, snake_case
- **Docstrings/Comments:** French (e.g., "Analyse les moments viraux")

### Naming
```python
# Functions/variables: snake_case
def analyze_viral_moments(video_path: str) -> List[ViralMoment]:
    min_duration = 30.0

# Classes: PascalCase
class SmartCropper:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CLIP_DURATION = 90.0

# Private: leading underscore
def _internal_helper(self):
    self._cache = {}
```

### Type Hints (Required)
```python
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass

def process_clips(
    moments: List[ViralMoment],
    config: ClipConfig
) -> Optional[List[str]]:
    pass

@dataclass
class ViralMoment:
    start_time: float
    end_time: float
    score: float
    reason: str
```

### Import Order
```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import List, Optional

# 2. Third-party
import click
import numpy as np
from rich.console import Console
from moviepy import VideoFileClip

# 3. Local modules (relative imports in src/)
from .viral_detector import ViralMomentDetector
from .smart_cropper import SmartCropper
```

### Formatting
- **Indentation:** 4 spaces
- **Line length:** Prefer <100 chars
- **Quotes:** Single `'` for internal strings, double `"` for user-facing messages
- **CLI:** Use `click` decorators
- **Console output:** Use `rich.console.Console()`

### Error Handling & Resource Management
```python
# CRITICAL: Always use try/finally for VideoFileClip to prevent file locks
video = None
try:
    video = VideoFileClip(video_path)
    # Process video...
finally:
    if video:
        video.close()
    gc.collect()  # Force garbage collection for MoviePy
```

### Configuration Pattern
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

## Architecture

```
App/
├── app.py                 # Entry point (PyWebView)
├── web_app.py             # Flask server
├── src/                   # Source modules
│   ├── downloader.py          # yt-dlp video downloader
│   ├── viral_detector.py      # Audio/video energy analysis (fallback)
│   ├── ai_analyzer.py         # Phi-4-mini local analysis (100% offline)
│   ├── local_llm.py           # Phi-4-mini via llama.cpp
│   ├── smart_cropper.py       # MediaPipe face detection + blur-fill
│   ├── clip_generator.py      # Main video processing pipeline
│   ├── subtitles.py           # Whisper transcription (local)
│   ├── enriched_subtitles.py  # TikTok-style animated captions
│   ├── hook_optimizer.py      # First 3s optimization
│   ├── adaptive_duration.py   # Platform-specific duration
│   ├── presets.py             # Visual/subtitle presets
│   └── ...
├── tests/                 # Test files (custom framework)
└── web/templates/         # Frontend HTML
```

### Pipeline Flow
```
Input -> VideoDownloader -> LocalAIViralAnalyzer (Phi-4-mini) -> HookOptimizer
  -> SmartCropper -> AdaptiveDurationManager -> ClipGenerator
  -> SubtitleGenerator -> EnrichedSubtitleProcessor -> Output
```

## Common Pitfalls

1. **MoviePy file locks:** Always `try/finally` with `video.close()` and `gc.collect()`
2. **AAC audio errors:** Use `sanitize_audio()` from `clip_generator.py`
3. **MediaPipe warnings:** Suppressed via `os.environ['GLOG_minloglevel']`
4. **LLM model not found:** Download Phi-4-mini GGUF to `models/` folder

## Adding Features

1. Create module in `src/` with French docstrings
2. Use `@dataclass` for configuration objects
3. Add CLI option in `main.py` using `@click.option`
4. Handle errors gracefully with fallbacks
5. Add tests in `tests/`
