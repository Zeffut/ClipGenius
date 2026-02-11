# ClipGenius

Generateur de clips viraux 100% local. Transforme des videos YouTube ou locales en shorts verticaux (9:16) optimises pour TikTok, Instagram Reels et YouTube Shorts.

Toute l'analyse est faite sur votre machine : aucune donnee ne quitte votre ordinateur.

---

## Fonctionnalites

| Module | Description |
|--------|-------------|
| **Analyse IA** | Detection des moments viraux via Phi-4-mini (offline, llama.cpp) |
| **Recadrage intelligent** | Suivi de visage MediaPipe + remplissage flou pour le format vertical |
| **Sous-titres animes** | Transcription Whisper locale, rendu style TikTok |
| **Hook Optimizer** | Optimisation automatique des 3 premieres secondes |
| **Presets** | Podcast, Gaming, Vlog, Tutoriel -- detection automatique du type |
| **Interface native** | Flask + PyWebView, pas de navigateur externe |

---

## Prerequis

- macOS (Apple Silicon recommande)
- Python 3.10+
- FFmpeg

```bash
# Installer FFmpeg (macOS)
brew install ffmpeg
```

---

## Installation

```bash
git clone https://github.com/votre-username/ClipGenius.git
cd ClipGenius/App
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Le modele Phi-4-mini est telecharge automatiquement au premier lancement.

---

## Utilisation

```bash
cd App
python app.py
```

L'interface s'ouvre dans une fenetre native. Le workflow :

1. **Source** -- Coller une URL YouTube ou selectionner un fichier local
2. **Preset** -- Choix automatique selon le contenu (ou selection manuelle)
3. **Parametres** -- Ajuster duree, nombre de clips, score minimum
4. **Generation** -- Les clips sont crees et sauvegardes dans `~/Telechargements`

---

## Presets

| Preset | Duree | Score min | Adapte pour |
|--------|-------|-----------|-------------|
| Podcast | 30-90s | 0.75 | Interviews, discussions, debates |
| Gaming | 15-60s | 0.80 | Clutch, moments epiques, highlights |
| Vlog | 15-60s | 0.70 | Storytelling, challenges, routines |
| Tutoriel | 45-120s | 0.70 | How-to, tips, demonstrations |

---

## Structure du projet

```
App/
  app.py                  Point d'entree (PyWebView)
  web_app.py              Serveur Flask + SocketIO
  src/
    ai_analyzer.py        Analyse IA locale (Phi-4-mini)
    local_llm.py          Interface llama.cpp
    viral_detector.py     Detection audio/video (fallback)
    smart_cropper.py      Recadrage + suivi MediaPipe
    clip_generator.py     Pipeline de generation
    subtitles.py          Transcription Whisper
    enriched_subtitles.py Sous-titres animes
    hook_optimizer.py     Optimisation du hook
    adaptive_duration.py  Durees par plateforme
    presets.py            Configuration des presets
  web/templates/
    index.html            Interface complete (SPA)
  tests/                  Suite de tests
```

---

## Tests

```bash
cd App

# Tests complets
python tests/test_pipeline.py

# Tests rapides (imports uniquement)
python tests/test_pipeline.py --quick

# Tests individuels
python tests/test_import.py
python tests/test_local_llm.py
python tests/test_subtitles.py
```

---

## Depannage

**FFmpeg introuvable** -- Verifier avec `ffmpeg -version`, reinstaller via `brew install ffmpeg`.

**Aucun clip genere** -- Baisser le score minimum (0.70 recommande) ou la duree minimale.

**Port occupe** -- Liberer le port 5001 : `lsof -ti:5001 | xargs kill -9`, puis relancer.

---

## Stack technique

Python -- MoviePy -- OpenCV -- MediaPipe -- Whisper (mlx-whisper) -- FFmpeg -- Flask -- SocketIO -- PyWebView -- Click -- Rich -- llama.cpp (Phi-4-mini)

---

## Licence

MIT

---
