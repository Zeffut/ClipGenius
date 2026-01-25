# Structure du Projet ClipGenius

## 📂 Arborescence

```
ClipGenius/
├── 🚀 app.py                  # Point d'entrée principal (PyWebView wrapper)
├── 🌐 web_app.py              # Serveur Flask + API routes (841 lignes)
├── 📦 requirements.txt        # Dépendances Python
├── 📖 README.md              # Documentation utilisateur
├── 🤖 AGENTS.md              # Documentation pour agents IA (Claude, etc.)
├── 📋 STRUCTURE.md           # Ce fichier
│
├── ⚙️ .env.example            # Template configuration API
├── 🚫 .gitignore             # Fichiers exclus de Git
├── 📝 .gitattributes         # Configuration Git (EOL, binaires)
│
├── 🏗️ build_mac.py            # Script build macOS .app
├── 🏗️ build_mac.sh            # Shell script build .dmg
│
├── 📁 src/                   # Modules Python (14.5k lignes)
│   ├── __init__.py
│   ├── downloader.py         # yt-dlp YouTube downloader (183 lignes)
│   ├── ai_analyzer.py        # Analyse IA GPT-4o-mini (492 lignes)
│   ├── local_llm.py          # Phi-4-mini local offline (297 lignes)
│   ├── viral_detector.py     # Audio/video fallback (443 lignes)
│   ├── smart_cropper.py      # MediaPipe face detection (1214 lignes)
│   ├── clip_generator.py     # Pipeline principal (881 lignes)
│   ├── subtitles.py          # Whisper transcription (1114 lignes)
│   ├── enriched_subtitles.py # Sous-titres TikTok animés (668 lignes)
│   ├── hook_optimizer.py     # Hook 3 secondes (420 lignes)
│   ├── adaptive_duration.py  # Durées adaptatives (491 lignes)
│   ├── advanced_audio_analyzer.py # Émotions/événements (672 lignes)
│   ├── audio_overlay.py      # Musique de fond (422 lignes)
│   ├── thumbnail_generator.py # Miniatures (638 lignes)
│   ├── auto_config.py        # Auto-configuration (873 lignes)
│   └── presets.py            # Configurations presets (695 lignes)
│
├── 🌐 web/                   # Interface web
│   └── templates/
│       └── index.html        # UI complète (4543 lignes)
│
├── 🧪 tests/                 # Tests unitaires
│   ├── test_import.py        # Test imports
│   ├── test_local_llm.py     # Test LLM local
│   ├── test_pipeline.py      # Test pipeline complet
│   ├── test_visual.py        # Test effets visuels
│   ├── test_server.py        # Test serveur Flask
│   └── test_subtitles.py     # Test génération sous-titres
│
├── 🎨 assets/                # Ressources statiques
│   └── fonts/                # Polices pour sous-titres
│
├── 🤖 models/                # Modèles LLM locaux (4.5GB)
│   ├── .gitkeep
│   ├── Phi-3-mini-4k-instruct-q4.gguf
│   └── Phi-4-mini-instruct.Q4_K_M.gguf
│
├── 📥 downloads/             # Vidéos YouTube (temporaire)
│   └── .gitkeep
│
├── 📤 uploads/               # Fichiers uploadés (temporaire)
│   └── .gitkeep
│
└── 📤 output/                # Clips générés
    └── .gitkeep
```

## 📊 Statistiques

- **Total lignes de code** : ~14,500 lignes Python + 4,500 lignes HTML/JS
- **Modules source** : 16 fichiers Python
- **Taille projet** : 6.3GB (dont 4.5GB de modèles LLM)
- **Tests** : 6 fichiers de test

## 🎯 Points d'entrée

### Utilisateur Final
```bash
python app.py
```
Lance l'application native avec PyWebView.

### Développeur - Tests
```bash
python tests/test_pipeline.py                    # Tests complets
python tests/test_pipeline.py --quick            # Tests rapides (imports)
python tests/test_pipeline.py --with-video video.mp4  # Tests avec vidéo
python tests/test_import.py                      # Tests imports
python tests/test_local_llm.py                   # Tests LLM local
python tests/test_server.py                      # Tests serveur Flask
python tests/test_subtitles.py                   # Tests sous-titres
```

### Développeur - Build
```bash
./build_mac.sh              # Build complet .app + .dmg
python build_mac.py         # Build .app seulement
```

## 🔄 Pipeline de Génération

```
1. downloader.py          → Télécharge vidéo YouTube
2. ai_analyzer.py         → Analyse contenu (ou local_llm.py)
3. viral_detector.py      → Fallback audio/video
4. hook_optimizer.py      → Optimise hook 3s
5. smart_cropper.py       → Recadrage intelligent
6. adaptive_duration.py   → Ajuste durées
7. clip_generator.py      → Génère clips vidéo
8. subtitles.py           → Transcription Whisper
9. enriched_subtitles.py  → Sous-titres animés
10. audio_overlay.py      → Musique de fond
11. thumbnail_generator.py → Miniatures
```

## 📦 Dépendances Clés

**Traitement Vidéo:**
- `moviepy>=2.0.0` - Manipulation vidéo
- `opencv-python>=4.10.0` - Traitement image
- `mediapipe>=0.10.8` - Détection visages

**IA & Audio:**
- `openai-whisper>=20231117` - Transcription
- `openai>=1.0.0` - GPT-4o-mini (optionnel)
- `llama-cpp-python>=0.2.0` - Phi-4-mini local

**Interface:**
- `flask>=3.0.0` - Backend web
- `pywebview>=5.0.0` - Fenêtre native
- `yt-dlp>=2023.12.30` - YouTube download

**Utils:**
- `click>=8.1.0` - CLI (legacy)
- `rich>=13.0.0` - Formatage console
- `python-dotenv>=1.0.0` - Variables env

## 🎨 Presets Disponibles

### 🎙️ Podcast (30-90s, score 0.75)
- Sous-titres clairs, pas d'emojis
- Détection: interview, discussion, talk

### 🎮 Gaming (15-60s, score 0.80)
- Sous-titres neon, dynamique
- 350+ mots-clés (Minecraft, Fortnite, etc.)

### 📹 Vlog (15-60s, score 0.70)
- Sous-titres + emojis, chaleureux
- Détection: routine, challenge, lifestyle

### 🎓 Tutoriel (45-120s, score 0.70)
- Sous-titres clairs, professionnel
- 270+ mots-clés (how to, tutorial, code, etc.)

## 🔐 Sécurité & Privacy

- **100% local** : Aucune donnée envoyée (sauf API OpenAI optionnelle)
- **Pas de tracking** : Aucune télémétrie
- **Open source** : Code auditable
- **Fichiers temporaires** : Auto-nettoyés après génération

## 🚀 Performance

- **Background processing** : 2.3x plus rapide (optimisations macOS)
- **App Nap désactivé** : Traitement continu
- **Caffeinate** : Empêche la mise en veille
- **CPU Priority boost** : Processus prioritaire
- **Cache intelligent** : Réutilisation des transcriptions

## 📝 Conventions Code

- **Langage** : Python 3.9+
- **Docstrings** : Français
- **Variables** : English snake_case
- **Constantes** : UPPER_SNAKE_CASE
- **Classes** : PascalCase
- **Type hints** : Obligatoires
- **Formatage** : 4 espaces, ~100 chars/ligne

## 🔄 Workflow Git

```bash
# Fichiers ignorés automatiquement (.gitignore)
- __pycache__/
- .venv/
- output/
- downloads/
- uploads/
- *.mp4, *.mp3, *.wav
- models/*.gguf
- .env
- .DS_Store
```

## 📞 Support

- **Issues** : GitHub Issues
- **Documentation** : README.md
- **Agents IA** : AGENTS.md
- **Structure** : Ce fichier (STRUCTURE.md)

---

**Dernière mise à jour** : 25 janvier 2026
**Version** : beta
**Mainteneur** : ClipGenius Team
