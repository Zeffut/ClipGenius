# 🧹 Workspace Organization - ClipGenius

## 📂 Structure Finale (25 janvier 2026)

```
ClipGenius/
├── 📄 Fichiers racine
│   ├── app.py                  # Wrapper PyWebView → lance web_app.py
│   ├── web_app.py              # Serveur Flask principal (50k lignes)
│   ├── requirements.txt        # Dépendances Python
│   ├── README.md              # Documentation utilisateur
│   ├── AGENTS.md              # Instructions pour agents IA
│   ├── STRUCTURE.md           # Architecture détaillée
│   ├── WORKSPACE.md           # Ce fichier (organisation)
│   ├── CHANGELOG.md           # Historique des versions
│   ├── build_mac.py           # Script build macOS .app
│   └── build_mac.sh           # Script build .dmg
│
├── 📁 src/                     # Code source (16 modules Python)
│   ├── downloader.py          # YouTube downloader
│   ├── ai_analyzer.py         # Analyse IA locale Phi-4-mini
│   ├── local_llm.py           # LLM local Phi-4-mini
│   ├── viral_detector.py      # Fallback audio/video
│   ├── smart_cropper.py       # MediaPipe face detection
│   ├── clip_generator.py      # Pipeline principal
│   ├── subtitles.py           # Whisper transcription
│   ├── enriched_subtitles.py  # Sous-titres TikTok animés
│   ├── hook_optimizer.py      # Hook 3 secondes
│   ├── adaptive_duration.py   # Durées adaptatives
│   ├── advanced_audio_analyzer.py  # Émotions/événements
│   ├── audio_overlay.py       # Musique de fond
│   ├── thumbnail_generator.py # Miniatures
│   ├── auto_config.py         # Auto-configuration
│   ├── presets.py             # Configurations presets
│   ├── ass_to_moviepy.py      # Conversion ASS
│   └── tiktok_captions.py     # Style TikTok
│
├── 🧪 tests/                   # Tests unitaires (6 fichiers)
│   ├── test_import.py         # Tests imports
│   ├── test_local_llm.py      # Tests LLM local
│   ├── test_pipeline.py       # Tests pipeline complet
│   ├── test_visual.py         # Tests effets visuels
│   ├── test_server.py         # Tests serveur Flask ⬅️ DÉPLACÉ
│   └── test_subtitles.py      # Tests sous-titres ⬅️ DÉPLACÉ
│
├── 🌐 web/                     # Interface web
│   └── templates/
│       └── index.html         # UI complète (4.5k lignes)
│
├── 🎨 assets/                  # Ressources statiques
│   └── fonts/                 # Polices pour sous-titres
│
├── 🤖 models/                  # Modèles LLM (4.5GB)
│   ├── Phi-3-mini-4k-instruct-q4.gguf
│   └── Phi-4-mini-instruct.Q4_K_M.gguf
│
├── 📥 downloads/               # Vidéos YouTube (temporaire) - NETTOYÉ ✅
│   └── .gitkeep
│
├── 📤 uploads/                 # Fichiers uploadés (temporaire)
│   └── .gitkeep
│
└── 📤 output/                  # Clips générés - NETTOYÉ ✅
    └── .gitkeep
```

---

## ✅ Nettoyage effectué

### 1. Fichiers déplacés
- ✅ `test_server.py` → `tests/test_server.py`
- ✅ `test_subtitles.py` → `tests/test_subtitles.py`

### 2. Fichiers supprimés
- ✅ Tous les `__pycache__/` (racine + src/)
- ✅ Tous les `.DS_Store` (récursif)
- ✅ Vidéos dans `downloads/` (921MB libérés)
- ✅ Clips dans `output/` (609MB libérés)
- ✅ Fichier `.sanitized_*.mp4` résiduel (274MB libérés)

**Total libéré** : ~1.8GB

### 3. .gitignore amélioré
- ✅ Ajout de `**/.DS_Store` pour ignorer récursivement

### 4. Documentation mise à jour
- ✅ `STRUCTURE.md` : Ajout des 2 nouveaux tests
- ✅ Date mise à jour : 25 janvier 2026

---

## 📊 Statistiques finales

```bash
# Structure du projet
$ du -sh src/ tests/ web/
442K    src/
76K     tests/
20K     web/

# Taille totale (sans modèles)
$ du -sh --exclude=models --exclude=.venv .
~1.2GB (dont 538KB de code source)

# Modèles LLM
$ du -sh models/
4.5GB   models/
```

---

## 🔐 Fichiers .gitignore

### Ignorés automatiquement
```
__pycache__/          # Cache Python
.venv/                # Environnement virtuel
*.DS_Store            # macOS
output/               # Clips générés
downloads/            # Vidéos YouTube
uploads/              # Uploads temporaires
*.mp4, *.mp3, *.wav   # Fichiers médias
models/*.gguf         # Modèles LLM
.env                  # Secrets
```

### Trackés dans Git
```
output/.gitkeep       # Garde le dossier
downloads/.gitkeep    # Garde le dossier
uploads/.gitkeep      # Garde le dossier
models/.gitkeep       # Garde le dossier
```

---

## 🎯 Points d'entrée

### Production
```bash
python app.py                  # Lance l'app native (PyWebView)
python web_app.py              # Lance le serveur Flask seul
```

### Tests
```bash
python tests/test_pipeline.py              # Tests complets
python tests/test_pipeline.py --quick      # Tests rapides
python tests/test_server.py                # Tests serveur
python tests/test_subtitles.py             # Tests sous-titres
```

### Build
```bash
./build_mac.sh                 # Build .app + .dmg
python build_mac.py            # Build .app seul
```

---

## 📝 Conventions de nommage

### Fichiers
- `snake_case.py` - Modules Python
- `PascalCase` - Classes
- `UPPER_SNAKE_CASE` - Constantes
- `test_*.py` - Tests (dans `tests/`)

### Architecture
```
app.py            ← Wrapper (PyWebView)
   ↓ import
web_app.py        ← Serveur Flask (routes, SSE)
   ↓ import
src/*.py          ← Logique métier (16 modules)
```

---

## 🚀 Workflow Git

### Avant commit
```bash
# Vérifier les fichiers non trackés
git status

# Vérifier que les dossiers temporaires sont vides
ls output/ downloads/ uploads/

# Vérifier l'absence de __pycache__
find . -name "__pycache__" -type d
```

### Commit propre
```bash
# Tous les fichiers sources trackés
git add src/ tests/ web/

# Fichiers racine
git add app.py web_app.py requirements.txt

# Documentation
git add README.md AGENTS.md STRUCTURE.md WORKSPACE.md

# Config
git add .gitignore .gitattributes .env.example

# Build
git add build_mac.py build_mac.sh

# .gitkeep pour dossiers vides
git add output/.gitkeep downloads/.gitkeep uploads/.gitkeep
```

---

## 🔄 Maintenance régulière

### Nettoyage hebdomadaire
```bash
# Supprimer les vidéos téléchargées
rm -rf downloads/*
touch downloads/.gitkeep

# Supprimer les clips générés
rm -rf output/*
touch output/.gitkeep

# Supprimer les uploads
rm -rf uploads/*
touch uploads/.gitkeep

# Nettoyer le cache Python
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Nettoyer .DS_Store
find . -name ".DS_Store" -delete
```

### Vérification de la structure
```bash
# Afficher l'arborescence
tree -L 2 -I ".venv|__pycache__|*.pyc|.DS_Store"

# Vérifier les tailles
du -sh src/ tests/ web/ models/
du -sh output/ downloads/ uploads/
```

---

## 📞 Support

- **Issues** : GitHub Issues
- **Documentation** : README.md
- **Architecture** : STRUCTURE.md
- **Organisation** : WORKSPACE.md (ce fichier)
- **Agents IA** : AGENTS.md

---

**Dernière organisation** : 25 janvier 2026  
**Status** : ✅ Workspace ultra-propre et organisé  
**Prêt pour commit** : ✅ OUI
