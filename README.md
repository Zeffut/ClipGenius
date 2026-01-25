# ClipGenius beta - Générateur de Clips Viraux

Application native 100% locale qui transforme automatiquement des vidéos YouTube ou locales en clips viraux optimisés pour TikTok, Instagram Reels et YouTube Shorts.

## 🎯 Fonctionnalités

- **🤖 Analyse IA des moments viraux** : GPT-4o-mini ou Phi-4-mini local (100% offline)
- **🎨 4 Presets intelligents** : Podcast, Gaming, Vlog, Tutoriel (détection automatique)
- **🎬 Recadrage intelligent** : Détection de visages MediaPipe + blur-fill élégant
- **📝 Sous-titres animés** : Style TikTok avec emojis et animations
- **⚡ Hook Optimizer** : Optimise les 3 premières secondes pour maximum d'engagement
- **🎵 Audio overlay** : Ajout de musique de fond automatique
- **📊 Interface web moderne** : UI native avec PyWebView (pas de navigateur)
- **🔒 100% Local** : Toutes les données restent sur votre machine

## 📋 Prérequis

- **Python 3.9+**
- **FFmpeg** (dans le PATH)
- **macOS** (optimisé pour Mac, compatible Linux/Windows)
- **GPU recommandé** pour Whisper (plus rapide)

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/votre-username/ClipGenius.git
cd ClipGenius
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# ou
.venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances

```bash
cd App
pip install -r requirements.txt
```

### 4. Installer FFmpeg

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
```bash
# Avec Chocolatey
choco install ffmpeg
```

### 5. Configuration (optionnel)

Pour utiliser l'analyse IA cloud (GPT-4o-mini):

```bash
cd App
cp .env.example .env
# Éditer .env et ajouter: OPENAI_API_KEY=sk-votre-cle
```

**Sans clé API** : ClipGenius utilisera Phi-4-mini en local (100% offline).

## ▶️ Utilisation

### Lancer l'application

```bash
cd App
python app.py
```

L'interface web s'ouvre automatiquement dans une fenêtre native.

### Workflow

1. **Source** : Coller une URL YouTube OU sélectionner un fichier local
2. **Preset** : Choisir automatiquement selon le titre (ou manuel)
3. **Paramètres** : Ajuster durée min/max et nombre de clips
4. **Générer** : Cliquer "Lancer la génération"
5. **Résultat** : Clips générés dans `~/Downloads` (copie automatique)

## 🎨 Les 4 Presets

### 🎙️ Podcast
- **Durée** : 30-90s
- **Score** : 0.75 (moments forts)
- **Style** : Sous-titres clairs, pas d'emojis
- **Détection** : Interview, discussion, talk, conversation

### 🎮 Gaming
- **Durée** : 15-60s
- **Score** : 0.80 (moments épiques)
- **Style** : Sous-titres neon, dynamique
- **Détection** : 350+ mots-clés gaming (Minecraft, Fortnite, Valorant, etc.)

### 📹 Vlog
- **Durée** : 15-60s
- **Score** : 0.70 (moments engageants)
- **Style** : Sous-titres + emojis, chaleureux
- **Détection** : GRWM, routine, challenge, lifestyle

### 🎓 Tutoriel
- **Durée** : 45-120s (plus long pour expliquer)
- **Score** : 0.70 (moments "aha")
- **Style** : Sous-titres clairs, professionnel
- **Détection** : 270+ mots-clés (how to, tutorial, code, photoshop, etc.)

## 📁 Structure du Projet

```
ClipGenius/
├── App/                      # Application principale
│   ├── app.py                   # Entry point (PyWebView)
│   ├── web_app.py               # Serveur Flask
│   ├── requirements.txt         # Dépendances Python
│   ├── .env.example             # Template configuration
│   │
│   ├── src/                     # Modules Python
│   │   ├── downloader.py           # yt-dlp YouTube downloader
│   │   ├── ai_analyzer.py          # Analyse IA (GPT-4o-mini)
│   │   ├── local_llm.py            # Phi-4-mini local (offline)
│   │   ├── viral_detector.py       # Détection audio/video (fallback)
│   │   ├── smart_cropper.py        # MediaPipe face detection
│   │   ├── clip_generator.py       # Pipeline génération clips
│   │   ├── subtitles.py            # Whisper transcription
│   │   ├── enriched_subtitles.py   # Sous-titres animés TikTok
│   │   ├── hook_optimizer.py       # Optimisation hook 3s
│   │   ├── adaptive_duration.py    # Durées adaptatives
│   │   ├── audio_overlay.py        # Musique de fond
│   │   ├── thumbnail_generator.py  # Miniatures
│   │   ├── auto_config.py          # Auto-configuration
│   │   └── presets.py              # Configurations presets
│   │
│   ├── web/                     # Interface web
│   │   └── templates/
│   │       └── index.html          # UI complète
│   │
│   ├── tests/                   # Tests
│   ├── models/                  # Modèles LLM locaux (gitignored)
│   ├── downloads/               # Vidéos YouTube temporaires
│   ├── uploads/                 # Vidéos locales uploadées
│   └── output/                  # Clips générés
│
├── Site/                     # Website (coming soon)
├── AGENTS.md                 # Documentation pour agents IA
├── README.md                 # Ce fichier
├── CHANGELOG.md              # Historique des versions
└── STRUCTURE.md              # Documentation technique
```

## 🔧 Configuration Avancée

### Variables d'environnement (.env)

```bash
# Analyse IA cloud (optionnel)
OPENAI_API_KEY=sk-xxx

# Emojis dans sous-titres (optionnel)
PYCAPS_OPENAI_API_KEY=sk-xxx
```

### Modèles Whisper disponibles

- `tiny` - Le plus rapide (39M params)
- `base` - Rapide (74M params)
- `small` - Équilibré (244M params)
- `medium` - Précis (769M params)
- `large` - Le plus précis (1550M params)
- `turbo` - **Recommandé** - Rapide + précis (809M params)

### Qualité vidéo

- `480p` - Rapide, faible qualité
- `720p` - Équilibré
- `1080p` - **Recommandé** - Haute qualité
- `best` - Maximum (peut être lourd)

## 💡 Conseils pour Clips Viraux

### 📊 D'après les meilleures pratiques TikTok/Reels

1. **Hook dans les 3s** : Les utilisateurs scrollent vite, captez l'attention immédiatement
2. **Durée optimale** : 60-90s pour maximiser le taux de visionnage complet
3. **Sous-titres obligatoires** : 85% des vidéos sont regardées sans son
4. **Format vertical 9:16** : Optimisé pour mobile (déjà géré par ClipGenius)
5. **Postez aux bonnes heures** : 12h-14h et 19h-22h
6. **3-5 hashtags pertinents** : Évitez #fyp trop générique
7. **Musique tendance** : Ajoutez une musique populaire si approprié
8. **Engagez rapidement** : Répondez aux commentaires dans la première heure

### 🎯 Stratégies par type de contenu

**Podcast** : Extraire les moments controversés, drôles ou inspirants
**Gaming** : Clutch, moments épiques, funny moments, world records
**Vlog** : Storytelling, transformations, challenges, routines
**Tutoriel** : Résultats avant/après, tips rapides, erreurs courantes

## 🐛 Troubleshooting

### "FFmpeg not found"
```bash
# Vérifier l'installation
ffmpeg -version

# Réinstaller si nécessaire
brew install ffmpeg  # Mac
```

### "CUDA out of memory" (Whisper)
```bash
# Utiliser un modèle plus petit dans l'interface
# OU forcer CPU dans code
```

### Téléchargement YouTube lent
- YouTube peut limiter la vitesse
- Essayer qualité 720p au lieu de 1080p
- Vérifier votre connexion internet

### Aucun clip généré
- Vérifier que min_score n'est pas trop élevé (0.70-0.75 recommandé)
- Vérifier que min_duration n'est pas trop long (30-60s recommandé)
- Certaines vidéos n'ont pas de moments suffisamment viraux

### Interface ne s'ouvre pas
```bash
# Vérifier que port 5001 est libre
lsof -ti:5001 | xargs kill -9

# Relancer
python app.py
```

## 🏗️ Build Application Standalone (macOS)

```bash
# Build .app + .dmg
./build_mac.sh

# Ou Python script
python build_mac.py
```

Génère `ClipGenius.app` et `ClipGenius.dmg` dans `dist/`.

## 🧪 Tests

```bash
# Test complet
python tests/test_pipeline.py

# Test rapide (imports only)
python tests/test_pipeline.py --quick

# Test avec vidéo
python tests/test_pipeline.py --with-video path/to/video.mp4
```

## 🤝 Contribution

Les contributions sont bienvenues ! 

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 Licence

MIT License - Utilisez librement pour projets personnels et commerciaux.

## 🙏 Remerciements

- **OpenAI Whisper** - Transcription audio
- **MediaPipe** - Détection de visages
- **yt-dlp** - Téléchargement YouTube
- **MoviePy** - Traitement vidéo
- **PyWebView** - Interface native

---

**Créé avec ❤️ pour les créateurs de contenu**
