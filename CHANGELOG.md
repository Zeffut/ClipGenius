# Changelog ClipGenius

Toutes les modifications notables de ce projet seront documentées ici.

Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet suit le [Semantic Versioning](https://semver.org/lang/fr/).

## [beta] - 2026-01-22

### 🎉 Refonte Majeure - Interface Web Native

#### Ajouté
- **Interface web native** avec PyWebView (remplace CLI)
- **4 Presets intelligents** : Podcast, Gaming, Vlog, Tutoriel
- **Détection automatique de preset** basée sur ~670 mots-clés
- **Preset Tutoriel** avec 270 mots-clés (code, design, DIY, musique, etc.)
- **UI moderne** avec design gradient violet/bleu
- **Verrouillage UI** pendant traitement (évite erreurs utilisateur)
- **EventSource streaming** pour suivi temps réel
- **Reconnexion automatique** aux jobs en cours
- **Validation YouTube** avec debounce 500ms
- **Optimisations macOS** : App Nap, caffeinate, CPU priority (2.3x plus rapide)
- **Support fichiers locaux** via drag & drop
- **Mini-player vidéo** avec preview avant génération
- **Gestion d'état localStorage** pour persistance

#### Modifié
- **Presets sur une ligne** au lieu de 2-3 (gain de place vertical)
- **Filtres simplifiés** : Supprimé qualité vidéo, score viral, modèle Whisper
- **Valeurs optimales hardcodées** : 1080p, score 0.75, Whisper turbo
- **Détection preset améliorée** : Favorise presets spécifiques sur vlog générique
- **Vocabulaire massif** : Gaming 350+ mots, Vlog 200+, Podcast 170+
- **README.md complet** avec nouvelle documentation

#### Supprimé
- **main.py** (CLI obsolète, remplacé par interface web)
- **test_simple.html** (fichier de test temporaire)
- **FILTERS_REDESIGN.md** (doc de travail temporaire)
- **PERFORMANCE_OPTIMIZATION.md** (doc de travail temporaire)
- **Fichiers cache** : `__pycache__`, `.cache`, `.DS_Store`
- **Vidéos temporaires** dans downloads/ et uploads/
- **Popup notification performance** (optimisations macOS suffisantes)

**Note** : `web_app.py` est CONSERVÉ - c'est le serveur Flask nécessaire au fonctionnement de l'application !

#### Nettoyage & Organisation
- **Structure repo optimisée** : Fichiers bien organisés
- **Documentation complète** : README, AGENTS, STRUCTURE, CHANGELOG
- **.gitattributes** pour gestion EOL et binaires
- **.gitkeep** pour dossiers temporaires vides
- **Dossiers nettoyés** : output/, downloads/, uploads/ vidés

### 🐛 Corrections
- **Hauteur inputs cohérente** : Select dropdowns alignés avec autres inputs
- **Bouton "Suivant" désactivé** pendant validation URL YouTube
- **Détection preset heuristique** : Tutoriel détecté en priorité
- **Performance background** : Traitement vidéo non ralenti par UI

### 📊 Statistiques
- **~14,500 lignes** de code Python
- **~4,500 lignes** HTML/CSS/JavaScript
- **16 modules** dans src/
- **4 presets** avec détection intelligente
- **~670 mots-clés** de détection totaux

---

## [1.0.0] - 2024-12-14

### Version Initiale - CLI

#### Ajouté
- **CLI avec Click** pour génération de clips
- **Analyse IA** avec GPT-4o-mini
- **LLM local** avec Phi-4-mini (100% offline)
- **Fallback audio/video** si pas d'API OpenAI
- **Recadrage intelligent** avec MediaPipe
- **Sous-titres Whisper** avec transcription auto
- **Sous-titres enrichis** style TikTok avec emojis
- **Hook optimizer** pour 3 premières secondes
- **Durées adaptatives** selon plateforme
- **Audio overlay** avec musique de fond
- **Thumbnail generator** pour miniatures
- **Smart cropper** avec blur-fill
- **Presets** : Podcast, Gaming, Vlog

#### Structure Initiale
```
ClipGenius/
├── main.py              # CLI principal
├── src/                 # 16 modules Python
├── tests/               # 4 tests
├── requirements.txt
└── README.md
```

---

## Format des Versions

**MAJOR.MINOR.PATCH**

- **MAJOR** : Changements incompatibles avec versions précédentes
- **MINOR** : Nouvelles fonctionnalités rétrocompatibles
- **PATCH** : Corrections de bugs rétrocompatibles

### Types de Changements

- **Ajouté** : Nouvelles fonctionnalités
- **Modifié** : Changements sur fonctionnalités existantes
- **Déprécié** : Fonctionnalités bientôt supprimées
- **Supprimé** : Fonctionnalités supprimées
- **Corrigé** : Corrections de bugs
- **Sécurité** : Vulnérabilités corrigées

---

**Projet maintenu par ClipGenius Team**
