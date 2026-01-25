#!/usr/bin/env python3
"""
Script de packaging pour créer ClipGenius.app
Crée une application Mac autonome (.app bundle)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

print("\n" + "="*60)
print("  ClipGenius - Création de l'application Mac")
print("="*60 + "\n")

# Vérifier que py2app est installé
try:
    import py2app
    print("✓ py2app installé")
except ImportError:
    print("❌ py2app n'est pas installé")
    print("\nInstallation requise:")
    print("  pip install py2app\n")
    sys.exit(1)

# Vérifier les autres dépendances
required_packages = ['webview', 'flask', 'moviepy', 'whisper']
missing = []

for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg} installé")
    except ImportError:
        missing.append(pkg)
        print(f"❌ {pkg} manquant")

if missing:
    print(f"\n❌ Packages manquants: {', '.join(missing)}")
    print("Installez-les avec: pip install -r requirements.txt\n")
    sys.exit(1)

print("\n🔨 Création du bundle .app...")

# Configuration py2app
setup_py_content = """
from setuptools import setup

APP = ['mac_app.py']
DATA_FILES = [
    ('web/templates', ['web/templates/index.html']),
    ('src', [
        'src/__init__.py',
        'src/downloader.py',
        'src/viral_detector.py',
        'src/smart_cropper.py',
        'src/clip_generator.py',
        'src/subtitles.py',
        'src/ai_analyzer.py',
        'src/local_llm.py',
        'src/hook_optimizer.py',
        'src/advanced_audio_analyzer.py',
        'src/enriched_subtitles.py',
        'src/adaptive_duration.py',
        'src/thumbnail_generator.py',
        'src/audio_overlay.py',
        'src/auto_config.py',
        'src/presets.py',
    ]),
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'assets/icon.icns' if os.path.exists('assets/icon.icns') else None,
    'plist': {
        'CFBundleName': 'ClipGenius',
        'CFBundleDisplayName': 'ClipGenius',
        'CFBundleIdentifier': 'com.clipgenius.app',
        'CFBundleVersion': '0.9.0',
        'CFBundleShortVersionString': 'beta',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15.0',
    },
    'packages': [
        'flask',
        'webview',
        'moviepy',
        'whisper',
        'cv2',
        'numpy',
        'PIL',
        'librosa',
        'mediapipe',
        'click',
        'rich',
        'openai',
        'yt_dlp',
    ],
    'includes': [
        'web_app',
        'src.downloader',
        'src.viral_detector',
        'src.smart_cropper',
        'src.clip_generator',
        'src.subtitles',
        'src.ai_analyzer',
        'src.local_llm',
        'src.hook_optimizer',
        'src.advanced_audio_analyzer',
        'src.enriched_subtitles',
        'src.adaptive_duration',
        'src.thumbnail_generator',
        'src.audio_overlay',
        'src.auto_config',
        'src.presets',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'scipy.spatial.cKDTree',
    ],
    'strip': False,
    'optimize': 2,
    'resources': [
        'web',
        'src',
    ],
}

setup(
    name='ClipGenius',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
"""

# Écrire le fichier setup.py temporaire
setup_file = Path('setup_mac.py')
with open(setup_file, 'w') as f:
    f.write(setup_py_content)

print("✓ Configuration créée")

# Nettoyer les builds précédents
for folder in ['build', 'dist']:
    if os.path.exists(folder):
        print(f"🗑️  Nettoyage de {folder}/")
        shutil.rmtree(folder)

# Lancer py2app
print("\n🔧 Construction de l'application (cela peut prendre quelques minutes)...")
print("   Utilisez le mode alias pour les tests rapides:")
print("   python setup_mac.py py2app -A\n")

try:
    # Mode production (complet)
    result = subprocess.run(
        [sys.executable, 'setup_mac.py', 'py2app'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Erreur lors du build:")
        print(result.stderr)
        sys.exit(1)
    
    print("✓ Build terminé")
    
    # Vérifier que l'app existe
    app_path = Path('dist/ClipGenius.app')
    if app_path.exists():
        print(f"\n✅ Application créée avec succès!")
        print(f"   Chemin: {app_path.absolute()}")
        print(f"   Taille: {sum(f.stat().st_size for f in app_path.rglob('*') if f.is_file()) / (1024*1024):.1f} MB")
        
        print("\n📦 Prochaines étapes:")
        print("   1. Tester l'app: open dist/ClipGenius.app")
        print("   2. Déplacer vers /Applications:")
        print("      cp -r dist/ClipGenius.app /Applications/")
        print("   3. Créer un DMG pour distribution:")
        print("      hdiutil create -volname ClipGenius -srcfolder dist/ClipGenius.app -ov -format UDZO ClipGenius.dmg")
        
    else:
        print("❌ L'application n'a pas été créée")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Nettoyer le fichier setup temporaire
    if setup_file.exists():
        setup_file.unlink()

print("\n" + "="*60)
print("  Build terminé!")
print("="*60 + "\n")
