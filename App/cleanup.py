"""Nettoyage des fichiers temporaires."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional


def cleanup_residual_files(video_path: Optional[str] = None, output_dir: str = "output", keep_user_files: bool = False):
    """
    Nettoie TOUS les fichiers résiduels générés par ClipGenius.

    Args:
        video_path: Chemin de la vidéo source à supprimer (si non-utilisateur)
        output_dir: Dossier de sortie à nettoyer
        keep_user_files: Si True, ne supprime pas la vidéo source

    Nettoie:
    - Vidéo source téléchargée (si non-utilisateur)
    - Fichiers temporaires MoviePy (*TEMP_MPY_*.mp4)
    - Fichiers audio nettoyés (.sanitized_*.mp4)
    - Fichiers de sous-titres (*.ass, *_sub.json)
    - Dossiers temporaires pycaps (pycaps_viral_*)
    """

    # 1. Vidéo source (sauf si fichier utilisateur)
    if not keep_user_files and video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass

    # 2. Fichiers temporaires (patterns)
    temp_patterns = [
        '*TEMP_MPY_*.mp4',      # Fichiers temporaires MoviePy
        '*_sub.json',            # Fichiers de sous-titres JSON
        '*.ass',                 # Fichiers ASS temporaires
        '.sanitized_*.mp4',      # Vidéos avec audio nettoyé
    ]

    for pattern in temp_patterns:
        # Répertoire courant
        for f in Path('.').glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass

        # Dossier output
        for f in Path(output_dir).glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass

    # 3. Dossiers temporaires pycaps
    temp_base = Path(tempfile.gettempdir())
    for pycaps_dir in temp_base.glob('pycaps_viral_*'):
        try:
            shutil.rmtree(pycaps_dir, ignore_errors=True)
        except Exception:
            pass
