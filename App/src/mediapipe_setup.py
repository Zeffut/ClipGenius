"""Configuration et import lazy de MediaPipe."""

import os
import sys
import shutil
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()

# Supprimer les warnings MediaPipe/Abseil AVANT tout import
os.environ['GLOG_minloglevel'] = '2'  # Désactive les logs INFO et WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Désactive les warnings TensorFlow

# Variable globale pour le chemin temporaire MediaPipe
_MP_TEMP_DIR = None
_MP_WORKAROUND_APPLIED = False


def _setup_mediapipe_workaround():
    """
    Contourne le bug MediaPipe avec les chemins contenant des espaces sur Windows.
    Doit être appelé AVANT tout import de mediapipe.
    """
    global _MP_TEMP_DIR, _MP_WORKAROUND_APPLIED

    if _MP_WORKAROUND_APPLIED:
        return True

    if sys.platform != 'win32':
        _MP_WORKAROUND_APPLIED = True
        return False

    # Vérifier si mediapipe est déjà importé
    if 'mediapipe' in sys.modules:
        console.print("[yellow]MediaPipe deja importe, workaround impossible[/yellow]")
        _MP_WORKAROUND_APPLIED = True
        return False

    try:
        # Trouver le chemin de mediapipe sans l'importer
        # On utilise importlib pour trouver le spec
        import importlib.util
        spec = importlib.util.find_spec('mediapipe')

        if spec is None or spec.origin is None:
            _MP_WORKAROUND_APPLIED = True
            return False

        mp_path = Path(spec.origin).parent

        # Vérifier si le chemin contient des espaces
        if ' ' not in str(mp_path):
            _MP_WORKAROUND_APPLIED = True
            return False

        console.print("[yellow]Chemin MediaPipe avec espaces detecte, application du workaround...[/yellow]")

        # Créer un répertoire temporaire sans espaces
        temp_base = Path(tempfile.gettempdir()) / "mp_fix"
        temp_base.mkdir(exist_ok=True)

        # Destination pour mediapipe
        mp_dst = temp_base / "mediapipe"

        # Lire la version depuis le fichier __init__.py ou setup
        version_file_path = mp_path / "__init__.py"
        current_version = "unknown"
        if version_file_path.exists():
            content = version_file_path.read_text(encoding='utf-8', errors='ignore')
            for line in content.split('\n'):
                if '__version__' in line:
                    try:
                        current_version = line.split('=')[1].strip().strip('"\'')
                    except (IndexError, ValueError):
                        pass
                    break

        # Vérifier si on doit recopier
        our_version_file = temp_base / "mp_version.txt"

        need_copy = True
        if our_version_file.exists() and mp_dst.exists():
            try:
                if our_version_file.read_text().strip() == current_version:
                    need_copy = False
            except OSError:
                pass

        if need_copy:
            console.print(f"[dim]Copie de MediaPipe vers {temp_base}...[/dim]")
            if mp_dst.exists():
                shutil.rmtree(mp_dst, ignore_errors=True)
            shutil.copytree(mp_path, mp_dst)
            our_version_file.write_text(current_version)
            console.print("[dim]Copie terminee.[/dim]")

        _MP_TEMP_DIR = temp_base

        # Ajouter le nouveau chemin en PREMIER dans sys.path
        temp_base_str = str(temp_base)
        if temp_base_str in sys.path:
            sys.path.remove(temp_base_str)
        sys.path.insert(0, temp_base_str)

        console.print("[green]Workaround MediaPipe applique![/green]")
        _MP_WORKAROUND_APPLIED = True
        return True

    except Exception as e:
        console.print(f"[red]Erreur workaround MediaPipe: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        _MP_WORKAROUND_APPLIED = True

    return False


# Lazy import de MediaPipe pour éviter les blocages au démarrage
_mp_instance = None

def _get_mediapipe():
    """Retourne le module MediaPipe (import lazy avec workaround)"""
    global _mp_instance
    if _mp_instance is None:
        # Appliquer le workaround AVANT d'importer mediapipe
        _setup_mediapipe_workaround()

        # Maintenant importer mediapipe (depuis le nouveau chemin si workaround appliqué)
        import mediapipe as mp
        _mp_instance = mp

    return _mp_instance
