"""Pre-traitement audio FFmpeg pour corriger les erreurs AAC."""

import os
import subprocess
import tempfile
from rich.console import Console

console = Console()

# Constantes de configuration
SANITIZE_AUDIO_TIMEOUT: int = 300           # Timeout FFmpeg pour le nettoyage audio (secondes)


def sanitize_audio(video_path: str, output_path: str) -> bool:
    """
    Pré-traite l'audio d'une vidéo pour corriger les erreurs AAC.

    Utilise FFmpeg avec des options de tolérance aux erreurs pour:
    - Ignorer les erreurs de décodage AAC (channel element, reserved bit, etc.)
    - Ré-encoder l'audio en AAC propre
    - Copier la vidéo sans modification

    Args:
        video_path: Chemin vers la vidéo source
        output_path: Chemin vers la vidéo de sortie avec audio corrigé

    Returns:
        True si succès, False sinon
    """
    try:
        cmd = [
            'ffmpeg', '-y',
            # Options d'entrée pour tolérer les erreurs
            '-err_detect', 'ignore_err',
            '-i', video_path,
            # Copier la vidéo sans modification
            '-c:v', 'copy',
            # Ré-encoder l'audio en AAC propre
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ac', '2',  # Stéréo
            '-ar', '44100',  # Sample rate standard
            # Options pour ignorer les erreurs
            '-max_muxing_queue_size', '9999',
            '-movflags', '+faststart',
            output_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SANITIZE_AUDIO_TIMEOUT
        )

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    except Exception as e:
        console.print(f"[yellow]Avertissement: Impossible de nettoyer l'audio: {e}[/yellow]")
        return False
