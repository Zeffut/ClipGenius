"""Transcription audio via Whisper (mlx-whisper ou openai-whisper)."""

import time
import platform
import threading
from typing import List, Optional
from dataclasses import dataclass
from rich.console import Console

console = Console()

# Détecter si on est sur Mac avec Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Vérifier si mlx-whisper est disponible (optimisé pour Mac Apple Silicon)
MLX_WHISPER_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx_whisper
        MLX_WHISPER_AVAILABLE = True
        console.print("[green]mlx-whisper détecté - transcription optimisée Mac (fallback)[/green]")
    except ImportError:
        console.print("[dim]mlx-whisper non installé - pip install mlx-whisper pour accélérer sur Mac[/dim]")

# Note: mlx_whisper utilise déjà un cache singleton interne (ModelHolder)
# Le modèle n'est chargé qu'une fois et réutilisé automatiquement


@dataclass
class SubtitleSegment:
    """Représente un segment de sous-titre (compatibilité avec l'ancien code)"""
    start: float
    end: float
    text: str


@dataclass
class WordTimestamp:
    """Représente un mot avec son timestamp exact"""
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """Résultat complet d'une transcription Whisper avec word timestamps"""
    segments: List[SubtitleSegment]
    words: List[WordTimestamp]  # Tous les mots avec leurs timestamps
    language: str

    def get_words_for_segment(self, start_time: float, end_time: float, offset: float = 0) -> List[WordTimestamp]:
        """
        Retourne les mots qui tombent dans une plage de temps donnée.

        Args:
            start_time: Début du segment (temps absolu dans la vidéo originale)
            end_time: Fin du segment
            offset: Décalage à appliquer aux timestamps (pour les clips)

        Returns:
            Liste de WordTimestamp avec timestamps ajustés
        """
        result = []
        for w in self.words:
            # Le mot chevauche le segment si son début est avant la fin ET sa fin est après le début
            if w.start < end_time and w.end > start_time:
                # Ajuster les timestamps relatifs au clip
                adjusted_start = max(0, w.start - start_time - offset)
                adjusted_end = w.end - start_time - offset
                result.append(WordTimestamp(
                    word=w.word,
                    start=adjusted_start,
                    end=adjusted_end
                ))
        return result


def transcribe_with_mlx(
    video_path: str,
    model_size: str,
    language: Optional[str],
    video_duration: Optional[float],
    report_progress: callable
) -> dict:
    """Transcription optimisée avec mlx-whisper pour Mac Apple Silicon.

    Args:
        video_path: Chemin vers la vidéo à transcrire
        model_size: Taille du modèle Whisper ('tiny', 'base', 'small', 'medium', 'large', 'turbo')
        language: Code de langue (None = auto-détection)
        video_duration: Durée de la vidéo en secondes (pour estimation progression)
        report_progress: Fonction de callback (percent, message)

    Returns:
        Résultat brut de mlx_whisper.transcribe (dict)
    """
    import mlx_whisper

    report_progress(10, "Initialisation du moteur audio...")
    console.print(f"[green]Moteur audio optimisé activé[/green]")

    # Mapping des noms de modèles
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "turbo": "mlx-community/whisper-large-v3-turbo",
    }

    model_path = model_map.get(model_size, f"mlx-community/whisper-{model_size}-mlx")

    report_progress(15, "Chargement du modèle d'analyse...")

    transcribe_options = {
        "path_or_hf_repo": model_path,
        "word_timestamps": True,
        "verbose": False,
    }

    if language:
        transcribe_options["language"] = language

    report_progress(25, "Initialisation de la transcription...")
    console.print("[cyan]Transcription en cours...[/cyan]")

    # MLX Whisper affiche sa progression via tqdm sur stderr
    # On va capturer stderr pour extraire le pourcentage réel
    transcription_result = [None]
    transcription_error = [None]
    current_whisper_progress = [0]  # Pourcentage réel de Whisper (0-100)

    import sys
    import io
    import re

    def run_transcription():
        try:
            # Capturer stderr où tqdm affiche la progression
            stderr_capture = io.StringIO()

            # Créer un wrapper qui capture stderr ET l'affiche dans la console
            class TeeStderr:
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, data):
                    for stream in self.streams:
                        stream.write(data)
                    # Extraire le pourcentage de tqdm (format: " 23%|███..." ou "100%|███...")
                    match = re.search(r'(\d+)%\|', data)
                    if match:
                        current_whisper_progress[0] = int(match.group(1))
                def flush(self):
                    for stream in self.streams:
                        stream.flush()

            # Rediriger stderr vers notre wrapper
            old_stderr = sys.stderr
            sys.stderr = TeeStderr(stderr_capture, old_stderr)

            try:
                transcription_result[0] = mlx_whisper.transcribe(video_path, **transcribe_options)
            finally:
                sys.stderr = old_stderr

        except Exception as e:
            transcription_error[0] = e

    transcription_thread = threading.Thread(target=run_transcription)
    transcription_thread.start()

    # Reporter la vraie progression de Whisper
    last_reported_whisper_progress = 0
    first_progress_received = False

    while transcription_thread.is_alive():
        whisper_pct = current_whisper_progress[0]

        # Envoyer directement le pourcentage Whisper (0-100)
        # Le mapping vers la plage globale sera fait par web_app.py

        # Ignorer les sauts suspects (bug tqdm qui affiche 100% au début)
        # Si on passe directement de 0% à >50%, c'est probablement un bug
        if not first_progress_received and whisper_pct > 50:
            # Premier message et déjà >50% ? Ignorer, c'est un bug tqdm
            continue

        if whisper_pct > 0:
            first_progress_received = True

        # Plafonner à 99% pendant le traitement (100% réservé pour la fin)
        if whisper_pct >= 100:
            whisper_pct = 99

        # Envoyer la progression si elle a changé d'au moins 2%
        if whisper_pct >= last_reported_whisper_progress + 2 and whisper_pct > 0:
            report_progress(whisper_pct, f"Transcription {whisper_pct}%")
            last_reported_whisper_progress = whisper_pct

        time.sleep(0.5)  # Vérifier 2x par seconde

    transcription_thread.join()

    if transcription_error[0]:
        raise transcription_error[0]

    result = transcription_result[0]
    report_progress(95, "Extraction des mots...")

    return result


def transcribe_with_openai_whisper(
    video_path: str,
    model_size: str,
    language: Optional[str],
    video_duration: Optional[float],
    report_progress: callable
) -> dict:
    """Transcription avec Whisper standard (modèle local).

    Args:
        video_path: Chemin vers la vidéo à transcrire
        model_size: Taille du modèle Whisper
        language: Code de langue (None = auto-détection)
        video_duration: Durée de la vidéo en secondes (pour estimation progression)
        report_progress: Fonction de callback (percent, message)

    Returns:
        Résultat brut de whisper.transcribe (dict)
    """
    import torch
    import whisper

    # Détecter le device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(f"[green]GPU NVIDIA détecté: {torch.cuda.get_device_name(0)}[/green]")
        report_progress(10, f"GPU détecté: {torch.cuda.get_device_name(0)}")
    elif IS_APPLE_SILICON:
        console.print("[yellow]Tip: pip install mlx-whisper pour accélérer 10x sur Mac[/yellow]")
        report_progress(10, "Processeur Apple Silicon détecté")
    else:
        report_progress(10, "Utilisation du CPU")

    report_progress(15, "Chargement du modèle d'analyse...")
    console.print(f"[cyan]Chargement du modèle audio...[/cyan]")
    model = whisper.load_model(model_size, device=device)

    report_progress(25, "Transcription...")
    console.print("[cyan]Transcription en cours...[/cyan]")

    transcribe_options = {
        "task": "transcribe",
        "verbose": False,
        "word_timestamps": True
    }

    if language:
        transcribe_options["language"] = language

    transcription_result = [None]
    transcription_error = [None]
    current_whisper_progress = [0]  # Pourcentage réel de Whisper (0-100)

    import sys
    import io
    import re

    def run_transcription():
        try:
            # Capturer stderr où tqdm affiche la progression
            stderr_capture = io.StringIO()

            # Créer un wrapper qui capture stderr ET l'affiche dans la console
            class TeeStderr:
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, data):
                    for stream in self.streams:
                        stream.write(data)
                    # Extraire le pourcentage de tqdm (format: " 23%|███..." ou "100%|███...")
                    match = re.search(r'(\d+)%\|', data)
                    if match:
                        current_whisper_progress[0] = int(match.group(1))
                def flush(self):
                    for stream in self.streams:
                        stream.flush()

            # Rediriger stderr vers notre wrapper
            old_stderr = sys.stderr
            sys.stderr = TeeStderr(stderr_capture, old_stderr)

            try:
                transcription_result[0] = model.transcribe(video_path, **transcribe_options)
            finally:
                sys.stderr = old_stderr

        except Exception as e:
            transcription_error[0] = e

    transcription_thread = threading.Thread(target=run_transcription)
    transcription_thread.start()

    # Reporter la vraie progression de Whisper
    last_reported_whisper_progress = 0

    while transcription_thread.is_alive():
        whisper_pct = current_whisper_progress[0]

        # Mapper la progression Whisper (0-100%) vers la plage analyze (25-85%)
        analyze_progress = 25 + int(whisper_pct * 0.6)  # 0-100 -> 25-85

        # Envoyer la progression si elle a changé d'au moins 2%
        if whisper_pct >= last_reported_whisper_progress + 2:
            report_progress(analyze_progress, f"Transcription {whisper_pct}%")
            last_reported_whisper_progress = whisper_pct

        time.sleep(1)  # Vérifier toutes les secondes

    transcription_thread.join()

    if transcription_error[0]:
        raise transcription_error[0]

    result = transcription_result[0]
    report_progress(85, "Extraction des mots...")

    return result


def parse_whisper_result(
    result: dict,
    language: Optional[str],
    report_progress: callable
) -> TranscriptionResult:
    """Parse le résultat Whisper (compatible OpenAI et MLX).

    Args:
        result: Résultat brut de whisper.transcribe
        language: Langue forcée (None = utiliser la langue détectée)
        report_progress: Fonction de callback (percent, message)

    Returns:
        TranscriptionResult avec segments et mots
    """
    detected_language = result.get("language", "unknown")
    if not language:
        console.print(f"[green]Langue détectée: {detected_language}[/green]")
        report_progress(88, f"Langue détectée: {detected_language}")

    # Convertir en segments
    segments = []
    all_words = []

    total_segments = len(result.get("segments", []))
    for i, segment in enumerate(result.get("segments", [])):
        segments.append(SubtitleSegment(
            start=segment["start"],
            end=segment["end"],
            text=segment["text"].strip()
        ))

        # Extraire les mots avec timestamps
        for word_data in segment.get("words", []):
            all_words.append(WordTimestamp(
                word=word_data.get("word", "").strip(),
                start=word_data.get("start", 0),
                end=word_data.get("end", 0)
            ))

    report_progress(95, f"Finalisation: {len(segments)} segments, {len(all_words)} mots")
    console.print(f"[green]Transcription terminée: {len(segments)} segments, {len(all_words)} mots[/green]")

    report_progress(100, "Transcription complète")

    return TranscriptionResult(
        segments=segments,
        words=all_words,
        language=detected_language
    )
