"""
Module d'analyse adaptative des segments vidéo
Détecte le type de contenu par segment et utilise la stratégie de crop appropriée
"""

import cv2
from typing import List, Optional, Tuple, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .smart_cropper import (
    SmartCropper, FocusPoint, AdaptiveCropManager, ContentType
)
from .viral_detector import ViralMoment

console = Console()

# FPS fallback par défaut
DEFAULT_ANALYSIS_FPS: float = 30.0


def analyze_adaptive_segments(
    video_path: str,
    moments: List[ViralMoment],
    adaptive_manager: AdaptiveCropManager,
    cropper: SmartCropper,
    sample_rate: int = 2
) -> List[Tuple[float, FocusPoint]]:
    """
    Analyse les segments avec AdaptiveCropManager pour détecter le type de contenu
    et utiliser la stratégie de crop appropriée pour chaque segment.

    Args:
        video_path: Chemin vers la vidéo
        moments: Liste des moments viraux à analyser
        adaptive_manager: Gestionnaire de crop adaptatif
        cropper: SmartCropper pour le lissage des points de focus
        sample_rate: Nombre de frames par seconde à analyser

    Returns:
        Liste de (timestamp, FocusPoint) avec stratégie adaptée
    """
    if not moments:
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_ANALYSIS_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        # Analyser chaque segment et stocker son type
        segment_types: Dict[int, Tuple[ContentType, Any]] = {}
        for i, moment in enumerate(moments):
            content_type, analysis = adaptive_manager.analyze_and_select_strategy(
                video_path, moment.start_time, moment.end_time
            )
            segment_types[i] = (content_type, analysis)

        # Résumé des types détectés
        type_counts: Dict[str, int] = {}
        for _, (ct, _) in segment_types.items():
            type_counts[ct.name] = type_counts.get(ct.name, 0) + 1
        console.print(f"[dim]Types détectés: {type_counts}[/dim]")

        # Calculer les frames à analyser
        total_segment_duration = sum(m.end_time - m.start_time for m in moments)
        frame_interval = max(1, int(fps / sample_rate))
        frames_to_analyze = int(total_segment_duration * sample_rate)

        console.print(
            f"[dim]Analyse adaptative: {total_segment_duration:.0f}s"
            f" sur {len(moments)} segment(s)[/dim]"
        )

        focus_points: List[Tuple[float, FocusPoint]] = []
        prev_focus: Optional[FocusPoint] = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Analyse adaptative ({total_segment_duration:.0f}s)...",
                total=frames_to_analyze
            )

            for seg_idx, moment in enumerate(moments):
                # Sélectionner la stratégie pour ce segment
                content_type, _ = segment_types[seg_idx]
                adaptive_manager._current_strategy = adaptive_manager.strategies[content_type]

                start_frame = int(moment.start_time * fps)
                end_frame = int(moment.end_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                frame_count = start_frame

                while frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if (frame_count - start_frame) % frame_interval == 0:
                        timestamp = frame_count / fps
                        try:
                            # Utiliser la stratégie adaptative
                            focus_point = adaptive_manager.get_focus_point(frame, prev_focus)
                            prev_focus = focus_point
                        except Exception:
                            # Fallback en cas d'erreur
                            focus_point = FocusPoint(x=0.5, y=0.5, confidence=0.1)

                        focus_points.append((timestamp, focus_point))
                        progress.update(task, advance=1)

                    frame_count += 1
    finally:
        cap.release()

    # Appliquer le lissage temporel
    if len(focus_points) > 3:
        focus_points = cropper.smooth_focus_points(focus_points)
        console.print(f"[dim]Lissage temporel appliqué[/dim]")

    console.print(
        f"[green]Analyse adaptative terminée:"
        f" {len(focus_points)} points de focus[/green]"
    )

    return focus_points
