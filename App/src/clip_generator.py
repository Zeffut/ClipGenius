"""
Module de génération de clips viraux
Combine toutes les fonctionnalités pour créer des clips optimisés
"""

import os
import gc
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from moviepy import VideoFileClip
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from concurrent.futures import ThreadPoolExecutor, as_completed

from .viral_detector import ViralMoment, ViralMomentDetector
from .smart_cropper import (
    SmartCropper, FocusPoint, CropResult, create_blur_filled_frame,
    AdaptiveCropManager, ContentType
)
from .blur_fill import create_letterbox_frame
from .visual_effects import apply_color_grading, apply_sharpening
from .audio_sanitizer import sanitize_audio
from .clip_encoder import encode_single_clip, IS_MACOS
from .adaptive_analysis import analyze_adaptive_segments

# Nouveau système de tracking (remplace l'ancien)
try:
    from .focus_tracker import (
        FocusTracker, TrackingPreset, TrackingConfig, PRESET_CONFIGS,
        FocusPoint as NewFocusPoint, analyze_content_type
    )
    NEW_TRACKER_AVAILABLE = True
except ImportError:
    NEW_TRACKER_AVAILABLE = False

console = Console()

# Import des sous-titres (Whisper pour transcription, ASS pour rendering)
try:
    from .subtitles import TranscriptionResult, SubtitleGenerator
    from .tiktok_captions import generate_tiktok_ass
    SUBTITLES_AVAILABLE = True
except ImportError:
    SUBTITLES_AVAILABLE = False

# Import de l'optimiseur de hooks
try:
    from .hook_optimizer import HookOptimizer, optimize_clip_hooks
    HOOK_OPTIMIZER_AVAILABLE = True
except ImportError:
    HOOK_OPTIMIZER_AVAILABLE = False

# Constantes de configuration
DEFAULT_OUTPUT_WIDTH: int = 1080            # Largeur de sortie par défaut (pixels)
DEFAULT_OUTPUT_HEIGHT: int = 1920           # Hauteur de sortie par défaut (pixels, ratio 9:16)
DEFAULT_OUTPUT_FPS: int = 30               # FPS de sortie par défaut


@dataclass
class ClipConfig:
    """Configuration pour la génération de clips"""
    # Format de sortie
    output_width: int = DEFAULT_OUTPUT_WIDTH       # Largeur en pixels
    output_height: int = DEFAULT_OUTPUT_HEIGHT      # Hauteur en pixels (9:16)
    output_fps: int = DEFAULT_OUTPUT_FPS           # FPS de sortie (30 ou 60)

    # Détection des moments (nombre automatique basé sur la qualité)
    min_clip_duration: float = 60.0    # Durée minimum: 60s
    max_clip_duration: float = 90.0    # Durée maximum: 90s
    min_viral_score: float = 0.80      # Score minimum pour être viral
    max_clips: Optional[int] = None    # None = automatique

    # === MODE RAPIDE ===
    fast_mode: bool = False            # Désactivé pour meilleure qualité
    use_hardware_accel: bool = False   # Désactivé: VideoToolbox peut causer des saccades avec MoviePy
    parallel_processing: bool = False  # Générer plusieurs clips en parallèle (DÉSACTIVÉ : conflits Whisper Metal)
    max_parallel_clips: int = 3        # Nombre de clips à traiter en parallèle (2-4 recommandé)

    # Qualité d'encodage vidéo (HAUTE QUALITÉ par défaut)
    video_bitrate: str = "25M"         # Bitrate vidéo très élevé pour 1080x1920
    audio_bitrate: str = "192k"        # Bitrate audio AAC

    # Options d'encodage haute qualité
    crf: Optional[int] = 15            # Constant Rate Factor (0=lossless, 15=excellent, 18=très bon, 22=bon)
                                        # 15 = qualité quasi parfaite, fichiers plus gros
                                        # None = utiliser video_bitrate à la place
    preset: str = "slow"               # Preset FFmpeg: ultrafast, fast, medium, slow, veryslow
                                        # slow = meilleure qualité (plus lent)
    video_profile: str = "high"        # Profil H.264: baseline, main, high
    video_level: str = "4.2"           # Niveau H.264 pour compatibilité (4.2 = 1080p60)

    # Style (anciennes options - conservées pour compatibilité)
    add_subtitles: bool = True
    subtitle_font: str = "Arial-Bold"
    subtitle_fontsize: int = 60
    subtitle_color: str = "white"
    subtitle_stroke_color: str = "black"
    subtitle_stroke_width: int = 3

    # Sous-titres animés (pycaps) - nouvelles options
    subtitle_enriched: bool = True              # Utiliser mots-clés colorés
    subtitle_theme: str = "viral"               # Thème de base
    subtitle_max_words: int = 3                 # Mots par segment
    subtitle_use_emojis: bool = True            # Ajouter émojis
    subtitle_custom_colors: Optional[dict] = None  # Couleurs/styles du preset

    # Zoom effect pour plus de dynamisme (DÉSACTIVÉ: peut causer des saccades)
    enable_zoom_effect: bool = False
    zoom_factor: float = 1.05
    zoom_style: str = "ease_out"       # "ease_out", "ease_in_out", "breathing", "pulse"

    # Blur fill pour les visages trop bas
    enable_blur_fill: bool = True
    blur_strength: int = 51  # Force du flou (doit être impair)
    # Seuil de confiance en dessous duquel on utilise le letterbox (fond flouté symétrique)
    letterbox_confidence_threshold: float = 0.25

    # Recadrage intelligent (désactiver pour accélérer)
    smart_crop: bool = True  # False = crop centré simple
    use_adaptive_crop: bool = True  # True = utilise AdaptiveCropManager (détecte le type de contenu)
    use_new_tracker: bool = True  # True = utilise le nouveau FocusTracker (plus fluide)
    tracking_preset: str = "auto"  # Preset de tracking: auto, podcast, vlog, gaming, etc.

    # Qualité de redimensionnement
    use_lanczos: bool = True           # Utiliser LANCZOS4 pour meilleure qualité (plus lent)

    # Effets cinématiques avancés
    enable_color_grading: bool = True   # Correction colorimétrique cinématique
    color_grading_style: str = "warm"   # "warm", "cool", "vibrant", "cinematic", "none"
    enable_sharpening: bool = True      # Netteté améliorée
    sharpening_strength: float = 0.3    # Force du sharpening (0-1)
    enable_vignette: bool = False       # Vignette sur tout le clip
    vignette_strength: float = 0.15     # Force de la vignette (0-1)

    # Ken Burns effect (mouvement subtil)
    enable_ken_burns: bool = False      # Mouvement panoramique subtil
    ken_burns_intensity: float = 0.02   # Intensité du mouvement (0-0.1)


class ClipGenerator:
    """
    Génère des clips viraux à partir d'une vidéo source
    """

    def __init__(
        self,
        config: Optional[ClipConfig] = None,
        transcription_result: Optional['TranscriptionResult'] = None
    ):
        """
        Initialise le générateur de clips.

        Args:
            config: Configuration du générateur
            transcription_result: Résultat de transcription pré-calculée (optionnel)
        """
        self.config = config or ClipConfig()
        self.cropper = SmartCropper(
            target_aspect_ratio=self.config.output_width / self.config.output_height
        )
        self.detector = ViralMomentDetector(
            min_clip_duration=self.config.min_clip_duration,
            max_clip_duration=self.config.max_clip_duration,
            min_viral_score=self.config.min_viral_score,
            max_clips=self.config.max_clips
        )
        # Gestionnaire de crop adaptatif (sélection auto de stratégie par segment)
        self.adaptive_manager: Optional[AdaptiveCropManager] = None
        if self.config.use_adaptive_crop and self.config.smart_crop and not self.config.use_new_tracker:
            self.adaptive_manager = AdaptiveCropManager()

        # Nouveau système de tracking (remplace l'ancien si disponible)
        self.focus_tracker: Optional['FocusTracker'] = None
        if self.config.use_new_tracker and self.config.smart_crop and NEW_TRACKER_AVAILABLE:
            # Convertir le preset string en enum
            preset_map = {
                'auto': TrackingPreset.AUTO,
                'podcast': TrackingPreset.PODCAST,
                'interview': TrackingPreset.INTERVIEW,
                'vlog': TrackingPreset.VLOG,
                'gaming': TrackingPreset.GAMING,
                'tutorial': TrackingPreset.TUTORIAL,
                'action': TrackingPreset.ACTION,
                'presentation': TrackingPreset.PRESENTATION,
            }
            preset = preset_map.get(self.config.tracking_preset.lower(), TrackingPreset.AUTO)
            self.focus_tracker = FocusTracker(preset=preset)
            console.print(f"[green]Nouveau tracker activé (preset: {preset.value})[/green]")

        # Cache pour la transcription globale (évite de re-transcrire pour chaque clip)
        self.transcription_result: Optional['TranscriptionResult'] = transcription_result

    def generate_clips(
        self,
        video_path: str,
        output_dir: str = "output",
        moments: Optional[List[ViralMoment]] = None,
        start_index: int = 1
    ) -> List[str]:
        """
        Génère des clips viraux à partir d'une vidéo

        Args:
            video_path: Chemin vers la vidéo source
            output_dir: Dossier de sortie
            moments: Moments viraux (détectés automatiquement si non fournis)
            start_index: Index de départ pour la numérotation des clips

        Returns:
            Liste des chemins vers les clips générés
        """
        video_path_obj = Path(video_path)
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold cyan]ClipGenius - Génération de clips viraux[/bold cyan]")
        console.print(f"Source: {video_path_obj.name}")

        # Pré-traiter l'audio pour éviter les erreurs AAC
        sanitized_video_path = None
        actual_video_path = str(video_path_obj)

        try:
            # Créer un fichier temporaire pour la vidéo nettoyée
            sanitized_video_path = str(output_dir_obj / f".sanitized_{video_path_obj.name}")
            console.print("[dim]Nettoyage de l'audio...[/dim]")

            if sanitize_audio(str(video_path_obj), sanitized_video_path):
                actual_video_path = sanitized_video_path
                console.print("[green]Audio nettoyé avec succès[/green]")
            else:
                console.print("[dim]Utilisation de l'audio original[/dim]")
                sanitized_video_path = None
        except Exception as e:
            console.print(f"[dim]Pré-traitement audio ignoré: {e}[/dim]")
            sanitized_video_path = None

        # Charger la vidéo (nettoyée ou originale)
        video = None
        try:
            video = VideoFileClip(actual_video_path)

            # Détecter les moments viraux si nécessaire
            if moments is None:
                moments = self.detector.analyze(str(video_path_obj))

            if not moments:
                console.print("[yellow]Aucun moment suffisamment viral détecté.[/yellow]")
                return []

            # Sécurité: limiter au max_clips configuré
            if self.config.max_clips is not None and len(moments) > self.config.max_clips:
                moments = sorted(moments, key=lambda m: m.score, reverse=True)[:self.config.max_clips]
                console.print(f"[dim]Limité à {self.config.max_clips} meilleurs clips[/dim]")

            # Analyser les points de focus si smart_crop activé
            focus_points = self._analyze_focus_points(str(video_path_obj), moments)

            # Générer chaque clip (parallèle ou séquentiel)
            generated_clips = self._generate_all_clips(
                video, moments, focus_points, actual_video_path,
                video_path_obj, output_dir_obj, start_index
            )

        finally:
            self._cleanup_resources(video, sanitized_video_path)

        console.print(f"\n[bold green]{len(generated_clips)} clips générés avec succès![/bold green]")
        return generated_clips

    def _analyze_focus_points(
        self,
        video_path: str,
        moments: List[ViralMoment]
    ) -> List[Tuple[float, FocusPoint]]:
        """
        Analyse les points de focus selon la stratégie configurée.

        Args:
            video_path: Chemin vers la vidéo source
            moments: Moments viraux à analyser

        Returns:
            Liste de (timestamp, FocusPoint)
        """
        if not self.config.smart_crop:
            console.print(f"[dim]Mode rapide: crop centré (--no-smart-crop)[/dim]")
            return []

        # En mode rapide: sample_rate plus bas (1 fps au lieu de 2)
        sample_rate = 1 if self.config.fast_mode else 2

        # === NOUVEAU TRACKER (plus fluide) ===
        if self.focus_tracker is not None:
            return self._analyze_with_new_tracker(video_path, moments)

        # === ANCIEN SYSTÈME ADAPTATIF (fallback) ===
        if self.adaptive_manager is not None:
            console.print("[cyan]Analyse adaptative du contenu (ancien système)...[/cyan]")
            return analyze_adaptive_segments(
                video_path, moments, self.adaptive_manager,
                self.cropper, sample_rate=sample_rate
            )

        # Mode classique: SmartCropper avec détection de visage
        segments = [(m.start_time, m.end_time) for m in moments]
        return self.cropper.analyze_video_segments(
            video_path, segments, sample_rate=sample_rate
        )

    def _analyze_with_new_tracker(
        self,
        video_path: str,
        moments: List[ViralMoment]
    ) -> List[Tuple[float, FocusPoint]]:
        """
        Analyse avec le nouveau FocusTracker (plus fluide).

        Args:
            video_path: Chemin vers la vidéo source
            moments: Moments viraux à analyser

        Returns:
            Liste de (timestamp, FocusPoint)
        """
        console.print("[cyan]Analyse du contenu (nouveau tracker)...[/cyan]")
        segments = [(m.start_time, m.end_time) for m in moments]

        # Auto-détection du preset si "auto"
        if self.config.tracking_preset.lower() == 'auto':
            detected_preset, analysis = analyze_content_type(video_path)
            old_tracker = self.focus_tracker
            self.focus_tracker = FocusTracker(preset=detected_preset)
            try:
                old_tracker.close()
            except Exception:
                pass
            console.print(
                f"[dim]Type détecté: {detected_preset.value}"
                f" (faces={analysis.get('has_face', False)},"
                f" motion={analysis.get('motion_level', 0):.2f})[/dim]"
            )

        # Analyser la vidéo avec le nouveau tracker
        new_focus_points = self.focus_tracker.analyze_video(
            video_path,
            segments=segments,
            progress_callback=lambda p, m: console.print(f"[dim]{m}[/dim]") if p % 25 == 0 else None
        )

        # Convertir en format legacy (timestamp, FocusPoint)
        focus_points = [
            (fp.timestamp, FocusPoint(x=fp.x, y=fp.y, confidence=fp.confidence))
            for fp in new_focus_points
        ]
        console.print(f"[green]{len(focus_points)} points de focus analysés[/green]")
        return focus_points

    def _generate_all_clips(
        self,
        video: VideoFileClip,
        moments: List[ViralMoment],
        focus_points: List[Tuple[float, FocusPoint]],
        actual_video_path: str,
        video_path_obj: Path,
        output_dir_obj: Path,
        start_index: int
    ) -> List[str]:
        """
        Génère tous les clips (parallèle ou séquentiel).

        Args:
            video: Vidéo source ouverte
            moments: Moments viraux
            focus_points: Points de focus analysés
            actual_video_path: Chemin effectif de la vidéo (sanitisée ou originale)
            video_path_obj: Path de la vidéo source
            output_dir_obj: Path du dossier de sortie
            start_index: Index de départ pour la numérotation

        Returns:
            Liste des chemins des clips générés
        """
        generated_clips: List[str] = []

        if self.config.parallel_processing and len(moments) > 1:
            return self._generate_clips_parallel(
                moments, focus_points, actual_video_path,
                video_path_obj, output_dir_obj, start_index
            )

        return self._generate_clips_sequential(
            video, moments, focus_points,
            video_path_obj, output_dir_obj, start_index
        )

    def _generate_clips_parallel(
        self,
        moments: List[ViralMoment],
        focus_points: List[Tuple[float, FocusPoint]],
        actual_video_path: str,
        video_path_obj: Path,
        output_dir_obj: Path,
        start_index: int
    ) -> List[str]:
        """
        Génère les clips en parallèle (2-4x plus rapide).
        """
        generated_clips: List[str] = []

        console.print(
            f"[cyan]Génération parallèle de {len(moments)} clips"
            f" ({self.config.max_parallel_clips} en parallèle)...[/cyan]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Génération des clips...", total=len(moments))

            def generate_clip_task(args):
                i, moment = args
                clip_name = f"{video_path_obj.stem}_clip_{i:02d}.mp4"
                clip_path = output_dir_obj / clip_name

                video_thread = None
                try:
                    video_thread = VideoFileClip(actual_video_path)

                    for attempt in range(2):
                        try:
                            encode_single_clip(
                                video=video_thread,
                                moment=moment,
                                focus_points=focus_points,
                                output_path=str(clip_path),
                                clip_number=i,
                                config=self.config,
                                process_frame_func=self._process_frame
                            )
                            return (i, str(clip_path), True, None)
                        except Exception as e:
                            if attempt == 0:
                                time.sleep(1)
                                gc.collect()
                            else:
                                return (i, str(clip_path), False, str(e))

                    return (i, str(clip_path), False, "Erreur inconnue")
                except Exception as e:
                    return (i, str(clip_path), False, f"Erreur ouverture vidéo: {e}")
                finally:
                    if video_thread:
                        try:
                            video_thread.close()
                        except Exception:
                            pass
                    gc.collect()

            tasks = [(i, moment) for i, moment in enumerate(moments, start_index)]

            with ThreadPoolExecutor(max_workers=self.config.max_parallel_clips) as executor:
                futures = {executor.submit(generate_clip_task, t): t for t in tasks}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is None:
                            continue

                        i, clip_path, success, error = result

                        if success:
                            generated_clips.append(clip_path)
                            moment = [m for idx, m in enumerate(moments, start_index) if idx == i][0]
                            console.print(
                                f"  [green]OK[/green] Clip {i}:"
                                f" {moment.start_time:.1f}s - {moment.end_time:.1f}s"
                            )
                        else:
                            console.print(f"  [red]X[/red] Erreur clip {i}: {error}")
                    except Exception as e:
                        console.print(f"  [red]X[/red] Erreur future: {e}")

                    progress.update(task, advance=1)

            generated_clips.sort()

        return generated_clips

    def _generate_clips_sequential(
        self,
        video: VideoFileClip,
        moments: List[ViralMoment],
        focus_points: List[Tuple[float, FocusPoint]],
        video_path_obj: Path,
        output_dir_obj: Path,
        start_index: int
    ) -> List[str]:
        """
        Génère les clips séquentiellement (compatible, fallback).
        """
        generated_clips: List[str] = []

        console.print(f"[dim]Génération séquentielle de {len(moments)} clips...[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Génération des clips...", total=len(moments))

            max_retries = 3
            for i, moment in enumerate(moments, start_index):
                clip_name = f"{video_path_obj.stem}_clip_{i:02d}.mp4"
                clip_path = output_dir_obj / clip_name

                gc.collect()

                for attempt in range(max_retries):
                    try:
                        encode_single_clip(
                            video=video,
                            moment=moment,
                            focus_points=focus_points,
                            output_path=str(clip_path),
                            clip_number=i,
                            config=self.config,
                            process_frame_func=self._process_frame
                        )
                        generated_clips.append(str(clip_path))
                        console.print(
                            f"  [green]OK[/green] Clip {i}:"
                            f" {moment.start_time:.1f}s - {moment.end_time:.1f}s"
                        )
                        break
                    except Exception as e:
                        error_msg = str(e)
                        gc.collect()

                        if (attempt < max_retries - 1
                                and ("stdout" in error_msg
                                     or "NoneType" in error_msg
                                     or "Proc" in error_msg)):
                            wait_time = (attempt + 1) * 2
                            console.print(
                                f"  [yellow]![/yellow] Clip {i}:"
                                f" Retry ({attempt + 1}/{max_retries})"
                                f" dans {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            console.print(f"  [red]X[/red] Erreur clip {i}: {e}")
                            break

                progress.update(task, advance=1)

        return generated_clips

    def _cleanup_resources(
        self,
        video: Optional[VideoFileClip],
        sanitized_video_path: Optional[str]
    ) -> None:
        """
        Nettoie toutes les ressources (vidéo, cropper, tracker, cache, fichiers temp).
        """
        if video is not None:
            try:
                video.close()
            except Exception:
                pass
        try:
            self.cropper.close()
        except Exception:
            pass
        if self.adaptive_manager is not None:
            try:
                self.adaptive_manager.close()
            except Exception:
                pass
        if self.focus_tracker is not None:
            try:
                self.focus_tracker.close()
            except Exception:
                pass

        gc.collect()

        # Nettoyer le fichier audio sanitisé temporaire
        if sanitized_video_path and os.path.exists(sanitized_video_path):
            try:
                os.remove(sanitized_video_path)
            except Exception:
                pass

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        focus_points: List[Tuple[float, FocusPoint]]
    ) -> np.ndarray:
        """
        Traite une frame: recadrage intelligent + effets visuels cinématiques.

        Pipeline de traitement:
        1. Recadrage intelligent centré sur le sujet
        2. Blur fill si nécessaire
        3. Color grading cinématique (désactivé en fast_mode)
        4. Sharpening intelligent (désactivé en fast_mode)

        Cache: Les frames sont mises en cache pour éviter de les redécoder
        """
        if frame is None or frame.size == 0:
            return np.zeros((self.config.output_height, self.config.output_width, 3), dtype=np.uint8)

        h, w = frame.shape[:2]

        # Obtenir le point de focus interpolé
        if self.focus_tracker is not None and focus_points:
            # Nouveau système: interpolation intégrée
            x, y, confidence = self.focus_tracker.get_focus_at(
                timestamp,
                [NewFocusPoint(x=fp.x, y=fp.y, confidence=fp.confidence, timestamp=t)
                 for t, fp in focus_points]
            )
            focus = FocusPoint(x=x, y=y, confidence=confidence)
        else:
            # Ancien système
            focus = self.cropper.get_interpolated_focus(focus_points, timestamp)

        # Pas de visage détecté → letterbox avec fond flouté symétrique haut/bas
        if (self.config.enable_blur_fill
                and focus.confidence < self.config.letterbox_confidence_threshold):
            return create_letterbox_frame(
                frame,
                target_width=self.config.output_width,
                target_height=self.config.output_height,
                blur_strength=self.config.blur_strength,
            )

        # Calculer la région de recadrage avec détection du besoin de blur fill
        crop_result = self.cropper.calculate_crop_region_extended(
            focus, w, h,
            enable_blur_fill=self.config.enable_blur_fill
        )

        # Si le visage est trop bas et qu'on a besoin d'un fond flouté
        if crop_result.needs_blur_fill and self.config.enable_blur_fill:
            crop_width = crop_result.x2 - crop_result.x1
            crop_height = crop_result.y2 - crop_result.y1

            result = create_blur_filled_frame(
                frame,
                crop_result,
                target_width=crop_width,
                target_height=crop_height,
                blur_strength=self.config.blur_strength
            )

            if result is None or result.size == 0:
                return np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
        else:
            # Recadrage normal
            x1, y1, x2, y2 = crop_result.x1, crop_result.y1, crop_result.x2, crop_result.y2

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, w)
            y2 = min(y2, h)

            if x2 <= x1 or y2 <= y1:
                crop_width = crop_result.x2 - crop_result.x1
                crop_height = crop_result.y2 - crop_result.y1
                return np.zeros((max(1, crop_height), max(1, crop_width), 3), dtype=np.uint8)

            result = frame[y1:y2, x1:x2]

            if result.size == 0:
                crop_width = crop_result.x2 - crop_result.x1
                crop_height = crop_result.y2 - crop_result.y1
                return np.zeros((max(1, crop_height), max(1, crop_width), 3), dtype=np.uint8)

        # Appliquer les effets visuels cinématiques (sauf en mode rapide)
        if not self.config.fast_mode:
            if self.config.enable_color_grading:
                result = apply_color_grading(result, self.config.color_grading_style)

            if self.config.enable_sharpening:
                result = apply_sharpening(result, self.config.sharpening_strength)

        return result


def generate_viral_clips(
    video_path: str,
    output_dir: str = "output",
    min_duration: float = 60.0,
    max_duration: float = 90.0,
    min_viral_score: float = 0.80,
    max_clips: Optional[int] = None,
    add_subtitles: bool = True
) -> List[str]:
    """
    Fonction utilitaire pour générer des clips viraux

    Le nombre de clips est déterminé automatiquement en fonction
    de la qualité des moments détectés.

    Args:
        video_path: Chemin vers la vidéo source
        output_dir: Dossier de sortie
        min_duration: Durée minimum des clips (défaut: 60s)
        max_duration: Durée maximum des clips (défaut: 90s)
        min_viral_score: Score minimum pour qu'un moment soit viral (0-1)
        max_clips: Nombre max de clips (None = automatique)
        add_subtitles: Ajouter des sous-titres automatiques

    Returns:
        Liste des chemins vers les clips générés
    """
    config = ClipConfig(
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
        min_viral_score=min_viral_score,
        max_clips=max_clips,
        add_subtitles=add_subtitles
    )

    generator = ClipGenerator(config)
    return generator.generate_clips(video_path, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        clips = generate_viral_clips(sys.argv[1])
        print(f"Clips générés: {clips}")
