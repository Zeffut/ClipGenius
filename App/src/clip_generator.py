"""
Module de génération de clips viraux
Combine toutes les fonctionnalités pour créer des clips optimisés
"""

import os
import gc
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from moviepy import (
    VideoFileClip, 
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .viral_detector import ViralMoment, ViralMomentDetector
from .smart_cropper import SmartCropper, FocusPoint, CropResult, create_blur_filled_frame

import subprocess
import tempfile
import shutil

console = Console()

# Import des sous-titres (Whisper pour transcription, ASS pour rendering)
try:
    from .subtitles import TranscriptionResult, SubtitleGenerator
    from .tiktok_captions import generate_tiktok_ass
    SUBTITLES_AVAILABLE = True
except ImportError:
    SUBTITLES_AVAILABLE = False

# Cache global pour les frames décodées (économise du temps de décodage)
_frame_cache = {}
_cache_lock = threading.Lock()


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
            timeout=300  # 5 minutes max
        )

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    except Exception as e:
        console.print(f"[yellow]Avertissement: Impossible de nettoyer l'audio: {e}[/yellow]")
        return False


import platform

# Détecter si on est sur Mac pour l'accélération matérielle VideoToolbox
IS_MACOS = platform.system() == "Darwin"


@dataclass
class ClipConfig:
    """Configuration pour la génération de clips"""
    # Format de sortie
    output_width: int = 1080       # Largeur en pixels
    output_height: int = 1920      # Hauteur en pixels (9:16)
    output_fps: int = 30           # FPS de sortie (30 ou 60)

    # Détection des moments (nombre automatique basé sur la qualité)
    min_clip_duration: float = 60.0    # Durée minimum: 60s
    max_clip_duration: float = 90.0    # Durée maximum: 90s
    min_viral_score: float = 0.80      # Score minimum pour être viral
    max_clips: Optional[int] = None    # None = automatique

    # === MODE RAPIDE ===
    fast_mode: bool = True             # Active les optimisations de vitesse
    use_hardware_accel: bool = True    # Utiliser GPU (VideoToolbox Mac, NVENC Nvidia)
    parallel_processing: bool = False  # Générer plusieurs clips en parallèle (DÉSACTIVÉ : conflits Whisper Metal)
    max_parallel_clips: int = 3        # Nombre de clips à traiter en parallèle (2-4 recommandé)

    # Qualité d'encodage vidéo
    video_bitrate: str = "8M"          # Bitrate vidéo (utilisé si crf est None)
    audio_bitrate: str = "192k"        # Bitrate audio AAC

    # Options d'encodage haute qualité
    crf: Optional[int] = 22            # Constant Rate Factor (0=lossless, 18=parfait, 22=excellent, 28=moyen)
                                        # None = utiliser video_bitrate à la place
    preset: str = "ultrafast"          # Preset FFmpeg: ultrafast, fast, medium, slow, veryslow
                                        # Plus lent = meilleure qualité/compression
    video_profile: str = "high"        # Profil H.264: baseline, main, high
    video_level: str = "4.1"           # Niveau H.264 pour compatibilité
    
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
    subtitle_custom_colors: Optional[Dict[str, Any]] = None  # Couleurs/styles du preset
    
    # Zoom effect pour plus de dynamisme
    enable_zoom_effect: bool = True
    zoom_factor: float = 1.05
    zoom_style: str = "ease_out"       # "ease_out", "ease_in_out", "breathing", "pulse"
    
    # Blur fill pour les visages trop bas
    enable_blur_fill: bool = True
    blur_strength: int = 51  # Force du flou (doit être impair)
    
    # Recadrage intelligent (désactiver pour accélérer)
    smart_crop: bool = True  # False = crop centré simple
    
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
        video = VideoFileClip(actual_video_path)
        
        # Détecter les moments viraux si nécessaire
        if moments is None:
            moments = self.detector.analyze(str(video_path_obj))
        
        if not moments:
            console.print("[yellow]Aucun moment suffisamment viral détecté.[/yellow]")
            video.close()
            self.cropper.close()
            return []
        
        # Analyser les points de focus si smart_crop activé
        if self.config.smart_crop:
            # Analyse MediaPipe des segments
            segments = [(m.start_time, m.end_time) for m in moments]
            # En mode rapide: sample_rate plus bas (1 fps au lieu de 2)
            sample_rate = 1 if self.config.fast_mode else 2
            focus_points = self.cropper.analyze_video_segments(
                str(video_path_obj), segments, sample_rate=sample_rate
            )
        else:
            # Mode rapide: crop centré, pas d'analyse MediaPipe
            console.print(f"[dim]Mode rapide: crop centré (--no-smart-crop)[/dim]")
            focus_points = []
        
        # Note: La transcription est maintenant faite PAR CLIP avec 'turbo' (plus rapide et plus précis)
        # Au lieu de transcrire toute la vidéo avec 'base'
        
        # Générer chaque clip (parallèle ou séquentiel)
        generated_clips: List[str] = []
        
        if self.config.parallel_processing and len(moments) > 1:
            # === MODE PARALLÈLE (2-4x plus rapide) ===
            console.print(f"[cyan]Génération parallèle de {len(moments)} clips ({self.config.max_parallel_clips} en parallèle)...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Génération des clips...", total=len(moments))
                
                # Fonction pour générer un clip (thread-safe)
                def generate_clip_task(args):
                    i, moment = args
                    clip_name = f"{video_path_obj.stem}_clip_{i:02d}.mp4"
                    clip_path = output_dir_obj / clip_name
                    
                    # Chaque thread doit avoir sa propre instance VideoFileClip
                    video_thread = None
                    try:
                        video_thread = VideoFileClip(actual_video_path)
                        
                        # Essayer jusqu'à 2 fois en mode parallèle
                        for attempt in range(2):
                            try:
                                self._generate_single_clip(
                                    video=video_thread,
                                    moment=moment,
                                    focus_points=focus_points,
                                    output_path=str(clip_path),
                                    clip_number=i
                                )
                                return (i, str(clip_path), True, None)
                            except Exception as e:
                                if attempt == 0:
                                    time.sleep(1)
                                    gc.collect()
                                else:
                                    return (i, str(clip_path), False, str(e))
                        
                        # Si on arrive ici, c'est qu'il y a eu un problème
                        return (i, str(clip_path), False, "Erreur inconnue")
                    except Exception as e:
                        # Erreur lors de l'ouverture de la vidéo
                        return (i, str(clip_path), False, f"Erreur ouverture vidéo: {e}")
                    finally:
                        if video_thread:
                            try:
                                video_thread.close()
                            except Exception:
                                pass
                        gc.collect()
                
                # Créer les tâches
                tasks = [(i, moment) for i, moment in enumerate(moments, start_index)]
                
                # Exécuter en parallèle avec ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.config.max_parallel_clips) as executor:
                    futures = {executor.submit(generate_clip_task, task): task for task in tasks}
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is None:
                                continue
                            
                            i, clip_path, success, error = result
                            
                            if success:
                                generated_clips.append(clip_path)
                                moment = [m for idx, m in enumerate(moments, start_index) if idx == i][0]
                                console.print(f"  [green]OK[/green] Clip {i}: {moment.start_time:.1f}s - {moment.end_time:.1f}s")
                            else:
                                console.print(f"  [red]X[/red] Erreur clip {i}: {error}")
                        except Exception as e:
                            console.print(f"  [red]X[/red] Erreur future: {e}")
                        
                        progress.update(task, advance=1)
                
                # Trier par ordre de génération
                generated_clips.sort()
        
        else:
            # === MODE SÉQUENTIEL (compatible, fallback) ===
            console.print(f"[dim]Génération séquentielle de {len(moments)} clips...[/dim]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Génération des clips...", total=len(moments))
                
                for i, moment in enumerate(moments, start_index):
                    clip_name = f"{video_path_obj.stem}_clip_{i:02d}.mp4"
                    clip_path = output_dir_obj / clip_name
                    
                    # Libérer la mémoire avant chaque clip pour éviter les problèmes FFmpeg
                    gc.collect()
                    
                    # Essayer jusqu'à 3 fois en cas d'erreur FFmpeg
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self._generate_single_clip(
                                video=video,
                                moment=moment,
                                focus_points=focus_points,
                                output_path=str(clip_path),
                                clip_number=i
                            )
                            generated_clips.append(str(clip_path))
                            console.print(f"  [green]OK[/green] Clip {i}: {moment.start_time:.1f}s - {moment.end_time:.1f}s")
                            break  # Succès, sortir de la boucle de retry
                        except Exception as e:
                            error_msg = str(e)
                            # Libérer les ressources avant de réessayer
                            gc.collect()
                            
                            if attempt < max_retries - 1 and ("stdout" in error_msg or "NoneType" in error_msg or "Proc" in error_msg):
                                # Erreur FFmpeg, réessayer avec délai croissant
                                wait_time = (attempt + 1) * 2  # 2s, 4s
                                console.print(f"  [yellow]![/yellow] Clip {i}: Retry ({attempt + 1}/{max_retries}) dans {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                console.print(f"  [red]X[/red] Erreur clip {i}: {e}")
                                break
                    
                    progress.update(task, advance=1)
        
        video.close()
        self.cropper.close()
        
        # Nettoyer le cache de frames
        global _frame_cache
        with _cache_lock:
            _frame_cache.clear()
        gc.collect()

        # Nettoyer le fichier audio sanitisé temporaire
        if sanitized_video_path and os.path.exists(sanitized_video_path):
            try:
                os.remove(sanitized_video_path)
            except Exception:
                pass

        console.print(f"\n[bold green]{len(generated_clips)} clips générés avec succès![/bold green]")
        return generated_clips
    
    def _generate_single_clip(
        self,
        video: VideoFileClip,
        moment: ViralMoment,
        focus_points: List[Tuple[float, FocusPoint]],
        output_path: str,
        clip_number: int
    ):
        """
        Génère un seul clip avec gestion robuste des ressources.
        
        Pipeline complet:
        1. Extraction du segment
        2. Recadrage intelligent + effets frame-by-frame
        3. Resize à la taille finale
        4. Effet de zoom dynamique
        5. Vignette (optionnel)
        6. Encodage haute qualité
        """
        subclip = None
        processed_clip = None
        
        try:
            # Extraire le segment
            subclip = video.with_subclip(moment.start_time, moment.end_time)
            
            if subclip.duration is None or subclip.duration <= 0:
                raise ValueError(f"Duree du subclip invalide: {subclip.duration}")
            
            # Appliquer le recadrage intelligent + effets frame par frame
            def process_frame_func(get_frame, t):
                frame = get_frame(t)
                return self._process_frame(
                    frame, 
                    t + moment.start_time, 
                    focus_points
                )
            
            processed_clip = subclip.transform(process_frame_func)
            
            # Redimensionner à la taille finale
            processed_clip = processed_clip.resized(
                (self.config.output_width, self.config.output_height)
            )
            
            # Appliquer l'effet de zoom si activé (sauf en mode rapide)
            if self.config.enable_zoom_effect and not self.config.fast_mode:
                processed_clip = self._apply_zoom_effect(processed_clip)

            # Appliquer l'effet Ken Burns si activé (après zoom, avant vignette)
            if self.config.enable_ken_burns and not self.config.fast_mode:
                processed_clip = self._apply_ken_burns_effect(processed_clip)
            
            # Appliquer la vignette si activée (après resize pour meilleures perfs, sauf fast mode)
            if self.config.enable_vignette and not self.config.fast_mode:
                def vignette_effect(get_frame, t):
                    frame = get_frame(t)
                    return self._apply_vignette(frame)
                processed_clip = processed_clip.transform(vignette_effect)
            
            # Déterminer le codec et les paramètres selon le mode
            use_hw = self.config.use_hardware_accel and IS_MACOS

            if use_hw:
                # === ENCODAGE GPU (VideoToolbox sur Mac) ===
                # Beaucoup plus rapide que libx264
                codec = 'h264_videotoolbox'
                ffmpeg_params = [
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-allow_sw', '1',  # Fallback software si GPU occupé
                    '-realtime', '0',  # Pas de limite temps réel
                    '-max_muxing_queue_size', '9999',  # Éviter les overflow de queue
                ]
                # VideoToolbox utilise bitrate au lieu de CRF
                # q:v 65 ≈ CRF 22 en qualité
                ffmpeg_params.extend(['-q:v', '65'])
                console.print("[green]Encodage GPU (VideoToolbox) activé[/green]")
            else:
                # === ENCODAGE CPU (libx264) ===
                codec = 'libx264'
                ffmpeg_params = [
                    '-profile:v', self.config.video_profile,
                    '-level', self.config.video_level,
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-preset', self.config.preset,
                    '-max_muxing_queue_size', '9999',  # Éviter les overflow de queue
                ]

                # Optimisations FFmpeg pour presets rapides (fast, medium, etc.)
                if self.config.preset in ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium']:
                    ffmpeg_params.extend([
                        '-tune', 'fastdecode',  # Optimise pour décodage rapide (moins de refs)
                    ])
                # Ajouter des paramètres d'optimisation pour presets lents
                elif self.config.preset in ['slow', 'slower', 'veryslow']:
                    ffmpeg_params.extend([
                        '-tune', 'film',
                        '-x264opts', 'rc-lookahead=60:ref=6',
                    ])

                # Utiliser CRF
                if self.config.crf is not None:
                    ffmpeg_params.extend(['-crf', str(self.config.crf)])

            # Nombre de threads (auto = tous les cores)
            import os
            num_threads = os.cpu_count() or 8

            # Paramètres de qualité vidéo
            write_params = {
                'fps': self.config.output_fps,
                'codec': codec,
                'audio_codec': 'aac',
                'audio_bitrate': self.config.audio_bitrate,
                'logger': None,
                'threads': num_threads,
                'ffmpeg_params': ffmpeg_params,
            }

            # Bitrate pour mode non-CRF
            if not use_hw and self.config.crf is None:
                write_params['bitrate'] = self.config.video_bitrate

            # =====================================================
            # OPTIMISATION SINGLE-PASS: Sous-titres intégrés AVANT encodage
            # Évite le double encodage (économise ~50% du temps d'encodage)
            # =====================================================
            final_clip_to_encode = processed_clip
            subtitle_clips = []
            temp_audio_file = None
            ass_file = None
            
            if self.config.add_subtitles:
                console.print(f"[cyan]Préparation sous-titres TikTok (single-pass)...[/cyan]")
                try:
                    from .tiktok_captions import generate_tiktok_ass
                    from .subtitles import SubtitleGenerator
                    from .ass_to_moviepy import render_ass_to_moviepy
                    
                    # 1. Extraire l'audio du subclip vers un fichier temporaire
                    temp_audio_file = output_path.replace('.mp4', '_temp_audio.wav')
                    try:
                        console.print(f"[dim]Extraction audio pour transcription...[/dim]")
                        subclip.audio.write_audiofile(
                            temp_audio_file, 
                            fps=16000,  # 16kHz optimal pour Whisper
                            nbytes=2,   # 16-bit
                            logger=None
                        )
                    except Exception as e:
                        console.print(f"[yellow]⚠ Erreur extraction audio: {e}[/yellow]")
                        temp_audio_file = None
                    
                    # 2. Transcrire l'audio avec Whisper turbo
                    clip_words = None
                    if temp_audio_file and os.path.exists(temp_audio_file):
                        console.print(f"[cyan]Transcription du clip {clip_number} avec Whisper turbo...[/cyan]")
                        try:
                            subtitle_gen = SubtitleGenerator(model_size="turbo", language=None)
                            clip_transcription = subtitle_gen.transcribe_with_words(temp_audio_file)
                            
                            if clip_transcription and clip_transcription.words:
                                clip_words = clip_transcription.words
                                console.print(f"[green]✓ Clip transcrit: {len(clip_words)} mots[/green]")
                            else:
                                console.print(f"[yellow]⚠ Aucun mot détecté dans le clip[/yellow]")
                        except Exception as e:
                            console.print(f"[yellow]⚠ Erreur transcription clip: {e}[/yellow]")
                    
                    # 3. Générer le fichier ASS et les TextClips
                    if clip_words:
                        ass_file = output_path.replace('.mp4', '.ass')
                        generate_tiktok_ass(
                            words=clip_words,
                            preset_colors=self.config.subtitle_custom_colors,
                            output_path=ass_file,
                            style_config={
                                'theme': self.config.subtitle_theme,
                                'max_words': self.config.subtitle_max_words,
                                'use_emojis': self.config.subtitle_use_emojis
                            }
                        )
                        
                        # 4. Convertir ASS en TextClips MoviePy
                        subtitle_clips = render_ass_to_moviepy(
                            ass_file,
                            video_width=self.config.output_width,
                            video_height=self.config.output_height
                        )
                        
                        if subtitle_clips:
                            # 5. Composite: processed_clip + sous-titres = clip final
                            console.print(f"[cyan]Intégration sous-titres (single-pass)...[/cyan]")
                            final_clip_to_encode = CompositeVideoClip(
                                [processed_clip] + subtitle_clips,
                                size=(self.config.output_width, self.config.output_height)
                            )
                            console.print(f"[green]✓ {len(subtitle_clips)} sous-titres intégrés[/green]")
                        else:
                            console.print(f"[yellow]⚠ Aucun sous-titre généré[/yellow]")
                    else:
                        console.print(f"[yellow]⚠ Aucune transcription disponible pour ce clip[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠ Échec préparation sous-titres: {e}[/yellow]")
                    # Fallback: encoder sans sous-titres
                    final_clip_to_encode = processed_clip
                finally:
                    # Cleanup fichiers temporaires
                    if temp_audio_file and os.path.exists(temp_audio_file):
                        try:
                            os.remove(temp_audio_file)
                        except Exception:
                            pass
                    if ass_file and os.path.exists(ass_file):
                        try:
                            os.remove(ass_file)
                        except Exception:
                            pass
            
            # Écrire le fichier final (UN SEUL encodage)
            final_clip_to_encode.write_videofile(output_path, **write_params)
            
            # Cleanup des TextClips
            for tc in subtitle_clips:
                try:
                    tc.close()
                except Exception:
                    pass
            
        finally:
            if processed_clip is not None:
                try:
                    processed_clip.close()
                except Exception:
                    pass
            if subclip is not None:
                try:
                    subclip.close()
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
        5. Vignette (optionnel)
        
        Cache: Les frames sont mises en cache pour éviter de les redécoder
        """
        if frame is None or frame.size == 0:
            return np.zeros((self.config.output_height, self.config.output_width, 3), dtype=np.uint8)
        
        # Clé de cache basée sur timestamp (arrondi à 0.1s pour éviter trop de variations)
        cache_key = f"{id(frame)}_{timestamp:.1f}"
        
        # Vérifier le cache (désactivé en mode parallèle pour éviter les race conditions)
        if not self.config.parallel_processing:
            with _cache_lock:
                if cache_key in _frame_cache:
                    return _frame_cache[cache_key].copy()
        
        h, w = frame.shape[:2]
        
        # Obtenir le point de focus interpolé
        focus = self.cropper.get_interpolated_focus(focus_points, timestamp)
        
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
            # 1. Color grading
            if self.config.enable_color_grading:
                result = self._apply_color_grading(result)

            # 2. Sharpening
            if self.config.enable_sharpening:
                result = self._apply_sharpening(result)
        
        # 3. Vignette (après resize final pour de meilleures performances)
        # Note: vignette appliquée après le resize dans _generate_single_clip
        
        # Mettre en cache le résultat (limiter la taille du cache à 100 frames max)
        if not self.config.parallel_processing:
            with _cache_lock:
                if len(_frame_cache) > 500:  # Augmenté de 100 à 500 pour meilleure performance
                    # Supprimer 20% des frames les plus anciennes
                    keys_to_remove = list(_frame_cache.keys())[:100]  # Supprimer 100 au lieu de 20
                    for key in keys_to_remove:
                        del _frame_cache[key]
                
                _frame_cache[cache_key] = result.copy()
        
        return result
    
    def _apply_zoom_effect(self, clip: VideoFileClip) -> VideoFileClip:
        """
        Applique un effet de zoom dynamique avec différents styles.
        
        Styles disponibles:
        - ease_out: Zoom rapide au début, ralentit à la fin (cinématique)
        - ease_in_out: Accélération douce, décélération douce
        - breathing: Micro-oscillations comme une respiration
        - pulse: Pulsations subtiles au rythme
        
        Utilise LANCZOS4 pour une meilleure qualité de redimensionnement.
        """
        duration = clip.duration
        use_lanczos = self.config.use_lanczos
        zoom_factor = self.config.zoom_factor
        zoom_style = self.config.zoom_style
        
        def ease_out_quad(t: float) -> float:
            """Courbe ease-out quadratique"""
            return 1 - (1 - t) * (1 - t)
        
        def ease_out_cubic(t: float) -> float:
            """Courbe ease-out cubique (plus prononcée)"""
            return 1 - pow(1 - t, 3)
        
        def ease_in_out_sine(t: float) -> float:
            """Courbe ease-in-out sinusoïdale (très douce)"""
            import math
            return -(math.cos(math.pi * t) - 1) / 2
        
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            
            import math
            
            # Calculer le zoom selon le style
            if zoom_style == "ease_out":
                # Zoom rapide au début, ralentit à la fin
                eased_progress = ease_out_cubic(progress)
                base_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
                # Breathing subtil
                breath = math.sin(t * 0.5 * 2 * math.pi) * 0.002
                current_zoom = base_zoom + breath
                
            elif zoom_style == "ease_in_out":
                # Accélération et décélération douces
                eased_progress = ease_in_out_sine(progress)
                base_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
                current_zoom = base_zoom
                
            elif zoom_style == "breathing":
                # Oscillations comme une respiration
                # Zoom de base plus léger
                base_zoom = 1.0 + (zoom_factor - 1.0) * 0.5 * progress
                # Breathing principal (cycle de 3 secondes)
                breath_main = math.sin(t * (2 * math.pi / 3)) * 0.015
                # Harmonique secondaire pour plus de naturel
                breath_secondary = math.sin(t * (2 * math.pi / 1.7)) * 0.005
                current_zoom = base_zoom + breath_main + breath_secondary
                
            elif zoom_style == "pulse":
                # Pulsations subtiles
                eased_progress = ease_out_cubic(progress)
                base_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
                # Pulse toutes les 0.8 secondes avec decay
                pulse_freq = 1.25
                pulse = math.sin(t * pulse_freq * 2 * math.pi)
                pulse = max(0, pulse)  # Garder seulement les pics positifs
                pulse_decay = math.exp(-t * 0.1)  # Decay progressif
                current_zoom = base_zoom + pulse * 0.008 * pulse_decay
                
            else:
                # Fallback: zoom linéaire simple
                current_zoom = 1.0 + (zoom_factor - 1.0) * progress
            
            # Assurer un zoom minimum de 1.0
            current_zoom = max(1.0, current_zoom)
            
            h, w = frame.shape[:2]
            new_h, new_w = int(h / current_zoom), int(w / current_zoom)
            
            # Calculer les offsets pour centrer
            y_offset = (h - new_h) // 2
            x_offset = (w - new_w) // 2
            
            # S'assurer que les dimensions sont valides
            y_offset = max(0, y_offset)
            x_offset = max(0, x_offset)
            end_y = min(y_offset + new_h, h)
            end_x = min(x_offset + new_w, w)
            
            # Recadrer et redimensionner avec haute qualité
            cropped = frame[y_offset:end_y, x_offset:end_x]
            
            # Choisir l'interpolation: LANCZOS4 (meilleure) ou LINEAR (rapide)
            interpolation = cv2.INTER_LANCZOS4 if use_lanczos else cv2.INTER_LINEAR
            resized = cv2.resize(cropped, (w, h), interpolation=interpolation)
            
            return resized
        
        return clip.transform(zoom_effect)
    
    def _apply_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """
        Applique une correction colorimétrique cinématique OPTIMISÉE.
        
        Styles:
        - warm: Tons chauds dorés (style lifestyle/vlog)
        - cool: Tons froids bleutés (style tech/corporate)
        - vibrant: Couleurs saturées et contrastées
        - cinematic: Look film avec ombres teintées
        
        Optimisation: Opérations vectorisées, pas de conversions HSV multiples
        """
        style = self.config.color_grading_style
        
        if style == "none":
            return frame
        
        # Note: Pas de conversion float si style simple (warm/cool)
        if style in ["warm", "cool"]:
            # Opération directe sur uint8 pour vitesse maximale
            if style == "warm":
                # Boost rouge/jaune, réduire bleu
                result = frame.copy()
                result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
                result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
                return result
            else:  # cool
                result = frame.copy()
                result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
                result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
                return result
        
        # Pour vibrant et cinematic, on garde la conversion float (nécessaire)
        img = frame.astype(np.float32) / 255.0
        
        if style == "vibrant":
            # Saturation et contraste - UNE SEULE conversion HSV
            hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
            # Contraste
            img = np.clip((img - 0.5) * 1.15 + 0.5, 0, 1)
            
        elif style == "cinematic":
            # Look cinéma simplifié - SANS conversion HSV (plus rapide)
            # Teinter directement dans BGR
            shadows = np.clip(img, 0, 0.3) / 0.3
            highlights = np.clip((img - 0.7) / 0.3, 0, 1)
            
            img[:, :, 0] = img[:, :, 0] + shadows[:, :, 0] * 0.03  # Bleu dans ombres
            img[:, :, 2] = img[:, :, 2] + highlights[:, :, 2] * 0.04  # Rouge dans highlights
            
            # Contraste simplifié
            img = np.clip((img - 0.5) * 1.08 + 0.5, 0, 1)
        
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """
        Applique un sharpening intelligent qui préserve les détails
        sans amplifier le bruit.
        """
        strength = self.config.sharpening_strength
        
        if strength <= 0:
            return frame
        
        # Unsharp mask: sharpen = original + strength * (original - blur)
        # Utiliser un blur léger pour préserver les détails
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.5)
        
        # Calculer le masque de netteté
        sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
        
        return sharpened
    
    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """
        Applique un effet vignette subtil pour focaliser l'attention.
        """
        h, w = frame.shape[:2]
        strength = self.config.vignette_strength
        
        # Créer le masque de vignette
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Distance radiale elliptique (adaptée au format 9:16)
        radius = np.sqrt((X * 0.8) ** 2 + Y ** 2)
        
        # Vignette douce avec falloff gaussien
        vignette = 1 - np.clip(radius - 0.5, 0, 1) * strength * 2
        vignette = np.clip(vignette, 1 - strength, 1)
        
        # Appliquer
        result = (frame.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        return result
    
    def _apply_ken_burns_effect(self, clip: VideoFileClip) -> VideoFileClip:
        """
        Applique l'effet Ken Burns : mouvement panoramique subtil + zoom lent.
        
        Crée un effet documentaire/cinématique en combinant:
        - Un zoom progressif très lent (1.0 → 1.0 + intensity)
        - Un léger mouvement panoramique (pan) horizontal ou vertical
        - Des transitions douces avec easing
        
        L'effet est subtil pour ne pas distraire du contenu principal.
        """
        import math
        
        duration = clip.duration
        intensity = self.config.ken_burns_intensity
        use_lanczos = self.config.use_lanczos
        
        # Choisir une direction de pan aléatoire mais cohérente pour le clip
        # On utilise le hash de la durée pour avoir une direction reproductible
        pan_seed = int(duration * 1000) % 4
        pan_directions = [
            (1, 0),    # Droite
            (-1, 0),   # Gauche
            (0, 1),    # Bas
            (0, -1),   # Haut
        ]
        pan_x_dir, pan_y_dir = pan_directions[pan_seed]
        
        def ease_in_out_cubic(t: float) -> float:
            """Courbe ease-in-out cubique pour transitions douces"""
            if t < 0.5:
                return 4 * t * t * t
            else:
                return 1 - pow(-2 * t + 2, 3) / 2
        
        def ken_burns_transform(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            
            # Appliquer l'easing pour un mouvement naturel
            eased_progress = ease_in_out_cubic(progress)
            
            # Zoom progressif très lent (commence à 1.0, finit à 1.0 + intensity)
            # L'intensité est divisée par 2 car le zoom est appliqué en crop
            current_zoom = 1.0 + (intensity * eased_progress)
            
            # Pan subtil dans la direction choisie
            # Le pan est proportionnel à l'intensité et au progrès
            pan_amount = intensity * 0.3  # Le pan est plus subtil que le zoom
            pan_x = pan_x_dir * pan_amount * eased_progress
            pan_y = pan_y_dir * pan_amount * eased_progress
            
            h, w = frame.shape[:2]
            
            # Calculer la région de crop avec zoom et pan
            # new_w et new_h sont les dimensions de la fenêtre de crop
            new_w = int(w / current_zoom)
            new_h = int(h / current_zoom)
            
            # Position centrale avec décalage du pan
            center_x = w / 2 + (pan_x * w / 2)
            center_y = h / 2 + (pan_y * h / 2)
            
            # Calculer les coordonnées de crop
            x1 = int(center_x - new_w / 2)
            y1 = int(center_y - new_h / 2)
            x2 = x1 + new_w
            y2 = y1 + new_h
            
            # S'assurer qu'on reste dans les limites de l'image
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x2 > w:
                x1 -= (x2 - w)
                x2 = w
            if y2 > h:
                y1 -= (y2 - h)
                y2 = h
            
            # Clamp final
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Crop et resize
            cropped = frame[y1:y2, x1:x2]
            
            # Choisir l'interpolation
            interpolation = cv2.INTER_LANCZOS4 if use_lanczos else cv2.INTER_LINEAR
            
            # Redimensionner à la taille originale
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                result = cv2.resize(cropped, (w, h), interpolation=interpolation)
            else:
                result = frame
            
            return result
        
        return clip.transform(ken_burns_transform)


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
