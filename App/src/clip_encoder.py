"""
Module d'encodage de clips individuels
Gère le pipeline complet: extraction, recadrage, sous-titres, encodage
"""

import os
import gc
import platform
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from moviepy import (
    VideoFileClip,
    CompositeVideoClip,
)
from rich.console import Console

from .smart_cropper import FocusPoint
from .visual_effects import (
    apply_zoom_effect,
    apply_vignette,
    apply_ken_burns_effect,
)
from .viral_detector import ViralMoment

console = Console()

# Détecter si on est sur Mac pour l'accélération matérielle VideoToolbox
IS_MACOS = platform.system() == "Darwin"


def encode_single_clip(
    video: VideoFileClip,
    moment: ViralMoment,
    focus_points: List[Tuple[float, FocusPoint]],
    output_path: str,
    clip_number: int,
    config: 'ClipConfig',
    process_frame_func,
    transcription_result: Optional[Any] = None
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

    Args:
        video: Vidéo source ouverte
        moment: Moment viral à extraire
        focus_points: Points de focus pour le recadrage
        output_path: Chemin du fichier de sortie
        clip_number: Numéro du clip (pour affichage)
        config: Configuration du clip
        process_frame_func: Fonction de traitement frame par frame
        transcription_result: Résultat de transcription pré-calculée (optionnel)
    """
    subclip = None
    processed_clip = None

    try:
        # Extraire le segment
        subclip = video.with_subclip(moment.start_time, moment.end_time)

        if subclip.duration is None or subclip.duration <= 0:
            raise ValueError(f"Duree du subclip invalide: {subclip.duration}")

        # Récupérer le FPS source pour le préserver
        source_fps = subclip.fps if hasattr(subclip, 'fps') and subclip.fps else config.output_fps

        # Appliquer le recadrage intelligent + effets frame par frame
        def frame_transform(get_frame, t):
            frame = get_frame(t)
            return process_frame_func(
                frame,
                t + moment.start_time,
                focus_points
            )

        processed_clip = subclip.transform(frame_transform)

        # Forcer le FPS pour éviter les saccades (IMPORTANT)
        processed_clip = processed_clip.with_fps(source_fps)

        # Redimensionner à la taille finale
        processed_clip = processed_clip.resized(
            (config.output_width, config.output_height)
        )

        # Appliquer l'effet de zoom si activé (sauf en mode rapide)
        if config.enable_zoom_effect and not config.fast_mode:
            processed_clip = apply_zoom_effect(
                processed_clip, config.zoom_factor,
                config.zoom_style, config.use_lanczos
            )

        # Appliquer l'effet Ken Burns si activé (après zoom, avant vignette)
        if config.enable_ken_burns and not config.fast_mode:
            processed_clip = apply_ken_burns_effect(
                processed_clip, config.ken_burns_intensity,
                config.use_lanczos
            )

        # Appliquer la vignette si activée (après resize pour meilleures perfs, sauf fast mode)
        if config.enable_vignette and not config.fast_mode:
            def vignette_effect(get_frame, t):
                frame = get_frame(t)
                return apply_vignette(frame, config.vignette_strength)
            processed_clip = processed_clip.transform(vignette_effect)

        # Construire les paramètres d'encodage
        write_params, output_fps = _build_encoding_params(config, source_fps)

        # =====================================================
        # OPTIMISATION SINGLE-PASS: Sous-titres intégrés AVANT encodage
        # Évite le double encodage (économise ~50% du temps d'encodage)
        # =====================================================
        final_clip_to_encode = processed_clip
        subtitle_clips = []
        temp_audio_file = None
        ass_file = None

        if config.add_subtitles:
            final_clip_to_encode, subtitle_clips, temp_audio_file, ass_file = (
                _integrate_subtitles(
                    subclip, processed_clip, output_path,
                    clip_number, config
                )
            )

        # Écrire le fichier final (UN SEUL encodage)
        final_clip_to_encode.write_videofile(output_path, **write_params)

        # Cleanup des TextClips
        for tc in subtitle_clips:
            try:
                tc.close()
            except Exception:
                pass

    finally:
        # Fermer le CompositeVideoClip s'il est différent du processed_clip
        if final_clip_to_encode is not None and final_clip_to_encode is not processed_clip:
            try:
                final_clip_to_encode.close()
            except Exception:
                pass
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


def _build_encoding_params(
    config: 'ClipConfig',
    source_fps: float
) -> Tuple[Dict[str, Any], int]:
    """
    Construit les paramètres d'encodage FFmpeg selon la config.

    Args:
        config: Configuration du clip
        source_fps: FPS de la vidéo source

    Returns:
        Tuple (paramètres write_videofile, fps de sortie)
    """
    use_hw = config.use_hardware_accel and IS_MACOS
    output_fps = int(source_fps) if source_fps else config.output_fps
    gop_size = output_fps * 2  # Keyframe toutes les 2 secondes

    if use_hw:
        # === ENCODAGE GPU (VideoToolbox sur Mac) ===
        codec = 'h264_videotoolbox'
        ffmpeg_params = [
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-allow_sw', '1',
            '-realtime', '0',
            '-max_muxing_queue_size', '9999',
            '-profile:v', 'high',
            '-vsync', 'cfr',
            '-r', str(output_fps),
            '-g', str(gop_size),
            '-bf', '2',
        ]
        ffmpeg_params.extend(['-q:v', '40'])
        ffmpeg_params.extend(['-b:v', config.video_bitrate])
        console.print(
            f"[green]Encodage GPU (VideoToolbox) @ {output_fps}fps"
            f" - Haute qualité (q:v=40, {config.video_bitrate})[/green]"
        )
    else:
        # === ENCODAGE CPU (libx264) ===
        codec = 'libx264'
        ffmpeg_params = [
            '-profile:v', config.video_profile,
            '-level', config.video_level,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-preset', config.preset,
            '-max_muxing_queue_size', '9999',
            '-vsync', 'cfr',
            '-r', str(output_fps),
            '-g', str(gop_size),
            '-bf', '2',
            '-sc_threshold', '0',
            '-video_track_timescale', str(output_fps * 1000),
            '-tune', 'film',
            '-x264opts', 'rc-lookahead=60:ref=6:deblock=-1,-1:aq-mode=3',
        ]
        console.print(
            f"[dim]Encodage CPU (libx264) @ {output_fps}fps"
            f" - CRF {config.crf}, preset {config.preset}[/dim]"
        )

        if config.crf is not None:
            ffmpeg_params.extend(['-crf', str(config.crf)])

    num_threads = os.cpu_count() or 8

    write_params: Dict[str, Any] = {
        'fps': output_fps,
        'codec': codec,
        'audio_codec': 'aac',
        'audio_bitrate': config.audio_bitrate,
        'logger': None,
        'threads': num_threads,
        'ffmpeg_params': ffmpeg_params,
    }

    if not use_hw and config.crf is None:
        write_params['bitrate'] = config.video_bitrate

    return write_params, output_fps


def _integrate_subtitles(
    subclip: VideoFileClip,
    processed_clip,
    output_path: str,
    clip_number: int,
    config: 'ClipConfig'
) -> Tuple[Any, List, Optional[str], Optional[str]]:
    """
    Intègre les sous-titres TikTok dans le clip (single-pass).

    Args:
        subclip: Sous-clip vidéo pour extraction audio
        processed_clip: Clip traité (recadré, effets appliqués)
        output_path: Chemin de sortie (pour fichiers temporaires)
        clip_number: Numéro du clip
        config: Configuration du clip

    Returns:
        Tuple (clip final à encoder, liste de TextClips, chemin audio temp, chemin ASS)
    """
    subtitle_clips: List = []
    temp_audio_file: Optional[str] = None
    ass_file: Optional[str] = None
    final_clip = processed_clip

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
                fps=16000,
                nbytes=2,
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

                    # Analyser la qualité du hook (3 premières secondes)
                    _analyze_hook_quality(clip_words)
                else:
                    console.print(f"[yellow]⚠ Aucun mot détecté dans le clip[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠ Erreur transcription clip: {e}[/yellow]")

        # 3. Générer le fichier ASS et les TextClips
        if clip_words:
            ass_file = output_path.replace('.mp4', '.ass')
            generate_tiktok_ass(
                words=clip_words,
                preset_colors=config.subtitle_custom_colors,
                output_path=ass_file,
                style_config={
                    'theme': config.subtitle_theme,
                    'max_words': config.subtitle_max_words,
                    'use_emojis': config.subtitle_use_emojis
                }
            )

            # 4. Convertir ASS en TextClips MoviePy
            subtitle_clips = render_ass_to_moviepy(
                ass_file,
                video_width=config.output_width,
                video_height=config.output_height
            )

            if subtitle_clips:
                # 5. Composite: processed_clip + sous-titres = clip final
                console.print(f"[cyan]Intégration sous-titres (single-pass)...[/cyan]")
                final_clip = CompositeVideoClip(
                    [processed_clip] + subtitle_clips,
                    size=(config.output_width, config.output_height)
                )
                console.print(f"[green]✓ {len(subtitle_clips)} sous-titres intégrés[/green]")
            else:
                console.print(f"[yellow]⚠ Aucun sous-titre généré[/yellow]")
        else:
            console.print(f"[yellow]⚠ Aucune transcription disponible pour ce clip[/yellow]")

    except Exception as e:
        console.print(f"[yellow]⚠ Échec préparation sous-titres: {e}[/yellow]")
        final_clip = processed_clip
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

    return final_clip, subtitle_clips, temp_audio_file, ass_file


def _analyze_hook_quality(clip_words: List) -> None:
    """
    Analyse la qualité du hook (3 premières secondes) et affiche le résultat.

    Args:
        clip_words: Liste des mots transcrits avec timestamps
    """
    try:
        from .hook_optimizer import HookOptimizer
        HOOK_OPTIMIZER_AVAILABLE = True
    except ImportError:
        HOOK_OPTIMIZER_AVAILABLE = False

    if not HOOK_OPTIMIZER_AVAILABLE or not clip_words:
        return

    try:
        optimizer = HookOptimizer(min_hook_score=0.4, hook_duration=3.0)
        hook_words = [
            w for w in clip_words
            if hasattr(w, 'start') and w.start <= 3.0
        ]
        if hook_words:
            hook_text = " ".join(
                w.word if hasattr(w, 'word') else str(w)
                for w in hook_words
            )
            analysis = optimizer.analyze_hook(hook_text)
            if analysis.score >= 0.6:
                console.print(
                    f"[green]✓ Hook qualité:"
                    f" {analysis.score:.0%}"
                    f" ({analysis.hook_type})[/green]"
                )
            elif analysis.score >= 0.4:
                console.print(
                    f"[yellow]⚠ Hook moyen:"
                    f" {analysis.score:.0%}"
                    f" - {hook_text[:50]}...[/yellow]"
                )
            else:
                console.print(
                    f"[red]⚠ Hook faible:"
                    f" {analysis.score:.0%}"
                    f" - considérer un autre"
                    f" point de départ[/red]"
                )
    except Exception:
        pass  # Silently ignore hook analysis errors
