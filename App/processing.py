"""Pipeline de traitement video pour ClipGenius."""

import os
import sys
import gc
import time
import threading
import shutil
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from cleanup import cleanup_residual_files


def process_video(job_id: str, url: str, options: dict, jobs: dict, jobs_lock: threading.Lock):
    """Traite une vidéo YouTube ou locale en arrière-plan"""

    # === LAZY IMPORTS (chargés ici pour démarrage rapide de Flask) ===
    from src.downloader import VideoDownloader
    from src.viral_detector import ViralMoment, ViralMomentDetector, ContentType
    from src.clip_generator import ClipGenerator, ClipConfig
    from src.subtitles import SubtitleGenerator
    from src.ai_analyzer import TranscriptSegment, analyze_with_ai, DEFAULT_MAX_CLIPS
    from src.auto_config import AutoConfigurator, GeneratedConfig

    # Import différé pour éviter les imports circulaires
    from web_app import LogCapture

    logger = LogCapture(job_id)
    jobs[job_id] = {"status": "running", "clips": [], "error": None, "_created": time.time()}

    video_path = None
    is_downloaded = False
    is_uploaded = False  # Pour les fichiers uploadés
    transcription_result = None  # Sera récupéré du job d'analyse si disponible

    try:
        # Vérifier si c'est un fichier local ou une URL YouTube
        local_file = options.get('local_file')
        skip_download = options.get('skip_download', False)

        # Si on réutilise une analyse, commencer directement à l'étape "generate"
        if skip_download and local_file:
            video_path = local_file
            is_uploaded = True

            if not Path(video_path).exists():
                raise Exception("Fichier introuvable")

            # Démarrer directement à "generate" avec 5% pour initialiser la barre bleue
            logger.log("Démarrage de la génération...", "info", "generate", 5)
        elif local_file:
            # === FICHIER LOCAL (quasi instantané) ===
            video_path = local_file
            is_uploaded = True

            if not Path(video_path).exists():
                raise Exception("Fichier introuvable")

            file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            logger.log(f"Fichier: {Path(video_path).name} ({file_size_mb:.1f} MB)", "success", "download", 100)
        else:
            # === URL YOUTUBE ===
            logger.log("Initialisation du téléchargement...", "info", "download", 0)

            downloader = VideoDownloader(output_dir="downloads")

            # Récupérer les infos vidéo
            try:
                logger.log("Récupération des informations...", "info", "download", 2)
                info = downloader.get_video_info(url)
                logger.log(f"Titre: {info['title']}", "info", "download", 5)
                duration_min = info['duration'] // 60
                duration_sec = info['duration'] % 60
                logger.log(f"Durée: {duration_min}:{duration_sec:02d}", "info", "download", 7)
                logger.log(f"Auteur: {info['uploader']}", "info", "download", 10)
            except Exception as e:
                logger.log(f"Impossible de récupérer les infos: {e}", "warning", "download", 10)

            # Callback de progression pour le téléchargement
            def download_progress(percent: int, message: str):
                logger.log(message, "info", "download", percent)

            video_path = downloader.download(
                url,
                quality=options.get('quality', '1080p'),
                progress_callback=download_progress
            )
            is_downloaded = True

            if not video_path:
                raise Exception("Échec du téléchargement")

            logger.log(f"Vidéo téléchargée: {Path(video_path).name}", "success", "download", 100)

        # === ÉTAPE 1b: Vérifier si on peut réutiliser la transcription 'base' du job d'analyse ===
        analysis_job_id = options.get('analysis_job_id')

        if analysis_job_id and analysis_job_id in jobs:
            analysis_data = jobs[analysis_job_id].get('data', {})

            if analysis_data and 'transcription' in analysis_data:
                transcription_result = analysis_data['transcription'].get('result')
                if transcription_result:
                    # Transcription récupérée depuis l'analyse!
                    nb_segments = len(transcription_result.segments) if transcription_result.segments else 0
                    nb_words = len(transcription_result.words) if transcription_result.words else 0
                    logger.log(f"✓ Transcription réutilisée depuis analyse ({nb_segments} segments, {nb_words} mots)", "success", "generate", 5)

        # === ÉTAPE 1c: Transcription Whisper (seulement si pas déjà disponible) ===
        if transcription_result is None and options.get('use_ai', True):
            logger.log("Transcription Whisper du contenu...", "info", "transcribe", 0)

            try:
                subtitle_gen = SubtitleGenerator(
                    model_size=options.get('whisper_model', 'turbo'),
                    language=options.get('language', None)
                )

                # Callback pour reporter la progression
                def transcription_progress(percent: int, message: str):
                    logger.log(message, "info", "transcribe", percent)

                transcription_result = subtitle_gen.transcribe_with_words(
                    video_path,
                    progress_callback=transcription_progress
                )

                nb_segments = len(transcription_result.segments)
                nb_words = len(transcription_result.words)
                logger.log(f"Transcription: {nb_segments} segments, {nb_words} mots", "success", "transcribe", 100)

            except Exception as e:
                logger.log(f"Erreur transcription: {e}", "warning", "transcribe", 100)
                logger.log("Génération sans sous-titres", "info", "transcribe", 100)
                transcription_result = None

        # === ÉTAPE 1c: Auto-configuration (avec transcription si disponible) ===
        auto_generated_config: Optional[GeneratedConfig] = None
        if options.get('auto_config', False):
            logger.log("Auto-configuration intelligente", "info", "autoconfig", 0)

            try:
                configurator = AutoConfigurator(
                    max_analysis_duration=180.0,
                    verbose=False
                )

                platform = options.get('platform', 'reels')
                auto_generated_config, analysis = configurator.analyze_and_configure(
                    video_path,
                    platform=platform,
                    transcription_result=transcription_result  # ✨ Nouvelle feature!
                )

                # Afficher le résultat
                confidence_pct = auto_generated_config.content_confidence * 100
                logger.log(
                    f"Contenu: {auto_generated_config.detected_content_type} (confiance: {confidence_pct:.0f}%)",
                    "success", "autoconfig", 40
                )

                # Verbose
                if options.get('auto_config_verbose', False):
                    logger.log(f"Énergie moyenne: {analysis.energy_profile.average:.2f}", "info", "autoconfig", 50)
                    logger.log(f"Variance: {analysis.energy_profile.variance:.2f}", "info", "autoconfig", 52)
                    logger.log(f"Pics: {analysis.excitement_peaks}", "info", "autoconfig", 54)
                    logger.log(f"Émotion: {analysis.dominant_emotion.value}", "info", "autoconfig", 56)

                logger.log("Config:", "info", "autoconfig", 70)
                logger.log(f"  Durées: {auto_generated_config.min_duration:.0f}-{auto_generated_config.max_duration:.0f}s", "info", "autoconfig", 72)
                logger.log(f"  Score min: {auto_generated_config.min_viral_score:.2f}", "info", "autoconfig", 74)
                logger.log(f"  Zoom: {auto_generated_config.zoom_style} ({auto_generated_config.zoom_factor}x)", "info", "autoconfig", 76)
                logger.log(f"  Couleurs: {auto_generated_config.color_grading}", "info", "autoconfig", 78)
                logger.log(f"  Sous-titres: {auto_generated_config.subtitle_theme}", "info", "autoconfig", 80)

                # Appliquer
                options['min_duration'] = auto_generated_config.min_duration
                options['max_duration'] = auto_generated_config.max_duration
                options['min_score'] = auto_generated_config.min_viral_score
                options['max_words'] = auto_generated_config.subtitle_max_words
                options['emojis'] = auto_generated_config.subtitle_emojis

                logger.log("Auto-config appliquée!", "success", "autoconfig", 100)

            except Exception as e:
                logger.log(f"Erreur auto-config: {e}", "warning", "autoconfig", 100)
                logger.log("Paramètres par défaut utilisés", "info", "autoconfig", 100)
                auto_generated_config = None

        # Extraire le type de contenu détecté (pour adapter l'analyse)
        detected_content_type = "unknown"
        if auto_generated_config and auto_generated_config.detected_content_type:
            detected_content_type = auto_generated_config.detected_content_type.lower()

        # === ÉTAPE 2: Analyse des moments viraux (25-50%) ===
        logger.log("Analyse des moments viraux...", "info", "analyze", 0)

        min_score = options.get('min_score', 0.80)
        min_duration = options.get('min_duration', 60)
        max_duration = options.get('max_duration', 90)
        max_clips = options.get('max_clips', None)

        moments = []

        if options.get('use_ai', True) and transcription_result:
            # Transcription déjà faite, passer directement à l'analyse
            try:
                if transcription_result.segments:
                    # Durée vidéo
                    logger.log("Analyse de la structure vidéo...", "info", "analyze", 5)
                    from moviepy import VideoFileClip
                    with VideoFileClip(video_path) as video:
                        video_duration = video.duration

                    # Analyse IA
                    logger.log("Détection des moments clés...", "info", "analyze", 10)
                    transcript_segments = [
                        TranscriptSegment(start=s.start, end=s.end, text=s.text)
                        for s in transcription_result.segments
                    ]

                    # Callback pour la progression de l'analyse IA
                    def ai_progress_callback(progress: float, message: str):
                        # Mapper la progression IA (0-1) sur la plage 10-95% de l'étape analyze
                        mapped_progress = int(10 + progress * 85)
                        logger.log(message, "info", "analyze", mapped_progress)

                    # Utilise le LLM local Phi-4-mini
                    logger.log("Détection moments viraux...", "info", "analyze", 15)
                    ai_moments = analyze_with_ai(
                        segments=transcript_segments,
                        video_duration=video_duration,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_clips=max_clips or DEFAULT_MAX_CLIPS,  # Défaut réduit de 10 à 5
                        min_viral_score=min_score,  # Propager le seuil utilisateur
                        video_path=video_path,
                        progress_callback=ai_progress_callback,
                        content_type=detected_content_type  # Adapter l'analyse au type de contenu
                    )

                    logger.log(f"IA a retourné {len(ai_moments)} moments", "info", "analyze", 95)

                    # Convertir tous les moments (déjà filtrés par min_viral_score)
                    all_moments = [
                        ViralMoment(
                            start_time=ai_m.start_time,
                            end_time=ai_m.end_time,
                            score=ai_m.score,
                            reason=f"[{ai_m.emotion}] {ai_m.reason}"
                        )
                        for ai_m in ai_moments
                    ]

                    # Les moments sont déjà filtrés par l'analyseur, pas besoin de re-filtrer
                    moments = all_moments

                    if moments:
                        logger.log(f"{len(moments)} moments viraux détectés", "info", "analyze", 98)
                    else:
                        logger.log("L'IA n'a trouvé aucun moment suffisamment viral", "warning", "analyze", 98)

            except Exception as e:
                logger.log(f"Erreur analyse IA: {e}", "error", "analyze", 90)
                moments = []

        # === FALLBACK: Analyse audio/vidéo si l'IA n'a rien trouvé ===
        if not moments:
            logger.log("Fallback: Analyse audio/vidéo...", "info", "analyze", 75)
            try:
                # Convertir la string en enum ContentType
                try:
                    content_type_enum = ContentType(detected_content_type)
                except ValueError:
                    content_type_enum = ContentType.UNKNOWN

                detector = ViralMomentDetector(
                    min_clip_duration=min_duration,
                    max_clip_duration=max_duration,
                    min_viral_score=min_score,
                    max_clips=max_clips or DEFAULT_MAX_CLIPS,
                    content_type=content_type_enum
                )
                moments = detector.analyze(video_path)

                if moments:
                    logger.log(f"Analyse audio/vidéo: {len(moments)} moments trouvés", "success", "analyze", 95)
                else:
                    logger.log("Analyse audio/vidéo: aucun moment trouvé", "warning", "analyze", 95)
            except Exception as e:
                logger.log(f"Erreur fallback: {e}", "error", "analyze", 95)
                moments = []

        if not moments:
            # Aucun moment trouvé - terminer gracieusement sans erreur
            logger.log("Aucun moment viral détecté dans cette vidéo", "warning", "analyze", 100)
            logger.log("La vidéo ne contient pas de moments suffisamment engageants pour créer des clips", "info", "complete", 100)

            # Nettoyer les fichiers temporaires (sauf si fichier uploadé par l'utilisateur)
            if is_downloaded and video_path:
                cleanup_residual_files(video_path, "output", keep_user_files=False)

            # Marquer le job comme terminé (pas d'erreur, juste 0 clips)
            jobs[job_id] = {
                "status": "completed",
                "clips": [],
                "message": "Aucun moment viral détecté. La vidéo ne contient pas de contenu suffisamment engageant pour créer des clips courts.",
                "_created": time.time()
            }
            logger.close()
            return

        # Afficher les moments trouvés
        for i, m in enumerate(moments, 1):
            duration = m.end_time - m.start_time
            logger.log(
                f"Moment {i}: {m.start_time:.1f}s - {m.end_time:.1f}s ({duration:.0f}s) - Score: {m.score:.0%}",
                "info", "analyze", 92 + i
            )

        logger.log(f"{len(moments)} moments viraux trouvés", "success", "analyze", 100)

        # === ÉTAPE 3: Génération des clips (50-75%) ===
        # Sécurité: limiter le nombre de clips au maximum configuré
        effective_max_clips = max_clips or DEFAULT_MAX_CLIPS
        if len(moments) > effective_max_clips:
            moments.sort(key=lambda m: m.score, reverse=True)
            moments = moments[:effective_max_clips]
            logger.log(f"Limité à {effective_max_clips} meilleurs clips", "info", "analyze", 99)

        total_clips = len(moments)
        logger.log(f"Génération de {total_clips} clip(s) vertical(s)...", "info", "generate", 0)

        output_dir = options.get('output_dir', 'output')

        # Créer la config avec les paramètres auto-générés si disponibles
        if auto_generated_config:
            config = ClipConfig(
                min_clip_duration=min_duration,
                max_clip_duration=max_duration,
                min_viral_score=min_score,
                max_clips=max_clips or DEFAULT_MAX_CLIPS,
                add_subtitles=True,
                enable_zoom_effect=True,
                zoom_factor=auto_generated_config.zoom_factor,
                zoom_style=auto_generated_config.zoom_style,
                enable_color_grading=True,
                color_grading_style=auto_generated_config.color_grading,
                enable_sharpening=True,
                sharpening_strength=auto_generated_config.sharpening_strength,
                enable_vignette=auto_generated_config.vignette_enabled,
                vignette_strength=auto_generated_config.vignette_strength,
            )
            logger.log(f"Effets visuels: {auto_generated_config.color_grading}, zoom {auto_generated_config.zoom_style}", "info", "generate", 3)
        else:
            config = ClipConfig(
                min_clip_duration=min_duration,
                max_clip_duration=max_duration,
                min_viral_score=min_score,
                max_clips=max_clips or DEFAULT_MAX_CLIPS,
                add_subtitles=True,
                enable_zoom_effect=True
            )

        generator = ClipGenerator(config, transcription_result=transcription_result)

        logger.log("Initialisation du générateur...", "info", "generate", 5)

        # === HEARTBEAT THREAD pour maintenir la connexion pendant les opérations longues ===
        heartbeat_stop = threading.Event()

        def heartbeat_thread():
            progress = 15
            while not heartbeat_stop.is_set():
                logger.heartbeat("generate", progress)
                progress = min(progress + 2, 90)  # Augmente lentement jusqu'à 90%
                heartbeat_stop.wait(10)  # Toutes les 10 secondes

        hb_thread = threading.Thread(target=heartbeat_thread, daemon=True)
        hb_thread.start()

        try:
            # === MODE PARALLÈLE: Générer tous les clips en même temps ===
            if config.parallel_processing and len(moments) > 1:
                logger.log(
                    f"Traitement parallèle activé ({config.max_parallel_clips} clips simultanés)...",
                    "info", "generate", 10
                )

                # Callback pour suivre la progression
                def progress_callback(clip_num, total, message):
                    progress = 10 + int((clip_num / total) * 85)
                    logger.log(message, "info", "generate", progress)

                try:
                    # Générer tous les clips en parallèle
                    clips = generator.generate_clips(video_path, output_dir, moments, start_index=1)

                    if clips:
                        logger.log(
                            f"{len(clips)} clips générés avec succès",
                            "success", "generate", 95
                        )
                except Exception as e:
                    logger.log(f"Erreur génération: {e}", "warning", "generate", 95)
                    clips = []

            else:
                # === MODE SÉQUENTIEL: Générer les clips un par un pour le suivi ===
                logger.log("Génération séquentielle...", "info", "generate", 10)
                clips = []
                for i, moment in enumerate(moments):
                    clip_num = i + 1
                    # Progression de 10% à 95% répartie entre les clips
                    progress_start = 10 + int((i / total_clips) * 85)
                    progress_end = 10 + int(((i + 1) / total_clips) * 85)

                    duration = moment.end_time - moment.start_time
                    logger.log(
                        f"Clip {clip_num}/{total_clips}: extraction ({duration:.0f}s)...",
                        "info", "generate", progress_start
                    )

                    # Sous-étapes de progression
                    logger.log(
                        f"Clip {clip_num}/{total_clips}: analyse des points de focus...",
                        "info", "generate", progress_start + int((progress_end - progress_start) * 0.2)
                    )

                    # Générer ce clip
                    try:
                        logger.log(
                            f"Clip {clip_num}/{total_clips}: encodage vidéo...",
                            "info", "generate", progress_start + int((progress_end - progress_start) * 0.5)
                        )

                        clip_paths = generator.generate_clips(video_path, output_dir, [moment], start_index=clip_num)

                        if clip_paths:
                            clips.extend(clip_paths)
                            logger.log(
                                f"Clip {clip_num}/{total_clips}: {Path(clip_paths[0]).name}",
                                "success", "generate", progress_end
                            )
                    except Exception as e:
                        logger.log(f"Erreur clip {clip_num}: {e}", "warning", "generate", progress_end)
                        print(f"[ERREUR GÉNÉRATION] Clip {clip_num}: {e}")  # Log serveur
                        traceback.print_exc()
        finally:
            # Arrêter le thread heartbeat
            heartbeat_stop.set()

        if not clips:
            raise Exception("Aucun clip généré")

        print(f"[GÉNÉRATION OK] {len(clips)} clips générés: {clips}")  # Log serveur
        logger.log(f"{len(clips)} clips générés avec succès", "success", "generate", 100)

        # === ÉTAPE 4: Sous-titres (maintenant intégrés dans la génération) ===
        # Les sous-titres sont ajoutés directement pendant la génération des clips
        # via le système ASS/FFmpeg dans ClipGenerator._generate_single_clip()
        if options.get('subtitles', True):
            logger.log("Sous-titres ajoutés pendant la génération", "success", "subtitles", 100)
        else:
            logger.log("Sous-titres désactivés", "info", "subtitles", 100)

        # === NOTE: Le message "complete" sera envoyé APRÈS la copie des clips ===
        # pour éviter que le frontend fetch le status avant que clip_data soit prêt

        # === COPIE AUTOMATIQUE VERS DOSSIER TÉLÉCHARGEMENTS ===
        # Détecter le dossier Téléchargements de l'utilisateur
        home = Path.home()
        downloads_dir = home / "Downloads"

        try:
            logger.log(f"Déplacement des clips vers {downloads_dir}...", "info", "complete", 100)

            moved_count = 0
            clips_to_delete = []  # Liste des clips à supprimer après extraction métadonnées

            for clip_path in clips:
                src = Path(clip_path)
                dst = downloads_dir / src.name

                if src.exists():
                    # Copier vers Downloads
                    shutil.copy2(src, dst)
                    moved_count += 1
                    # Marquer pour suppression après extraction métadonnées
                    clips_to_delete.append(src)
                    logger.log(f"✓ {src.name} → Téléchargements", "success", "complete", 100)

            logger.log(f"{moved_count} clips déplacés dans Téléchargements", "success", "complete", 100)

        except Exception as e:
            logger.log(f"Avertissement: impossible de déplacer vers Téléchargements: {e}", "warning", "complete", 100)
            clips_to_delete = []  # Ne pas supprimer si la copie a échoué

        # Convertir les chemins en URLs relatives + extraire métadonnées
        clip_data = []
        for i, clip_path in enumerate(clips):
            try:
                clip_file = Path(clip_path)

                # Les clips ont été copiés vers Downloads, chercher là-bas
                downloads_clip = downloads_dir / clip_file.name

                # Vérifier que le fichier existe dans Downloads
                if not downloads_clip.exists():
                    logger.log(f"⚠️ Clip introuvable dans Downloads: {downloads_clip}", "warning", "complete", 100)
                    # Fallback: essayer dans output/ (si copie a échoué)
                    if clip_file.exists():
                        downloads_clip = clip_file
                        logger.log(f"  → Utilisation depuis output/: {clip_file}", "info", "complete", 100)
                    else:
                        logger.log(f"⚠️ Clip introuvable partout, skip", "warning", "complete", 100)
                        continue

                # Extraire les métadonnées du fichier dans Downloads
                # Utiliser /clips/ au lieu de /output/ car on sert depuis ~/Downloads
                metadata = {
                    "url": f"/clips/{downloads_clip.name}",
                    "name": downloads_clip.name,
                    "index": i,
                    "size": round(downloads_clip.stat().st_size / (1024 * 1024), 1),  # MB
                }

                # Extraire durée et score à partir du nom de fichier ou via ffprobe
                try:
                    from moviepy import VideoFileClip
                    with VideoFileClip(str(downloads_clip)) as vc:
                        metadata["duration"] = round(vc.duration, 1)
                except Exception as e:
                    logger.log(f"⚠️ Impossible d'extraire la durée de {downloads_clip.name}: {e}", "warning", "complete", 100)
                    metadata["duration"] = 60.0  # Valeur par défaut

                # Score estimé - les premiers clips ont les meilleurs scores
                # (décroît progressivement de 0.95 à 0.80)
                # TODO: extraire le vrai score depuis les moments viraux
                score_range = 0.15  # 0.95 - 0.80 = 0.15
                score_decrement = score_range / max(len(clips), 1)
                metadata["score"] = round(0.95 - (i * score_decrement), 2)

                clip_data.append(metadata)
                logger.log(f"✓ Métadonnées extraites pour {downloads_clip.name}", "info", "complete", 100)

            except Exception as e:
                logger.log(f"❌ Erreur extraction métadonnées pour {clip_path}: {e}", "error", "complete", 100)
                # Ajouter quand même un objet minimal
                clip_data.append({
                    "url": f"/clips/{Path(clip_path).name}",
                    "name": Path(clip_path).name,
                    "index": i,
                    "size": 5.0,
                    "duration": 60.0,
                    "score": 0.85
                })

        logger.log(f"📊 {len(clip_data)} clips prêts avec métadonnées", "success", "complete", 100)

        # Log des URLs générées pour debug
        for i, clip in enumerate(clip_data):
            logger.log(f"  Clip {i+1}: {clip['url']} ({clip['size']} MB)", "info", "complete", 100)

        jobs[job_id] = {"status": "completed", "clips": clip_data, "error": None, "_created": time.time()}

        # === TERMINÉ - Envoyer APRÈS mise à jour du job ===
        # Le frontend va fetch /api/status dès réception de ce message
        # clip_data doit être prêt AVANT d'envoyer "finished"
        # NOTE: On utilise "finished" au lieu de "complete" pour distinguer du step de progression
        logger.log(f"Traitement terminé: {len(clip_data)} clips créés!", "success", "finished", 100)

        # === NETTOYAGE IMMÉDIAT D'OUTPUT/ ===
        # Les clips sont maintenant servis depuis ~/Downloads via /clips/
        # On peut donc nettoyer output/ immédiatement
        if clips_to_delete:
            logger.log(f"🗑️ Nettoyage immédiat de {len(clips_to_delete)} clips d'output/ (servis depuis Downloads)", "info", "complete", 100)
            deleted_count = 0
            for clip_file in clips_to_delete:
                try:
                    if clip_file.exists():
                        clip_file.unlink()
                        deleted_count += 1
                except Exception:
                    pass
            if deleted_count > 0:
                logger.log(f"✓ {deleted_count} clips supprimés d'output/", "info", "complete", 100)

        # Nettoyer aussi les autres fichiers temporaires
        temp_patterns = ['*TEMP_MPY_*.mp4', '*_sub.json', '*.ass', '.sanitized_*.mp4']
        for pattern in temp_patterns:
            for f in Path('.').glob(pattern):
                try:
                    f.unlink()
                except Exception:
                    pass
            for f in Path(output_dir).glob(pattern):
                try:
                    f.unlink()
                except Exception:
                    pass

        # Nettoyer pycaps
        temp_base = Path(tempfile.gettempdir())
        for pycaps_dir in temp_base.glob('pycaps_viral_*'):
            try:
                shutil.rmtree(pycaps_dir, ignore_errors=True)
            except Exception:
                pass

        # === Supprimer la vidéo source YouTube après succès ===
        # Uniquement pour les vidéos téléchargées, pas les fichiers importés
        if is_downloaded and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.log("Vidéo YouTube supprimée", "info", "complete", 100)
            except Exception:
                pass
        else:
            logger.log("Vidéo source préservée", "info", "complete", 100)

    except Exception as e:
        error_msg = str(e)
        logger.log(f"ERREUR: {error_msg}", "error", "error", 0)
        logger.log(traceback.format_exc(), "error", "error", 0)
        jobs[job_id] = {"status": "failed", "clips": [], "error": error_msg, "_created": time.time()}

    finally:
        # Attendre 2 secondes avant de fermer pour laisser le temps au frontend de recevoir le dernier message
        time.sleep(2)
        logger.close()


def analyze_video(job_id: str, url_or_path: str, is_local: bool, jobs: dict, jobs_lock: threading.Lock):
    """Analyse une vidéo (téléchargement + transcription + auto-config uniquement)"""

    # === LAZY IMPORTS ===
    from src.downloader import VideoDownloader
    from src.subtitles import SubtitleGenerator
    from src.auto_config import AutoConfigurator
    from moviepy import VideoFileClip

    # Import différé pour éviter les imports circulaires
    from web_app import LogCapture

    logger = LogCapture(job_id)
    jobs[job_id] = {"status": "running", "data": None, "error": None, "_created": time.time()}

    video_path = None

    try:
        # === ÉTAPE 1: Téléchargement (si YouTube) ===
        if is_local:
            # Fichier local
            video_path = url_or_path
            if not Path(video_path).exists():
                raise Exception("Fichier introuvable")

            file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            logger.log(f"Fichier: {Path(video_path).name} ({file_size_mb:.1f} MB)", "success", "download", 100)
        else:
            # URL YouTube - téléchargement avec progression
            logger.log("Initialisation du téléchargement...", "info", "download", 0)

            downloader = VideoDownloader(output_dir="downloads")

            # Callback de progression
            def download_progress(percent: int, message: str):
                logger.log(message, "info", "download", percent)

            video_path = downloader.download(
                url_or_path,
                quality='1080p',
                progress_callback=download_progress
            )

            if not video_path:
                raise Exception("Échec du téléchargement")

            logger.log(f"Vidéo téléchargée: {Path(video_path).name}", "success", "download", 100)

        # === ÉTAPE 2: Transcription Whisper ===
        logger.log("Transcription Whisper du contenu...", "info", "transcribe", 0)

        subtitle_gen = SubtitleGenerator(model_size='base', language=None)

        def transcription_progress(percent: int, message: str):
            logger.log(message, "info", "transcribe", percent)

        transcription_result = subtitle_gen.transcribe_with_words(
            video_path,
            progress_callback=transcription_progress
        )

        nb_segments = len(transcription_result.segments)
        nb_words = len(transcription_result.words)
        logger.log(f"Transcription: {nb_segments} segments, {nb_words} mots", "success", "transcribe", 100)

        # === ÉTAPE 3: Auto-configuration ===
        logger.log("Détection du type de contenu...", "info", "detect", 0)

        configurator = AutoConfigurator()

        with VideoFileClip(video_path) as video:
            total_duration = video.duration

        # Callback pour la progression de détection
        def detection_progress(percent: int, message: str):
            logger.log(message, "info", "detect", percent)

        config, analysis = configurator.analyze_and_configure(
            video_path,
            platform='reels',
            total_duration=total_duration,
            transcription_result=transcription_result,
            progress_callback=detection_progress
        )

        confidence_pct = analysis.content_confidence * 100
        logger.log(f"Configuration optimale générée", "success", "detect", 100)

        # === RÉSULTATS ===
        result_data = {
            "success": True,
            "transcription": {
                "word_count": nb_words,
                "segment_count": nb_segments,
                "result": transcription_result  # Sauvegarder l'objet complet pour réutilisation
            },
            "content_type": analysis.content_type.value,
            "confidence": analysis.content_confidence,
            "config": {
                "min_duration": config.min_duration,
                "max_duration": config.max_duration,
                "color_grading": config.color_grading,
                "zoom_style": config.zoom_style
            },
            "video_path": video_path
        }

        jobs[job_id] = {"status": "completed", "data": result_data, "error": None, "_created": time.time()}
        logger.log("Analyse terminée!", "success", "complete", 100)

        # === NETTOYAGE - DÉSACTIVÉ pour la vidéo source ===
        # La vidéo doit rester disponible pour la génération de clips!
        cleanup_residual_files(
            video_path=None,  # NE PAS supprimer la vidéo téléchargée
            output_dir="output",
            keep_user_files=is_local
        )

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"\n❌ ERREUR ANALYSE:\n{traceback_str}")
        logger.log(f"Erreur: {error_msg}", "error", "error", 0)
        jobs[job_id] = {"status": "failed", "data": None, "error": error_msg, "_created": time.time()}

    finally:
        # === NETTOYAGE FINAL (même en cas d'erreur) ===
        # ⚠️ NE PAS supprimer la vidéo ici car elle sera réutilisée pour la génération!
        # La vidéo sera supprimée après la génération dans process_video()
        # On nettoie seulement les fichiers temporaires, pas la vidéo source
        cleanup_residual_files(
            video_path=None,  # Ne pas supprimer la vidéo téléchargée
            output_dir="output",
            keep_user_files=is_local
        )

        # Attendre 2 secondes avant de fermer pour laisser le temps au frontend de recevoir le dernier message
        time.sleep(2)
        logger.close()
