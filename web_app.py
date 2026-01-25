#!/usr/bin/env python3
"""
ClipGenius - Interface Web
Interface de test pour visualiser le processus de génération de clips
"""

import os
import sys

# Supprimer les warnings MediaPipe/Abseil AVANT tout import
os.environ['GLOG_minloglevel'] = '2'  # Désactive les logs INFO et WARNING de MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Désactive les warnings TensorFlow

import json
import queue
import threading
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Generator

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent))

# === LAZY IMPORTS (chargés à la demande pour démarrage rapide) ===
# Ces imports lourds sont déplacés dans les fonctions qui les utilisent:
# - src.downloader (VideoDownloader)
# - src.viral_detector (ViralMoment, ViralMomentDetector)
# - src.smart_cropper (SmartCropper)
# - src.clip_generator (ClipGenerator, ClipConfig)
# - src.subtitles (SubtitleGenerator, TranscriptionResult)
# - src.ai_analyzer (TranscriptSegment, ViralMomentAI, analyze_with_ai)
# - src.auto_config (AutoConfigurator, GeneratedConfig)

app = Flask(__name__, template_folder='web/templates')
app.config['SECRET_KEY'] = 'clipgenius-dev-key'

# Queue globale pour les messages de log
log_queues = {}

# Buffer pour l'historique des logs (pour reconnexion)
log_buffers = {}

# État global des jobs
jobs = {}


class LogCapture:
    """Capture les logs et les envoie à la queue SSE + buffer pour reconnexion"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.queue = queue.Queue()
        self.buffer = []
        log_queues[job_id] = self.queue
        log_buffers[job_id] = self.buffer

    def log(self, message: str, level: str = "info", step: Optional[str] = None, progress: Optional[int] = None):
        """Envoie un message de log"""
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
            "step": step,
            "progress": progress
        }
        self.buffer.append(data)
        self.queue.put(data)

    def close(self):
        """Ferme la queue (garde le buffer pour reconnexion)"""
        if self.job_id in log_queues:
            del log_queues[self.job_id]


def process_video(job_id: str, url: str, options: dict):
    """Traite une vidéo YouTube ou locale en arrière-plan"""
    
    # === LAZY IMPORTS (chargés ici pour démarrage rapide de Flask) ===
    from src.downloader import VideoDownloader
    from src.viral_detector import ViralMoment, ViralMomentDetector
    from src.clip_generator import ClipGenerator, ClipConfig
    from src.subtitles import SubtitleGenerator
    from src.ai_analyzer import TranscriptSegment, analyze_with_ai
    from src.auto_config import AutoConfigurator, GeneratedConfig
    
    logger = LogCapture(job_id)
    jobs[job_id] = {"status": "running", "clips": [], "error": None}
    
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
                        max_clips=max_clips or 10,
                        video_path=video_path,
                        progress_callback=ai_progress_callback
                    )

                    logger.log(f"IA a retourné {len(ai_moments)} moments", "info", "analyze", 95)

                    # Convertir tous les moments
                    all_moments = [
                        ViralMoment(
                            start_time=ai_m.start_time,
                            end_time=ai_m.end_time,
                            score=ai_m.score,
                            reason=f"[{ai_m.emotion}] {ai_m.reason}"
                        )
                        for ai_m in ai_moments
                    ]

                    # Filtrer par score minimum
                    moments = [m for m in all_moments if m.score >= min_score]

                    if moments:
                        logger.log(f"{len(moments)} moments retenus (score >= {min_score})", "info", "analyze", 98)
                    elif all_moments:
                        # Aucun ne passe le filtre → prendre le meilleur quand même
                        best = max(all_moments, key=lambda m: m.score)
                        moments = [best]
                        logger.log(f"Aucun moment >= {min_score}, meilleur: {best.score:.0%}", "warning", "analyze", 98)
                    else:
                        logger.log("L'IA n'a trouvé aucun moment", "warning", "analyze", 98)

            except Exception as e:
                logger.log(f"Erreur analyse IA: {e}", "error", "analyze", 90)
                moments = []

        # === FALLBACK: Analyse audio/vidéo si l'IA n'a rien trouvé ===
        if not moments:
            logger.log("Fallback: Analyse audio/vidéo...", "info", "analyze", 75)
            try:
                detector = ViralMomentDetector(
                    min_clip_duration=min_duration,
                    max_clip_duration=max_duration,
                    min_viral_score=min_score,
                    max_clips=max_clips
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
            error_msg = f"Aucun moment viral détecté (score minimum: {min_score}, durée: {min_duration}-{max_duration}s). Essayez de réduire le score minimum."
            logger.log(error_msg, "error", "error", 0)
            jobs[job_id] = {"status": "failed", "clips": [], "error": error_msg}
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
        total_clips = len(moments)
        logger.log(f"Génération de {total_clips} clip(s) vertical(s)...", "info", "generate", 0)
        
        output_dir = options.get('output_dir', 'output')

        # Créer la config avec les paramètres auto-générés si disponibles
        if auto_generated_config:
            config = ClipConfig(
                min_clip_duration=min_duration,
                max_clip_duration=max_duration,
                min_viral_score=min_score,
                max_clips=max_clips,
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
                max_clips=max_clips,
                add_subtitles=True,
                enable_zoom_effect=True
            )
        
        generator = ClipGenerator(config, transcription_result=transcription_result)
        
        logger.log("Initialisation du générateur...", "info", "generate", 5)
        
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
        
        if not clips:
            raise Exception("Aucun clip généré")
        
        logger.log(f"{len(clips)} clips générés avec succès", "success", "generate", 100)
        
        # === ÉTAPE 4: Sous-titres (maintenant intégrés dans la génération) ===
        # Les sous-titres sont ajoutés directement pendant la génération des clips
        # via le système ASS/FFmpeg dans ClipGenerator._generate_single_clip()
        if options.get('subtitles', True):
            logger.log("Sous-titres ajoutés pendant la génération", "success", "subtitles", 100)
        else:
            logger.log("Sous-titres désactivés", "info", "subtitles", 100)
        
        # === TERMINÉ ===
        logger.log("Traitement terminé avec succès!", "success", "complete", 100)
        
        # === COPIE AUTOMATIQUE VERS DOSSIER TÉLÉCHARGEMENTS ===
        try:
            import shutil
            from pathlib import Path as PathLib
            
            # Détecter le dossier Téléchargements de l'utilisateur
            home = PathLib.home()
            downloads_dir = home / "Downloads"
            
            logger.log(f"Copie des clips vers {downloads_dir}...", "info", "complete", 100)
            
            copied_count = 0
            for clip_path in clips:
                src = Path(clip_path)
                dst = downloads_dir / src.name
                
                if src.exists():
                    shutil.copy2(src, dst)
                    copied_count += 1
                    logger.log(f"✓ {src.name} → Téléchargements", "success", "complete", 100)
            
            logger.log(f"{copied_count} clips copiés dans Téléchargements", "success", "complete", 100)
            
        except Exception as e:
            logger.log(f"Avertissement: impossible de copier vers Téléchargements: {e}", "warning", "complete", 100)
        
        # Convertir les chemins en URLs relatives + extraire métadonnées
        clip_data = []
        for i, clip_path in enumerate(clips):
            try:
                clip_file = Path(clip_path)
                
                # Vérifier que le fichier existe avant d'extraire les métadonnées
                if not clip_file.exists():
                    logger.log(f"⚠️ Clip introuvable: {clip_path}", "warning", "complete", 100)
                    # Utiliser le chemin absolu si relatif
                    clip_file = Path(__file__).parent / clip_path
                    if not clip_file.exists():
                        logger.log(f"⚠️ Clip introuvable (absolu): {clip_file}", "warning", "complete", 100)
                        continue
                
                # Extraire les métadonnées du fichier
                metadata = {
                    "url": f"/output/{clip_file.name}",
                    "name": clip_file.name,
                    "index": i,
                    "size": round(clip_file.stat().st_size / (1024 * 1024), 1),  # MB
                }
                
                # Extraire durée et score à partir du nom de fichier ou via ffprobe
                try:
                    from moviepy import VideoFileClip
                    with VideoFileClip(str(clip_file)) as vc:
                        metadata["duration"] = round(vc.duration, 1)
                except Exception as e:
                    logger.log(f"⚠️ Impossible d'extraire la durée de {clip_file.name}: {e}", "warning", "complete", 100)
                    metadata["duration"] = 60.0  # Valeur par défaut
                
                # Score estimé - les premiers clips ont les meilleurs scores
                # (décroît progressivement de 0.95 à 0.80)
                # TODO: extraire le vrai score depuis les moments viraux
                score_range = 0.15  # 0.95 - 0.80 = 0.15
                score_decrement = score_range / max(len(clips), 1)
                metadata["score"] = round(0.95 - (i * score_decrement), 2)
                
                clip_data.append(metadata)
                logger.log(f"✓ Métadonnées extraites pour {clip_file.name}", "info", "complete", 100)
                
            except Exception as e:
                logger.log(f"❌ Erreur extraction métadonnées pour {clip_path}: {e}", "error", "complete", 100)
                # Ajouter quand même un objet minimal
                clip_data.append({
                    "url": f"/output/{Path(clip_path).name}",
                    "name": Path(clip_path).name,
                    "index": i,
                    "size": 5.0,
                    "duration": 60.0,
                    "score": 0.85
                })
        
        logger.log(f"📊 {len(clip_data)} clips prêts avec métadonnées", "success", "complete", 100)
        jobs[job_id] = {"status": "completed", "clips": clip_data, "error": None}
        
        # Nettoyage des fichiers sources (sauf fichiers utilisateur)
        is_user_file = options.get('is_user_file', False)
        if (is_downloaded or is_uploaded) and not is_user_file and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.log("Fichier source nettoyé", "info", "complete", 100)
            except:
                pass
        
        # Nettoyer les fichiers temp
        for pattern in ['*TEMP_MPY_*.mp4', '*_sub.json']:
            for f in Path('.').glob(pattern):
                try:
                    f.unlink()
                except:
                    pass
            for f in Path(output_dir).glob(pattern.replace('*TEMP', '*')):
                try:
                    f.unlink()
                except:
                    pass
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.log(f"ERREUR: {error_msg}", "error", "error", 0)
        logger.log(traceback.format_exc(), "error", "error", 0)
        jobs[job_id] = {"status": "failed", "clips": [], "error": error_msg}
    
    finally:
        # Attendre 2 secondes avant de fermer pour laisser le temps au frontend de recevoir le dernier message
        time.sleep(2)
        logger.close()


@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def start_process():
    """Démarre le traitement d'une vidéo"""
    data = request.json
    url = data.get('url', '').strip()
    video_path = data.get('video_path', '').strip()
    
    # Accepter soit une URL YouTube, soit un video_path pré-téléchargé
    if not url and not video_path:
        return jsonify({"error": "URL ou video_path requis"}), 400
    
    if url and 'youtube.com' not in url and 'youtu.be' not in url:
        return jsonify({"error": "URL YouTube invalide"}), 400
    
    # Générer un ID unique
    job_id = f"job_{int(time.time() * 1000)}"
    
    # Options
    options = {
        'quality': data.get('quality', '1080p'),
        'min_score': float(data.get('min_score', 0.80)),
        'min_duration': float(data.get('min_duration', 60)),
        'max_duration': float(data.get('max_duration', 90)),
        'max_clips': int(data.get('max_clips')) if data.get('max_clips') else None,
        'subtitles': data.get('subtitles', True),
        'emojis': data.get('emojis', True),
        'max_words': int(data.get('max_words', 3)),
        'whisper_model': data.get('whisper_model', 'turbo'),
        'use_ai': data.get('use_ai', True),
        'language': data.get('language', None),
        'output_dir': 'output',
        'auto_config': data.get('auto_config', False),
        'auto_config_verbose': data.get('auto_config_verbose', False),
        'platform': data.get('platform', 'reels'),
        'analysis_job_id': data.get('analysis_job_id'),  # ✨ Pour réutiliser la transcription
        'local_file': video_path if video_path else None,  # Si vidéo déjà téléchargée
        'skip_download': data.get('skip_download', False),  # Flag pour skip download
    }
    
    # Lancer le traitement en arrière-plan
    thread = threading.Thread(target=process_video, args=(job_id, url, options))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})


def analyze_video(job_id: str, url_or_path: str, is_local: bool):
    """Analyse une vidéo (téléchargement + transcription + auto-config uniquement)"""
    
    # === LAZY IMPORTS ===
    from src.downloader import VideoDownloader
    from src.subtitles import SubtitleGenerator
    from src.auto_config import AutoConfigurator
    from moviepy import VideoFileClip
    
    logger = LogCapture(job_id)
    jobs[job_id] = {"status": "running", "data": None, "error": None}
    
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
        
        jobs[job_id] = {"status": "completed", "data": result_data, "error": None}
        logger.log("Analyse terminée!", "success", "complete", 100)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"\n❌ ERREUR ANALYSE:\n{traceback_str}")
        logger.log(f"Erreur: {error_msg}", "error", "error", 0)
        jobs[job_id] = {"status": "failed", "data": None, "error": error_msg}
    
    finally:
        # Attendre 2 secondes avant de fermer pour laisser le temps au frontend de recevoir le dernier message
        time.sleep(2)
        logger.close()


@app.route('/api/process-local', methods=['POST'])
def start_process_local():
    """Démarre le traitement d'un fichier vidéo local (chemin direct)"""
    data = request.json
    file_path = data.get('file_path', '').strip()
    
    if not file_path:
        return jsonify({"error": "Chemin du fichier requis"}), 400
    
    # Vérifier que le fichier existe
    if not Path(file_path).exists():
        return jsonify({"error": "Fichier introuvable"}), 400
    
    # Vérifier l'extension
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    ext = Path(file_path).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Format non supporté. Utilisez: {', '.join(allowed_extensions)}"}), 400
    
    # Générer un ID unique
    job_id = f"job_{int(time.time() * 1000)}"
    
    # Options
    options = {
        'quality': data.get('quality', '1080p'),
        'min_score': float(data.get('min_score', 0.80)),
        'min_duration': float(data.get('min_duration', 60)),
        'max_duration': float(data.get('max_duration', 90)),
        'max_clips': int(data.get('max_clips')) if data.get('max_clips') else None,
        'subtitles': data.get('subtitles', True),
        'emojis': data.get('emojis', True),
        'max_words': int(data.get('max_words', 3)),
        'whisper_model': data.get('whisper_model', 'turbo'),
        'use_ai': data.get('use_ai', True),
        'language': data.get('language', None),
        'output_dir': 'output',
        'auto_config': data.get('auto_config', False),
        'auto_config_verbose': data.get('auto_config_verbose', False),
        'platform': data.get('platform', 'reels'),
        'local_file': file_path,  # Chemin direct du fichier
        'is_user_file': True,  # Flag pour ne pas supprimer le fichier après traitement
        'analysis_job_id': data.get('analysis_job_id'),  # ✨ Pour réutiliser la transcription de l'analyse
    }
    
    # Lancer le traitement en arrière-plan
    thread = threading.Thread(target=process_video, args=(job_id, None, options))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})


@app.route('/api/analyze-local', methods=['POST'])
def analyze_local():
    """Analyse une vidéo locale (transcription + auto-config uniquement)"""
    data = request.json
    file_path = data.get('file_path', '').strip()
    
    if not file_path or not Path(file_path).exists():
        return jsonify({"error": "Fichier introuvable"}), 400
    
    try:
        # === LAZY IMPORTS ===
        from src.subtitles import SubtitleGenerator
        from src.auto_config import AutoConfigurator
        
        # 1. Transcription
        subtitle_gen = SubtitleGenerator(model_size='base', language=None)
        transcription_result = subtitle_gen.transcribe_with_words(file_path)
        
        # 2. Auto-config avec transcription
        configurator = AutoConfigurator()
        
        # Extraire premiers 180s de texte
        words_180s = [w for w in transcription_result.words if w.start <= 180.0]
        transcription_text = " ".join([w.word for w in words_180s])
        
        from moviepy import VideoFileClip
        with VideoFileClip(file_path) as video:
            total_duration = video.duration
        
        config, analysis = configurator.analyze_and_configure(
            file_path,
            platform='reels',
            total_duration=total_duration,
            transcription_result=transcription_result
        )
        
        # Retourner les résultats
        return jsonify({
            "success": True,
            "transcription": {
                "word_count": len(transcription_result.words),
                "segment_count": len(transcription_result.segments)
            },
            "content_type": analysis.content_type.value,
            "confidence": analysis.content_confidence,
            "config": {
                "min_duration": config.min_duration,
                "max_duration": config.max_duration,
                "color_grading": config.color_grading,
                "zoom_style": config.zoom_style
            },
            "video_path": file_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_youtube():
    """Analyse une vidéo YouTube (transcription + auto-config uniquement)"""
    data = request.json
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({"error": "URL YouTube requise"}), 400
    
    try:
        # === LAZY IMPORTS ===
        from src.downloader import VideoDownloader
        from src.subtitles import SubtitleGenerator
        from src.auto_config import AutoConfigurator
        
        # 1. Télécharger
        downloader = VideoDownloader()
        video_path = downloader.download(url, quality='1080p', output_dir='output')
        
        # 2. Transcription
        subtitle_gen = SubtitleGenerator(model_size='base', language=None)
        transcription_result = subtitle_gen.transcribe_with_words(video_path)
        
        # 3. Auto-config avec transcription
        configurator = AutoConfigurator()
        
        # Extraire premiers 180s de texte
        words_180s = [w for w in transcription_result.words if w.start <= 180.0]
        transcription_text = " ".join([w.word for w in words_180s])
        
        from moviepy import VideoFileClip
        with VideoFileClip(video_path) as video:
            total_duration = video.duration
        
        config, analysis = configurator.analyze_and_configure(
            video_path,
            platform='reels',
            total_duration=total_duration,
            transcription_result=transcription_result
        )
        
        # Retourner les résultats
        return jsonify({
            "success": True,
            "transcription": {
                "word_count": len(transcription_result.words),
                "segment_count": len(transcription_result.segments)
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
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze-stream-youtube', methods=['POST'])
def start_analyze_youtube():
    """Démarre l'analyse d'une vidéo YouTube avec SSE"""
    data = request.json
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({"error": "URL YouTube requise"}), 400
    
    # Générer un ID unique
    job_id = f"analyze_{int(time.time() * 1000)}"
    
    # Lancer l'analyse en arrière-plan
    thread = threading.Thread(target=analyze_video, args=(job_id, url, False))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})


@app.route('/api/analyze-stream-local', methods=['POST'])
def start_analyze_local():
    """Démarre l'analyse d'un fichier local avec SSE"""
    data = request.json
    file_path = data.get('file_path', '').strip()
    
    if not file_path:
        return jsonify({"error": "Chemin du fichier requis"}), 400
    
    if not Path(file_path).exists():
        return jsonify({"error": "Fichier introuvable"}), 400
    
    # Générer un ID unique
    job_id = f"analyze_{int(time.time() * 1000)}"
    
    # Lancer l'analyse en arrière-plan
    thread = threading.Thread(target=analyze_video, args=(job_id, file_path, True))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})


@app.route('/api/analyze-result/<job_id>')
def get_analyze_result(job_id: str):
    """Récupère le résultat final de l'analyse"""
    if job_id not in jobs:
        return jsonify({"error": "Job non trouvé"}), 404
    
    job = jobs[job_id]
    
    if job["status"] == "running":
        return jsonify({"status": "running"}), 202
    elif job["status"] == "failed":
        return jsonify({"error": job["error"]}), 500
    else:
        return jsonify(job["data"])


@app.route('/api/process-file', methods=['POST'])
def start_process_file():
    """Démarre le traitement d'un fichier vidéo uploadé"""
    if 'video' not in request.files:
        return jsonify({"error": "Fichier vidéo requis"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    # Vérifier l'extension
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    ext = Path(video_file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Format non supporté. Utilisez: {', '.join(allowed_extensions)}"}), 400
    
    # Sauvegarder le fichier temporairement
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)
    
    # Nom unique pour éviter les conflits
    timestamp = int(time.time() * 1000)
    safe_filename = f"{timestamp}_{video_file.filename}"
    video_path = upload_dir / safe_filename
    video_file.save(str(video_path))
    
    # Générer un ID unique
    job_id = f"job_{timestamp}"
    
    # Options depuis FormData
    options = {
        'quality': request.form.get('quality', '1080p'),
        'min_score': float(request.form.get('min_score', 0.80)),
        'min_duration': float(request.form.get('min_duration', 60)),
        'max_duration': float(request.form.get('max_duration', 90)),
        'max_clips': int(request.form.get('max_clips')) if request.form.get('max_clips') else None,
        'subtitles': request.form.get('subtitles', 'true').lower() == 'true',
        'emojis': request.form.get('emojis', 'true').lower() == 'true',
        'max_words': int(request.form.get('max_words', 3)),
        'whisper_model': request.form.get('whisper_model', 'turbo'),
        'use_ai': request.form.get('use_ai', 'true').lower() == 'true',
        'language': request.form.get('language', None),
        'output_dir': 'output',
        'auto_config': request.form.get('auto_config', 'false').lower() == 'true',
        'auto_config_verbose': request.form.get('auto_config_verbose', 'false').lower() == 'true',
        'platform': request.form.get('platform', 'reels'),
        'local_file': str(video_path),  # Chemin du fichier uploadé
    }
    
    # Lancer le traitement en arrière-plan
    thread = threading.Thread(target=process_video, args=(job_id, None, options))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})


@app.route('/api/stream/<job_id>')
def stream_logs(job_id: str):
    """Stream SSE des logs en temps réel avec support reconnexion"""
    # Récupérer l'index de départ depuis le header (pour éviter les doublons)
    last_index = request.args.get('from', 0, type=int)

    def generate() -> Generator[str, None, None]:
        nonlocal last_index

        # Attendre que la queue soit créée ou vérifier si le job existe dans le buffer
        timeout = 10
        start = time.time()
        while job_id not in log_queues and job_id not in log_buffers and time.time() - start < timeout:
            time.sleep(0.1)

        # Si le job n'existe pas du tout
        if job_id not in log_queues and job_id not in log_buffers:
            yield f"data: {json.dumps({'level': 'error', 'message': 'Job non trouvé'})}\n\n"
            return

        # Si le job existe dans le buffer (reconnexion après fermeture)
        if job_id in log_buffers:
            buffer = log_buffers[job_id]
            # Envoyer les messages non encore reçus
            for i, data in enumerate(buffer[last_index:], start=last_index):
                yield f"data: {json.dumps(data)}\n\n"
                last_index = i + 1

            # Si le job est terminé, arrêter
            if buffer and buffer[-1].get('step') in ['complete', 'error']:
                return

        # Si la queue n'existe plus (job terminé), arrêter
        if job_id not in log_queues:
            return

        q = log_queues[job_id]

        # Connection timeout: 10 minutes max (pour les longues vidéos)
        connection_start = time.time()
        max_connection_time = 600  # 10 minutes
        
        # Heartbeat counter
        heartbeat_count = 0

        while True:
            try:
                # Check connection timeout
                if time.time() - connection_start > max_connection_time:
                    yield f"data: {json.dumps({'level': 'info', 'message': 'Connection timeout - reconnect'})}\n\n"
                    break

                # Attendre un message avec timeout de 15 secondes
                data = q.get(timeout=15)
                
                # Limit message size to prevent buffer overflow
                message_str = json.dumps(data)
                if len(message_str) > 8192:  # 8KB max per message
                    data['message'] = data.get('message', '')[:1000] + '... (truncated)'
                    message_str = json.dumps(data)
                
                yield f"data: {message_str}\n\n"

                # Si c'est la fin, arrêter
                if data.get('step') in ['complete', 'error']:
                    break

            except queue.Empty:
                # Envoyer un heartbeat régulier (toutes les 15 secondes)
                heartbeat_count += 1
                yield f"data: {json.dumps({'level': 'heartbeat', 'count': heartbeat_count})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )


@app.route('/api/status/<job_id>')
def get_status(job_id: str):
    """Retourne le statut d'un job avec infos pour reconnexion"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    result = dict(jobs[job_id])

    # Ajouter les infos du buffer si disponibles
    if job_id in log_buffers:
        buffer = log_buffers[job_id]
        result['log_count'] = len(buffer)
        if buffer:
            last_log = buffer[-1]
            result['last_step'] = last_log.get('step')
            result['last_progress'] = last_log.get('progress')

    return jsonify(result)



@app.route('/api/youtube/duration/<video_id>')
def get_youtube_duration(video_id: str):
    """Récupère la durée d'une vidéo YouTube via yt-dlp"""
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            
            if info and 'duration' in info:
                total_seconds = int(info['duration'])
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                hours = minutes // 60
                minutes = minutes % 60
                
                if hours > 0:
                    duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = f"{minutes}:{seconds:02d}"
                
                return jsonify({
                    'success': True,
                    'duration': duration_str,
                    'total_seconds': total_seconds
                })
        
        return jsonify({'success': False, 'error': 'Duration not found'})
        
    except Exception as e:
        print(f"Error fetching YouTube duration: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/output/<path:filename>')
def serve_output(filename: str):
    """Sert les fichiers de sortie avec le bon type MIME pour les vidéos"""
    # Chemin absolu vers le dossier output
    output_dir = Path(__file__).parent / 'output'
    
    # Déterminer le type MIME
    mimetype = None
    if filename.lower().endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.lower().endswith('.webm'):
        mimetype = 'video/webm'
    elif filename.lower().endswith('.mov'):
        mimetype = 'video/quicktime'
    
    # Envoyer le fichier avec support des range requests (nécessaire pour les vidéos)
    response = send_from_directory(output_dir, filename, mimetype=mimetype)
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


# Créer le dossier templates
templates_dir = Path(__file__).parent / 'web' / 'templates'
templates_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # Créer le dossier output
    Path('output').mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("  ClipGenius - Interface Web")
    print("  http://localhost:5001")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
