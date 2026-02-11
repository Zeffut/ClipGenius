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
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Generator

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from cleanup import cleanup_residual_files
from processing import process_video, analyze_video

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

# Constantes de configuration
SOCKETIO_PING_TIMEOUT = 120          # 2 minutes avant timeout
SOCKETIO_PING_INTERVAL = 25         # Ping toutes les 25 secondes
SOCKETIO_MAX_BUFFER_MB = 10         # Taille max des messages HTTP en Mo
HEARTBEAT_DEFAULT_PROGRESS = 50     # Progression par défaut du heartbeat
DEFAULT_MAX_CLIPS = 5               # Nombre max de clips par défaut

app = Flask(__name__, template_folder='web/templates')
app.config['SECRET_KEY'] = os.urandom(24).hex()

# === SOCKET.IO CONFIGURATION ===
# async_mode='threading' pour compatibilité avec les threads de traitement
# ping_timeout/ping_interval élevés pour les longues opérations (transcription, génération)
socketio = SocketIO(
    app,
    cors_allowed_origins=["http://127.0.0.1:5001", "http://localhost:5001"],
    async_mode='threading',
    ping_timeout=SOCKETIO_PING_TIMEOUT,
    ping_interval=SOCKETIO_PING_INTERVAL,
    max_http_buffer_size=SOCKETIO_MAX_BUFFER_MB * 1024 * 1024
)

# État global des jobs
jobs = {}
_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 3600  # 1 heure


def _cleanup_old_jobs():
    """Supprime les jobs expirés (plus vieux que JOB_TTL_SECONDS)"""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, job in jobs.items()
                   if now - job.get('_created', 0) > JOB_TTL_SECONDS]
        for jid in expired:
            del jobs[jid]


class LogCapture:
    """Capture les logs et les envoie via WebSocket en temps réel"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.buffer = []  # Buffer pour historique (reconnexion)

    def log(self, message: str, level: str = "info", step: Optional[str] = None, progress: Optional[int] = None):
        """Envoie un message de log via WebSocket"""
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
            "step": step,
            "progress": progress,
            "job_id": self.job_id
        }
        self.buffer.append(data)

        # Émettre via WebSocket vers la room du job
        socketio.emit('progress', data, room=self.job_id)

    def heartbeat(self, step: str = "generate", progress: int = HEARTBEAT_DEFAULT_PROGRESS):
        """Envoie un heartbeat pour maintenir la connexion active"""
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": "heartbeat",
            "message": "Traitement en cours...",
            "step": step,
            "progress": progress,
            "job_id": self.job_id
        }
        socketio.emit('progress', data, room=self.job_id)

    def close(self):
        """Marque la fin du job (le buffer reste pour reconnexion)"""
        pass  # Buffer conservé pour reconnexion éventuelle


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
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())

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
        'analysis_job_id': data.get('analysis_job_id'),  # Pour réutiliser la transcription
        'local_file': video_path if video_path else None,  # Si vidéo déjà téléchargée
        'skip_download': data.get('skip_download', False),  # Flag pour skip download
    }

    # Lancer le traitement en arrière-plan
    thread = threading.Thread(target=process_video, args=(job_id, url, options, jobs, _jobs_lock))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


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
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())

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
        'analysis_job_id': data.get('analysis_job_id'),  # Pour réutiliser la transcription de l'analyse
    }

    # Lancer le traitement en arrière-plan
    thread = threading.Thread(target=process_video, args=(job_id, None, options, jobs, _jobs_lock))
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
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())

    # Lancer l'analyse en arrière-plan
    thread = threading.Thread(target=analyze_video, args=(job_id, url, False, jobs, _jobs_lock))
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
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())

    # Lancer l'analyse en arrière-plan
    thread = threading.Thread(target=analyze_video, args=(job_id, file_path, True, jobs, _jobs_lock))
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
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())

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
    thread = threading.Thread(target=process_video, args=(job_id, None, options, jobs, _jobs_lock))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


# === SOCKET.IO EVENT HANDLERS ===

@socketio.on('connect')
def handle_connect():
    """Gère la connexion d'un client WebSocket"""
    print(f"[WebSocket] Client connecté: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Gère la déconnexion d'un client WebSocket"""
    print(f"[WebSocket] Client déconnecté: {request.sid}")

@socketio.on('join_job')
def handle_join_job(data):
    """Rejoint la room d'un job pour recevoir les mises à jour de progression"""
    job_id = data.get('job_id')
    if job_id:
        join_room(job_id)
        print(f"[WebSocket] Client {request.sid} a rejoint le job {job_id}")

        # Si le job existe déjà, envoyer l'historique des logs
        if job_id in jobs:
            job = jobs[job_id]
            # Envoyer le statut actuel
            emit('job_status', {
                'job_id': job_id,
                'status': job.get('status'),
                'clips': job.get('clips', []),
                'error': job.get('error')
            })

@socketio.on('leave_job')
def handle_leave_job(data):
    """Quitte la room d'un job"""
    job_id = data.get('job_id')
    if job_id:
        leave_room(job_id)
        print(f"[WebSocket] Client {request.sid} a quitté le job {job_id}")


@app.route('/api/status/<job_id>')
def get_status(job_id: str):
    """Retourne le statut d'un job"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    result = dict(jobs[job_id])
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
    file_path = output_dir / filename

    # Log pour tracer les requêtes
    if file_path.exists():
        print(f"✅ [serve_output] Fichier trouvé: {filename} ({file_path.stat().st_size} bytes)")
    else:
        print(f"❌ [serve_output] Fichier INTROUVABLE: {filename} (chemin: {file_path})")

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


@app.route('/clips/<path:filename>')
def serve_clips(filename: str):
    """Sert les clips finaux depuis ~/Downloads (~/Téléchargements sur Mac FR)"""
    # Validation du nom de fichier pour éviter les traversées de chemin
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return "Forbidden", 403

    # N'autoriser que les fichiers vidéo
    if not filename.lower().endswith(('.mp4', '.webm', '.mov')):
        return "Forbidden", 403

    # Déterminer le dossier Downloads selon l'OS et la langue
    downloads_dir = Path.home() / 'Downloads'

    # Sur Mac avec locale française, le dossier peut être "Téléchargements"
    if not downloads_dir.exists():
        downloads_dir = Path.home() / 'Téléchargements'

    # Déterminer le type MIME
    mimetype = None
    if filename.lower().endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.lower().endswith('.webm'):
        mimetype = 'video/webm'
    elif filename.lower().endswith('.mov'):
        mimetype = 'video/quicktime'

    # Envoyer le fichier avec support des range requests (nécessaire pour les vidéos)
    response = send_from_directory(downloads_dir, filename, mimetype=mimetype)
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


# Créer le dossier templates
templates_dir = Path(__file__).parent / 'web' / 'templates'
templates_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # Créer le dossier output
    output_path = Path('output')
    output_path.mkdir(exist_ok=True)

    # === NETTOYAGE AU DÉMARRAGE ===
    # Supprimer tous les clips résiduels de la session précédente
    print("\n🧹 Nettoyage des fichiers résiduels...")
    cleanup_residual_files(video_path=None, output_dir="output", keep_user_files=False)

    # === DÉSACTIVÉ: Ne plus supprimer downloads/ au démarrage ===
    # La vidéo doit rester disponible entre l'analyse et la génération
    # downloads_path = Path('downloads')
    # if downloads_path.exists():
    #     for video_file in downloads_path.glob('*.mp4'):
    #         try:
    #             video_file.unlink()
    #             print(f"  🗑️ Supprimé: {video_file.name}")
    #         except:
    #             pass
    print("  ⏭️ downloads/ préservé (nécessaire pour génération)")

    print("✅ Nettoyage terminé\n")

    print("="*50)
    print("  ClipGenius - Interface Web")
    print("  http://localhost:5001")
    print("="*50 + "\n")

    # Utiliser socketio.run() pour supporter WebSocket
    socketio.run(app, debug=True, host='127.0.0.1', port=5001)
