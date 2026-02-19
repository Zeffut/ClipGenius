#!/usr/bin/env python3
"""
ClipGenius - Application principale
Application native avec interface web embarquée

Usage:
    python app.py
"""

import sys
import os

# Supprimer TOUS les warnings avant tout import
os.environ["PYTHONWARNINGS"] = "ignore"
# Silence MediaPipe/Abseil C++ threads (GLOG_minloglevel=3 = FATAL uniquement)
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import warnings
warnings.filterwarnings("ignore")

import threading
import time
import urllib.request
import subprocess
import signal
from pathlib import Path
from typing import Optional

# Ajouter le dossier du projet au path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Constantes de configuration
DOWNLOAD_TIMEOUT_SECONDS = 30
SPEED_SMOOTHING_ALPHA_MAX = 0.3
SPEED_SMOOTHING_ALPHA_MIN = 0.05
SPEED_SMOOTHING_DIFF_FACTOR = 0.5
NOTIFY_INTERVAL_SECONDS = 0.25

# Serveur Flask embarque
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5001

MODELS_DIR = "models"
# Phi-4-mini-instruct (recommandé, meilleure qualité)
PHI4_MODEL_NAME = "Phi-4-mini-instruct.Q4_K_M.gguf"
PHI4_MODEL_URL = "https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"
PHI4_MODEL_SIZE_MB = 2400
# Fallback: Phi-3-mini (plus petit, moins performant)
PHI3_MODEL_NAME = "Phi-3-mini-4k-instruct-q4.gguf"
PHI3_MODEL_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
PHI3_MODEL_SIZE_MB = 2300


def get_project_root() -> Path:
    """Retourne le dossier racine du projet"""
    return Path(__file__).parent.resolve()


def get_models_dir() -> Path:
    """Retourne le dossier des modèles (le crée si nécessaire)"""
    models_dir = get_project_root() / MODELS_DIR
    models_dir.mkdir(exist_ok=True)
    return models_dir


def find_phi_model() -> Optional[Path]:
    """Cherche le modèle Phi (4 en priorité, puis 3 en fallback) dans les emplacements connus"""
    search_dirs = [
        get_project_root() / MODELS_DIR,
        get_project_root(),
        Path.home() / ".cache" / "clipgenius",
        Path.home() / ".cache" / "huggingface",
    ]
    # Phi-4 en priorité (meilleure qualité)
    phi4_names = [
        "Phi-4-mini-instruct.Q4_K_M.gguf",
        "Phi-4-mini-instruct-Q4_K_M.gguf",
        "phi-4-mini-instruct-Q4_K_M.gguf",
        "phi-4-mini-instruct-q4_k_m.gguf",
        "phi4-mini-instruct-q4.gguf",
    ]
    # Phi-3 en fallback
    phi3_names = [
        "Phi-3-mini-4k-instruct-q4.gguf",
        "phi-3-mini-4k-instruct-q4.gguf",
        "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    ]

    # Chercher Phi-4 d'abord
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in phi4_names:
            model_path = directory / name
            if model_path.exists() and model_path.stat().st_size > 1_000_000_000:
                return model_path

    # Fallback sur Phi-3
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in phi3_names:
            model_path = directory / name
            if model_path.exists() and model_path.stat().st_size > 1_000_000_000:
                return model_path

    return None


# Alias pour compatibilité
def find_phi3_model() -> Optional[Path]:
    """Alias pour find_phi_model (compatibilité)"""
    return find_phi_model()


def download_phi_model(progress_callback=None) -> Optional[Path]:
    """
    Télécharge le modèle Phi-4 (ou Phi-3 en fallback) avec progression.

    Args:
        progress_callback: Fonction appelée avec (downloaded_mb, total_mb, speed_mb_s, eta_text)

    Returns:
        Path vers le modèle téléchargé, ou None si échec
    """
    models_dir = get_models_dir()

    # Essayer Phi-4 d'abord
    model_path = models_dir / PHI4_MODEL_NAME
    temp_path = models_dir / f"{PHI4_MODEL_NAME}.tmp"
    model_url = PHI4_MODEL_URL
    model_size = PHI4_MODEL_SIZE_MB

    # Vérifier si déjà présent
    if model_path.exists() and model_path.stat().st_size > 1_000_000_000:
        return model_path

    try:
        req = urllib.request.Request(model_url)
        req.add_header('User-Agent', 'ClipGenius/beta')

        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            total = int(response.headers.get('Content-Length', model_size * 1024 * 1024))
            total_mb = total / (1024 * 1024)
            downloaded = 0
            last_speed_time = time.time()
            last_speed_bytes = 0
            last_notify_time = time.time()
            current_speed = 0.0

            # Lissage ETA
            smoothed_speed = 0.0
            displayed_eta = None
            speed_samples = []

            with open(temp_path, 'wb') as f:
                while True:
                    chunk = response.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()

                    # Calculer le débit instantané toutes les 0.5s
                    speed_elapsed = now - last_speed_time
                    if speed_elapsed >= 0.5:
                        bytes_since = downloaded - last_speed_bytes
                        current_speed = bytes_since / speed_elapsed / (1024 * 1024)
                        last_speed_time = now
                        last_speed_bytes = downloaded

                        # Lissage de la vitesse pour l'ETA
                        if current_speed > 0:
                            if len(speed_samples) < 5:
                                speed_samples.append(current_speed)
                                smoothed_speed = sum(speed_samples) / len(speed_samples)
                            else:
                                diff_ratio = abs(current_speed - smoothed_speed) / max(smoothed_speed, 0.1)
                                alpha = min(SPEED_SMOOTHING_ALPHA_MAX, max(SPEED_SMOOTHING_ALPHA_MIN, diff_ratio * SPEED_SMOOTHING_DIFF_FACTOR))
                                smoothed_speed = alpha * current_speed + (1 - alpha) * smoothed_speed

                    # Notifier toutes les 250ms
                    notify_elapsed = now - last_notify_time
                    if notify_elapsed >= NOTIFY_INTERVAL_SECONDS and progress_callback:
                        mb = downloaded / (1024 * 1024)

                        # Calculer ETA
                        eta_text = ""
                        if smoothed_speed > 0:
                            remaining_mb = total_mb - mb
                            raw_eta = remaining_mb / smoothed_speed

                            if displayed_eta is None:
                                displayed_eta = raw_eta
                            else:
                                if raw_eta < displayed_eta:
                                    displayed_eta = 0.7 * displayed_eta + 0.3 * raw_eta
                                else:
                                    displayed_eta = 0.95 * displayed_eta + 0.05 * raw_eta

                            eta_seconds = max(0, displayed_eta)
                            if eta_seconds < 60:
                                eta_text = f"{int(eta_seconds)}s"
                            else:
                                eta_text = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"

                        progress_callback(mb, total_mb, current_speed, eta_text)
                        last_notify_time = now

        # Renommer le fichier temporaire
        temp_path.rename(model_path)
        return model_path

    except Exception as e:
        print(f"Erreur de téléchargement: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None


# Alias pour compatibilité
def download_phi3_model(progress_callback=None) -> Optional[Path]:
    """Alias pour download_phi_model (compatibilité)"""
    return download_phi_model(progress_callback)


def ensure_phi_model() -> Optional[Path]:
    """
    S'assure que le modèle Phi-4 (ou Phi-3) est disponible.
    Le télécharge si nécessaire avec progression console.

    Returns:
        Path vers le modèle, ou None si échec
    """
    # Chercher un modèle existant
    model_path = find_phi_model()
    if model_path:
        return model_path

    print("\n" + "="*50)
    print("  Téléchargement du modèle AI (première utilisation)")
    print("="*50 + "\n")
    print(f"  Modèle: {PHI4_MODEL_NAME}")
    print(f"  Taille: ~{PHI4_MODEL_SIZE_MB} MB")
    print()

    def show_progress(downloaded, total, speed, eta):
        pct = int(downloaded / total * 100)
        bar_width = 30
        filled = int(bar_width * downloaded / total)
        bar = "█" * filled + "░" * (bar_width - filled)

        speed_text = f"{speed:.1f} MB/s" if speed > 0 else "..."
        eta_text = f" - {eta} restant" if eta else ""

        print(f"\r  [{bar}] {pct}% - {downloaded:.0f}/{total:.0f} MB - {speed_text}{eta_text}    ", end="", flush=True)

    result = download_phi_model(progress_callback=show_progress)

    if result:
        print("\n\n  ✓ Modèle téléchargé avec succès!\n")
    else:
        print("\n\n  ✗ Échec du téléchargement\n")

    return result


# Alias pour compatibilité
def ensure_phi3_model() -> Optional[Path]:
    """Alias pour ensure_phi_model (compatibilité)"""
    return ensure_phi_model()


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

# Variable globale pour webview (initialisée dans run_main_app)
webview = None

# Event pour signaler l'arrêt de l'application
shutdown_event = threading.Event()

# Process caffeinate (macOS) pour empêcher la mise en veille
caffeinate_process = None


def start_caffeinate():
    """
    Lance caffeinate sur macOS pour empêcher la mise en veille pendant le traitement.
    caffeinate empêche le Mac de dormir pendant que l'app tourne.
    """
    global caffeinate_process
    import platform

    if platform.system() == 'Darwin':  # macOS seulement
        try:
            # Options caffeinate:
            # -d : Empêche l'écran de s'éteindre
            # -i : Empêche le système de dormir pendant les processus actifs
            # -m : Empêche le disque de dormir
            # -s : Empêche la mise en veille même sur batterie
            caffeinate_process = subprocess.Popen(
                ['caffeinate', '-dims'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("☕ Caffeinate activé - Mac ne se mettra pas en veille")
            return True
        except Exception as e:
            print(f"⚠️  Impossible de lancer caffeinate: {e}")
            return False
    return False


def stop_caffeinate():
    """Arrête caffeinate si actif"""
    global caffeinate_process
    if caffeinate_process:
        try:
            caffeinate_process.terminate()
            caffeinate_process.wait(timeout=2)
            print("☕ Caffeinate arrêté")
        except Exception as e:
            print(f"⚠️  Erreur arrêt caffeinate: {e}")
        finally:
            caffeinate_process = None


class Api:
    """API exposée à JavaScript via PyWebView"""

    def __init__(self, window=None):
        self._window = window

    def set_window(self, window):
        self._window = window

    def select_video_file(self):
        """
        Ouvre un dialogue de sélection de fichier vidéo.
        Retourne le chemin complet du fichier ou None si annulé.
        """
        global webview
        if not self._window or not webview:
            return None

        import locale
        lang = locale.getdefaultlocale()[0] or ''
        if lang.startswith('fr'):
            file_types = ('Fichiers vidéo (*.mp4;*.mov;*.avi;*.mkv;*.webm)',)
        else:
            file_types = ('Video files (*.mp4;*.mov;*.avi;*.mkv;*.webm)',)
        result = self._window.create_file_dialog(
            dialog_type=webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=file_types
        )

        if result and len(result) > 0:
            return result[0]
        return None

    def open_output_folder(self):
        """Ouvre le dossier output dans l'explorateur de fichiers"""
        import subprocess
        import platform

        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)

        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open', str(output_dir)])
        elif system == 'Windows':
            subprocess.run(['explorer', str(output_dir)])
        else:  # Linux
            subprocess.run(['xdg-open', str(output_dir)])

        return True

    def get_video_as_base64(self, file_path):
        """Convertit une vidéo en base64 pour l'afficher dans PyWebView"""
        import base64
        try:
            with open(file_path, 'rb') as f:
                video_data = f.read()
                base64_data = base64.b64encode(video_data).decode('utf-8')
                return f"data:video/mp4;base64,{base64_data}"
        except Exception as e:
            print(f"Erreur get_video_as_base64: {e}")
            return None

    def get_file_info(self, file_path):
        """Retourne les informations sur un fichier"""
        try:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                return {
                    'name': path.name,
                    'size': stat.st_size,
                    'path': str(path.absolute())
                }
        except Exception as e:
            print(f"Erreur get_file_info: {e}")
        return None


def run_main_app():
    """Lance l'application principale"""
    global webview

    # Référence module-level pour empêcher le garbage collection de l'activité NSProcessInfo
    _ns_activity = None

    # Supprimer les logs verbeux
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # ⚡ Désactiver App Nap et optimiser les performances (macOS)
    import platform
    if platform.system() == 'Darwin':
        try:
            # Marquer le processus comme prioritaire avec pyobjc
            try:
                from Foundation import NSProcessInfo
                from AppKit import NSActivityUserInitiated, NSActivityLatencyCritical, NSActivityIdleSystemSleepDisabled

                info = NSProcessInfo.processInfo()
                # Combinaison de flags pour performances maximales :
                # - NSActivityUserInitiated (0x00FFFFFF) : Priorité utilisateur haute
                # - NSActivityLatencyCritical (0xFF00000000) : Pas de throttling CPU
                # - NSActivityIdleSystemSleepDisabled (1 << 20) : Empêche App Nap
                activity_options = (
                    NSActivityUserInitiated |
                    NSActivityLatencyCritical |
                    NSActivityIdleSystemSleepDisabled
                )

                # Démarrer une activité en arrière-plan avec ces options
                _ns_activity = info.beginActivityWithOptions_reason_(
                    activity_options,
                    "ClipGenius: Analyse et génération de clips vidéo"
                )
                print("⚡ Optimisations macOS activées (App Nap désactivé)")

            except ImportError:
                # pyobjc pas installé, utiliser des alternatives
                print("⚠️  pyobjc non disponible, performances en arrière-plan potentiellement réduites")
                print("   Installation recommandée: pip install pyobjc-framework-Cocoa")

        except Exception as e:
            print(f"⚠️  Impossible d'optimiser les performances: {e}")

    try:
        import webview as wv
        webview = wv  # Stocker dans la variable globale
        WEBVIEW_AVAILABLE = True
    except ImportError:
        WEBVIEW_AVAILABLE = False

    from dotenv import load_dotenv
    load_dotenv()

    # === NETTOYAGE AU DÉMARRAGE ===
    print("\n🧹 Nettoyage des fichiers résiduels...")

    # Nettoyer output/
    output_path = Path('output')
    output_path.mkdir(exist_ok=True)
    for pattern in ['*.mp4', '*.webm', '*.mov', '*.avi', '*TEMP_MPY_*', '*.ass', '*_sub.json', '.sanitized_*']:
        for f in output_path.glob(pattern):
            try:
                f.unlink()
                print(f"  🗑️ Supprimé: output/{f.name}")
            except OSError:
                pass

    # === DÉSACTIVÉ: Ne plus supprimer downloads/ au démarrage ===
    # La vidéo doit rester disponible entre l'analyse et la génération
    # downloads_path = Path('downloads')
    # if downloads_path.exists():
    #     for video_file in downloads_path.glob('*.mp4'):
    #         try:
    #             video_file.unlink()
    #             print(f"  🗑️ Supprimé: downloads/{video_file.name}")
    #         except OSError:
    #             pass
    print("  ⏭️ downloads/ préservé (nécessaire pour génération)")

    # Nettoyer temp pycaps
    import tempfile
    temp_base = Path(tempfile.gettempdir())
    for pycaps_dir in temp_base.glob('pycaps_viral_*'):
        try:
            import shutil
            shutil.rmtree(pycaps_dir, ignore_errors=True)
            print(f"  🗑️ Supprimé: {pycaps_dir.name}")
        except OSError:
            pass

    print("✅ Nettoyage terminé\n")

    # === LAZY IMPORT: Charger Flask seulement maintenant ===
    print("⏳ Chargement de l'interface...")
    from web_app import app as flask_app
    print("✓ Interface prête!")

    if WEBVIEW_AVAILABLE:
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        # Démarrer Flask en arrière-plan avec thread daemon
        flask_thread = threading.Thread(
            target=lambda: flask_app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True, use_reloader=False),
            daemon=True,
            name="FlaskServer"
        )
        flask_thread.start()
        time.sleep(1)

        # Créer l'API
        api = Api()

        # ☕ Lancer caffeinate pour empêcher la mise en veille
        start_caffeinate()

        # Handler de fermeture pour cleanup propre
        def on_closing():
            """Appelé quand l'utilisateur ferme la fenêtre"""
            print("\n🛑 Fermeture de l'application...")
            stop_caffeinate()  # Arrêter caffeinate
            shutdown_event.set()  # Signaler l'arrêt aux threads
            time.sleep(0.5)  # Laisser les threads se terminer
            return True  # Permettre la fermeture

        # Créer la fenêtre native avec l'API
        window = webview.create_window(
            title='ClipGenius',
            url=f'http://{SERVER_HOST}:{SERVER_PORT}',
            width=1200,
            height=800,
            resizable=False,  # Taille fixe, pas de resize
            fullscreen=False,  # Pas de plein écran
            background_color='#0d0d0d',
            js_api=api,
            on_top=False,
            confirm_close=False
        )

        # Donner la référence de la fenêtre à l'API
        api.set_window(window)

        # Démarrer webview avec handler de fermeture
        # Note: PyWebView ne supporte pas directement on_closing callback
        # On utilise signal handler à la place
        def signal_handler(sig, frame):
            print("\n🛑 Signal d'arrêt reçu...")
            stop_caffeinate()  # Arrêter caffeinate
            shutdown_event.set()
            time.sleep(0.5)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Démarrer avec debug=False pour éviter la suspension
        webview.start(debug=False)
    else:
        # Fallback navigateur
        import webbrowser

        threading.Thread(
            target=lambda: (time.sleep(1), webbrowser.open(f'http://{SERVER_HOST}:{SERVER_PORT}')),
            daemon=True
        ).start()

        flask_app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True, use_reloader=False)


def ensure_dependencies():
    """
    Auto-installe les dépendances depuis requirements.txt.

    Utilise un hash MD5 pour détecter les changements : l'installation
    ne se relance que si requirements.txt a été modifié depuis la dernière fois.
    Le fichier .deps_hash (ignoré par git) stocke le hash de la dernière installation.
    """
    import hashlib, os, tempfile

    req_file = Path(__file__).parent / 'requirements.txt'
    hash_file = Path(__file__).parent / '.deps_hash'

    if not req_file.exists():
        return

    current_hash = hashlib.md5(req_file.read_bytes()).hexdigest()

    # Déjà installé avec ce requirements.txt exact → rien à faire
    if hash_file.exists() and hash_file.read_text().strip() == current_hash:
        return

    print("📦 Installation des dépendances (première utilisation ou mise à jour)...")
    print("   Cela peut prendre quelques minutes.\n")

    # Pinning setuptools<75 : les versions récentes ont retiré pkg_resources,
    # qui est requis par des packages legacy comme openai-whisper.
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet',
         'pip', 'setuptools<75', 'wheel'],
        capture_output=False,
    )

    # PIP_CONSTRAINT propage le pin aux environnements de build isolés que pip
    # crée en interne : chaque sous-processus pip en hérite automatiquement.
    constraints_content = 'setuptools<75\n'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     prefix='cg_pip_') as f:
        f.write(constraints_content)
        constraints_file = f.name

    env = os.environ.copy()
    env['PIP_CONSTRAINT'] = constraints_file

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(req_file),
             '--quiet', '--upgrade'],
            capture_output=False,
            env=env,
        )
    finally:
        os.unlink(constraints_file)

    if result.returncode == 0:
        hash_file.write_text(current_hash)
        print("\n✅ Dépendances installées avec succès!\n")
    else:
        print("\n⚠️  Certaines dépendances n'ont pas pu être installées. "
              "L'application continue avec ce qui est disponible.\n")


def main():
    """Point d'entrée principal"""
    print("\n" + "="*50)
    print("  ClipGenius beta")
    print("="*50 + "\n")

    # Auto-installer les dépendances si nécessaire
    ensure_dependencies()

    # Vérifier/télécharger le modèle AI au premier démarrage
    model_path = find_phi3_model()
    if not model_path:
        model_path = ensure_phi3_model()
        if not model_path:
            print("⚠️  L'application fonctionnera sans analyse AI locale.\n")
    else:
        print(f"✓ Modèle AI: {model_path.name}\n")

    print("🚀 Démarrage de l'application...\n")

    try:
        run_main_app()
    except KeyboardInterrupt:
        print("\n👋 Fermé")
        stop_caffeinate()  # Arrêter caffeinate proprement
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        stop_caffeinate()  # Arrêter caffeinate même en cas d'erreur
        sys.exit(1)
    finally:
        # S'assurer que caffeinate est bien arrêté
        stop_caffeinate()


if __name__ == '__main__':
    main()
