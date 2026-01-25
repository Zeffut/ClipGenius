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

MODELS_DIR = "models"
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


def find_phi3_model() -> Optional[Path]:
    """Cherche le modèle Phi-3 dans les emplacements connus"""
    search_dirs = [
        get_project_root() / MODELS_DIR,
        get_project_root(),
        Path.home() / ".cache" / "clipgenius",
        Path.home() / ".cache" / "huggingface",
    ]
    names = [
        "Phi-3-mini-4k-instruct-q4.gguf",
        "phi-3-mini-4k-instruct-q4.gguf",
        "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in names:
            model_path = directory / name
            if model_path.exists() and model_path.stat().st_size > 1_000_000_000:
                return model_path
    return None


def download_phi3_model(progress_callback=None) -> Optional[Path]:
    """
    Télécharge le modèle Phi-3 avec progression.
    
    Args:
        progress_callback: Fonction appelée avec (downloaded_mb, total_mb, speed_mb_s, eta_text)
    
    Returns:
        Path vers le modèle téléchargé, ou None si échec
    """
    models_dir = get_models_dir()
    model_path = models_dir / PHI3_MODEL_NAME
    temp_path = models_dir / f"{PHI3_MODEL_NAME}.tmp"
    
    # Vérifier si déjà présent
    if model_path.exists() and model_path.stat().st_size > 1_000_000_000:
        return model_path
    
    try:
        req = urllib.request.Request(PHI3_MODEL_URL)
        req.add_header('User-Agent', 'ClipGenius/2.0')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get('Content-Length', PHI3_MODEL_SIZE_MB * 1024 * 1024))
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
                                alpha = min(0.3, max(0.05, diff_ratio * 0.5))
                                smoothed_speed = alpha * current_speed + (1 - alpha) * smoothed_speed
                    
                    # Notifier toutes les 250ms
                    notify_elapsed = now - last_notify_time
                    if notify_elapsed >= 0.25 and progress_callback:
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


def ensure_phi3_model() -> Optional[Path]:
    """
    S'assure que le modèle Phi-3 est disponible.
    Le télécharge si nécessaire avec progression console.
    
    Returns:
        Path vers le modèle, ou None si échec
    """
    # Chercher un modèle existant
    model_path = find_phi3_model()
    if model_path:
        return model_path
    
    print("\n" + "="*50)
    print("  Téléchargement du modèle AI (première utilisation)")
    print("="*50 + "\n")
    print(f"  Modèle: {PHI3_MODEL_NAME}")
    print(f"  Taille: ~{PHI3_MODEL_SIZE_MB} MB")
    print()
    
    def show_progress(downloaded, total, speed, eta):
        pct = int(downloaded / total * 100)
        bar_width = 30
        filled = int(bar_width * downloaded / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        speed_text = f"{speed:.1f} MB/s" if speed > 0 else "..."
        eta_text = f" - {eta} restant" if eta else ""
        
        print(f"\r  [{bar}] {pct}% - {downloaded:.0f}/{total:.0f} MB - {speed_text}{eta_text}    ", end="", flush=True)
    
    result = download_phi3_model(progress_callback=show_progress)
    
    if result:
        print("\n\n  ✓ Modèle téléchargé avec succès!\n")
    else:
        print("\n\n  ✗ Échec du téléchargement\n")
    
    return result


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
        
        file_types = ('Fichiers vidéo (*.mp4;*.mov;*.avi;*.mkv;*.webm)',)
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
                activity = info.beginActivityWithOptions_reason_(
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
    
    # === LAZY IMPORT: Charger Flask seulement maintenant ===
    print("⏳ Chargement de l'interface...")
    from web_app import app as flask_app
    print("✓ Interface prête!")
    
    # Créer le dossier output
    Path('output').mkdir(exist_ok=True)
    
    if WEBVIEW_AVAILABLE:
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        
        # Démarrer Flask en arrière-plan avec thread daemon
        flask_thread = threading.Thread(
            target=lambda: flask_app.run(host='127.0.0.1', port=5001, debug=False, threaded=True, use_reloader=False),
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
            url='http://127.0.0.1:5001',
            width=1200,
            height=800,  # Réduit de 900 à 800 pour laisser de l'espace
            resizable=True,
            min_size=(800, 600),
            background_color='#0d0d0d',  # Fond sombre comme le HTML
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
            target=lambda: (time.sleep(1), webbrowser.open('http://127.0.0.1:5001')),
            daemon=True
        ).start()
        
        flask_app.run(host='127.0.0.1', port=5001, debug=False, threaded=True, use_reloader=False)


def main():
    """Point d'entrée principal"""
    print("\n" + "="*50)
    print("  ClipGenius v2.0")
    print("="*50 + "\n")
    
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
