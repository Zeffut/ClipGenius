"""
Module de gestion des runtimes JavaScript pour yt-dlp
Télécharge et configure automatiquement deno pour permettre le téléchargement
de vidéos YouTube en haute qualité (contourne les restrictions SABR)
"""

import os
import sys
import shutil
import platform
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

console = Console()

# Version de deno à télécharger
DENO_VERSION = "2.1.4"

# URLs de téléchargement par plateforme
DENO_DOWNLOAD_URLS = {
    ("Darwin", "arm64"): f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}/deno-aarch64-apple-darwin.zip",
    ("Darwin", "x86_64"): f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}/deno-x86_64-apple-darwin.zip",
    ("Linux", "x86_64"): f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip",
    ("Linux", "aarch64"): f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}/deno-aarch64-unknown-linux-gnu.zip",
    ("Windows", "AMD64"): f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}/deno-x86_64-pc-windows-msvc.zip",
}


class RuntimeManager:
    """Gère les runtimes JavaScript nécessaires pour yt-dlp"""
    
    def __init__(self, app_dir: Optional[str] = None):
        """
        Initialise le gestionnaire de runtimes
        
        Args:
            app_dir: Répertoire de l'application. Par défaut, utilise le répertoire parent de src/
        """
        if app_dir is None:
            # Remonter depuis src/ vers App/
            app_dir = Path(__file__).parent.parent
        
        self.app_dir = Path(app_dir)
        self.bin_dir = self.app_dir / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemin vers deno
        self.deno_name = "deno.exe" if platform.system() == "Windows" else "deno"
        self.deno_path = self.bin_dir / self.deno_name
        
    def get_platform_key(self) -> Tuple[str, str]:
        """Retourne la clé de plateforme (OS, architecture)"""
        system = platform.system()
        machine = platform.machine()
        
        # Normalisation des architectures
        if machine in ("arm64", "aarch64"):
            machine = "arm64" if system == "Darwin" else "aarch64"
        elif machine in ("x86_64", "AMD64"):
            machine = "x86_64" if system != "Windows" else "AMD64"
            
        return (system, machine)
    
    def is_deno_installed(self) -> bool:
        """Vérifie si deno est installé (localement ou globalement)"""
        # Vérifier d'abord le deno local
        if self.deno_path.exists():
            return True
            
        # Vérifier le deno global
        return shutil.which("deno") is not None
    
    def get_deno_path(self) -> Optional[str]:
        """
        Retourne le chemin vers deno
        Priorité: local > global
        """
        if self.deno_path.exists():
            return str(self.deno_path)
            
        global_deno = shutil.which("deno")
        if global_deno:
            return global_deno
            
        return None
    
    def download_deno(self, progress_callback=None) -> bool:
        """
        Télécharge et installe deno localement
        
        Args:
            progress_callback: Callback optionnel (percent, message)
            
        Returns:
            True si succès, False sinon
        """
        platform_key = self.get_platform_key()
        
        if platform_key not in DENO_DOWNLOAD_URLS:
            console.print(f"[red]Plateforme non supportée: {platform_key}[/red]")
            return False
            
        url = DENO_DOWNLOAD_URLS[platform_key]
        zip_path = self.bin_dir / "deno.zip"
        
        try:
            if progress_callback:
                progress_callback(10, "Téléchargement de deno...")
            else:
                console.print(f"[cyan]Téléchargement de deno depuis {url}...[/cyan]")
            
            # Télécharger le fichier
            urllib.request.urlretrieve(url, zip_path)
            
            if progress_callback:
                progress_callback(60, "Extraction de deno...")
            else:
                console.print("[cyan]Extraction de deno...[/cyan]")
            
            # Extraire (avec protection contre zip-slip)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    member_path = (self.bin_dir / member).resolve()
                    if not str(member_path).startswith(str(self.bin_dir.resolve())):
                        raise ValueError(f"Zip path traversal detected: {member}")
                zip_ref.extractall(self.bin_dir)
            
            # Rendre exécutable sur Unix
            if platform.system() != "Windows":
                os.chmod(self.deno_path, 0o755)
            
            # Nettoyer
            zip_path.unlink()
            
            # Vérifier l'installation
            if self.deno_path.exists():
                if progress_callback:
                    progress_callback(100, "deno installé avec succès")
                else:
                    console.print(f"[green]deno installé: {self.deno_path}[/green]")
                return True
            else:
                if progress_callback:
                    progress_callback(0, "Échec de l'installation de deno")
                return False
                
        except Exception as e:
            console.print(f"[red]Erreur lors du téléchargement de deno: {e}[/red]")
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    def ensure_deno(self, progress_callback=None) -> Optional[str]:
        """
        S'assure que deno est disponible, le télécharge si nécessaire
        
        Args:
            progress_callback: Callback optionnel (percent, message)
            
        Returns:
            Chemin vers deno ou None si échec
        """
        deno_path = self.get_deno_path()
        
        if deno_path:
            return deno_path
            
        # Télécharger deno
        console.print("[yellow]deno non trouvé, téléchargement en cours...[/yellow]")
        
        if self.download_deno(progress_callback):
            return str(self.deno_path)
            
        return None
    
    def get_yt_dlp_env(self) -> dict:
        """
        Retourne les variables d'environnement pour yt-dlp avec deno configuré
        
        Returns:
            Dictionnaire des variables d'environnement
        """
        env = os.environ.copy()
        
        deno_path = self.get_deno_path()
        if deno_path:
            # Ajouter le répertoire de deno au PATH
            deno_dir = str(Path(deno_path).parent)
            current_path = env.get("PATH", "")
            
            # Mettre deno en premier dans le PATH
            if deno_dir not in current_path:
                env["PATH"] = f"{deno_dir}{os.pathsep}{current_path}"
                
        return env
    
    def verify_deno(self) -> Tuple[bool, str]:
        """
        Vérifie que deno fonctionne correctement
        
        Returns:
            (success, version_string ou error_message)
        """
        deno_path = self.get_deno_path()
        
        if not deno_path:
            return False, "deno non installé"
            
        try:
            result = subprocess.run(
                [deno_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                return True, version
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)


# Instance globale pour faciliter l'accès
_runtime_manager: Optional[RuntimeManager] = None


def get_runtime_manager() -> RuntimeManager:
    """Retourne l'instance globale du RuntimeManager"""
    global _runtime_manager
    if _runtime_manager is None:
        _runtime_manager = RuntimeManager()
    return _runtime_manager


def ensure_deno_available(progress_callback=None) -> Optional[str]:
    """
    Fonction utilitaire pour s'assurer que deno est disponible
    
    Returns:
        Chemin vers deno ou None
    """
    manager = get_runtime_manager()
    return manager.ensure_deno(progress_callback)


def get_yt_dlp_environment() -> dict:
    """
    Retourne l'environnement configuré pour yt-dlp avec deno
    
    Returns:
        Variables d'environnement
    """
    manager = get_runtime_manager()
    return manager.get_yt_dlp_env()
