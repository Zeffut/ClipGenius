"""
Module de téléchargement de vidéos YouTube
Utilise yt-dlp pour télécharger des vidéos de haute qualité
Gère automatiquement le runtime deno pour contourner les restrictions YouTube
"""

import os
import time
import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from rich.console import Console

from .runtime_manager import get_runtime_manager, ensure_deno_available

console = Console()

# Type pour le callback de progression
ProgressCallback = Callable[[int, str], None]  # (percent, message)


class VideoDownloader:
    """Télécharge des vidéos YouTube avec yt-dlp"""
    
    def __init__(self, output_dir: str = "downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback: Optional[ProgressCallback] = None
        
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Récupère les informations de la vidéo sans télécharger"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            # Pas de format spécifique pour get_info (évite les erreurs de format)
            'skip_download': True,
            # Utiliser android_vr pour contourner les restrictions SABR
            'extractor_args': {
                'youtube': {
                    'player_client': ['android_vr'],
                }
            },
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'unknown'),
                'duration': info.get('duration', 0),
                'description': info.get('description', ''),
                'uploader': info.get('uploader', ''),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'id': info.get('id', ''),
            }
    
    def _make_progress_hook(self):
        """Crée un hook de progression pour yt-dlp"""
        last_percent = [0]  # Utiliser une liste pour modifier dans la closure
        
        def hook(d):
            if d['status'] == 'downloading':
                # Calculer le pourcentage
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                
                if total > 0:
                    percent = int((downloaded / total) * 100)
                    # Ne reporter que si le pourcentage a changé d'au moins 2%
                    if percent >= last_percent[0] + 2 or percent == 100:
                        last_percent[0] = percent
                        speed = d.get('speed', 0)
                        speed_str = f"{speed/1024/1024:.1f} MB/s" if speed else ""
                        eta = d.get('eta', 0)
                        eta_str = f"ETA: {eta}s" if eta else ""
                        
                        msg = f"Téléchargement: {percent}%"
                        if speed_str:
                            msg += f" ({speed_str})"
                        if eta_str:
                            msg += f" - {eta_str}"
                        
                        if self._progress_callback:
                            # Mapper 0-100% du download vers 20-90% de l'étape
                            mapped_percent = 20 + int(percent * 0.7)
                            self._progress_callback(mapped_percent, msg)
                            
            elif d['status'] == 'finished':
                if self._progress_callback:
                    self._progress_callback(92, "Téléchargement terminé, conversion en cours...")
                    
        return hook
    
    def download(self, url: str, quality: str = "best", 
                 progress_callback: Optional[ProgressCallback] = None,
                 max_retries: int = 3) -> Optional[str]:
        """
        Télécharge une vidéo YouTube avec retry automatique en cas d'erreur réseau.
        
        Args:
            url: URL de la vidéo YouTube
            quality: Qualité souhaitée ('best', '1080p', '720p', '480p')
            progress_callback: Callback pour la progression (percent, message)
            max_retries: Nombre maximum de tentatives (défaut: 3)
            
        Returns:
            Chemin vers le fichier téléchargé ou None si échec
        """
        self._progress_callback = progress_callback
        
        # S'assurer que deno est disponible pour YouTube (haute qualité)
        runtime_manager = get_runtime_manager()
        deno_path = runtime_manager.ensure_deno(progress_callback)
        
        if deno_path:
            console.print(f"[green]Runtime deno disponible: {deno_path}[/green]")
        else:
            console.print("[yellow]deno non disponible, qualité peut être limitée[/yellow]")
        
        # Configurer l'environnement avec deno dans le PATH
        env = runtime_manager.get_yt_dlp_env()
        
        # Configuration de la qualité
        format_spec = self._get_format_spec(quality)
        
        # Nom du fichier de sortie
        output_template = str(self.output_dir / '%(title)s.%(ext)s')
        
        ydl_opts = {
            'format': format_spec,
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'quiet': False,  # Afficher les messages pour debug
            'no_warnings': False,
            'progress_hooks': [self._make_progress_hook()],
            # Utiliser android_vr pour contourner les restrictions SABR de YouTube
            # Ce client n'a pas besoin de PO Token et offre tous les formats HD
            'extractor_args': {
                'youtube': {
                    'player_client': ['android_vr'],
                }
            },
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
            },
        }
        
        # Sauvegarder le PATH original avant mutation
        original_path = os.environ.get("PATH", "")
        
        # Mettre à jour l'environnement du processus pour yt-dlp
        os.environ.update(env)
        
        last_error: Optional[Exception] = None
        
        try:
            for attempt in range(1, max_retries + 1):
                try:
                    if progress_callback:
                        if attempt > 1:
                            progress_callback(5, f"Tentative {attempt}/{max_retries}...")
                        else:
                            progress_callback(5, "Connexion à YouTube...")
                    else:
                        if attempt > 1:
                            console.print(f"[yellow]Tentative {attempt}/{max_retries}...[/yellow]")
                        else:
                            console.print(f"[cyan]Téléchargement de la vidéo...[/cyan]")
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        if progress_callback:
                            progress_callback(10, "Récupération des métadonnées...")
                            
                        info = ydl.extract_info(url, download=True)
                        
                        # Trouver le fichier téléchargé
                        if info:
                            filename = ydl.prepare_filename(info)
                            # Assurer l'extension .mp4
                            base = os.path.splitext(filename)[0]
                            final_path = base + '.mp4'
                            
                            if os.path.exists(final_path):
                                if progress_callback:
                                    progress_callback(100, f"Téléchargement terminé")
                                else:
                                    console.print(f"[green]Vidéo téléchargée: {final_path}[/green]")
                                return final_path
                            elif os.path.exists(filename):
                                if progress_callback:
                                    progress_callback(100, f"Téléchargement terminé")
                                else:
                                    console.print(f"[green]Vidéo téléchargée: {filename}[/green]")
                                return filename
                    
                    # Si on arrive ici, le téléchargement n'a rien produit
                    last_error = Exception("Aucun fichier produit par yt-dlp")
                            
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        # Délai progressif: 3s, 6s entre les tentatives
                        delay = attempt * 3
                        console.print(
                            f"[yellow]Erreur: {e}[/yellow]\n"
                            f"[yellow]Nouvelle tentative dans {delay}s...[/yellow]"
                        )
                        if progress_callback:
                            progress_callback(0, f"Erreur réseau, retry dans {delay}s...")
                        time.sleep(delay)
                    else:
                        # Dernière tentative échouée
                        if progress_callback:
                            progress_callback(0, f"Erreur après {max_retries} tentatives: {e}")
                        else:
                            console.print(f"[red]Erreur après {max_retries} tentatives: {e}[/red]")
            
            return None
            
        finally:
            self._progress_callback = None
            # Restaurer le PATH original pour éviter une mutation permanente
            os.environ["PATH"] = original_path
    
    def _get_format_spec(self, quality: str) -> str:
        """
        Retourne la spécification de format pour yt-dlp
        Privilégie H.264 (avc1) pour un meilleur bitrate et compatibilité
        """
        # Préférer H.264 (avc1) car il a généralement un bitrate plus élevé que AV1/VP9
        # Format: meilleur H.264 vidéo + meilleur audio, sinon meilleur disponible
        quality_map = {
            # Préférer avc1 (H.264) pour meilleur bitrate, puis vp9, puis av01
            'best': 'bv[vcodec^=avc1]+ba/bv[vcodec^=vp9]+ba/bv+ba/b',
            '1080p': 'bv[height<=1080][vcodec^=avc1]+ba/bv[height<=1080][vcodec^=vp9]+ba/bv[height<=1080]+ba/b',
            '720p': 'bv[height<=720][vcodec^=avc1]+ba/bv[height<=720][vcodec^=vp9]+ba/bv[height<=720]+ba/b',
            '480p': 'bv[height<=480][vcodec^=avc1]+ba/bv[height<=480][vcodec^=vp9]+ba/bv[height<=480]+ba/b',
        }
        return quality_map.get(quality, quality_map['best'])


def download_video(url: str, output_dir: str = "downloads", quality: str = "best") -> Optional[str]:
    """Fonction utilitaire pour télécharger une vidéo"""
    downloader = VideoDownloader(output_dir)
    return downloader.download(url, quality)


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        result = download_video(sys.argv[1])
        print(f"Résultat: {result}")
