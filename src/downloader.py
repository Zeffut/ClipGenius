"""
Module de téléchargement de vidéos YouTube
Utilise yt-dlp pour télécharger des vidéos de haute qualité
"""

import os
import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from rich.console import Console

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
            # Options anti-blocage YouTube
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
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
                 progress_callback: Optional[ProgressCallback] = None) -> Optional[str]:
        """
        Télécharge une vidéo YouTube
        
        Args:
            url: URL de la vidéo YouTube
            quality: Qualité souhaitée ('best', '1080p', '720p', '480p')
            progress_callback: Callback pour la progression (percent, message)
            
        Returns:
            Chemin vers le fichier téléchargé ou None si échec
        """
        self._progress_callback = progress_callback
        
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
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self._make_progress_hook()],
            # Options anti-blocage YouTube (403 Forbidden)
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
        }
        
        try:
            if progress_callback:
                progress_callback(5, "Connexion à YouTube...")
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
                        
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Erreur: {e}")
            else:
                console.print(f"[red]Erreur lors du téléchargement: {e}[/red]")
            return None
        finally:
            self._progress_callback = None
            
        return None
    
    def _get_format_spec(self, quality: str) -> str:
        """Retourne la spécification de format pour yt-dlp"""
        quality_map = {
            'best': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '1080p': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]',
            '720p': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]',
            '480p': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]',
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
