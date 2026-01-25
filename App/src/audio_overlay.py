"""
Module d'overlay audio/musique pour ClipGenius

Permet d'ajouter de la musique de fond aux clips générés:
- Musique de fond avec volume ajustable
- Fade-in / fade-out automatique
- Duck automatique (baisse le volume pendant la parole)
- Bibliothèque de musiques libres de droits

Usage:
    from src.audio_overlay import add_background_music, AudioOverlayConfig
    
    # Ajouter de la musique simple
    add_background_music("clip.mp4", "music.mp3", "output.mp4")
    
    # Avec configuration avancée
    config = AudioOverlayConfig(volume=0.15, fade_duration=2.0, enable_ducking=True)
    add_background_music("clip.mp4", "music.mp3", "output.mp4", config)
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from rich.console import Console

console = Console()


@dataclass
class AudioOverlayConfig:
    """Configuration pour l'overlay audio"""
    # Volume de la musique (0.0 - 1.0)
    music_volume: float = 0.15  # 15% du volume original par défaut
    
    # Fade in/out
    fade_in_duration: float = 1.5   # Durée du fade-in en secondes
    fade_out_duration: float = 2.0  # Durée du fade-out en secondes
    
    # Ducking (baisse du volume pendant la parole)
    enable_ducking: bool = True     # Activer le ducking automatique
    ducking_threshold: float = 0.1  # Seuil de détection de parole
    ducking_ratio: float = 0.3      # Volume pendant la parole (30% du volume normal)
    ducking_attack: float = 0.1     # Temps d'attaque du ducking (secondes)
    ducking_release: float = 0.3    # Temps de release du ducking (secondes)
    
    # Options
    loop_music: bool = True         # Boucler la musique si plus courte que le clip
    normalize_music: bool = True    # Normaliser le volume de la musique
    preserve_original_audio: bool = True  # Garder l'audio original du clip


class AudioOverlay:
    """
    Gestionnaire d'overlay audio pour ajouter de la musique de fond.
    
    Fonctionnalités:
    - Mixage audio avec volume ajustable
    - Fade-in/out pour transitions douces
    - Ducking automatique pendant les dialogues
    - Support de boucle pour musiques courtes
    """
    
    def __init__(self, config: Optional[AudioOverlayConfig] = None):
        self.config = config or AudioOverlayConfig()
    
    def add_music(
        self,
        video_path: str,
        music_path: str,
        output_path: str,
        start_offset: float = 0.0
    ) -> bool:
        """
        Ajoute de la musique de fond à une vidéo.
        
        Args:
            video_path: Chemin vers la vidéo source
            music_path: Chemin vers le fichier audio (mp3, wav, etc.)
            output_path: Chemin de sortie
            start_offset: Offset de début dans la musique (secondes)
            
        Returns:
            True si succès, False sinon
        """
        video_path = Path(video_path)
        music_path = Path(music_path)
        
        if not video_path.exists():
            console.print(f"[red]Vidéo introuvable: {video_path}[/red]")
            return False
        
        if not music_path.exists():
            console.print(f"[red]Fichier audio introuvable: {music_path}[/red]")
            return False
        
        video = None
        music = None
        final_clip = None
        
        try:
            # Charger la vidéo
            video = VideoFileClip(str(video_path))
            video_duration = video.duration
            
            # Charger la musique
            music = AudioFileClip(str(music_path))
            
            # Appliquer l'offset de début
            if start_offset > 0 and start_offset < music.duration:
                music = music.with_subclip(start_offset)
            
            # Boucler la musique si nécessaire
            if self.config.loop_music and music.duration < video_duration:
                music = self._loop_audio(music, video_duration)
            
            # Couper la musique à la durée de la vidéo
            if music.duration > video_duration:
                music = music.with_subclip(0, video_duration)
            
            # Normaliser le volume si demandé
            if self.config.normalize_music:
                music = self._normalize_audio(music)
            
            # Appliquer le volume de base
            music = music.with_volume_scaled(self.config.music_volume)
            
            # Appliquer les fades
            music = self._apply_fades(music)
            
            # Appliquer le ducking si activé et si la vidéo a de l'audio
            if self.config.enable_ducking and video.audio is not None:
                music = self._apply_ducking(music, video.audio)
            
            # Mixer l'audio
            if self.config.preserve_original_audio and video.audio is not None:
                # Combiner l'audio original avec la musique
                final_audio = CompositeAudioClip([video.audio, music])
            else:
                # Remplacer l'audio par la musique seule
                final_audio = music
            
            # Créer le clip final
            final_clip = video.with_audio(final_audio)
            
            # Exporter
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                audio_bitrate='192k',
                logger=None
            )
            
            return True
            
        except Exception as e:
            console.print(f"[red]Erreur lors de l'ajout de musique: {e}[/red]")
            return False
            
        finally:
            # Libérer les ressources
            if final_clip:
                try:
                    final_clip.close()
                except Exception:
                    pass
            if video:
                try:
                    video.close()
                except Exception:
                    pass
            if music:
                try:
                    music.close()
                except Exception:
                    pass
    
    def _loop_audio(self, audio: AudioFileClip, target_duration: float) -> AudioFileClip:
        """Boucle l'audio pour atteindre la durée cible."""
        from moviepy import concatenate_audioclips
        
        loops_needed = int(np.ceil(target_duration / audio.duration))
        clips = [audio] * loops_needed
        
        looped = concatenate_audioclips(clips)
        
        # Couper à la durée exacte
        return looped.with_subclip(0, target_duration)
    
    def _normalize_audio(self, audio: AudioFileClip) -> AudioFileClip:
        """Normalise le volume de l'audio."""
        # Obtenir les échantillons
        fps = audio.fps or 44100
        samples = audio.to_soundarray(fps=fps)
        
        # Calculer le facteur de normalisation
        max_amplitude = np.max(np.abs(samples))
        
        if max_amplitude > 0:
            # Normaliser à 0.9 pour éviter le clipping
            normalize_factor = 0.9 / max_amplitude
            # Limiter la normalisation pour éviter d'amplifier le bruit
            normalize_factor = min(normalize_factor, 2.0)
            
            if normalize_factor != 1.0:
                audio = audio.with_volume_scaled(normalize_factor)
        
        return audio
    
    def _apply_fades(self, audio: AudioFileClip) -> AudioFileClip:
        """Applique les fade-in et fade-out."""
        duration = audio.duration
        
        # Fade-in
        if self.config.fade_in_duration > 0:
            audio = audio.with_effects([
                lambda clip: clip.audio_fadein(min(self.config.fade_in_duration, duration / 4))
            ]) if hasattr(audio, 'audio_fadein') else self._manual_fade_in(audio)
        
        # Fade-out
        if self.config.fade_out_duration > 0:
            audio = audio.with_effects([
                lambda clip: clip.audio_fadeout(min(self.config.fade_out_duration, duration / 4))
            ]) if hasattr(audio, 'audio_fadeout') else self._manual_fade_out(audio)
        
        return audio
    
    def _manual_fade_in(self, audio: AudioFileClip) -> AudioFileClip:
        """Applique un fade-in manuel."""
        fade_duration = min(self.config.fade_in_duration, audio.duration / 4)
        
        def volume_filter(get_frame, t):
            frame = get_frame(t)
            if t < fade_duration:
                factor = t / fade_duration
                return frame * factor
            return frame
        
        return audio.transform(volume_filter, keep_duration=True)
    
    def _manual_fade_out(self, audio: AudioFileClip) -> AudioFileClip:
        """Applique un fade-out manuel."""
        duration = audio.duration
        fade_duration = min(self.config.fade_out_duration, duration / 4)
        fade_start = duration - fade_duration
        
        def volume_filter(get_frame, t):
            frame = get_frame(t)
            if t > fade_start:
                factor = (duration - t) / fade_duration
                return frame * factor
            return frame
        
        return audio.transform(volume_filter, keep_duration=True)
    
    def _apply_ducking(
        self,
        music: AudioFileClip,
        voice: AudioFileClip
    ) -> AudioFileClip:
        """
        Applique le ducking automatique.
        
        Baisse le volume de la musique quand il y a de la parole.
        """
        fps = music.fps or 44100
        duration = music.duration
        
        # Analyser le volume de la voix
        voice_samples = voice.to_soundarray(fps=fps)
        if len(voice_samples.shape) > 1:
            voice_mono = np.mean(voice_samples, axis=1)
        else:
            voice_mono = voice_samples
        
        # Calculer l'enveloppe du volume
        window_size = int(fps * 0.05)  # Fenêtre de 50ms
        envelope = np.zeros(len(voice_mono))
        
        for i in range(0, len(voice_mono) - window_size, window_size):
            envelope[i:i+window_size] = np.max(np.abs(voice_mono[i:i+window_size]))
        
        # Détecter les zones de parole
        threshold = self.config.ducking_threshold * np.max(envelope)
        speech_mask = envelope > threshold
        
        # Appliquer l'attaque et le release
        attack_samples = int(self.config.ducking_attack * fps)
        release_samples = int(self.config.ducking_release * fps)
        
        # Lisser le masque
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        speech_mask_float = speech_mask.astype(float)
        speech_mask_float = maximum_filter1d(speech_mask_float, release_samples)
        speech_mask_float = minimum_filter1d(speech_mask_float, attack_samples)
        
        # Calculer le facteur de volume
        ducking_factor = 1.0 - (1.0 - self.config.ducking_ratio) * speech_mask_float
        
        # Appliquer le ducking à la musique
        def apply_ducking_filter(get_frame, t):
            frame = get_frame(t)
            sample_idx = int(t * fps)
            
            if sample_idx < len(ducking_factor):
                factor = ducking_factor[sample_idx]
            else:
                factor = 1.0
            
            return frame * factor
        
        return music.transform(apply_ducking_filter, keep_duration=True)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def add_background_music(
    video_path: str,
    music_path: str,
    output_path: str,
    volume: float = 0.15,
    fade_duration: float = 2.0,
    enable_ducking: bool = True
) -> bool:
    """
    Ajoute rapidement de la musique de fond à une vidéo.
    
    Args:
        video_path: Chemin vers la vidéo
        music_path: Chemin vers la musique
        output_path: Chemin de sortie
        volume: Volume de la musique (0.0 - 1.0)
        fade_duration: Durée des fades
        enable_ducking: Activer le ducking automatique
        
    Returns:
        True si succès
    """
    config = AudioOverlayConfig(
        music_volume=volume,
        fade_in_duration=fade_duration,
        fade_out_duration=fade_duration,
        enable_ducking=enable_ducking
    )
    
    overlay = AudioOverlay(config)
    return overlay.add_music(video_path, music_path, output_path)


def add_music_to_clips(
    clips: List[str],
    music_path: str,
    output_dir: str,
    config: Optional[AudioOverlayConfig] = None
) -> List[str]:
    """
    Ajoute de la musique à plusieurs clips.
    
    Args:
        clips: Liste des chemins vers les clips
        music_path: Chemin vers la musique
        output_dir: Dossier de sortie
        config: Configuration audio
        
    Returns:
        Liste des chemins des clips avec musique
    """
    config = config or AudioOverlayConfig()
    overlay = AudioOverlay(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for clip_path in clips:
        clip_name = Path(clip_path).stem
        output_path = output_dir / f"{clip_name}_music.mp4"
        
        if overlay.add_music(clip_path, music_path, str(output_path)):
            results.append(str(output_path))
            console.print(f"  [green]✓[/green] {clip_name}")
        else:
            console.print(f"  [yellow]✗[/yellow] {clip_name}")
    
    return results


# Bibliothèque de musiques recommandées (liens vers ressources libres de droits)
MUSIC_RESOURCES = {
    "youtube_audio_library": "https://studio.youtube.com/channel/UC/music",
    "mixkit": "https://mixkit.co/free-stock-music/",
    "pixabay": "https://pixabay.com/music/",
    "uppbeat": "https://uppbeat.io/",
    "epidemic_sound": "https://www.epidemicsound.com/",
}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 4:
        video = sys.argv[1]
        music = sys.argv[2]
        output = sys.argv[3]
        
        success = add_background_music(video, music, output)
        
        if success:
            print(f"Musique ajoutée: {output}")
        else:
            print("Erreur lors de l'ajout de musique")
            sys.exit(1)
    else:
        print("Usage: python -m src.audio_overlay <video> <music> <output>")
        print("\nRessources de musique libre de droits:")
        for name, url in MUSIC_RESOURCES.items():
            print(f"  - {name}: {url}")
