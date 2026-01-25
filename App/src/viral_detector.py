"""
Module de détection des moments viraux
Analyse l'audio et la vidéo pour identifier les moments à fort potentiel viral
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from moviepy import VideoFileClip
from scipy.signal import find_peaks
from rich.console import Console
from rich.progress import Progress

console = Console()

# Seuil minimum de score pour qu'un moment soit considéré comme viral
VIRAL_SCORE_THRESHOLD = 0.5


@dataclass
class ViralMoment:
    """Représente un moment potentiellement viral dans la vidéo"""
    start_time: float  # Temps de début en secondes
    end_time: float    # Temps de fin en secondes
    score: float       # Score de viralité (0-1)
    reason: str        # Raison de la sélection
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ViralMomentDetector:
    """
    Détecte les moments à fort potentiel viral dans une vidéo
    
    Critères analysés:
    - Pics d'énergie audio (moments excitants, rires, surprises)
    - Changements de scène rapides
    - Variations de volume (climax)
    - Patterns de parole (phrases accrocheuses)
    
    Le nombre de clips est déterminé automatiquement en fonction de la qualité
    des moments détectés. Seuls les moments avec un score suffisant sont gardés.
    """
    
    def __init__(
        self,
        min_clip_duration: float = 60.0,
        max_clip_duration: float = 90.0,
        min_viral_score: float = 0.80,
        max_clips: Optional[int] = None,
    ):
        """
        Args:
            min_clip_duration: Durée minimum des clips en secondes (défaut: 60s)
            max_clip_duration: Durée maximum des clips en secondes (défaut: 90s)
            min_viral_score: Score minimum pour qu'un moment soit considéré viral (0-1)
            max_clips: Nombre maximum de clips (None = pas de limite, déterminé par qualité)
        """
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.min_viral_score = min_viral_score
        self.max_clips = max_clips
        
    def analyze(self, video_path: str) -> List[ViralMoment]:
        """
        Analyse une vidéo et retourne les moments viraux potentiels
        
        Le nombre de clips est déterminé automatiquement:
        - Seuls les moments avec un score >= min_viral_score sont gardés
        - Les clips ne se chevauchent pas
        - Préfère moins de clips de haute qualité que beaucoup de clips moyens
        
        Args:
            video_path: Chemin vers le fichier vidéo
            
        Returns:
            Liste des moments viraux triés par score
        """
        console.print("[cyan]Analyse de la vidéo pour détecter les moments viraux...[/cyan]")
        
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Vidéo non trouvée: {video_path_obj}")
        
        # Charger la vidéo
        video = VideoFileClip(str(video_path_obj))
        duration = video.duration
        
        console.print(f"  Durée de la vidéo: {duration/60:.1f} minutes")
        
        # Vérifier si la vidéo est assez longue
        if duration < self.min_clip_duration:
            console.print(f"[yellow]Vidéo trop courte ({duration:.0f}s < {self.min_clip_duration:.0f}s minimum)[/yellow]")
            video.close()
            return []
        
        # Extraire et analyser l'audio
        audio_path = self._extract_audio(video, video_path_obj)
        audio_moments = self._analyze_audio(audio_path, duration)
        
        # Analyser les changements de scène
        scene_moments = self._analyze_scene_changes(video)
        
        # Combiner et scorer les moments
        all_moments = self._combine_moments(audio_moments, scene_moments, duration)
        
        # Sélectionner uniquement les meilleurs clips (filtrage par qualité)
        selected_moments = self._select_best_clips(all_moments, duration)
        
        video.close()
        
        if selected_moments:
            console.print(f"[green]{len(selected_moments)} moment(s) viral(aux) détecté(s)![/green]")
            for i, m in enumerate(selected_moments, 1):
                console.print(f"  [dim]{i}. Score: {m.score:.2f} - {m.reason}[/dim]")
        else:
            console.print("[yellow]Aucun moment suffisamment viral détecté dans cette vidéo.[/yellow]")
        
        return selected_moments
    
    def _extract_audio(self, video: VideoFileClip, video_path: Path) -> str:
        """Extrait la piste audio de la vidéo"""
        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
        
        if video.audio is not None:
            video.audio.write_audiofile(
                str(audio_path),
                fps=22050,
                logger=None
            )
        
        return str(audio_path)
    
    def _analyze_audio(self, audio_path: str, video_duration: float) -> List[Dict[str, Any]]:
        """
        Analyse l'audio pour détecter les moments intéressants
        
        Détecte:
        - Pics d'énergie (moments excitants)
        - Variations de volume importantes
        - Onset strength (débuts de sons/paroles)
        """
        moments = []
        
        try:
            # Charger l'audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 1. Analyse de l'énergie RMS (pics d'excitation)
            rms = librosa.feature.rms(y=y)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            
            # Normaliser
            rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
            
            # Trouver les pics d'énergie avec des seuils plus stricts
            peaks, properties = find_peaks(
                rms_normalized,
                height=0.7,  # Seuil plus élevé pour ne garder que les vrais pics
                distance=sr // 512 * 5,  # Au moins 5 secondes entre les pics
                prominence=0.25
            )
            
            for peak in peaks:
                if peak < len(rms_times):
                    time = rms_times[peak]
                    score = float(rms_normalized[peak])
                    moments.append({
                        'time': time,
                        'score': score,
                        'type': 'energy_peak',
                        'reason': 'Pic d\'énergie audio'
                    })
            
            # 2. Analyse des onsets (débuts de sons)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
            
            # Détecter les clusters d'onsets (moments très actifs)
            window_size = int(sr / 512 * 3)  # Fenêtre de 3 secondes
            onset_density = np.convolve(onset_env, np.ones(window_size)/window_size, mode='same')
            onset_density_normalized = (onset_density - onset_density.min()) / (onset_density.max() - onset_density.min() + 1e-8)
            
            density_peaks, _ = find_peaks(
                onset_density_normalized,
                height=0.6,  # Seuil plus strict
                distance=sr // 512 * 8,
                prominence=0.2
            )
            
            for peak in density_peaks:
                if peak < len(onset_times):
                    time = onset_times[peak]
                    # Éviter les doublons proches
                    if not any(abs(m['time'] - time) < 5 for m in moments):
                        moments.append({
                            'time': time,
                            'score': float(onset_density_normalized[peak]) * 0.9,
                            'type': 'onset_cluster',
                            'reason': 'Moment très actif'
                        })
            
            # 3. Analyse spectrale pour détecter les variations tonales
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(spectral_contrast, axis=0)
            contrast_times = librosa.frames_to_time(np.arange(len(contrast_mean)), sr=sr)
            
            # Détecter les changements brusques de contraste spectral
            contrast_diff = np.abs(np.diff(contrast_mean))
            contrast_diff_normalized = (contrast_diff - contrast_diff.min()) / (contrast_diff.max() - contrast_diff.min() + 1e-8)
            
            change_peaks, _ = find_peaks(
                contrast_diff_normalized,
                height=0.75,  # Seuil plus strict
                distance=sr // 512 * 6
            )
            
            for peak in change_peaks:
                if peak < len(contrast_times) - 1:
                    time = contrast_times[peak]
                    if not any(abs(m['time'] - time) < 5 for m in moments):
                        moments.append({
                            'time': time,
                            'score': float(contrast_diff_normalized[peak]) * 0.8,
                            'type': 'spectral_change',
                            'reason': 'Changement tonal marqué'
                        })
            
        except Exception as e:
            console.print(f"[yellow]Avertissement lors de l'analyse audio: {e}[/yellow]")
        
        # Nettoyer le fichier audio temporaire
        try:
            Path(audio_path).unlink()
        except:
            pass
            
        return moments
    
    def _analyze_scene_changes(self, video: VideoFileClip) -> List[Dict[str, Any]]:
        """
        Analyse les changements de scène dans la vidéo
        
        Utilise la différence entre frames consécutives pour détecter les cuts
        """
        moments = []
        
        try:
            # Échantillonner la vidéo à 2 fps pour l'efficacité
            fps = 2
            prev_frame = None
            frame_diffs = []
            times = []
            
            console.print("[cyan]Analyse des changements de scène...[/cyan]")
            
            for t in np.arange(0, video.duration, 1/fps):
                try:
                    frame = video.get_frame(t)
                    # Réduire la résolution pour accélérer
                    frame_small = frame[::4, ::4, :].astype(np.float32)
                    
                    if prev_frame is not None:
                        # Calculer la différence
                        diff = np.mean(np.abs(frame_small - prev_frame))
                        frame_diffs.append(diff)
                        times.append(t)
                    
                    prev_frame = frame_small
                except:
                    continue
            
            if frame_diffs:
                frame_diffs = np.array(frame_diffs)
                diffs_normalized = (frame_diffs - frame_diffs.min()) / (frame_diffs.max() - frame_diffs.min() + 1e-8)
                
                # Détecter uniquement les changements très significatifs
                change_peaks, _ = find_peaks(
                    diffs_normalized,
                    height=0.6,  # Seuil plus strict
                    distance=fps * 5  # Au moins 5 secondes entre les détections
                )
                
                for peak in change_peaks:
                    moments.append({
                        'time': times[peak],
                        'score': float(diffs_normalized[peak]) * 0.7,
                        'type': 'scene_change',
                        'reason': 'Changement de scène important'
                    })
                    
        except Exception as e:
            console.print(f"[yellow]Avertissement lors de l'analyse de scène: {e}[/yellow]")
            
        return moments
    
    def _combine_moments(
        self,
        audio_moments: List[Dict],
        scene_moments: List[Dict],
        duration: float
    ) -> List[Dict[str, Any]]:
        """Combine et pondère tous les moments détectés"""
        all_moments = audio_moments + scene_moments
        
        # Boost pour les moments où plusieurs signaux convergent
        for moment in all_moments:
            time = moment['time']
            # Compter les autres moments proches
            nearby_count = sum(
                1 for m in all_moments 
                if m != moment and abs(m['time'] - time) < 10
            )
            if nearby_count > 0:
                # Boost si plusieurs signaux proches
                moment['score'] *= (1 + 0.15 * nearby_count)
                moment['reason'] += f' (+{nearby_count} signaux)'
        
        # Léger boost pour les moments en début de vidéo (hook potentiel)
        for moment in all_moments:
            if moment['time'] < duration * 0.15:
                moment['score'] *= 1.1
        
        # Normaliser les scores pour qu'ils restent dans [0, 1]
        if all_moments:
            max_score = max(m['score'] for m in all_moments)
            if max_score > 1:
                for m in all_moments:
                    m['score'] /= max_score
        
        # Trier par score décroissant
        all_moments.sort(key=lambda x: x['score'], reverse=True)
        
        return all_moments
    
    def _select_best_clips(
        self,
        moments: List[Dict],
        video_duration: float
    ) -> List[ViralMoment]:
        """
        Sélectionne les meilleurs clips en fonction de leur score viral
        
        - Ne garde que les moments avec score >= min_viral_score
        - Évite les chevauchements
        - Limite au max_clips si défini
        """
        selected = []
        used_ranges = []
        
        # Calculer le nombre maximum théorique de clips non-chevauchants
        max_possible_clips = int(video_duration / self.min_clip_duration)
        
        for moment in moments:
            # Arrêter si on a atteint la limite (si définie)
            if self.max_clips is not None and len(selected) >= self.max_clips:
                break
            
            # Arrêter si on ne peut plus ajouter de clips
            if len(selected) >= max_possible_clips:
                break
            
            # Ne garder que les moments vraiment viraux
            if moment['score'] < self.min_viral_score:
                continue
            
            center_time = moment['time']
            
            # Calculer le début et la fin du clip
            clip_duration = self.min_clip_duration
            
            # Centrer le clip sur le moment, avec le pic au premier tiers
            start_time = max(0, center_time - clip_duration * 0.3)
            end_time = min(video_duration, start_time + clip_duration)
            
            # Ajuster si on dépasse la fin
            if end_time - start_time < self.min_clip_duration:
                start_time = max(0, end_time - self.min_clip_duration)
            
            # Vérifier si le clip est assez long
            if end_time - start_time < self.min_clip_duration:
                continue
            
            # Vérifier les chevauchements
            overlaps = False
            for used_start, used_end in used_ranges:
                # Ajouter une marge de 5 secondes entre les clips
                if not (end_time + 5 <= used_start or start_time >= used_end + 5):
                    overlaps = True
                    break
            
            if not overlaps:
                selected.append(ViralMoment(
                    start_time=start_time,
                    end_time=end_time,
                    score=moment['score'],
                    reason=moment['reason']
                ))
                used_ranges.append((start_time, end_time))
        
        # Trier par temps de début
        selected.sort(key=lambda x: x.start_time)
        
        return selected


def detect_viral_moments(
    video_path: str,
    min_duration: float = 60.0,
    max_duration: float = 90.0,
    min_score: float = 0.80,
    max_clips: Optional[int] = None
) -> List[ViralMoment]:
    """
    Fonction utilitaire pour détecter les moments viraux
    
    Args:
        video_path: Chemin vers la vidéo
        min_duration: Durée minimum des clips (défaut: 60s)
        max_duration: Durée maximum des clips (défaut: 90s)
        min_score: Score minimum de viralité (0-1, défaut: 0.80)
        max_clips: Nombre max de clips (None = automatique)
    """
    detector = ViralMomentDetector(
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
        min_viral_score=min_score,
        max_clips=max_clips
    )
    return detector.analyze(video_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        moments = detect_viral_moments(sys.argv[1])
        for i, moment in enumerate(moments, 1):
            print(f"Clip {i}: {moment.start_time:.1f}s - {moment.end_time:.1f}s "
                  f"(score: {moment.score:.2f}, raison: {moment.reason})")
