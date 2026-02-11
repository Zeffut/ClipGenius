"""
Module de détection avancée des moments viraux
Analyse les émotions vocales, événements audio et patterns de parole
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from rich.console import Console

console = Console()


@dataclass
class AudioEvent:
    """Événement audio détecté"""
    time: float
    duration: float
    event_type: str  # 'laugh', 'applause', 'music', 'silence', 'exclamation', 'speech_burst'
    confidence: float
    description: str


@dataclass
class EmotionSegment:
    """Segment avec analyse émotionnelle"""
    start: float
    end: float
    energy_level: float  # 0-1, niveau d'énergie moyen
    speech_rate: float  # Mots par seconde estimé
    pitch_variation: float  # Variation de pitch (excitation)
    dominant_emotion: str  # 'neutral', 'excited', 'calm', 'intense'
    viral_potential: float  # 0-1


@dataclass
class AdvancedViralMoment:
    """Moment viral avec analyse avancée"""
    start_time: float
    end_time: float
    score: float
    reason: str
    audio_events: List[AudioEvent] = field(default_factory=list)
    emotion_profile: Optional[EmotionSegment] = None
    hook_strength: float = 0.5


class AdvancedAudioAnalyzer:
    """
    Analyseur audio avancé pour la détection de moments viraux.

    Fonctionnalités:
    - Détection des émotions vocales (excitation, calme, intensité)
    - Détection des événements audio (rires, applaudissements, musique)
    - Analyse de la vitesse de parole
    - Détection des silences dramatiques
    - Analyse des variations de pitch

    Modes:
    - Normal: analyse complète haute qualité
    - Quick: analyse rapide pour pré-configuration (sample_rate réduit)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        emotion_window: float = 5.0,  # Fenêtre d'analyse émotionnelle en secondes
        quick_mode: bool = False
    ):
        """
        Initialise l'analyseur audio avancé.

        Args:
            sample_rate: Taux d'échantillonnage (défaut: 22050)
            hop_length: Pas d'analyse (défaut: 512)
            emotion_window: Fenêtre d'analyse émotionnelle en secondes
            quick_mode: Mode rapide avec paramètres optimisés pour la vitesse
        """
        self.quick_mode = quick_mode

        if quick_mode:
            # Paramètres optimisés pour l'analyse rapide
            self.sample_rate = 11025
            self.hop_length = 1024
            self.emotion_window = 10.0
        else:
            self.sample_rate = sample_rate
            self.hop_length = hop_length
            self.emotion_window = emotion_window
    
    def analyze_audio(
        self,
        audio_path: str,
        video_duration: float
    ) -> Tuple[List[AudioEvent], List[EmotionSegment]]:
        """
        Analyse complète de l'audio.
        
        Args:
            audio_path: Chemin vers le fichier audio
            video_duration: Durée totale de la vidéo
            
        Returns:
            Tuple (événements audio, segments émotionnels)
        """
        console.print("[cyan]Analyse audio avancée en cours...[/cyan]")
        
        try:
            # Charger l'audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 1. Détecter les événements audio
            events = self._detect_audio_events(y, sr)
            console.print(f"  [dim]{len(events)} événement(s) audio détecté(s)[/dim]")
            
            # 2. Analyser les émotions par segment
            emotions = self._analyze_emotions(y, sr, video_duration)
            console.print(f"  [dim]{len(emotions)} segment(s) émotionnel(s) analysé(s)[/dim]")
            
            return events, emotions

        except Exception as e:
            console.print(f"[yellow]Erreur analyse audio avancée: {e}[/yellow]")
            return [], []

    def analyze_sample(
        self,
        audio_path: str,
        max_duration: float = 180.0,
        total_duration: Optional[float] = None
    ) -> Tuple[List[AudioEvent], List[EmotionSegment]]:
        """
        Analyse un échantillon partiel de l'audio (mode rapide).

        Utile pour la pré-analyse et l'auto-configuration.

        Args:
            audio_path: Chemin vers le fichier audio
            max_duration: Durée maximale à analyser (défaut: 180s)
            total_duration: Durée totale de la vidéo (pour le contexte)

        Returns:
            Tuple (événements audio, segments émotionnels)
        """
        if not self.quick_mode:
            console.print("[dim]Note: analyze_sample fonctionne mieux en quick_mode[/dim]")

        try:
            # Charger seulement une portion de l'audio
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=max_duration
            )

            actual_duration = len(y) / sr
            console.print(f"[dim]Analyse d'un échantillon de {actual_duration:.0f}s[/dim]")

            # Détecter les événements audio
            events = self._detect_audio_events(y, sr)

            # Analyser les émotions par segment
            emotions = self._analyze_emotions(y, sr, actual_duration)

            return events, emotions

        except Exception as e:
            console.print(f"[yellow]Erreur analyse échantillon: {e}[/yellow]")
            return [], []

    def _detect_audio_events(
        self,
        y: np.ndarray,
        sr: int
    ) -> List[AudioEvent]:
        """Détecte les événements audio distinctifs"""
        events = []
        
        # 1. Détecter les rires/applaudissements (haute fréquence + énergie variable)
        laugh_events = self._detect_laughter_applause(y, sr)
        events.extend(laugh_events)
        
        # 2. Détecter les silences dramatiques
        silence_events = self._detect_dramatic_silences(y, sr)
        events.extend(silence_events)
        
        # 3. Détecter les exclamations/cris
        exclamation_events = self._detect_exclamations(y, sr)
        events.extend(exclamation_events)
        
        # 4. Détecter les bursts de parole (moments très actifs)
        burst_events = self._detect_speech_bursts(y, sr)
        events.extend(burst_events)
        
        # Trier par temps
        events.sort(key=lambda e: e.time)
        
        return events
    
    def _detect_laughter_applause(
        self,
        y: np.ndarray,
        sr: int
    ) -> List[AudioEvent]:
        """
        Détecte les rires et applaudissements.
        
        Caractéristiques:
        - Haute énergie dans les hautes fréquences
        - Pattern rythmique irrégulier
        - Spectral flatness élevé
        """
        events = []
        
        # Calculer le spectral flatness (élevé pour le bruit/rires)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        flatness_times = librosa.frames_to_time(np.arange(len(flatness)), sr=sr, hop_length=self.hop_length)
        
        # Calculer l'énergie RMS
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Normaliser
        flatness_norm = (flatness - flatness.min()) / (flatness.max() - flatness.min() + 1e-8)
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        
        # Combiner: haute flatness + haute énergie = rire/applaudissement potentiel
        laugh_score = flatness_norm * rms_norm
        
        # Lisser pour éviter les faux positifs
        laugh_score_smooth = uniform_filter1d(laugh_score, size=10)
        
        # Trouver les pics
        peaks, properties = find_peaks(
            laugh_score_smooth,
            height=0.3,
            distance=sr // self.hop_length * 2,  # Au moins 2 secondes entre les pics
            prominence=0.1
        )
        
        for peak in peaks:
            if peak < len(flatness_times):
                events.append(AudioEvent(
                    time=float(flatness_times[peak]),
                    duration=1.5,
                    event_type='laugh_or_applause',
                    confidence=float(laugh_score_smooth[peak]),
                    description="Rire ou applaudissement détecté"
                ))
        
        return events
    
    def _detect_dramatic_silences(
        self,
        y: np.ndarray,
        sr: int
    ) -> List[AudioEvent]:
        """
        Détecte les silences dramatiques (pauses avant un moment fort).
        
        Un silence dramatique est suivi d'une augmentation d'énergie.
        """
        events = []
        
        # Calculer l'énergie RMS
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)
        
        # Normaliser
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        
        # Trouver les zones de faible énergie (silences)
        silence_threshold = 0.1
        min_silence_frames = int(sr / self.hop_length * 0.5)  # Au moins 0.5 seconde
        
        in_silence = False
        silence_start = 0
        
        for i, energy in enumerate(rms_norm):
            if energy < silence_threshold and not in_silence:
                in_silence = True
                silence_start = i
            elif energy >= silence_threshold and in_silence:
                in_silence = False
                silence_duration = i - silence_start
                
                if silence_duration >= min_silence_frames:
                    # Vérifier si c'est suivi d'une montée d'énergie
                    if i + 10 < len(rms_norm):
                        post_silence_energy = np.mean(rms_norm[i:i+10])
                        if post_silence_energy > 0.3:  # Montée significative
                            events.append(AudioEvent(
                                time=float(rms_times[silence_start]),
                                duration=float(rms_times[i] - rms_times[silence_start]),
                                event_type='dramatic_silence',
                                confidence=min(1.0, post_silence_energy),
                                description="Silence dramatique suivi d'un moment fort"
                            ))
        
        return events
    
    def _detect_exclamations(
        self,
        y: np.ndarray,
        sr: int
    ) -> List[AudioEvent]:
        """
        Détecte les exclamations et cris.
        
        Caractéristiques:
        - Pic d'énergie soudain
        - Haute fréquence fondamentale
        """
        events = []
        
        # Calculer l'énergie RMS
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)
        
        # Calculer la dérivée de l'énergie (changements soudains)
        rms_diff = np.diff(rms)
        rms_diff = np.concatenate([[0], rms_diff])
        
        # Normaliser
        rms_diff_norm = (rms_diff - rms_diff.min()) / (rms_diff.max() - rms_diff.min() + 1e-8)
        
        # Trouver les pics de changement (exclamations)
        peaks, _ = find_peaks(
            rms_diff_norm,
            height=0.5,  # Changement significatif
            distance=sr // self.hop_length * 1  # Au moins 1 seconde entre les pics
        )
        
        for peak in peaks:
            if peak < len(rms_times) and rms[peak] > np.mean(rms) * 1.5:
                events.append(AudioEvent(
                    time=float(rms_times[peak]),
                    duration=0.5,
                    event_type='exclamation',
                    confidence=float(rms_diff_norm[peak]),
                    description="Exclamation ou cri détecté"
                ))
        
        return events
    
    def _detect_speech_bursts(
        self,
        y: np.ndarray,
        sr: int
    ) -> List[AudioEvent]:
        """
        Détecte les bursts de parole (moments très actifs vocalement).
        
        Utilise l'onset strength pour détecter l'activité vocale.
        """
        events = []
        
        # Calculer l'onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=self.hop_length)
        
        # Calculer la densité d'onsets (activité vocale)
        window_size = int(sr / self.hop_length * 2)  # Fenêtre de 2 secondes
        onset_density = uniform_filter1d(onset_env, size=window_size)
        
        # Normaliser
        onset_density_norm = (onset_density - onset_density.min()) / (onset_density.max() - onset_density.min() + 1e-8)
        
        # Trouver les zones de haute activité
        peaks, _ = find_peaks(
            onset_density_norm,
            height=0.6,
            distance=sr // self.hop_length * 5,  # Au moins 5 secondes entre les pics
            prominence=0.15
        )
        
        for peak in peaks:
            if peak < len(onset_times):
                events.append(AudioEvent(
                    time=float(onset_times[peak]),
                    duration=3.0,
                    event_type='speech_burst',
                    confidence=float(onset_density_norm[peak]),
                    description="Moment de parole très actif"
                ))
        
        return events
    
    def _analyze_emotions(
        self,
        y: np.ndarray,
        sr: int,
        duration: float
    ) -> List[EmotionSegment]:
        """
        Analyse les émotions par segment temporel.
        """
        segments = []
        window_samples = int(self.emotion_window * sr)
        hop_samples = window_samples // 2  # 50% overlap
        
        for start_sample in range(0, len(y) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            segment_audio = y[start_sample:end_sample]
            
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            # Calculer les features émotionnelles
            energy = self._compute_energy(segment_audio)
            speech_rate = self._estimate_speech_rate(segment_audio, sr)
            pitch_var = self._compute_pitch_variation(segment_audio, sr)
            
            # Déterminer l'émotion dominante
            emotion, viral_potential = self._classify_emotion(energy, speech_rate, pitch_var)
            
            segments.append(EmotionSegment(
                start=start_time,
                end=end_time,
                energy_level=energy,
                speech_rate=speech_rate,
                pitch_variation=pitch_var,
                dominant_emotion=emotion,
                viral_potential=viral_potential
            ))
        
        return segments
    
    def _compute_energy(self, audio: np.ndarray) -> float:
        """Calcule le niveau d'énergie normalisé"""
        rms = np.sqrt(np.mean(audio ** 2))
        # Normaliser entre 0 et 1 (approximatif)
        return min(1.0, rms * 10)
    
    def _estimate_speech_rate(self, audio: np.ndarray, sr: int) -> float:
        """Estime la vitesse de parole via les onsets"""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        # Compter les onsets significatifs
        peaks, _ = find_peaks(onset_env, height=np.mean(onset_env))
        # Convertir en "syllabes par seconde" approximatif
        duration = len(audio) / sr
        return len(peaks) / duration if duration > 0 else 0
    
    def _compute_pitch_variation(self, audio: np.ndarray, sr: int) -> float:
        """Calcule la variation de pitch (indicateur d'excitation)"""
        try:
            # Extraire le pitch avec librosa
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            
            # Prendre les pitches les plus forts
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 1:
                # Coefficient de variation du pitch
                return float(np.std(pitch_values) / (np.mean(pitch_values) + 1e-8))
            return 0.0
        except Exception:
            return 0.0
    
    def _classify_emotion(
        self,
        energy: float,
        speech_rate: float,
        pitch_var: float
    ) -> Tuple[str, float]:
        """
        Classifie l'émotion et calcule le potentiel viral.
        
        Returns:
            Tuple (emotion, viral_potential)
        """
        # Calculer un score de potentiel viral basé sur les features
        viral_potential = 0.3  # Base
        
        # Haute énergie = plus viral
        if energy > 0.6:
            viral_potential += 0.25
            emotion = "excited"
        elif energy < 0.2:
            emotion = "calm"
        else:
            emotion = "neutral"
        
        # Vitesse de parole élevée = plus excitant
        if speech_rate > 4:  # Plus de 4 "syllabes" par seconde
            viral_potential += 0.15
            if emotion == "neutral":
                emotion = "intense"
        
        # Variation de pitch élevée = plus engageant
        if pitch_var > 0.3:
            viral_potential += 0.15
            if emotion == "neutral":
                emotion = "excited"
        
        # Combiner les signaux
        if energy > 0.7 and speech_rate > 5 and pitch_var > 0.4:
            emotion = "very_excited"
            viral_potential = min(1.0, viral_potential + 0.2)
        
        return emotion, min(1.0, viral_potential)
    
    def get_viral_moments_from_analysis(
        self,
        events: List[AudioEvent],
        emotions: List[EmotionSegment],
        min_duration: float = 60.0,
        max_duration: float = 90.0,
        video_duration: float = 0
    ) -> List[AdvancedViralMoment]:
        """
        Génère des moments viraux à partir de l'analyse audio avancée.
        
        Args:
            events: Événements audio détectés
            emotions: Segments émotionnels
            min_duration: Durée minimum des clips
            max_duration: Durée maximum des clips
            video_duration: Durée totale de la vidéo
            
        Returns:
            Liste de moments viraux avec analyse avancée
        """
        moments = []
        
        # 1. Identifier les pics émotionnels
        high_emotion_segments = [
            e for e in emotions 
            if e.viral_potential >= 0.5
        ]
        
        # 2. Identifier les événements importants
        important_events = [
            e for e in events 
            if e.confidence >= 0.4
        ]
        
        # 3. Combiner pour créer des moments
        # Grouper les segments émotionnels proches
        if high_emotion_segments:
            current_start = high_emotion_segments[0].start
            current_events = []
            
            for seg in high_emotion_segments:
                # Ajouter les événements dans ce segment
                seg_events = [
                    e for e in important_events 
                    if seg.start <= e.time <= seg.end
                ]
                
                # Si le segment est proche du précédent, l'étendre
                if seg.start - current_start < max_duration:
                    current_events.extend(seg_events)
                else:
                    # Créer un moment avec les segments accumulés
                    if current_events:
                        end_time = min(current_start + max_duration, video_duration)
                        
                        # Calculer le score moyen
                        avg_score = np.mean([e.confidence for e in current_events]) if current_events else 0.5
                        
                        moments.append(AdvancedViralMoment(
                            start_time=current_start,
                            end_time=end_time,
                            score=avg_score,
                            reason=f"Segment émotionnel avec {len(current_events)} événement(s)",
                            audio_events=current_events
                        ))
                    
                    # Réinitialiser
                    current_start = seg.start
                    current_events = seg_events
            
            # Dernier moment
            if current_events and current_start + min_duration <= video_duration:
                end_time = min(current_start + max_duration, video_duration)
                avg_score = np.mean([e.confidence for e in current_events])
                
                moments.append(AdvancedViralMoment(
                    start_time=current_start,
                    end_time=end_time,
                    score=avg_score,
                    reason=f"Segment émotionnel avec {len(current_events)} événement(s)",
                    audio_events=current_events
                ))
        
        # 4. Ajouter des moments basés uniquement sur les événements importants
        for event in important_events:
            # Vérifier si cet événement n'est pas déjà couvert
            is_covered = any(
                m.start_time <= event.time <= m.end_time 
                for m in moments
            )
            
            if not is_covered and event.confidence >= 0.6:
                # Créer un moment centré sur cet événement
                start = max(0, event.time - min_duration * 0.3)
                end = min(video_duration, start + min_duration)
                
                if end - start >= min_duration * 0.8:
                    moments.append(AdvancedViralMoment(
                        start_time=start,
                        end_time=end,
                        score=event.confidence,
                        reason=event.description,
                        audio_events=[event]
                    ))
        
        # Trier par score décroissant
        moments.sort(key=lambda m: m.score, reverse=True)
        
        return moments


def analyze_audio_advanced(
    audio_path: str,
    video_duration: float,
    min_duration: float = 60.0,
    max_duration: float = 90.0
) -> List[AdvancedViralMoment]:
    """
    Fonction utilitaire pour l'analyse audio avancée.
    
    Args:
        audio_path: Chemin vers le fichier audio
        video_duration: Durée de la vidéo
        min_duration: Durée minimum des clips
        max_duration: Durée maximum des clips
        
    Returns:
        Liste de moments viraux avec analyse avancée
    """
    analyzer = AdvancedAudioAnalyzer()
    events, emotions = analyzer.analyze_audio(audio_path, video_duration)
    
    return analyzer.get_viral_moments_from_analysis(
        events, emotions, min_duration, max_duration, video_duration
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Test avec un fichier audio
        from moviepy import VideoFileClip
        
        video_path = sys.argv[1]
        video = None
        try:
            video = VideoFileClip(video_path)
            
            # Extraire l'audio temporairement
            audio_path = "temp_audio.wav"
            if video.audio:
                video.audio.write_audiofile(audio_path, fps=22050, logger=None)
            
            moments = analyze_audio_advanced(audio_path, video.duration)
            
            print(f"\n{len(moments)} moments viraux détectés:")
            for i, m in enumerate(moments, 1):
                print(f"  {i}. {m.start_time:.1f}s - {m.end_time:.1f}s (score: {m.score:.2f})")
                print(f"     {m.reason}")
            
            Path(audio_path).unlink(missing_ok=True)
        finally:
            if video:
                video.close()
