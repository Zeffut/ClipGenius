"""
Module d'auto-configuration intelligente pour ClipGenius

Analyse automatiquement la vidéo source et génère une configuration
optimale basée sur le type de contenu détecté.

Usage:
    from src.auto_config import AutoConfigurator

    configurator = AutoConfigurator()
    config = configurator.analyze_and_configure(video_path, platform="tiktok")
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

console = Console()


# =============================================================================
# DATACLASSES
# =============================================================================

class ContentType(Enum):
    """Types de contenu détectables"""
    PODCAST = "podcast"
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    VLOG = "vlog"
    COMEDY = "comedy"
    GAMING = "gaming"
    MUSIC = "music"
    NEWS = "news"
    MOTIVATIONAL = "motivational"
    UNKNOWN = "unknown"


class EmotionProfile(Enum):
    """Profils émotionnels"""
    EXCITED = "excited"
    VERY_EXCITED = "very_excited"
    CALM = "calm"
    INTENSE = "intense"
    NEUTRAL = "neutral"


@dataclass
class EnergyProfile:
    """Profil énergétique de l'audio"""
    average: float  # Moyenne de l'énergie (0-1)
    variance: float  # Variance (indicateur de dynamisme)
    peak_count: int  # Nombre de pics d'énergie
    peak_times: List[float] = field(default_factory=list)
    calm_ratio: float = 0.5  # Ratio de segments calmes


@dataclass
class AudioEventsSummary:
    """Résumé des événements audio détectés"""
    laughter_count: int = 0
    applause_count: int = 0
    dramatic_silence_count: int = 0
    exclamation_count: int = 0
    speech_burst_count: int = 0


@dataclass
class SpeechCharacteristics:
    """Caractéristiques de la parole"""
    rate: float  # Vitesse moyenne (syllabes/seconde estimées)
    density: float  # Densité de parole (0-1)
    pitch_variation: float  # Variation de pitch moyenne


@dataclass
class QuickAnalysisResult:
    """Résultats de l'analyse rapide"""
    # Profil énergétique
    energy_profile: EnergyProfile

    # Émotion dominante
    dominant_emotion: EmotionProfile
    excitement_peaks: int

    # Type de contenu
    content_type: ContentType
    content_confidence: float  # 0-1

    # Événements audio
    audio_events: AudioEventsSummary

    # Caractéristiques de parole
    speech: SpeechCharacteristics

    # Durée analysée
    analyzed_duration: float
    total_duration: float


@dataclass
class GeneratedConfig:
    """Configuration générée automatiquement"""
    # Durées
    min_duration: float
    max_duration: float

    # Score viral
    min_viral_score: float

    # Effets visuels
    zoom_style: str  # 'pulse', 'breathing', 'ease_out', 'ease_in_out'
    zoom_factor: float
    color_grading: str  # 'warm', 'cool', 'vibrant', 'cinematic'
    sharpening_strength: float
    vignette_enabled: bool
    vignette_strength: float

    # Sous-titres
    subtitle_theme: str  # 'viral', 'neon', 'minimal', 'professional'
    subtitle_max_words: int
    subtitle_emojis: bool

    # Options de traitement
    smart_crop: bool
    optimize_hooks: bool
    advanced_audio: bool

    # Métadonnées
    detected_content_type: str
    content_confidence: float
    reasoning: List[str] = field(default_factory=list)


# =============================================================================
# DURÉES OPTIMALES PAR TYPE DE CONTENU
# =============================================================================

# Constantes de configuration
DURATION_SHORT_MIN = 15       # Durée minimale courte (secondes)
DURATION_SHORT_MID = 20       # Durée minimale courte-moyenne (secondes)
DURATION_MEDIUM_MIN = 30      # Durée minimale moyenne (secondes)
DURATION_MEDIUM_MAX = 45      # Durée maximale moyenne (secondes)
DURATION_LONG_MIN = 60        # Durée minimale longue (secondes)
DURATION_LONG_MID = 75        # Durée maximale longue-moyenne (secondes)
DURATION_LONG_MAX = 90        # Durée maximale longue (secondes)

CONTENT_DURATION_MAP = {
    ContentType.COMEDY: (DURATION_SHORT_MIN, DURATION_MEDIUM_MAX),
    ContentType.MUSIC: (DURATION_SHORT_MIN, DURATION_MEDIUM_MIN),
    ContentType.TUTORIAL: (DURATION_MEDIUM_MIN, DURATION_LONG_MIN),
    ContentType.VLOG: (DURATION_MEDIUM_MIN, DURATION_LONG_MIN),
    ContentType.PODCAST: (DURATION_LONG_MIN, DURATION_LONG_MAX),
    ContentType.INTERVIEW: (DURATION_MEDIUM_MAX, DURATION_LONG_MID),
    ContentType.MOTIVATIONAL: (DURATION_SHORT_MID, DURATION_MEDIUM_MAX),
    ContentType.NEWS: (DURATION_MEDIUM_MIN, DURATION_LONG_MIN),
    ContentType.UNKNOWN: (DURATION_MEDIUM_MIN, DURATION_LONG_MIN),
}

# Ajustements par plateforme
PLATFORM_DURATION_ADJUSTMENTS = {
    'tiktok': (-5, -15),  # Plus court
    'reels': (0, 0),      # Standard
    'shorts': (-5, -5),   # Légèrement plus court
    'all': (0, 0),
}


# =============================================================================
# QUICK ANALYZER
# =============================================================================

class QuickAnalyzer:
    """
    Analyseur rapide pour la pré-analyse de contenu.

    Utilise des paramètres optimisés pour une analyse rapide:
    - Sample rate réduit (11025 Hz)
    - Hop length augmenté
    - Fenêtre d'émotion élargie
    """

    def __init__(
        self,
        sample_rate: int = 11025,
        hop_length: int = 1024,
        emotion_window: float = 10.0,
        max_analysis_duration: float = 180.0
    ):
        """
        Initialise l'analyseur rapide.

        Args:
            sample_rate: Taux d'échantillonnage (défaut: 11025 Hz pour rapidité)
            hop_length: Pas d'analyse (défaut: 1024 pour rapidité)
            emotion_window: Fenêtre d'analyse émotionnelle en secondes
            max_analysis_duration: Durée maximale à analyser
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.emotion_window = emotion_window
        self.max_analysis_duration = max_analysis_duration

    def analyze(
        self,
        audio_path: str,
        total_duration: Optional[float] = None,
        transcription_text: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> QuickAnalysisResult:
        """
        Effectue une analyse rapide de l'audio + transcription.

        Args:
            audio_path: Chemin vers le fichier audio
            total_duration: Durée totale de la vidéo (optionnel)
            transcription_text: Texte transcrit (optionnel, améliore précision)
            progress_callback: Callback pour la progression (percent, message)

        Returns:
            QuickAnalysisResult avec toutes les métriques
        """
        def report_progress(percent: int, message: str):
            """Helper pour reporter la progression"""
            if progress_callback:
                progress_callback(percent, message)

        # Charger l'audio avec la durée limitée
        report_progress(0, "Chargement de l'audio...")
        y, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=self.max_analysis_duration
        )

        actual_duration = len(y) / sr
        report_progress(10, "Audio chargé, analyse énergétique...")

        # 1. Analyser le profil énergétique
        energy_profile = self._analyze_energy(y, sr)
        report_progress(30, "Profil énergétique calculé")

        # 2. Détecter les événements audio
        report_progress(35, "Détection des événements audio...")
        audio_events = self._detect_events(y, sr)
        report_progress(55, "Événements audio détectés")

        # 3. Analyser les caractéristiques de parole
        report_progress(60, "Analyse des caractéristiques vocales...")
        speech = self._analyze_speech(y, sr)
        report_progress(75, "Caractéristiques vocales analysées")

        # 4. Déterminer l'émotion dominante
        report_progress(80, "Classification émotionnelle...")
        dominant_emotion, excitement_peaks = self._classify_emotion(
            energy_profile, speech
        )

        # 5. Classifier le type de contenu (avec transcription si disponible)
        report_progress(85, "Classification du type de contenu...")
        content_type, confidence = self._classify_content(
            energy_profile, audio_events, speech, dominant_emotion,
            transcription_text=transcription_text
        )
        report_progress(100, f"Analyse terminée: {content_type.value}")

        return QuickAnalysisResult(
            energy_profile=energy_profile,
            dominant_emotion=dominant_emotion,
            excitement_peaks=excitement_peaks,
            content_type=content_type,
            content_confidence=confidence,
            audio_events=audio_events,
            speech=speech,
            analyzed_duration=actual_duration,
            total_duration=total_duration or actual_duration
        )

    def _analyze_energy(self, y: np.ndarray, sr: int) -> EnergyProfile:
        """Analyse le profil énergétique"""
        # Calculer l'énergie RMS
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=self.hop_length
        )

        # Normaliser
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

        # Calculer les statistiques
        average = float(np.mean(rms_norm))
        variance = float(np.var(rms_norm))

        # Trouver les pics d'énergie
        peaks, _ = find_peaks(
            rms_norm,
            height=0.6,
            distance=sr // self.hop_length * 3
        )

        peak_times = [float(rms_times[p]) for p in peaks if p < len(rms_times)]

        # Calculer le ratio de segments calmes
        calm_threshold = 0.3
        calm_ratio = float(np.mean(rms_norm < calm_threshold))

        return EnergyProfile(
            average=average,
            variance=variance,
            peak_count=len(peaks),
            peak_times=peak_times,
            calm_ratio=calm_ratio
        )

    def _detect_events(self, y: np.ndarray, sr: int) -> AudioEventsSummary:
        """Détecte les événements audio de manière simplifiée"""
        summary = AudioEventsSummary()

        # Spectral flatness pour rires/applaudissements
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Normaliser
        flatness_norm = (flatness - flatness.min()) / (flatness.max() - flatness.min() + 1e-8)
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

        # Score de rire/applaudissement
        laugh_score = flatness_norm * rms_norm
        laugh_score_smooth = uniform_filter1d(laugh_score, size=10)

        # Compter les pics
        laugh_peaks, _ = find_peaks(laugh_score_smooth, height=0.3, distance=sr // self.hop_length * 2)
        # Répartir entre rires et applaudissements (approximatif)
        summary.laughter_count = len(laugh_peaks) // 2
        summary.applause_count = len(laugh_peaks) - summary.laughter_count

        # Silences dramatiques
        silence_count = 0
        min_silence_frames = int(sr / self.hop_length * 0.5)
        in_silence = False
        silence_start = 0

        for i, energy in enumerate(rms_norm):
            if energy < 0.1 and not in_silence:
                in_silence = True
                silence_start = i
            elif energy >= 0.1 and in_silence:
                in_silence = False
                if i - silence_start >= min_silence_frames:
                    # Vérifier si suivi d'une montée
                    if i + 10 < len(rms_norm) and np.mean(rms_norm[i:i+10]) > 0.3:
                        silence_count += 1

        summary.dramatic_silence_count = silence_count

        # Exclamations (changements soudains d'énergie)
        rms_diff = np.diff(rms_norm)
        rms_diff_norm = (rms_diff - rms_diff.min()) / (rms_diff.max() - rms_diff.min() + 1e-8)
        exclaim_peaks, _ = find_peaks(rms_diff_norm, height=0.5, distance=sr // self.hop_length)
        summary.exclamation_count = len(exclaim_peaks)

        # Speech bursts
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        window_size = int(sr / self.hop_length * 2)
        onset_density = uniform_filter1d(onset_env, size=window_size)
        onset_density_norm = (onset_density - onset_density.min()) / (onset_density.max() - onset_density.min() + 1e-8)
        burst_peaks, _ = find_peaks(onset_density_norm, height=0.6, distance=sr // self.hop_length * 5)
        summary.speech_burst_count = len(burst_peaks)

        return summary

    def _analyze_speech(self, y: np.ndarray, sr: int) -> SpeechCharacteristics:
        """Analyse les caractéristiques de parole"""
        # Estimer la vitesse de parole via les onsets
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        peaks, _ = find_peaks(onset_env, height=np.mean(onset_env) * 0.8)

        duration = len(y) / sr
        rate = len(peaks) / duration if duration > 0 else 0

        # Densité de parole (basée sur l'énergie RMS)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        density = float(np.mean(rms_norm > 0.15))

        # Variation de pitch
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                if pitches[idx, t] > 0:
                    pitch_values.append(pitches[idx, t])

            if len(pitch_values) > 10:
                pitch_variation = float(np.std(pitch_values) / (np.mean(pitch_values) + 1e-8))
            else:
                pitch_variation = 0.0
        except Exception:
            pitch_variation = 0.0

        return SpeechCharacteristics(
            rate=rate,
            density=density,
            pitch_variation=pitch_variation
        )

    def _classify_emotion(
        self,
        energy: EnergyProfile,
        speech: SpeechCharacteristics
    ) -> Tuple[EmotionProfile, int]:
        """Classifie l'émotion dominante"""
        # Calculer le nombre de pics d'excitation
        excitement_peaks = energy.peak_count

        # Déterminer l'émotion
        if energy.average > 0.6 and speech.rate > 4 and speech.pitch_variation > 0.3:
            return EmotionProfile.VERY_EXCITED, excitement_peaks
        elif energy.average > 0.5 or speech.rate > 3.5 or speech.pitch_variation > 0.25:
            return EmotionProfile.EXCITED, excitement_peaks
        elif energy.calm_ratio > 0.7:
            return EmotionProfile.CALM, excitement_peaks
        elif energy.variance > 0.15 and speech.rate > 3:
            return EmotionProfile.INTENSE, excitement_peaks
        else:
            return EmotionProfile.NEUTRAL, excitement_peaks

    def _classify_content(
        self,
        energy: EnergyProfile,
        events: AudioEventsSummary,
        speech: SpeechCharacteristics,
        emotion: EmotionProfile,
        transcription_text: Optional[str] = None
    ) -> Tuple[ContentType, float]:
        """
        Classifie le type de contenu en utilisant audio ET transcription.

        Args:
            transcription_text: Texte transcrit (optionnel, améliore la précision)
        """
        scores = {ct: 0.0 for ct in ContentType}

        # === ANALYSE PAR TRANSCRIPTION (si disponible) ===
        if transcription_text:
            text_lower = transcription_text.lower()

            # Gaming: mots-clés gaming (élargi pour Minecraft)
            gaming_keywords = [
                # Combat/Action
                'kill', 'headshot', 'victory', 'victoire', 'win', 'lose', 'perdu',
                'weapon', 'arme', 'skill', 'combo', 'damage', 'dégâts', 'attack', 'defend',

                # Gaming général
                'game', 'play', 'joue', 'playing', 'gamer', 'gaming',
                'level', 'niveau', 'boss', 'enemy', 'ennemi',
                'gg', 'ez', 'noob', 'pro', 'clutch', 'ace', 'pentakill',

                # Jeux spécifiques
                'fortnite', 'minecraft', 'valorant', 'league', 'cod', 'warzone',
                'roblox', 'terraria', 'rust', 'apex', 'overwatch',

                # Streaming
                'stream', 'twitch', 'youtube gaming', 'speedrun', 'gameplay',

                # Minecraft spécifique
                'craft', 'crafting', 'mine', 'mining', 'build', 'building', 'builder',
                'block', 'blocks', 'diamond', 'diamonds', 'creeper', 'enderman',
                'nether', 'portal', 'enchant', 'enchanting', 'cave', 'caves',
                'village', 'villager', 'zombie', 'skeleton', 'spawn', 'biome',
                'redstone', 'piston', 'tnt', 'chest', 'pickaxe', 'sword', 'armor',
                'survival', 'creative', 'hardcore', 'adventure', 'dungeon',
                'mob', 'mobs', 'server', 'multiplayer', 'world', 'seed'
            ]
            gaming_count = sum(1 for kw in gaming_keywords if kw in text_lower)

            # Seuil réduit : 2-3 mots-clés suffisent pour Minecraft
            if gaming_count >= 3:
                scores[ContentType.GAMING] += 0.7  # Boost augmenté
                console.print(f"[dim]🎮 Gaming détecté: {gaming_count} mots-clés[/dim]")
            elif gaming_count >= 2:
                scores[ContentType.GAMING] += 0.4  # Boost modéré

            # Comedy: expressions humoristiques
            comedy_keywords = [
                'mdr', 'lol', 'ptdr', 'mort de rire', 'hilarant', 'dr\u00f4le',
                'blague', 'joke', 'funny', 'haha', 'hehe', 'rigole',
                'sketch', 'parodie', 'imitation', 'troll'
            ]
            comedy_count = sum(1 for kw in comedy_keywords if kw in text_lower)
            if comedy_count >= 3:
                scores[ContentType.COMEDY] += 0.5

            # Podcast/Interview: marqueurs de conversation
            podcast_keywords = [
                'interview', 'question', 'r\u00e9ponse', 'discussion', 'parle',
                '\u00e9pisode', 'podcast', 'invit\u00e9', 'guest', 'aujourd\'hui',
                'pense que', 'je crois', '\u00e0 mon avis', 'perspective',
                'exp\u00e9rience', 'carri\u00e8re', 'parcours'
            ]
            podcast_count = sum(1 for kw in podcast_keywords if kw in text_lower)
            if podcast_count >= 4:
                scores[ContentType.PODCAST] += 0.5
                scores[ContentType.INTERVIEW] += 0.4

            # Tutorial: vocabulaire instructif
            tutorial_keywords = [
                'comment', 'tutoriel', 'tutorial', 'apprendre', 'montrer',
                '\u00e9tape', 'step', 'facile', 'simple', 'expliquer',
                'd\'abord', 'ensuite', 'puis', 'finalement', 'voil\u00e0',
                'faire', 'cr\u00e9er', 'installer', 'utiliser'
            ]
            tutorial_count = sum(1 for kw in tutorial_keywords if kw in text_lower)
            if tutorial_count >= 5:
                scores[ContentType.TUTORIAL] += 0.6

            # Motivational: langage inspirant
            motivational_keywords = [
                'motivation', 'inspir', 'r\u00e9ussir', 'succ\u00e8s', 'objectif',
                'r\u00eave', 'dream', 'possible', 'impossible', 'force',
                'courage', 'd\u00e9termination', 'persev\u00e9rance', 'battre',
                'champion', 'winner', 'gagnant', 'victoire'
            ]
            motivational_count = sum(1 for kw in motivational_keywords if kw in text_lower)
            if motivational_count >= 3:
                scores[ContentType.MOTIVATIONAL] += 0.5

        # === ANALYSE AUDIO (comme avant) ===

        # Comedy: beaucoup de rires, énergie variable, exclamations
        if events.laughter_count >= 3:
            scores[ContentType.COMEDY] += 0.4
        if events.exclamation_count >= 3:
            scores[ContentType.COMEDY] += 0.2
        if energy.variance > 0.15:
            scores[ContentType.COMEDY] += 0.1

        # Podcast: énergie stable, densité de parole élevée, peu de pics
        if energy.variance < 0.1:
            scores[ContentType.PODCAST] += 0.3
        if speech.density > 0.6:
            scores[ContentType.PODCAST] += 0.2
        if energy.peak_count < 5:
            scores[ContentType.PODCAST] += 0.15
        if energy.calm_ratio > 0.5:
            scores[ContentType.PODCAST] += 0.1

        # Interview: similaire au podcast mais plus de variations
        if 0.08 < energy.variance < 0.18:
            scores[ContentType.INTERVIEW] += 0.25
        if speech.density > 0.5:
            scores[ContentType.INTERVIEW] += 0.2
        if events.dramatic_silence_count >= 2:
            scores[ContentType.INTERVIEW] += 0.15

        # Tutorial: parole régulière, énergie moyenne, peu d'exclamations
        if 0.35 < speech.density < 0.7:
            scores[ContentType.TUTORIAL] += 0.25
        if 2 < speech.rate < 4:
            scores[ContentType.TUTORIAL] += 0.2
        if events.exclamation_count < 2:
            scores[ContentType.TUTORIAL] += 0.15

        # Vlog: énergie variable, discours naturel
        if 0.1 < energy.variance < 0.25:
            scores[ContentType.VLOG] += 0.2
        if emotion in [EmotionProfile.EXCITED, EmotionProfile.NEUTRAL]:
            scores[ContentType.VLOG] += 0.15

        # Music: énergie soutenue, peu de parole
        if speech.density < 0.3:
            scores[ContentType.MUSIC] += 0.35
        if energy.average > 0.5:
            scores[ContentType.MUSIC] += 0.2

        # News: parole dense, énergie stable, peu d'émotions
        if speech.density > 0.7:
            scores[ContentType.NEWS] += 0.2
        if energy.variance < 0.08:
            scores[ContentType.NEWS] += 0.2
        if emotion == EmotionProfile.NEUTRAL:
            scores[ContentType.NEWS] += 0.15

        # Motivational: haute énergie, variations de pitch élevées
        if speech.pitch_variation > 0.3:
            scores[ContentType.MOTIVATIONAL] += 0.3
        if emotion in [EmotionProfile.EXCITED, EmotionProfile.VERY_EXCITED]:
            scores[ContentType.MOTIVATIONAL] += 0.25
        if events.dramatic_silence_count >= 1:
            scores[ContentType.MOTIVATIONAL] += 0.1

        # Trouver le type avec le score maximum (analyse audio+mots-clés)
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Normaliser la confiance
        total = sum(scores.values())
        if total > 0:
            confidence = best_score / total
        else:
            confidence = 0.0

        # Si confiance trop basse, retourner UNKNOWN
        if confidence < 0.2 or best_score < 0.3:
            audio_type = ContentType.UNKNOWN
            audio_confidence = confidence
        else:
            audio_type = best_type
            audio_confidence = min(1.0, confidence * 1.5)  # Boost légèrement la confiance

        # ✨ NOUVEAU: Utiliser le LLM pour affiner la détection si transcription disponible
        if transcription_text and len(transcription_text) > 100:
            try:
                final_type, final_confidence, llm_reasoning = detect_content_type_with_llm(
                    transcription_text=transcription_text,
                    audio_based_type=audio_type,
                    audio_confidence=audio_confidence
                )
                console.print(f"[cyan]✨ Détection combinée: {final_type.value} ({final_confidence:.0%})[/cyan]")
                return final_type, final_confidence
            except Exception as e:
                console.print(f"[yellow]⚠️ Erreur LLM, utilisation analyse audio: {e}[/yellow]")
                return audio_type, audio_confidence
        else:
            # Pas de transcription ou trop courte → utiliser uniquement l'audio
            return audio_type, audio_confidence


# =============================================================================
# CONFIG GENERATOR
# =============================================================================

class ConfigGenerator:
    """
    Génère une configuration optimale basée sur l'analyse.
    """

    def generate(
        self,
        analysis: QuickAnalysisResult,
        platform: str = "reels"
    ) -> GeneratedConfig:
        """
        Génère une configuration basée sur l'analyse.

        Args:
            analysis: Résultat de l'analyse rapide
            platform: Plateforme cible

        Returns:
            GeneratedConfig avec tous les paramètres optimisés
        """
        reasoning = []

        # 1. Durées basées sur le contenu et la plateforme
        base_min, base_max = CONTENT_DURATION_MAP.get(
            analysis.content_type,
            CONTENT_DURATION_MAP[ContentType.UNKNOWN]
        )

        # Ajustement plateforme
        adj_min, adj_max = PLATFORM_DURATION_ADJUSTMENTS.get(platform, (0, 0))
        min_duration = max(15, base_min + adj_min)
        max_duration = max(min_duration + 15, base_max + adj_max)

        reasoning.append(
            f"Durées: {min_duration:.0f}-{max_duration:.0f}s "
            f"({analysis.content_type.value} + {platform})"
        )

        # 2. Score viral minimum basé sur la confiance
        if analysis.content_confidence >= 0.6:
            min_score = 0.55
            reasoning.append("Score minimum: 0.55 (haute confiance)")
        else:
            min_score = 0.65
            reasoning.append("Score minimum: 0.65 (confiance modérée)")

        # 3. Effets visuels basés sur le profil émotionnel
        zoom_style, zoom_factor = self._determine_zoom(analysis, reasoning)
        color_grading = self._determine_color_grading(analysis, reasoning)
        sharpening = self._determine_sharpening(analysis, reasoning)
        vignette_enabled, vignette_strength = self._determine_vignette(analysis, reasoning)

        # 4. Style de sous-titres
        theme, max_words, emojis = self._determine_subtitles(analysis, reasoning)

        # 5. Options de traitement
        smart_crop = True
        optimize_hooks = analysis.content_type != ContentType.NEWS
        advanced_audio = True

        return GeneratedConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_viral_score=min_score,
            zoom_style=zoom_style,
            zoom_factor=zoom_factor,
            color_grading=color_grading,
            sharpening_strength=sharpening,
            vignette_enabled=vignette_enabled,
            vignette_strength=vignette_strength,
            subtitle_theme=theme,
            subtitle_max_words=max_words,
            subtitle_emojis=emojis,
            smart_crop=smart_crop,
            optimize_hooks=optimize_hooks,
            advanced_audio=advanced_audio,
            detected_content_type=analysis.content_type.value,
            content_confidence=analysis.content_confidence,
            reasoning=reasoning
        )

    def _determine_zoom(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> Tuple[str, float]:
        """Détermine le style et facteur de zoom"""
        energy = analysis.energy_profile
        emotion = analysis.dominant_emotion

        if energy.variance > 0.4:
            reasoning.append("Zoom: pulse (1.06x) - contenu très dynamique")
            return "pulse", 1.06
        elif energy.calm_ratio > 0.7:
            reasoning.append("Zoom: breathing (1.03x) - contenu calme")
            return "breathing", 1.03
        elif emotion in [EmotionProfile.EXCITED, EmotionProfile.VERY_EXCITED]:
            reasoning.append("Zoom: pulse (1.05x) - haute énergie")
            return "pulse", 1.05
        else:
            reasoning.append("Zoom: ease_out (1.04x) - standard")
            return "ease_out", 1.04

    def _determine_color_grading(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> str:
        """Détermine le style de color grading"""
        emotion = analysis.dominant_emotion
        content = analysis.content_type

        if emotion in [EmotionProfile.EXCITED, EmotionProfile.VERY_EXCITED]:
            reasoning.append("Couleurs: vibrant - haute énergie")
            return "vibrant"
        elif content in [ContentType.PODCAST, ContentType.INTERVIEW]:
            reasoning.append("Couleurs: warm - conversation")
            return "warm"
        elif emotion == EmotionProfile.CALM:
            reasoning.append("Couleurs: cool - contenu calme")
            return "cool"
        elif content == ContentType.NEWS:
            reasoning.append("Couleurs: cool - professionnel")
            return "cool"
        elif analysis.audio_events.dramatic_silence_count >= 2:
            reasoning.append("Couleurs: cinematic - silences dramatiques")
            return "cinematic"
        else:
            reasoning.append("Couleurs: warm - standard")
            return "warm"

    def _determine_sharpening(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> float:
        """Détermine la force du sharpening"""
        if analysis.excitement_peaks > 5:
            reasoning.append("Netteté: 0.35 - contenu dynamique")
            return 0.35
        elif analysis.energy_profile.calm_ratio > 0.6:
            reasoning.append("Netteté: 0.20 - contenu calme")
            return 0.20
        else:
            reasoning.append("Netteté: 0.28 - standard")
            return 0.28

    def _determine_vignette(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> Tuple[bool, float]:
        """Détermine si la vignette est activée et sa force"""
        events = analysis.audio_events

        if events.dramatic_silence_count >= 2:
            reasoning.append("Vignette: activée (0.15) - silences dramatiques")
            return True, 0.15
        elif analysis.content_type in [ContentType.PODCAST, ContentType.INTERVIEW]:
            reasoning.append("Vignette: activée (0.10) - focus conversation")
            return True, 0.10
        else:
            return False, 0.0

    def _determine_subtitles(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> Tuple[str, int, bool]:
        """Détermine le style des sous-titres"""
        events = analysis.audio_events
        speech = analysis.speech
        content = analysis.content_type

        # Thème
        if events.laughter_count >= 3:
            theme = "neon"
            emojis = True
            reasoning.append("Sous-titres: neon + emojis - présence de rires")
        elif content in [ContentType.PODCAST, ContentType.INTERVIEW, ContentType.NEWS]:
            theme = "professional"
            emojis = False
            reasoning.append("Sous-titres: professional - contenu formel")
        elif content == ContentType.MOTIVATIONAL:
            theme = "viral"
            emojis = True
            reasoning.append("Sous-titres: viral + emojis - contenu motivant")
        else:
            theme = "viral"
            emojis = content not in [ContentType.TUTORIAL, ContentType.NEWS]
            reasoning.append(f"Sous-titres: viral - standard")

        # Nombre de mots par segment
        if speech.rate > 4:
            max_words = 4
            reasoning.append("Max mots: 4 - parole rapide")
        elif speech.rate < 2:
            max_words = 2
            reasoning.append("Max mots: 2 - parole lente")
        else:
            max_words = 3

        return theme, max_words, emojis


# =============================================================================
# LLM CONTENT TYPE DETECTOR
# =============================================================================

def detect_content_type_with_llm(
    transcription_text: str,
    audio_based_type: ContentType,
    audio_confidence: float
) -> Tuple[ContentType, float, str]:
    """
    Utilise le LLM local (Phi-4-mini) pour affiner la détection du type de contenu.

    Combine l'analyse audio avec l'analyse sémantique de la transcription pour
    améliorer la précision de détection.

    Args:
        transcription_text: Texte transcrit (premiers ~500 mots)
        audio_based_type: Type détecté par l'analyse audio
        audio_confidence: Confiance de la détection audio (0-1)

    Returns:
        Tuple (ContentType final, confiance finale, raisonnement)
    """
    try:
        from .local_llm import LocalLLM

        # Prendre uniquement les premiers 500 mots pour limiter les tokens
        words = transcription_text.split()[:500]
        sample_text = ' '.join(words)

        # Prompt pour le LLM
        prompt = f"""Tu es un expert en classification de contenu vidéo pour les réseaux sociaux (TikTok, Reels, Shorts).

Analyse cette transcription et détermine le type de contenu parmi:
- PODCAST: Conversations longues, interviews approfondies, discussions
- INTERVIEW: Questions-réponses, format structuré
- TUTORIAL: Instructions, explications étape par étape, éducatif
- VLOG: Contenu personnel, journée type, storytelling
- COMEDY: Humour, sketches, blagues
- GAMING: Jeux vidéo, gameplay, commentaires de parties, Minecraft, streams gaming
- MUSIC: Chansons, performances musicales
- NEWS: Actualités, informations factuelles
- MOTIVATIONAL: Inspiration, développement personnel, encouragement (SANS lien avec gaming)

IMPORTANT:
- Si la transcription parle de JEUX VIDÉO (Minecraft, Fortnite, etc.) → GAMING, PAS motivational
- GAMING inclut: gameplay, exploration de mondes virtuels, construction, combat dans des jeux
- MOTIVATIONAL est uniquement pour: développement personnel, inspiration de vie, coaching
- NE CONFONDS PAS une voix énergique dans un jeu avec du contenu motivational

TRANSCRIPTION:
{sample_text}

ANALYSE AUDIO PRÉLIMINAIRE:
- Type détecté: {audio_based_type.value}
- Confiance: {audio_confidence:.0%}

INSTRUCTIONS:
1. Lis attentivement la transcription
2. Identifie le SUJET PRINCIPAL (jeu vidéo? développement perso? interview?)
3. Si c'est lié aux jeux vidéo → GAMING
4. Sinon, choisis parmi les autres types

Réponds UNIQUEMENT au format:
TYPE: [un seul mot parmi la liste]
CONFIANCE: [0-100]
RAISON: [1 phrase courte expliquant pourquoi]"""

        # Initialiser le LLM
        llm = LocalLLM()

        # Générer la réponse
        llm_response = llm.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,  # Bas pour plus de déterminisme
            stop=["\n\n", "TYPE:", "CONFIANCE:", "RAISON:"]
        )

        # Extraire le texte de la réponse
        response = llm_response.text

        # Parser la réponse
        lines = response.strip().split('\n')
        detected_type = audio_based_type  # Fallback
        confidence = audio_confidence
        reasoning = f"Détection audio: {audio_based_type.value}"

        for line in lines:
            line = line.strip()
            if line.startswith('TYPE:'):
                type_str = line.replace('TYPE:', '').strip().lower()
                # Mapper vers ContentType
                type_mapping = {
                    'podcast': ContentType.PODCAST,
                    'interview': ContentType.INTERVIEW,
                    'tutorial': ContentType.TUTORIAL,
                    'vlog': ContentType.VLOG,
                    'comedy': ContentType.COMEDY,
                    'gaming': ContentType.GAMING,
                    'music': ContentType.MUSIC,
                    'news': ContentType.NEWS,
                    'motivational': ContentType.MOTIVATIONAL,
                }
                detected_type = type_mapping.get(type_str, audio_based_type)

            elif line.startswith('CONFIANCE:'):
                try:
                    conf_str = line.replace('CONFIANCE:', '').strip().replace('%', '')
                    confidence = float(conf_str) / 100.0
                except Exception:
                    pass

            elif line.startswith('RAISON:'):
                reasoning = line.replace('RAISON:', '').strip()

        # Combiner avec l'analyse audio (pondération 70% LLM, 30% audio)
        if detected_type == audio_based_type:
            # Accord entre LLM et audio → boost confiance
            final_confidence = min(1.0, (confidence * 0.7 + audio_confidence * 0.3) * 1.3)
        else:
            # Désaccord → favoriser le LLM (il comprend le contexte mieux que l'audio)
            final_confidence = confidence * 0.85

        console.print(f"[dim]🤖 LLM: {detected_type.value} ({final_confidence:.0%}) - {reasoning}[/dim]")
        return detected_type, final_confidence, reasoning

    except Exception as e:
        console.print(f"[yellow]⚠️ LLM indisponible: {e}[/yellow]")
        # Fallback: utiliser uniquement l'analyse audio
        return audio_based_type, audio_confidence, f"Audio seul: {audio_based_type.value}"


# =============================================================================
# AUTO CONFIGURATOR (Orchestrateur principal)
# =============================================================================

class AutoConfigurator:
    """
    Orchestrateur principal de l'auto-configuration.

    Coordonne l'analyse rapide et la génération de configuration.
    """

    def __init__(
        self,
        max_analysis_duration: float = 180.0,
        verbose: bool = False
    ):
        """
        Initialise l'auto-configurateur.

        Args:
            max_analysis_duration: Durée maximale à analyser (défaut: 180s)
            verbose: Afficher le raisonnement détaillé
        """
        self.max_analysis_duration = max_analysis_duration
        self.verbose = verbose
        self.analyzer = QuickAnalyzer(max_analysis_duration=max_analysis_duration)
        self.generator = ConfigGenerator()

    def analyze_and_configure(
        self,
        video_path: str,
        platform: str = "reels",
        total_duration: Optional[float] = None,
        transcription_result: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[GeneratedConfig, QuickAnalysisResult]:
        """
        Analyse la vidéo et génère une configuration optimale.

        Args:
            video_path: Chemin vers la vidéo
            platform: Plateforme cible
            total_duration: Durée totale de la vidéo (optionnel)
            transcription_result: Résultat de transcription Whisper (optionnel, améliore précision)
            progress_callback: Callback pour la progression (percent, message)

        Returns:
            Tuple (GeneratedConfig, QuickAnalysisResult)
        """
        def report_progress(percent: int, message: str):
            """Helper pour reporter la progression"""
            if progress_callback:
                progress_callback(percent, message)

        from moviepy import VideoFileClip

        report_progress(5, "Préparation de l'analyse...")

        # Obtenir la durée totale si non fournie
        if total_duration is None:
            with VideoFileClip(video_path) as video:
                total_duration = video.duration

        # Extraire l'audio temporairement
        audio_path = Path(video_path).parent / f"{Path(video_path).stem}_autoconfig.wav"

        try:
            report_progress(10, "Extraction de l'audio...")

            with VideoFileClip(video_path) as video:
                if video.audio:
                    # Extraire seulement la portion nécessaire
                    duration_to_extract = min(self.max_analysis_duration, video.duration)
                    video.subclipped(0, duration_to_extract).audio.write_audiofile(
                        str(audio_path),
                        fps=11025,  # Sample rate réduit pour rapidité
                        logger=None
                    )

            if not audio_path.exists():
                raise ValueError("Impossible d'extraire l'audio")

            report_progress(25, "Audio extrait, préparation transcription...")

            # Extraire le texte transcrit si disponible
            transcription_text = None
            if transcription_result is not None:
                try:
                    # Extraire seulement les 180 premières secondes
                    words_180s = [w for w in transcription_result.words if w.start <= 180.0]
                    transcription_text = " ".join([w.word for w in words_180s])
                    console.print(f"[dim]Transcription utilisée: {len(transcription_text)} caractères[/dim]")
                    report_progress(35, f"Transcription prête ({len(transcription_text)} caractères)")
                except Exception as e:
                    console.print(f"[yellow]⚠ Impossible d'utiliser la transcription: {e}[/yellow]")

            # Analyser (avec transcription si disponible)
            # Mapper les pourcentages de l'analyseur (0-100) vers notre plage (40-90)
            def analyzer_progress(percent: int, message: str):
                # Mapper 0-100% de l'analyzer vers 40-90% de l'étape globale
                mapped_percent = 40 + int(percent * 0.5)
                report_progress(mapped_percent, message)

            analysis = self.analyzer.analyze(
                str(audio_path),
                total_duration,
                transcription_text=transcription_text,
                progress_callback=analyzer_progress
            )

            # Générer la configuration
            report_progress(92, "Génération de la configuration optimale...")
            config = self.generator.generate(analysis, platform)

            report_progress(95, f"Configuration générée: {analysis.content_type.value}")

            # Afficher les résultats si verbose
            if self.verbose:
                self._print_verbose_results(analysis, config)

            return config, analysis

        finally:
            # Nettoyer
            if audio_path.exists():
                audio_path.unlink()

    def _print_verbose_results(
        self,
        analysis: QuickAnalysisResult,
        config: GeneratedConfig
    ):
        """Affiche les résultats détaillés de l'analyse"""
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Pre-Analysis: Auto-Configuration[/bold cyan]",
            border_style="cyan"
        ))

        # Type de contenu
        confidence_pct = analysis.content_confidence * 100
        console.print(
            f"\n  Contenu détecté: [cyan]{analysis.content_type.value}[/cyan]"
            f" (confiance: {confidence_pct:.0f}%)"
        )

        # Profil émotionnel
        console.print(f"\n  [bold]Profil émotionnel:[/bold]")
        console.print(f"    - Énergie moyenne: {analysis.energy_profile.average:.2f}")
        variance_label = 'dynamique' if analysis.energy_profile.variance > 0.15 else 'stable'
        console.print(
            f"    - Variance: {analysis.energy_profile.variance:.2f}"
            f" ({variance_label})"
        )
        console.print(f"    - Pics d'excitation: {analysis.excitement_peaks}")
        console.print(f"    - Émotion dominante: {analysis.dominant_emotion.value}")

        # Événements audio
        events = analysis.audio_events
        console.print(f"\n  [bold]Événements audio:[/bold]")
        console.print(f"    - Rires: {events.laughter_count}")
        console.print(f"    - Applaudissements: {events.applause_count}")
        console.print(f"    - Silences dramatiques: {events.dramatic_silence_count}")

        # Configuration générée
        console.print(f"\n  [bold]Configuration générée:[/bold]")
        for reason in config.reasoning:
            console.print(f"    - {reason}")

        console.print()


def auto_configure(
    video_path: str,
    platform: str = "reels",
    verbose: bool = False,
    max_analysis_duration: float = 180.0
) -> Tuple[GeneratedConfig, QuickAnalysisResult]:
    """
    Fonction utilitaire pour l'auto-configuration.

    Args:
        video_path: Chemin vers la vidéo
        platform: Plateforme cible
        verbose: Afficher le raisonnement détaillé
        max_analysis_duration: Durée maximale à analyser

    Returns:
        Tuple (GeneratedConfig, QuickAnalysisResult)
    """
    configurator = AutoConfigurator(
        max_analysis_duration=max_analysis_duration,
        verbose=verbose
    )
    return configurator.analyze_and_configure(video_path, platform)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_config.py <video_path> [platform]")
        sys.exit(1)

    video_path = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else "reels"

    config, analysis = auto_configure(video_path, platform, verbose=True)

    print("\nConfiguration finale:")
    print(f"  - Durées: {config.min_duration:.0f}-{config.max_duration:.0f}s")
    print(f"  - Score minimum: {config.min_viral_score}")
    print(f"  - Zoom: {config.zoom_style} ({config.zoom_factor}x)")
    print(f"  - Couleurs: {config.color_grading}")
    print(f"  - Sous-titres: {config.subtitle_theme}")
