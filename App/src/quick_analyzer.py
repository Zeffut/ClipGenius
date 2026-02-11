"""Analyseur audio rapide pour la detection de contenu."""

import numpy as np
import librosa
from typing import List, Dict, Any, Optional, Tuple, Callable
from rich.console import Console
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from .auto_config import (
    ContentType, EmotionProfile, EnergyProfile,
    AudioEventsSummary, SpeechCharacteristics, QuickAnalysisResult
)

console = Console()


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
                from .config_generator import detect_content_type_with_llm

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
