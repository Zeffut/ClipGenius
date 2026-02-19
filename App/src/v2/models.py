"""Modeles de donnees pour le pipeline v2 de detection virale."""

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict


class ContentType(enum.Enum):
    """Types de contenu detectes."""
    PODCAST = 'podcast'
    INTERVIEW = 'interview'
    TUTORIAL = 'tutorial'
    VLOG = 'vlog'
    COMEDY = 'comedy'
    GAMING = 'gaming'
    MUSIC = 'music'
    NEWS = 'news'
    MOTIVATIONAL = 'motivational'
    ACTION = 'action'
    UNKNOWN = 'unknown'


@dataclass
class WordTimestamp:
    """Mot avec son timestamp exact."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    """Segment de transcription avec timestamps."""
    start: float
    end: float
    text: str


@dataclass
class AudioFeatures:
    """Features audio extraites pour un segment temporel."""
    start: float
    end: float
    # Energie
    rms_energy: float = 0.0
    energy_peak: bool = False
    # Pitch / F0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0
    # Debit de parole
    speech_rate: float = 0.0         # mots/seconde
    speech_rate_delta: float = 0.0   # variation par rapport a la moyenne
    # Emotion vocale (speechbrain)
    emotion: str = 'neutral'
    emotion_confidence: float = 0.0
    # Diarisation (pyannote)
    num_speakers: int = 0
    speaker_changes: int = 0
    overlap_ratio: float = 0.0      # % du segment avec parole chevauchee


@dataclass
class AudioEngagementScore:
    """Score d'engagement audio agrege pour un segment."""
    start: float
    end: float
    score: float = 0.0              # Score composite 0-1
    # Composantes du score
    energy_score: float = 0.0
    pitch_score: float = 0.0
    speech_dynamics_score: float = 0.0
    emotion_score: float = 0.0
    interaction_score: float = 0.0  # overlap + speaker changes


@dataclass
class LLMChapter:
    """Chapitre identifie par le LLM (passe 1)."""
    start: float
    end: float
    title: str
    summary: str


@dataclass
class LLMScoredMoment:
    """Moment score par le LLM avec scoring multi-dimensionnel (passe 2)."""
    start: float
    end: float
    # Scores individuels (1-10, normalises en 0-1 pour la fusion)
    hook_strength: float = 0.0
    emotional_intensity: float = 0.0
    standalone_clarity: float = 0.0
    quotability: float = 0.0
    tension_arc: float = 0.0
    controversy: float = 0.0
    # Meta
    hook_text: str = ''
    emotion: str = ''
    reason: str = ''
    composite_score: float = 0.0    # Score LLM composite

    def compute_composite(self) -> float:
        """Calcule le score composite a partir des dimensions individuelles."""
        weights = {
            'hook_strength': 0.25,
            'emotional_intensity': 0.20,
            'standalone_clarity': 0.20,
            'quotability': 0.15,
            'tension_arc': 0.10,
            'controversy': 0.10,
        }
        self.composite_score = sum(
            getattr(self, dim) * w for dim, w in weights.items()
        )
        return self.composite_score


@dataclass
class ViralCandidate:
    """Candidat viral avec tous les scores fusionnes."""
    start: float
    end: float
    # Scores par source
    llm_score: float = 0.0
    audio_score: float = 0.0
    speech_dynamics_score: float = 0.0
    structural_score: float = 0.0
    hook_score: float = 0.0
    # Score final fusionne
    final_score: float = 0.0
    # Metadata
    hook_text: str = ''
    emotion: str = ''
    reason: str = ''
    chapter_title: str = ''

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_viral_moment(self):
        """Convertit en ViralMoment pour compatibilite avec le pipeline existant."""
        from ..viral_detector import ViralMoment
        return ViralMoment(
            start_time=self.start,
            end_time=self.end,
            score=self.final_score,
            reason=f'[{self.emotion}] {self.reason}' if self.emotion else self.reason
        )


@dataclass
class PipelineConfig:
    """Configuration du pipeline v2."""
    # Durees
    min_clip_duration: float = 30.0
    max_clip_duration: float = 90.0
    target_clip_duration: float = 60.0
    # Nombre de clips
    max_clips: int = 5
    clips_per_minutes: float = 1/3  # 1 clip par 3 minutes
    # Seuils
    min_viral_score: float = 0.45
    # Poids de fusion
    weight_llm: float = 0.40
    weight_audio: float = 0.30
    weight_speech: float = 0.20
    weight_structural: float = 0.10
    # Audio
    segment_duration: float = 10.0  # duree des segments d'analyse audio
    # LLM
    llm_model: str = 'qwen2.5-7b'
    llm_temperature: float = 0.3
    llm_max_retries: int = 3
    # Content type
    content_type: ContentType = ContentType.UNKNOWN
    # Features optionnelles
    use_diarization: bool = True
    use_emotion: bool = True
    use_pitch: bool = True
