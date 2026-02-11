"""Pipeline v2 de detection de moments viraux - approche multi-signal."""

from .models import (
    AudioEngagementScore,
    AudioFeatures,
    ContentType,
    LLMChapter,
    LLMScoredMoment,
    PipelineConfig,
    TranscriptSegment,
    ViralCandidate,
    WordTimestamp,
)
from .pipeline import ViralDetectorV2, detect_viral_moments

__all__ = [
    'AudioEngagementScore',
    'AudioFeatures',
    'ContentType',
    'LLMChapter',
    'LLMScoredMoment',
    'PipelineConfig',
    'TranscriptSegment',
    'ViralCandidate',
    'ViralDetectorV2',
    'WordTimestamp',
    'detect_viral_moments',
]
