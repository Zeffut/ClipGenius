"""
Module d'auto-configuration intelligente pour ClipGenius

Analyse automatiquement la vidéo source et génère une configuration
optimale basée sur le type de contenu détecté.

Usage:
    from src.auto_config import AutoConfigurator

    configurator = AutoConfigurator()
    config = configurator.analyze_and_configure(video_path, platform="tiktok")
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console
from rich.panel import Panel

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

        from .quick_analyzer import QuickAnalyzer
        from .config_generator import ConfigGenerator

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
                    video.with_subclip(0, duration_to_extract).audio.write_audiofile(
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
