"""Pipeline v2 de detection de moments viraux — orchestrateur principal.

Ce module coordonne les differentes phases du pipeline multi-signal :
1. Extraction de features audio
2. Agregation des scores audio
3. Analyse LLM (chapitrage + scoring)
4. Fusion des scores
5. Conversion en ViralMoment (compatibilite pipeline existant)

Gestion robuste des erreurs : si le LLM echoue, on se rabat sur
l'audio seul ; si l'audio echoue, on se rabat sur le LLM seul ;
si les deux echouent, on retourne une liste vide.
"""

from typing import Callable, List, Optional, Tuple

from rich.console import Console

from ..viral_detector import ViralMoment
from .audio_features import AudioFeatureExtractor
from .fusion import ScoreFusion
from .models import (
    AudioEngagementScore,
    ContentType,
    LLMScoredMoment,
    PipelineConfig,
    TranscriptSegment,
    ViralCandidate,
    WordTimestamp,
)
from .score_aggregator import AudioScoreAggregator

console = Console()

# Poids de progression pour chaque phase du pipeline
_PROGRESS_AUDIO_FEATURES: float = 0.20
_PROGRESS_AUDIO_SCORES: float = 0.05
_PROGRESS_LLM_ANALYSIS: float = 0.60
_PROGRESS_FUSION: float = 0.10
_PROGRESS_CONVERSION: float = 0.05


def _safe_progress(
    callback: Optional[Callable[[float, str], None]],
    progress: float,
    message: str,
) -> None:
    """Appelle le callback de progression de maniere securisee.

    Args:
        callback: Fonction de callback (progress, message) ou None.
        progress: Avancement entre 0.0 et 1.0.
        message: Message descriptif de l'etape en cours.
    """
    if callback is not None:
        try:
            callback(min(1.0, max(0.0, progress)), message)
        except Exception:
            pass


class ViralDetectorV2:
    """Pipeline v2 de detection de moments viraux multi-signal."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Methode publique principale
    # ------------------------------------------------------------------

    def detect(
        self,
        video_path: str,
        transcript_segments: Optional[List[TranscriptSegment]] = None,
        words: Optional[List[WordTimestamp]] = None,
        content_type: str = 'unknown',
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[ViralMoment]:
        """Detecte les moments viraux dans une video.

        Compatible avec le pipeline existant : retourne List[ViralMoment].

        Args:
            video_path: Chemin vers la video.
            transcript_segments: Segments de transcription (optionnel,
                sera transcrit sinon).
            words: Mots avec timestamps (optionnel).
            content_type: Type de contenu (ex. 'podcast', 'comedy').
            progress_callback: Callback (0-1, message) pour le suivi
                de progression.

        Returns:
            Liste de ViralMoment tries par score (compatible pipeline
            existant).
        """
        console.print('[bold cyan]Pipeline v2 de detection virale[/bold cyan]')

        # -- Configuration selon le type de contenu --
        self._apply_content_type(content_type)

        # Variables de resultats intermediaires
        audio_scores: List[AudioEngagementScore] = []
        audio_peaks: List[Tuple[float, float, float]] = []
        llm_moments: List[LLMScoredMoment] = []
        audio_ok = False
        llm_ok = False

        # ====================================================
        # Phase 1 : Extraction de features audio (20%)
        # ====================================================
        _safe_progress(progress_callback, 0.0, "Extraction des features audio...")
        audio_features = self._phase_audio_features(video_path, words, progress_callback)

        # ====================================================
        # Phase 2 : Agregation des scores audio (5%)
        # ====================================================
        if audio_features:
            _safe_progress(
                progress_callback,
                _PROGRESS_AUDIO_FEATURES,
                "Agregation des scores audio...",
            )
            audio_scores, audio_peaks = self._phase_audio_scores(audio_features)
            audio_ok = True

        _safe_progress(
            progress_callback,
            _PROGRESS_AUDIO_FEATURES + _PROGRESS_AUDIO_SCORES,
            "Analyse LLM en cours...",
        )

        # ====================================================
        # Phase 3 : Analyse LLM (60%)
        # ====================================================
        llm_moments = self._phase_llm_analysis(
            video_path, transcript_segments, words, progress_callback
        )
        if llm_moments:
            llm_ok = True

        # ====================================================
        # Phase 4 : Fusion (10%)
        # ====================================================
        progress_before_fusion = (
            _PROGRESS_AUDIO_FEATURES + _PROGRESS_AUDIO_SCORES + _PROGRESS_LLM_ANALYSIS
        )
        _safe_progress(progress_callback, progress_before_fusion, "Fusion des scores...")

        candidates = self._phase_fusion(
            llm_moments, audio_scores, audio_peaks, words,
            audio_ok, llm_ok,
        )

        # ====================================================
        # Phase 5 : Conversion en ViralMoment (5%)
        # ====================================================
        progress_before_conversion = progress_before_fusion + _PROGRESS_FUSION
        _safe_progress(
            progress_callback,
            progress_before_conversion,
            "Conversion des resultats...",
        )

        results = self._phase_conversion(candidates)

        _safe_progress(progress_callback, 1.0, "Detection terminee.")
        console.print(
            f'[bold green]Pipeline v2 termine : '
            f'{len(results)} moment(s) viral(aux) detecte(s)[/bold green]'
        )
        return results

    # ------------------------------------------------------------------
    # Phases du pipeline
    # ------------------------------------------------------------------

    def _phase_audio_features(
        self,
        video_path: str,
        words: Optional[List[WordTimestamp]],
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> list:
        """Phase 1 : Extraction des features audio.

        L'extracteur gere en interne la diarisation et l'emotion
        si les bibliotheques sont disponibles.

        Args:
            video_path: Chemin vers la video.
            words: Mots avec timestamps.
            progress_callback: Callback de progression.

        Returns:
            Liste d'AudioFeatures ou liste vide en cas d'erreur.
        """
        try:
            extractor = AudioFeatureExtractor()
            features = extractor.extract_all(
                audio_path=video_path,
                segment_duration=self.config.segment_duration,
                words=words,
            )
            return features

        except Exception as exc:
            console.print(
                f'[yellow]Phase audio echouee : {exc}[/yellow]'
            )
            return []

    def _phase_audio_scores(
        self,
        audio_features: list,
    ) -> Tuple[List[AudioEngagementScore], List[Tuple[float, float, float]]]:
        """Phase 2 : Agregation des scores audio et detection des pics.

        Args:
            audio_features: Liste d'AudioFeatures extraites en phase 1.

        Returns:
            Tuple (scores d'engagement, regions de pics).
        """
        try:
            aggregator = AudioScoreAggregator(content_type=self.config.content_type)
            scores = aggregator.aggregate(audio_features)
            peaks = aggregator.find_peak_regions(
                scores,
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
            )
            console.print(
                f'[dim]Audio : {len(scores)} segments scores, '
                f'{len(peaks)} region(s) de pic detectee(s)[/dim]'
            )
            return scores, peaks

        except Exception as exc:
            console.print(
                f'[yellow]Agregation audio echouee : {exc}[/yellow]'
            )
            return [], []

    def _phase_llm_analysis(
        self,
        video_path: str,
        transcript_segments: Optional[List[TranscriptSegment]],
        words: Optional[List[WordTimestamp]],
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> List[LLMScoredMoment]:
        """Phase 3 : Analyse LLM en deux passes (chapitrage + scoring).

        Args:
            video_path: Chemin vers la video (pour estimer la duree).
            transcript_segments: Segments de transcription.
            words: Mots avec timestamps.
            progress_callback: Callback de progression.

        Returns:
            Liste de LLMScoredMoment ou liste vide en cas d'erreur.
        """
        if not transcript_segments:
            console.print(
                '[yellow]Pas de transcription disponible — '
                'analyse LLM impossible[/yellow]'
            )
            return []

        try:
            from .llm_analyzer import LLMAnalyzer

            analyzer = LLMAnalyzer(
                content_type=self.config.content_type.value,
                temperature=self.config.llm_temperature,
                max_retries=self.config.llm_max_retries,
            )

            # Estimer la duree de la video a partir des segments
            video_duration = self._estimate_duration_from_segments(
                transcript_segments
            )

            # Passe 1 : Chapitrage
            progress_base = _PROGRESS_AUDIO_FEATURES + _PROGRESS_AUDIO_SCORES
            _safe_progress(
                progress_callback,
                progress_base + 0.05,
                "LLM passe 1 : identification des chapitres...",
            )

            chapters = analyzer.analyze_chapters(
                segments=transcript_segments,
                video_duration=video_duration,
            )

            if not chapters:
                console.print(
                    '[yellow]LLM passe 1 : aucun chapitre identifie[/yellow]'
                )
                return []

            console.print(
                f'[dim]LLM passe 1 : {len(chapters)} chapitre(s) identifie(s)[/dim]'
            )

            # Passe 2 : Scoring multi-dimensionnel
            _safe_progress(
                progress_callback,
                progress_base + 0.35,
                "LLM passe 2 : scoring multi-dimensionnel...",
            )

            moments = analyzer.score_moments(
                chapters=chapters,
                segments=transcript_segments,
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
            )

            # Calculer le score composite pour chaque moment
            for moment in moments:
                moment.compute_composite()

            console.print(
                f'[dim]LLM passe 2 : {len(moments)} moment(s) score(s)[/dim]'
            )
            return moments

        except ImportError:
            console.print(
                '[yellow]Module llm_analyzer non disponible — '
                'analyse LLM desactivee[/yellow]'
            )
            return []
        except Exception as exc:
            console.print(
                f'[yellow]Analyse LLM echouee : {exc}[/yellow]'
            )
            return []

    @staticmethod
    def _estimate_duration_from_segments(
        segments: List[TranscriptSegment],
    ) -> float:
        """Estime la duree totale de la video a partir des segments.

        Args:
            segments: Segments de transcription.

        Returns:
            Duree estimee en secondes.
        """
        if not segments:
            return 0.0
        return max(seg.end for seg in segments)

    def _phase_fusion(
        self,
        llm_moments: List[LLMScoredMoment],
        audio_scores: List[AudioEngagementScore],
        audio_peaks: List[Tuple[float, float, float]],
        words: Optional[List[WordTimestamp]],
        audio_ok: bool,
        llm_ok: bool,
    ) -> List[ViralCandidate]:
        """Phase 4 : Fusion des signaux en candidats viraux.

        Gere les cas de degradation gracieuse :
        - Si les deux sources sont disponibles, fusion complete.
        - Si seul l'audio est disponible, candidats audio uniquement.
        - Si seul le LLM est disponible, candidats LLM uniquement.
        - Si aucune source, retourne une liste vide.

        Args:
            llm_moments: Moments scores par le LLM.
            audio_scores: Scores d'engagement audio.
            audio_peaks: Regions de pics audio.
            words: Mots avec timestamps.
            audio_ok: True si l'audio a ete traite avec succes.
            llm_ok: True si le LLM a ete traite avec succes.

        Returns:
            Liste de ViralCandidate.
        """
        if not audio_ok and not llm_ok:
            console.print(
                "[red]Aucune source de signal disponible — "
                "impossible de detecter des moments viraux[/red]"
            )
            return []

        # Ajuster les poids si une source est manquante
        config = self._adjust_weights_for_fallback(audio_ok, llm_ok)
        fusion = ScoreFusion(config)

        try:
            candidates = fusion.fuse(
                llm_moments=llm_moments,
                audio_scores=audio_scores,
                audio_peaks=audio_peaks,
                words=words,
            )
            return candidates

        except Exception as exc:
            console.print(f'[red]Erreur lors de la fusion : {exc}[/red]')
            return []

    @staticmethod
    def _phase_conversion(
        candidates: List[ViralCandidate],
    ) -> List[ViralMoment]:
        """Phase 5 : Conversion des candidats en ViralMoment.

        Args:
            candidates: Liste de ViralCandidate.

        Returns:
            Liste de ViralMoment compatibles avec le pipeline existant.
        """
        results: List[ViralMoment] = []
        for candidate in candidates:
            try:
                results.append(candidate.to_viral_moment())
            except Exception as exc:
                console.print(
                    f'[yellow]Conversion echouee pour le candidat '
                    f'{candidate.start:.1f}-{candidate.end:.1f}s : {exc}[/yellow]'
                )
        return results

    # ------------------------------------------------------------------
    # Helpers de configuration
    # ------------------------------------------------------------------

    def _apply_content_type(self, content_type: str) -> None:
        """Applique le type de contenu a la configuration du pipeline.

        Args:
            content_type: Type de contenu sous forme de chaine (ex. 'podcast').
        """
        try:
            ct = ContentType(content_type.lower())
        except ValueError:
            ct = ContentType.UNKNOWN

        self.config.content_type = ct

        console.print(
            f'[dim]Type de contenu : {ct.value}[/dim]'
        )

    def _adjust_weights_for_fallback(
        self, audio_ok: bool, llm_ok: bool
    ) -> PipelineConfig:
        """Ajuste les poids de fusion en cas de source manquante.

        Si une source est absente, on redistribue son poids aux
        autres sources proportionnellement.

        Args:
            audio_ok: True si les scores audio sont disponibles.
            llm_ok: True si les scores LLM sont disponibles.

        Returns:
            PipelineConfig avec les poids ajustes.
        """
        from copy import copy
        config = copy(self.config)

        if audio_ok and llm_ok:
            # Tout est normal
            return config

        if llm_ok and not audio_ok:
            # Redistribuer le poids audio vers le LLM et le structurel
            console.print(
                '[yellow]Mode degrade : audio indisponible, '
                'redistribution des poids vers le LLM[/yellow]'
            )
            audio_weight = config.weight_audio + config.weight_speech
            config.weight_llm += audio_weight * 0.75
            config.weight_structural += audio_weight * 0.25
            config.weight_audio = 0.0
            config.weight_speech = 0.0

        elif audio_ok and not llm_ok:
            # Redistribuer le poids LLM vers l'audio et le structurel
            console.print(
                '[yellow]Mode degrade : LLM indisponible, '
                'redistribution des poids vers l\'audio[/yellow]'
            )
            llm_weight = config.weight_llm
            config.weight_audio += llm_weight * 0.50
            config.weight_speech += llm_weight * 0.30
            config.weight_structural += llm_weight * 0.20
            config.weight_llm = 0.0

        # Abaisser le seuil minimum en mode degrade pour ne pas tout filtrer
        config.min_viral_score = max(0.3, config.min_viral_score - 0.15)

        return config


# ======================================================================
# Fonction utilitaire de commodite
# ======================================================================

def detect_viral_moments(
    video_path: str,
    transcript_segments: Optional[List[TranscriptSegment]] = None,
    words: Optional[List[WordTimestamp]] = None,
    min_duration: float = 30.0,
    max_duration: float = 90.0,
    max_clips: int = 5,
    min_viral_score: float = 0.65,
    content_type: str = 'unknown',
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[ViralMoment]:
    """Fonction utilitaire pour detecter les moments viraux avec le pipeline v2.

    Cree une instance de ViralDetectorV2 avec la configuration fournie et
    lance la detection. Compatible avec le pipeline existant.

    Args:
        video_path: Chemin vers la video.
        transcript_segments: Segments de transcription (optionnel).
        words: Mots avec timestamps (optionnel).
        min_duration: Duree minimale des clips en secondes.
        max_duration: Duree maximale des clips en secondes.
        max_clips: Nombre maximal de clips a generer.
        min_viral_score: Score minimum pour qu'un moment soit retenu.
        content_type: Type de contenu (ex. 'podcast', 'comedy').
        progress_callback: Callback (0-1, message) pour le suivi.

    Returns:
        Liste de ViralMoment tries par score decroissant.
    """
    config = PipelineConfig(
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
        max_clips=max_clips,
        min_viral_score=min_viral_score,
    )

    detector = ViralDetectorV2(config=config)
    return detector.detect(
        video_path=video_path,
        transcript_segments=transcript_segments,
        words=words,
        content_type=content_type,
        progress_callback=progress_callback,
    )
