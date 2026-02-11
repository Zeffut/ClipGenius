"""Systeme de fusion des scores pour le pipeline v2 de detection virale.

Combine les scores LLM et audio en candidats viraux finaux, en tenant
compte de la structure du texte, de la qualite du hook et de la position
dans la video.
"""

from typing import List, Optional, Tuple

from rich.console import Console

from .models import (
    AudioEngagementScore,
    LLMScoredMoment,
    PipelineConfig,
    ViralCandidate,
    WordTimestamp,
)

console = Console()

# Seuil de chevauchement au-dela duquel on supprime le candidat le plus faible
_OVERLAP_REMOVAL_RATIO: float = 0.30

# Penalite pour les moments dans les 10 premieres ou dernieres secondes
_EDGE_PENALTY_SECONDS: float = 10.0
_EDGE_PENALTY_VALUE: float = 0.10

# Bonus pour une duree proche de la duree cible
_DURATION_BONUS_MAX: float = 0.08

# Bonus pour un debut de phrase propre
_SENTENCE_START_BONUS: float = 0.06

# Bonus pour une fin de phrase propre
_SENTENCE_END_BONUS: float = 0.04


class ScoreFusion:
    """Fusionne les scores LLM et audio en candidats viraux."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Methode principale
    # ------------------------------------------------------------------

    def fuse(
        self,
        llm_moments: List[LLMScoredMoment],
        audio_scores: List[AudioEngagementScore],
        audio_peaks: List[Tuple[float, float, float]],
        words: Optional[List[WordTimestamp]] = None,
    ) -> List[ViralCandidate]:
        """Fusionne les signaux LLM, audio et structurels en candidats viraux.

        Algorithme :
        1. Partir des moments LLM comme candidats primaires.
        2. Pour chaque moment, calculer le score audio moyen des segments
           qui chevauchent la fenetre temporelle du moment.
        3. Optionnellement scorer le hook via HookOptimizer.
        4. Calculer un score structurel (position, limites de phrase).
        5. Appliquer les poids de fusion de PipelineConfig.
        6. Ajouter les pics audio non couverts par un moment LLM.
        7. Trier par score final decroissant.
        8. Supprimer les chevauchements (garder le meilleur).
        9. Limiter a max_clips.

        Args:
            llm_moments: Moments scores par le LLM (passe 2).
            audio_scores: Scores d'engagement audio par segment.
            audio_peaks: Regions a fort engagement audio (start, end, avg_score).
            words: Mots avec timestamps pour l'analyse structurelle.

        Returns:
            Liste de ViralCandidate tries par score final decroissant.
        """
        candidates: List[ViralCandidate] = []

        # Determiner la duree totale de la video a partir des signaux disponibles
        video_duration = self._estimate_video_duration(
            llm_moments, audio_scores, audio_peaks
        )

        # -- 1. Candidats issus des moments LLM --
        for moment in llm_moments:
            audio_avg = self._compute_overlapping_audio_score(
                moment.start, moment.end, audio_scores
            )
            speech_dynamics = self._compute_overlapping_speech_dynamics(
                moment.start, moment.end, audio_scores
            )
            hook_score = self._compute_hook_score(moment.hook_text)
            structural = self._compute_structural_score(
                moment.start, moment.end, video_duration, words
            )

            final_score = (
                self.config.weight_llm * moment.composite_score
                + self.config.weight_audio * audio_avg
                + self.config.weight_speech * speech_dynamics
                + self.config.weight_structural * structural
            )

            candidates.append(ViralCandidate(
                start=moment.start,
                end=moment.end,
                llm_score=moment.composite_score,
                audio_score=audio_avg,
                speech_dynamics_score=speech_dynamics,
                structural_score=structural,
                hook_score=hook_score,
                final_score=final_score,
                hook_text=moment.hook_text,
                emotion=moment.emotion,
                reason=moment.reason,
            ))

        # -- 2. Pics audio non couverts par un moment LLM --
        for peak_start, peak_end, peak_score in audio_peaks:
            if self._is_covered_by_llm(peak_start, peak_end, llm_moments):
                continue

            speech_dynamics = self._compute_overlapping_speech_dynamics(
                peak_start, peak_end, audio_scores
            )
            structural = self._compute_structural_score(
                peak_start, peak_end, video_duration, words
            )

            # Pas de score LLM pour ces candidats
            final_score = (
                self.config.weight_audio * peak_score
                + self.config.weight_speech * speech_dynamics
                + self.config.weight_structural * structural
            )

            candidates.append(ViralCandidate(
                start=peak_start,
                end=peak_end,
                llm_score=0.0,
                audio_score=peak_score,
                speech_dynamics_score=speech_dynamics,
                structural_score=structural,
                hook_score=0.0,
                final_score=final_score,
                hook_text='',
                emotion='',
                reason='Pic audio sans analyse LLM',
            ))

        # -- 3. Tri par score final decroissant --
        candidates.sort(key=lambda c: c.final_score, reverse=True)

        # -- 4. Suppression des chevauchements --
        candidates = self._remove_overlaps(candidates)

        # -- 5. Filtrage par seuil minimum --
        candidates = [
            c for c in candidates
            if c.final_score >= self.config.min_viral_score
        ]

        # -- 6. Limitation au nombre maximal de clips --
        candidates = candidates[:self.config.max_clips]

        console.print(
            f'[dim]Fusion terminee : {len(candidates)} candidat(s) viral(aux) retenus[/dim]'
        )
        return candidates

    # ------------------------------------------------------------------
    # Calcul du score structurel
    # ------------------------------------------------------------------

    def _compute_structural_score(
        self,
        start: float,
        end: float,
        video_duration: float,
        words: Optional[List[WordTimestamp]] = None,
    ) -> float:
        """Calcule un score structurel pour un candidat.

        Criteres :
        - Bonus si le moment commence par un debut de phrase (majuscule
          apres un point).
        - Bonus si le moment se termine par une ponctuation de fin de
          phrase.
        - Petite penalite si le moment est dans les 10 premieres ou
          dernieres secondes de la video.
        - Petit bonus si la duree est proche de target_clip_duration.

        Args:
            start: Debut du moment (secondes).
            end: Fin du moment (secondes).
            video_duration: Duree totale de la video (secondes).
            words: Mots avec timestamps (optionnel).

        Returns:
            Score structurel entre 0.0 et 1.0.
        """
        score = 0.5  # score de base neutre

        # -- Penalite pour les bords de la video --
        if video_duration > 0:
            if start < _EDGE_PENALTY_SECONDS:
                score -= _EDGE_PENALTY_VALUE
            if video_duration - end < _EDGE_PENALTY_SECONDS:
                score -= _EDGE_PENALTY_VALUE

        # -- Bonus de duree --
        duration = end - start
        target = self.config.target_clip_duration
        if target > 0:
            # Plus la duree est proche de la cible, plus le bonus est eleve
            distance_ratio = abs(duration - target) / target
            duration_bonus = _DURATION_BONUS_MAX * max(0.0, 1.0 - distance_ratio)
            score += duration_bonus

        # -- Analyse des mots aux frontieres du moment --
        if words:
            score += self._sentence_boundary_bonus(start, end, words)

        # Clamper dans [0, 1]
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Helpers prives
    # ------------------------------------------------------------------

    @staticmethod
    def _sentence_boundary_bonus(
        start: float,
        end: float,
        words: List[WordTimestamp],
    ) -> float:
        """Calcule le bonus lie aux limites de phrase.

        Bonus si le premier mot du moment est un debut de phrase
        (majuscule apres un point, ou premier mot de la transcription).
        Bonus si le dernier mot du moment se termine par une ponctuation
        de fin de phrase (. ! ?).

        Args:
            start: Debut du moment (secondes).
            end: Fin du moment (secondes).
            words: Mots avec timestamps.

        Returns:
            Bonus total (>= 0).
        """
        if not words:
            return 0.0

        bonus = 0.0

        # Trouver le premier mot dont le centre est dans le moment
        first_word: Optional[WordTimestamp] = None
        first_word_idx: int = -1
        for i, w in enumerate(words):
            word_center = (w.start + w.end) / 2.0
            if word_center >= start:
                if word_center <= end:
                    first_word = w
                    first_word_idx = i
                break

        # Trouver le dernier mot dont le centre est dans le moment
        last_word: Optional[WordTimestamp] = None
        for w in reversed(words):
            word_center = (w.start + w.end) / 2.0
            if word_center <= end:
                if word_center >= start:
                    last_word = w
                break

        # Bonus debut de phrase
        if first_word is not None:
            text = first_word.word.strip()
            if text and text[0].isupper():
                # Verifier si le mot precedent se termine par une ponctuation
                if first_word_idx == 0:
                    bonus += _SENTENCE_START_BONUS
                elif first_word_idx > 0:
                    prev_text = words[first_word_idx - 1].word.strip()
                    if prev_text and prev_text[-1] in '.!?':
                        bonus += _SENTENCE_START_BONUS

        # Bonus fin de phrase
        if last_word is not None:
            text = last_word.word.strip()
            if text and text[-1] in '.!?':
                bonus += _SENTENCE_END_BONUS

        return bonus

    @staticmethod
    def _compute_overlapping_audio_score(
        start: float,
        end: float,
        audio_scores: List[AudioEngagementScore],
    ) -> float:
        """Calcule le score audio moyen sur les segments qui chevauchent [start, end].

        Args:
            start: Debut de la fenetre (secondes).
            end: Fin de la fenetre (secondes).
            audio_scores: Liste d'AudioEngagementScore.

        Returns:
            Score audio moyen (0.0 si aucun segment ne chevauche).
        """
        overlapping = [
            s.score for s in audio_scores
            if s.end > start and s.start < end
        ]
        if not overlapping:
            return 0.0
        return sum(overlapping) / len(overlapping)

    @staticmethod
    def _compute_overlapping_speech_dynamics(
        start: float,
        end: float,
        audio_scores: List[AudioEngagementScore],
    ) -> float:
        """Calcule le score de dynamique vocale moyen sur la fenetre.

        Args:
            start: Debut de la fenetre (secondes).
            end: Fin de la fenetre (secondes).
            audio_scores: Liste d'AudioEngagementScore.

        Returns:
            Score de dynamique vocale moyen (0.0 si aucun segment).
        """
        overlapping = [
            s.speech_dynamics_score for s in audio_scores
            if s.end > start and s.start < end
        ]
        if not overlapping:
            return 0.0
        return sum(overlapping) / len(overlapping)

    @staticmethod
    def _compute_hook_score(hook_text: str) -> float:
        """Calcule le score du hook en utilisant HookOptimizer si disponible.

        Args:
            hook_text: Texte du hook a evaluer.

        Returns:
            Score du hook entre 0.0 et 1.0 (0.0 si indisponible).
        """
        if not hook_text or not hook_text.strip():
            return 0.0

        try:
            from ..hook_optimizer import HookOptimizer
            optimizer = HookOptimizer()
            analysis = optimizer.analyze_hook(hook_text)
            return analysis.score
        except (ImportError, Exception):
            # HookOptimizer non disponible ou erreur — on ignore silencieusement
            return 0.0

    @staticmethod
    def _is_covered_by_llm(
        peak_start: float,
        peak_end: float,
        llm_moments: List[LLMScoredMoment],
    ) -> bool:
        """Verifie si un pic audio est deja couvert par un moment LLM.

        Un pic est considere couvert si plus de 50% de sa duree
        chevauche un moment LLM.

        Args:
            peak_start: Debut du pic audio.
            peak_end: Fin du pic audio.
            llm_moments: Liste des moments LLM.

        Returns:
            True si le pic est suffisamment couvert.
        """
        peak_duration = peak_end - peak_start
        if peak_duration <= 0:
            return True

        for moment in llm_moments:
            inter_start = max(peak_start, moment.start)
            inter_end = min(peak_end, moment.end)
            intersection = max(0.0, inter_end - inter_start)

            if intersection / peak_duration > 0.50:
                return True

        return False

    @staticmethod
    def _remove_overlaps(
        candidates: List[ViralCandidate],
    ) -> List[ViralCandidate]:
        """Supprime les candidats qui se chevauchent a plus de 30%.

        Les candidats doivent etre tries par score final decroissant
        avant l'appel. On garde toujours le candidat avec le meilleur
        score.

        Args:
            candidates: Liste de ViralCandidate tries par score decroissant.

        Returns:
            Liste filtree sans chevauchements excessifs.
        """
        if len(candidates) <= 1:
            return candidates

        kept: List[ViralCandidate] = []

        for candidate in candidates:
            c_start = candidate.start
            c_end = candidate.end
            c_duration = c_end - c_start

            is_overlapping = False
            for kept_candidate in kept:
                k_start = kept_candidate.start
                k_end = kept_candidate.end
                k_duration = k_end - k_start

                # Calcul de l'intersection
                inter_start = max(c_start, k_start)
                inter_end = min(c_end, k_end)
                intersection = max(0.0, inter_end - inter_start)

                # Ratio de chevauchement sur le plus petit des deux
                min_duration = min(c_duration, k_duration)
                if min_duration > 0 and intersection / min_duration > _OVERLAP_REMOVAL_RATIO:
                    is_overlapping = True
                    break

            if not is_overlapping:
                kept.append(candidate)

        return kept

    @staticmethod
    def _estimate_video_duration(
        llm_moments: List[LLMScoredMoment],
        audio_scores: List[AudioEngagementScore],
        audio_peaks: List[Tuple[float, float, float]],
    ) -> float:
        """Estime la duree totale de la video a partir des signaux disponibles.

        On prend le maximum de toutes les fins de segments/moments.

        Args:
            llm_moments: Moments LLM.
            audio_scores: Scores audio par segment.
            audio_peaks: Pics audio.

        Returns:
            Duree estimee en secondes (0.0 si aucune donnee).
        """
        max_end = 0.0

        for m in llm_moments:
            if m.end > max_end:
                max_end = m.end

        for s in audio_scores:
            if s.end > max_end:
                max_end = s.end

        for _, peak_end, _ in audio_peaks:
            if peak_end > max_end:
                max_end = peak_end

        return max_end
