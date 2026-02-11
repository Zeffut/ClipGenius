"""Agregation des features audio en scores d'engagement par segment.

Ce module convertit les features audio brutes en scores d'engagement
normalises, puis identifie les regions candidates pour des moments viraux
en se basant uniquement sur les signaux audio.
"""

import numpy as np
from typing import Dict, List, Tuple

from .models import AudioFeatures, AudioEngagementScore, ContentType, PipelineConfig


# Seuil d'ecart entre segments pour la fusion (en secondes)
_MERGE_GAP_THRESHOLD: float = 5.0

# Decalage au-dessus de la moyenne pour definir le seuil de pic
_PEAK_STD_FACTOR: float = 0.5

# Mapping emotion -> score d'engagement brut
_EMOTION_ENGAGEMENT: Dict[str, float] = {
    'happy': 0.9,
    'angry': 0.85,
    'surprise': 1.0,
    'sad': 0.55,
    'neutral': 0.25,
    'fear': 0.7,
    'disgust': 0.65,
}

# Score par defaut pour les emotions non reconnues
_EMOTION_DEFAULT: float = 0.3


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Applique la fonction sigmoide element par element."""
    return 1.0 / (1.0 + np.exp(-x))


def _zscore_normalize(values: np.ndarray) -> np.ndarray:
    """Normalise par z-score puis projette dans [0, 1] via sigmoide.

    Gere le cas ou l'ecart-type est nul (tous les elements identiques)
    en renvoyant 0.5 pour chaque element.
    """
    if values.size == 0:
        return values
    std = np.std(values)
    if std == 0.0:
        return np.full_like(values, 0.5, dtype=np.float64)
    z = (values - np.mean(values)) / std
    return _sigmoid(z)


class AudioScoreAggregator:
    """Agrege les features audio en scores d'engagement par segment."""

    # Poids par defaut pour chaque composante
    DEFAULT_WEIGHTS: Dict[str, float] = {
        'energy': 0.25,
        'pitch': 0.20,
        'speech_dynamics': 0.25,
        'emotion': 0.20,
        'interaction': 0.10,
    }

    # Poids adaptes par type de contenu
    CONTENT_WEIGHTS: Dict[ContentType, Dict[str, float]] = {
        ContentType.PODCAST: {
            'energy': 0.10,
            'pitch': 0.20,
            'speech_dynamics': 0.30,
            'emotion': 0.25,
            'interaction': 0.15,
        },
        ContentType.INTERVIEW: {
            'energy': 0.10,
            'pitch': 0.20,
            'speech_dynamics': 0.30,
            'emotion': 0.25,
            'interaction': 0.15,
        },
        ContentType.COMEDY: {
            'energy': 0.30,
            'pitch': 0.15,
            'speech_dynamics': 0.15,
            'emotion': 0.30,
            'interaction': 0.10,
        },
        ContentType.GAMING: {
            'energy': 0.30,
            'pitch': 0.15,
            'speech_dynamics': 0.15,
            'emotion': 0.30,
            'interaction': 0.10,
        },
        ContentType.ACTION: {
            'energy': 0.35,
            'pitch': 0.15,
            'speech_dynamics': 0.10,
            'emotion': 0.30,
            'interaction': 0.10,
        },
        ContentType.TUTORIAL: {
            'energy': 0.10,
            'pitch': 0.15,
            'speech_dynamics': 0.35,
            'emotion': 0.15,
            'interaction': 0.25,
        },
        ContentType.NEWS: {
            'energy': 0.10,
            'pitch': 0.15,
            'speech_dynamics': 0.35,
            'emotion': 0.20,
            'interaction': 0.20,
        },
        ContentType.VLOG: {
            'energy': 0.20,
            'pitch': 0.20,
            'speech_dynamics': 0.20,
            'emotion': 0.25,
            'interaction': 0.15,
        },
        ContentType.MUSIC: {
            'energy': 0.35,
            'pitch': 0.30,
            'speech_dynamics': 0.05,
            'emotion': 0.25,
            'interaction': 0.05,
        },
        ContentType.MOTIVATIONAL: {
            'energy': 0.20,
            'pitch': 0.20,
            'speech_dynamics': 0.20,
            'emotion': 0.30,
            'interaction': 0.10,
        },
    }

    def __init__(self, content_type: ContentType = ContentType.UNKNOWN) -> None:
        """Initialise l'agregateur avec les poids adaptes au type de contenu.

        Args:
            content_type: Type de contenu de la video. Determine les poids
                appliques a chaque composante du score d'engagement.
        """
        self._content_type: ContentType = content_type
        self._weights: Dict[str, float] = self.CONTENT_WEIGHTS.get(
            content_type, self.DEFAULT_WEIGHTS
        ).copy()

    @property
    def content_type(self) -> ContentType:
        """Retourne le type de contenu courant."""
        return self._content_type

    @property
    def weights(self) -> Dict[str, float]:
        """Retourne une copie des poids courants."""
        return self._weights.copy()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def aggregate(
        self, features: List[AudioFeatures]
    ) -> List[AudioEngagementScore]:
        """Convertit une liste de features audio en scores d'engagement.

        Chaque dimension est normalisee (z-score + sigmoide) sur l'ensemble
        des segments, puis ponderee selon le type de contenu.

        Args:
            features: Liste de features audio brutes par segment.

        Returns:
            Liste d'AudioEngagementScore, un par segment d'entree.
        """
        if not features:
            return []

        n = len(features)

        # -- Extraction des vecteurs bruts --
        raw_energy = np.array([f.rms_energy for f in features], dtype=np.float64)
        raw_pitch_std = np.array([f.pitch_std for f in features], dtype=np.float64)
        raw_speech_delta = np.array(
            [abs(f.speech_rate_delta) for f in features], dtype=np.float64
        )
        raw_overlap = np.array([f.overlap_ratio for f in features], dtype=np.float64)
        raw_speaker_changes = np.array(
            [float(f.speaker_changes) for f in features], dtype=np.float64
        )

        # -- Normalisation z-score + sigmoide --
        norm_energy = _zscore_normalize(raw_energy)
        norm_pitch = _zscore_normalize(raw_pitch_std)
        norm_speech = _zscore_normalize(raw_speech_delta)
        norm_overlap = _zscore_normalize(raw_overlap)
        norm_speaker = _zscore_normalize(raw_speaker_changes)

        # -- Construction des scores par segment --
        scores: List[AudioEngagementScore] = []

        for i in range(n):
            f = features[i]

            energy_score = float(norm_energy[i])
            pitch_score = float(norm_pitch[i])
            speech_dynamics_score = float(norm_speech[i])
            emotion_score = self._compute_emotion_score(f.emotion, f.emotion_confidence)
            interaction_score = self._compute_interaction_score(
                float(norm_overlap[i]), float(norm_speaker[i])
            )

            # Score composite pondere
            composite = (
                self._weights['energy'] * energy_score
                + self._weights['pitch'] * pitch_score
                + self._weights['speech_dynamics'] * speech_dynamics_score
                + self._weights['emotion'] * emotion_score
                + self._weights['interaction'] * interaction_score
            )

            scores.append(
                AudioEngagementScore(
                    start=f.start,
                    end=f.end,
                    score=composite,
                    energy_score=energy_score,
                    pitch_score=pitch_score,
                    speech_dynamics_score=speech_dynamics_score,
                    emotion_score=emotion_score,
                    interaction_score=interaction_score,
                )
            )

        return scores

    # ------------------------------------------------------------------
    # Detection de regions
    # ------------------------------------------------------------------

    def find_peak_regions(
        self,
        scores: List[AudioEngagementScore],
        min_duration: float = 30.0,
        max_duration: float = 90.0,
    ) -> List[Tuple[float, float, float]]:
        """Identifie les regions temporelles a fort engagement audio.

        Algorithme :
        1. Calcule un seuil = mean + 0.5 * std des scores composites.
        2. Selectionne les segments au-dessus du seuil.
        3. Fusionne les segments separes par moins de 5 secondes.
        4. Ajuste chaque region pour respecter min/max duration.
        5. Trie par score moyen decroissant.

        Args:
            scores: Liste d'AudioEngagementScore (issus de aggregate()).
            min_duration: Duree minimale d'une region candidate (secondes).
            max_duration: Duree maximale d'une region candidate (secondes).

        Returns:
            Liste de tuples (start, end, avg_score), tries par avg_score
            decroissant.
        """
        if not scores:
            return []

        score_values = np.array([s.score for s in scores], dtype=np.float64)
        mean_score = float(np.mean(score_values))
        std_score = float(np.std(score_values))
        threshold = mean_score + _PEAK_STD_FACTOR * std_score

        # -- Etape 1 : segments au-dessus du seuil --
        above: List[AudioEngagementScore] = [s for s in scores if s.score >= threshold]

        if not above:
            return []

        # -- Etape 2 : regrouper en regions contigues --
        regions: List[List[AudioEngagementScore]] = []
        current_region: List[AudioEngagementScore] = [above[0]]

        for seg in above[1:]:
            prev_end = current_region[-1].end
            if seg.start - prev_end <= _MERGE_GAP_THRESHOLD:
                current_region.append(seg)
            else:
                regions.append(current_region)
                current_region = [seg]
        regions.append(current_region)

        # -- Etape 3 : convertir en (start, end, avg_score) et ajuster durees --
        # On a besoin de l'etendue temporelle totale pour borner l'expansion
        global_start = scores[0].start
        global_end = scores[-1].end

        candidates: List[Tuple[float, float, float]] = []

        for region in regions:
            r_start = region[0].start
            r_end = region[-1].end
            avg_score = float(np.mean([s.score for s in region]))

            r_start, r_end = self._adjust_region(
                r_start, r_end, min_duration, max_duration,
                global_start, global_end, scores,
            )

            # Recalculer le score moyen sur la region ajustee
            avg_score = self._region_avg_score(scores, r_start, r_end)
            candidates.append((r_start, r_end, avg_score))

        # -- Deduplication de regions qui se chevauchent fortement --
        candidates = self._deduplicate_regions(candidates)

        # -- Tri par score decroissant --
        candidates.sort(key=lambda c: c[2], reverse=True)

        return candidates

    # ------------------------------------------------------------------
    # Helpers prives
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_emotion_score(emotion: str, confidence: float) -> float:
        """Calcule le score d'engagement lie a l'emotion detectee.

        Mappe l'emotion a un score de base, puis le pondere par la
        confiance du modele.
        """
        base = _EMOTION_ENGAGEMENT.get(emotion.lower(), _EMOTION_DEFAULT)
        # La confiance est attendue dans [0, 1] ; on clamp par securite
        conf = max(0.0, min(1.0, confidence))
        # Melange : si la confiance est faible, on tire vers 'neutral'
        neutral_base = _EMOTION_ENGAGEMENT['neutral']
        return conf * base + (1.0 - conf) * neutral_base

    @staticmethod
    def _compute_interaction_score(
        norm_overlap: float, norm_speaker: float
    ) -> float:
        """Combine les signaux d'interaction (chevauchement + changements).

        Les deux composantes sont deja normalisees dans [0, 1].
        On utilise une moyenne ponderee : 40% overlap, 60% speaker_changes.
        """
        return 0.4 * norm_overlap + 0.6 * norm_speaker

    @staticmethod
    def _adjust_region(
        start: float,
        end: float,
        min_duration: float,
        max_duration: float,
        global_start: float,
        global_end: float,
        scores: List[AudioEngagementScore],
    ) -> Tuple[float, float]:
        """Ajuste une region pour respecter les contraintes de duree.

        - Si la region est trop courte, on l'etend symetriquement.
        - Si la region est trop longue, on la reduit en gardant le centre
          le plus engage.
        """
        duration = end - start

        if duration < min_duration:
            # Expansion symetrique
            deficit = min_duration - duration
            expand_before = deficit / 2.0
            expand_after = deficit / 2.0

            new_start = start - expand_before
            new_end = end + expand_after

            # Clamper aux bornes globales
            if new_start < global_start:
                # Reporter le surplus vers la fin
                new_end += (global_start - new_start)
                new_start = global_start
            if new_end > global_end:
                # Reporter le surplus vers le debut
                new_start -= (new_end - global_end)
                new_end = global_end

            # Clamper une derniere fois
            new_start = max(new_start, global_start)
            new_end = min(new_end, global_end)

            start, end = new_start, new_end

        elif duration > max_duration:
            # Reduction : trouver la sous-fenetre de max_duration avec le
            # meilleur score moyen
            best_start = start
            best_avg = -1.0
            step = 1.0  # pas de recherche en secondes

            t = start
            while t + max_duration <= end + 0.01:
                avg = AudioScoreAggregator._region_avg_score(
                    scores, t, t + max_duration
                )
                if avg > best_avg:
                    best_avg = avg
                    best_start = t
                t += step

            start = best_start
            end = start + max_duration

        return start, end

    @staticmethod
    def _region_avg_score(
        scores: List[AudioEngagementScore],
        start: float,
        end: float,
    ) -> float:
        """Calcule le score moyen des segments qui chevauchent [start, end].

        Un segment est inclus si son intersection avec la region est non nulle.
        """
        included = [
            s.score for s in scores
            if s.end > start and s.start < end
        ]
        if not included:
            return 0.0
        return float(np.mean(included))

    @staticmethod
    def _deduplicate_regions(
        candidates: List[Tuple[float, float, float]],
        overlap_threshold: float = 0.5,
    ) -> List[Tuple[float, float, float]]:
        """Supprime les regions qui se chevauchent trop.

        Lorsque deux regions se chevauchent a plus de overlap_threshold
        (ratio de l'intersection sur la plus petite des deux), on garde
        celle avec le meilleur score.
        """
        if len(candidates) <= 1:
            return candidates

        # Trier par score decroissant pour garder les meilleures en priorite
        sorted_candidates = sorted(candidates, key=lambda c: c[2], reverse=True)
        kept: List[Tuple[float, float, float]] = []

        for candidate in sorted_candidates:
            c_start, c_end, c_score = candidate
            c_duration = c_end - c_start

            is_duplicate = False
            for k_start, k_end, _k_score in kept:
                # Calcul de l'intersection
                inter_start = max(c_start, k_start)
                inter_end = min(c_end, k_end)
                intersection = max(0.0, inter_end - inter_start)

                k_duration = k_end - k_start
                min_duration = min(c_duration, k_duration)

                if min_duration > 0 and intersection / min_duration > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(candidate)

        return kept
