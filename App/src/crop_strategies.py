"""Strategies de recadrage adaptatif."""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from rich.console import Console

from .mediapipe_setup import _get_mediapipe
from .content_detector import FocusPoint, ContentType, ContentAnalysis, ContentTypeDetector
from .smoothing import KalmanFilter1D, smooth_value

console = Console()


# ============================================================================
# STRATÉGIES DE CROP ADAPTATIVES
# ============================================================================

class CropStrategy:
    """
    Classe de base pour les stratégies de recadrage.

    Chaque stratégie implémente une logique différente pour déterminer
    le point de focus optimal selon le type de contenu.
    """

    def get_focus_point(
        self,
        frame: np.ndarray,
        prev_focus: Optional[FocusPoint] = None,
        analysis: Optional[ContentAnalysis] = None
    ) -> FocusPoint:
        """
        Calcule le point de focus pour une frame.

        Args:
            frame: Frame BGR à analyser
            prev_focus: Point de focus précédent pour le lissage
            analysis: Analyse de contenu (optionnel)

        Returns:
            FocusPoint avec les coordonnées du centre d'intérêt
        """
        raise NotImplementedError


class FaceTrackingStrategy(CropStrategy):
    """
    Stratégie de tracking de visage avec lissage Kalman.

    Idéale pour: Podcasts, interviews, vlogs, tutoriels face-cam.
    Le visage est maintenu dans le tiers supérieur avec des mouvements fluides.
    """

    def __init__(self):
        self._face_detector = None
        self.kalman_x = KalmanFilter1D(process_variance=0.003, measurement_variance=0.03)
        self.kalman_y = KalmanFilter1D(process_variance=0.003, measurement_variance=0.03)

    def _get_detector(self):
        if self._face_detector is None:
            mp = _get_mediapipe()
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
        return self._face_detector

    def get_focus_point(
        self,
        frame: np.ndarray,
        prev_focus: Optional[FocusPoint] = None,
        analysis: Optional[ContentAnalysis] = None
    ) -> FocusPoint:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detector = self._get_detector()
        results = detector.process(rgb)

        if results.detections:
            # Prendre le plus grand visage
            best = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width *
                             d.location_data.relative_bounding_box.height
            )
            bbox = best.location_data.relative_bounding_box

            raw_x = bbox.xmin + bbox.width / 2
            raw_y = bbox.ymin + bbox.height / 2

            # Appliquer le filtre de Kalman pour lisser
            smooth_x = self.kalman_x.update(raw_x)
            smooth_y = self.kalman_y.update(raw_y)

            return FocusPoint(
                x=smooth_x,
                y=smooth_y,
                confidence=best.score[0],
                width=bbox.width * 2,
                height=bbox.height * 2
            )

        # Pas de visage détecté: utiliser le précédent ou le centre
        if prev_focus:
            return prev_focus
        return FocusPoint(x=0.5, y=0.4, confidence=0.1)

    def reset(self):
        """Réinitialise les filtres de Kalman"""
        self.kalman_x.reset()
        self.kalman_y.reset()

    def close(self):
        if self._face_detector:
            try:
                self._face_detector.close()
            except Exception:
                pass
            self._face_detector = None


class CenterWeightedStrategy(CropStrategy):
    """
    Stratégie de crop centrée avec pondération.

    Idéale pour: Screencasts, tutoriels code, présentations.
    Maintient le centre de l'écran avec léger ajustement vers les zones d'activité.
    """

    def __init__(self, center_weight: float = 0.8):
        """
        Args:
            center_weight: Pondération vers le centre (0.5-1.0)
        """
        self.center_weight = center_weight
        self._prev_gray = None

    def get_focus_point(
        self,
        frame: np.ndarray,
        prev_focus: Optional[FocusPoint] = None,
        analysis: Optional[ContentAnalysis] = None
    ) -> FocusPoint:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Par défaut: centre
        focus_x, focus_y = 0.5, 0.5

        # Détecter les zones d'activité (changement depuis la frame précédente)
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)

            # Diviser en grille 3x3 et trouver la zone avec le plus de changement
            grid_h, grid_w = h // 3, w // 3
            max_activity = 0
            active_region = (1, 1)  # Centre par défaut

            for row in range(3):
                for col in range(3):
                    region = diff[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w]
                    activity = np.mean(region)
                    if activity > max_activity:
                        max_activity = activity
                        active_region = (row, col)

            # Ajuster légèrement vers la zone active
            if max_activity > 5:  # Seuil minimum d'activité
                region_y = (active_region[0] + 0.5) / 3
                region_x = (active_region[1] + 0.5) / 3

                # Pondérer entre le centre et la zone active
                focus_x = 0.5 * self.center_weight + region_x * (1 - self.center_weight)
                focus_y = 0.5 * self.center_weight + region_y * (1 - self.center_weight)

        self._prev_gray = gray.copy()

        return FocusPoint(
            x=focus_x,
            y=focus_y,
            confidence=0.7,
            width=0.6,
            height=0.8
        )

    def reset(self):
        self._prev_gray = None


class MotionTrackingStrategy(CropStrategy):
    """
    Stratégie de tracking basée sur le mouvement.

    Idéale pour: Gaming, sport, action rapide.
    Suit les zones de mouvement intense avec réactivité.
    """

    def __init__(self, reactivity: float = 0.3):
        """
        Args:
            reactivity: Vitesse de réaction aux mouvements (0.1-0.5)
        """
        self.reactivity = reactivity
        self._prev_gray = None
        self._smooth_x = 0.5
        self._smooth_y = 0.5

    def get_focus_point(
        self,
        frame: np.ndarray,
        prev_focus: Optional[FocusPoint] = None,
        analysis: Optional[ContentAnalysis] = None
    ) -> FocusPoint:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        target_x, target_y = 0.5, 0.5
        confidence = 0.5

        if self._prev_gray is not None:
            # Calculer le flux optique simplifié (différence de frames)
            diff = cv2.absdiff(self._prev_gray, gray)

            # Flouter pour réduire le bruit
            diff_blur = cv2.GaussianBlur(diff, (21, 21), 0)

            # Trouver le centre de masse du mouvement
            moments = cv2.moments(diff_blur)
            if moments['m00'] > 1000:  # Seuil minimum de mouvement
                target_x = moments['m10'] / moments['m00'] / w
                target_y = moments['m01'] / moments['m00'] / h
                confidence = min(0.9, moments['m00'] / (w * h * 50))

        # Lissage exponentiel vers la cible
        self._smooth_x = self._smooth_x + self.reactivity * (target_x - self._smooth_x)
        self._smooth_y = self._smooth_y + self.reactivity * (target_y - self._smooth_y)

        self._prev_gray = gray.copy()

        return FocusPoint(
            x=self._smooth_x,
            y=self._smooth_y,
            confidence=confidence,
            width=0.4,
            height=0.5
        )

    def reset(self):
        self._prev_gray = None
        self._smooth_x = 0.5
        self._smooth_y = 0.5


class MultiSubjectStrategy(CropStrategy):
    """
    Stratégie pour scènes avec plusieurs sujets.

    Idéale pour: Conversations, panels, scènes de groupe.
    Trouve le meilleur cadrage pour inclure tous les sujets importants.
    """

    def __init__(self, max_subjects: int = 3):
        """
        Args:
            max_subjects: Nombre maximum de sujets à considérer
        """
        self.max_subjects = max_subjects
        self._face_detector = None
        self.kalman_x = KalmanFilter1D(process_variance=0.005, measurement_variance=0.05)
        self.kalman_y = KalmanFilter1D(process_variance=0.005, measurement_variance=0.05)

    def _get_detector(self):
        if self._face_detector is None:
            mp = _get_mediapipe()
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.4
            )
        return self._face_detector

    def get_focus_point(
        self,
        frame: np.ndarray,
        prev_focus: Optional[FocusPoint] = None,
        analysis: Optional[ContentAnalysis] = None
    ) -> FocusPoint:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detector = self._get_detector()
        results = detector.process(rgb)

        if results.detections:
            # Trier par taille et prendre les N plus grands
            sorted_detections = sorted(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width *
                             d.location_data.relative_bounding_box.height,
                reverse=True
            )[:self.max_subjects]

            # Calculer le centre de tous les visages
            sum_x, sum_y = 0, 0
            total_weight = 0

            for detection in sorted_detections:
                bbox = detection.location_data.relative_bounding_box
                weight = bbox.width * bbox.height  # Pondérer par taille
                sum_x += (bbox.xmin + bbox.width / 2) * weight
                sum_y += (bbox.ymin + bbox.height / 2) * weight
                total_weight += weight

            if total_weight > 0:
                raw_x = sum_x / total_weight
                raw_y = sum_y / total_weight

                # Appliquer le lissage
                smooth_x = self.kalman_x.update(raw_x)
                smooth_y = self.kalman_y.update(raw_y)

                # Calculer la zone englobante
                x_positions = [d.location_data.relative_bounding_box.xmin +
                              d.location_data.relative_bounding_box.width / 2
                              for d in sorted_detections]
                y_positions = [d.location_data.relative_bounding_box.ymin +
                              d.location_data.relative_bounding_box.height / 2
                              for d in sorted_detections]

                width = max(0.4, max(x_positions) - min(x_positions) + 0.2)
                height = max(0.5, max(y_positions) - min(y_positions) + 0.3)

                return FocusPoint(
                    x=smooth_x,
                    y=smooth_y,
                    confidence=0.8,
                    width=width,
                    height=height
                )

        # Fallback
        if prev_focus:
            return prev_focus
        return FocusPoint(x=0.5, y=0.5, confidence=0.3)

    def reset(self):
        self.kalman_x.reset()
        self.kalman_y.reset()

    def close(self):
        if self._face_detector:
            try:
                self._face_detector.close()
            except Exception:
                pass
            self._face_detector = None


class AdaptiveCropManager:
    """
    Gestionnaire de crop adaptatif qui sélectionne automatiquement
    la meilleure stratégie selon le type de contenu détecté.

    Fonctionnalités:
    - Détection automatique du type de contenu
    - Sélection dynamique de la stratégie de crop
    - Transitions fluides entre stratégies
    - Cache des analyses pour la performance
    """

    def __init__(self):
        self.detector = ContentTypeDetector()
        self.strategies: Dict[ContentType, CropStrategy] = {
            ContentType.FACE_CENTRIC: FaceTrackingStrategy(),
            ContentType.SCREEN_CONTENT: CenterWeightedStrategy(),
            ContentType.ACTION_CONTENT: MotionTrackingStrategy(),
            ContentType.MULTI_SUBJECT: MultiSubjectStrategy(),
            ContentType.UNKNOWN: FaceTrackingStrategy(),  # Fallback
        }
        self._current_strategy: Optional[CropStrategy] = None
        self._current_type: ContentType = ContentType.UNKNOWN
        self._analysis_cache: Dict[str, ContentAnalysis] = {}

    def analyze_and_select_strategy(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> Tuple[ContentType, ContentAnalysis]:
        """
        Analyse un segment et sélectionne la stratégie appropriée.

        Args:
            video_path: Chemin vers la vidéo
            start_time: Début du segment
            end_time: Fin du segment

        Returns:
            Tuple (ContentType détecté, ContentAnalysis complète)
        """
        cache_key = f"{video_path}_{start_time:.1f}_{end_time:.1f}"

        if cache_key in self._analysis_cache:
            analysis = self._analysis_cache[cache_key]
        else:
            analysis = self.detector.analyze_segment(video_path, start_time, end_time)
            self._analysis_cache[cache_key] = analysis

        self._current_type = analysis.content_type
        self._current_strategy = self.strategies[analysis.content_type]

        # Réinitialiser la stratégie sélectionnée
        if hasattr(self._current_strategy, 'reset'):
            self._current_strategy.reset()

        console.print(f"[cyan]Type de contenu détecté: {analysis.content_type.name} "
                     f"(confiance: {analysis.confidence:.0%})[/cyan]")

        return analysis.content_type, analysis

    def get_focus_point(
        self,
        frame: np.ndarray,
        prev_focus: Optional[FocusPoint] = None
    ) -> FocusPoint:
        """
        Obtient le point de focus en utilisant la stratégie actuelle.

        Args:
            frame: Frame à analyser
            prev_focus: Point de focus précédent

        Returns:
            FocusPoint optimal selon la stratégie
        """
        if self._current_strategy is None:
            # Fallback si pas de stratégie sélectionnée
            self._current_strategy = self.strategies[ContentType.FACE_CENTRIC]

        return self._current_strategy.get_focus_point(frame, prev_focus)

    def get_current_type(self) -> ContentType:
        """Retourne le type de contenu actuel"""
        return self._current_type

    def clear_cache(self):
        """Vide le cache d'analyses"""
        self._analysis_cache.clear()

    def close(self):
        """Libère toutes les ressources"""
        self.detector.close()
        for strategy in self.strategies.values():
            if hasattr(strategy, 'close'):
                strategy.close()
