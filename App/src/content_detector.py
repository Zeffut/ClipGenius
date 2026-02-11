"""Detection du type de contenu video."""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum, auto
from rich.console import Console

from .mediapipe_setup import _get_mediapipe

console = Console()

# Constantes de configuration
DEFAULT_SAMPLE_FRAMES = 10
FALLBACK_FPS = 30.0
PIXEL_VALUE_MAX = 255.0
EDGE_ANALYSIS_WIDTH = 320
EDGE_ANALYSIS_HEIGHT = 180
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150


@dataclass
class FocusPoint:
    """Point de focus dans une frame"""
    x: float  # Position X normalisée (0-1)
    y: float  # Position Y normalisée (0-1)
    confidence: float
    width: float = 0.3  # Largeur normalisée de la zone d'intérêt
    height: float = 0.5  # Hauteur normalisée de la zone d'intérêt


# ============================================================================
# SYSTÈME DE DÉTECTION DE TYPE DE CONTENU - Smart Crop Intelligent
# ============================================================================


class ContentType(Enum):
    """
    Types de contenu vidéo détectables pour adapter la stratégie de crop.

    Chaque type a des caractéristiques visuelles distinctes qui nécessitent
    une approche de recadrage différente pour un résultat optimal.
    """
    FACE_CENTRIC = auto()      # Podcast, interview, vlog - visage dominant
    SCREEN_CONTENT = auto()    # Tutoriel, coding, screencast - peu de mouvement, texte
    ACTION_CONTENT = auto()    # Gaming, sport, action - mouvement rapide
    MULTI_SUBJECT = auto()     # Plusieurs personnes, scènes variées
    UNKNOWN = auto()           # Type non identifié, utiliser stratégie par défaut


@dataclass
class ContentAnalysis:
    """
    Résultat de l'analyse de contenu d'un segment vidéo.

    Contient toutes les métriques calculées pour déterminer le type de contenu
    et la stratégie de crop optimale.
    """
    content_type: ContentType
    confidence: float                    # Confiance dans la détection (0-1)

    # Métriques de visage
    face_presence_ratio: float = 0.0     # % de frames avec visage détecté
    face_size_avg: float = 0.0           # Taille moyenne du visage (% de la frame)
    face_count_avg: float = 0.0          # Nombre moyen de visages par frame
    face_stability: float = 0.0          # Stabilité de la position du visage (0-1)

    # Métriques de mouvement
    motion_intensity: float = 0.0        # Intensité moyenne du mouvement (0-1)
    motion_variance: float = 0.0         # Variance du mouvement (pics d'action)
    optical_flow_magnitude: float = 0.0  # Magnitude moyenne du flux optique

    # Métriques de contenu écran
    edge_density: float = 0.0            # Densité de contours (texte, UI)
    color_uniformity: float = 0.0        # Uniformité des couleurs (fond uni = screencast)
    text_region_ratio: float = 0.0       # % de la frame avec du texte potentiel

    # Métriques de scène
    scene_change_count: int = 0          # Nombre de changements de scène
    dominant_region: str = "center"      # Région dominante (center, top, bottom, left, right)


@dataclass
class CropResult:
    """Résultat du calcul de crop avec info sur le fond flouté"""
    x1: int
    y1: int
    x2: int
    y2: int
    needs_blur_fill: bool = False  # True si le visage est trop bas et nécessite un fond flouté
    blur_fill_height: int = 0  # Hauteur du fond flouté à ajouter en bas
    face_y_in_crop: float = 0.0  # Position Y du visage dans le crop (0-1)


class ContentTypeDetector:
    """
    Détecteur intelligent de type de contenu vidéo.

    Analyse les caractéristiques visuelles d'un segment vidéo pour déterminer
    automatiquement le type de contenu et recommander la meilleure stratégie
    de recadrage.

    Utilise plusieurs techniques d'analyse:
    - Détection de visages (MediaPipe)
    - Analyse de mouvement (optical flow)
    - Détection de contours (texte/UI)
    - Analyse colorimétrique
    - Détection de changements de scène
    """

    def __init__(self, sample_frames: int = DEFAULT_SAMPLE_FRAMES):
        """
        Args:
            sample_frames: Nombre de frames à analyser par segment
        """
        self.sample_frames = sample_frames
        self._face_detector = None
        self._prev_frame_gray = None

    def _get_face_detector(self):
        """Initialisation lazy du détecteur de visages"""
        if self._face_detector is None:
            mp = _get_mediapipe()
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.4
            )
        return self._face_detector

    def analyze_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> ContentAnalysis:
        """
        Analyse un segment vidéo pour déterminer son type de contenu.

        Args:
            video_path: Chemin vers la vidéo
            start_time: Début du segment en secondes
            end_time: Fin du segment en secondes

        Returns:
            ContentAnalysis avec le type détecté et les métriques
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = FALLBACK_FPS  # Fallback FPS par défaut
        duration = end_time - start_time

        # Calculer les frames à analyser
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame

        # Échantillonner uniformément
        frame_indices = np.linspace(start_frame, end_frame - 1, self.sample_frames, dtype=int)

        # Métriques à collecter
        faces_detected = []
        face_sizes = []
        face_positions = []
        motion_scores = []
        edge_densities = []
        color_uniformities = []

        self._prev_frame_gray = None

        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                h, w = frame.shape[:2]

                # 1. Analyse des visages
                face_info = self._analyze_faces(frame, w, h)
                faces_detected.append(face_info['count'])
                if face_info['size'] > 0:
                    face_sizes.append(face_info['size'])
                    face_positions.append((face_info['x'], face_info['y']))

                # 2. Analyse du mouvement
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = self._analyze_motion(gray)
                motion_scores.append(motion)

                # 3. Analyse des contours (texte/UI)
                edge_density = self._analyze_edges(gray)
                edge_densities.append(edge_density)

                # 4. Analyse de l'uniformité des couleurs
                uniformity = self._analyze_color_uniformity(frame)
                color_uniformities.append(uniformity)

                self._prev_frame_gray = gray
        finally:
            cap.release()

        # Calculer les métriques agrégées
        analysis = self._compute_analysis(
            faces_detected, face_sizes, face_positions,
            motion_scores, edge_densities, color_uniformities
        )

        return analysis

    def _analyze_faces(self, frame: np.ndarray, w: int, h: int) -> Dict[str, float]:
        """Analyse les visages dans une frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector = self._get_face_detector()
        results = detector.process(rgb_frame)

        if not results.detections:
            return {'count': 0, 'size': 0, 'x': 0.5, 'y': 0.5}

        # Prendre le plus grand visage
        best = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width *
                         d.location_data.relative_bounding_box.height
        )
        bbox = best.location_data.relative_bounding_box

        return {
            'count': len(results.detections),
            'size': bbox.width * bbox.height,
            'x': bbox.xmin + bbox.width / 2,
            'y': bbox.ymin + bbox.height / 2
        }

    def _analyze_motion(self, gray: np.ndarray) -> float:
        """Analyse le mouvement entre frames via différence absolue"""
        if self._prev_frame_gray is None:
            return 0.0

        # Différence absolue (plus rapide que optical flow)
        diff = cv2.absdiff(self._prev_frame_gray, gray)
        motion = np.mean(diff) / PIXEL_VALUE_MAX

        return motion

    def _analyze_edges(self, gray: np.ndarray) -> float:
        """Analyse la densité de contours (indicateur de texte/UI)"""
        # Réduire la résolution pour la vitesse
        small = cv2.resize(gray, (EDGE_ANALYSIS_WIDTH, EDGE_ANALYSIS_HEIGHT))
        edges = cv2.Canny(small, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        density = np.mean(edges) / PIXEL_VALUE_MAX
        return density

    def _analyze_color_uniformity(self, frame: np.ndarray) -> float:
        """Analyse l'uniformité des couleurs (fond uni = screencast probable)"""
        # Réduire la résolution
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Écart-type de la saturation et de la valeur
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])

        # Plus l'écart-type est bas, plus c'est uniforme
        uniformity = 1.0 - min(1.0, (s_std + v_std) / 200.0)
        return uniformity

    def _compute_analysis(
        self,
        faces_detected: List[int],
        face_sizes: List[float],
        face_positions: List[Tuple[float, float]],
        motion_scores: List[float],
        edge_densities: List[float],
        color_uniformities: List[float]
    ) -> ContentAnalysis:
        """Calcule l'analyse finale à partir des métriques collectées"""

        # Métriques de visage
        face_presence = sum(1 for f in faces_detected if f > 0) / max(1, len(faces_detected))
        face_size_avg = np.mean(face_sizes) if face_sizes else 0.0
        face_count_avg = np.mean(faces_detected) if faces_detected else 0.0

        # Stabilité du visage (variance de position)
        face_stability = 0.0
        if len(face_positions) > 1:
            x_positions = [p[0] for p in face_positions]
            y_positions = [p[1] for p in face_positions]
            position_variance = np.std(x_positions) + np.std(y_positions)
            face_stability = max(0, 1.0 - position_variance * 5)

        # Métriques de mouvement
        motion_intensity = np.mean(motion_scores) if motion_scores else 0.0
        motion_variance = np.std(motion_scores) if motion_scores else 0.0

        # Métriques de contenu écran
        edge_density = np.mean(edge_densities) if edge_densities else 0.0
        color_uniformity = np.mean(color_uniformities) if color_uniformities else 0.0

        # === CLASSIFICATION DU TYPE DE CONTENU ===
        content_type = ContentType.UNKNOWN
        confidence = 0.5

        # Règles de classification avec scores pondérés
        scores = {
            ContentType.FACE_CENTRIC: 0.0,
            ContentType.SCREEN_CONTENT: 0.0,
            ContentType.ACTION_CONTENT: 0.0,
            ContentType.MULTI_SUBJECT: 0.0
        }

        # FACE_CENTRIC: Visage présent, stable, taille moyenne à grande
        if face_presence > 0.6:
            scores[ContentType.FACE_CENTRIC] += face_presence * 0.4
            if face_size_avg > 0.02:  # Visage occupe >2% de la frame
                scores[ContentType.FACE_CENTRIC] += min(0.3, face_size_avg * 5)
            if face_stability > 0.7:
                scores[ContentType.FACE_CENTRIC] += 0.2
            if face_count_avg < 1.5:  # Généralement 1 visage
                scores[ContentType.FACE_CENTRIC] += 0.1

        # SCREEN_CONTENT: Peu de mouvement, beaucoup de contours, fond uniforme
        if motion_intensity < 0.03:
            scores[ContentType.SCREEN_CONTENT] += 0.3
        if edge_density > 0.15:  # Beaucoup de contours (texte/code)
            scores[ContentType.SCREEN_CONTENT] += min(0.3, edge_density * 2)
        if color_uniformity > 0.5:
            scores[ContentType.SCREEN_CONTENT] += 0.2
        if face_presence < 0.3:
            scores[ContentType.SCREEN_CONTENT] += 0.2

        # ACTION_CONTENT: Mouvement élevé, variance de mouvement
        if motion_intensity > 0.05:
            scores[ContentType.ACTION_CONTENT] += min(0.4, motion_intensity * 5)
        if motion_variance > 0.02:
            scores[ContentType.ACTION_CONTENT] += min(0.3, motion_variance * 10)
        if face_stability < 0.3:  # Visage instable = action
            scores[ContentType.ACTION_CONTENT] += 0.2

        # MULTI_SUBJECT: Plusieurs visages
        if face_count_avg > 1.5:
            scores[ContentType.MULTI_SUBJECT] += 0.5
            if face_stability < 0.5:
                scores[ContentType.MULTI_SUBJECT] += 0.2

        # Sélectionner le type avec le score le plus élevé
        best_type = max(scores.items(), key=lambda x: x[1])
        if best_type[1] > 0.3:
            content_type = best_type[0]
            confidence = min(0.95, best_type[1] + 0.3)

        return ContentAnalysis(
            content_type=content_type,
            confidence=confidence,
            face_presence_ratio=face_presence,
            face_size_avg=face_size_avg,
            face_count_avg=face_count_avg,
            face_stability=face_stability,
            motion_intensity=motion_intensity,
            motion_variance=motion_variance,
            edge_density=edge_density,
            color_uniformity=color_uniformity
        )

    def close(self):
        """Libère les ressources"""
        if self._face_detector:
            try:
                self._face_detector.close()
            except Exception:
                pass
            self._face_detector = None
