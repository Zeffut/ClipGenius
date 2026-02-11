#!/usr/bin/env python3
"""
FocusTracker - Système de tracking caméra intelligent pour ClipGenius

Ce module remplace l'ancien système de smart_cropper avec une approche simplifiée
et plus efficace basée sur:
- Détection de visage MediaPipe optimisée
- Filtre de Kalman prédictif (utilise la vélocité)
- Smoothing unique et configurable
- Presets adaptés au type de contenu

Auteur: ClipGenius
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
from rich.console import Console

console = Console()


# =============================================================================
# PRESETS DE TRACKING
# =============================================================================

class TrackingPreset(Enum):
    """Presets de tracking adaptés au type de contenu"""
    PODCAST = "podcast"           # 1-2 personnes, mouvements lents
    INTERVIEW = "interview"       # 2 personnes, changements de speaker
    VLOG = "vlog"                 # 1 personne, mouvements modérés
    GAMING = "gaming"             # Focus écran, visage en coin
    TUTORIAL = "tutorial"         # Écran principal, visage secondaire
    ACTION = "action"             # Mouvements rapides, suivi dynamique
    PRESENTATION = "presentation" # Speaker + slides
    AUTO = "auto"                 # Détection automatique


@dataclass
class TrackingConfig:
    """Configuration du tracking pour un preset donné"""
    # Réactivité: 0.0 = très lent (stable), 1.0 = très réactif (saccadé)
    reactivity: float = 0.15
    
    # Vitesse max de déplacement (% de l'écran par seconde)
    max_velocity: float = 0.5
    
    # Zone morte: pas de mouvement si le sujet est dans cette zone centrale (%)
    dead_zone: float = 0.15
    
    # Priorité visage vs centre (0.0 = centre, 1.0 = visage)
    face_priority: float = 0.9
    
    # Sample rate pour l'analyse (FPS)
    sample_rate: int = 5
    
    # Délai de prédiction (secondes) - anticipe le mouvement
    prediction_delay: float = 0.1
    
    # Seuil de confiance minimum pour la détection
    min_confidence: float = 0.5
    
    # Position par défaut si pas de détection (0.5 = centre)
    default_x: float = 0.5
    default_y: float = 0.4  # Légèrement au-dessus du centre
    
    # Multi-sujet: suivre le plus grand visage ou le centroïde
    multi_subject_mode: str = "largest"  # "largest", "centroid", "speaking"


# Configurations par preset
PRESET_CONFIGS: Dict[TrackingPreset, TrackingConfig] = {
    TrackingPreset.PODCAST: TrackingConfig(
        reactivity=0.08,
        max_velocity=0.3,
        dead_zone=0.20,
        face_priority=1.0,
        sample_rate=4,
        multi_subject_mode="largest"
    ),
    TrackingPreset.INTERVIEW: TrackingConfig(
        reactivity=0.12,
        max_velocity=0.6,
        dead_zone=0.15,
        face_priority=1.0,
        sample_rate=5,
        multi_subject_mode="speaking"  # TODO: implémenter détection speaker
    ),
    TrackingPreset.VLOG: TrackingConfig(
        reactivity=0.15,
        max_velocity=0.5,
        dead_zone=0.12,
        face_priority=0.95,
        sample_rate=5,
    ),
    TrackingPreset.GAMING: TrackingConfig(
        reactivity=0.05,
        max_velocity=0.2,
        dead_zone=0.30,
        face_priority=0.3,  # Focus sur l'écran principalement
        sample_rate=3,
        default_x=0.5,
        default_y=0.45,
    ),
    TrackingPreset.TUTORIAL: TrackingConfig(
        reactivity=0.06,
        max_velocity=0.25,
        dead_zone=0.25,
        face_priority=0.4,
        sample_rate=3,
    ),
    TrackingPreset.ACTION: TrackingConfig(
        reactivity=0.25,
        max_velocity=0.8,
        dead_zone=0.08,
        face_priority=0.7,
        sample_rate=8,
        prediction_delay=0.15,
    ),
    TrackingPreset.PRESENTATION: TrackingConfig(
        reactivity=0.10,
        max_velocity=0.4,
        dead_zone=0.18,
        face_priority=0.8,
        sample_rate=4,
    ),
    TrackingPreset.AUTO: TrackingConfig(
        reactivity=0.12,
        max_velocity=0.5,
        dead_zone=0.15,
        face_priority=0.85,
        sample_rate=5,
    ),
}


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class FocusPoint:
    """Point de focus avec métadonnées"""
    x: float              # Position X normalisée (0-1)
    y: float              # Position Y normalisée (0-1)
    confidence: float     # Confiance de la détection (0-1)
    timestamp: float      # Timestamp en secondes
    face_size: float = 0  # Taille relative du visage (0-1)
    num_faces: int = 0    # Nombre de visages détectés


@dataclass
class TrackerState:
    """État interne du tracker (Kalman)"""
    x: float = 0.5
    y: float = 0.4
    vx: float = 0.0  # Vélocité X
    vy: float = 0.0  # Vélocité Y
    last_update: float = 0.0
    confidence: float = 0.5


# =============================================================================
# FILTRE DE KALMAN PRÉDICTIF 2D
# =============================================================================

class PredictiveKalman2D:
    """
    Filtre de Kalman 2D avec prédiction de vélocité.
    
    Contrairement à un Kalman classique, celui-ci:
    - Utilise la vélocité pour prédire la position future
    - S'adapte dynamiquement au bruit de mesure
    - Gère les gaps de détection avec la prédiction
    """
    
    def __init__(
        self,
        process_noise: float = 0.02,
        measurement_noise: float = 0.1,
        initial_x: float = 0.5,
        initial_y: float = 0.4
    ):
        """
        Args:
            process_noise: Bruit du processus (mouvement naturel)
            measurement_noise: Bruit de mesure (erreur détection)
            initial_x, initial_y: Position initiale
        """
        # État: [x, y, vx, vy]
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)
        
        # Covariance de l'état (incertitude)
        self.P = np.eye(4) * 0.1
        
        # Bruit du processus
        self.Q = np.eye(4) * process_noise
        self.Q[2, 2] *= 2  # Plus de bruit sur la vélocité
        self.Q[3, 3] *= 2
        
        # Bruit de mesure de base
        self.base_R = measurement_noise
        
        # Matrice d'observation (on observe x, y seulement)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        self.last_time = 0.0
        self._initialized = False
    
    def predict(self, dt: float) -> Tuple[float, float]:
        """
        Prédit la position future basée sur la vélocité actuelle.
        
        Args:
            dt: Delta temps en secondes
            
        Returns:
            (x, y) position prédite
        """
        if dt <= 0:
            return self.state[0], self.state[1]
        
        # Matrice de transition d'état
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Prédiction de l'état
        predicted_state = F @ self.state
        
        # Prédiction de la covariance
        self.P = F @ self.P @ F.T + self.Q * dt
        
        self.state = predicted_state
        return self.state[0], self.state[1]
    
    def update(
        self,
        measured_x: float,
        measured_y: float,
        confidence: float,
        timestamp: float
    ) -> Tuple[float, float]:
        """
        Met à jour l'état avec une nouvelle mesure.
        
        Args:
            measured_x, measured_y: Position mesurée (0-1)
            confidence: Confiance de la mesure (0-1)
            timestamp: Timestamp en secondes
            
        Returns:
            (x, y) position filtrée
        """
        dt = timestamp - self.last_time if self._initialized else 0.033
        self.last_time = timestamp
        
        if not self._initialized:
            self.state[0] = measured_x
            self.state[1] = measured_y
            self._initialized = True
            return measured_x, measured_y
        
        # Prédiction
        self.predict(dt)
        
        # Bruit de mesure adaptatif (moins de confiance = plus de bruit)
        R = np.eye(2) * self.base_R * (2.0 - confidence)
        
        # Mesure
        z = np.array([measured_x, measured_y])
        
        # Innovation (différence entre mesure et prédiction)
        y = z - self.H @ self.state
        
        # Covariance de l'innovation
        S = self.H @ self.P @ self.H.T + R
        
        # Gain de Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Mise à jour de l'état
        self.state = self.state + K @ y
        
        # Mise à jour de la covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        # Clamp les positions entre 0 et 1
        self.state[0] = np.clip(self.state[0], 0.05, 0.95)
        self.state[1] = np.clip(self.state[1], 0.05, 0.95)
        
        return self.state[0], self.state[1]
    
    def get_predicted_position(self, future_dt: float) -> Tuple[float, float]:
        """
        Retourne la position prédite dans future_dt secondes.
        Sans modifier l'état interne.
        """
        x = self.state[0] + self.state[2] * future_dt
        y = self.state[1] + self.state[3] * future_dt
        return np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Retourne la vélocité actuelle (vx, vy)"""
        return self.state[2], self.state[3]
    
    @property
    def position(self) -> Tuple[float, float]:
        """Retourne la position actuelle (x, y)"""
        return self.state[0], self.state[1]


# =============================================================================
# DÉTECTEUR DE VISAGE OPTIMISÉ
# =============================================================================

class FaceDetector:
    """
    Détecteur de visage optimisé avec MediaPipe.
    
    Fonctionnalités:
    - Détection multi-visages
    - Sélection du visage principal (plus grand, plus centré, speaker)
    - Cache des résultats pour les frames proches
    """
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Args:
            min_confidence: Confiance minimum pour une détection valide
        """
        self.min_confidence = min_confidence
        self._detector = None
        self._cache: Dict[int, List[dict]] = {}  # frame_idx -> faces
        self._cache_max_size = 100
    
    def _get_detector(self):
        """Lazy initialization du détecteur MediaPipe"""
        if self._detector is None:
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 = full range (meilleur pour vidéo)
                min_detection_confidence=self.min_confidence
            )
        return self._detector
    
    def detect(self, frame: np.ndarray, frame_idx: int = -1) -> List[dict]:
        """
        Détecte les visages dans une frame.
        
        Args:
            frame: Image BGR (numpy array)
            frame_idx: Index de la frame (pour le cache)
            
        Returns:
            Liste de dictionnaires avec les infos de chaque visage:
            {
                'x': float,       # Centre X normalisé (0-1)
                'y': float,       # Centre Y normalisé (0-1)
                'width': float,   # Largeur normalisée
                'height': float,  # Hauteur normalisée
                'confidence': float,
                'area': float     # Surface relative
            }
        """
        # Check cache
        if frame_idx >= 0 and frame_idx in self._cache:
            return self._cache[frame_idx]
        
        detector = self._get_detector()
        h, w = frame.shape[:2]
        
        # Convertir en RGB pour MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Centre du visage
                cx = bbox.xmin + bbox.width / 2
                cy = bbox.ymin + bbox.height / 2
                
                # Surface relative
                area = bbox.width * bbox.height
                
                faces.append({
                    'x': cx,
                    'y': cy,
                    'width': bbox.width,
                    'height': bbox.height,
                    'confidence': detection.score[0],
                    'area': area
                })
        
        # Trier par taille (plus grand en premier)
        faces.sort(key=lambda f: f['area'], reverse=True)
        
        # Cache
        if frame_idx >= 0:
            self._cache[frame_idx] = faces
            # Nettoyer le cache si trop grand
            if len(self._cache) > self._cache_max_size:
                oldest = min(self._cache.keys())
                del self._cache[oldest]
        
        return faces
    
    def select_primary_face(
        self,
        faces: List[dict],
        mode: str = "largest",
        prev_position: Optional[Tuple[float, float]] = None
    ) -> Optional[dict]:
        """
        Sélectionne le visage principal selon le mode.
        
        Args:
            faces: Liste des visages détectés
            mode: "largest", "centroid", "nearest"
            prev_position: Position précédente (pour mode "nearest")
            
        Returns:
            Le visage sélectionné ou None
        """
        if not faces:
            return None
        
        if mode == "largest":
            return faces[0]  # Déjà trié par taille
        
        elif mode == "centroid":
            # Calculer le centroïde pondéré par la taille
            total_weight = sum(f['area'] for f in faces)
            if total_weight == 0:
                return faces[0]
            
            cx = sum(f['x'] * f['area'] for f in faces) / total_weight
            cy = sum(f['y'] * f['area'] for f in faces) / total_weight
            avg_conf = sum(f['confidence'] * f['area'] for f in faces) / total_weight
            
            return {
                'x': cx,
                'y': cy,
                'width': faces[0]['width'],
                'height': faces[0]['height'],
                'confidence': avg_conf,
                'area': total_weight / len(faces)
            }
        
        elif mode == "nearest" and prev_position:
            # Le plus proche de la position précédente
            px, py = prev_position
            return min(faces, key=lambda f: (f['x'] - px)**2 + (f['y'] - py)**2)
        
        return faces[0]
    
    def close(self):
        """Libère les ressources"""
        if self._detector:
            self._detector.close()
            self._detector = None
        self._cache.clear()


# =============================================================================
# FOCUS TRACKER PRINCIPAL
# =============================================================================

class FocusTracker:
    """
    Système de tracking caméra intelligent.
    
    Utilise la détection de visage + filtre de Kalman prédictif
    pour un suivi fluide et naturel.
    
    Usage:
        tracker = FocusTracker(preset=TrackingPreset.PODCAST)
        
        # Analyser une vidéo
        focus_points = tracker.analyze_video(video_path, segments)
        
        # Obtenir la position pour une frame
        x, y = tracker.get_focus_at(timestamp, focus_points)
    """
    
    def __init__(
        self,
        preset: TrackingPreset = TrackingPreset.AUTO,
        config: Optional[TrackingConfig] = None
    ):
        """
        Args:
            preset: Preset de tracking à utiliser
            config: Configuration custom (override le preset)
        """
        self.preset = preset
        self.config = config or PRESET_CONFIGS.get(preset, TrackingConfig())
        
        self.face_detector = FaceDetector(min_confidence=self.config.min_confidence)
        self.kalman = PredictiveKalman2D(
            process_noise=0.01 * self.config.reactivity,
            measurement_noise=0.1 / self.config.reactivity,
            initial_x=self.config.default_x,
            initial_y=self.config.default_y
        )
        
        self._last_detection_time = 0.0
        self._frames_without_detection = 0
    
    def analyze_video(
        self,
        video_path: str,
        segments: Optional[List[Tuple[float, float]]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[FocusPoint]:
        """
        Analyse une vidéo et retourne les points de focus.
        
        Args:
            video_path: Chemin vers la vidéo
            segments: Liste de (start, end) à analyser. None = toute la vidéo
            progress_callback: Fonction appelée avec (progress_percent, message)
            
        Returns:
            Liste de FocusPoint pour chaque frame analysée
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            console.print(f"[red]Impossible d'ouvrir la vidéo: {video_path}[/red]")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculer l'intervalle d'échantillonnage
        frame_interval = max(1, int(fps / self.config.sample_rate))
        
        # Si pas de segments, analyser toute la vidéo
        if segments is None:
            segments = [(0, duration)]
        
        focus_points: List[FocusPoint] = []
        frames_analyzed = 0
        total_segment_frames = sum(
            int((end - start) * fps / frame_interval) 
            for start, end in segments
        )
        
        for seg_idx, (start_time, end_time) in enumerate(segments):
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            for frame_idx in range(start_frame, end_frame, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                timestamp = frame_idx / fps
                
                # Détecter les visages
                faces = self.face_detector.detect(frame, frame_idx)
                
                # Sélectionner le visage principal
                primary_face = self.face_detector.select_primary_face(
                    faces,
                    mode=self.config.multi_subject_mode,
                    prev_position=self.kalman.position
                )
                
                if primary_face:
                    # Mise à jour Kalman avec la détection
                    x, y = self.kalman.update(
                        primary_face['x'],
                        primary_face['y'],
                        primary_face['confidence'],
                        timestamp
                    )
                    confidence = primary_face['confidence']
                    face_size = primary_face['area']
                    self._frames_without_detection = 0
                else:
                    # Pas de visage: utiliser la prédiction
                    self._frames_without_detection += 1
                    
                    # Après plusieurs frames sans détection, revenir doucement au centre
                    if self._frames_without_detection > 10:
                        # Blend vers le centre
                        blend = min(0.1, self._frames_without_detection * 0.01)
                        pred_x, pred_y = self.kalman.position
                        target_x = pred_x + (self.config.default_x - pred_x) * blend
                        target_y = pred_y + (self.config.default_y - pred_y) * blend
                        x, y = self.kalman.update(target_x, target_y, 0.3, timestamp)
                    else:
                        # Utiliser la prédiction basée sur la vélocité
                        x, y = self.kalman.get_predicted_position(1/fps * frame_interval)
                    
                    confidence = max(0.1, 0.5 - self._frames_without_detection * 0.05)
                    face_size = 0
                
                focus_points.append(FocusPoint(
                    x=x,
                    y=y,
                    confidence=confidence,
                    timestamp=timestamp,
                    face_size=face_size,
                    num_faces=len(faces)
                ))
                
                frames_analyzed += 1
                
                # Progress callback
                if progress_callback and frames_analyzed % 10 == 0:
                    progress = int(100 * frames_analyzed / max(1, total_segment_frames))
                    progress_callback(progress, f"Analyse frame {frames_analyzed}")
        
        cap.release()
        
        # Post-processing: lisser les points de focus
        focus_points = self._smooth_focus_points(focus_points)
        
        if progress_callback:
            progress_callback(100, f"{len(focus_points)} points analysés")
        
        return focus_points
    
    def _smooth_focus_points(self, points: List[FocusPoint]) -> List[FocusPoint]:
        """
        Lissage final des points de focus.
        
        Applique un filtre simple pour éliminer les dernières saccades
        sans introduire trop de lag.
        """
        if len(points) < 3:
            return points
        
        smoothed = []
        window = 3  # Fenêtre de lissage
        
        for i, fp in enumerate(points):
            # Fenêtre autour du point actuel
            start = max(0, i - window // 2)
            end = min(len(points), i + window // 2 + 1)
            
            # Moyenne pondérée par la confiance
            total_weight = 0
            sum_x = 0
            sum_y = 0
            
            for j in range(start, end):
                w = points[j].confidence
                # Plus de poids au point central
                if j == i:
                    w *= 2
                sum_x += points[j].x * w
                sum_y += points[j].y * w
                total_weight += w
            
            if total_weight > 0:
                smoothed.append(FocusPoint(
                    x=sum_x / total_weight,
                    y=sum_y / total_weight,
                    confidence=fp.confidence,
                    timestamp=fp.timestamp,
                    face_size=fp.face_size,
                    num_faces=fp.num_faces
                ))
            else:
                smoothed.append(fp)
        
        return smoothed
    
    def get_focus_at(
        self,
        timestamp: float,
        focus_points: List[FocusPoint]
    ) -> Tuple[float, float, float]:
        """
        Retourne la position de focus interpolée pour un timestamp donné.
        
        Args:
            timestamp: Timestamp en secondes
            focus_points: Liste des points de focus analysés
            
        Returns:
            (x, y, confidence) - Position normalisée et confiance
        """
        if not focus_points:
            return self.config.default_x, self.config.default_y, 0.1
        
        # Trouver les points encadrants
        prev_fp = None
        next_fp = None
        
        for i, fp in enumerate(focus_points):
            if fp.timestamp <= timestamp:
                prev_fp = fp
                if i + 1 < len(focus_points):
                    next_fp = focus_points[i + 1]
            else:
                if prev_fp is None:
                    prev_fp = fp
                next_fp = fp
                break
        
        if prev_fp is None:
            prev_fp = focus_points[0]
        if next_fp is None:
            next_fp = focus_points[-1]
        
        # Interpolation linéaire entre les deux points
        if prev_fp.timestamp == next_fp.timestamp:
            return prev_fp.x, prev_fp.y, prev_fp.confidence
        
        t = (timestamp - prev_fp.timestamp) / (next_fp.timestamp - prev_fp.timestamp)
        t = np.clip(t, 0, 1)
        
        # Easing pour un mouvement plus naturel
        t = self._ease_in_out(t)
        
        x = prev_fp.x + (next_fp.x - prev_fp.x) * t
        y = prev_fp.y + (next_fp.y - prev_fp.y) * t
        confidence = prev_fp.confidence + (next_fp.confidence - prev_fp.confidence) * t
        
        # Appliquer la dead zone
        center_x, center_y = 0.5, 0.5
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        if dist_from_center < self.config.dead_zone:
            # Dans la dead zone: garder au centre
            blend = dist_from_center / self.config.dead_zone
            x = center_x + (x - center_x) * blend
            y = center_y + (y - center_y) * blend
        
        return x, y, confidence
    
    def _ease_in_out(self, t: float) -> float:
        """Fonction d'easing pour un mouvement naturel"""
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - pow(-2 * t + 2, 2) / 2
    
    def close(self):
        """Libère les ressources"""
        self.face_detector.close()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_preset_for_content(
    has_face: bool,
    face_ratio: float,
    motion_level: float,
    num_speakers: int = 1
) -> TrackingPreset:
    """
    Détermine le preset optimal basé sur l'analyse du contenu.
    
    Args:
        has_face: Visage détecté
        face_ratio: Taille du visage par rapport à l'écran (0-1)
        motion_level: Niveau de mouvement (0-1)
        num_speakers: Nombre de personnes
        
    Returns:
        TrackingPreset recommandé
    """
    if not has_face:
        if motion_level > 0.5:
            return TrackingPreset.ACTION
        else:
            return TrackingPreset.TUTORIAL
    
    if num_speakers >= 2:
        return TrackingPreset.INTERVIEW
    
    if face_ratio > 0.15:  # Gros plan
        if motion_level < 0.2:
            return TrackingPreset.PODCAST
        else:
            return TrackingPreset.VLOG
    
    if face_ratio < 0.05:  # Visage petit (écran gaming, etc.)
        return TrackingPreset.GAMING
    
    if motion_level > 0.4:
        return TrackingPreset.ACTION
    
    return TrackingPreset.VLOG


def analyze_content_type(
    video_path: str,
    sample_duration: float = 5.0
) -> Tuple[TrackingPreset, dict]:
    """
    Analyse le début de la vidéo pour déterminer le type de contenu.
    
    Args:
        video_path: Chemin vers la vidéo
        sample_duration: Durée d'analyse en secondes
        
    Returns:
        (preset, analysis_info)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return TrackingPreset.AUTO, {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = min(int(sample_duration * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    detector = FaceDetector(min_confidence=0.5)
    
    face_counts = []
    face_sizes = []
    motion_scores = []
    prev_gray = None
    
    frame_interval = max(1, int(fps / 3))  # 3 FPS pour l'analyse
    
    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Détection de visage
        faces = detector.detect(frame, frame_idx)
        face_counts.append(len(faces))
        if faces:
            face_sizes.append(max(f['area'] for f in faces))
        
        # Mouvement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)
        
        prev_gray = gray
    
    cap.release()
    detector.close()
    
    # Calculer les stats
    has_face = len(face_sizes) > 0
    avg_face_ratio = np.mean(face_sizes) if face_sizes else 0
    motion_level = np.mean(motion_scores) if motion_scores else 0
    avg_speakers = np.mean(face_counts) if face_counts else 0
    
    analysis = {
        'has_face': has_face,
        'avg_face_ratio': avg_face_ratio,
        'motion_level': motion_level,
        'avg_speakers': avg_speakers
    }
    
    preset = get_preset_for_content(
        has_face=has_face,
        face_ratio=avg_face_ratio,
        motion_level=motion_level,
        num_speakers=int(round(avg_speakers))
    )
    
    return preset, analysis
