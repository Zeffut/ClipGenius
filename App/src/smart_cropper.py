"""
Module de recadrage intelligent
Utilise la détection de visages et de sujets pour centrer le contenu
"""

import os
import sys
import shutil
import tempfile

# Supprimer les warnings MediaPipe/Abseil AVANT tout import
os.environ['GLOG_minloglevel'] = '2'  # Désactive les logs INFO et WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Désactive les warnings TensorFlow

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from rich.console import Console

console = Console()

# Variable globale pour le chemin temporaire MediaPipe
_MP_TEMP_DIR = None
_MP_WORKAROUND_APPLIED = False


def _setup_mediapipe_workaround():
    """
    Contourne le bug MediaPipe avec les chemins contenant des espaces sur Windows.
    Doit être appelé AVANT tout import de mediapipe.
    """
    global _MP_TEMP_DIR, _MP_WORKAROUND_APPLIED
    
    if _MP_WORKAROUND_APPLIED:
        return True
    
    if sys.platform != 'win32':
        _MP_WORKAROUND_APPLIED = True
        return False
    
    # Vérifier si mediapipe est déjà importé
    if 'mediapipe' in sys.modules:
        console.print("[yellow]MediaPipe deja importe, workaround impossible[/yellow]")
        _MP_WORKAROUND_APPLIED = True
        return False
    
    try:
        # Trouver le chemin de mediapipe sans l'importer
        # On utilise importlib pour trouver le spec
        import importlib.util
        spec = importlib.util.find_spec('mediapipe')
        
        if spec is None or spec.origin is None:
            _MP_WORKAROUND_APPLIED = True
            return False
        
        mp_path = Path(spec.origin).parent
        
        # Vérifier si le chemin contient des espaces
        if ' ' not in str(mp_path):
            _MP_WORKAROUND_APPLIED = True
            return False
        
        console.print("[yellow]Chemin MediaPipe avec espaces detecte, application du workaround...[/yellow]")
        
        # Créer un répertoire temporaire sans espaces
        temp_base = Path(tempfile.gettempdir()) / "mp_fix"
        temp_base.mkdir(exist_ok=True)
        
        # Destination pour mediapipe
        mp_dst = temp_base / "mediapipe"
        
        # Lire la version depuis le fichier __init__.py ou setup
        version_file_path = mp_path / "__init__.py"
        current_version = "unknown"
        if version_file_path.exists():
            content = version_file_path.read_text(encoding='utf-8', errors='ignore')
            for line in content.split('\n'):
                if '__version__' in line:
                    try:
                        current_version = line.split('=')[1].strip().strip('"\'')
                    except:
                        pass
                    break
        
        # Vérifier si on doit recopier
        our_version_file = temp_base / "mp_version.txt"
        
        need_copy = True
        if our_version_file.exists() and mp_dst.exists():
            try:
                if our_version_file.read_text().strip() == current_version:
                    need_copy = False
            except:
                pass
        
        if need_copy:
            console.print(f"[dim]Copie de MediaPipe vers {temp_base}...[/dim]")
            if mp_dst.exists():
                shutil.rmtree(mp_dst, ignore_errors=True)
            shutil.copytree(mp_path, mp_dst)
            our_version_file.write_text(current_version)
            console.print("[dim]Copie terminee.[/dim]")
        
        _MP_TEMP_DIR = temp_base
        
        # Ajouter le nouveau chemin en PREMIER dans sys.path
        temp_base_str = str(temp_base)
        if temp_base_str in sys.path:
            sys.path.remove(temp_base_str)
        sys.path.insert(0, temp_base_str)
        
        console.print("[green]Workaround MediaPipe applique![/green]")
        _MP_WORKAROUND_APPLIED = True
        return True
            
    except Exception as e:
        console.print(f"[red]Erreur workaround MediaPipe: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        _MP_WORKAROUND_APPLIED = True
    
    return False


# Lazy import de MediaPipe pour éviter les blocages au démarrage
_mp_instance = None

def _get_mediapipe():
    """Retourne le module MediaPipe (import lazy avec workaround)"""
    global _mp_instance
    if _mp_instance is None:
        # Appliquer le workaround AVANT d'importer mediapipe
        _setup_mediapipe_workaround()

        # Maintenant importer mediapipe (depuis le nouveau chemin si workaround appliqué)
        import mediapipe as mp
        _mp_instance = mp

    return _mp_instance


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

from enum import Enum, auto


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
    
    def __init__(self, sample_frames: int = 10):
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
            fps = 30.0  # Fallback FPS par défaut
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
        motion = np.mean(diff) / 255.0
        
        return motion
    
    def _analyze_edges(self, gray: np.ndarray) -> float:
        """Analyse la densité de contours (indicateur de texte/UI)"""
        # Réduire la résolution pour la vitesse
        small = cv2.resize(gray, (320, 180))
        edges = cv2.Canny(small, 50, 150)
        density = np.mean(edges) / 255.0
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


# ============================================================================
# FIN DU SYSTÈME DE SMART CROP INTELLIGENT
# ============================================================================


def ease_in_out_cubic(t: float) -> float:
    """
    Fonction d'interpolation ease-in-out cubique.
    Produit un mouvement plus naturel avec accélération/décélération.
    
    Args:
        t: Valeur entre 0 et 1
        
    Returns:
        Valeur transformée entre 0 et 1
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_quad(t: float) -> float:
    """
    Fonction d'interpolation ease-out quadratique.
    Mouvement rapide au début, ralentit à la fin.
    """
    return 1 - (1 - t) * (1 - t)


def ease_out_expo(t: float) -> float:
    """
    Fonction d'interpolation ease-out exponentielle.
    Décélération très prononcée, idéale pour les mouvements de caméra.
    """
    return 1 if t == 1 else 1 - pow(2, -10 * t)


def ease_in_out_sine(t: float) -> float:
    """
    Fonction d'interpolation ease-in-out sinusoïdale.
    Mouvement très doux et naturel.
    """
    import math
    return -(math.cos(math.pi * t) - 1) / 2


def smooth_value(current: float, target: float, smoothing: float = 0.15) -> float:
    """
    Lissage exponentiel pour des mouvements fluides.
    
    Args:
        current: Valeur actuelle
        target: Valeur cible
        smoothing: Facteur de lissage (0.1 = très lisse, 0.5 = réactif)
        
    Returns:
        Nouvelle valeur lissée
    """
    return current + (target - current) * smoothing


def critically_damped_spring(
    current: float, 
    target: float, 
    velocity: float, 
    smoothness: float = 0.25,
    delta_time: float = 1/30
) -> Tuple[float, float]:
    """
    Simulation d'un ressort critiquement amorti pour des mouvements ultra-fluides.
    Utilisé par les systèmes de caméra professionnels.
    
    Args:
        current: Position actuelle
        target: Position cible
        velocity: Vélocité actuelle
        smoothness: Temps de lissage en secondes (plus grand = plus lent)
        delta_time: Intervalle de temps
        
    Returns:
        Tuple (nouvelle_position, nouvelle_vélocité)
    """
    omega = 2.0 / smoothness  # Fréquence naturelle
    
    x = current - target
    exp_term = np.exp(-omega * delta_time)
    
    new_pos = target + (x + (velocity + omega * x) * delta_time) * exp_term
    new_vel = (velocity - omega * (velocity + omega * x) * delta_time) * exp_term
    
    return new_pos, new_vel


class KalmanFilter1D:
    """
    Filtre de Kalman 1D pour un lissage prédictif optimal.
    Utilisé pour stabiliser les mouvements de caméra en réduisant le bruit
    tout en conservant la réactivité aux vrais mouvements.
    """
    
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """
        Args:
            process_variance: Variance du processus (bruit du système)
            measurement_variance: Variance de mesure (bruit de détection)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0
        self.velocity = 0.0
        self.initialized = False
    
    def update(self, measurement: float, dt: float = 1/30) -> float:
        """
        Met à jour le filtre avec une nouvelle mesure.
        
        Args:
            measurement: Nouvelle mesure brute
            dt: Intervalle de temps depuis la dernière mesure
            
        Returns:
            Estimation filtrée
        """
        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
            return measurement
        
        # Prédiction
        predicted_estimate = self.estimate + self.velocity * dt
        predicted_error = self.estimate_error + self.process_variance
        
        # Mise à jour
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
        
        # Nouvelle estimation
        new_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        
        # Estimer la vélocité
        self.velocity = (new_estimate - self.estimate) / dt if dt > 0 else 0
        
        self.estimate = new_estimate
        self.estimate_error = (1 - kalman_gain) * predicted_error
        
        return self.estimate
    
    def reset(self):
        """Réinitialise le filtre"""
        self.estimate = 0.0
        self.estimate_error = 1.0
        self.velocity = 0.0
        self.initialized = False


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


class SmartCropper:
    """
    Recadrage intelligent basé sur la détection de visages et de pose
    
    Utilise MediaPipe pour détecter:
    - Visages (priorité haute)
    - Pose du corps (fallback)
    - Centre de mouvement (dernier recours)
    """
    
    def __init__(self, target_aspect_ratio: float = 9/16):
        """
        Args:
            target_aspect_ratio: Ratio largeur/hauteur cible (9:16 pour vertical)
        """
        self.target_aspect_ratio = target_aspect_ratio

        # MediaPipe sera importé à la demande dans _ensure_detectors()
        self.mp_face = None
        self.mp_pose = None

        # Les détecteurs seront initialisés à la demande
        self.face_detector = None
        self.pose_detector = None

    def _ensure_detectors(self):
        """Initialise ou réinitialise les détecteurs MediaPipe si nécessaire."""
        # Import lazy de MediaPipe au premier besoin
        if self.mp_face is None or self.mp_pose is None:
            mp = _get_mediapipe()
            self.mp_face = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose

        if self.face_detector is None:
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=1,  # 1 = meilleure détection à distance
                min_detection_confidence=0.5
            )

        if self.pose_detector is None:
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,  # Mode vidéo = plus rapide
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5  # Requis pour le mode vidéo
            )

    def reset_detectors(self):
        """Réinitialise les détecteurs MediaPipe (à appeler entre les clips)."""
        if self.face_detector is not None:
            try:
                self.face_detector.close()
            except Exception:
                pass
            self.face_detector = None

        if self.pose_detector is not None:
            try:
                self.pose_detector.close()
            except Exception:
                pass
            self.pose_detector = None
        
    def find_focus_point(self, frame: np.ndarray) -> FocusPoint:
        """
        Trouve le point de focus optimal dans une frame

        Args:
            frame: Image BGR (OpenCV)

        Returns:
            FocusPoint avec la position du sujet principal
        """
        # S'assurer que les détecteurs sont initialisés
        self._ensure_detectors()

        h, w = frame.shape[:2]

        # Convertir en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Essayer la détection de visage
        face_focus = self._detect_face(rgb_frame, w, h)
        if face_focus and face_focus.confidence > 0.5:
            return face_focus
        
        # 2. Essayer la détection de pose
        pose_focus = self._detect_pose(rgb_frame, w, h)
        if pose_focus and pose_focus.confidence > 0.4:
            return pose_focus
        
        # 3. Fallback: centre de l'image
        return FocusPoint(x=0.5, y=0.5, confidence=0.1)
    
    def _detect_face(self, rgb_frame: np.ndarray, w: int, h: int) -> Optional[FocusPoint]:
        """Détecte les visages et retourne le centre du visage principal"""
        results = self.face_detector.process(rgb_frame)
        
        if results.detections:
            # Prendre le visage le plus grand (probablement le plus proche/important)
            best_detection = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width *
                              d.location_data.relative_bounding_box.height
            )
            
            bbox = best_detection.location_data.relative_bounding_box
            
            # Centre du visage
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            
            return FocusPoint(
                x=center_x,
                y=center_y,
                confidence=best_detection.score[0],
                width=bbox.width * 2,  # Zone plus large autour du visage
                height=bbox.height * 2
            )
        
        return None
    
    def _detect_pose(self, rgb_frame: np.ndarray, w: int, h: int) -> Optional[FocusPoint]:
        """Détecte la pose et retourne le centre du corps"""
        results = self.pose_detector.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calculer le centre du corps (moyenne des épaules et hanches)
            key_points = [
                landmarks[11],  # Épaule gauche
                landmarks[12],  # Épaule droite
                landmarks[23],  # Hanche gauche
                landmarks[24],  # Hanche droite
            ]
            
            # Filtrer les points visibles
            visible_points = [p for p in key_points if p.visibility > 0.5]
            
            if visible_points:
                center_x = sum(p.x for p in visible_points) / len(visible_points)
                center_y = sum(p.y for p in visible_points) / len(visible_points)
                avg_visibility = sum(p.visibility for p in visible_points) / len(visible_points)
                
                return FocusPoint(
                    x=center_x,
                    y=center_y,
                    confidence=avg_visibility,
                    width=0.5,
                    height=0.8
                )
        
        return None
    
    def calculate_crop_region(
        self,
        focus_point: FocusPoint,
        frame_width: int,
        frame_height: int,
        smooth_center: Optional[Tuple[float, float]] = None,
        smoothing_factor: float = 0.1,
        vertical_bias: float = 0.35
    ) -> Tuple[int, int, int, int]:
        """
        Calcule la région de recadrage optimale (version simple)
        
        Returns:
            Tuple (x1, y1, x2, y2) de la région de recadrage
        """
        result = self.calculate_crop_region_extended(
            focus_point, frame_width, frame_height,
            smooth_center, smoothing_factor, vertical_bias
        )
        return (result.x1, result.y1, result.x2, result.y2)
    
    def calculate_crop_region_extended(
        self,
        focus_point: FocusPoint,
        frame_width: int,
        frame_height: int,
        smooth_center: Optional[Tuple[float, float]] = None,
        smoothing_factor: float = 0.1,
        vertical_bias: float = 0.35,
        enable_blur_fill: bool = True,
        max_face_y_in_crop: float = 0.70  # Position Y max du visage dans le crop avant blur fill
    ) -> CropResult:
        """
        Calcule la région de recadrage optimale avec détection du besoin de fond flouté
        
        Le blur fill n'est activé que si, après contrainte du crop aux limites de l'image,
        le visage serait positionné trop bas dans le crop (au-dela de max_face_y_in_crop).
        
        Args:
            focus_point: Point de focus détecté
            frame_width: Largeur de la frame originale
            frame_height: Hauteur de la frame originale
            smooth_center: Centre précédent pour le lissage
            smoothing_factor: Facteur de lissage (0 = pas de lissage, 1 = instantané)
            vertical_bias: Position verticale idéale du visage dans le crop (0.35 = tiers supérieur)
            enable_blur_fill: Activer la détection du besoin de fond flouté
            max_face_y_in_crop: Position Y max du visage avant d'activer le blur fill (0.55 = milieu)
            
        Returns:
            CropResult avec les coordonnées et info sur le fond flouté
        """
        # Calculer les dimensions du crop pour le ratio cible
        if frame_width / frame_height > self.target_aspect_ratio:
            # Vidéo plus large que haute: crop en largeur
            crop_height = frame_height
            crop_width = int(crop_height * self.target_aspect_ratio)
        else:
            # Vidéo plus haute que large: crop en hauteur
            crop_width = frame_width
            crop_height = int(crop_width / self.target_aspect_ratio)
        
        # Centre idéal basé sur le focus point
        ideal_center_x = focus_point.x * frame_width
        
        # Position Y du visage dans la frame
        focus_y_in_frame = focus_point.y * frame_height
        
        # Position idéale du centre du crop pour placer le visage à vertical_bias
        ideal_center_y = focus_y_in_frame + (0.5 - vertical_bias) * crop_height
        
        # Appliquer le lissage si disponible
        if smooth_center:
            ideal_center_x = smooth_center[0] + smoothing_factor * (ideal_center_x - smooth_center[0])
            ideal_center_y = smooth_center[1] + smoothing_factor * (ideal_center_y - smooth_center[1])
        
        # Calculer les coordonnées de crop idéales
        x1 = int(ideal_center_x - crop_width / 2)
        y1_ideal = int(ideal_center_y - crop_height / 2)
        
        # S'assurer que X reste dans les limites
        x1 = max(0, min(x1, frame_width - crop_width))
        
        # D'abord, essayer un crop normal contraint aux limites
        y1_constrained = max(0, min(y1_ideal, frame_height - crop_height))
        
        # Calculer où serait le visage dans ce crop contraint
        face_y_in_constrained_crop = (focus_y_in_frame - y1_constrained) / crop_height
        
        # Décider si on a besoin du blur fill
        needs_blur_fill = False
        blur_fill_height = 0
        y1 = y1_constrained
        
        if enable_blur_fill and face_y_in_constrained_crop > max_face_y_in_crop:
            # Le visage serait trop bas dans le crop normal
            # On doit utiliser le blur fill pour le remonter
            
            # Utiliser le y1 idéal (non contraint) pour garder le visage bien positionné
            y1 = max(0, y1_ideal)  # Au minimum 0
            y2_with_ideal = y1 + crop_height
            
            if y2_with_ideal > frame_height:
                # Le crop dépasse en bas: on a besoin du blur fill
                blur_fill_height = y2_with_ideal - frame_height
                needs_blur_fill = True
        
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        
        # Calculer la position finale du visage dans le crop
        face_y_in_crop = (focus_y_in_frame - y1) / crop_height if crop_height > 0 else 0.5
        
        return CropResult(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            needs_blur_fill=needs_blur_fill,
            blur_fill_height=blur_fill_height,
            face_y_in_crop=face_y_in_crop
        )
    
    def analyze_video_focus(
        self,
        video_path: str,
        sample_rate: int = 2
    ) -> List[Tuple[float, FocusPoint]]:
        """
        Analyse une vidéo pour déterminer les points de focus
        
        Args:
            video_path: Chemin vers la vidéo
            sample_rate: Nombre de frames par seconde à analyser
            
        Returns:
            Liste de tuples (timestamp, FocusPoint)
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback FPS par défaut
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        focus_points = []
        frame_interval = max(1, int(fps / sample_rate))
        frames_to_analyze = total_frames // frame_interval
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    f"Analyse des points de focus ({duration:.0f}s de video)...", 
                    total=frames_to_analyze
                )
                
                frame_count = 0
                analyzed_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        timestamp = frame_count / fps
                        focus_point = self.find_focus_point(frame)
                        focus_points.append((timestamp, focus_point))
                        analyzed_count += 1
                        progress.update(task, advance=1)
                    
                    frame_count += 1
        finally:
            cap.release()
        console.print(f"[green]Analyse terminee: {len(focus_points)} points de focus[/green]")
        
        return focus_points
    
    def analyze_video_segments(
        self,
        video_path: str,
        segments: List[Tuple[float, float]],
        sample_rate: int = 2,
        margin: float = 1.0
    ) -> List[Tuple[float, FocusPoint]]:
        """
        Analyse uniquement les segments spécifiés d'une vidéo (beaucoup plus rapide)
        
        Args:
            video_path: Chemin vers la vidéo
            segments: Liste de tuples (start_time, end_time) à analyser
            sample_rate: Nombre de frames par seconde à analyser
            margin: Marge en secondes avant/après chaque segment
            
        Returns:
            Liste de tuples (timestamp, FocusPoint)
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
        
        if not segments:
            return []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback FPS par défaut
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        # Fusionner les segments qui se chevauchent et ajouter les marges
        merged_segments = []
        for start, end in sorted(segments):
            start = max(0, start - margin)
            end = min(video_duration, end + margin)
            
            if merged_segments and start <= merged_segments[-1][1]:
                # Fusionner avec le segment précédent
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], end))
            else:
                merged_segments.append((start, end))
        
        # Calculer le nombre total de frames à analyser
        total_segment_duration = sum(end - start for start, end in merged_segments)
        frame_interval = max(1, int(fps / sample_rate))
        frames_to_analyze = int(total_segment_duration * sample_rate)
        
        console.print(f"[dim]Analyse optimisée: {total_segment_duration:.0f}s sur {len(merged_segments)} segment(s)[/dim]")
        
        focus_points = []
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    f"Analyse des points de focus ({total_segment_duration:.0f}s)...", 
                    total=frames_to_analyze
                )
                
                for seg_idx, (seg_start, seg_end) in enumerate(merged_segments):
                    # Réinitialiser les détecteurs entre les segments pour éviter les erreurs MediaPipe
                    if seg_idx > 0:
                        self.reset_detectors()

                    # Positionner au début du segment
                    start_frame = int(seg_start * fps)
                    end_frame = int(seg_end * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    frame_count = start_frame

                    while frame_count < end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if (frame_count - start_frame) % frame_interval == 0:
                            timestamp = frame_count / fps
                            try:
                                focus_point = self.find_focus_point(frame)
                            except Exception as e:
                                # En cas d'erreur MediaPipe, réinitialiser et utiliser le centre
                                self.reset_detectors()
                                focus_point = FocusPoint(x=0.5, y=0.5, confidence=0.1)
                            focus_points.append((timestamp, focus_point))
                            progress.update(task, advance=1)
                        
                        frame_count += 1
        finally:
            cap.release()
        
        # Appliquer le lissage temporel pour des mouvements plus fluides
        if len(focus_points) > 3:
            focus_points = self.smooth_focus_points(focus_points)
            console.print(f"[dim]Lissage temporel appliqué[/dim]")
        
        console.print(f"[green]Analyse terminee: {len(focus_points)} points de focus (segments)[/green]")
        
        
        return focus_points
    
    def get_interpolated_focus(
        self,
        focus_points: List[Tuple[float, FocusPoint]],
        timestamp: float,
        use_easing: bool = True,
        lookahead: bool = True
    ) -> FocusPoint:
        """
        Interpole le focus point pour un timestamp donné avec courbes avancées.
        
        Utilise une interpolation Catmull-Rom spline pour des mouvements
        encore plus fluides et naturels qu'une simple interpolation linéaire.
        
        Args:
            focus_points: Liste de (timestamp, FocusPoint) analysés
            timestamp: Timestamp pour lequel on veut le focus
            use_easing: Utiliser l'interpolation ease-in-out
            lookahead: Utiliser les points suivants pour une meilleure interpolation
            
        Returns:
            FocusPoint interpolé
        """
        if not focus_points:
            return FocusPoint(x=0.5, y=0.5, confidence=0.1)
        
        if len(focus_points) == 1:
            return focus_points[0][1]
        
        # Trouver les points encadrants et les points voisins pour Catmull-Rom
        p0 = p1 = p2 = p3 = None
        idx = 0
        
        for i, (t, fp) in enumerate(focus_points):
            if t >= timestamp:
                idx = i
                break
            idx = i
        
        # Points pour l'interpolation
        p1_idx = max(0, idx - 1) if idx > 0 else 0
        p2_idx = min(idx, len(focus_points) - 1)
        
        if p1_idx == p2_idx:
            return focus_points[p1_idx][1]
        
        p1 = focus_points[p1_idx]
        p2 = focus_points[p2_idx]
        
        # Points supplémentaires pour Catmull-Rom (si lookahead activé)
        if lookahead and len(focus_points) >= 4:
            p0_idx = max(0, p1_idx - 1)
            p3_idx = min(p2_idx + 1, len(focus_points) - 1)
            p0 = focus_points[p0_idx]
            p3 = focus_points[p3_idx]
        
        # Calculer le facteur d'interpolation
        t1, fp1 = p1
        t2, fp2 = p2
        
        if t2 == t1:
            alpha = 0
        else:
            alpha = (timestamp - t1) / (t2 - t1)
            alpha = max(0, min(1, alpha))
            
            # Appliquer la courbe ease-in-out
            if use_easing:
                alpha = ease_in_out_sine(alpha)  # Utiliser sine pour plus de douceur
        
        # Interpolation Catmull-Rom si on a 4 points
        if p0 and p3 and lookahead:
            _, fp0 = p0
            _, fp3 = p3
            
            # Catmull-Rom spline
            t = alpha
            t2_val = t * t
            t3_val = t2_val * t
            
            # Coefficients Catmull-Rom
            c0 = -0.5 * t3_val + t2_val - 0.5 * t
            c1 = 1.5 * t3_val - 2.5 * t2_val + 1.0
            c2 = -1.5 * t3_val + 2.0 * t2_val + 0.5 * t
            c3 = 0.5 * t3_val - 0.5 * t2_val
            
            interp_x = c0 * fp0.x + c1 * fp1.x + c2 * fp2.x + c3 * fp3.x
            interp_y = c0 * fp0.y + c1 * fp1.y + c2 * fp2.y + c3 * fp3.y
            
            # Limiter aux bornes raisonnables
            interp_x = max(0.1, min(0.9, interp_x))
            interp_y = max(0.1, min(0.9, interp_y))
            
            return FocusPoint(
                x=interp_x,
                y=interp_y,
                confidence=fp1.confidence + alpha * (fp2.confidence - fp1.confidence)
            )
        
        # Fallback: interpolation linéaire avec easing
        return FocusPoint(
            x=fp1.x + alpha * (fp2.x - fp1.x),
            y=fp1.y + alpha * (fp2.y - fp1.y),
            confidence=fp1.confidence + alpha * (fp2.confidence - fp1.confidence)
        )
    
    def smooth_focus_points(
        self,
        focus_points: List[Tuple[float, FocusPoint]],
        window_size: int = 7,
        outlier_threshold: float = 0.12,
        use_kalman: bool = True
    ) -> List[Tuple[float, FocusPoint]]:
        """
        Applique un lissage temporel avancé sur les points de focus.
        
        Combine plusieurs techniques pour des mouvements ultra-fluides:
        1. Filtre de Kalman pour réduire le bruit de détection
        2. Moyenne mobile pondérée gaussienne
        3. Détection et correction des outliers
        4. Lissage des transitions brusques
        
        Args:
            focus_points: Liste de (timestamp, FocusPoint) bruts
            window_size: Taille de la fenêtre de lissage (impair recommandé)
            outlier_threshold: Seuil de distance pour détecter un outlier (0-1)
            use_kalman: Utiliser le filtre de Kalman pour le pré-lissage
            
        Returns:
            Liste lissée de (timestamp, FocusPoint)
        """
        if len(focus_points) < 3:
            return focus_points
        
        n = len(focus_points)
        half_window = window_size // 2
        
        # Étape 1: Pré-lissage avec filtre de Kalman
        if use_kalman:
            kalman_x = KalmanFilter1D(process_variance=0.005, measurement_variance=0.05)
            kalman_y = KalmanFilter1D(process_variance=0.005, measurement_variance=0.05)
            
            kalman_filtered = []
            prev_t = focus_points[0][0]
            
            for t, fp in focus_points:
                dt = max(0.001, t - prev_t)
                filtered_x = kalman_x.update(fp.x, dt)
                filtered_y = kalman_y.update(fp.y, dt)
                
                kalman_filtered.append((t, FocusPoint(
                    x=filtered_x,
                    y=filtered_y,
                    confidence=fp.confidence,
                    width=fp.width,
                    height=fp.height
                )))
                prev_t = t
            
            focus_points = kalman_filtered
        
        # Étape 2: Détection des outliers et moyenne mobile pondérée
        smoothed = []
        
        for i in range(n):
            t, fp = focus_points[i]
            
            # Définir la fenêtre
            start_idx = max(0, i - half_window)
            end_idx = min(n, i + half_window + 1)
            window = focus_points[start_idx:end_idx]
            
            # Calculer les poids gaussiens
            weights = []
            for j, (wt, wfp) in enumerate(window):
                # Distance temporelle relative
                dist = abs(j - (i - start_idx))
                # Poids gaussien avec sigma adaptatif
                sigma = half_window / 2.0 + 0.5
                weight = np.exp(-0.5 * (dist / sigma) ** 2)
                # Pondérer aussi par la confiance
                weight *= (wfp.confidence ** 1.5)  # Exposant pour accentuer les bonnes détections
                weights.append(weight)
            
            total_weight = sum(weights) if sum(weights) > 0 else 1
            
            # Moyenne pondérée
            avg_x = sum(w * wfp.x for w, (_, wfp) in zip(weights, window)) / total_weight
            avg_y = sum(w * wfp.y for w, (_, wfp) in zip(weights, window)) / total_weight
            avg_conf = sum(w * wfp.confidence for w, (_, wfp) in zip(weights, window)) / total_weight
            
            # Détection d'outlier améliorée
            distance = np.sqrt((fp.x - avg_x) ** 2 + (fp.y - avg_y) ** 2)
            
            # Calculer l'écart-type local pour un seuil adaptatif
            local_std = np.sqrt(
                sum(w * ((wfp.x - avg_x) ** 2 + (wfp.y - avg_y) ** 2) 
                    for w, (_, wfp) in zip(weights, window)) / total_weight
            )
            adaptive_threshold = max(outlier_threshold, local_std * 2)
            
            if distance > adaptive_threshold and fp.confidence < 0.7:
                # Outlier détecté - utiliser la moyenne
                smoothed.append((t, FocusPoint(
                    x=avg_x,
                    y=avg_y,
                    confidence=avg_conf * 0.8,  # Réduire la confiance
                    width=fp.width,
                    height=fp.height
                )))
            else:
                # Lissage adaptatif basé sur la confiance
                # Haute confiance = plus proche de l'original
                blend = 0.5 + fp.confidence * 0.4  # 0.5 à 0.9
                smoothed.append((t, FocusPoint(
                    x=fp.x * blend + avg_x * (1 - blend),
                    y=fp.y * blend + avg_y * (1 - blend),
                    confidence=fp.confidence,
                    width=fp.width,
                    height=fp.height
                )))
        
        # Étape 3: Lissage final des transitions brusques
        final_smoothed = [smoothed[0]]
        max_jump = 0.08  # Maximum 8% de saut entre frames
        
        for i in range(1, len(smoothed)):
            t, fp = smoothed[i]
            _, prev_fp = final_smoothed[-1]
            
            dx = fp.x - prev_fp.x
            dy = fp.y - prev_fp.y
            jump = np.sqrt(dx * dx + dy * dy)
            
            if jump > max_jump:
                # Limiter le saut
                scale = max_jump / jump
                new_x = prev_fp.x + dx * scale
                new_y = prev_fp.y + dy * scale
                final_smoothed.append((t, FocusPoint(
                    x=new_x,
                    y=new_y,
                    confidence=fp.confidence,
                    width=fp.width,
                    height=fp.height
                )))
            else:
                final_smoothed.append((t, fp))
        
        return final_smoothed
    
    def close(self):
        """Libère les ressources"""
        self.reset_detectors()


def create_smart_cropper(aspect_ratio: float = 9/16) -> SmartCropper:
    """Crée un SmartCropper configuré"""
    return SmartCropper(target_aspect_ratio=aspect_ratio)


def create_blur_filled_frame(
    frame: np.ndarray,
    crop_result: CropResult,
    target_width: int,
    target_height: int,
    blur_strength: int = 51,
    gradient_height: int = 60,
    darken_factor: float = 0.45,
    add_vignette: bool = True,
    add_grain: bool = False,
    saturation_boost: float = 1.1,
    use_lanczos: bool = True  # Nouvelle option pour qualité maximale
) -> np.ndarray:
    """
    Crée une frame avec le sujet en haut et un fond flouté cinématique en bas.
    
    Style TikTok/Reels premium: quand le visage est trop bas dans la source,
    on remonte le visage et on remplit le bas avec une version floutée et stylisée.
    
    Améliorations visuelles:
    - Dégradé de transition multi-couches pour une fusion naturelle
    - Vignette subtile sur le fond pour attirer l'attention sur le sujet
    - Boost de saturation léger pour des couleurs plus vivantes
    - Grain optionnel pour un look cinématique
    - LANCZOS4 pour un redimensionnement haute qualité (anti-aliasing supérieur)
    
    Args:
        frame: Frame source BGR
        crop_result: Résultat du calcul de crop
        target_width: Largeur cible de sortie
        target_height: Hauteur cible de sortie
        blur_strength: Force du flou (doit être impair)
        gradient_height: Hauteur de la zone de transition (dégradé)
        darken_factor: Facteur d'assombrissement du fond (0.45 = 55% luminosité)
        add_vignette: Ajouter un effet vignette sur le fond
        add_grain: Ajouter un grain cinématique subtil
        saturation_boost: Facteur de boost de saturation (1.0 = pas de changement)
        use_lanczos: Utiliser LANCZOS4 pour redimensionnement haute qualité
        
    Returns:
        Frame composite avec le sujet en haut, transition et blur en bas
    """
    h, w = frame.shape[:2]
    
    # Choisir l'interpolation: LANCZOS4 (meilleure qualité) ou LINEAR (plus rapide)
    interpolation = cv2.INTER_LANCZOS4 if use_lanczos else cv2.INTER_LINEAR
    
    # Calculer les dimensions du crop
    crop_width = crop_result.x2 - crop_result.x1
    crop_height = crop_result.y2 - crop_result.y1
    
    # Hauteur disponible dans la frame source
    available_height = min(crop_result.y2, h) - crop_result.y1
    
    if available_height <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Extraire la partie valide du crop
    x1, y1 = crop_result.x1, crop_result.y1
    x2 = crop_result.x2
    y2 = min(crop_result.y2, h)
    
    # S'assurer que les coordonnées sont valides
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, w)
    
    cropped_content = frame[y1:y2, x1:x2]
    
    if cropped_content.size == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculer le ratio de la partie valide
    valid_ratio = available_height / crop_height
    
    # Hauteur du contenu principal dans l'image finale
    content_height = int(target_height * valid_ratio)
    blur_height = target_height - content_height
    
    if content_height <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Redimensionner le contenu principal avec interpolation haute qualité
    content_resized = cv2.resize(cropped_content, (target_width, content_height), interpolation=interpolation)
    
    # Appliquer un boost de saturation subtil au contenu principal
    if saturation_boost != 1.0:
        content_hsv = cv2.cvtColor(content_resized, cv2.COLOR_BGR2HSV).astype(np.float32)
        content_hsv[:, :, 1] = np.clip(content_hsv[:, :, 1] * saturation_boost, 0, 255)
        content_resized = cv2.cvtColor(content_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Créer le fond flouté à partir de la partie basse de la frame
    blur_source_height = min(h // 3, 300)
    blur_source_y = max(0, h - blur_source_height)
    blur_source = frame[blur_source_y:h, x1:x2]
    
    if blur_source.size == 0:
        blur_source = cropped_content[-min(100, available_height):, :]
    
    # Redimensionner et flouter
    if blur_source.size > 0 and blur_height > 0:
        blur_resized = cv2.resize(blur_source, (target_width, blur_height), interpolation=interpolation)
        
        # Flou gaussien - UNE SEULE passe au lieu de 3 (3x plus rapide)
        # La qualité reste excellente avec un seul blur bien paramétré
        blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred = cv2.GaussianBlur(blur_resized, (blur_strength, blur_strength), 0)
        
        # Assombrissement optimisé - vectorisé au lieu de boucle row-by-row (10x plus rapide)
        # Créer un gradient vertical une seule fois
        gradient = np.linspace(darken_factor, darken_factor * 0.65, blur_height)
        gradient = 1 - np.power(1 - (gradient - darken_factor) / (darken_factor * 0.35), 2)  # ease-out
        gradient = darken_factor - (darken_factor * 0.35 * gradient)
        # Appliquer le gradient à toute l'image en une fois
        blurred = (blurred.astype(np.float32) * gradient[:, np.newaxis, np.newaxis]).clip(0, 255).astype(np.uint8)
        
        # Ajouter l'effet vignette sur le fond
        if add_vignette:
            vignette = create_vignette_mask(target_width, blur_height, strength=0.4)
            blurred = (blurred.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        # Ajouter du grain cinématique optionnel
        if add_grain:
            grain = np.random.normal(0, 3, blurred.shape).astype(np.float32)
            blurred = np.clip(blurred.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    else:
        blurred = np.zeros((max(1, blur_height), target_width, 3), dtype=np.uint8)
    
    # Créer une zone de transition améliorée (dégradé multi-couches)
    if blur_height > 0 and gradient_height > 0:
        actual_gradient_height = min(gradient_height, content_height // 3, blur_height)
        
        if actual_gradient_height > 4:
            # Extraire les zones de transition
            content_bottom = content_resized[-actual_gradient_height:, :].copy()
            blur_top = blurred[:actual_gradient_height, :].copy()
            
            # Créer un dégradé avec courbe ease-in-out pour une transition plus douce
            for row in range(actual_gradient_height):
                t = row / actual_gradient_height
                # Utiliser ease-in-out sine pour une transition ultra-douce
                alpha = 1 - ease_in_out_sine(t)
                
                # Mélanger les pixels avec pondération
                content_bottom[row] = (
                    content_bottom[row].astype(np.float32) * alpha +
                    blur_top[row].astype(np.float32) * (1 - alpha)
                ).clip(0, 255).astype(np.uint8)
            
            # Remplacer la zone de transition dans le contenu
            content_resized[-actual_gradient_height:, :] = content_bottom
    
    # Assembler: contenu en haut, blur en bas
    if blur_height > 0:
        result = np.vstack([content_resized, blurred])
    else:
        result = content_resized
    
    # S'assurer de la bonne taille finale
    if result.shape[0] != target_height or result.shape[1] != target_width:
        result = cv2.resize(result, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    return result


def create_vignette_mask(width: int, height: int, strength: float = 0.5) -> np.ndarray:
    """
    Crée un masque de vignette pour assombrir les bords.
    
    Args:
        width: Largeur de l'image
        height: Hauteur de l'image
        strength: Force de la vignette (0-1)
        
    Returns:
        Masque 2D de valeurs entre (1-strength) et 1
    """
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance radiale depuis le centre
    radius = np.sqrt(X**2 + Y**2)
    
    # Normaliser et inverser (centre = 1, bords = 1-strength)
    vignette = 1 - (radius / radius.max()) * strength
    
    return vignette.astype(np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cropper = SmartCropper()
        focus_points = cropper.analyze_video_focus(sys.argv[1])
        for t, fp in focus_points[:10]:
            print(f"t={t:.1f}s: x={fp.x:.2f}, y={fp.y:.2f}, conf={fp.confidence:.2f}")
        cropper.close()
