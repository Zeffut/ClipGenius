"""
Module de génération de thumbnails pour ClipGenius

Génère des thumbnails optimisés pour les différentes plateformes:
- Extraction de frames clés (visages, action)
- Application des effets visuels (color grading, etc.)
- Redimensionnement aux formats spécifiques
- Ajout optionnel de texte/overlay

Usage:
    from src.thumbnail_generator import ThumbnailGenerator, generate_thumbnail

    # Générer un thumbnail pour un clip
    thumb_path = generate_thumbnail("clip.mp4", "output/thumbnail.jpg")

    # Avec options avancées
    generator = ThumbnailGenerator(ThumbnailConfig(style="vibrant"))
    thumb_path = generator.generate("clip.mp4", "output/")
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from rich.console import Console

console = Console()

# Constantes de configuration
VERTICAL_WIDTH = 1080           # Largeur standard 9:16 (TikTok, Reels, Shorts)
VERTICAL_HEIGHT = 1920          # Hauteur standard 9:16
YOUTUBE_WIDTH = 1280            # Largeur standard 16:9
YOUTUBE_HEIGHT = 720            # Hauteur standard 16:9
SQUARE_SIZE = 1080              # Taille standard 1:1
PREVIEW_WIDTH = 540             # Largeur aperçu 9:16 demi-résolution

# Seuils d'ecart-type de niveaux de gris pour le scoring de contraste
_CONTRAST_HIGH_THRESHOLD: float = 50.0
_CONTRAST_MEDIUM_THRESHOLD: float = 30.0
PREVIEW_HEIGHT = 960            # Hauteur aperçu 9:16 demi-résolution


class ThumbnailSize(Enum):
    """Tailles de thumbnails pour différentes plateformes"""
    TIKTOK = (VERTICAL_WIDTH, VERTICAL_HEIGHT)       # 9:16
    REELS = (VERTICAL_WIDTH, VERTICAL_HEIGHT)        # 9:16
    SHORTS = (VERTICAL_WIDTH, VERTICAL_HEIGHT)       # 9:16
    YOUTUBE = (YOUTUBE_WIDTH, YOUTUBE_HEIGHT)         # 16:9
    SQUARE = (SQUARE_SIZE, SQUARE_SIZE)               # 1:1
    PREVIEW = (PREVIEW_WIDTH, PREVIEW_HEIGHT)         # 9:16 demi-résolution


@dataclass
class ThumbnailConfig:
    """Configuration pour la génération de thumbnails"""
    # Format de sortie
    size: ThumbnailSize = ThumbnailSize.TIKTOK
    quality: int = 95              # Qualité JPEG (1-100)
    format: str = "jpg"            # jpg, png, webp

    # Extraction de frame
    extract_method: str = "smart"  # "smart", "middle", "faces", "action"
    num_candidates: int = 5        # Nombre de frames candidates à analyser

    # Effets visuels
    apply_color_grading: bool = True
    color_style: str = "vibrant"   # warm, cool, vibrant, cinematic
    apply_sharpening: bool = True
    sharpening_strength: float = 0.4
    apply_vignette: bool = True
    vignette_strength: float = 0.12
    brightness_boost: float = 1.05  # Légère augmentation de luminosité

    # Overlay texte (optionnel)
    add_text_overlay: bool = False
    overlay_text: Optional[str] = None
    text_position: str = "bottom"   # top, middle, bottom
    text_font_scale: float = 1.5
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_shadow: bool = True


class ThumbnailGenerator:
    """
    Générateur de thumbnails optimisés pour les réseaux sociaux.

    Utilise la détection de visages et l'analyse de contenu pour
    sélectionner la meilleure frame du clip.
    """

    def __init__(self, config: Optional[ThumbnailConfig] = None):
        self.config = config or ThumbnailConfig()
        self.face_detector = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
            except Exception:
                pass

    def generate(
        self,
        video_path: str,
        output_dir: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Génère un thumbnail optimisé pour une vidéo.

        Args:
            video_path: Chemin vers la vidéo
            output_dir: Dossier de sortie
            filename: Nom du fichier (auto-généré si None)

        Returns:
            Chemin du thumbnail généré ou None si erreur
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            console.print(f"[red]Vidéo introuvable: {video_path}[/red]")
            return None

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            console.print(f"[red]Impossible d'ouvrir: {video_path}[/red]")
            return None

        try:
            # Extraire les informations vidéo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Sélectionner la meilleure frame
            best_frame = self._select_best_frame(cap, total_frames, duration)

            if best_frame is None:
                console.print("[yellow]Aucune frame valide trouvée[/yellow]")
                return None

            # Appliquer les effets visuels
            processed_frame = self._apply_effects(best_frame)

            # Redimensionner au format cible
            target_size = self.config.size.value
            final_frame = self._resize_and_crop(processed_frame, target_size)

            # Ajouter l'overlay texte si configuré
            if self.config.add_text_overlay and self.config.overlay_text:
                final_frame = self._add_text_overlay(final_frame)

            # Générer le nom de fichier
            if filename is None:
                filename = f"{video_path.stem}_thumb.{self.config.format}"

            output_path = output_dir / filename

            # Sauvegarder
            if self.config.format.lower() == "jpg":
                cv2.imwrite(
                    str(output_path),
                    final_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
                )
            elif self.config.format.lower() == "png":
                cv2.imwrite(
                    str(output_path),
                    final_frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9 - self.config.quality // 12]
                )
            elif self.config.format.lower() == "webp":
                cv2.imwrite(
                    str(output_path),
                    final_frame,
                    [cv2.IMWRITE_WEBP_QUALITY, self.config.quality]
                )
            else:
                cv2.imwrite(str(output_path), final_frame)

            return str(output_path)

        finally:
            cap.release()

    def generate_multiple(
        self,
        video_path: str,
        output_dir: str,
        count: int = 3
    ) -> List[str]:
        """
        Génère plusieurs thumbnails candidats pour une vidéo.

        Args:
            video_path: Chemin vers la vidéo
            output_dir: Dossier de sortie
            count: Nombre de thumbnails à générer

        Returns:
            Liste des chemins des thumbnails générés
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Obtenir plusieurs frames candidates
            candidates = self._get_candidate_frames(cap, total_frames, duration, count * 2)

            # Scorer et trier les candidates
            scored_frames = []
            for frame, timestamp in candidates:
                score = self._score_frame(frame)
                scored_frames.append((frame, timestamp, score))

            scored_frames.sort(key=lambda x: x[2], reverse=True)

            # Générer les thumbnails pour les meilleures frames
            results = []
            for i, (frame, timestamp, score) in enumerate(scored_frames[:count]):
                processed = self._apply_effects(frame)
                final = self._resize_and_crop(processed, self.config.size.value)

                filename = f"{video_path.stem}_thumb_{i+1:02d}.{self.config.format}"
                output_path = output_dir / filename

                cv2.imwrite(
                    str(output_path),
                    final,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
                )
                results.append(str(output_path))

            return results

        finally:
            cap.release()

    def _select_best_frame(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        duration: float
    ) -> Optional[np.ndarray]:
        """Sélectionne la meilleure frame selon la méthode configurée."""

        if self.config.extract_method == "middle":
            # Frame du milieu
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            return frame if ret else None

        elif self.config.extract_method == "smart" or self.config.extract_method == "faces":
            # Analyse intelligente avec détection de visages
            candidates = self._get_candidate_frames(
                cap, total_frames, duration, self.config.num_candidates
            )

            best_frame = None
            best_score = -1

            for frame, timestamp in candidates:
                score = self._score_frame(frame)
                if score > best_score:
                    best_score = score
                    best_frame = frame

            return best_frame

        elif self.config.extract_method == "action":
            # Chercher une frame avec du mouvement/action
            candidates = self._get_candidate_frames(
                cap, total_frames, duration, self.config.num_candidates * 2
            )

            best_frame = None
            best_variance = -1

            for frame, timestamp in candidates:
                # Calculer la variance (indicateur d'activité visuelle)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()

                if variance > best_variance:
                    best_variance = variance
                    best_frame = frame

            return best_frame

        else:
            # Fallback: milieu
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            return frame if ret else None

    def _get_candidate_frames(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        duration: float,
        count: int
    ) -> List[Tuple[np.ndarray, float]]:
        """Extrait des frames candidates réparties dans la vidéo."""
        candidates = []

        # Éviter le tout début et la toute fin
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)

        if end_frame <= start_frame:
            start_frame = 0
            end_frame = total_frames - 1

        step = max(1, (end_frame - start_frame) // count)

        for i in range(count):
            frame_idx = start_frame + i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                timestamp = frame_idx / cap.get(cv2.CAP_PROP_FPS)
                candidates.append((frame, timestamp))

        return candidates

    def _score_frame(self, frame: np.ndarray) -> float:
        """
        Score une frame selon plusieurs critères:
        - Présence de visages
        - Luminosité
        - Netteté
        - Composition
        """
        score = 0.0

        # 1. Détection de visages (+40 points par visage)
        if self.face_detector and MEDIAPIPE_AVAILABLE:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb)
                if results.detections:
                    score += len(results.detections) * 40

                    # Bonus si visage bien centré
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        center_x = bbox.xmin + bbox.width / 2
                        center_y = bbox.ymin + bbox.height / 2

                        # Bonus pour visage proche du centre (règle des tiers)
                        dist_from_center = abs(center_x - 0.5) + abs(center_y - 0.33)
                        score += max(0, 20 - dist_from_center * 40)
            except Exception:
                pass

        # 2. Luminosité (ni trop sombre ni trop clair)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Idéal entre 80 et 180
        if 80 <= brightness <= 180:
            score += 20
        elif 60 <= brightness <= 200:
            score += 10

        # 3. Netteté (variance du Laplacien)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var > 500:
            score += 20
        elif laplacian_var > 200:
            score += 10
        elif laplacian_var > 100:
            score += 5

        # 4. Contraste
        contrast = np.std(gray)
        if contrast > _CONTRAST_HIGH_THRESHOLD:
            score += 15
        elif contrast > _CONTRAST_MEDIUM_THRESHOLD:
            score += 8

        return score

    def _apply_effects(self, frame: np.ndarray) -> np.ndarray:
        """Applique les effets visuels configurés."""
        result = frame.copy()

        # 1. Brightness boost
        if self.config.brightness_boost != 1.0:
            result = cv2.convertScaleAbs(result, alpha=self.config.brightness_boost, beta=5)

        # 2. Color grading
        if self.config.apply_color_grading:
            result = self._apply_color_grading(result)

        # 3. Sharpening
        if self.config.apply_sharpening:
            result = self._apply_sharpening(result)

        # 4. Vignette
        if self.config.apply_vignette:
            result = self._apply_vignette(result)

        return result

    def _apply_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """Applique le color grading selon le style configuré."""
        style = self.config.color_style
        img = frame.astype(np.float32) / 255.0

        if style == "warm":
            img[:, :, 2] = np.clip(img[:, :, 2] * 1.1, 0, 1)  # Rouge
            img[:, :, 1] = np.clip(img[:, :, 1] * 1.03, 0, 1)  # Vert
            img[:, :, 0] = np.clip(img[:, :, 0] * 0.92, 0, 1)  # Bleu
            img = np.clip((img - 0.5) * 1.1 + 0.5, 0, 1)

        elif style == "cool":
            img[:, :, 0] = np.clip(img[:, :, 0] * 1.1, 0, 1)  # Bleu
            img[:, :, 1] = np.clip(img[:, :, 1] * 1.03, 0, 1)  # Vert
            img[:, :, 2] = np.clip(img[:, :, 2] * 0.92, 0, 1)  # Rouge

        elif style == "vibrant":
            hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)  # Saturation +35%
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 0, 255)  # Valeur +8%
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
            img = np.clip((img - 0.5) * 1.15 + 0.5, 0, 1)

        elif style == "cinematic":
            shadows = np.clip(img, 0, 0.3) / 0.3
            highlights = np.clip((img - 0.7) / 0.3, 0, 1)
            img[:, :, 0] = img[:, :, 0] + shadows[:, :, 0] * 0.04
            img[:, :, 2] = img[:, :, 2] + highlights[:, :, 2] * 0.05
            img = np.clip((img - 0.5) * 1.1 + 0.5, 0, 1)

        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Applique un sharpening pour une image plus nette."""
        strength = self.config.sharpening_strength
        blurred = cv2.GaussianBlur(frame, (0, 0), 2)
        sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Applique un effet vignette."""
        h, w = frame.shape[:2]
        strength = self.config.vignette_strength

        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)

        # Distance radiale adaptée au ratio
        aspect = w / h
        radius = np.sqrt((X * aspect * 0.5) ** 2 + Y ** 2)

        vignette = 1 - np.clip(radius - 0.6, 0, 1) * strength * 2.5
        vignette = np.clip(vignette, 1 - strength * 1.5, 1)

        result = (frame.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        return result

    def _resize_and_crop(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Redimensionne et recadre l'image pour le format cible."""
        target_w, target_h = target_size
        h, w = frame.shape[:2]

        target_ratio = target_w / target_h
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Image plus large : crop horizontal
            new_w = int(h * target_ratio)
            x_start = (w - new_w) // 2
            cropped = frame[:, x_start:x_start + new_w]
        else:
            # Image plus haute : crop vertical
            new_h = int(w / target_ratio)
            y_start = (h - new_h) // 2
            cropped = frame[y_start:y_start + new_h, :]

        # Redimensionner
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return resized

    def _add_text_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Ajoute un overlay texte sur l'image."""
        if not self.config.overlay_text:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        text = self.config.overlay_text
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = self.config.text_font_scale
        thickness = 2

        # Calculer la taille du texte
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        # Position selon configuration
        x = (w - text_w) // 2
        if self.config.text_position == "top":
            y = text_h + 40
        elif self.config.text_position == "middle":
            y = (h + text_h) // 2
        else:  # bottom
            y = h - 40

        # Ombre
        if self.config.text_shadow:
            cv2.putText(result, text, (x + 3, y + 3), font, scale, (0, 0, 0), thickness + 2)

        # Texte
        cv2.putText(result, text, (x, y), font, scale, self.config.text_color, thickness)

        return result

    def close(self):
        """Libère les ressources."""
        if self.face_detector:
            self.face_detector.close()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def generate_thumbnail(
    video_path: str,
    output_path: Optional[str] = None,
    style: str = "vibrant"
) -> Optional[str]:
    """
    Génère rapidement un thumbnail pour une vidéo.

    Args:
        video_path: Chemin vers la vidéo
        output_path: Chemin de sortie (auto si None)
        style: Style de color grading (warm, cool, vibrant, cinematic)

    Returns:
        Chemin du thumbnail généré ou None
    """
    config = ThumbnailConfig(
        color_style=style,
        extract_method="smart"
    )

    generator = ThumbnailGenerator(config)

    try:
        if output_path:
            output_dir = str(Path(output_path).parent)
            filename = Path(output_path).name
        else:
            output_dir = str(Path(video_path).parent)
            filename = None

        return generator.generate(video_path, output_dir, filename)

    finally:
        generator.close()


def generate_thumbnails_for_clips(
    clips: List[str],
    output_dir: str,
    style: str = "vibrant"
) -> Dict[str, str]:
    """
    Génère des thumbnails pour une liste de clips.

    Args:
        clips: Liste des chemins vers les clips
        output_dir: Dossier de sortie
        style: Style de color grading

    Returns:
        Dict {clip_path: thumbnail_path}
    """
    config = ThumbnailConfig(color_style=style)
    generator = ThumbnailGenerator(config)

    results = {}

    try:
        for clip_path in clips:
            thumb_path = generator.generate(clip_path, output_dir)
            if thumb_path:
                results[clip_path] = thumb_path
                console.print(f"  [green]✓[/green] {Path(clip_path).name} → {Path(thumb_path).name}")
            else:
                console.print(f"  [yellow]✗[/yellow] {Path(clip_path).name}")

        return results

    finally:
        generator.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None

        result = generate_thumbnail(video, output)

        if result:
            print(f"Thumbnail généré: {result}")
        else:
            print("Erreur lors de la génération")
            sys.exit(1)
    else:
        print("Usage: python -m src.thumbnail_generator <video> [output]")
