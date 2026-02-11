"""Effets visuels pour les clips (zoom, color grading, sharpening, vignette, Ken Burns)."""

import cv2
import numpy as np
import math
from rich.console import Console

console = Console()


def apply_zoom_effect(clip, zoom_factor: float, zoom_style: str, use_lanczos: bool = True):
    """
    Applique un effet de zoom dynamique avec différents styles.

    Styles disponibles:
    - ease_out: Zoom rapide au début, ralentit à la fin (cinématique)
    - ease_in_out: Accélération douce, décélération douce
    - breathing: Micro-oscillations comme une respiration
    - pulse: Pulsations subtiles au rythme

    Utilise LANCZOS4 pour une meilleure qualité de redimensionnement.
    """
    duration = clip.duration

    def ease_out_quad(t: float) -> float:
        """Courbe ease-out quadratique"""
        return 1 - (1 - t) * (1 - t)

    def ease_out_cubic(t: float) -> float:
        """Courbe ease-out cubique (plus prononcée)"""
        return 1 - pow(1 - t, 3)

    def ease_in_out_sine(t: float) -> float:
        """Courbe ease-in-out sinusoïdale (très douce)"""
        return -(math.cos(math.pi * t) - 1) / 2

    def zoom_effect(get_frame, t):
        frame = get_frame(t)
        progress = t / duration

        # Calculer le zoom selon le style
        if zoom_style == "ease_out":
            # Zoom rapide au début, ralentit à la fin
            eased_progress = ease_out_cubic(progress)
            base_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
            # Breathing subtil
            breath = math.sin(t * 0.5 * 2 * math.pi) * 0.002
            current_zoom = base_zoom + breath

        elif zoom_style == "ease_in_out":
            # Accélération et décélération douces
            eased_progress = ease_in_out_sine(progress)
            base_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
            current_zoom = base_zoom

        elif zoom_style == "breathing":
            # Oscillations comme une respiration
            # Zoom de base plus léger
            base_zoom = 1.0 + (zoom_factor - 1.0) * 0.5 * progress
            # Breathing principal (cycle de 3 secondes)
            breath_main = math.sin(t * (2 * math.pi / 3)) * 0.015
            # Harmonique secondaire pour plus de naturel
            breath_secondary = math.sin(t * (2 * math.pi / 1.7)) * 0.005
            current_zoom = base_zoom + breath_main + breath_secondary

        elif zoom_style == "pulse":
            # Pulsations subtiles
            eased_progress = ease_out_cubic(progress)
            base_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
            # Pulse toutes les 0.8 secondes avec decay
            pulse_freq = 1.25
            pulse = math.sin(t * pulse_freq * 2 * math.pi)
            pulse = max(0, pulse)  # Garder seulement les pics positifs
            pulse_decay = math.exp(-t * 0.1)  # Decay progressif
            current_zoom = base_zoom + pulse * 0.008 * pulse_decay

        else:
            # Fallback: zoom linéaire simple
            current_zoom = 1.0 + (zoom_factor - 1.0) * progress

        # Assurer un zoom minimum de 1.0
        current_zoom = max(1.0, current_zoom)

        h, w = frame.shape[:2]
        new_h, new_w = int(h / current_zoom), int(w / current_zoom)

        # Calculer les offsets pour centrer
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2

        # S'assurer que les dimensions sont valides
        y_offset = max(0, y_offset)
        x_offset = max(0, x_offset)
        end_y = min(y_offset + new_h, h)
        end_x = min(x_offset + new_w, w)

        # Recadrer et redimensionner avec haute qualité
        cropped = frame[y_offset:end_y, x_offset:end_x]

        # Choisir l'interpolation: LANCZOS4 (meilleure) ou LINEAR (rapide)
        interpolation = cv2.INTER_LANCZOS4 if use_lanczos else cv2.INTER_LINEAR
        resized = cv2.resize(cropped, (w, h), interpolation=interpolation)

        return resized

    return clip.transform(zoom_effect)


def apply_color_grading(frame: np.ndarray, style: str) -> np.ndarray:
    """
    Applique une correction colorimétrique cinématique OPTIMISÉE.

    Styles:
    - warm: Tons chauds dorés (style lifestyle/vlog)
    - cool: Tons froids bleutés (style tech/corporate)
    - vibrant: Couleurs saturées et contrastées
    - cinematic: Look film avec ombres teintées

    Optimisation: Opérations vectorisées, pas de conversions HSV multiples
    """
    if style == "none":
        return frame

    # Note: Pas de conversion float si style simple (warm/cool)
    if style in ["warm", "cool"]:
        # Opération directe sur uint8 pour vitesse maximale
        if style == "warm":
            # Boost rouge/jaune, réduire bleu
            result = frame.copy()
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            return result
        else:  # cool
            result = frame.copy()
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            return result

    # Pour vibrant et cinematic, on garde la conversion float (nécessaire)
    img = frame.astype(np.float32) / 255.0

    if style == "vibrant":
        # Saturation et contraste - UNE SEULE conversion HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        # Contraste
        img = np.clip((img - 0.5) * 1.15 + 0.5, 0, 1)

    elif style == "cinematic":
        # Look cinéma simplifié - SANS conversion HSV (plus rapide)
        # Teinter directement dans BGR
        shadows = np.clip(img, 0, 0.3) / 0.3
        highlights = np.clip((img - 0.7) / 0.3, 0, 1)

        img[:, :, 0] = img[:, :, 0] + shadows[:, :, 0] * 0.03  # Bleu dans ombres
        img[:, :, 2] = img[:, :, 2] + highlights[:, :, 2] * 0.04  # Rouge dans highlights

        # Contraste simplifié
        img = np.clip((img - 0.5) * 1.08 + 0.5, 0, 1)

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def apply_sharpening(frame: np.ndarray, strength: float) -> np.ndarray:
    """
    Applique un sharpening intelligent qui préserve les détails
    sans amplifier le bruit.
    """
    if strength <= 0:
        return frame

    # Unsharp mask: sharpen = original + strength * (original - blur)
    # Utiliser un blur léger pour préserver les détails
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.5)

    # Calculer le masque de netteté
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)

    return sharpened


def apply_vignette(frame: np.ndarray, strength: float) -> np.ndarray:
    """
    Applique un effet vignette subtil pour focaliser l'attention.
    """
    h, w = frame.shape[:2]

    # Créer le masque de vignette
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    # Distance radiale elliptique (adaptée au format 9:16)
    radius = np.sqrt((X * 0.8) ** 2 + Y ** 2)

    # Vignette douce avec falloff gaussien
    vignette = 1 - np.clip(radius - 0.5, 0, 1) * strength * 2
    vignette = np.clip(vignette, 1 - strength, 1)

    # Appliquer
    result = (frame.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)

    return result


def apply_ken_burns_effect(clip, ken_burns_intensity: float, use_lanczos: bool = True):
    """
    Applique l'effet Ken Burns : mouvement panoramique subtil + zoom lent.

    Crée un effet documentaire/cinématique en combinant:
    - Un zoom progressif très lent (1.0 → 1.0 + intensity)
    - Un léger mouvement panoramique (pan) horizontal ou vertical
    - Des transitions douces avec easing

    L'effet est subtil pour ne pas distraire du contenu principal.
    """
    duration = clip.duration
    intensity = ken_burns_intensity

    # Choisir une direction de pan aléatoire mais cohérente pour le clip
    # On utilise le hash de la durée pour avoir une direction reproductible
    pan_seed = int(duration * 1000) % 4
    pan_directions = [
        (1, 0),    # Droite
        (-1, 0),   # Gauche
        (0, 1),    # Bas
        (0, -1),   # Haut
    ]
    pan_x_dir, pan_y_dir = pan_directions[pan_seed]

    def ease_in_out_cubic(t: float) -> float:
        """Courbe ease-in-out cubique pour transitions douces"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def ken_burns_transform(get_frame, t):
        frame = get_frame(t)
        progress = t / duration

        # Appliquer l'easing pour un mouvement naturel
        eased_progress = ease_in_out_cubic(progress)

        # Zoom progressif très lent (commence à 1.0, finit à 1.0 + intensity)
        # L'intensité est divisée par 2 car le zoom est appliqué en crop
        current_zoom = 1.0 + (intensity * eased_progress)

        # Pan subtil dans la direction choisie
        # Le pan est proportionnel à l'intensité et au progrès
        pan_amount = intensity * 0.3  # Le pan est plus subtil que le zoom
        pan_x = pan_x_dir * pan_amount * eased_progress
        pan_y = pan_y_dir * pan_amount * eased_progress

        h, w = frame.shape[:2]

        # Calculer la région de crop avec zoom et pan
        # new_w et new_h sont les dimensions de la fenêtre de crop
        new_w = int(w / current_zoom)
        new_h = int(h / current_zoom)

        # Position centrale avec décalage du pan
        center_x = w / 2 + (pan_x * w / 2)
        center_y = h / 2 + (pan_y * h / 2)

        # Calculer les coordonnées de crop
        x1 = int(center_x - new_w / 2)
        y1 = int(center_y - new_h / 2)
        x2 = x1 + new_w
        y2 = y1 + new_h

        # S'assurer qu'on reste dans les limites de l'image
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # Clamp final
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop et resize
        cropped = frame[y1:y2, x1:x2]

        # Choisir l'interpolation
        interpolation = cv2.INTER_LANCZOS4 if use_lanczos else cv2.INTER_LINEAR

        # Redimensionner à la taille originale
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            result = cv2.resize(cropped, (w, h), interpolation=interpolation)
        else:
            result = frame

        return result

    return clip.transform(ken_burns_transform)
