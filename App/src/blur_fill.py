"""Creation de frames avec fond floute cinematique."""

import cv2
import numpy as np
from typing import Optional, Tuple

from .smoothing import ease_in_out_sine
from .content_detector import CropResult


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


def create_letterbox_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    blur_strength: int = 61,
    darken_factor: float = 0.40,
) -> np.ndarray:
    """Affiche la frame source en letterbox centré avec fond flouté haut/bas.

    Utilisé quand aucun visage n'est détecté : on montre la totalité de la
    largeur source (scalée à target_width) centrée verticalement, et on
    remplit les barres haut/bas avec une version fortement floutée et
    assombrie de la même frame.

    Args:
        frame:          Frame source BGR
        target_width:   Largeur cible (ex: 1080)
        target_height:  Hauteur cible (ex: 1920)
        blur_strength:  Taille du kernel gaussien (doit être impair)
        darken_factor:  Assombrissement du fond (0.40 = 60 % luminosité)

    Returns:
        Frame composite target_width × target_height
    """
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    interpolation = cv2.INTER_LANCZOS4

    # --- Contenu : ajuster la largeur à target_width ---
    scale = target_width / w
    content_h = int(h * scale)
    content = cv2.resize(frame, (target_width, content_h), interpolation=interpolation)

    if content_h >= target_height:
        # La vidéo est déjà plus haute que cible (rare) : recadrer au centre
        y0 = (content_h - target_height) // 2
        return content[y0:y0 + target_height, :]

    # --- Fond : scale to fill target_height, center-crop width ---
    bg_scale = target_height / h
    bg_w = int(w * bg_scale)
    bg = cv2.resize(frame, (bg_w, target_height), interpolation=cv2.INTER_LINEAR)
    if bg_w > target_width:
        bx = (bg_w - target_width) // 2
        bg = bg[:, bx:bx + target_width]
    elif bg_w < target_width:
        pad = target_width - bg_w
        bg = cv2.copyMakeBorder(bg, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_REFLECT)

    # Flou gaussien lourd + assombrissement
    ks = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    bg = cv2.GaussianBlur(bg, (ks, ks), 0)
    bg = (bg.astype(np.float32) * (1.0 - darken_factor)).clip(0, 255).astype(np.uint8)

    # --- Composite : coller le contenu centré sur le fond ---
    y_off = (target_height - content_h) // 2
    result = bg.copy()
    result[y_off:y_off + content_h, :] = content

    # Dégradé de transition haut/bas pour une fusion naturelle
    fade_h = min(40, content_h // 8, y_off) if y_off > 0 else 0
    if fade_h > 2:
        for row in range(fade_h):
            alpha = row / fade_h
            top_row = y_off + row
            bot_row = y_off + content_h - 1 - row
            result[top_row] = (
                content[row].astype(np.float32) * alpha
                + bg[top_row].astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)
            result[bot_row] = (
                content[content_h - 1 - row].astype(np.float32) * alpha
                + bg[bot_row].astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)

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
