"""
Système de presets unifié pour ClipGenius

Permet de combiner facilement différents effets visuels, styles de sous-titres,
et paramètres d'encodage en configurations prédéfinies.

Usage:
    from src.presets import get_preset, list_presets, PRESETS

    preset = get_preset("tiktok_viral")
    config = preset.to_clip_config()
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

# Constantes de configuration par défaut pour les presets visuels et sous-titres
DEFAULT_ZOOM_FACTOR: float = 1.05
DEFAULT_SHARPENING_STRENGTH: float = 0.3
DEFAULT_VIGNETTE_STRENGTH: float = 0.15
DEFAULT_KEN_BURNS_INTENSITY: float = 0.02
DEFAULT_BLUR_STRENGTH: int = 51
DEFAULT_BASE_FONT_SIZE: int = 56
DEFAULT_KEYWORD_SIZE_MULTIPLIER: float = 1.18


class PresetCategory(Enum):
    """Catégories de presets par type de contenu"""
    PODCAST = "podcast"       # Interviews, discussions, talk-shows
    GAMING = "gaming"         # Gameplay, streams, esport
    VLOG = "vlog"             # Vlogs, lifestyle, voyages
    UTILITY = "utility"       # Rapide, preview, sans effets
    QUALITY = "quality"       # Presets de qualité d'encodage


@dataclass
class VisualPreset:
    """Configuration des effets visuels"""
    # Color grading
    color_grading_enabled: bool = True
    color_grading_style: str = "warm"  # warm, cool, vibrant, cinematic, none

    # Zoom effect
    zoom_enabled: bool = True
    zoom_factor: float = DEFAULT_ZOOM_FACTOR
    zoom_style: str = "ease_out"  # ease_out, ease_in_out, breathing, pulse

    # Post-processing
    sharpening_enabled: bool = True
    sharpening_strength: float = DEFAULT_SHARPENING_STRENGTH
    vignette_enabled: bool = False
    vignette_strength: float = DEFAULT_VIGNETTE_STRENGTH

    # Ken Burns
    ken_burns_enabled: bool = False
    ken_burns_intensity: float = DEFAULT_KEN_BURNS_INTENSITY

    # Blur fill
    blur_fill_enabled: bool = True
    blur_strength: int = DEFAULT_BLUR_STRENGTH


@dataclass
class SubtitlePreset:
    """Configuration des sous-titres"""
    enabled: bool = True
    enriched: bool = True  # Mots-clés colorés
    theme: str = "viral"   # viral, professional, minimal, neon, gaming, podcast
    max_words: int = 3
    use_emojis: bool = True

    # Couleurs personnalisées (optionnel, sinon défaut du thème)
    primary_color: Optional[str] = None          # Couleur principale des mots
    keyword_color: Optional[str] = None          # Couleur des mots-clés
    number_color: Optional[str] = None           # Couleur des chiffres
    emphasis_color: Optional[str] = None         # Couleur des mots d'emphase
    highlight_color: Optional[str] = None        # Couleur du mot actuel

    # Tailles
    base_font_size: int = DEFAULT_BASE_FONT_SIZE                     # Taille de police de base
    keyword_size_multiplier: float = DEFAULT_KEYWORD_SIZE_MULTIPLIER        # Multiplicateur pour mots-clés

    # Style de police
    font_family: str = "Poppins"                 # Police à utiliser
    text_transform: str = "uppercase"            # uppercase, capitalize, none
    uppercase_keywords: bool = True              # Mettre les mots-clés en majuscules

    # Effets visuels
    enable_animations: bool = True               # Activer les animations de mots
    enable_glow: bool = True                     # Activer l'effet de glow
    enable_3d_shadow: bool = True                # Ombres 3D


@dataclass
class EncodingPreset:
    """Configuration de l'encodage"""
    # Qualité vidéo
    crf: int = 18  # 0=lossless, 18=quasi parfait, 23=défaut
    preset: str = "slow"  # ultrafast, fast, medium, slow, veryslow
    video_profile: str = "high"
    video_level: str = "4.1"

    # Bitrates
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"

    # Format
    output_fps: int = 30
    output_width: int = 1080
    output_height: int = 1920
    use_lanczos: bool = True


@dataclass
class ClipGeniusPreset:
    """Preset complet combinant tous les paramètres"""
    name: str
    description: str
    category: PresetCategory

    # Sous-configurations
    visual: VisualPreset = field(default_factory=VisualPreset)
    subtitle: SubtitlePreset = field(default_factory=SubtitlePreset)
    encoding: EncodingPreset = field(default_factory=EncodingPreset)

    # Paramètres de détection
    min_duration: float = 30.0
    max_duration: float = 90.0
    min_viral_score: float = 0.60

    # Options de traitement
    smart_crop: bool = True
    optimize_hooks: bool = True
    advanced_audio: bool = True
    adaptive_duration: bool = True
    use_ai: bool = True

    def _build_subtitle_colors_dict(self) -> Optional[Dict[str, Any]]:
        """
        Construit le dictionnaire de couleurs/styles personnalisés pour les sous-titres
        """
        subtitle_colors = {}

        # Couleurs
        if self.subtitle.primary_color:
            subtitle_colors['primary_color'] = self.subtitle.primary_color
        if self.subtitle.keyword_color:
            subtitle_colors['keyword_color'] = self.subtitle.keyword_color
        if self.subtitle.number_color:
            subtitle_colors['number_color'] = self.subtitle.number_color
        if self.subtitle.emphasis_color:
            subtitle_colors['emphasis_color'] = self.subtitle.emphasis_color
        if self.subtitle.highlight_color:
            subtitle_colors['highlight_color'] = self.subtitle.highlight_color

        # Tailles
        subtitle_colors['base_font_size'] = self.subtitle.base_font_size
        subtitle_colors['keyword_size_multiplier'] = self.subtitle.keyword_size_multiplier

        # Styles
        subtitle_colors['font_family'] = self.subtitle.font_family
        subtitle_colors['text_transform'] = self.subtitle.text_transform
        subtitle_colors['uppercase_keywords'] = self.subtitle.uppercase_keywords

        # Effets
        subtitle_colors['enable_animations'] = self.subtitle.enable_animations
        subtitle_colors['enable_glow'] = self.subtitle.enable_glow
        subtitle_colors['enable_3d_shadow'] = self.subtitle.enable_3d_shadow

        return subtitle_colors if subtitle_colors else None

    def to_clip_config(self) -> Dict[str, Any]:
        """
        Convertit le preset en paramètres pour ClipConfig
        """
        return {
            # Format de sortie
            'output_width': self.encoding.output_width,
            'output_height': self.encoding.output_height,
            'output_fps': self.encoding.output_fps,

            # Durées
            'min_clip_duration': self.min_duration,
            'max_clip_duration': self.max_duration,
            'min_viral_score': self.min_viral_score,

            # Encodage
            'crf': self.encoding.crf,
            'preset': self.encoding.preset,
            'video_profile': self.encoding.video_profile,
            'video_level': self.encoding.video_level,
            'video_bitrate': self.encoding.video_bitrate,
            'audio_bitrate': self.encoding.audio_bitrate,
            'use_lanczos': self.encoding.use_lanczos,

            # Sous-titres (config de base + options personnalisées)
            'add_subtitles': self.subtitle.enabled,
            'subtitle_enriched': self.subtitle.enriched,
            'subtitle_theme': self.subtitle.theme,
            'subtitle_max_words': self.subtitle.max_words,
            'subtitle_use_emojis': self.subtitle.use_emojis,
            'subtitle_custom_colors': self._build_subtitle_colors_dict(),

            # Zoom
            'enable_zoom_effect': self.visual.zoom_enabled,
            'zoom_factor': self.visual.zoom_factor,
            'zoom_style': self.visual.zoom_style,

            # Blur fill
            'enable_blur_fill': self.visual.blur_fill_enabled,
            'blur_strength': self.visual.blur_strength,

            # Smart crop
            'smart_crop': self.smart_crop,

            # Effets cinématiques
            'enable_color_grading': self.visual.color_grading_enabled,
            'color_grading_style': self.visual.color_grading_style,
            'enable_sharpening': self.visual.sharpening_enabled,
            'sharpening_strength': self.visual.sharpening_strength,
            'enable_vignette': self.visual.vignette_enabled,
            'vignette_strength': self.visual.vignette_strength,

            # Ken Burns
            'enable_ken_burns': self.visual.ken_burns_enabled,
            'ken_burns_intensity': self.visual.ken_burns_intensity,
        }

    def get_subtitle_options(self) -> Dict[str, Any]:
        """
        Retourne les options pour les sous-titres
        """
        return {
            'enabled': self.subtitle.enabled,
            'enriched': self.subtitle.enriched,
            'theme': self.subtitle.theme,
            'max_words': self.subtitle.max_words,
            'use_emojis': self.subtitle.use_emojis,
            'primary_color': self.subtitle.primary_color,
            'keyword_color': self.subtitle.keyword_color,
            'number_color': self.subtitle.number_color,
            'emphasis_color': self.subtitle.emphasis_color,
            'highlight_color': self.subtitle.highlight_color,
            'base_font_size': self.subtitle.base_font_size,
            'keyword_size_multiplier': self.subtitle.keyword_size_multiplier,
            'font_family': self.subtitle.font_family,
            'text_transform': self.subtitle.text_transform,
            'uppercase_keywords': self.subtitle.uppercase_keywords,
            'enable_animations': self.subtitle.enable_animations,
            'enable_glow': self.subtitle.enable_glow,
            'enable_3d_shadow': self.subtitle.enable_3d_shadow,
        }


# =============================================================================
# PRESETS PRÉDÉFINIS - importés depuis preset_definitions.py
# =============================================================================

from .preset_definitions import PRESETS


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_preset(name: str) -> ClipGeniusPreset:
    """
    Récupère un preset par son nom.

    Args:
        name: Nom du preset (insensible à la casse)

    Returns:
        Le preset demandé

    Raises:
        KeyError: Si le preset n'existe pas
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower in PRESETS:
        return PRESETS[name_lower]

    # Recherche partielle
    matches = [k for k in PRESETS if name_lower in k]
    if len(matches) == 1:
        return PRESETS[matches[0]]

    available = ", ".join(sorted(PRESETS.keys()))
    raise KeyError(f"Preset '{name}' non trouvé. Disponibles: {available}")


def list_presets(category: Optional[PresetCategory] = None) -> List[ClipGeniusPreset]:
    """
    Liste tous les presets disponibles.

    Args:
        category: Filtrer par catégorie (optionnel)

    Returns:
        Liste des presets
    """
    if category is None:
        return list(PRESETS.values())

    return [p for p in PRESETS.values() if p.category == category]


def get_preset_names() -> List[str]:
    """
    Retourne la liste des noms de presets disponibles.
    """
    return sorted(PRESETS.keys())


def print_presets_table():
    """
    Affiche un tableau formaté de tous les presets disponibles.
    À utiliser avec Rich console.
    """
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title="Presets ClipGenius", show_header=True, header_style="bold cyan")

    table.add_column("Nom", style="cyan", width=15)
    table.add_column("Catégorie", style="magenta", width=10)
    table.add_column("Description", style="white", width=50)
    table.add_column("Durée", justify="right", width=10)

    for name, preset in sorted(PRESETS.items()):
        duration = f"{preset.min_duration:.0f}-{preset.max_duration:.0f}s"
        table.add_row(
            name,
            preset.category.value,
            preset.description[:48] + "..." if len(preset.description) > 50 else preset.description,
            duration
        )

    console.print(table)


# =============================================================================
# CLI INTÉGRATION
# =============================================================================

def apply_preset_to_args(preset: ClipGeniusPreset, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applique un preset aux arguments CLI existants.
    Les arguments explicites de l'utilisateur ont priorité sur le preset.

    Args:
        preset: Le preset à appliquer
        args: Arguments CLI existants

    Returns:
        Arguments fusionnés
    """
    # Paramètres du preset qui peuvent être overridés
    preset_defaults = {
        'min_duration': preset.min_duration,
        'max_duration': preset.max_duration,
        'min_score': preset.min_viral_score,
        'smart_crop': preset.smart_crop,
        'optimize_hooks': preset.optimize_hooks,
        'advanced_audio': preset.advanced_audio,
        'adaptive_duration': preset.adaptive_duration,
        'use_ai': preset.use_ai,
        'subtitles': preset.subtitle.enabled,
        'enriched_subtitles': preset.subtitle.enriched,
        'max_words': preset.subtitle.max_words,
        'emojis': preset.subtitle.use_emojis,
    }

    # Fusionner: preset d'abord, puis override avec args explicites
    result = {}
    for key, preset_value in preset_defaults.items():
        # Si l'utilisateur n'a pas spécifié la valeur, utiliser celle du preset
        if key not in args or args[key] is None:
            result[key] = preset_value
        else:
            result[key] = args[key]

    # Ajouter les autres args non couverts par le preset
    for key, value in args.items():
        if key not in result:
            result[key] = value

    return result


# =============================================================================
# AUTO-CONFIG INTÉGRATION
# =============================================================================

def create_preset_from_auto_config(
    generated_config: 'GeneratedConfig',
    name: str = "Auto-Generated"
) -> ClipGeniusPreset:
    """
    Crée un preset ClipGenius à partir d'une configuration auto-générée.

    Args:
        generated_config: Configuration générée par AutoConfigurator
        name: Nom à donner au preset

    Returns:
        ClipGeniusPreset configuré selon l'analyse
    """
    # Déterminer le thème de sous-titres
    subtitle_theme = generated_config.subtitle_theme
    if subtitle_theme not in ['viral', 'neon', 'minimal', 'professional']:
        subtitle_theme = 'viral'

    return ClipGeniusPreset(
        name=name,
        description=f"Configuration auto-générée pour contenu {generated_config.detected_content_type}",
        category=PresetCategory.VLOG,  # Par défaut

        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style=generated_config.color_grading,
            zoom_enabled=True,
            zoom_factor=generated_config.zoom_factor,
            zoom_style=generated_config.zoom_style,
            sharpening_enabled=True,
            sharpening_strength=generated_config.sharpening_strength,
            vignette_enabled=generated_config.vignette_enabled,
            vignette_strength=generated_config.vignette_strength,
        ),

        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme=subtitle_theme,
            max_words=generated_config.subtitle_max_words,
            use_emojis=generated_config.subtitle_emojis,
        ),

        encoding=EncodingPreset(
            crf=18,
            preset="slow",
            output_fps=30,
        ),

        min_duration=generated_config.min_duration,
        max_duration=generated_config.max_duration,
        min_viral_score=generated_config.min_viral_score,

        smart_crop=generated_config.smart_crop,
        optimize_hooks=generated_config.optimize_hooks,
        advanced_audio=generated_config.advanced_audio,
        adaptive_duration=True,
        use_ai=True,
    )


# Type hint forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.auto_config import GeneratedConfig


if __name__ == "__main__":
    # Afficher les presets disponibles
    print_presets_table()
