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
    zoom_factor: float = 1.05
    zoom_style: str = "ease_out"  # ease_out, ease_in_out, breathing, pulse
    
    # Post-processing
    sharpening_enabled: bool = True
    sharpening_strength: float = 0.3
    vignette_enabled: bool = False
    vignette_strength: float = 0.15
    
    # Ken Burns
    ken_burns_enabled: bool = False
    ken_burns_intensity: float = 0.02
    
    # Blur fill
    blur_fill_enabled: bool = True
    blur_strength: int = 51


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
    base_font_size: int = 56                     # Taille de police de base
    keyword_size_multiplier: float = 1.18        # Multiplicateur pour mots-clés
    
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
# PRESETS PRÉDÉFINIS PAR TYPE DE CONTENU
# =============================================================================

PRESETS: Dict[str, ClipGeniusPreset] = {
    
    # -------------------------------------------------------------------------
    # PODCAST / INTERVIEW - Focus sur la parole et les échanges
    # -------------------------------------------------------------------------
    
    "podcast": ClipGeniusPreset(
        name="Podcast",
        description="Optimisé pour les podcasts et interviews. Focus sur la clarté du discours et les moments forts.",
        category=PresetCategory.PODCAST,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="warm",
            zoom_enabled=True,
            zoom_factor=1.03,
            zoom_style="breathing",
            sharpening_enabled=True,
            sharpening_strength=0.2,
            vignette_enabled=True,
            vignette_strength=0.1,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="podcast",
            max_words=4,
            use_emojis=False,
            # Style professionnel épuré pour podcasts
            primary_color="#FFFFFF",
            keyword_color="#FFD700",  # Or doux
            number_color="#87CEEB",   # Bleu ciel
            emphasis_color="#FFA07A",  # Saumon
            highlight_color="#32CD32", # Vert lime
            base_font_size=54,
            keyword_size_multiplier=1.15,
            text_transform="capitalize",  # Capitalisation naturelle
            uppercase_keywords=False,
            enable_animations=True,
            enable_glow=False,  # Pas de glow pour un look propre
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=18,
            preset="slow",
            audio_bitrate="256k",  # Audio haute qualité pour podcast
            output_fps=30,
        ),
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.55,
    ),
    
    "interview": ClipGeniusPreset(
        name="Interview",
        description="Pour les interviews et discussions. Transitions douces, sous-titres lisibles.",
        category=PresetCategory.PODCAST,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="cool",
            zoom_enabled=True,
            zoom_factor=1.02,
            zoom_style="ease_in_out",
            sharpening_enabled=True,
            sharpening_strength=0.2,
            vignette_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="professional",
            max_words=5,
            use_emojis=False,
            # Style corporatif élégant
            primary_color="#F5F5F5",
            keyword_color="#4A90E2",  # Bleu corporate
            number_color="#7ED321",   # Vert success
            emphasis_color="#F5A623",  # Orange attention
            highlight_color="#50E3C2", # Turquoise
            base_font_size=52,
            keyword_size_multiplier=1.12,
            text_transform="capitalize",
            uppercase_keywords=False,
            enable_animations=True,
            enable_glow=False,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=17,
            preset="slow",
            audio_bitrate="192k",
            output_fps=30,
        ),
        min_duration=30.0,
        max_duration=120.0,
        min_viral_score=0.50,
        optimize_hooks=False,
    ),
    
    # -------------------------------------------------------------------------
    # GAMING - Dynamique et énergique
    # -------------------------------------------------------------------------
    
    "gaming": ClipGeniusPreset(
        name="Gaming",
        description="Pour les clips de jeux vidéo. Couleurs vibrantes, effets dynamiques, sous-titres flashy.",
        category=PresetCategory.GAMING,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="vibrant",
            zoom_enabled=True,
            zoom_factor=1.06,
            zoom_style="pulse",
            sharpening_enabled=True,
            sharpening_strength=0.35,
            vignette_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="gaming",
            max_words=3,
            use_emojis=True,
            # Style gaming énergique avec néons
            primary_color="#FFFFFF",
            keyword_color="#FF00FF",  # Magenta vif
            number_color="#00FFFF",   # Cyan électrique
            emphasis_color="#FF0066",  # Rose flash
            highlight_color="#00FF00", # Vert néon
            base_font_size=58,
            keyword_size_multiplier=1.25,
            text_transform="uppercase",
            uppercase_keywords=True,
            enable_animations=True,
            enable_glow=True,  # Glow néon fort
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=18,
            preset="medium",
            output_fps=30,
        ),
        min_duration=15.0,
        max_duration=60.0,
        min_viral_score=0.55,
    ),
    
    "stream": ClipGeniusPreset(
        name="Stream",
        description="Pour les meilleurs moments de stream. Capture les réactions et moments forts.",
        category=PresetCategory.GAMING,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="vibrant",
            zoom_enabled=True,
            zoom_factor=1.05,
            zoom_style="ease_out",
            sharpening_enabled=True,
            sharpening_strength=0.3,
            vignette_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="viral",
            max_words=3,
            use_emojis=True,
        ),
        encoding=EncodingPreset(
            crf=19,
            preset="medium",
            output_fps=30,
        ),
        min_duration=15.0,
        max_duration=60.0,
        min_viral_score=0.50,
    ),
    
    # -------------------------------------------------------------------------
    # VLOG - Lifestyle et voyage
    # -------------------------------------------------------------------------
    
    "vlog": ClipGeniusPreset(
        name="Vlog",
        description="Pour les vlogs lifestyle et voyage. Look chaleureux et authentique.",
        category=PresetCategory.VLOG,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="warm",
            zoom_enabled=True,
            zoom_factor=1.04,
            zoom_style="breathing",
            sharpening_enabled=True,
            sharpening_strength=0.25,
            vignette_enabled=True,
            vignette_strength=0.12,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="viral",
            max_words=3,
            use_emojis=True,
            # Style vlog chaleureux et amical
            primary_color="#FFFFFF",
            keyword_color="#FFB84D",  # Orange doré chaud
            number_color="#5FC3E4",   # Bleu ciel
            emphasis_color="#FF6B9D",  # Rose vif
            highlight_color="#00FF88", # Vert néon
            base_font_size=56,
            keyword_size_multiplier=1.18,
            text_transform="uppercase",
            uppercase_keywords=True,
            enable_animations=True,
            enable_glow=True,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=18,
            preset="slow",
            output_fps=30,
        ),
        min_duration=15.0,
        max_duration=60.0,
        min_viral_score=0.55,
    ),
    
    "lifestyle": ClipGeniusPreset(
        name="Lifestyle",
        description="Pour le contenu lifestyle haut de gamme. Look épuré et élégant.",
        category=PresetCategory.VLOG,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="cinematic",
            zoom_enabled=True,
            zoom_factor=1.03,
            zoom_style="ease_in_out",
            sharpening_enabled=True,
            sharpening_strength=0.2,
            vignette_enabled=True,
            vignette_strength=0.15,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=False,
            theme="minimal",
            max_words=4,
            use_emojis=False,
            # Style minimaliste chic
            primary_color="#F8F8F8",
            keyword_color="#D4AF37",  # Or antique élégant
            number_color="#B4C7DC",   # Bleu gris doux
            emphasis_color="#C9A98F",  # Beige rosé
            highlight_color="#E8D5C4", # Crème doré
            base_font_size=52,
            keyword_size_multiplier=1.10,
            text_transform="capitalize",
            uppercase_keywords=False,
            enable_animations=True,
            enable_glow=False,  # Pas de glow pour look épuré
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=16,
            preset="slow",
            output_fps=30,
        ),
        min_duration=20.0,
        max_duration=60.0,
        min_viral_score=0.60,
    ),
    
    # -------------------------------------------------------------------------
    # QUALITÉ D'ENCODAGE - Presets axés sur la qualité vidéo
    # -------------------------------------------------------------------------
    
    "standard": ClipGeniusPreset(
        name="Standard",
        description="Qualité standard. Bon équilibre qualité/taille. Encodage rapide.",
        category=PresetCategory.QUALITY,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="warm",
            zoom_enabled=True,
            zoom_factor=1.04,
            zoom_style="ease_out",
            sharpening_enabled=True,
            sharpening_strength=0.25,
            vignette_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="viral",
            max_words=3,
            use_emojis=True,
        ),
        encoding=EncodingPreset(
            crf=22,                     # CRF 22 = bonne qualité, fichiers raisonnables
            preset="fast",              # Encodage rapide
            video_bitrate="10M",        # Bitrate de fallback
            audio_bitrate="192k",
            output_fps=30,
            use_lanczos=False,          # LINEAR pour vitesse
        ),
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.60,
    ),
    
    "high": ClipGeniusPreset(
        name="Haute Qualité",
        description="Haute qualité. CRF 18, preset medium. Idéal pour publication finale.",
        category=PresetCategory.QUALITY,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="warm",
            zoom_enabled=True,
            zoom_factor=1.05,
            zoom_style="ease_out",
            sharpening_enabled=True,
            sharpening_strength=0.3,
            vignette_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="viral",
            max_words=3,
            use_emojis=True,
            base_font_size=58,
            enable_glow=True,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=18,                     # CRF 18 = quasi parfait visuellement
            preset="medium",            # Bon équilibre vitesse/qualité
            video_bitrate="15M",        # Bitrate élevé pour 1080x1920
            audio_bitrate="192k",
            video_level="4.2",          # Niveau 4.2 pour 1080p60
            output_fps=30,
            use_lanczos=True,           # LANCZOS4 pour meilleure qualité
        ),
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.60,
    ),
    
    "ultra": ClipGeniusPreset(
        name="Ultra Qualité",
        description="Qualité maximale. CRF 15, preset slow. Pour les créateurs exigeants.",
        category=PresetCategory.QUALITY,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="cinematic",
            zoom_enabled=True,
            zoom_factor=1.05,
            zoom_style="ease_out",
            sharpening_enabled=True,
            sharpening_strength=0.35,   # Sharpening plus fort
            vignette_enabled=True,
            vignette_strength=0.1,
            ken_burns_enabled=False,    # Disponible mais désactivé par défaut
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="viral",
            max_words=3,
            use_emojis=True,
            base_font_size=60,
            keyword_size_multiplier=1.2,
            enable_glow=True,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=15,                     # CRF 15 = qualité quasi-lossless
            preset="slow",              # Meilleure compression, plus lent
            video_bitrate="20M",        # Bitrate très élevé
            audio_bitrate="256k",       # Audio haute qualité
            video_profile="high",
            video_level="4.2",
            output_fps=30,
            use_lanczos=True,           # Obligatoire pour ultra
        ),
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.55,           # Seuil légèrement plus bas pour plus de contenu
    ),
    
    "master": ClipGeniusPreset(
        name="Master",
        description="Qualité master/archive. CRF 12, preset veryslow. Fichiers très lourds.",
        category=PresetCategory.QUALITY,
        visual=VisualPreset(
            color_grading_enabled=True,
            color_grading_style="cinematic",
            zoom_enabled=True,
            zoom_factor=1.05,
            zoom_style="ease_out",
            sharpening_enabled=True,
            sharpening_strength=0.4,
            vignette_enabled=True,
            vignette_strength=0.12,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=True,
            theme="viral",
            max_words=3,
            use_emojis=True,
            base_font_size=62,
            keyword_size_multiplier=1.22,
            enable_glow=True,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=12,                     # CRF 12 = pratiquement lossless
            preset="veryslow",          # Compression maximale (très lent)
            video_bitrate="30M",        # Bitrate très élevé
            audio_bitrate="320k",       # Audio master
            video_profile="high",
            video_level="5.1",          # Niveau 5.1 pour résolutions élevées
            output_fps=30,
            use_lanczos=True,
        ),
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.50,
    ),
    
    # -------------------------------------------------------------------------
    # UTILITAIRES - Modes spéciaux
    # -------------------------------------------------------------------------
    
    "fast": ClipGeniusPreset(
        name="Rapide",
        description="Encodage ultra-rapide. Qualité réduite mais traitement 3x plus rapide.",
        category=PresetCategory.UTILITY,
        visual=VisualPreset(
            color_grading_enabled=False,
            zoom_enabled=True,
            zoom_factor=1.03,
            zoom_style="ease_out",
            sharpening_enabled=False,
            vignette_enabled=False,
            blur_fill_enabled=True,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=False,
            theme="viral",
            max_words=3,
            use_emojis=False,
            # Style basique pour performance
            base_font_size=56,
            enable_animations=False,  # Désactivé pour performance
            enable_glow=False,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=23,
            preset="ultrafast",
            use_lanczos=False,
            output_fps=30,
        ),
        smart_crop=False,
        advanced_audio=False,
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.60,
    ),
    
    "clean": ClipGeniusPreset(
        name="Sans effets",
        description="Sans effets visuels. Recadrage intelligent et sous-titres uniquement.",
        category=PresetCategory.UTILITY,
        visual=VisualPreset(
            color_grading_enabled=False,
            zoom_enabled=False,
            sharpening_enabled=False,
            vignette_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=True,
            enriched=False,
            theme="minimal",
            max_words=5,
            use_emojis=False,
            # Style minimal propre
            primary_color="#FFFFFF",
            base_font_size=54,
            text_transform="capitalize",
            enable_animations=False,
            enable_glow=False,
            enable_3d_shadow=True,
        ),
        encoding=EncodingPreset(
            crf=18,
            preset="medium",
            output_fps=30,
        ),
        min_duration=30.0,
        max_duration=90.0,
        min_viral_score=0.60,
    ),
    
    "preview": ClipGeniusPreset(
        name="Preview",
        description="Mode preview pour tester rapidement. Basse qualité, encodage instantané.",
        category=PresetCategory.UTILITY,
        visual=VisualPreset(
            color_grading_enabled=False,
            zoom_enabled=False,
            sharpening_enabled=False,
            vignette_enabled=False,
            blur_fill_enabled=False,
        ),
        subtitle=SubtitlePreset(
            enabled=False,
        ),
        encoding=EncodingPreset(
            crf=28,
            preset="ultrafast",
            use_lanczos=False,
            output_width=720,
            output_height=1280,
            output_fps=24,
        ),
        smart_crop=False,
        optimize_hooks=False,
        advanced_audio=False,
        adaptive_duration=False,
        use_ai=False,
        min_duration=15.0,
        max_duration=60.0,
        min_viral_score=0.50,
    ),
    
}


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
