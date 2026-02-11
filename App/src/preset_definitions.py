"""Definitions des presets ClipGenius."""

from typing import Dict

from .presets import ClipGeniusPreset, PresetCategory, VisualPreset, SubtitlePreset, EncodingPreset


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
