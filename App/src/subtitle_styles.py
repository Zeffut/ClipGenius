"""Styles CSS et templates pour les sous-titres animes."""

# CSS pour le style viral TikTok - Police Poppins SemiBold (style officiel TikTok)
# Design moderne avec ombres portées, couleurs vibrantes et effets
VIRAL_CSS = """
@font-face {
    font-family: 'Poppins';
    src: url('Poppins-SemiBold.ttf') format('truetype');
    font-weight: 600;
    font-style: normal;
}

.word {
    font-family: 'Poppins', 'Arial', sans-serif;
    font-size: 56px;
    color: white;
    font-weight: 700;
    /* Ombre portée douce pour profondeur */
    text-shadow:
        /* Contour noir net */
        3px 3px 0px #000,
        -3px -3px 0px #000,
        3px -3px 0px #000,
        -3px 3px 0px #000,
        0px 3px 0px #000,
        0px -3px 0px #000,
        3px 0px 0px #000,
        -3px 0px 0px #000,
        /* Ombre portée douce */
        4px 4px 8px rgba(0, 0, 0, 0.5),
        0px 6px 12px rgba(0, 0, 0, 0.3);
    letter-spacing: 0.5px;
    padding: 0 6px;
    line-height: 1.3;
    text-transform: uppercase;
}

.word-being-narrated {
    color: #00FF88;
    transform: scale(1.05);
    text-shadow:
        /* Contour noir */
        3px 3px 0px #000,
        -3px -3px 0px #000,
        3px -3px 0px #000,
        -3px 3px 0px #000,
        0px 3px 0px #000,
        0px -3px 0px #000,
        3px 0px 0px #000,
        -3px 0px 0px #000,
        /* Glow vert */
        0px 0px 20px rgba(0, 255, 136, 0.6),
        0px 0px 40px rgba(0, 255, 136, 0.3);
}

.word-already-narrated {
    color: #FFFFFF;
}

.word-not-narrated-yet {
    color: rgba(255, 255, 255, 0.85);
}

.segment {
    text-align: center;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 12px;
    padding: 12px 24px;
    /* Fond semi-transparent optionnel pour meilleure lisibilité */
    background: linear-gradient(
        to bottom,
        rgba(0, 0, 0, 0) 0%,
        rgba(0, 0, 0, 0.1) 50%,
        rgba(0, 0, 0, 0) 100%
    );
    border-radius: 8px;
}

.emoji {
    font-size: 60px;
    filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.5));
}
"""

# Template JSON pour le style TikTok viral - SANS ZOOM NI GLOW
VIRAL_TEMPLATE = {
    "css": "styles.css",
    "layout": {
        "max_width_ratio": 0.9,
        "max_number_of_lines": 2,
        "min_number_of_lines": 1,
        "vertical_align": {
            "align": "center",
            "offset": 0.30
        }
    },
    "splitters": [
        {
            "type": "limit_by_words",
            "limit": 3
        }
    ],
    "effects": [
        {
            "type": "remove_punctuation_marks",
            "punctuation_marks": [".", ","],
            "exception_marks": ["...", "!", "?"]
        },
        {
            "type": "emoji_in_segment",
            "chance_to_apply": 0.8,
            "align": "random",
            "max_uses_of_each_emoji": 2,
            "max_consecutive_segments_with_emoji": 2
        },
        {
            "type": "animate_segment_emojis"
        }
    ],
    "animations": [
        {
            "type": "fade_in",
            "when": "narration-starts",
            "what": "segment",
            "duration": 0.1
        },
        {
            "type": "fade_out",
            "when": "narration-ends",
            "what": "segment",
            "duration": 0.1
        }
    ],
    "tagger_rules": []
}
