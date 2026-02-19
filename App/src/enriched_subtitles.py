"""
Module de sous-titres enrichis avec mise en valeur des mots-clés
Style professionnel type Opus Clip / Captions.ai
"""

import re
import os
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from rich.console import Console

console = Console()

# Constantes de configuration des sous-titres
DEFAULT_BASE_FONT_SIZE: int = 56
DEFAULT_KEYWORD_SIZE_MULTIPLIER: float = 1.18
DEFAULT_NUMBER_SIZE_MULTIPLIER: float = 1.12
EMPHASIS_SIZE_MULTIPLIER: float = 1.1
QUESTION_SIZE_MULTIPLIER: float = 1.05
DEFAULT_LETTER_SPACING: float = 0.5
DEFAULT_MAX_WORDS_PER_SEGMENT: int = 3


@dataclass
class EnrichedWord:
    """Mot enrichi avec métadonnées de styling"""
    word: str
    start: float
    end: float
    importance: str  # 'normal', 'keyword', 'number', 'emphasis', 'question'
    color: Optional[str] = None
    size_multiplier: float = 1.0
    animation: Optional[str] = None  # 'pop', 'bounce', 'highlight'


@dataclass
class SubtitleStyle:
    """Configuration du style des sous-titres"""
    # Couleurs par type
    normal_color: str = "#FFFFFF"
    keyword_color: str = "#FFD700"  # Or pour les mots-clés
    number_color: str = "#00FFFF"   # Cyan pour les chiffres
    emphasis_color: str = "#FF6B6B"  # Rouge pour l'emphase
    question_color: str = "#9B59B6"  # Violet pour les questions
    highlight_color: str = "#00FF88"  # Vert néon pour le mot actuel

    # Couleurs additionnelles
    action_color: str = "#FF9500"    # Orange pour les verbes d'action
    urgency_color: str = "#FF3B30"   # Rouge vif pour l'urgence
    value_color: str = "#34C759"     # Vert pour la valeur

    # Tailles
    base_font_size: int = DEFAULT_BASE_FONT_SIZE
    keyword_size_multiplier: float = DEFAULT_KEYWORD_SIZE_MULTIPLIER
    number_size_multiplier: float = DEFAULT_NUMBER_SIZE_MULTIPLIER

    # Animations
    enable_animations: bool = True
    keyword_animation: str = "pop"
    enable_3d_effect: bool = True     # Effet 3D sur les mots
    enable_gradient_text: bool = True # Dégradé sur les mots-clés

    # Style global
    font_family: str = "Poppins"
    text_transform: str = "uppercase"  # "uppercase", "capitalize", "none"
    letter_spacing: float = DEFAULT_LETTER_SPACING

    # Autres options
    max_words_per_segment: int = DEFAULT_MAX_WORDS_PER_SEGMENT
    uppercase_keywords: bool = True

    # Thème prédéfini
    theme: str = "viral"  # "viral", "professional", "minimal", "neon"


# Mots-clés à mettre en valeur (par catégorie)
KEYWORD_CATEGORIES = {
    # Impact / Émotion
    'impact': {
        'fr': ['incroyable', 'extraordinaire', 'génial', 'parfait', 'terrible',
               'énorme', 'fou', 'dingue', 'malade', 'ouf', 'grave', 'trop',
               'jamais', 'toujours', 'absolument', 'totalement', 'vraiment',
               'meilleur', 'pire', 'premier', 'dernier', 'unique', 'secret'],
        'en': ['amazing', 'incredible', 'awesome', 'perfect', 'terrible',
               'huge', 'crazy', 'insane', 'never', 'always', 'absolutely',
               'totally', 'really', 'best', 'worst', 'first', 'last', 'only', 'secret']
    },
    # Action / Verbes forts
    'action': {
        'fr': ['découvrez', 'regardez', 'écoutez', 'imaginez', 'arrêtez',
               'commencez', 'changez', 'transformez', 'révèle', 'montre'],
        'en': ['discover', 'watch', 'listen', 'imagine', 'stop',
               'start', 'change', 'transform', 'reveal', 'show']
    },
    # Urgence
    'urgency': {
        'fr': ['maintenant', 'aujourd\'hui', 'vite', 'urgent', 'attention',
               'important', 'critique', 'immédiatement'],
        'en': ['now', 'today', 'quick', 'urgent', 'attention',
               'important', 'critical', 'immediately']
    },
    # Valeur
    'value': {
        'fr': ['gratuit', 'offert', 'bonus', 'exclusif', 'limité',
               'spécial', 'nouveau', 'astuce', 'conseil', 'hack'],
        'en': ['free', 'bonus', 'exclusive', 'limited', 'special',
               'new', 'tip', 'trick', 'hack', 'secret']
    }
}

# Mots d'emphase (à toujours mettre en valeur)
EMPHASIS_WORDS = {
    'fr': ['mais', 'sauf', 'attention', 'stop', 'non', 'oui', 'wow', 'oh', 'ah'],
    'en': ['but', 'except', 'attention', 'stop', 'no', 'yes', 'wow', 'oh', 'ah']
}

# Patterns de questions
QUESTION_PATTERNS = [
    r'^(pourquoi|comment|quand|où|qui|que|quoi|est-ce|quel|quelle)',
    r'^(why|how|when|where|who|what|which|is|are|do|does|can|could|would)',
    r'\?$'
]


class EnrichedSubtitleProcessor:
    """
    Processeur de sous-titres enrichis.

    Analyse chaque mot pour déterminer son importance et son styling:
    - Mots-clés: couleur dorée, taille plus grande
    - Chiffres/statistiques: couleur cyan
    - Mots d'emphase: couleur rouge
    - Questions: couleur violette
    """

    def __init__(
        self,
        style: Optional[SubtitleStyle] = None,
        language: str = 'fr'
    ):
        self.style = style or SubtitleStyle()
        self.language = language
        self._build_keyword_sets()

    def _build_keyword_sets(self):
        """Construit les ensembles de mots-clés pour recherche rapide"""
        self.keywords: Set[str] = set()
        self.emphasis_words: Set[str] = set()

        # Ajouter les mots-clés de toutes les catégories
        for category, langs in KEYWORD_CATEGORIES.items():
            for lang in ['fr', 'en']:  # Supporter les deux langues
                self.keywords.update(word.lower() for word in langs.get(lang, []))

        # Ajouter les mots d'emphase
        for lang in ['fr', 'en']:
            self.emphasis_words.update(word.lower() for word in EMPHASIS_WORDS.get(lang, []))

    def classify_word(self, word: str, context: List[str] = None) -> Tuple[str, float]:
        """
        Classifie un mot et retourne son type d'importance.

        Args:
            word: Le mot à classifier
            context: Liste des mots environnants (optionnel)

        Returns:
            Tuple (importance_type, size_multiplier)
        """
        word_lower = word.lower().strip('.,!?;:\'\"()[]{}')

        # 1. Vérifier si c'est un chiffre ou contient des statistiques
        if self._is_number_or_stat(word):
            return 'number', self.style.number_size_multiplier

        # 2. Vérifier si c'est un mot-clé
        if word_lower in self.keywords:
            return 'keyword', self.style.keyword_size_multiplier

        # 3. Vérifier si c'est un mot d'emphase
        if word_lower in self.emphasis_words:
            return 'emphasis', EMPHASIS_SIZE_MULTIPLIER

        # 4. Vérifier si c'est une question (premier mot)
        if context and len(context) > 0 and context[0].lower() == word_lower:
            for pattern in QUESTION_PATTERNS:
                if re.match(pattern, word_lower, re.IGNORECASE):
                    return 'question', QUESTION_SIZE_MULTIPLIER

        return 'normal', 1.0

    def _is_number_or_stat(self, word: str) -> bool:
        """Vérifie si le mot est un nombre ou une statistique"""
        # Patterns de nombres/stats
        patterns = [
            r'^\d+$',           # Nombre simple
            r'^\d+[%€$£¥]$',    # Nombre avec symbole
            r'^\d+[kKmMbB]$',   # Nombre abrégé (1k, 5M, etc.)
            r'^\d+[,\.]\d+',    # Nombre décimal
            r'^#\d+$',          # Classement (#1)
            r'^\d+(er|ère|ème|st|nd|rd|th)$',  # Ordinaux
        ]

        for pattern in patterns:
            if re.match(pattern, word):
                return True

        return False

    def process_words(
        self,
        words: List[Dict]
    ) -> List[EnrichedWord]:
        """
        Traite une liste de mots et retourne des mots enrichis.

        Args:
            words: Liste de dicts avec 'word', 'start', 'end'

        Returns:
            Liste de EnrichedWord avec styling
        """
        # Extraire le contexte (liste des mots)
        context = [w.get('word', w.get('text', '')) for w in words]

        enriched = []
        for w in words:
            word_text = w.get('word', w.get('text', ''))
            start = w.get('start', w.get('start_time', 0))
            end = w.get('end', w.get('end_time', 0))

            # Classifier le mot
            importance, size_mult = self.classify_word(word_text, context)

            # Déterminer la couleur
            color = self._get_color_for_importance(importance)

            # Déterminer l'animation
            animation = None
            if self.style.enable_animations and importance in ['keyword', 'number']:
                animation = self.style.keyword_animation

            # Appliquer la mise en majuscule pour les mots-clés si configuré
            display_word = word_text
            if self.style.uppercase_keywords and importance == 'keyword':
                display_word = word_text.upper()

            enriched.append(EnrichedWord(
                word=display_word,
                start=start,
                end=end,
                importance=importance,
                color=color,
                size_multiplier=size_mult,
                animation=animation
            ))

        return enriched

    def _get_color_for_importance(self, importance: str) -> str:
        """Retourne la couleur pour un type d'importance"""
        color_map = {
            'normal': self.style.normal_color,
            'keyword': self.style.keyword_color,
            'number': self.style.number_color,
            'emphasis': self.style.emphasis_color,
            'question': self.style.question_color
        }
        return color_map.get(importance, self.style.normal_color)

    def generate_enhanced_css(self) -> str:
        """
        Génère le CSS enrichi pour pycaps avec les styles de mots-clés.
        Design moderne avec ombres portées, glow, effets 3D et animations fluides.
        """
        # Définir les ombres 3D selon le thème
        if self.style.theme == "neon":
            base_shadow = """
                0 0 5px #fff,
                0 0 10px #fff,
                0 0 20px currentColor,
                0 0 40px currentColor;
            """
            highlight_shadow = """
                0 0 10px #fff,
                0 0 20px currentColor,
                0 0 40px currentColor,
                0 0 80px currentColor;
            """
        elif self.style.theme == "minimal":
            base_shadow = "2px 2px 4px rgba(0, 0, 0, 0.5);"
            highlight_shadow = "2px 2px 8px rgba(0, 0, 0, 0.7);"
        else:  # viral ou professional
            base_shadow = """
                /* Contour noir épais pour lisibilité */
                3px 3px 0px #000,
                -3px -3px 0px #000,
                3px -3px 0px #000,
                -3px 3px 0px #000,
                0px 3px 0px #000,
                0px -3px 0px #000,
                3px 0px 0px #000,
                -3px 0px 0px #000,
                /* Ombre portée douce */
                5px 5px 15px rgba(0, 0, 0, 0.7);
            """
            highlight_shadow = """
                3px 3px 0px #000,
                -3px -3px 0px #000,
                3px -3px 0px #000,
                -3px 3px 0px #000,
                0px 3px 0px #000,
                0px -3px 0px #000,
                3px 0px 0px #000,
                -3px 0px 0px #000,
                /* Glow effect puissant */
                0px 0px 30px rgba(0, 255, 136, 0.8),
                0px 0px 60px rgba(0, 255, 136, 0.5);
            """

        css = f"""
@font-face {{
    font-family: 'Poppins';
    src: url('Poppins-SemiBold.ttf') format('truetype');
    font-weight: 600;
    font-style: normal;
}}

/* ===== BASE WORD STYLE ===== */
.word {{
    font-family: '{self.style.font_family}', 'Arial Black', sans-serif;
    font-size: {self.style.base_font_size}px;
    color: {self.style.normal_color};
    font-weight: 800;
    text-shadow: {base_shadow}
    letter-spacing: {self.style.letter_spacing}px;
    padding: 4px 10px 10px 10px;
    line-height: 1.5;
    text-transform: {self.style.text_transform};
    transition: all 0.12s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    display: inline-block;
}}

/* ===== NARRATION STATES ===== */
.word-being-narrated {{
    color: {self.style.highlight_color};
    transform: scale(1.12);
    text-shadow: {highlight_shadow}
}}

.word-already-narrated {{
    color: {self.style.normal_color};
    opacity: 0.95;
}}

.word-not-narrated-yet {{
    color: rgba(255, 255, 255, 0.75);
}}

/* ===== KEYWORD STYLES - OR avec glow doré ===== */
.word-keyword {{
    color: {self.style.keyword_color};
    font-size: {int(self.style.base_font_size * self.style.keyword_size_multiplier)}px;
    font-weight: 900;
    background: linear-gradient(180deg, #FFE566 0%, #FFD700 50%, #FFA500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 2px rgba(255, 215, 0, 0.5));
}}

.word-keyword.word-being-narrated {{
    transform: scale(1.22);
    filter: drop-shadow(0 0 8px rgba(255, 215, 0, 0.9))
            drop-shadow(0 0 20px rgba(255, 215, 0, 0.6));
    animation: keyword-pulse 0.4s ease-out;
}}

/* ===== NUMBER STYLES - CYAN électrique ===== */
.word-number {{
    color: {self.style.number_color};
    font-size: {int(self.style.base_font_size * self.style.number_size_multiplier)}px;
    font-weight: 900;
    background: linear-gradient(180deg, #66FFFF 0%, #00FFFF 50%, #00CCCC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.word-number.word-being-narrated {{
    transform: scale(1.18);
    filter: drop-shadow(0 0 6px rgba(0, 255, 255, 0.9))
            drop-shadow(0 0 15px rgba(0, 255, 255, 0.6));
    animation: number-flash 0.3s ease-out;
}}

/* ===== EMPHASIS STYLES - ROUGE vif ===== */
.word-emphasis {{
    color: {self.style.emphasis_color};
    font-weight: 900;
}}

.word-emphasis.word-being-narrated {{
    transform: scale(1.15);
    filter: drop-shadow(0 0 8px rgba(255, 107, 107, 0.8));
}}

/* ===== ACTION STYLES - ORANGE dynamique ===== */
.word-action {{
    color: {self.style.action_color};
    font-weight: 800;
    font-style: italic;
}}

.word-action.word-being-narrated {{
    transform: scale(1.1) skewX(-3deg);
    filter: drop-shadow(0 0 6px rgba(255, 149, 0, 0.8));
}}

/* ===== URGENCY STYLES - ROUGE urgent ===== */
.word-urgency {{
    color: {self.style.urgency_color};
    font-weight: 900;
    text-transform: uppercase;
}}

.word-urgency.word-being-narrated {{
    transform: scale(1.2);
    animation: urgency-shake 0.3s ease-out;
    filter: drop-shadow(0 0 10px rgba(255, 59, 48, 0.9));
}}

/* ===== QUESTION STYLES - VIOLET mystérieux ===== */
.word-question {{
    color: {self.style.question_color};
}}

.word-question.word-being-narrated {{
    transform: scale(1.1);
    filter: drop-shadow(0 0 8px rgba(155, 89, 182, 0.8));
}}

/* ===== SEGMENT CONTAINER ===== */
.segment {{
    text-align: center;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 14px;
    padding: 16px 32px;
    background: linear-gradient(
        to bottom,
        rgba(0, 0, 0, 0) 0%,
        rgba(0, 0, 0, 0.2) 50%,
        rgba(0, 0, 0, 0) 100%
    );
    border-radius: 12px;
}}

/* ===== EMOJI STYLES ===== */
.emoji {{
    font-size: 64px;
    filter: drop-shadow(4px 4px 8px rgba(0, 0, 0, 0.6));
    animation: emoji-bounce 0.5s ease-out;
}}

/* ===== ANIMATIONS ===== */
@keyframes pop {{
    0% {{ transform: scale(1); }}
    40% {{ transform: scale(1.3); }}
    70% {{ transform: scale(0.95); }}
    100% {{ transform: scale(1.12); }}
}}

@keyframes keyword-pulse {{
    0% {{ transform: scale(1); filter: brightness(1); }}
    50% {{ transform: scale(1.3); filter: brightness(1.3); }}
    100% {{ transform: scale(1.22); filter: brightness(1.1); }}
}}

@keyframes number-flash {{
    0% {{ transform: scale(1); filter: brightness(1); }}
    30% {{ transform: scale(1.25); filter: brightness(1.5); }}
    100% {{ transform: scale(1.18); filter: brightness(1); }}
}}

@keyframes urgency-shake {{
    0%, 100% {{ transform: translateX(0) scale(1.2); }}
    25% {{ transform: translateX(-4px) scale(1.2); }}
    75% {{ transform: translateX(4px) scale(1.2); }}
}}

@keyframes emoji-bounce {{
    0% {{ transform: scale(0) rotate(-20deg); }}
    50% {{ transform: scale(1.2) rotate(10deg); }}
    70% {{ transform: scale(0.9) rotate(-5deg); }}
    100% {{ transform: scale(1) rotate(0deg); }}
}}

@keyframes glow-pulse {{
    0%, 100% {{ filter: drop-shadow(0 0 5px currentColor); }}
    50% {{ filter: drop-shadow(0 0 20px currentColor); }}
}}

/* Animation classes pour JS */
.word-keyword.animate-pop {{
    animation: pop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}}

.word-number.animate-pop {{
    animation: number-flash 0.25s ease-out;
}}

/* ===== RESPONSIVE ===== */
@media (max-width: 720px) {{
    .word {{
        font-size: {int(self.style.base_font_size * 0.85)}px;
    }}
    .word-keyword {{
        font-size: {int(self.style.base_font_size * self.style.keyword_size_multiplier * 0.85)}px;
    }}
}}
"""
        return css

    def create_word_tagger_rules(self) -> List[Dict]:
        """
        Crée les règles de tagging pour pycaps.
        Ces règles associent des classes CSS aux mots selon leur type.
        """
        rules = []

        # Règles pour les mots-clés
        for word in self.keywords:
            rules.append({
                "pattern": f"\\b{re.escape(word)}\\b",
                "case_sensitive": False,
                "tags": ["word-keyword"]
            })

        # Règles pour les mots d'emphase
        for word in self.emphasis_words:
            rules.append({
                "pattern": f"\\b{re.escape(word)}\\b",
                "case_sensitive": False,
                "tags": ["word-emphasis"]
            })

        # Règles pour les nombres et statistiques
        number_patterns = [
            r"\d+%",
            r"\d+[€$£¥]",
            r"\d+[kKmMbB]",
            r"#\d+",
        ]
        for pattern in number_patterns:
            rules.append({
                "pattern": pattern,
                "case_sensitive": False,
                "tags": ["word-number"]
            })

        return rules


def _apply_custom_colors(style: 'SubtitleStyle', custom_colors: Dict[str, str]) -> None:
    """Applique les couleurs/options personnalisées sur un SubtitleStyle.

    Args:
        style: Instance de SubtitleStyle à modifier en place
        custom_colors: Dictionnaire de couleurs et options personnalisées
    """
    # Mapping direct: clé du dict -> attribut du style
    _DIRECT_MAPPINGS = {
        'primary_color': 'normal_color',
        'keyword_color': 'keyword_color',
        'number_color': 'number_color',
        'emphasis_color': 'emphasis_color',
        'highlight_color': 'highlight_color',
        'base_font_size': 'base_font_size',
        'keyword_size_multiplier': 'keyword_size_multiplier',
        'font_family': 'font_family',
        'text_transform': 'text_transform',
    }

    for key, attr in _DIRECT_MAPPINGS.items():
        if key in custom_colors and custom_colors[key]:
            setattr(style, attr, custom_colors[key])

    # Champs booléens (vérifier is not None car False est une valeur valide)
    _BOOL_MAPPINGS = {
        'uppercase_keywords': 'uppercase_keywords',
        'enable_animations': 'enable_animations',
        'enable_3d_shadow': 'enable_3d_effect',
    }

    for key, attr in _BOOL_MAPPINGS.items():
        if key in custom_colors and custom_colors[key] is not None:
            setattr(style, attr, custom_colors[key])


def create_enriched_subtitle_template(
    style: Optional[SubtitleStyle] = None,
    max_words: int = 3,
    use_emojis: bool = True,
    custom_colors: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Crée un template pycaps enrichi avec styling des mots-clés.

    Args:
        style: Configuration du style
        max_words: Nombre max de mots par segment
        use_emojis: Activer les émojis
        custom_colors: Dictionnaire de couleurs personnalisées à appliquer au style

    Returns:
        Dict de configuration template pycaps
    """
    # Créer ou copier le style
    if style is None:
        style = SubtitleStyle()

    # Appliquer les couleurs personnalisées si fournies
    if custom_colors:
        _apply_custom_colors(style, custom_colors)
        if 'enable_glow' in custom_colors and custom_colors['enable_glow'] is not None:
            # Le glow affecte le thème
            if not custom_colors['enable_glow'] and style.theme in ['viral', 'neon', 'gaming']:
                style.theme = 'minimal'

    processor = EnrichedSubtitleProcessor(style=style)

    template = {
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
                "limit": max_words
            }
        ],
        "effects": [
            {
                "type": "remove_punctuation_marks",
                "punctuation_marks": [".", ","],
                "exception_marks": ["...", "!", "?"]
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
        "tagger_rules": processor.create_word_tagger_rules()
    }

    # Ajouter les effets d'emoji si activés
    if use_emojis:
        template["effects"].extend([
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
        ])

    return template


def get_enriched_css(style: Optional[SubtitleStyle] = None, custom_colors: Optional[Dict[str, str]] = None) -> str:
    """
    Fonction utilitaire pour obtenir le CSS enrichi.

    Args:
        style: Configuration du style
        custom_colors: Dictionnaire de couleurs personnalisées
    """
    if style is None:
        style = SubtitleStyle()

    # Appliquer les couleurs personnalisées
    if custom_colors:
        _apply_custom_colors(style, custom_colors)
        if 'theme' in custom_colors and custom_colors['theme']:
            style.theme = custom_colors['theme']

    processor = EnrichedSubtitleProcessor(style=style)
    return processor.generate_enhanced_css()


if __name__ == "__main__":
    # Test du processeur
    processor = EnrichedSubtitleProcessor()

    test_words = [
        {"word": "Incroyable", "start": 0.0, "end": 0.5},
        {"word": "découverte", "start": 0.5, "end": 1.0},
        {"word": "90%", "start": 1.0, "end": 1.3},
        {"word": "des", "start": 1.3, "end": 1.5},
        {"word": "gens", "start": 1.5, "end": 1.8},
        {"word": "ne", "start": 1.8, "end": 2.0},
        {"word": "savent", "start": 2.0, "end": 2.3},
        {"word": "pas", "start": 2.3, "end": 2.5},
        {"word": "ça", "start": 2.5, "end": 2.7},
        {"word": "!", "start": 2.7, "end": 2.8},
    ]

    enriched = processor.process_words(test_words)

    print("Mots enrichis:")
    for w in enriched:
        print(f"  '{w.word}' -> {w.importance} (color: {w.color}, size: {w.size_multiplier:.2f})")

    print("\nCSS généré (extrait):")
    css = processor.generate_enhanced_css()
    print(css[:500] + "...")
