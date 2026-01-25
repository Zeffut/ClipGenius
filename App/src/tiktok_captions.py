"""
Module de génération de sous-titres ASS/SSA style TikTok avec pysubs2
Remplace pycaps pour les sous-titres animés avec mots-clés colorés
"""

import os
import pysubs2
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console

console = Console()

# Import du module de classification des mots
try:
    from .enriched_subtitles import EnrichedSubtitleProcessor, SubtitleStyle as EnrichedSubtitleStyle
    ENRICHED_AVAILABLE = True
except ImportError:
    ENRICHED_AVAILABLE = False


def rgb_to_ass_color(hex_color: str) -> str:
    """
    Convertit une couleur RGB hex (#RRGGBB) en couleur ASS (&H00BBGGRR).
    
    Args:
        hex_color: Couleur en hex RGB (ex: #FF0000)
    
    Returns:
        Couleur au format ASS &H00BBGGRR (sans & final pour pysubs2)
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # pysubs2 attend le format &HAABBGGRR (sans & final)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def create_ass_style(
    name: str,
    font_size: int,
    primary_color: str,
    font_name: str = "Poppins",
    bold: bool = True,
    border_width: int = 3,
    shadow_depth: int = 2,
    alignment: int = 2  # 2 = center bottom
) -> pysubs2.SSAStyle:
    """
    Crée un style ASS pour les sous-titres.
    
    Args:
        name: Nom du style
        font_size: Taille de la police
        primary_color: Couleur principale (hex RGB)
        font_name: Nom de la police
        bold: Police en gras
        border_width: Épaisseur du contour
        shadow_depth: Profondeur de l'ombre
        alignment: Position (1-9, numpad style)
    
    Returns:
        SSAStyle configuré
    """
    ass_color = rgb_to_ass_color(primary_color)
    
    return pysubs2.SSAStyle(
        fontname=font_name,
        fontsize=font_size,
        primarycolor=ass_color,
        secondarycolor=ass_color,
        outlinecolor=pysubs2.Color(0, 0, 0),  # Noir pour le contour
        backcolor=pysubs2.Color(0, 0, 0, 128),  # Noir semi-transparent pour l'ombre
        bold=bold,
        italic=False,
        underline=False,
        strikeout=False,
        scalex=100.0,
        scaley=100.0,
        spacing=0.5,
        angle=0.0,
        borderstyle=1,
        outline=border_width,
        shadow=shadow_depth,
        alignment=alignment,
        marginl=10,
        marginr=10,
        marginv=50  # Marge verticale depuis le bas
    )


def generate_tiktok_ass(
    words: List,
    preset_colors: Optional[Dict[str, str]],
    output_path: str,
    style_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Génère un fichier ASS avec sous-titres style TikTok.
    
    Args:
        words: Liste de WordTimestamp avec timestamps
        preset_colors: Dictionnaire de couleurs du preset (keyword_color, number_color, etc.)
        output_path: Chemin de sortie du fichier .ass
        style_config: Configuration additionnelle (theme, max_words, use_emojis)
        
    Returns:
        Chemin du fichier .ass généré
    """
    if not words:
        console.print("[yellow]⚠ Aucun mot à sous-titrer[/yellow]")
        return None
    
    # Configuration par défaut
    config = style_config or {}
    theme = config.get('theme', 'viral')
    max_words = config.get('max_words', 3)
    use_emojis = config.get('use_emojis', True)
    
    # Couleurs par défaut (TikTok viral style)
    colors = preset_colors or {}
    primary_color = colors.get('primary_color', '#FFFFFF')
    keyword_color = colors.get('keyword_color', '#FFD700')
    number_color = colors.get('number_color', '#00FFFF')
    emphasis_color = colors.get('emphasis_color', '#FF6B6B')
    highlight_color = colors.get('highlight_color', '#00FF88')
    
    # Configuration de police
    font_name = colors.get('font_family', 'Poppins')
    base_font_size = colors.get('base_font_size', 56)
    keyword_size_mult = colors.get('keyword_size_multiplier', 1.18)
    
    # Créer le fichier ASS
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"  # Résolution 9:16
    subs.info["PlayResY"] = "1920"
    
    # Créer les styles pour chaque type de mot
    subs.styles["Normal"] = create_ass_style(
        name="Normal",
        font_size=base_font_size,
        primary_color=primary_color,
        font_name=font_name
    )
    
    subs.styles["Keyword"] = create_ass_style(
        name="Keyword",
        font_size=int(base_font_size * keyword_size_mult),
        primary_color=keyword_color,
        font_name=font_name,
        bold=True
    )
    
    subs.styles["Number"] = create_ass_style(
        name="Number",
        font_size=int(base_font_size * 1.12),
        primary_color=number_color,
        font_name=font_name,
        bold=True
    )
    
    subs.styles["Emphasis"] = create_ass_style(
        name="Emphasis",
        font_size=base_font_size,
        primary_color=emphasis_color,
        font_name=font_name,
        bold=True
    )
    
    subs.styles["Highlight"] = create_ass_style(
        name="Highlight",
        font_size=base_font_size,
        primary_color=highlight_color,
        font_name=font_name,
        bold=True
    )
    
    # Classifier les mots avec enriched_subtitles si disponible
    if ENRICHED_AVAILABLE and colors:
        # Créer le style enrichi avec les couleurs du preset
        enriched_style = EnrichedSubtitleStyle()
        enriched_style.normal_color = primary_color
        enriched_style.keyword_color = keyword_color
        enriched_style.number_color = number_color
        enriched_style.emphasis_color = emphasis_color
        enriched_style.highlight_color = highlight_color
        enriched_style.base_font_size = base_font_size
        enriched_style.keyword_size_multiplier = keyword_size_mult
        
        processor = EnrichedSubtitleProcessor(style=enriched_style)
        
        # Convertir les mots en format dict pour le processeur
        words_dict = [{'word': w.word, 'start': w.start, 'end': w.end} for w in words]
        enriched_words = processor.process_words(words_dict)
    else:
        # Fallback: tous les mots en normal
        from dataclasses import dataclass
        
        @dataclass
        class SimpleEnrichedWord:
            word: str
            start: float
            end: float
            importance: str = 'normal'
            color: str = '#FFFFFF'
            size_multiplier: float = 1.0
            animation: Optional[str] = None
        
        enriched_words = [
            SimpleEnrichedWord(
                word=w.word,
                start=w.start,
                end=w.end,
                importance='normal',
                color=primary_color
            )
            for w in words
        ]
    
    # Grouper les mots en segments (max_words mots par segment)
    segments = []
    current_segment = []
    
    for enriched_word in enriched_words:
        current_segment.append(enriched_word)
        
        if len(current_segment) >= max_words:
            segments.append(current_segment)
            current_segment = []
    
    # Ajouter le dernier segment s'il reste des mots
    if current_segment:
        segments.append(current_segment)
    
    # Générer les événements ASS pour chaque segment
    for segment_words in segments:
        if not segment_words:
            continue
        
        # Timing du segment (du premier au dernier mot)
        start_time = int(segment_words[0].start * 1000)  # Millisecondes
        end_time = int(segment_words[-1].end * 1000)
        
        # Construire le texte du segment avec styles inline
        segment_text_parts = []
        
        for i, word in enumerate(segment_words):
            # Déterminer le style selon l'importance
            importance = word.importance
            
            # Mapper l'importance vers le nom de style
            style_map = {
                'keyword': 'Keyword',
                'number': 'Number',
                'emphasis': 'Emphasis',
                'normal': 'Normal'
            }
            style_name = style_map.get(importance, 'Normal')
            
            # Texte du mot (avec espaces)
            word_text = word.word
            if i < len(segment_words) - 1:
                word_text += " "
            
            # Animation karaoke + pop-in pour les mots importants
            if importance in ['keyword', 'number']:
                # Effet pop-in: scale de 120% puis retour à 100%
                # Karaoke: highlight progressif
                # Format: {\k<duration>\t(\fscx120\fscy120)\t(\fscx100\fscy100)}
                word_duration = int((word.end - word.start) * 100)  # Centisecondes
                word_text = f"{{\\k{word_duration}\\t(\\fscx120\\fscy120)\\t(\\fscx100\\fscy100)}}{word_text}"
            else:
                # Effet karaoke simple
                word_duration = int((word.end - word.start) * 100)
                word_text = f"{{\\k{word_duration}}}{word_text}"
            
            segment_text_parts.append(word_text)
        
        # Joindre tous les mots
        segment_text = "".join(segment_text_parts)
        
        # Ajouter le style par défaut du segment (Normal)
        # Les overrides inline changeront le style de mots spécifiques
        event = pysubs2.SSAEvent(
            start=start_time,
            end=end_time,
            text=segment_text,
            style="Normal"
        )
        
        subs.events.append(event)
    
    # Sauvegarder le fichier ASS
    subs.save(output_path)
    console.print(f"[green]✓ Fichier ASS généré: {Path(output_path).name} ({len(subs.events)} segments)[/green]")
    
    return output_path


if __name__ == "__main__":
    # Test du module
    from dataclasses import dataclass
    
    @dataclass
    class TestWord:
        word: str
        start: float
        end: float
    
    test_words = [
        TestWord("Incroyable", 0.0, 0.5),
        TestWord("découverte", 0.5, 1.0),
        TestWord("90%", 1.0, 1.3),
        TestWord("des", 1.3, 1.5),
        TestWord("gens", 1.5, 1.8),
        TestWord("ne", 1.8, 2.0),
        TestWord("savent", 2.0, 2.3),
        TestWord("pas", 2.3, 2.5),
        TestWord("ça", 2.5, 2.7),
    ]
    
    test_colors = {
        'primary_color': '#FFFFFF',
        'keyword_color': '#FF00FF',
        'number_color': '#00FFFF',
        'emphasis_color': '#FF6B6B'
    }
    
    output = "/tmp/test_tiktok.ass"
    generate_tiktok_ass(
        words=test_words,
        preset_colors=test_colors,
        output_path=output,
        style_config={'theme': 'gaming', 'max_words': 3, 'use_emojis': False}
    )
    
    print(f"Fichier test généré: {output}")
