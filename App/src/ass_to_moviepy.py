"""
Convertisseur ASS vers MoviePy TextClip
Fallback temporaire en attendant FFmpeg avec libass
"""

import pysubs2
from moviepy import TextClip, CompositeVideoClip
from typing import List, Optional
from pathlib import Path
from rich.console import Console

console = Console()

# Constantes de configuration
DEFAULT_VIDEO_WIDTH: int = 1080
DEFAULT_VIDEO_HEIGHT: int = 1920
DEFAULT_FONT_SIZE: int = 56
STROKE_WIDTH: int = 3
HORIZONTAL_MARGIN: int = 100
BOTTOM_MARGIN: int = 150
MS_PER_SECOND: float = 1000.0


def parse_ass_color(ass_color) -> str:
    """
    Convertit une couleur ASS BGR (int ou string) en RGB hex (#RRGGBB).

    Args:
        ass_color: Couleur ASS (int comme 16777215 ou string comme &H00FFFFFF&)

    Returns:
        Couleur au format RGB hex (ex: #FFFFFF)
    """
    # pysubs2 retourne un int (ex: 16777215 pour blanc)
    if isinstance(ass_color, int):
        # Format: 0xAABBGGRR (alpha, blue, green, red)
        r = (ass_color & 0x000000FF)
        g = (ass_color & 0x0000FF00) >> 8
        b = (ass_color & 0x00FF0000) >> 16
        return f'#{r:02X}{g:02X}{b:02X}'

    # Fallback pour format string
    if isinstance(ass_color, str) and ass_color.startswith('&H'):
        hex_part = ass_color.replace('&H', '').replace('&', '')
        if len(hex_part) >= 6:
            b = hex_part[-6:-4]
            g = hex_part[-4:-2]
            r = hex_part[-2:]
            return f'#{r.upper()}{g.upper()}{b.upper()}'

    return '#FFFFFF'


def strip_ass_tags(text: str) -> str:
    """
    Enlève les tags ASS d'un texte (karaoke, animations, etc.).

    Args:
        text: Texte avec tags ASS

    Returns:
        Texte nettoyé
    """
    import re
    # Enlever les tags entre {}
    cleaned = re.sub(r'\{[^}]*\}', '', text)
    return cleaned.strip()


def render_ass_to_moviepy(
    ass_file: str,
    video_width: int = DEFAULT_VIDEO_WIDTH,
    video_height: int = DEFAULT_VIDEO_HEIGHT,
    font_path: Optional[str] = None
) -> List[TextClip]:
    """
    Convertit un fichier ASS en liste de TextClip MoviePy.

    Args:
        ass_file: Chemin vers le fichier .ass
        video_width: Largeur de la vidéo (défaut: 1080)
        video_height: Hauteur de la vidéo (défaut: 1920)
        font_path: Chemin vers la police (optionnel)

    Returns:
        Liste de TextClip à overlayer sur la vidéo
    """
    if not Path(ass_file).exists():
        console.print(f"[yellow]⚠ Fichier ASS introuvable: {ass_file}[/yellow]")
        return []

    try:
        # Charger le fichier ASS
        subs = pysubs2.load(ass_file)

        # Obtenir le style par défaut
        default_style = subs.styles.get('Normal') or subs.styles.get('Default')

        if not default_style:
            console.print("[yellow]⚠ Aucun style trouvé dans le fichier ASS[/yellow]")
            return []

        # Extraire les propriétés du style
        font_name_ass = default_style.fontname or 'Arial'

        # MoviePy/Pillow nécessite soit un chemin de font, soit un nom basique
        # Sur Mac, utiliser les polices système par défaut
        font_name = 'Arial'  # Fallback simple qui fonctionne avec ImageMagick

        font_size = int(default_style.fontsize) if default_style.fontsize else DEFAULT_FONT_SIZE
        primary_color = parse_ass_color(default_style.primarycolor)

        # Créer les TextClips
        text_clips = []

        for event in subs.events:
            # Ignorer les événements vides
            if not event.text or event.text.strip() == '':
                continue

            # Nettoyer le texte des tags ASS
            clean_text = strip_ass_tags(event.text)

            if not clean_text:
                continue

            # Convertir les timestamps (millisecondes -> secondes)
            start_time = event.start / MS_PER_SECOND
            end_time = event.end / MS_PER_SECOND
            duration = end_time - start_time

            if duration <= 0:
                continue

            try:
                # Créer le TextClip (MoviePy 2.x: font est le 1er param, text le 2e)
                text_clip = TextClip(
                    font_name,  # 1er param: font
                    text=clean_text,  # 2e param: text
                    font_size=font_size,
                    color=primary_color,
                    stroke_color='black',
                    stroke_width=STROKE_WIDTH,
                    method='caption',
                    size=(video_width - HORIZONTAL_MARGIN, None),  # Largeur max avec marges
                    text_align='center'
                )

                # Positionner en bas (style TikTok)
                margin_bottom = BOTTOM_MARGIN
                text_clip = text_clip.with_position(('center', video_height - margin_bottom - text_clip.h))

                # Définir le timing
                text_clip = text_clip.with_start(start_time).with_duration(duration)

                text_clips.append(text_clip)

            except Exception as e:
                console.print(f"[yellow]⚠ Erreur création TextClip pour '{clean_text}': {e}[/yellow]")
                continue

        console.print(f"[green]✓ {len(text_clips)} sous-titres convertis depuis ASS[/green]")
        return text_clips

    except Exception as e:
        console.print(f"[red]Erreur lecture ASS: {e}[/red]")
        return []


def add_subtitles_moviepy(
    video_clip,
    ass_file: str
) -> CompositeVideoClip:
    """
    Ajoute des sous-titres ASS à un clip vidéo MoviePy.

    Args:
        video_clip: VideoFileClip MoviePy
        ass_file: Chemin vers le fichier .ass

    Returns:
        CompositeVideoClip avec sous-titres
    """
    # Obtenir les dimensions de la vidéo
    video_width, video_height = video_clip.size

    # Convertir ASS en TextClips
    subtitle_clips = render_ass_to_moviepy(
        ass_file,
        video_width=video_width,
        video_height=video_height
    )

    if not subtitle_clips:
        console.print("[yellow]⚠ Aucun sous-titre à ajouter[/yellow]")
        return video_clip

    # Créer le composite
    final_clip = CompositeVideoClip([video_clip] + subtitle_clips)

    return final_clip


if __name__ == "__main__":
    # Test du module
    print("Test de conversion ASS -> MoviePy")

    test_ass = "/tmp/clipgenius_test.ass"

    if Path(test_ass).exists():
        clips = render_ass_to_moviepy(test_ass)
        print(f"✓ {len(clips)} TextClips créés")

        for i, clip in enumerate(clips[:3]):  # Afficher les 3 premiers
            print(f"  Clip {i+1}: durée={clip.duration:.2f}s, start={clip.start:.2f}s")
    else:
        print("Fichier de test non trouvé")
