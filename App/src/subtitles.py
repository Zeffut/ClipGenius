"""
Module de génération de sous-titres automatiques animés
Utilise pycaps pour générer des sous-titres style TikTok avec animations et émojis
"""

import os
import gc
import json
import time
import tempfile
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from rich.console import Console

# Imports depuis les modules extraits
from .transcription import (
    SubtitleSegment,
    WordTimestamp,
    TranscriptionResult,
    IS_APPLE_SILICON,
    MLX_WHISPER_AVAILABLE,
    transcribe_with_mlx,
    transcribe_with_openai_whisper,
    parse_whisper_result,
)
from .subtitle_styles import VIRAL_CSS, VIRAL_TEMPLATE

console = Console()

# Note: Les émojis sont générés localement via Phi-4-mini (100% offline)


# =============================================================================
# CUSTOM LLM PROVIDER - Émojis icônes uniquement (pas de visages)
# Utilise uniquement le LLM local Phi-4-mini (100% offline)
# =============================================================================


class IconOnlyLocalLLM:
    """
    Provider d'emojis avec Phi-3-mini local (100% offline).
    Alternative à gpt-5-nano pour les sous-titres animés.
    """

    SYSTEM_PROMPT = """You are an emoji selector for video subtitles.
Respond with ONLY a single emoji icon that matches the sentiment/context.
NEVER use face or hand emojis.
If no emoji fits, respond with exactly "None"."""

    def __init__(self):
        """Initialise le provider avec le LLM local"""
        try:
            from .local_llm import LocalLLM
            self.llm = LocalLLM()
            self._enabled = True
        except Exception as e:
            console.print(f"[dim]Emojis locaux indisponibles: {e}[/dim]")
            self._enabled = False

    def send_message(self, prompt: str) -> str:
        """Génère un emoji avec le LLM local"""
        if not self._enabled:
            return "None"

        try:
            # Prompt ultra court pour performance
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nText: {prompt}\nEmoji:"

            response = self.llm.generate(
                full_prompt,
                max_tokens=5,
                temperature=0.1,
                stop=["\n", " "]
            )

            emoji = response.text.strip()

            # Vérifier que c'est un emoji (très basique)
            if emoji and emoji != "None":
                return emoji[:1]  # Prendre le premier caractère (l'emoji)
            return "None"

        except Exception as e:
            console.print(f"[dim]Emoji génération échouée: {e}[/dim]")
            return "None"

    def is_enabled(self) -> bool:
        """Vérifie si le provider est disponible"""
        return self._enabled


def _setup_custom_emoji_provider():
    """Configure pycaps pour utiliser notre LLM local avec émojis icônes.

    Utilise uniquement le LLM local Phi-4-mini (100% offline).
    """
    try:
        from pycaps.ai import LlmProvider

        # Utiliser le LLM local uniquement (100% offline)
        local_provider = IconOnlyLocalLLM()
        if local_provider.is_enabled():
            LlmProvider.set(local_provider)
            console.print("[dim]Provider émojis local (Phi-4-mini) activé[/dim]")
            return

        console.print("[yellow]Provider emoji local non disponible[/yellow]")

    except ImportError:
        pass


# Configurer le provider personnalisé au chargement du module
_setup_custom_emoji_provider()

# Vérifier si pycaps est disponible
PYCAPS_AVAILABLE = False
try:
    from pycaps import CapsPipelineBuilder, TemplateLoader
    from pycaps.animation import FadeIn, PopIn, ZoomIn
    from pycaps.common import EventType, ElementType
    from pycaps.common.models import Document, Segment, Line, Word, TimeFragment
    from pycaps.transcriber.splitter import LimitByWordsSplitter
    from pycaps.transcriber.base_transcriber import AudioTranscriber
    from pycaps.layout.definitions import SubtitleLayoutOptions, VerticalAlignment
    from pycaps.effect import EmojiInSegmentEffect, AnimateSegmentEmojisEffect, RemovePunctuationMarksEffect
    from pycaps.effect.text.emoji_in_segment_effect import EmojiAlign
    PYCAPS_AVAILABLE = True
except ImportError:
    # pycaps non disponible - sous-titres animés TikTok-style désactivés
    # L'app utilisera des sous-titres simples à la place
    pass

# Import du module de sous-titres enrichis
try:
    from .enriched_subtitles import (
        EnrichedSubtitleProcessor,
        SubtitleStyle as EnrichedSubtitleStyle,
        get_enriched_css,
        create_enriched_subtitle_template
    )
    ENRICHED_SUBTITLES_AVAILABLE = True
except ImportError:
    ENRICHED_SUBTITLES_AVAILABLE = False


class PreTranscribedAudioTranscriber:
    """
    Transcriber personnalisé pour pycaps qui utilise des mots pré-transcrits.
    Évite de refaire la transcription Whisper pour chaque clip.
    """

    def __init__(self, words: List[WordTimestamp]):
        """
        Args:
            words: Liste de mots avec leurs timestamps (relatifs au clip, pas à la vidéo originale)
        """
        self.words = self._merge_contractions(words)

    @staticmethod
    def _merge_contractions(words: List[WordTimestamp]) -> List[WordTimestamp]:
        """Fusionne les contractions françaises pour éviter les coupures.
        Ex: ['L'', 'application'] → ['L'application']
        """
        import copy
        merged: List[WordTimestamp] = []
        i = 0
        while i < len(words):
            w = words[i]
            text = (w.word or '').rstrip()
            if (text.endswith("'") or text.endswith('\u2019')) and i + 1 < len(words):
                nw = words[i + 1]
                m = copy.copy(w)
                m.word = text + (nw.word or '').lstrip()
                m.end = nw.end
                merged.append(m)
                i += 2
            else:
                merged.append(w)
                i += 1
        return merged

    def transcribe(self, audio_path: str) -> 'Document':
        """
        Retourne un Document pycaps à partir des mots pré-transcrits.
        Le paramètre audio_path est ignoré car on a déjà la transcription.
        """
        if not PYCAPS_AVAILABLE:
            raise ImportError("pycaps n'est pas disponible")

        document = Document()

        if not self.words:
            return document

        # Créer un seul segment contenant tous les mots
        # pycaps va ensuite splitter selon la config (ex: 3 mots par segment)
        segment = Segment(time=TimeFragment(
            start=self.words[0].start if self.words else 0,
            end=self.words[-1].end if self.words else 0
        ))

        # Créer une ligne contenant tous les mots
        line = Line()

        for w in self.words:
            word = Word(
                text=w.word,
                time=TimeFragment(start=w.start, end=w.end)
            )
            line.words.add(word)  # ElementContainer utilise add() au lieu de append()

        segment.lines.add(line)  # ElementContainer utilise add() au lieu de append()
        document.segments.add(segment)  # ElementContainer utilise add() au lieu de append()

        return document


# Configuration du nombre de mots par segment (style TikTok = 2-3 mots max)
MAX_WORDS_PER_SEGMENT = 3

# Chemin vers le dossier fonts du projet
FONTS_DIR = Path(__file__).parent.parent / "fonts"


class AnimatedSubtitleGenerator:
    """
    Génère des sous-titres animés style TikTok avec pycaps
    Supporte les sous-titres enrichis avec mots-clés colorés
    """

    def __init__(
        self,
        template: str = "viral",
        whisper_model: str = "base",
        language: Optional[str] = None,
        use_enriched: bool = False
    ):
        """
        Args:
            template: Style de sous-titres ('viral', 'bold', 'minimalist', 'enriched')
            whisper_model: Taille du modèle Whisper ('tiny', 'base', 'small', 'medium', 'large', 'turbo')
            language: Code de langue (None = auto-détection)
            use_enriched: Utiliser les sous-titres enrichis avec mots-clés colorés
        """
        self.template = template
        self.whisper_model = whisper_model
        self.language = language
        self.detected_language = None
        self.use_enriched = use_enriched or template == "enriched"

    def add_animated_subtitles(
        self,
        video_path: str,
        output_path: str,
        max_words: int = MAX_WORDS_PER_SEGMENT,
        use_emojis: bool = True,
        pretranscribed_words: Optional[List[WordTimestamp]] = None,
        use_enriched: Optional[bool] = None,
        custom_colors: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ajoute des sous-titres animés à une vidéo avec pycaps

        Args:
            video_path: Chemin vers la vidéo source
            output_path: Chemin de sortie
            max_words: Nombre maximum de mots par segment (défaut: 3)
            use_emojis: Ajouter des émojis automatiques (défaut: True)
            pretranscribed_words: Mots pré-transcrits (évite de refaire Whisper)
            use_enriched: Utiliser les sous-titres enrichis (mots-clés colorés)
            custom_colors: Couleurs et styles personnalisés du preset

        Returns:
            Chemin vers la vidéo avec sous-titres
        """
        if not PYCAPS_AVAILABLE:
            console.print("[yellow]⚠ pycaps non disponible, sous-titres désactivés[/yellow]")
            # Copier la vidéo sans sous-titres vers la destination
            import shutil
            if video_path != output_path:
                shutil.copy2(video_path, output_path)
                console.print(f"[dim]Clip copié sans sous-titres: {output_path}[/dim]")
            return output_path

        # Déterminer si on utilise les sous-titres enrichis
        enriched = use_enriched if use_enriched is not None else self.use_enriched

        emoji_status = "avec émojis" if use_emojis else "sans émojis"
        transcription_status = "pré-transcrit" if pretranscribed_words else "Whisper"
        enriched_status = "enrichis" if enriched else "standard"
        console.print(
            f"[cyan]Ajout des sous-titres animés"
            f" ({enriched_status}, {max_words} mots/segment,"
            f" {emoji_status}, {transcription_status})...[/cyan]"
        )

        # Créer un dossier template temporaire
        temp_dir = tempfile.mkdtemp(prefix="pycaps_viral_")
        css_path = os.path.join(temp_dir, "styles.css")
        template_path = os.path.join(temp_dir, "pycaps.template.json")

        # Variable pour capturer les erreurs du thread
        thread_error = [None]

        def run_pycaps_pipeline():
            """Exécute le pipeline pycaps dans un thread séparé pour éviter les conflits asyncio/Playwright"""
            try:
                # Choisir le template et CSS selon le mode
                if enriched and ENRICHED_SUBTITLES_AVAILABLE:
                    # Utiliser le template enrichi avec mots-clés colorés
                    template_config = create_enriched_subtitle_template(
                        max_words=max_words,
                        use_emojis=use_emojis,
                        custom_colors=custom_colors
                    )
                    css_content = get_enriched_css(custom_colors=custom_colors)
                else:
                    # Template viral standard
                    template_config = VIRAL_TEMPLATE.copy()

                    # Ajuster le nombre de mots
                    template_config["splitters"] = [
                        {
                            "type": "limit_by_words",
                            "limit": max_words
                        }
                    ]

                    # Désactiver les émojis si demandé
                    if not use_emojis:
                        template_config["effects"] = [
                            e for e in template_config["effects"]
                            if e.get("type") not in ["emoji_in_segment", "animate_segment_emojis"]
                        ]

                    css_content = VIRAL_CSS

                # Écrire le CSS
                with open(css_path, "w", encoding="utf-8") as f:
                    f.write(css_content)

                # Copier la police Poppins dans le dossier temporaire
                font_src = FONTS_DIR / "Poppins-SemiBold.ttf"
                if font_src.exists():
                    font_dst = os.path.join(temp_dir, "Poppins-SemiBold.ttf")
                    shutil.copy2(font_src, font_dst)

                # Écrire le template JSON
                with open(template_path, "w", encoding="utf-8") as f:
                    json.dump(template_config, f, indent=2)

                # Charger le template et construire le pipeline
                loader = TemplateLoader(temp_dir)
                loader.with_input_video(video_path)

                # Obtenir le builder pour personnaliser
                builder = loader.load(should_build_pipeline=False)

                # Utiliser le transcriber pré-transcrit ou Whisper
                if pretranscribed_words:
                    # Utiliser notre transcriber personnalisé
                    custom_transcriber = PreTranscribedAudioTranscriber(pretranscribed_words)
                    builder.with_custom_audio_transcriber(custom_transcriber)
                else:
                    # Configurer Whisper normalement
                    builder.with_whisper_config(
                        language=self.language,
                        model_size=self.whisper_model
                    )

                # Définir la sortie
                builder.with_output_video(output_path)

                # Construire et exécuter le pipeline
                pipeline = builder.build()
                pipeline.run()

            except Exception as e:
                thread_error[0] = e

        try:
            # Exécuter pycaps dans un thread séparé pour éviter:
            # "Playwright Sync API inside asyncio loop" error
            thread = threading.Thread(target=run_pycaps_pipeline)
            thread.start()
            thread.join(timeout=300)  # Timeout de 5 minutes

            if thread.is_alive():
                console.print(
                    "[yellow]⚠ Timeout du thread pycaps (5min),"
                    " la vidéo sera retournée sans sous-titres[/yellow]"
                )

            # Vérifier si une erreur s'est produite dans le thread
            if not thread.is_alive() and thread_error[0]:
                raise thread_error[0]

            console.print(f"[green]Sous-titres animés ajoutés![/green]")
            return output_path

        except Exception as e:
            console.print(f"[red]Erreur pycaps: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            console.print("[yellow]Retour à la vidéo sans sous-titres[/yellow]")
            # Copier la vidéo originale vers la sortie
            if not os.path.exists(output_path):
                shutil.copy2(video_path, output_path)
            return output_path

        finally:
            # Forcer le garbage collection pour libérer les handles de fichiers
            gc.collect()
            time.sleep(0.5)  # Petit délai pour laisser Windows libérer les fichiers

            # Nettoyer le dossier temporaire
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def add_animated_subtitles(
    video_path: str,
    output_path: Optional[str] = None,
    template: str = "viral",
    whisper_model: str = "base",
    language: Optional[str] = None,
    max_words: int = MAX_WORDS_PER_SEGMENT,
    use_emojis: bool = True,
    pretranscribed_words: Optional[List[WordTimestamp]] = None,
    use_enriched: bool = False,
    custom_colors: Optional[Dict[str, Any]] = None
) -> str:
    """
    Fonction utilitaire pour ajouter des sous-titres animés à une vidéo

    Args:
        video_path: Chemin vers la vidéo
        output_path: Chemin de sortie (auto-généré si non fourni)
        template: Style de sous-titres ('viral', 'bold', 'minimalist', 'enriched')
        whisper_model: Taille du modèle Whisper
        language: Langue (None = auto-détection)
        max_words: Nombre maximum de mots par segment (défaut: 3)
        use_emojis: Ajouter des émojis automatiques (défaut: True)
        pretranscribed_words: Mots pré-transcrits avec timestamps (évite de refaire Whisper)
        use_enriched: Utiliser les sous-titres enrichis avec mots-clés colorés (défaut: False)
        custom_colors: Couleurs et styles personnalisés du preset

    Returns:
        Chemin vers la vidéo avec sous-titres
    """
    if output_path is None:
        path = Path(video_path)
        output_path = str(path.parent / f"{path.stem}_captioned{path.suffix}")

    generator = AnimatedSubtitleGenerator(
        template=template,
        whisper_model=whisper_model,
        language=language,
        use_enriched=use_enriched or template == "enriched"
    )

    return generator.add_animated_subtitles(
        video_path, output_path,
        max_words=max_words,
        use_emojis=use_emojis,
        pretranscribed_words=pretranscribed_words,
        use_enriched=use_enriched,
        custom_colors=custom_colors
    )


# ============================================================================
# COMPATIBILITÉ AVEC L'ANCIEN CODE
# Ces classes et fonctions sont conservées pour ne pas casser le code existant
# ============================================================================

@dataclass
class SubtitleStyle:
    """Style des sous-titres (compatibilité)"""
    font: str = "Arial-Bold"
    fontsize: int = 60
    color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    bg_color: Optional[str] = None
    position: Tuple[str, str] = ("center", "bottom")
    margin_bottom: int = 150
    highlight_color: str = "yellow"
    highlight_words: Optional[List[str]] = None


class SubtitleGenerator:
    """
    Wrapper de compatibilité qui utilise pycaps en interne
    Conserve l'interface de l'ancien code pour ne rien casser
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        style: Optional[SubtitleStyle] = None
    ):
        self.model_size = model_size
        self.language = language
        self.style = style or SubtitleStyle()
        self.detected_language = None
        self._animated_generator = AnimatedSubtitleGenerator(
            template="viral",
            whisper_model=model_size,
            language=language
        )

    def transcribe(self, video_path: str) -> List[SubtitleSegment]:
        """
        Transcrit l'audio (compatibilité - utilise Whisper directement)
        Note: pycaps gère la transcription en interne, cette méthode est pour la compatibilité
        """
        result = self.transcribe_with_words(video_path)
        return result.segments

    def transcribe_with_words(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcrit l'audio avec word timestamps complets.
        Retourne un TranscriptionResult avec segments et mots individuels.

        Utilise automatiquement mlx-whisper sur Mac Apple Silicon pour une
        transcription jusqu'à 10x plus rapide.

        Args:
            video_path: Chemin vers la vidéo à transcrire
            progress_callback: Fonction de callback optionnelle appelée avec (progress_percent, message)
                               progress_percent: 0-100
                               message: description de l'étape en cours
        """
        def report_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)

        # Obtenir la durée de la vidéo pour estimer la progression
        video_duration = None
        try:
            from moviepy import VideoFileClip
            with VideoFileClip(video_path) as video:
                video_duration = video.duration
            report_progress(5, f"Durée vidéo: {video_duration:.0f}s")
        except Exception:
            pass

        # Transcription 100% locale: MLX Whisper (Mac Apple Silicon) > Whisper standard
        raw_result = None

        # 1. MLX Whisper sur Mac Apple Silicon (rapide et local)
        if MLX_WHISPER_AVAILABLE:
            raw_result = transcribe_with_mlx(
                video_path, self.model_size, self.language,
                video_duration, report_progress
            )

        # 2. Fallback final: Whisper standard (local)

        result = parse_whisper_result(raw_result, self.language, report_progress)
        self.detected_language = result.language

        return result

    def add_subtitles_to_video(
        self,
        video_path: str,
        output_path: str,
        segments: Optional[List[SubtitleSegment]] = None,
        start_offset: float = 0
    ) -> str:
        """
        Ajoute des sous-titres animés à une vidéo avec pycaps
        """
        # Utiliser pycaps pour les sous-titres animés
        return self._animated_generator.add_animated_subtitles(video_path, output_path)


def add_subtitles(
    video_path: str,
    output_path: Optional[str] = None,
    language: Optional[str] = None,
    model_size: str = "base"
) -> str:
    """
    Fonction utilitaire (compatibilité) - utilise maintenant pycaps
    """
    return add_animated_subtitles(
        video_path=video_path,
        output_path=output_path,
        template="viral",
        whisper_model=model_size,
        language=language
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = add_animated_subtitles(sys.argv[1])
        print(f"Vidéo avec sous-titres: {result}")
