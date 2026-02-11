"""
Module d'analyse intelligente 100% locale
Analyse le contenu transcrit pour identifier les moments à fort potentiel viral
Utilise Phi-4-mini via llama.cpp (aucune API cloud)

Phi-4-mini est un modèle 3.8B paramètres optimisé pour le raisonnement.
Il utilise le même format de prompt que Phi-3:
<|system|>...<|end|><|user|>...<|end|><|assistant|>
"""

import os
import json
import re
from typing import List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console

from .ai_prompts import GLOBAL_PROMPT_TEMPLATE, PROMPT_TEMPLATE
from .moment_processing import (
    ViralMomentAI, TranscriptSegment,
    parse_global_response, snap_to_sentence_boundaries,
    parse_response, create_moment, validate_moments
)

console = Console()

# Constantes de configuration
DEFAULT_MIN_CLIP_DURATION: float = 30.0
DEFAULT_MAX_CLIP_DURATION: float = 90.0
DEFAULT_MAX_CLIPS: int = 5
DEFAULT_MIN_VIRAL_SCORE: float = 0.70
CLIPS_PER_DURATION_RATIO: int = 180  # 1 clip toutes les 3 minutes
SECTION_DURATION_SECONDS: int = 900  # 15 minutes par section
LLM_SECTION_MAX_TOKENS: int = 800
LLM_SECTION_TEMPERATURE: float = 0.2
LLM_SECTION_TOP_P: float = 0.9
LLM_SEGMENT_MAX_TOKENS: int = 200
LLM_SEGMENT_TEMPERATURE: float = 0.1
MAX_LLM_RETRIES: int = 3
HOOK_DISPLAY_MAX_LENGTH: int = 100
SEGMENT_TEXT_MAX_LENGTH: int = 800


def _create_progress_bar():
    """Crée une barre de progression Rich standard pour l'analyse IA."""
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
        BarColumn, TaskProgressColumn, TimeRemainingColumn
    )
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    )


class LocalAIViralAnalyzer:
    """
    Analyse de contenu viral 100% locale avec Phi-4-mini via llama.cpp.

    Pas d'API cloud, pas de clé requise, fonctionne offline.
    Optimisé pour segments vidéo courts (30-90s).

    Phi-4-mini offre de meilleures performances de raisonnement que Phi-3,
    avec le même format de prompt.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        min_clip_duration: float = DEFAULT_MIN_CLIP_DURATION,
        max_clip_duration: float = DEFAULT_MAX_CLIP_DURATION,
        max_clips: int = DEFAULT_MAX_CLIPS,  # Réduit de 10 à 5 pour éviter trop de clips
        min_viral_score: float = DEFAULT_MIN_VIRAL_SCORE,  # Seuil minimum de qualité
        content_type: str = "unknown",  # Type de contenu pour adapter l'analyse
    ):
        """
        Initialise l'analyseur local.

        Args:
            model_path: Chemin vers le modèle GGUF (auto-detect par défaut)
            n_threads: Nombre de threads CPU
            min_clip_duration: Durée minimum des clips
            max_clip_duration: Durée maximum des clips
            max_clips: Nombre maximum de clips (défaut: 5)
            min_viral_score: Score minimum pour qu'un moment soit retenu (défaut: 0.70)
            content_type: Type de contenu (podcast, interview, comedy, etc.)
        """
        from .local_llm import LocalLLM

        self.llm = LocalLLM(model_path=model_path, n_threads=n_threads)
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.max_clips = max_clips
        self.min_viral_score = min_viral_score
        self.content_type = content_type.lower()

        # Types de contenu "calmes" où l'excitation audio n'est pas pertinente
        self.calm_content_types = {"podcast", "interview", "tutorial", "news", "motivational"}

    def analyze(
        self,
        segments: List[TranscriptSegment],
        video_duration: float,
        video_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[ViralMomentAI]:
        """
        Analyse les segments pour identifier les moments viraux.

        ✨ NOUVELLE APPROCHE RAPIDE: 1 seule analyse globale au lieu de N analyses individuelles.

        Args:
            segments: Segments de transcription avec timestamps
            video_duration: Durée totale de la vidéo
            video_path: Chemin vers la vidéo (optionnel)
            progress_callback: Callback optionnel (progress: float 0-1, message: str)

        Returns:
            Liste des moments viraux, triés par score
        """
        if not segments:
            console.print("[yellow]Aucune transcription à analyser[/yellow]")
            return []

        console.print("[cyan]Analyse locale du contenu (Phi-4-mini)...[/cyan]")

        # 🎯 NOUVELLE STRATÉGIE: Découper en sections de ~15 minutes
        # Pour chaque section → analyse indépendante → 2-3 meilleurs moments
        # Limiter le nombre total en fonction de la durée

        SECTION_DURATION = SECTION_DURATION_SECONDS

        # Calculer un max_clips intelligent basé sur la durée
        # Règle: ~1 clip par 3-4 minutes de vidéo, minimum 1, maximum self.max_clips
        smart_max_clips = max(1, min(self.max_clips, int(video_duration / CLIPS_PER_DURATION_RATIO)))  # 1 clip / 3 min
        console.print(f"[dim]Durée: {video_duration/60:.1f}min → max {smart_max_clips} clips[/dim]")

        # Découper la vidéo en sections de 15 minutes
        sections = self._split_into_sections(segments, video_duration, SECTION_DURATION)
        total_sections = len(sections)

        console.print(f"[cyan]📹 Vidéo découpée en {total_sections} sections de ~15min[/cyan]")

        all_moments = []

        with _create_progress_bar() as progress:
            task = progress.add_task("Analyse par sections...", total=total_sections)

            for i, section in enumerate(sections):
                try:
                    if progress_callback:
                        pct = i / total_sections
                        progress_callback(pct, f"Section {i+1}/{total_sections}")

                    # Analyser cette section (demande 3-5 moments)
                    section_moments = self._analyze_section(section, i+1, total_sections)
                    all_moments.extend(section_moments)

                    console.print(f"[dim]Section {i+1}/{total_sections}: {len(section_moments)} moments trouvés[/dim]")

                except Exception as e:
                    console.print(f"[yellow]⚠️ Erreur section {i+1}: {e}[/yellow]")
                    continue
                finally:
                    progress.update(task, advance=1)

        if progress_callback:
            progress_callback(1.0, f"Analyse terminée: {len(all_moments)} moments trouvés")

        console.print(f"[green]✅ {len(all_moments)} moments viraux détectés au total[/green]")

        # Trier par score et valider (utiliser smart_max_clips)
        all_moments.sort(key=lambda m: m.score, reverse=True)
        validated = validate_moments(
            all_moments, video_duration,
            max_clips=self.max_clips,
            min_duration=self.min_clip_duration,
            max_duration=self.max_clip_duration,
            min_viral_score=self.min_viral_score,
            max_clips_override=smart_max_clips
        )

        console.print(f"[cyan]📊 Meilleurs clips retenus: {len(validated)}/{len(all_moments)}[/cyan]")

        return validated

    def _build_timestamped_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Construit une transcription avec timestamps pour l'analyse globale."""
        lines = []
        for seg in segments:
            mins = int(seg.start // 60)
            secs = int(seg.start % 60)
            timestamp = f"[{mins:02d}:{secs:02d}]"
            lines.append(f"{timestamp} {seg.text}")
        return "\n".join(lines)

    def _split_into_sections(
        self,
        segments: List[TranscriptSegment],
        video_duration: float,
        section_duration: float = 900
    ) -> List[List[TranscriptSegment]]:
        """
        Découpe les segments en sections temporelles de ~15 minutes.

        Args:
            segments: Tous les segments de transcription
            video_duration: Durée totale de la vidéo
            section_duration: Durée cible d'une section (900s = 15min)

        Returns:
            Liste de sections, chaque section = liste de segments
        """
        sections = []
        current_section = []
        section_start = 0.0

        for seg in segments:
            # Si le segment dépasse la limite de temps, créer une nouvelle section
            if seg.start >= section_start + section_duration and current_section:
                sections.append(current_section)
                current_section = []
                section_start = seg.start

            current_section.append(seg)

        # Ajouter la dernière section
        if current_section:
            sections.append(current_section)

        return sections

    def _analyze_section(
        self,
        section_segments: List[TranscriptSegment],
        section_num: int,
        total_sections: int
    ) -> List[ViralMomentAI]:
        """
        Analyse une section de ~15 minutes et retourne 3-5 meilleurs moments.
        Réessaie jusqu'à 3 fois si le JSON est invalide.

        Args:
            section_segments: Segments de cette section
            section_num: Numéro de la section (pour affichage)
            total_sections: Nombre total de sections

        Returns:
            Liste de 3-5 moments viraux de cette section
        """
        if not section_segments:
            return []

        # Construire la transcription de cette section
        section_transcript = self._build_timestamped_transcript(section_segments)
        section_start = section_segments[0].start
        section_end = section_segments[-1].end
        section_duration = section_end - section_start

        # Adapter les critères selon le type de contenu
        is_calm_content = self.content_type in self.calm_content_types

        if is_calm_content:
            # === PROMPT POUR CONTENU CALME (podcast, interview, tutorial) ===
            # L'excitation audio n'est PAS un critère - on cherche la VALEUR du message
            criteria_text = """CRITÈRES POUR CONTENU CONVERSATIONNEL:
- Idée forte ou conseil actionnable (pas besoin d'excitation)
- Message clair et autonome (compréhensible seul)
- Point de vue intéressant ou contre-intuitif
- Révélation, anecdote marquante ou moment de vérité
- Phrase quotable ou mémorable

NE PAS chercher:
- Les moments "excités" ou à haute énergie
- Les rires ou réactions bruyantes
- L'intensité vocale (non pertinent pour ce type de contenu)"""
        else:
            # === PROMPT POUR CONTENU EXCITÉ (comedy, gaming, action) ===
            criteria_text = """CRITÈRES STRICTS:
- Accroche TRÈS forte dès la première phrase (capter l'attention en 3s)
- Émotion intense (humour fort, surprise majeure, tension palpable)
- Message complet et autonome (compréhensible hors contexte)
- Début ET fin sur des limites de phrases"""

        # Créer le prompt pour cette section
        prompt = f"""<|system|>
Tu es un expert en contenu viral pour TikTok/Reels/Shorts. Tu identifies les meilleurs moments d'une section de vidéo.
Réponds UNIQUEMENT avec un tableau JSON, rien d'autre.<|end|>
<|user|>
Voici la transcription de la SECTION {section_num}/{total_sections} (durée: {section_duration:.0f}s):

{section_transcript}

TYPE DE CONTENU: {self.content_type.upper()}

MISSION:
Identifie les 2 à 3 MEILLEURS moments de cette section (30-90s chacun).
Sois TRÈS SÉLECTIF - ne retiens que les moments vraiment pertinents.

RÈGLE CRITIQUE - HOOK (les 3 premières secondes):
Le clip DOIT commencer AU DÉBUT d'une phrase accrocheuse, JAMAIS en plein milieu.
Exemples de BONS hooks:
- Question rhétorique: "Est-ce que vous saviez que...?" / "Pourquoi personne ne parle de...?"
- Affirmation choc: "C'est la pire erreur que font 90% des gens"
- Interpellation: "Attendez, vous allez pas croire ça"
- Promesse de valeur: "Voici 3 secrets que personne ne vous dit"
- Début d'histoire: "Il y a 2 ans, j'ai découvert quelque chose..."

Exemples de MAUVAIS hooks (à éviter):
- "...et donc voilà pourquoi" (commence au milieu)
- "Euh... donc..." (hésitation)
- "Ouais c'est ça" (réponse sans contexte)

RÈGLE CRITIQUE - FIN DU CLIP:
Le clip DOIT finir À LA FIN d'une phrase, JAMAIS au milieu.
Cherche une ponctuation (. ! ?) ou une pause naturelle.

{criteria_text}

INSTRUCTIONS:
1. Analyse TOUTE cette section
2. Repère les 2-3 moments les plus pertinents uniquement
3. Pour chaque moment: vérifie que start = DÉBUT d'une phrase accrocheuse
4. Pour chaque moment: vérifie que end = FIN d'une phrase
5. Score: 0.85+ = excellent, 0.70-0.85 = bon potentiel, <0.70 = ne pas inclure

FORMAT JSON EXACT (tableau de 2-3 moments max):
[
  {{"start": 15, "end": 65, "score": 0.88, "hook": "La phrase d'accroche complète qui débute le clip", "emotion": "insight", "reason": "explication courte"}},
  {{"start": 120, "end": 180, "score": 0.75, "hook": "Autre phrase d'accroche", "emotion": "conseil", "reason": "pourquoi pertinent"}}
]

IMPORTANT: Le champ "hook" doit contenir la VRAIE première phrase du clip (copiée de la transcription).

Retourne UNIQUEMENT le tableau JSON (2-3 moments MAX), rien d'autre.<|end|>
<|assistant|>
"""

        # Système de retry (max 3 tentatives)
        max_retries = MAX_LLM_RETRIES
        for attempt in range(max_retries):
            try:
                # Générer avec le LLM
                response = self.llm.generate(
                    prompt,
                    max_tokens=LLM_SECTION_MAX_TOKENS,  # 3-5 moments = ~500-700 tokens
                    temperature=LLM_SECTION_TEMPERATURE,  # Un peu de créativité pour varier les sélections
                    top_p=LLM_SECTION_TOP_P,
                    stop=["<|end|>", "\n\n\n"]
                )

                # Parser la réponse
                moments = parse_global_response(
                    response.text, section_segments,
                    min_duration=self.min_clip_duration,
                    max_duration=self.max_clip_duration
                )

                # Si on a réussi à parser au moins 1 moment, c'est bon
                if moments:
                    return moments

                # Si aucun moment mais pas d'exception, retry
                if attempt < max_retries - 1:
                    continue
                else:
                    return []

            except Exception as e:
                # En cas d'erreur JSON, retry silencieusement
                if attempt < max_retries - 1:
                    continue
                else:
                    # Dernière tentative échouée, retourner liste vide (pas de message d'erreur)
                    return []

        return []

    def _analyze_segment_by_segment(
        self,
        segments: List[TranscriptSegment],
        video_duration: float,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[ViralMomentAI]:
        """Ancien système d'analyse segment par segment (fallback pour vidéos très longues)."""
        console.print("[yellow]Mode segment par segment (lent)[/yellow]")

        combined_segments = self._combine_segments(segments)
        total_to_analyze = len(combined_segments)
        console.print(f"[dim]{total_to_analyze} segments à analyser[/dim]")

        moments = []
        with _create_progress_bar() as progress:
            task = progress.add_task("Analyse IA...", total=total_to_analyze)

            for i, seg in enumerate(combined_segments):
                try:
                    moment = self._analyze_segment(seg, video_duration, i + 1, total_to_analyze)
                    if moment:
                        moments.append(moment)
                except Exception as e:
                    console.print(f"[yellow]Erreur segment {i+1}: {e}[/yellow]")
                    continue
                finally:
                    progress.update(task, advance=1, description=f"Segment {i+1}/{total_to_analyze}")

                    if progress_callback:
                        try:
                            pct = (i + 1) / total_to_analyze
                            progress_callback(pct, f"Analyse de la vidéo ({int(pct * 100)}%)")
                        except Exception:
                            pass

        # Trier par score et garder les meilleurs
        console.print(f"[green]✅ Analyse terminée: {len(moments)} moments trouvés[/green]")
        moments.sort(key=lambda m: m.score, reverse=True)

        # Valider et supprimer les chevauchements
        validated = validate_moments(
            moments, video_duration,
            max_clips=self.max_clips,
            min_duration=self.min_clip_duration,
            max_duration=self.max_clip_duration,
            min_viral_score=self.min_viral_score
        )

        console.print(f"[cyan]📊 Meilleurs clips retenus: {len(validated)}/{len(moments)}[/cyan]")
        return validated

    def _combine_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Combine les petits segments en blocs de 30-90s pour analyse."""
        if not segments:
            return []

        combined = []
        target_duration = (self.min_clip_duration + self.max_clip_duration) / 2

        i = 0
        while i < len(segments):
            current_start = segments[i].start
            current_texts = []
            current_end = segments[i].end
            j = i

            while j < len(segments):
                current_texts.append(segments[j].text)
                current_end = segments[j].end
                duration = current_end - current_start

                if duration >= self.min_clip_duration:
                    break
                j += 1

            if current_texts:
                combined.append(TranscriptSegment(
                    start=current_start,
                    end=current_end,
                    text=" ".join(current_texts)
                ))

            # Avancer sans chevauchement pour analyser toute la vidéo
            i = j + 1 if j < len(segments) - 1 else len(segments)

        return combined


    def _analyze_segment(
        self,
        segment: TranscriptSegment,
        video_duration: float,
        index: int,
        total: int
    ) -> Optional[ViralMomentAI]:
        """Analyse un segment unique avec le LLM local"""

        duration = segment.end - segment.start

        # Limiter le texte pour performance (mais garder assez de contexte)
        text = segment.text[:SEGMENT_TEXT_MAX_LENGTH]

        prompt = PROMPT_TEMPLATE.format(
            start=segment.start,
            end=segment.end,
            duration=duration,
            text=text
        )

        # Générer avec le LLM
        response = self.llm.generate(
            prompt,
            max_tokens=LLM_SEGMENT_MAX_TOKENS,
            temperature=LLM_SEGMENT_TEMPERATURE,  # Plus déterministe
            stop=["<|end|>", "\n\n", "```"]
        )

        # Parser la réponse JSON
        return parse_response(response.text, segment)


def analyze_with_ai(
    segments: List[TranscriptSegment],
    video_duration: float,
    min_duration: float = DEFAULT_MIN_CLIP_DURATION,
    max_duration: float = DEFAULT_MAX_CLIP_DURATION,
    max_clips: int = DEFAULT_MAX_CLIPS,
    min_viral_score: float = DEFAULT_MIN_VIRAL_SCORE,
    model_path: Optional[str] = None,
    video_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    content_type: str = "unknown"
) -> List[ViralMomentAI]:
    """
    Fonction utilitaire pour analyser avec l'IA locale.

    Utilise le LLM local (Phi-4-mini ou Phi-3-mini) pour détecter les moments viraux.
    100% offline, aucune API cloud requise.

    Args:
        segments: Segments de transcription
        video_duration: Durée de la vidéo
        min_duration: Durée min des clips
        max_duration: Durée max des clips
        max_clips: Nombre max de clips (défaut: 5)
        min_viral_score: Score minimum pour retenir un moment (défaut: 0.70)
        model_path: Chemin vers le modèle (optionnel, auto-detect)
        video_path: Chemin vers la vidéo (optionnel)
        progress_callback: Callback optionnel (progress: float 0-1, message: str)
        content_type: Type de contenu pour adapter l'analyse (podcast, interview, comedy, etc.)

    Returns:
        Liste des moments viraux
    """
    try:
        console.print("[dim]Initialisation du LLM local...[/dim]")
        analyzer = LocalAIViralAnalyzer(
            model_path=model_path,
            min_clip_duration=min_duration,
            max_clip_duration=max_duration,
            max_clips=max_clips,
            min_viral_score=min_viral_score,
            content_type=content_type
        )
        return analyzer.analyze(
            segments,
            video_duration,
            video_path=video_path,
            progress_callback=progress_callback
        )
    except Exception as e:
        console.print(f"[yellow]LLM local indisponible: {e}[/yellow]")
        console.print("[yellow]Utilisation du fallback audio/video...[/yellow]")
        return []


# Alias pour compatibilité
AIViralAnalyzer = LocalAIViralAnalyzer


if __name__ == "__main__":
    # Test
    test_segments = [
        TranscriptSegment(0, 10, "Bonjour à tous, aujourd'hui je vais vous révéler un secret incroyable"),
        TranscriptSegment(10, 20, "que personne ne connaît sur comment devenir riche rapidement"),
        TranscriptSegment(20, 35, "J'ai découvert cette méthode par hasard il y a 2 ans et ça a changé ma vie"),
        TranscriptSegment(35, 50, "La première étape c'est de comprendre comment fonctionne l'algorithme"),
        TranscriptSegment(50, 65, "Ensuite vous devez appliquer cette technique tous les jours"),
    ]

    moments = analyze_with_ai(test_segments, 300, min_duration=30, max_duration=60)
    for m in moments:
        print(f"[{m.start_time:.0f}s-{m.end_time:.0f}s] Score: {m.score:.2f} - {m.emotion}: {m.reason}")
