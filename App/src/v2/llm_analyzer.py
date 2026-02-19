"""
Analyseur LLM 3 passes pour la detection de moments viraux.

Architecture en 3 passes :
  1. Segmentation en chapitres (identification des transitions thematiques)
  2. Scoring multi-dimensionnel (6 axes d'evaluation par moment)
  3. Optimisation des bornes (alignement sur les phrases et les hooks)

Supporte deux formats de modeles :
  - ChatML (Qwen2.5-7B) : <|im_start|>role\n...<|im_end|>
  - Phi (Phi-3/Phi-4-mini) : <|system|>...<|end|><|user|>...<|end|><|assistant|>

Fonctionne 100% hors-ligne via llama.cpp.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from ..local_llm import LocalLLM, LLMResponse
from .models import LLMChapter, LLMScoredMoment, TranscriptSegment

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3
CHAPTER_MAX_TOKENS: int = 1500
SCORING_MAX_TOKENS: int = 2400
BOUNDARY_MAX_TOKENS: int = 800
DEFAULT_TEMPERATURE: float = 0.3
CHATML_STOP: List[str] = ['<|im_end|>', '<|endoftext|>']
PHI_STOP: List[str] = ['<|end|>']

# Mots frequents utilises pour la detection de langue
_FRENCH_MARKERS: List[str] = [
    ' le ', ' la ', ' les ', ' de ', ' des ', ' du ', ' un ', ' une ',
    ' et ', ' est ', ' que ', ' qui ', ' dans ', ' pour ', ' avec ',
    ' sur ', ' pas ', ' ce ', ' cette ', ' nous ', ' vous ', ' ils ',
    " c'est ", " j'ai ", " l'", " n'", " qu'", " d'",
]

# Seuil minimum de marqueurs francais pour considerer le texte comme francais
_FRENCH_DETECTION_THRESHOLD: int = 5

# Nombre de caracteres a analyser pour la detection de langue
_LANGUAGE_SAMPLE_LENGTH: int = 500

# Nombre minimum et maximum de chapitres attendus
_MIN_CHAPTERS: int = 2
_MAX_CHAPTERS: int = 10

# Ecart de temperature entre chaque retry
_TEMPERATURE_RETRY_INCREMENT: float = 0.1


class LLMAnalyzer:
    """Analyseur LLM 3 passes pour detection de moments viraux.

    Utilise un LLM local (Qwen2.5-7B ou Phi-4-mini) pour :
    1. Segmenter la transcription en chapitres thematiques
    2. Identifier et scorer les moments les plus viraux
    3. Optimiser les bornes temporelles des clips
    """

    def __init__(
        self,
        model_format: str = 'auto',
        content_type: str = 'unknown',
        language: str = 'auto',
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Initialise l'analyseur LLM.

        Args:
            model_format: Format du modele — 'chatml' pour Qwen2.5,
                'phi' pour Phi-3/Phi-4-mini, 'auto' pour detection
                automatique basee sur le modele charge.
            content_type: Type de contenu pour adapter les prompts
                (ex: 'podcast', 'interview', 'tutorial').
            language: Langue des prompts — 'fr', 'en', ou 'auto'
                pour detection automatique.
            temperature: Temperature de base pour la generation.
            max_retries: Nombre de tentatives en cas d'echec.
        """
        self._content_type: str = content_type
        self._language: str = language
        self._temperature: float = temperature
        self._max_retries: int = max_retries
        self._llm: LocalLLM = LocalLLM()

        # Auto-detection du format de modele
        if model_format == 'auto':
            model_format = self._detect_model_format()

        if model_format not in ('chatml', 'phi'):
            logger.warning(
                "model_format '%s' non reconnu, fallback sur 'phi'", model_format
            )
            model_format = 'phi'

        self._model_format: str = model_format
        self._stop_tokens: List[str] = (
            CHATML_STOP if model_format == 'chatml' else PHI_STOP
        )

    def _detect_model_format(self) -> str:
        """Detecte le format du modele a partir du singleton LocalLLM.

        Verifie le nom du fichier modele charge pour determiner
        s'il s'agit d'un modele ChatML (Qwen) ou Phi.

        Returns:
            'chatml' pour Qwen2.5, 'phi' pour Phi-3/Phi-4-mini.
        """
        try:
            if LocalLLM._llm is not None and hasattr(LocalLLM._llm, 'model_path'):
                model_path = str(LocalLLM._llm.model_path).lower()
                if 'qwen' in model_path:
                    return 'chatml'
        except Exception:
            pass
        return 'phi'

    # ===================================================================
    # Passe 1 : Segmentation en chapitres
    # ===================================================================

    def analyze_chapters(
        self,
        segments: List[TranscriptSegment],
        video_duration: float,
    ) -> List[LLMChapter]:
        """Identifie les chapitres thematiques de la transcription.

        Demande au LLM de reperer les transitions de sujets et de regrouper
        les segments en chapitres coherents (3 a 8 par video).

        Args:
            segments: Segments de transcription avec timestamps.
            video_duration: Duree totale de la video en secondes.

        Returns:
            Liste de LLMChapter tries par ordre chronologique.
        """
        if not segments:
            console.print('[yellow]Aucun segment de transcription fourni[/yellow]')
            return []

        lang = self._resolve_language(segments)
        transcript_text = self._build_timestamped_transcript(segments)

        system_prompt = self._chapter_system_prompt(lang, video_duration)
        user_prompt = self._chapter_user_prompt(lang, transcript_text, video_duration)

        prompt = self._format_prompt(system_prompt, user_prompt)

        console.print('[cyan]Passe 1/3 : Segmentation en chapitres...[/cyan]')

        raw_response = self._generate_with_retry(
            prompt,
            max_tokens=CHAPTER_MAX_TOKENS,
        )

        if not raw_response:
            console.print(
                '[yellow]Passe 1 echouee — fallback sur un chapitre unique[/yellow]'
            )
            return [LLMChapter(
                start=0.0,
                end=video_duration,
                title='Full Video',
                summary='Chapitre unique (fallback)',
            )]

        chapters = self._parse_chapters(raw_response, video_duration)

        if not chapters:
            console.print(
                '[yellow]Parsing des chapitres echoue — '
                'fallback sur un chapitre unique[/yellow]'
            )
            return [LLMChapter(
                start=0.0,
                end=video_duration,
                title='Full Video',
                summary='Chapitre unique (fallback)',
            )]

        console.print(
            f'[green]Passe 1 terminee : {len(chapters)} chapitres identifies[/green]'
        )
        return chapters

    # ===================================================================
    # Passe 2 : Scoring multi-dimensionnel
    # ===================================================================

    def score_moments(
        self,
        chapters: List[LLMChapter],
        segments: List[TranscriptSegment],
        min_duration: float = 30.0,
        max_duration: float = 90.0,
    ) -> List[LLMScoredMoment]:
        """Score les moments viraux pour chaque chapitre.

        Pour chaque chapitre, demande au LLM d'identifier 1 a 2 moments
        et de les scorer sur 6 dimensions (hook, emotion, clarte, etc.).

        Args:
            chapters: Chapitres identifies en passe 1.
            segments: Segments de transcription complets.
            min_duration: Duree minimale d'un clip en secondes.
            max_duration: Duree maximale d'un clip en secondes.

        Returns:
            Liste de LLMScoredMoment tries par score composite decroissant.
        """
        if not chapters or not segments:
            return []

        lang = self._resolve_language(segments)
        all_moments: List[LLMScoredMoment] = []

        console.print(
            f'[cyan]Passe 2/3 : Scoring de {len(chapters)} chapitres...[/cyan]'
        )

        for i, chapter in enumerate(chapters):
            chapter_segments = self._segments_in_range(
                segments, chapter.start, chapter.end
            )
            if not chapter_segments:
                continue

            chapter_transcript = self._build_timestamped_transcript(chapter_segments)

            system_prompt = self._scoring_system_prompt(lang)
            user_prompt = self._scoring_user_prompt(
                lang, chapter, chapter_transcript, min_duration, max_duration
            )

            prompt = self._format_prompt(system_prompt, user_prompt)

            console.print(
                f'[dim]  Chapitre {i + 1}/{len(chapters)} : '
                f'{chapter.title}[/dim]'
            )

            raw_response = self._generate_with_retry(
                prompt,
                max_tokens=SCORING_MAX_TOKENS,
            )

            if not raw_response:
                logger.warning(
                    'Scoring echoue pour le chapitre "%s"', chapter.title
                )
                continue

            moments = self._parse_scored_moments(
                raw_response, chapter, min_duration, max_duration
            )
            all_moments.extend(moments)

        # Calculer les scores composites et trier
        for moment in all_moments:
            moment.compute_composite()

        all_moments.sort(key=lambda m: m.composite_score, reverse=True)

        console.print(
            f'[green]Passe 2 terminee : {len(all_moments)} moments scores[/green]'
        )
        return all_moments

    # ===================================================================
    # Passe 3 : Optimisation des bornes (optionnelle)
    # ===================================================================

    def optimize_boundaries(
        self,
        moments: List[LLMScoredMoment],
        segments: List[TranscriptSegment],
    ) -> List[LLMScoredMoment]:
        """Affine les bornes temporelles pour un decoupage propre.

        Aligne start/end sur les frontieres de phrases et s'assure que
        chaque clip commence par un hook fort.

        Args:
            moments: Moments scores issus de la passe 2.
            segments: Segments de transcription complets.

        Returns:
            Liste de LLMScoredMoment avec bornes optimisees.
        """
        if not moments or not segments:
            return moments

        lang = self._resolve_language(segments)

        console.print(
            f'[cyan]Passe 3/3 : Optimisation des bornes '
            f'({len(moments)} moments)...[/cyan]'
        )

        optimized: List[LLMScoredMoment] = []

        for i, moment in enumerate(moments):
            # Recuperer les segments autour du moment (marge de 10s)
            context_segments = self._segments_in_range(
                segments,
                max(0.0, moment.start - 10.0),
                moment.end + 10.0,
            )
            if not context_segments:
                optimized.append(moment)
                continue

            context_transcript = self._build_timestamped_transcript(context_segments)

            system_prompt = self._boundary_system_prompt(lang)
            user_prompt = self._boundary_user_prompt(
                lang, moment, context_transcript
            )

            prompt = self._format_prompt(system_prompt, user_prompt)

            raw_response = self._generate_with_retry(
                prompt,
                max_tokens=BOUNDARY_MAX_TOKENS,
            )

            if not raw_response:
                optimized.append(moment)
                continue

            refined = self._parse_boundary_response(raw_response, moment)
            optimized.append(refined)

        console.print('[green]Passe 3 terminee : bornes optimisees[/green]')
        return optimized

    # ===================================================================
    # Formatage des prompts
    # ===================================================================

    def _format_prompt(self, system: str, user: str) -> str:
        """Formate un prompt systeme + utilisateur selon le format du modele.

        Args:
            system: Contenu du message systeme.
            user: Contenu du message utilisateur.

        Returns:
            Prompt formate pret a etre envoye au LLM.
        """
        if self._model_format == 'chatml':
            return (
                f'<|im_start|>system\n{system}<|im_end|>\n'
                f'<|im_start|>user\n{user}<|im_end|>\n'
                f'<|im_start|>assistant\n'
            )
        else:
            # Format Phi-3/Phi-4
            return (
                f'<|system|>\n{system}<|end|>\n'
                f'<|user|>\n{user}<|end|>\n'
                f'<|assistant|>\n'
            )

    # ===================================================================
    # Generation avec retry
    # ===================================================================

    def _generate_with_retry(
        self,
        prompt: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Appelle le LLM avec retry et augmentation progressive de la temperature.

        A chaque tentative echouee, la temperature augmente de 0.1 pour
        encourager des reponses differentes.

        Args:
            prompt: Prompt formate complet.
            max_tokens: Nombre maximum de tokens a generer.
            temperature: Temperature de base (defaut: self._temperature).
            stop: Sequences d'arret (defaut: self._stop_tokens).

        Returns:
            Texte genere par le LLM, ou chaine vide si toutes les tentatives
            ont echoue.
        """
        if temperature is None:
            temperature = self._temperature
        if stop is None:
            stop = self._stop_tokens

        for attempt in range(self._max_retries):
            current_temp = min(
                temperature + attempt * _TEMPERATURE_RETRY_INCREMENT, 1.0
            )
            try:
                response: LLMResponse = self._llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=current_temp,
                    stop=stop,
                )

                text = response.text.strip()
                if text:
                    logger.debug(
                        'Generation reussie (tentative %d/%d, %d tokens)',
                        attempt + 1, self._max_retries, response.tokens_used,
                    )
                    return text

                logger.warning(
                    'Reponse vide du LLM (tentative %d/%d)',
                    attempt + 1, self._max_retries,
                )

            except Exception as exc:
                logger.warning(
                    'Erreur LLM (tentative %d/%d) : %s',
                    attempt + 1, self._max_retries, exc,
                )
                if attempt == self._max_retries - 1:
                    console.print(
                        f'[red]LLM : {self._max_retries} tentatives echouees[/red]'
                    )

        return ''

    # ===================================================================
    # Extraction JSON robuste
    # ===================================================================

    def _extract_json(self, text: str) -> Any:
        """Extrait des donnees JSON depuis la reponse du LLM.

        Strategies de fallback :
        1. json.loads direct sur le texte complet
        2. Recherche d'un bloc JSON entre ``` markers
        3. Regex pour extraire le premier [...] (tableau)
        4. Regex pour extraire le premier {...} (objet)
        5. Recuperation ligne par ligne des objets JSON individuels

        Args:
            text: Texte brut de la reponse LLM.

        Returns:
            Donnees JSON deserialisees (list ou dict), ou None si
            aucune strategie ne fonctionne.
        """
        if not text:
            return None

        # Strategie 1 : parsing direct
        try:
            result = json.loads(text)
            # Encapsuler un objet unique dans une liste pour uniformiser
            if isinstance(result, dict):
                return [result]
            return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategie 2 : extraire le contenu entre ```json ... ```
        code_block_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL
        )
        if code_block_match:
            try:
                result = json.loads(code_block_match.group(1).strip())
                if isinstance(result, dict):
                    return [result]
                return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategie 3 : premier tableau JSON [...]
        array_match = re.search(r'(\[[\s\S]*\])', text)
        if array_match:
            candidate = array_match.group(1)
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                # Tentative de reparation : supprimer les virgules pendantes
                cleaned = self._repair_json(candidate)
                try:
                    return json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategie 4 : premier objet JSON {...}
        object_match = re.search(r'(\{[\s\S]*\})', text)
        if object_match:
            candidate = object_match.group(1)
            try:
                result = json.loads(candidate)
                # Encapsuler un objet unique dans une liste pour uniformiser
                return [result] if isinstance(result, dict) else result
            except (json.JSONDecodeError, ValueError):
                cleaned = self._repair_json(candidate)
                try:
                    result = json.loads(cleaned)
                    return [result] if isinstance(result, dict) else result
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategie 5 : recuperation ligne par ligne des objets JSON
        recovered = self._recover_json_objects(text)
        if recovered:
            return recovered

        # Strategie 6 : JSON tronque par la limite de tokens
        # Chercher depuis le premier '[' et fermer les structures ouvertes
        array_start = text.find('[')
        if array_start != -1:
            candidate = self._close_truncated_json(
                self._repair_json(text[array_start:])
            )
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning('Impossible d\'extraire du JSON de la reponse LLM')
        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """Tente de reparer du JSON malformate.

        Corrige les problemes courants :
        - Virgules pendantes avant ] ou }
        - Guillemets simples au lieu de doubles
        - Commentaires en fin de ligne
        """
        # Supprimer les commentaires en fin de ligne (// ...)
        repaired = re.sub(r'//[^\n]*', '', text)

        # Remplacer les guillemets simples par des doubles (hors des chaines)
        # Approche simplifiee : ne remplacer que les guillemets simples
        # utilises comme delimiteurs de cles/valeurs JSON
        repaired = re.sub(
            r"(?<=[{\[,:\s])'([^']*?)'(?=[}\],:\s])",
            r'"\1"',
            repaired,
        )

        # Supprimer les virgules pendantes avant ] ou }
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Supprimer les virgules pendantes en fin de texte
        repaired = repaired.rstrip().rstrip(',')

        return repaired

    @staticmethod
    def _recover_json_objects(text: str) -> Optional[List[Dict[str, Any]]]:
        """Recupere les objets JSON individuels depuis un texte malformate.

        Parcourt le texte ligne par ligne et tente de construire des
        objets JSON valides a partir des fragments trouves.

        Returns:
            Liste de dictionnaires recuperes, ou None si rien trouve.
        """
        objects: List[Dict[str, Any]] = []
        brace_depth = 0
        current_obj = ''

        for char in text:
            if char == '{':
                if brace_depth == 0:
                    current_obj = ''
                brace_depth += 1
                current_obj += char
            elif char == '}':
                brace_depth -= 1
                current_obj += char
                if brace_depth == 0 and current_obj:
                    try:
                        obj = json.loads(current_obj)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    current_obj = ''
            elif brace_depth > 0:
                current_obj += char

        return objects if objects else None

    @staticmethod
    def _close_truncated_json(text: str) -> str:
        """Ferme les structures JSON non terminees (tronquees par la limite de tokens).

        Parcourt le texte en tenant compte des chaines de caracteres pour
        compter les accolades et crochets ouverts, puis ajoute les fermetures
        manquantes.
        """
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        i = 0

        while i < len(text):
            char = text[i]
            if in_string:
                if char == '\\':
                    i += 2  # Skip escaped character
                    continue
                if char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    depth_brace += 1
                elif char == '}':
                    depth_brace = max(0, depth_brace - 1)
                elif char == '[':
                    depth_bracket += 1
                elif char == ']':
                    depth_bracket = max(0, depth_bracket - 1)
            i += 1

        stripped = text.rstrip()
        if stripped.endswith(','):
            stripped = stripped[:-1]

        return stripped + '}' * depth_brace + ']' * depth_bracket

    # ===================================================================
    # Construction de la transcription formatee
    # ===================================================================

    @staticmethod
    def _build_timestamped_transcript(segments: List[TranscriptSegment]) -> str:
        """Formate les segments en transcription avec timestamps [MM:SS].

        Args:
            segments: Segments de transcription a formater.

        Returns:
            Texte formate avec un segment par ligne : [MM:SS] texte
        """
        lines: List[str] = []
        for seg in segments:
            minutes = int(seg.start // 60)
            seconds = int(seg.start % 60)
            timestamp = f'[{minutes:02d}:{seconds:02d}]'
            text = seg.text.strip()
            if text:
                lines.append(f'{timestamp} {text}')
        return '\n'.join(lines)

    # ===================================================================
    # Detection de langue
    # ===================================================================

    def _resolve_language(self, segments: List[TranscriptSegment]) -> str:
        """Determine la langue effective a utiliser pour les prompts.

        Si la langue est fixee a 'fr' ou 'en', la retourne directement.
        Si 'auto', detecte la langue a partir du contenu de la transcription.

        Args:
            segments: Segments de transcription.

        Returns:
            Code de langue : 'fr' ou 'en'.
        """
        if self._language in ('fr', 'en'):
            return self._language
        return self._detect_language(segments)

    @staticmethod
    def _detect_language(segments: List[TranscriptSegment]) -> str:
        """Detecte la langue de la transcription (francais ou anglais).

        Analyse les premiers 500 caracteres de la transcription et compte
        les occurrences de mots-marqueurs francais. Si le nombre depasse
        le seuil, la langue est consideree comme francaise.

        Args:
            segments: Segments de transcription.

        Returns:
            'fr' si le texte semble etre en francais, 'en' sinon.
        """
        sample_text = ''
        for seg in segments:
            sample_text += ' ' + seg.text
            if len(sample_text) >= _LANGUAGE_SAMPLE_LENGTH:
                break

        sample_lower = sample_text[:_LANGUAGE_SAMPLE_LENGTH].lower()

        french_count = sum(
            1 for marker in _FRENCH_MARKERS if marker in sample_lower
        )

        if french_count >= _FRENCH_DETECTION_THRESHOLD:
            logger.debug(
                'Langue detectee : FR (%d marqueurs trouves)', french_count
            )
            return 'fr'

        logger.debug(
            'Langue detectee : EN (%d marqueurs FR trouves)', french_count
        )
        return 'en'

    # ===================================================================
    # Filtrage des segments par plage temporelle
    # ===================================================================

    @staticmethod
    def _segments_in_range(
        segments: List[TranscriptSegment],
        start: float,
        end: float,
    ) -> List[TranscriptSegment]:
        """Filtre les segments dont le centre tombe dans [start, end].

        Args:
            segments: Liste complete des segments.
            start: Borne inferieure en secondes.
            end: Borne superieure en secondes.

        Returns:
            Sous-liste des segments dans la plage temporelle.
        """
        result: List[TranscriptSegment] = []
        for seg in segments:
            center = (seg.start + seg.end) / 2.0
            if start <= center <= end:
                result.append(seg)
        return result

    # ===================================================================
    # Prompts — Passe 1 : Chapitres
    # ===================================================================

    @staticmethod
    def _chapter_system_prompt(lang: str, video_duration: float) -> str:
        """Construit le prompt systeme pour la passe 1 (chapitres).

        Args:
            lang: Code de langue ('fr' ou 'en').
            video_duration: Duree totale de la video en secondes.

        Returns:
            Prompt systeme.
        """
        duration_min = video_duration / 60.0

        if lang == 'fr':
            return (
                'Tu es un analyste de contenu video expert. '
                'Ta tache est de segmenter une transcription en chapitres thematiques.\n\n'
                'Regles :\n'
                f'- La video dure {duration_min:.1f} minutes\n'
                f'- Identifie entre {_MIN_CHAPTERS} et {_MAX_CHAPTERS} chapitres\n'
                '- Chaque chapitre doit correspondre a un sujet ou theme distinct\n'
                '- Les timestamps doivent etre en secondes (nombres entiers ou decimaux)\n'
                '- Les chapitres doivent couvrir toute la video sans trous ni chevauchements\n'
                '- Reponds UNIQUEMENT avec un tableau JSON valide, sans texte supplementaire'
            )
        else:
            return (
                'You are an expert video content analyst. '
                'Your task is to segment a transcript into thematic chapters.\n\n'
                'Rules:\n'
                f'- The video is {duration_min:.1f} minutes long\n'
                f'- Identify between {_MIN_CHAPTERS} and {_MAX_CHAPTERS} chapters\n'
                '- Each chapter must correspond to a distinct topic or theme\n'
                '- Timestamps must be in seconds (integers or decimals)\n'
                '- Chapters must cover the entire video without gaps or overlaps\n'
                '- Respond ONLY with a valid JSON array, no extra text'
            )

    @staticmethod
    def _chapter_user_prompt(
        lang: str,
        transcript: str,
        video_duration: float,
    ) -> str:
        """Construit le prompt utilisateur pour la passe 1 (chapitres).

        Args:
            lang: Code de langue ('fr' ou 'en').
            transcript: Transcription formatee avec timestamps.
            video_duration: Duree totale de la video en secondes.

        Returns:
            Prompt utilisateur.
        """
        if lang == 'fr':
            return (
                'Voici la transcription de la video :\n\n'
                f'{transcript}\n\n'
                'Analyse la transcription etape par etape :\n'
                '1. Identifie les points de transition thematique '
                '(changement de sujet, nouvelle question, nouvelle idee)\n'
                '2. Regroupe les segments adjacents qui traitent du meme sujet\n'
                '3. Donne un titre court et un resume pour chaque chapitre\n\n'
                'Reponds avec un tableau JSON de cette forme :\n'
                '[{"start": 0, "end": 180, "title": "Introduction", "summary": "..."},\n'
                ' {"start": 180, "end": 450, "title": "Sujet principal", "summary": "..."}]\n\n'
                f'Les timestamps doivent aller de 0 a {video_duration:.0f} secondes.\n'
                'JSON :'
            )
        else:
            return (
                'Here is the video transcript:\n\n'
                f'{transcript}\n\n'
                'Analyze the transcript step by step:\n'
                '1. Identify thematic transition points '
                '(topic changes, new questions, new ideas)\n'
                '2. Group adjacent segments that cover the same topic\n'
                '3. Give a short title and summary for each chapter\n\n'
                'Respond with a JSON array in this format:\n'
                '[{"start": 0, "end": 180, "title": "Introduction", "summary": "..."},\n'
                ' {"start": 180, "end": 450, "title": "Main topic", "summary": "..."}]\n\n'
                f'Timestamps must range from 0 to {video_duration:.0f} seconds.\n'
                'JSON:'
            )

    # ===================================================================
    # Prompts — Passe 2 : Scoring
    # ===================================================================

    @staticmethod
    def _scoring_system_prompt(lang: str) -> str:
        """Construit le prompt systeme pour la passe 2 (scoring).

        Args:
            lang: Code de langue ('fr' ou 'en').

        Returns:
            Prompt systeme.
        """
        if lang == 'fr':
            return (
                'Tu es un expert en contenu viral pour les reseaux sociaux '
                '(TikTok, Instagram Reels, YouTube Shorts).\n'
                'Ta tache est d\'identifier les 1 a 2 meilleurs moments viraux '
                'dans un chapitre de video.\n\n'
                'Pour chaque moment, tu dois :\n'
                '1. D\'abord RAISONNER sur pourquoi ce moment est viral '
                '(chain-of-thought)\n'
                '2. Puis SCORER chaque dimension de 1 a 10 :\n'
                '   - hook_strength : force du hook d\'ouverture '
                '(question, affirmation choc, promesse)\n'
                '   - emotional_intensity : intensite emotionnelle '
                '(rire, colere, surprise, inspiration)\n'
                '   - standalone_clarity : comprehensible SANS contexte exterieur\n'
                '   - quotability : contient une phrase memorable/partageable\n'
                '   - tension_arc : presence d\'un arc narratif '
                '(probleme -> resolution, build-up -> punchline)\n'
                '   - controversy : potentiel de debat ou de reactions polarisees\n\n'
                'Reponds UNIQUEMENT avec un tableau JSON valide.'
            )
        else:
            return (
                'You are an expert in viral social media content '
                '(TikTok, Instagram Reels, YouTube Shorts).\n'
                'Your task is to identify the 1-2 best viral moments '
                'in a video chapter.\n\n'
                'For each moment, you must:\n'
                '1. First REASON about why this moment is viral '
                '(chain-of-thought)\n'
                '2. Then SCORE each dimension from 1 to 10:\n'
                '   - hook_strength: opening hook power '
                '(question, shocking statement, promise)\n'
                '   - emotional_intensity: emotional intensity '
                '(laughter, anger, surprise, inspiration)\n'
                '   - standalone_clarity: understandable WITHOUT external context\n'
                '   - quotability: contains a memorable/shareable phrase\n'
                '   - tension_arc: presence of a narrative arc '
                '(problem -> resolution, build-up -> punchline)\n'
                '   - controversy: potential for debate or polarized reactions\n\n'
                'Respond ONLY with a valid JSON array.'
            )

    @staticmethod
    def _scoring_user_prompt(
        lang: str,
        chapter: LLMChapter,
        transcript: str,
        min_duration: float,
        max_duration: float,
    ) -> str:
        """Construit le prompt utilisateur pour la passe 2 (scoring).

        Args:
            lang: Code de langue ('fr' ou 'en').
            chapter: Chapitre a analyser.
            transcript: Transcription formatee du chapitre.
            min_duration: Duree minimale d'un clip en secondes.
            max_duration: Duree maximale d'un clip en secondes.

        Returns:
            Prompt utilisateur.
        """
        if lang == 'fr':
            return (
                f'Chapitre : "{chapter.title}"\n'
                f'Resume : {chapter.summary}\n'
                f'Plage : {chapter.start:.0f}s - {chapter.end:.0f}s\n\n'
                f'Transcription du chapitre :\n{transcript}\n\n'
                f'Trouve les 1 a 2 meilleurs moments viraux '
                f'(duree : {min_duration:.0f}-{max_duration:.0f} secondes).\n\n'
                'Mets ton raisonnement dans le champ "reasoning" de chaque objet JSON.\n\n'
                'Format JSON attendu :\n'
                '[{\n'
                '  "start": 15, "end": 65,\n'
                '  "reasoning": "Ce segment contient une question forte...",\n'
                '  "scores": {\n'
                '    "hook_strength": 8, "emotional_intensity": 7,\n'
                '    "standalone_clarity": 9, "quotability": 6,\n'
                '    "tension_arc": 5, "controversy": 4\n'
                '  },\n'
                '  "hook": "Premiere phrase du clip",\n'
                '  "emotion": "insight",\n'
                '  "reason": "Conseil autonome avec un hook clair"\n'
                '}]\n\n'
                'JSON :'
            )
        else:
            return (
                f'Chapter: "{chapter.title}"\n'
                f'Summary: {chapter.summary}\n'
                f'Range: {chapter.start:.0f}s - {chapter.end:.0f}s\n\n'
                f'Chapter transcript:\n{transcript}\n\n'
                f'Find the 1-2 best viral moments '
                f'(duration: {min_duration:.0f}-{max_duration:.0f} seconds).\n\n'
                'Put your reasoning inside the "reasoning" field of each JSON object.\n\n'
                'Expected JSON format:\n'
                '[{\n'
                '  "start": 15, "end": 65,\n'
                '  "reasoning": "This segment contains a strong opening question...",\n'
                '  "scores": {\n'
                '    "hook_strength": 8, "emotional_intensity": 7,\n'
                '    "standalone_clarity": 9, "quotability": 6,\n'
                '    "tension_arc": 5, "controversy": 4\n'
                '  },\n'
                '  "hook": "First sentence of the clip",\n'
                '  "emotion": "insight",\n'
                '  "reason": "Strong standalone advice with clear hook"\n'
                '}]\n\n'
                'JSON:'
            )

    # ===================================================================
    # Prompts — Passe 3 : Optimisation des bornes
    # ===================================================================

    @staticmethod
    def _boundary_system_prompt(lang: str) -> str:
        """Construit le prompt systeme pour la passe 3 (bornes).

        Args:
            lang: Code de langue ('fr' ou 'en').

        Returns:
            Prompt systeme.
        """
        if lang == 'fr':
            return (
                'Tu es un monteur video expert specialise dans les clips courts viraux.\n'
                'Ta tache est d\'optimiser les points de coupe (debut et fin) d\'un clip.\n\n'
                'Regles :\n'
                '- Le debut doit tomber sur une phrase forte (hook)\n'
                '- La fin doit tomber apres une phrase complete (pas de coupure au milieu)\n'
                '- Le clip doit etre comprehensible seul\n'
                '- Optimise pour un maximum d\'impact en 3 secondes d\'intro\n'
                '- Reponds UNIQUEMENT avec un objet JSON valide'
            )
        else:
            return (
                'You are an expert video editor specialized in short viral clips.\n'
                'Your task is to optimize the cut points (start and end) of a clip.\n\n'
                'Rules:\n'
                '- The start must land on a strong sentence (hook)\n'
                '- The end must land after a complete sentence (no mid-sentence cuts)\n'
                '- The clip must be understandable on its own\n'
                '- Optimize for maximum impact in the first 3 seconds\n'
                '- Respond ONLY with a valid JSON object'
            )

    @staticmethod
    def _boundary_user_prompt(
        lang: str,
        moment: LLMScoredMoment,
        transcript: str,
    ) -> str:
        """Construit le prompt utilisateur pour la passe 3 (bornes).

        Args:
            lang: Code de langue ('fr' ou 'en').
            moment: Moment a optimiser.
            transcript: Transcription formatee du contexte elargi.

        Returns:
            Prompt utilisateur.
        """
        if lang == 'fr':
            return (
                f'Moment actuel : {moment.start:.1f}s - {moment.end:.1f}s\n'
                f'Hook actuel : "{moment.hook_text}"\n\n'
                f'Transcription du contexte (avec marge de 10s) :\n{transcript}\n\n'
                'Propose des bornes optimisees pour ce clip.\n\n'
                'Format JSON attendu :\n'
                '{"start": 14.5, "end": 66.0, "hook": "Phrase d\'accroche optimisee"}\n\n'
                'JSON :'
            )
        else:
            return (
                f'Current moment: {moment.start:.1f}s - {moment.end:.1f}s\n'
                f'Current hook: "{moment.hook_text}"\n\n'
                f'Context transcript (with 10s margin):\n{transcript}\n\n'
                'Suggest optimized boundaries for this clip.\n\n'
                'Expected JSON format:\n'
                '{"start": 14.5, "end": 66.0, "hook": "Optimized hook sentence"}\n\n'
                'JSON:'
            )

    # ===================================================================
    # Parsing — Passe 1 : Chapitres
    # ===================================================================

    def _parse_chapters(
        self,
        raw_response: str,
        video_duration: float,
    ) -> List[LLMChapter]:
        """Parse la reponse JSON du LLM pour extraire les chapitres.

        Effectue une validation stricte des timestamps et corrige
        les incoherences (trous, chevauchements, bornes hors limites).

        Args:
            raw_response: Texte brut de la reponse LLM.
            video_duration: Duree totale de la video en secondes.

        Returns:
            Liste de LLMChapter valides, ou liste vide si parsing echoue.
        """
        data = self._extract_json(raw_response)

        if not isinstance(data, list) or not data:
            logger.warning('Reponse chapitres : JSON invalide ou vide')
            return []

        chapters: List[LLMChapter] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                start = float(item.get('start', -1))
                end = float(item.get('end', -1))
                title = str(item.get('title', '')).strip()
                summary = str(item.get('summary', '')).strip()
            except (TypeError, ValueError):
                continue

            # Validation des bornes
            if start < 0 or end <= start:
                continue

            # Clamper aux bornes de la video
            start = max(0.0, min(start, video_duration))
            end = max(start + 1.0, min(end, video_duration))

            if not title:
                title = f'Chapter {len(chapters) + 1}'

            chapters.append(LLMChapter(
                start=start,
                end=end,
                title=title,
                summary=summary or title,
            ))

        if not chapters:
            return []

        # Trier par timestamp de debut
        chapters.sort(key=lambda c: c.start)

        # S'assurer que le premier chapitre commence a 0
        if chapters[0].start > 5.0:
            chapters[0] = LLMChapter(
                start=0.0,
                end=chapters[0].end,
                title=chapters[0].title,
                summary=chapters[0].summary,
            )

        # S'assurer que le dernier chapitre couvre la fin de la video
        if chapters[-1].end < video_duration - 5.0:
            chapters[-1] = LLMChapter(
                start=chapters[-1].start,
                end=video_duration,
                title=chapters[-1].title,
                summary=chapters[-1].summary,
            )

        # Combler les trous entre chapitres consecutifs
        filled: List[LLMChapter] = [chapters[0]]
        for i in range(1, len(chapters)):
            prev = filled[-1]
            curr = chapters[i]

            if curr.start > prev.end + 1.0:
                # Trou detecte : etendre le chapitre precedent
                filled[-1] = LLMChapter(
                    start=prev.start,
                    end=curr.start,
                    title=prev.title,
                    summary=prev.summary,
                )

            if curr.start < prev.end - 1.0:
                # Chevauchement detecte : ajuster le debut du courant
                curr = LLMChapter(
                    start=prev.end,
                    end=curr.end,
                    title=curr.title,
                    summary=curr.summary,
                )

            if curr.end > curr.start:
                filled.append(curr)

        return filled

    # ===================================================================
    # Parsing — Passe 2 : Moments scores
    # ===================================================================

    def _parse_scored_moments(
        self,
        raw_response: str,
        chapter: LLMChapter,
        min_duration: float,
        max_duration: float,
    ) -> List[LLMScoredMoment]:
        """Parse la reponse JSON du LLM pour extraire les moments scores.

        Valide les timestamps, normalise les scores de 1-10 vers 0-1,
        et contraint les moments aux limites du chapitre.

        Args:
            raw_response: Texte brut de la reponse LLM.
            chapter: Chapitre parent pour la validation des bornes.
            min_duration: Duree minimale d'un clip en secondes.
            max_duration: Duree maximale d'un clip en secondes.

        Returns:
            Liste de LLMScoredMoment valides.
        """
        data = self._extract_json(raw_response)

        if not isinstance(data, list) or not data:
            logger.warning(
                'Reponse scoring : JSON invalide ou vide pour "%s"',
                chapter.title,
            )
            return []

        moments: List[LLMScoredMoment] = []
        score_dimensions = [
            'hook_strength', 'emotional_intensity', 'standalone_clarity',
            'quotability', 'tension_arc', 'controversy',
        ]

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                start = float(item.get('start', -1))
                end = float(item.get('end', -1))
            except (TypeError, ValueError):
                continue

            # Validation des bornes temporelles
            if start < 0 or end <= start:
                continue

            duration = end - start
            if duration < min_duration * 0.5:
                # Trop court meme avec tolerance
                continue
            if duration > max_duration * 1.5:
                # Trop long meme avec tolerance — tronquer
                end = start + max_duration

            # Clamper aux bornes du chapitre (avec tolerance de 5s)
            start = max(chapter.start - 5.0, start)
            end = min(chapter.end + 5.0, end)

            # Extraire les scores (nested "scores" object ou top-level)
            scores_raw = item.get('scores', {})
            if not isinstance(scores_raw, dict):
                scores_raw = {}

            # Normaliser les scores de 1-10 vers 0-1
            normalized_scores: Dict[str, float] = {}
            for dim in score_dimensions:
                # Priorite : scores_raw (nested) > top-level > defaut 5
                raw_score = scores_raw.get(dim, item.get(dim, 5))
                try:
                    raw_score = float(raw_score)
                except (TypeError, ValueError):
                    raw_score = 5.0
                # Clamper entre 1 et 10, puis normaliser
                raw_score = max(1.0, min(10.0, raw_score))
                normalized_scores[dim] = raw_score / 10.0

            # Extraire les metadonnees
            hook_text = str(item.get('hook', '')).strip()
            emotion = str(item.get('emotion', '')).strip()
            reason = str(item.get('reason', '')).strip()

            # Si le reasoning est present et qu'il n'y a pas de reason, utiliser le reasoning
            if not reason:
                reason = str(item.get('reasoning', '')).strip()

            moment = LLMScoredMoment(
                start=start,
                end=end,
                hook_strength=normalized_scores.get('hook_strength', 0.5),
                emotional_intensity=normalized_scores.get('emotional_intensity', 0.5),
                standalone_clarity=normalized_scores.get('standalone_clarity', 0.5),
                quotability=normalized_scores.get('quotability', 0.5),
                tension_arc=normalized_scores.get('tension_arc', 0.5),
                controversy=normalized_scores.get('controversy', 0.5),
                hook_text=hook_text,
                emotion=emotion,
                reason=reason,
            )

            moments.append(moment)

            # Maximum 2 moments par chapitre
            if len(moments) >= 2:
                break

        return moments

    # ===================================================================
    # Parsing — Passe 3 : Bornes optimisees
    # ===================================================================

    def _parse_boundary_response(
        self,
        raw_response: str,
        original: LLMScoredMoment,
    ) -> LLMScoredMoment:
        """Parse la reponse de la passe 3 et applique les bornes optimisees.

        En cas d'echec du parsing, retourne le moment original inchange.

        Args:
            raw_response: Texte brut de la reponse LLM.
            original: Moment original avant optimisation.

        Returns:
            Moment avec bornes optimisees, ou l'original en cas d'echec.
        """
        data = self._extract_json(raw_response)

        if data is None:
            return original

        # Si c'est une liste, prendre le premier element
        if isinstance(data, list):
            if not data:
                return original
            data = data[0]

        if not isinstance(data, dict):
            return original

        try:
            new_start = float(data.get('start', original.start))
            new_end = float(data.get('end', original.end))
        except (TypeError, ValueError):
            return original

        # Validation : les nouvelles bornes ne doivent pas trop devier
        max_shift = 15.0  # secondes de decalage maximum
        if abs(new_start - original.start) > max_shift:
            new_start = original.start
        if abs(new_end - original.end) > max_shift:
            new_end = original.end

        if new_end <= new_start:
            return original

        # Mettre a jour le hook si fourni
        new_hook = str(data.get('hook', '')).strip()
        if not new_hook:
            new_hook = original.hook_text

        # Creer un moment mis a jour en conservant tous les scores
        return LLMScoredMoment(
            start=new_start,
            end=new_end,
            hook_strength=original.hook_strength,
            emotional_intensity=original.emotional_intensity,
            standalone_clarity=original.standalone_clarity,
            quotability=original.quotability,
            tension_arc=original.tension_arc,
            controversy=original.controversy,
            hook_text=new_hook,
            emotion=original.emotion,
            reason=original.reason,
            composite_score=original.composite_score,
        )
