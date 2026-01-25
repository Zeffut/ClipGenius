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
import hashlib
from typing import List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console

console = Console()




@dataclass
class ViralMomentAI:
    """Moment viral détecté par l'IA"""
    start_time: float
    end_time: float
    score: float  # 0-1
    hook: str  # Phrase d'accroche suggérée
    reason: str  # Pourquoi c'est viral
    emotion: str  # Émotion principale (humour, surprise, émotion, tension, etc.)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass 
class TranscriptSegment:
    """Segment de transcription avec timestamps"""
    start: float
    end: float
    text: str


class LocalAIViralAnalyzer:
    """
    Analyse de contenu viral 100% locale avec Phi-4-mini via llama.cpp.

    Pas d'API cloud, pas de clé requise, fonctionne offline.
    Optimisé pour segments vidéo courts (30-90s).
    
    Phi-4-mini offre de meilleures performances de raisonnement que Phi-3,
    avec le même format de prompt.
    """

    # Format chat Phi-3/Phi-4 (compatible avec les deux modèles)
    # ✨ NOUVEAU PROMPT: Analyse globale au lieu de segment par segment
    GLOBAL_PROMPT_TEMPLATE = """<|system|>
Tu es un expert en contenu viral pour TikTok/Reels/Shorts. Tu identifies les meilleurs moments d'une vidéo.
Réponds UNIQUEMENT avec un tableau JSON, rien d'autre.<|end|>
<|user|>
Voici la transcription complète d'une vidéo de {duration:.0f}s:

{transcript}

MISSION:
Identifie TOUS les moments viraux potentiels (30-90s chacun).

CRITÈRES:
- Accroche forte au début du segment
- Émotion claire (humour, surprise, tension, inspiration)
- Message complet et autonome
- Potentiel de partage élevé

INSTRUCTIONS:
1. Lis TOUTE la transcription
2. Repère TOUS les moments intéressants (pas seulement les meilleurs)
3. Pour chaque moment, donne le timestamp de début et fin
4. Score entre 0.0 et 1.0 (inclure aussi les moments moyens > 0.5)
5. Cherche au moins 10-15 moments si la vidéo est longue

FORMAT JSON EXACT (tableau de moments):
[
  {{"start": 15, "end": 45, "score": 0.85, "hook": "phrase accrocheuse", "emotion": "humour", "reason": "explication courte"}},
  {{"start": 120, "end": 180, "score": 0.78, "hook": "autre phrase", "emotion": "surprise", "reason": "pourquoi viral"}}
]

Retourne UNIQUEMENT le tableau JSON, rien d'autre.<|end|>
<|assistant|>
"""

    # Ancien prompt pour analyse segment par segment (fallback si vidéo très longue)
    PROMPT_TEMPLATE = """<|system|>
Tu es un expert en contenu viral TikTok/Reels/Shorts. Tu analyses des segments vidéo et donnes un score de viralité.
IMPORTANT: Réponds UNIQUEMENT avec un objet JSON valide, rien d'autre.<|end|>
<|user|>
Évalue ce segment vidéo pour son potentiel viral:

SEGMENT [{start:.0f}s - {end:.0f}s] (durée: {duration:.0f}s)
TRANSCRIPTION: "{text}"

Critères d'évaluation:
- Accroche forte dès le début?
- Émotion (humour, surprise, inspiration, tension)?
- Message complet et clair?
- Potentiel de partage?

Donne un score entre 0.0 et 1.0:
- 0.8-1.0 = Excellent potentiel viral (accroche forte + émotion + message clair)
- 0.6-0.8 = Bon potentiel (2 critères sur 3)
- 0.4-0.6 = Potentiel moyen (contenu correct mais pas exceptionnel)
- 0.0-0.4 = Faible potentiel (ennuyeux, confus, ou incomplet)

Réponds avec CE FORMAT JSON EXACT:
{{"score": 0.75, "hook": "phrase accrocheuse du segment", "emotion": "type_emotion", "reason": "explication courte"}}<|end|>
<|assistant|>
"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        min_clip_duration: float = 30.0,
        max_clip_duration: float = 90.0,
        max_clips: int = 5,  # Réduit de 10 à 5 pour éviter trop de clips
        min_viral_score: float = 0.70,  # Seuil minimum de qualité
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
        """
        from .local_llm import LocalLLM

        self.llm = LocalLLM(model_path=model_path, n_threads=n_threads)
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.max_clips = max_clips
        self.min_viral_score = min_viral_score

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
        # Pour chaque section → analyse indépendante → 3-5 meilleurs moments
        # Résultat: Vidéo 1h30 → 6 sections → 18-30 clips au total
        
        SECTION_DURATION = 900  # 15 minutes = 900 secondes
        
        # Découper la vidéo en sections de 15 minutes
        sections = self._split_into_sections(segments, video_duration, SECTION_DURATION)
        total_sections = len(sections)
        
        console.print(f"[cyan]📹 Vidéo découpée en {total_sections} sections de ~15min[/cyan]")
        
        all_moments = []
        
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
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
        
        # Trier par score et valider
        all_moments.sort(key=lambda m: m.score, reverse=True)
        validated = self._validate_moments(all_moments, video_duration)
        
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
        
        # Créer le prompt pour cette section
        prompt = f"""<|system|>
Tu es un expert en contenu viral pour TikTok/Reels/Shorts. Tu identifies les meilleurs moments d'une section de vidéo.
Réponds UNIQUEMENT avec un tableau JSON, rien d'autre.<|end|>
<|user|>
Voici la transcription de la SECTION {section_num}/{total_sections} (durée: {section_duration:.0f}s):

{section_transcript}

MISSION:
Identifie les 2 à 3 MEILLEURS moments viraux de cette section (30-90s chacun).
Sois TRÈS SÉLECTIF - ne retiens que les moments vraiment exceptionnels.

CRITÈRES STRICTS:
- Accroche TRÈS forte au début (doit capter l'attention en 3s)
- Émotion intense (humour fort, surprise majeure, tension palpable)
- Message complet et autonome (compréhensible hors contexte)
- Fort potentiel de partage et d'engagement

INSTRUCTIONS:
1. Analyse TOUTE cette section
2. Repère les 2-3 moments les plus EXCEPTIONNELS uniquement
3. Score: 0.85+ = viral assuré, 0.70-0.85 = bon potentiel, <0.70 = ne pas inclure
4. Ne retourne QUE les moments avec score >= 0.70

FORMAT JSON EXACT (tableau de 2-3 moments max):
[
  {{"start": 15, "end": 65, "score": 0.88, "hook": "phrase accrocheuse", "emotion": "humour", "reason": "explication courte"}},
  {{"start": 120, "end": 180, "score": 0.75, "hook": "autre phrase", "emotion": "surprise", "reason": "pourquoi viral"}}
]

Retourne UNIQUEMENT le tableau JSON (2-3 moments MAX), rien d'autre.<|end|>
<|assistant|>
"""
        
        # Système de retry (max 3 tentatives)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Générer avec le LLM
                response = self.llm.generate(
                    prompt,
                    max_tokens=800,  # 3-5 moments = ~500-700 tokens
                    temperature=0.2,  # Un peu de créativité pour varier les sélections
                    top_p=0.9,
                    stop=["<|end|>", "\n\n\n"]
                )
                
                # Parser la réponse
                moments = self._parse_global_response(response.text, section_segments)
                
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
    
    def _parse_global_response(self, response_text: str, segments: List[TranscriptSegment]) -> List[ViralMomentAI]:
        """Parse la réponse globale du LLM (tableau JSON de moments)."""
        response_text = response_text.strip()
        moments = []
        
        try:
            # Extraire le tableau JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
                
                if isinstance(data, list):
                    for item in data:
                        try:
                            moment = ViralMomentAI(
                                start_time=float(item.get('start', 0)),
                                end_time=float(item.get('end', 0)),
                                score=float(item.get('score', 0.5)),
                                hook=item.get('hook', '')[:200],
                                emotion=item.get('emotion', 'neutre')[:20],
                                reason=item.get('reason', '')[:200]
                            )
                            
                            duration = moment.end_time - moment.start_time
                            if (self.min_clip_duration <= duration <= self.max_clip_duration 
                                and moment.score >= 0.5):  # Seuil pour filtrer les moments de qualité
                                moments.append(moment)
                        except (ValueError, KeyError) as e:
                            console.print(f"[dim]⚠️ Moment invalide ignoré: {e}[/dim]")
                            continue
                else:
                    raise ValueError("Réponse n'est pas un tableau JSON")
                    
        except Exception as e:
            console.print(f"[yellow]⚠️ Erreur parsing JSON: {e}[/yellow]")
            console.print(f"[dim]Réponse LLM: {response_text[:200]}...[/dim]")
        
        moments.sort(key=lambda m: m.score, reverse=True)
        # Retourner tous les moments trouvés (le tri final se fait dans analyze())
        return moments
    
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
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
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
        validated = self._validate_moments(moments, video_duration)
        
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
        text = segment.text[:800]

        prompt = self.PROMPT_TEMPLATE.format(
            start=segment.start,
            end=segment.end,
            duration=duration,
            text=text
        )

        # Générer avec le LLM
        response = self.llm.generate(
            prompt,
            max_tokens=200,
            temperature=0.1,  # Plus déterministe
            stop=["<|end|>", "\n\n", "```"]
        )

        # Parser la réponse JSON
        return self._parse_response(response.text, segment)

    def _parse_response(
        self, 
        response_text: str, 
        segment: TranscriptSegment
    ) -> Optional[ViralMomentAI]:
        """Parse la réponse du LLM avec plusieurs stratégies de fallback"""
        
        response_text = response_text.strip()
        
        # Stratégie 1: Parser directement si c'est du JSON
        if response_text.startswith('{'):
            try:
                data = json.loads(response_text)
                return self._create_moment(data, segment)
            except json.JSONDecodeError:
                pass
        
        # Stratégie 2: Extraire le JSON du texte
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._create_moment(data, segment)
            except json.JSONDecodeError:
                pass
        
        # Stratégie 3: Extraire le score avec regex
        score_match = re.search(r'"?score"?\s*[:=]\s*([\d.]+)', response_text)
        if score_match:
            try:
                score = float(score_match.group(1))
                # Extraire d'autres champs si possible
                hook_match = re.search(r'"?hook"?\s*[:=]\s*"([^"]+)"', response_text)
                emotion_match = re.search(r'"?emotion"?\s*[:=]\s*"?(\w+)"?', response_text)
                reason_match = re.search(r'"?reason"?\s*[:=]\s*"([^"]+)"', response_text)
                
                return ViralMomentAI(
                    start_time=segment.start,
                    end_time=segment.end,
                    score=min(1.0, max(0.0, score)),
                    hook=hook_match.group(1) if hook_match else "",
                    emotion=emotion_match.group(1) if emotion_match else "unknown",
                    reason=reason_match.group(1) if reason_match else ""
                )
            except (ValueError, AttributeError):
                pass
        
        # Stratégie 4: Donner un score par défaut basé sur le contenu
        # Si le LLM n'a pas pu parser mais le segment existe, on lui donne une chance
        if len(segment.text) > 50:
            # Score basé sur des heuristiques simples
            text_lower = segment.text.lower()
            base_score = 0.4
            
            # Bonus pour certains patterns
            if any(word in text_lower for word in ['incroyable', 'secret', 'révèle', 'découvr', 'important', 'attention']):
                base_score += 0.15
            if any(word in text_lower for word in ['?', '!', 'pourquoi', 'comment', 'voici']):
                base_score += 0.1
            if len(segment.text) > 100:
                base_score += 0.05
                
            return ViralMomentAI(
                start_time=segment.start,
                end_time=segment.end,
                score=min(0.7, base_score),  # Cap à 0.7 pour le fallback
                hook=segment.text[:50] + "...",
                emotion="unknown",
                reason="Score estimé (parsing LLM échoué)"
            )
        
        return None

    def _create_moment(self, data: dict, segment: TranscriptSegment) -> ViralMomentAI:
        """Crée un ViralMomentAI à partir des données parsées"""
        score = float(data.get("score", 0.5))
        hook = str(data.get("hook", ""))
        emotion = str(data.get("emotion", "unknown"))
        reason = str(data.get("reason", ""))

        return ViralMomentAI(
            start_time=segment.start,
            end_time=segment.end,
            score=min(1.0, max(0.0, score)),
            hook=hook[:100],
            reason=reason[:200],
            emotion=emotion[:20]
        )

    def _validate_moments(
        self,
        moments: List[ViralMomentAI],
        video_duration: float
    ) -> List[ViralMomentAI]:
        """
        Valide, fusionne et filtre les moments détectés.
        
        Améliorations:
        - Utilise self.min_viral_score au lieu d'un seuil hardcodé
        - Fusionne les moments adjacents/chevauchants
        - Seuil de chevauchement plus strict (30% au lieu de 50%)
        """

        if not moments:
            return []

        # Trier par temps de début pour faciliter la fusion
        moments.sort(key=lambda x: x.start_time)
        
        # === ÉTAPE 1: Fusion des moments adjacents/chevauchants ===
        merged = []
        for moment in moments:
            start = max(0, moment.start_time)
            end = min(video_duration, moment.end_time)
            
            if not merged:
                merged.append(ViralMomentAI(
                    start_time=start,
                    end_time=end,
                    score=moment.score,
                    hook=moment.hook,
                    reason=moment.reason,
                    emotion=moment.emotion
                ))
                continue
            
            last = merged[-1]
            # Fusionner si chevauchement > 10s ou écart < 5s
            gap = start - last.end_time
            if gap < 5:  # Moins de 5s d'écart = fusionner
                # Étendre le moment précédent
                new_end = max(last.end_time, end)
                # Limiter à max_clip_duration
                if new_end - last.start_time <= self.max_clip_duration * 1.2:
                    last.end_time = new_end
                    # Garder le meilleur score
                    if moment.score > last.score:
                        last.score = moment.score
                        last.hook = moment.hook
                        last.reason = moment.reason
                    continue
            
            # Pas de fusion, ajouter comme nouveau moment
            merged.append(ViralMomentAI(
                start_time=start,
                end_time=end,
                score=moment.score,
                hook=moment.hook,
                reason=moment.reason,
                emotion=moment.emotion
            ))
        
        # === ÉTAPE 2: Trier par score décroissant ===
        merged.sort(key=lambda x: x.score, reverse=True)
        
        # === ÉTAPE 3: Filtrer par score et chevauchement ===
        valid = []
        used_ranges = []

        for moment in merged:
            start = moment.start_time
            end = moment.end_time
            duration = end - start

            # Vérifier la durée minimum
            if duration < self.min_clip_duration * 0.5:
                continue
            
            # Tronquer si trop long
            if duration > self.max_clip_duration * 1.2:
                end = start + self.max_clip_duration
                duration = end - start

            # Vérifier le score (utilise le seuil configuré, pas un hardcodé)
            if moment.score < self.min_viral_score:
                continue

            # Vérifier le chevauchement (seuil strict: 30%)
            overlap = False
            for used_start, used_end in used_ranges:
                overlap_start = max(start, used_start)
                overlap_end = min(end, used_end)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    # Rejet si > 30% de chevauchement (plus strict que 50%)
                    if overlap_duration > duration * 0.3:
                        overlap = True
                        break
            
            if overlap:
                continue

            # Ajouter le moment validé
            moment.start_time = start
            moment.end_time = end
            valid.append(moment)
            used_ranges.append((start, end))

            # Limiter au nombre max de clips
            if len(valid) >= self.max_clips:
                break

        # Fallback: si aucun moment valide, prendre le meilleur candidat
        if not valid and merged:
            best = merged[0]
            best.start_time = max(0, best.start_time)
            best.end_time = min(video_duration, best.end_time)
            if best.end_time - best.start_time < self.min_clip_duration:
                best.end_time = min(video_duration, best.start_time + self.min_clip_duration)
            console.print(f"[dim]Fallback: meilleur score {best.score:.0%}[/dim]")
            return [best]

        # Trier par temps pour l'export
        valid.sort(key=lambda x: x.start_time)
        return valid


def analyze_with_ai(
    segments: List[TranscriptSegment],
    video_duration: float,
    min_duration: float = 30.0,
    max_duration: float = 90.0,
    max_clips: int = 5,
    min_viral_score: float = 0.70,
    model_path: Optional[str] = None,
    video_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
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
            min_viral_score=min_viral_score
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
