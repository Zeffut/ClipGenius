"""Parsing des reponses LLM et validation des moments viraux."""

import json
import re
from typing import List, Optional, Callable, Any
from dataclasses import dataclass
from rich.console import Console

console = Console()

# === Constantes de seuils et validation ===
MOMENT_MIN_SCORE_THRESHOLD: float = 0.5
MERGE_GAP_SECONDS: float = 5.0
DURATION_TOLERANCE_FACTOR: float = 1.2
MIN_DURATION_FACTOR: float = 0.5
OVERLAP_REJECTION_RATIO: float = 0.3
SENTENCE_SEARCH_WINDOW: float = 5.0

# === Constantes de fallback (quand le parsing LLM échoue) ===
FALLBACK_MIN_TEXT_LENGTH: int = 50
FALLBACK_BASE_SCORE: float = 0.4
FALLBACK_KEYWORD_BONUS: float = 0.15
FALLBACK_PUNCTUATION_BONUS: float = 0.1
FALLBACK_LENGTH_BONUS: float = 0.05
FALLBACK_LENGTH_THRESHOLD: int = 100
FALLBACK_SCORE_CAP: float = 0.7
FALLBACK_HOOK_PREVIEW_LENGTH: int = 50

# === Constantes de troncature des champs ===
HOOK_MAX_LENGTH: int = 200
EMOTION_MAX_LENGTH: int = 20
REASON_MAX_LENGTH: int = 200


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


def parse_global_response(
    response_text: str,
    segments: List[TranscriptSegment],
    min_duration: float,
    max_duration: float,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[ViralMomentAI]:
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
                            score=float(item.get('score', MOMENT_MIN_SCORE_THRESHOLD)),
                            hook=item.get('hook', '')[:HOOK_MAX_LENGTH],
                            emotion=item.get('emotion', 'neutre')[:EMOTION_MAX_LENGTH],
                            reason=item.get('reason', '')[:REASON_MAX_LENGTH]
                        )

                        duration = moment.end_time - moment.start_time
                        if (min_duration <= duration <= max_duration
                            and moment.score >= MOMENT_MIN_SCORE_THRESHOLD):  # Seuil pour filtrer les moments de qualité
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

    # Snap aux limites de phrases pour éviter de couper au milieu
    moments = [snap_to_sentence_boundaries(m, segments, min_duration, max_duration) for m in moments]

    return moments


def snap_to_sentence_boundaries(
    moment: ViralMomentAI,
    segments: List[TranscriptSegment],
    min_duration: float,
    max_duration: float
) -> ViralMomentAI:
    """
    Ajuste les timestamps pour commencer/finir sur des limites de phrases.

    Cherche le début de phrase le plus proche pour start_time,
    et la fin de phrase la plus proche pour end_time.
    """
    if not segments:
        return moment

    # Marge de recherche (en secondes)
    SEARCH_WINDOW = SENTENCE_SEARCH_WINDOW

    # === SNAP DU DÉBUT ===
    # Chercher le segment qui contient ou précède start_time
    best_start = moment.start_time
    for seg in segments:
        # Segment dans la fenêtre de recherche
        if abs(seg.start - moment.start_time) <= SEARCH_WINDOW:
            text = seg.text.strip()

            # Si le segment commence par une majuscule ou après ponctuation = bon début
            if text and (text[0].isupper() or seg.start == 0):
                # Préférer un début légèrement avant le timestamp LLM
                if seg.start <= moment.start_time:
                    best_start = seg.start
                    break
                # Ou légèrement après si pas d'autre option
                elif best_start == moment.start_time:
                    best_start = seg.start

    # === SNAP DE LA FIN ===
    # Chercher la fin de phrase la plus proche de end_time
    best_end = moment.end_time
    for seg in segments:
        if abs(seg.end - moment.end_time) <= SEARCH_WINDOW:
            text = seg.text.strip()

            # Si le segment finit par ponctuation forte = bonne fin
            if text and text[-1] in '.!?':
                # Préférer une fin légèrement après le timestamp LLM
                if seg.end >= moment.end_time:
                    best_end = seg.end
                    break
                # Ou légèrement avant si pas d'autre option
                elif best_end == moment.end_time:
                    best_end = seg.end

    # Vérifier que la durée reste valide
    new_duration = best_end - best_start
    if new_duration < min_duration:
        # Durée trop courte, étendre la fin
        best_end = best_start + min_duration
    elif new_duration > max_duration:
        # Durée trop longue, raccourcir la fin
        best_end = best_start + max_duration

    # Créer un nouveau moment avec les timestamps ajustés
    return ViralMomentAI(
        start_time=best_start,
        end_time=best_end,
        score=moment.score,
        hook=moment.hook,
        emotion=moment.emotion,
        reason=moment.reason
    )


def parse_response(
    response_text: str,
    segment: TranscriptSegment
) -> Optional[ViralMomentAI]:
    """Parse la réponse du LLM avec plusieurs stratégies de fallback"""

    response_text = response_text.strip()

    # Stratégie 1: Parser directement si c'est du JSON
    if response_text.startswith('{'):
        try:
            data = json.loads(response_text)
            return create_moment(
                start=segment.start,
                end=segment.end,
                score=float(data.get("score", MOMENT_MIN_SCORE_THRESHOLD)),
                hook=str(data.get("hook", "")),
                emotion=str(data.get("emotion", "unknown")),
                reason=str(data.get("reason", ""))
            )
        except json.JSONDecodeError:
            pass

    # Stratégie 2: Extraire le JSON du texte
    json_match = re.search(r'\{[^{}]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return create_moment(
                start=segment.start,
                end=segment.end,
                score=float(data.get("score", MOMENT_MIN_SCORE_THRESHOLD)),
                hook=str(data.get("hook", "")),
                emotion=str(data.get("emotion", "unknown")),
                reason=str(data.get("reason", ""))
            )
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
    if len(segment.text) > FALLBACK_MIN_TEXT_LENGTH:
        # Score basé sur des heuristiques simples
        text_lower = segment.text.lower()
        base_score = FALLBACK_BASE_SCORE

        # Bonus pour certains patterns
        if any(word in text_lower for word in [
            'incroyable', 'secret', 'révèle',
            'découvr', 'important', 'attention'
        ]):
            base_score += FALLBACK_KEYWORD_BONUS
        if any(word in text_lower for word in ['?', '!', 'pourquoi', 'comment', 'voici']):
            base_score += FALLBACK_PUNCTUATION_BONUS
        if len(segment.text) > FALLBACK_LENGTH_THRESHOLD:
            base_score += FALLBACK_LENGTH_BONUS

        return ViralMomentAI(
            start_time=segment.start,
            end_time=segment.end,
            score=min(FALLBACK_SCORE_CAP, base_score),  # Cap à 0.7 pour le fallback
            hook=segment.text[:FALLBACK_HOOK_PREVIEW_LENGTH] + "...",
            emotion="unknown",
            reason="Score estimé (parsing LLM échoué)"
        )

    return None


def create_moment(
    start: float,
    end: float,
    score: float,
    hook: str,
    emotion: str,
    reason: str
) -> ViralMomentAI:
    """Crée un ViralMomentAI à partir des données parsées"""
    return ViralMomentAI(
        start_time=start,
        end_time=end,
        score=min(1.0, max(0.0, score)),
        hook=hook[:HOOK_MAX_LENGTH],
        reason=reason[:REASON_MAX_LENGTH],
        emotion=emotion[:EMOTION_MAX_LENGTH]
    )


def validate_moments(
    moments: List[ViralMomentAI],
    video_duration: float,
    max_clips: int,
    min_duration: float,
    max_duration: float,
    min_viral_score: float,
    max_clips_override: Optional[int] = None
) -> List[ViralMomentAI]:
    """
    Valide, fusionne et filtre les moments détectés.

    Args:
        moments: Liste des moments à valider
        video_duration: Durée totale de la vidéo
        max_clips: Nombre maximum de clips
        min_duration: Durée minimum des clips
        max_duration: Durée maximum des clips
        min_viral_score: Score minimum pour qu'un moment soit retenu
        max_clips_override: Si défini, utilise cette valeur au lieu de max_clips

    Améliorations:
    - Utilise min_viral_score au lieu d'un seuil hardcodé
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
        if gap < MERGE_GAP_SECONDS:  # Moins de 5s d'écart = fusionner
            # Étendre le moment précédent
            new_end = max(last.end_time, end)
            # Limiter à max_clip_duration
            if new_end - last.start_time <= max_duration * DURATION_TOLERANCE_FACTOR:
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
        if duration < min_duration * MIN_DURATION_FACTOR:
            continue

        # Tronquer si trop long
        if duration > max_duration * DURATION_TOLERANCE_FACTOR:
            end = start + max_duration
            duration = end - start

        # Vérifier le score (utilise le seuil configuré, pas un hardcodé)
        if moment.score < min_viral_score:
            continue

        # Vérifier le chevauchement (seuil strict: 30%)
        overlap = False
        for used_start, used_end in used_ranges:
            overlap_start = max(start, used_start)
            overlap_end = min(end, used_end)
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                # Rejet si > 30% de chevauchement (plus strict que 50%)
                if overlap_duration > duration * OVERLAP_REJECTION_RATIO:
                    overlap = True
                    break

        if overlap:
            continue

        # Ajouter le moment validé
        moment.start_time = start
        moment.end_time = end
        valid.append(moment)
        used_ranges.append((start, end))

        # Limiter au nombre max de clips (utiliser override si fourni)
        effective_max_clips = max_clips_override if max_clips_override is not None else max_clips
        if len(valid) >= effective_max_clips:
            break

    # Fallback: si aucun moment valide, prendre le meilleur candidat
    if not valid and merged:
        best = merged[0]
        best.start_time = max(0, best.start_time)
        best.end_time = min(video_duration, best.end_time)
        if best.end_time - best.start_time < min_duration:
            best.end_time = min(video_duration, best.start_time + min_duration)
        console.print(f"[dim]Fallback: meilleur score {best.score:.0%}[/dim]")
        return [best]

    # Trier par temps pour l'export
    valid.sort(key=lambda x: x.start_time)
    return valid
