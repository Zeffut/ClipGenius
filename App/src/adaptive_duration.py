"""
Module de gestion des durées adaptatives et formats multiples
Adapte la durée des clips au contenu et génère différents formats
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console

console = Console()


class Platform(Enum):
    """Plateformes de destination"""
    TIKTOK = "tiktok"
    REELS = "reels"
    SHORTS = "shorts"
    ALL = "all"


@dataclass
class PlatformSpec:
    """Spécifications par plateforme"""
    name: str
    min_duration: float
    max_duration: float
    ideal_duration: float
    aspect_ratio: Tuple[int, int]  # (width, height)
    max_file_size_mb: int
    
    @property
    def aspect_ratio_float(self) -> float:
        return self.aspect_ratio[0] / self.aspect_ratio[1]


# Spécifications des plateformes
PLATFORM_SPECS = {
    Platform.TIKTOK: PlatformSpec(
        name="TikTok",
        min_duration=15.0,
        max_duration=60.0,  # 60s max pour TikTok standard
        ideal_duration=30.0,
        aspect_ratio=(9, 16),
        max_file_size_mb=287
    ),
    Platform.REELS: PlatformSpec(
        name="Instagram Reels",
        min_duration=15.0,
        max_duration=90.0,
        ideal_duration=60.0,
        aspect_ratio=(9, 16),
        max_file_size_mb=250
    ),
    Platform.SHORTS: PlatformSpec(
        name="YouTube Shorts",
        min_duration=15.0,
        max_duration=60.0,
        ideal_duration=45.0,
        aspect_ratio=(9, 16),
        max_file_size_mb=500
    ),
}


@dataclass
class ContentType:
    """Type de contenu détecté"""
    type_name: str  # 'podcast', 'tutorial', 'vlog', 'interview', 'music', 'comedy', 'news'
    confidence: float
    recommended_duration: Tuple[float, float]  # (min, max)
    
    
# Durées recommandées par type de contenu
CONTENT_DURATIONS = {
    'podcast': (60.0, 90.0),      # Conversations longues
    'interview': (45.0, 75.0),     # Q&R
    'tutorial': (30.0, 60.0),      # Instructions étape par étape
    'vlog': (30.0, 60.0),          # Moments de vie
    'comedy': (15.0, 45.0),        # Sketchs courts
    'music': (15.0, 30.0),         # Clips musicaux
    'news': (30.0, 60.0),          # Actualités
    'motivational': (20.0, 45.0),  # Citations / motivation
    'unknown': (30.0, 60.0),       # Par défaut
}


@dataclass
class AdaptiveMoment:
    """Moment avec durée adaptée"""
    start_time: float
    end_time: float
    original_start: float
    original_end: float
    score: float
    reason: str
    content_type: str
    platform: Platform
    is_complete_thought: bool  # True si le segment est une idée complète


class AdaptiveDurationManager:
    """
    Gère les durées adaptatives des clips.
    
    Fonctionnalités:
    - Détecte le type de contenu
    - Adapte les durées selon le contenu
    - Trouve les frontières naturelles (fin de phrase)
    - Génère des variantes pour chaque plateforme
    """
    
    def __init__(
        self,
        default_platform: Platform = Platform.REELS,
        respect_sentence_boundaries: bool = True,
        boundary_tolerance: float = 5.0  # Tolérance pour ajuster les frontières
    ):
        self.default_platform = default_platform
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.boundary_tolerance = boundary_tolerance
    
    def detect_content_type(
        self,
        transcript_segments: List[Dict],
        audio_features: Optional[Dict] = None
    ) -> ContentType:
        """
        Détecte le type de contenu à partir de la transcription.
        
        Args:
            transcript_segments: Segments de transcription
            audio_features: Features audio optionnelles
            
        Returns:
            ContentType avec recommandations
        """
        if not transcript_segments:
            return ContentType('unknown', 0.0, CONTENT_DURATIONS['unknown'])
        
        # Combiner tout le texte
        full_text = " ".join(
            s.get('text', s.get('word', '')) 
            for s in transcript_segments
        ).lower()
        
        scores = {
            'podcast': 0.0,
            'interview': 0.0,
            'tutorial': 0.0,
            'vlog': 0.0,
            'comedy': 0.0,
            'music': 0.0,
            'news': 0.0,
            'motivational': 0.0,
        }
        
        # Indicateurs de podcast/interview
        interview_keywords = ['question', 'réponse', 'alors', 'donc', 'tu penses', 
                             'vous pensez', 'selon toi', 'selon vous', 'invité']
        for kw in interview_keywords:
            if kw in full_text:
                scores['podcast'] += 0.15
                scores['interview'] += 0.2
        
        # Indicateurs de tutoriel
        tutorial_keywords = ['étape', 'premièrement', 'ensuite', 'puis', 'comment',
                           'voici', 'astuce', 'conseil', 'méthode', 'technique',
                           'step', 'first', 'then', 'how to', 'tip', 'trick']
        for kw in tutorial_keywords:
            if kw in full_text:
                scores['tutorial'] += 0.2
        
        # Indicateurs de comédie
        comedy_keywords = ['haha', 'mdr', 'lol', 'drôle', 'blague', 'humour',
                         'funny', 'joke', 'laugh', 'hilarious']
        for kw in comedy_keywords:
            if kw in full_text:
                scores['comedy'] += 0.25
        
        # Indicateurs de vlog
        vlog_keywords = ['aujourd\'hui', 'je vais', 'on va', 'avec moi', 'ma journée',
                        'today', 'i\'m going', 'we\'re going', 'my day', 'follow me']
        for kw in vlog_keywords:
            if kw in full_text:
                scores['vlog'] += 0.2
        
        # Indicateurs motivationnels
        motivational_keywords = ['croyez', 'possible', 'succès', 'réussir', 'motivation',
                                'believe', 'success', 'achieve', 'dream', 'possible']
        for kw in motivational_keywords:
            if kw in full_text:
                scores['motivational'] += 0.2
        
        # Indicateurs de news
        news_keywords = ['actualité', 'news', 'breaking', 'dernière heure', 'annonce',
                        'officiel', 'official', 'announcement']
        for kw in news_keywords:
            if kw in full_text:
                scores['news'] += 0.25
        
        # Trouver le type dominant
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Si aucun score significatif, utiliser 'unknown'
        if best_score < 0.2:
            return ContentType('unknown', 0.0, CONTENT_DURATIONS['unknown'])
        
        return ContentType(
            type_name=best_type,
            confidence=min(1.0, best_score),
            recommended_duration=CONTENT_DURATIONS[best_type]
        )
    
    def find_sentence_boundaries(
        self,
        words: List[Dict],
        start_time: float,
        end_time: float
    ) -> List[float]:
        """
        Trouve les frontières de phrases dans une plage de temps.
        
        Args:
            words: Liste de mots avec timestamps
            start_time: Début de la plage
            end_time: Fin de la plage
            
        Returns:
            Liste de timestamps de fin de phrases
        """
        boundaries = []
        
        # Ponctuation de fin de phrase
        end_punctuation = {'.', '!', '?', '...', '。', '！', '？'}
        
        for w in words:
            word_text = w.get('word', w.get('text', ''))
            word_end = w.get('end', w.get('end_time', 0))
            
            # Vérifier si le mot est dans la plage
            if start_time <= word_end <= end_time:
                # Vérifier si c'est une fin de phrase
                if any(word_text.endswith(p) for p in end_punctuation):
                    boundaries.append(word_end)
        
        return boundaries
    
    def adjust_to_boundaries(
        self,
        start_time: float,
        end_time: float,
        words: List[Dict],
        min_duration: float,
        max_duration: float
    ) -> Tuple[float, float, bool]:
        """
        Ajuste les timestamps pour respecter les frontières de phrases.
        
        Args:
            start_time: Début souhaité
            end_time: Fin souhaitée
            words: Liste de mots
            min_duration: Durée minimum
            max_duration: Durée maximum
            
        Returns:
            Tuple (new_start, new_end, is_complete_thought)
        """
        if not self.respect_sentence_boundaries or not words:
            return start_time, end_time, False
        
        current_duration = end_time - start_time
        
        # Chercher une fin de phrase proche de end_time
        search_start = end_time - self.boundary_tolerance
        search_end = min(end_time + self.boundary_tolerance, start_time + max_duration)
        
        boundaries = self.find_sentence_boundaries(words, search_start, search_end)
        
        if boundaries:
            # Trouver la frontière la plus proche de end_time
            best_boundary = min(boundaries, key=lambda b: abs(b - end_time))
            
            new_duration = best_boundary - start_time
            
            # Vérifier que la nouvelle durée est acceptable
            if min_duration <= new_duration <= max_duration:
                return start_time, best_boundary, True
        
        # Pas de frontière trouvée, garder les timestamps originaux
        return start_time, end_time, False
    
    def adapt_moments_for_platform(
        self,
        moments: List[Any],
        words: List[Dict],
        platform: Platform = None,
        content_type: Optional[ContentType] = None
    ) -> List[AdaptiveMoment]:
        """
        Adapte les moments pour une plateforme spécifique.
        
        Args:
            moments: Liste de moments viraux originaux
            words: Liste de mots avec timestamps
            platform: Plateforme cible (ou default)
            content_type: Type de contenu détecté
            
        Returns:
            Liste de moments adaptés
        """
        platform = platform or self.default_platform
        spec = PLATFORM_SPECS.get(platform, PLATFORM_SPECS[Platform.REELS])
        
        # Déterminer les durées à utiliser
        if content_type and content_type.confidence > 0.3:
            min_dur = max(spec.min_duration, content_type.recommended_duration[0])
            max_dur = min(spec.max_duration, content_type.recommended_duration[1])
        else:
            min_dur = spec.min_duration
            max_dur = spec.max_duration
        
        console.print(f"[dim]Adaptation pour {spec.name}: {min_dur:.0f}s - {max_dur:.0f}s[/dim]")
        
        adapted_moments = []
        
        for moment in moments:
            # Extraire les timestamps originaux (gérer objet ou dict)
            if hasattr(moment, 'start_time'):
                orig_start = moment.start_time
                orig_end = moment.end_time
            elif isinstance(moment, dict):
                orig_start = moment.get('start_time', 0)
                orig_end = moment.get('end_time', 0)
            else:
                orig_start = 0
                orig_end = 0
            orig_duration = orig_end - orig_start
            
            # Calculer la nouvelle durée cible
            if orig_duration > max_dur:
                # Clip trop long, réduire
                target_end = orig_start + max_dur
            elif orig_duration < min_dur:
                # Clip trop court, étendre si possible
                target_end = orig_start + min_dur
            else:
                target_end = orig_end
            
            # Ajuster aux frontières de phrases
            new_start, new_end, is_complete = self.adjust_to_boundaries(
                orig_start, target_end, words, min_dur, max_dur
            )
            
            # Extraire le score et la raison (gérer objet ou dict)
            if hasattr(moment, 'score'):
                score = moment.score
                reason = getattr(moment, 'reason', '')
            elif isinstance(moment, dict):
                score = moment.get('score', 0.5)
                reason = moment.get('reason', '')
            else:
                score = 0.5
                reason = ''
            
            adapted_moments.append(AdaptiveMoment(
                start_time=new_start,
                end_time=new_end,
                original_start=orig_start,
                original_end=orig_end,
                score=score,
                reason=reason,
                content_type=content_type.type_name if content_type else 'unknown',
                platform=platform,
                is_complete_thought=is_complete
            ))
        
        return adapted_moments
    
    def generate_multi_format(
        self,
        moments: List[Any],
        words: List[Dict],
        content_type: Optional[ContentType] = None,
        platforms: List[Platform] = None
    ) -> Dict[Platform, List[AdaptiveMoment]]:
        """
        Génère des variantes pour plusieurs plateformes.
        
        Args:
            moments: Moments viraux originaux
            words: Liste de mots
            content_type: Type de contenu
            platforms: Plateformes cibles (toutes si None)
            
        Returns:
            Dict avec moments adaptés par plateforme
        """
        if platforms is None:
            platforms = [Platform.TIKTOK, Platform.REELS, Platform.SHORTS]
        
        results = {}
        
        for platform in platforms:
            adapted = self.adapt_moments_for_platform(
                moments, words, platform, content_type
            )
            results[platform] = adapted
            
            # Log des adaptations
            complete_count = sum(1 for m in adapted if m.is_complete_thought)
            console.print(
                f"  [green]{platform.value}:[/green] {len(adapted)} clips, "
                f"{complete_count} avec fin de phrase propre"
            )
        
        return results


def create_adaptive_clips(
    moments: List[Any],
    words: List[Dict],
    transcript_segments: List[Dict] = None,
    platform: Platform = Platform.REELS
) -> List[AdaptiveMoment]:
    """
    Fonction utilitaire pour créer des clips avec durées adaptatives.
    
    Args:
        moments: Moments viraux détectés
        words: Liste de mots avec timestamps
        transcript_segments: Segments de transcription pour détection du type
        platform: Plateforme cible
        
    Returns:
        Liste de moments adaptés
    """
    manager = AdaptiveDurationManager()
    
    # Détecter le type de contenu
    content_type = None
    if transcript_segments:
        content_type = manager.detect_content_type(transcript_segments)
        if content_type.confidence > 0.3:
            console.print(
                f"[cyan]Type de contenu détecté:[/cyan] {content_type.type_name} "
                f"(confiance: {content_type.confidence:.0%})"
            )
    
    return manager.adapt_moments_for_platform(
        moments, words, platform, content_type
    )


def get_platform_spec(platform: Platform) -> PlatformSpec:
    """Retourne les spécifications d'une plateforme"""
    return PLATFORM_SPECS.get(platform, PLATFORM_SPECS[Platform.REELS])


if __name__ == "__main__":
    # Tests
    manager = AdaptiveDurationManager()
    
    # Test de détection de type
    test_segments = [
        {"text": "Aujourd'hui je vais vous montrer comment faire"},
        {"text": "Première étape, on va commencer par"},
        {"text": "Ensuite, voici l'astuce secrète"},
        {"text": "Et pour finir, le conseil le plus important"},
    ]
    
    content_type = manager.detect_content_type(test_segments)
    print(f"Type détecté: {content_type.type_name} ({content_type.confidence:.0%})")
    print(f"Durée recommandée: {content_type.recommended_duration}")
    
    # Test de frontières
    test_words = [
        {"word": "Bonjour.", "start": 0, "end": 0.5},
        {"word": "Aujourd'hui", "start": 0.6, "end": 1.0},
        {"word": "on", "start": 1.0, "end": 1.1},
        {"word": "parle", "start": 1.1, "end": 1.3},
        {"word": "de", "start": 1.3, "end": 1.4},
        {"word": "ça!", "start": 1.4, "end": 1.6},
        {"word": "C'est", "start": 1.7, "end": 1.9},
        {"word": "important.", "start": 1.9, "end": 2.3},
    ]
    
    boundaries = manager.find_sentence_boundaries(test_words, 0, 3)
    print(f"Frontières trouvées: {boundaries}")
