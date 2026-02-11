"""
Module d'optimisation des hooks (3 premières secondes)
Analyse et améliore le potentiel d'accroche des clips viraux
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from rich.console import Console

console = Console()

# Constantes de configuration du scoring des hooks
DEFAULT_MIN_HOOK_SCORE: float = 0.4
DEFAULT_HOOK_DURATION: float = 3.0
DEFAULT_SEARCH_WINDOW: float = 10.0
BASE_HOOK_SCORE: float = 0.5
PATTERN_MATCH_BONUS: float = 0.15
WEAK_WORD_PENALTY: float = 0.15
MAX_POWER_WORD_BONUS: float = 0.2
PER_POWER_WORD_BONUS: float = 0.05
HOOK_TOO_SHORT_PENALTY: float = 0.1
HOOK_TOO_LONG_PENALTY: float = 0.1
OPTIMAL_LENGTH_BONUS: float = 0.05
PROPER_START_BONUS: float = 0.05
EXPRESSIVE_PUNCTUATION_BONUS: float = 0.05
MIN_HOOK_WORD_COUNT: int = 3
MAX_HOOK_WORD_COUNT: int = 20
OPTIMAL_MIN_WORD_COUNT: int = 5
OPTIMAL_MAX_WORD_COUNT: int = 15


@dataclass
class HookAnalysis:
    """Résultat de l'analyse d'un hook"""
    score: float  # 0-1
    reasons: List[str]  # Raisons du score
    suggested_start: Optional[float]  # Nouveau timestamp de début suggéré
    hook_text: str  # Texte du hook
    hook_type: str  # Type de hook détecté


# Patterns de hooks puissants (phrases qui captent l'attention)
HOOK_PATTERNS = {
    # Questions rhétoriques (très engageantes)
    "question": [
        r"^(est-ce que|pourquoi|comment|qu['']est-ce"
        r"|savez-vous|vous savez|tu sais|c['']est quoi"
        r"|what|why|how|do you know|did you know)",
        r"\?$",
        r"^(et si|imagine|imaginez)",
    ],
    # Statements chocs / controverses
    "controversy": [
        r"(personne ne|tout le monde|jamais|toujours|impossible|incroyable|choquant|scandale|secret)",
        r"(nobody|everyone|never|always|impossible|incredible|shocking|scandal|secret)",
        r"(la vérité|the truth|en fait|actually|contrairement)",
    ],
    # Chiffres et statistiques (crédibilité + curiosité)
    "statistics": [
        r"\b\d+[%€$£]\b",
        r"\b\d+\s*(millions?|milliards?|billions?|k|K)\b",
        r"\b(premier|première|1er|1ère|numéro 1|#1|top \d+)\b",
    ],
    # Promesses de valeur
    "value_promise": [
        r"(je vais vous|i['']m going to|voici|here['']s|découvrez|discover)",
        r"(astuce|conseil|secret|hack|tips?|trick)",
        r"(en \d+ (secondes?|minutes?|étapes?))",
        r"(gratuit|free|sans payer)",
    ],
    # Urgence / FOMO
    "urgency": [
        r"(maintenant|now|aujourd['']hui|today|vite|quick|urgent|dernière chance|last chance)",
        r"(avant qu['']il|before it['']s|ne ratez pas|don['']t miss)",
    ],
    # Storytelling (accroche narrative)
    "story": [
        r"^(il y a \d+|there was|\d+ ans|years ago|un jour|one day|cette fois|this time)",
        r"(histoire vraie|true story|ça m['']est arrivé|it happened to me)",
        r"^(alors|so|donc|well|ok so|bon alors)",
    ],
    # Interpellation directe
    "direct_address": [
        r"^(vous|tu|toi|you|hey|salut|bonjour|yo|les (gars|amis|gens))",
        r"(écoute|écoutez|listen|regarde|regardez|watch|attends|attendez|wait)",
    ],
}

# Mots faibles qui réduisent l'impact d'un hook
WEAK_WORDS = [
    r"^(euh|hum|hmm|uh|um|er|ah|oh)",
    r"^(donc|so|alors|well|bon)\s*$",
    r"^(je pense que|i think|peut-être|maybe|probablement|probably)",
    r"^(en fait|actually|basically|genre|like)\s+",
]

# Mots puissants qui augmentent l'impact
POWER_WORDS = [
    "incroyable", "incredible", "amazing", "choquant", "shocking",
    "secret", "révélation", "révéler", "reveal", "découvrir", "discover",
    "jamais", "never", "toujours", "always", "premier", "first",
    "meilleur", "best", "pire", "worst", "erreur", "mistake",
    "gratuit", "free", "maintenant", "now", "urgent", "vite", "quick",
    "million", "milliard", "billion", "%", "euros", "dollars",
]


class HookOptimizer:
    """
    Optimise les hooks des clips pour maximiser l'engagement.

    Les 3 premières secondes sont cruciales pour la rétention:
    - 65% des viewers décident de rester ou partir dans les 3 premières secondes
    - Un bon hook peut doubler le taux de rétention
    """

    def __init__(
        self,
        min_hook_score: float = DEFAULT_MIN_HOOK_SCORE,
        hook_duration: float = DEFAULT_HOOK_DURATION,
        search_window: float = DEFAULT_SEARCH_WINDOW
    ):
        """
        Args:
            min_hook_score: Score minimum acceptable pour un hook
            hook_duration: Durée du hook à analyser (en secondes)
            search_window: Fenêtre de recherche pour un meilleur hook (en secondes)
        """
        self.min_hook_score = min_hook_score
        self.hook_duration = hook_duration
        self.search_window = search_window

    def analyze_hook(
        self,
        text: str,
        start_time: float = 0
    ) -> HookAnalysis:
        """
        Analyse le potentiel d'un hook.

        Args:
            text: Texte des premières secondes
            start_time: Timestamp de début

        Returns:
            HookAnalysis avec score et suggestions
        """
        if not text or not text.strip():
            return HookAnalysis(
                score=0.0,
                reasons=["Pas de texte dans le hook"],
                suggested_start=None,
                hook_text="",
                hook_type="empty"
            )

        text_lower = text.lower().strip()
        score = BASE_HOOK_SCORE  # Score de base
        reasons = []
        hook_type = "neutral"

        # 1. Détecter les patterns de hooks puissants
        best_pattern_score = 0
        for pattern_type, patterns in HOOK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_score = PATTERN_MATCH_BONUS
                    if best_pattern_score < pattern_score:
                        best_pattern_score = pattern_score
                        hook_type = pattern_type
                    break

        if best_pattern_score > 0:
            score += best_pattern_score
            reasons.append(f"Pattern '{hook_type}' détecté (+{best_pattern_score:.0%})")

        # 2. Détecter les mots faibles (pénalité)
        for weak_pattern in WEAK_WORDS:
            if re.search(weak_pattern, text_lower, re.IGNORECASE):
                score -= WEAK_WORD_PENALTY
                reasons.append(f"Mot faible détecté (-15%)")
                break

        # 3. Détecter les mots puissants (bonus)
        power_count = 0
        for word in POWER_WORDS:
            if word.lower() in text_lower:
                power_count += 1

        if power_count > 0:
            bonus = min(MAX_POWER_WORD_BONUS, power_count * PER_POWER_WORD_BONUS)
            score += bonus
            reasons.append(f"{power_count} mot(s) puissant(s) (+{bonus:.0%})")

        # 4. Longueur du hook (ni trop court, ni trop long)
        word_count = len(text.split())
        if word_count < 3:
            score -= 0.1
            reasons.append("Hook trop court (-10%)")
        elif word_count > 20:
            score -= 0.1
            reasons.append("Hook trop long (-10%)")
        elif 5 <= word_count <= 15:
            score += 0.05
            reasons.append("Longueur optimale (+5%)")

        # 5. Commence par une majuscule / phrase complète
        if text[0].isupper():
            score += 0.05
            reasons.append("Début de phrase propre (+5%)")

        # 6. Présence de ponctuation expressive
        if "!" in text or "?" in text:
            score += 0.05
            reasons.append("Ponctuation expressive (+5%)")

        # Normaliser le score entre 0 et 1
        score = max(0.0, min(1.0, score))

        return HookAnalysis(
            score=score,
            reasons=reasons,
            suggested_start=None,
            hook_text=text,
            hook_type=hook_type
        )

    def find_better_hook(
        self,
        words: List[Dict[str, Any]],
        current_start: float,
        clip_end: float,
        min_clip_duration: float = 30.0
    ) -> Tuple[float, HookAnalysis]:
        """
        Cherche un meilleur point de départ dans une fenêtre de recherche.

        Args:
            words: Liste de mots avec timestamps [{word, start, end}, ...]
            current_start: Timestamp de début actuel
            clip_end: Timestamp de fin du clip
            min_clip_duration: Durée minimum du clip à respecter

        Returns:
            Tuple (nouveau_start, HookAnalysis)
        """
        if not words:
            return current_start, self.analyze_hook("")

        best_start = current_start
        best_analysis = self.analyze_hook("")

        # Définir la fenêtre de recherche
        search_end = min(current_start + self.search_window, clip_end - min_clip_duration)

        # Trouver les mots dans la fenêtre
        window_words = [
            w for w in words
            if current_start <= w.get('start', w.get('start_time', 0)) <= search_end
        ]

        if not window_words:
            # Pas de mots dans la fenêtre, analyser le texte actuel
            current_words = [
                w for w in words
                if current_start <= w.get('start', w.get('start_time', 0)) <= current_start + self.hook_duration
            ]
            hook_text = " ".join(w.get('word', w.get('text', '')) for w in current_words)
            return current_start, self.analyze_hook(hook_text)

        # Tester différents points de départ (début de phrase)
        potential_starts = []

        for i, word in enumerate(window_words):
            word_text = word.get('word', word.get('text', ''))
            word_start = word.get('start', word.get('start_time', 0))

            # Détecter les débuts de phrase potentiels
            is_sentence_start = (
                i == 0 or  # Premier mot
                (word_text[0].isupper() if word_text else False) or  # Majuscule
                (any(window_words[i-1].get('word', '').endswith(p) for p in '.!?') if i > 0 else False)
            )

            if is_sentence_start:
                potential_starts.append((word_start, i))

        # Si pas de début de phrase trouvé, utiliser le début actuel
        if not potential_starts:
            potential_starts = [(current_start, 0)]

        # Évaluer chaque point de départ potentiel
        for start_time, start_idx in potential_starts:
            # Récupérer les mots du hook (3 premières secondes)
            hook_end = start_time + self.hook_duration
            hook_words = [
                w for w in words
                if start_time <= w.get('start', w.get('start_time', 0)) <= hook_end
            ]

            hook_text = " ".join(w.get('word', w.get('text', '')) for w in hook_words)
            analysis = self.analyze_hook(hook_text, start_time)

            if analysis.score > best_analysis.score:
                best_start = start_time
                best_analysis = analysis

        # Mettre à jour le suggested_start
        if best_start != current_start:
            best_analysis.suggested_start = best_start

        return best_start, best_analysis

    def optimize_moments(
        self,
        moments: List[Any],
        words: List[Dict[str, Any]],
        min_clip_duration: float = 30.0
    ) -> List[Tuple[Any, HookAnalysis]]:
        """
        Optimise les hooks de plusieurs moments.

        Args:
            moments: Liste de moments viraux
            words: Tous les mots de la transcription
            min_clip_duration: Durée minimum des clips

        Returns:
            Liste de tuples (moment_optimisé, HookAnalysis)
        """
        results = []

        # Convertir les mots au bon format si nécessaire (une seule fois)
        formatted_words = []
        for w in words:
            if hasattr(w, 'word'):
                formatted_words.append({
                    'word': w.word,
                    'start': w.start,
                    'end': w.end
                })
            else:
                formatted_words.append(w)

        for moment in moments:
            # Gérer les deux cas: objet ViralMoment ou dictionnaire
            if hasattr(moment, 'start_time'):
                start = moment.start_time
                end = moment.end_time
            elif isinstance(moment, dict):
                start = moment.get('start_time', 0)
                end = moment.get('end_time', 0)
            else:
                start = 0
                end = 0

            new_start, analysis = self.find_better_hook(
                formatted_words, start, end, min_clip_duration
            )

            # Mettre à jour le moment si un meilleur hook est trouvé
            if new_start != start and analysis.score >= self.min_hook_score:
                if hasattr(moment, 'start_time'):
                    moment.start_time = new_start
                else:
                    moment['start_time'] = new_start

                console.print(
                    f"  [green]Hook optimisé:[/green]"
                    f" {start:.1f}s -> {new_start:.1f}s"
                    f" (score: {analysis.score:.2f})"
                )

            results.append((moment, analysis))

        return results

    def validate_hook(
        self,
        words: List[Dict[str, Any]],
        start_time: float
    ) -> HookAnalysis:
        """
        Valide simplement un hook sans chercher d'alternative.

        Args:
            words: Liste de mots
            start_time: Timestamp de début

        Returns:
            HookAnalysis du hook actuel
        """
        hook_end = start_time + self.hook_duration
        hook_words = [
            w for w in words
            if start_time <= w.get('start', w.get('start_time', 0)) <= hook_end
        ]

        hook_text = " ".join(w.get('word', w.get('text', '')) for w in hook_words)
        return self.analyze_hook(hook_text, start_time)


def optimize_clip_hooks(
    moments: List[Any],
    words: List[Any],
    min_hook_score: float = 0.4,
    min_clip_duration: float = 30.0
) -> List[Tuple[Any, HookAnalysis]]:
    """
    Fonction utilitaire pour optimiser les hooks des clips.

    Args:
        moments: Liste de moments viraux
        words: Liste de mots avec timestamps
        min_hook_score: Score minimum pour un bon hook
        min_clip_duration: Durée minimum des clips

    Returns:
        Liste de tuples (moment, HookAnalysis)
    """
    optimizer = HookOptimizer(min_hook_score=min_hook_score)
    return optimizer.optimize_moments(moments, words, min_clip_duration)


if __name__ == "__main__":
    # Tests
    optimizer = HookOptimizer()

    test_hooks = [
        "Bonjour à tous, aujourd'hui on va parler de...",
        "Est-ce que vous saviez que 90% des gens font cette erreur?",
        "Euh... donc voilà quoi",
        "INCROYABLE! Ce secret va changer votre vie!",
        "Il y a 2 ans, j'ai découvert quelque chose qui a tout changé",
        "Personne ne vous dira jamais ça, mais...",
        "3 astuces pour gagner 1000€ par mois",
    ]

    for hook in test_hooks:
        analysis = optimizer.analyze_hook(hook)
        print(f"\n'{hook[:50]}...'")
        print(f"  Score: {analysis.score:.2f} | Type: {analysis.hook_type}")
        for reason in analysis.reasons:
            print(f"  - {reason}")
