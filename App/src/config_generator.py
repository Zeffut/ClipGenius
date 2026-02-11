"""Generation de configuration optimale a partir de l'analyse audio."""

from typing import List, Optional, Tuple
from rich.console import Console

from .auto_config import (
    ContentType, EmotionProfile, QuickAnalysisResult, GeneratedConfig,
    CONTENT_DURATION_MAP, PLATFORM_DURATION_ADJUSTMENTS
)

console = Console()


class ConfigGenerator:
    """
    Génère une configuration optimale basée sur l'analyse.
    """

    def generate(
        self,
        analysis: QuickAnalysisResult,
        platform: str = "reels"
    ) -> GeneratedConfig:
        """
        Génère une configuration basée sur l'analyse.

        Args:
            analysis: Résultat de l'analyse rapide
            platform: Plateforme cible

        Returns:
            GeneratedConfig avec tous les paramètres optimisés
        """
        reasoning = []

        # 1. Durées basées sur le contenu et la plateforme
        base_min, base_max = CONTENT_DURATION_MAP.get(
            analysis.content_type,
            CONTENT_DURATION_MAP[ContentType.UNKNOWN]
        )

        # Ajustement plateforme
        adj_min, adj_max = PLATFORM_DURATION_ADJUSTMENTS.get(platform, (0, 0))
        min_duration = max(15, base_min + adj_min)
        max_duration = max(min_duration + 15, base_max + adj_max)

        reasoning.append(
            f"Durées: {min_duration:.0f}-{max_duration:.0f}s "
            f"({analysis.content_type.value} + {platform})"
        )

        # 2. Score viral minimum basé sur la confiance
        if analysis.content_confidence >= 0.6:
            min_score = 0.55
            reasoning.append("Score minimum: 0.55 (haute confiance)")
        else:
            min_score = 0.65
            reasoning.append("Score minimum: 0.65 (confiance modérée)")

        # 3. Effets visuels basés sur le profil émotionnel
        zoom_style, zoom_factor = self._determine_zoom(analysis, reasoning)
        color_grading = self._determine_color_grading(analysis, reasoning)
        sharpening = self._determine_sharpening(analysis, reasoning)
        vignette_enabled, vignette_strength = self._determine_vignette(analysis, reasoning)

        # 4. Style de sous-titres
        theme, max_words, emojis = self._determine_subtitles(analysis, reasoning)

        # 5. Options de traitement
        smart_crop = True
        optimize_hooks = analysis.content_type != ContentType.NEWS
        advanced_audio = True

        return GeneratedConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_viral_score=min_score,
            zoom_style=zoom_style,
            zoom_factor=zoom_factor,
            color_grading=color_grading,
            sharpening_strength=sharpening,
            vignette_enabled=vignette_enabled,
            vignette_strength=vignette_strength,
            subtitle_theme=theme,
            subtitle_max_words=max_words,
            subtitle_emojis=emojis,
            smart_crop=smart_crop,
            optimize_hooks=optimize_hooks,
            advanced_audio=advanced_audio,
            detected_content_type=analysis.content_type.value,
            content_confidence=analysis.content_confidence,
            reasoning=reasoning
        )

    def _determine_zoom(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> Tuple[str, float]:
        """Détermine le style et facteur de zoom"""
        energy = analysis.energy_profile
        emotion = analysis.dominant_emotion

        if energy.variance > 0.4:
            reasoning.append("Zoom: pulse (1.06x) - contenu très dynamique")
            return "pulse", 1.06
        elif energy.calm_ratio > 0.7:
            reasoning.append("Zoom: breathing (1.03x) - contenu calme")
            return "breathing", 1.03
        elif emotion in [EmotionProfile.EXCITED, EmotionProfile.VERY_EXCITED]:
            reasoning.append("Zoom: pulse (1.05x) - haute énergie")
            return "pulse", 1.05
        else:
            reasoning.append("Zoom: ease_out (1.04x) - standard")
            return "ease_out", 1.04

    def _determine_color_grading(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> str:
        """Détermine le style de color grading"""
        emotion = analysis.dominant_emotion
        content = analysis.content_type

        if emotion in [EmotionProfile.EXCITED, EmotionProfile.VERY_EXCITED]:
            reasoning.append("Couleurs: vibrant - haute énergie")
            return "vibrant"
        elif content in [ContentType.PODCAST, ContentType.INTERVIEW]:
            reasoning.append("Couleurs: warm - conversation")
            return "warm"
        elif emotion == EmotionProfile.CALM:
            reasoning.append("Couleurs: cool - contenu calme")
            return "cool"
        elif content == ContentType.NEWS:
            reasoning.append("Couleurs: cool - professionnel")
            return "cool"
        elif analysis.audio_events.dramatic_silence_count >= 2:
            reasoning.append("Couleurs: cinematic - silences dramatiques")
            return "cinematic"
        else:
            reasoning.append("Couleurs: warm - standard")
            return "warm"

    def _determine_sharpening(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> float:
        """Détermine la force du sharpening"""
        if analysis.excitement_peaks > 5:
            reasoning.append("Netteté: 0.35 - contenu dynamique")
            return 0.35
        elif analysis.energy_profile.calm_ratio > 0.6:
            reasoning.append("Netteté: 0.20 - contenu calme")
            return 0.20
        else:
            reasoning.append("Netteté: 0.28 - standard")
            return 0.28

    def _determine_vignette(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> Tuple[bool, float]:
        """Détermine si la vignette est activée et sa force"""
        events = analysis.audio_events

        if events.dramatic_silence_count >= 2:
            reasoning.append("Vignette: activée (0.15) - silences dramatiques")
            return True, 0.15
        elif analysis.content_type in [ContentType.PODCAST, ContentType.INTERVIEW]:
            reasoning.append("Vignette: activée (0.10) - focus conversation")
            return True, 0.10
        else:
            return False, 0.0

    def _determine_subtitles(
        self,
        analysis: QuickAnalysisResult,
        reasoning: List[str]
    ) -> Tuple[str, int, bool]:
        """Détermine le style des sous-titres"""
        events = analysis.audio_events
        speech = analysis.speech
        content = analysis.content_type

        # Thème
        if events.laughter_count >= 3:
            theme = "neon"
            emojis = True
            reasoning.append("Sous-titres: neon + emojis - présence de rires")
        elif content in [ContentType.PODCAST, ContentType.INTERVIEW, ContentType.NEWS]:
            theme = "professional"
            emojis = False
            reasoning.append("Sous-titres: professional - contenu formel")
        elif content == ContentType.MOTIVATIONAL:
            theme = "viral"
            emojis = True
            reasoning.append("Sous-titres: viral + emojis - contenu motivant")
        else:
            theme = "viral"
            emojis = content not in [ContentType.TUTORIAL, ContentType.NEWS]
            reasoning.append(f"Sous-titres: viral - standard")

        # Nombre de mots par segment
        if speech.rate > 4:
            max_words = 4
            reasoning.append("Max mots: 4 - parole rapide")
        elif speech.rate < 2:
            max_words = 2
            reasoning.append("Max mots: 2 - parole lente")
        else:
            max_words = 3

        return theme, max_words, emojis


# =============================================================================
# LLM CONTENT TYPE DETECTOR
# =============================================================================

def detect_content_type_with_llm(
    transcription_text: str,
    audio_based_type: ContentType,
    audio_confidence: float
) -> Tuple[ContentType, float, str]:
    """
    Utilise le LLM local (Phi-4-mini) pour affiner la détection du type de contenu.

    Combine l'analyse audio avec l'analyse sémantique de la transcription pour
    améliorer la précision de détection.

    Args:
        transcription_text: Texte transcrit (premiers ~500 mots)
        audio_based_type: Type détecté par l'analyse audio
        audio_confidence: Confiance de la détection audio (0-1)

    Returns:
        Tuple (ContentType final, confiance finale, raisonnement)
    """
    try:
        from .local_llm import LocalLLM

        # Prendre uniquement les premiers 500 mots pour limiter les tokens
        words = transcription_text.split()[:500]
        sample_text = ' '.join(words)

        # Prompt pour le LLM
        prompt = f"""Tu es un expert en classification de contenu vidéo pour les réseaux sociaux (TikTok, Reels, Shorts).

Analyse cette transcription et détermine le type de contenu parmi:
- PODCAST: Conversations longues, interviews approfondies, discussions
- INTERVIEW: Questions-réponses, format structuré
- TUTORIAL: Instructions, explications étape par étape, éducatif
- VLOG: Contenu personnel, journée type, storytelling
- COMEDY: Humour, sketches, blagues
- GAMING: Jeux vidéo, gameplay, commentaires de parties, Minecraft, streams gaming
- MUSIC: Chansons, performances musicales
- NEWS: Actualités, informations factuelles
- MOTIVATIONAL: Inspiration, développement personnel, encouragement (SANS lien avec gaming)

IMPORTANT:
- Si la transcription parle de JEUX VIDÉO (Minecraft, Fortnite, etc.) → GAMING, PAS motivational
- GAMING inclut: gameplay, exploration de mondes virtuels, construction, combat dans des jeux
- MOTIVATIONAL est uniquement pour: développement personnel, inspiration de vie, coaching
- NE CONFONDS PAS une voix énergique dans un jeu avec du contenu motivational

TRANSCRIPTION:
{sample_text}

ANALYSE AUDIO PRÉLIMINAIRE:
- Type détecté: {audio_based_type.value}
- Confiance: {audio_confidence:.0%}

INSTRUCTIONS:
1. Lis attentivement la transcription
2. Identifie le SUJET PRINCIPAL (jeu vidéo? développement perso? interview?)
3. Si c'est lié aux jeux vidéo → GAMING
4. Sinon, choisis parmi les autres types

Réponds UNIQUEMENT au format:
TYPE: [un seul mot parmi la liste]
CONFIANCE: [0-100]
RAISON: [1 phrase courte expliquant pourquoi]"""

        # Initialiser le LLM
        llm = LocalLLM()

        # Générer la réponse
        llm_response = llm.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,  # Bas pour plus de déterminisme
            stop=["\n\n", "TYPE:", "CONFIANCE:", "RAISON:"]
        )

        # Extraire le texte de la réponse
        response = llm_response.text

        # Parser la réponse
        lines = response.strip().split('\n')
        detected_type = audio_based_type  # Fallback
        confidence = audio_confidence
        reasoning = f"Détection audio: {audio_based_type.value}"

        for line in lines:
            line = line.strip()
            if line.startswith('TYPE:'):
                type_str = line.replace('TYPE:', '').strip().lower()
                # Mapper vers ContentType
                type_mapping = {
                    'podcast': ContentType.PODCAST,
                    'interview': ContentType.INTERVIEW,
                    'tutorial': ContentType.TUTORIAL,
                    'vlog': ContentType.VLOG,
                    'comedy': ContentType.COMEDY,
                    'gaming': ContentType.GAMING,
                    'music': ContentType.MUSIC,
                    'news': ContentType.NEWS,
                    'motivational': ContentType.MOTIVATIONAL,
                }
                detected_type = type_mapping.get(type_str, audio_based_type)

            elif line.startswith('CONFIANCE:'):
                try:
                    conf_str = line.replace('CONFIANCE:', '').strip().replace('%', '')
                    confidence = float(conf_str) / 100.0
                except Exception:
                    pass

            elif line.startswith('RAISON:'):
                reasoning = line.replace('RAISON:', '').strip()

        # Combiner avec l'analyse audio (pondération 70% LLM, 30% audio)
        if detected_type == audio_based_type:
            # Accord entre LLM et audio → boost confiance
            final_confidence = min(1.0, (confidence * 0.7 + audio_confidence * 0.3) * 1.3)
        else:
            # Désaccord → favoriser le LLM (il comprend le contexte mieux que l'audio)
            final_confidence = confidence * 0.85

        console.print(f"[dim]🤖 LLM: {detected_type.value} ({final_confidence:.0%}) - {reasoning}[/dim]")
        return detected_type, final_confidence, reasoning

    except Exception as e:
        console.print(f"[yellow]⚠️ LLM indisponible: {e}[/yellow]")
        # Fallback: utiliser uniquement l'analyse audio
        return audio_based_type, audio_confidence, f"Audio seul: {audio_based_type.value}"
