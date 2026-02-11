#!/usr/bin/env python3
"""
Test complet du pipeline ClipGenius beta

Ce script teste tous les modules du pipeline sans nécessiter de vidéo externe.
Il vérifie que les imports et les logiques fonctionnent correctement.

Usage:
    python test_pipeline.py              # Test complet
    python test_pipeline.py --quick      # Test rapide (imports seulement)
    python test_pipeline.py --with-video # Test avec une vraie vidéo (si disponible)
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import argparse

# Ajouter le dossier racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constantes de test
TEST_NAME_COLUMN_WIDTH = 35
STATUS_COLUMN_WIDTH = 8

# Rich pour l'affichage
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    # Fallback sans rich
    class FakeConsole:
        def print(self, *args, **kwargs):
            text = args[0] if args else ""
            # Nettoyer les tags rich
            import re
            text = re.sub(r'\[.*?\]', '', str(text))
            print(text)
    console = FakeConsole()

    # Fake Panel pour le fallback
    class Panel:
        def __init__(self, text, **kwargs):
            self.text = text
        def __str__(self):
            return str(self.text)

    # Fake Table pour le fallback
    class Table:
        def __init__(self, **kwargs):
            self.rows = []
            self.columns = []
        def add_column(self, name, **kwargs):
            self.columns.append(name)
        def add_row(self, *args, **kwargs):
            self.rows.append(args)
        def __str__(self):
            lines = [" | ".join(self.columns)]
            for row in self.rows:
                lines.append(" | ".join(str(c) for c in row))
            return "\n".join(lines)


@dataclass
class TestResult:
    """Résultat d'un test"""
    name: str
    success: bool
    message: str
    details: Optional[str] = None


def test_imports() -> List[TestResult]:
    """Teste tous les imports des modules"""
    results = []

    modules = [
        ("src.downloader", "VideoDownloader"),
        ("src.viral_detector", "ViralMomentDetector, ViralMoment"),
        ("src.smart_cropper", "SmartCropper, FocusPoint, CropResult"),
        ("src.clip_generator", "ClipGenerator, ClipConfig"),
        ("src.subtitles", "add_animated_subtitles, SubtitleGenerator"),
        ("src.ai_analyzer", "AIViralAnalyzer, TranscriptSegment"),
        ("src.hook_optimizer", "HookOptimizer, HookAnalysis"),
        ("src.advanced_audio_analyzer", "AdvancedAudioAnalyzer, AudioEvent"),
        ("src.enriched_subtitles", "EnrichedSubtitleProcessor, SubtitleStyle"),
        ("src.adaptive_duration", "AdaptiveDurationManager, Platform"),
    ]

    for module_name, classes in modules:
        try:
            module = __import__(module_name, fromlist=classes.split(", "))
            # Vérifier que les classes existent
            for cls in classes.split(", "):
                cls = cls.strip()
                if not hasattr(module, cls):
                    results.append(TestResult(
                        name=f"Import {module_name}",
                        success=False,
                        message=f"Classe {cls} non trouvée"
                    ))
                    continue
            results.append(TestResult(
                name=f"Import {module_name}",
                success=True,
                message=f"OK ({classes})"
            ))
        except Exception as e:
            results.append(TestResult(
                name=f"Import {module_name}",
                success=False,
                message=str(e)
            ))

    return results


def test_hook_optimizer() -> List[TestResult]:
    """Teste le module d'optimisation des hooks"""
    results = []

    try:
        from src.hook_optimizer import HookOptimizer, HookAnalysis

        optimizer = HookOptimizer(min_hook_score=0.4)

        # Test de textes avec différentes qualités de hook
        test_cases = [
            # Bon hook avec mot puissant et question
            {
                "text": "Savez-vous pourquoi 90% des gens echouent?",
                "expected_score_min": 0.5
            },
            # Hook faible avec mot d'hésitation
            {
                "text": "Euh alors voila bon",
                "expected_score_max": 0.4
            },
        ]

        for i, case in enumerate(test_cases):
            analysis = optimizer.analyze_hook(case["text"])

            if "expected_score_min" in case:
                success = analysis.score >= case["expected_score_min"]
                msg = f"Score: {analysis.score:.2f} (attendu >= {case['expected_score_min']})"
            else:
                success = analysis.score <= case["expected_score_max"]
                msg = f"Score: {analysis.score:.2f} (attendu <= {case['expected_score_max']})"

            results.append(TestResult(
                name=f"HookOptimizer test {i+1}",
                success=success,
                message=msg,
                details=f"Type: {analysis.hook_type}, Reasons: {len(analysis.reasons)}"
            ))

    except Exception as e:
        results.append(TestResult(
            name="HookOptimizer",
            success=False,
            message=str(e)
        ))

    return results


def test_enriched_subtitles() -> List[TestResult]:
    """Teste le module de sous-titres enrichis"""
    results = []

    try:
        from src.enriched_subtitles import EnrichedSubtitleProcessor, SubtitleStyle

        processor = EnrichedSubtitleProcessor()

        # Test de classification de mots
        test_words = [
            {"word": "Incroyable", "start": 0.0, "end": 0.5},  # Mot-clé
            {"word": "90%", "start": 0.5, "end": 0.8},        # Nombre
            {"word": "mais", "start": 0.8, "end": 1.0},       # Emphase
            {"word": "normal", "start": 1.0, "end": 1.3},     # Normal
        ]

        enriched = processor.process_words(test_words)

        expected = [
            ("Incroyable", "keyword"),
            ("90%", "number"),
            ("mais", "emphasis"),
            ("normal", "normal"),
        ]

        for (exp_word, exp_type), enriched_word in zip(expected, enriched):
            success = enriched_word.importance == exp_type
            results.append(TestResult(
                name=f"EnrichedSubtitles '{exp_word}'",
                success=success,
                message=f"Type: {enriched_word.importance} (attendu: {exp_type})",
                details=f"Couleur: {enriched_word.color}"
            ))

        # Test de génération CSS
        css = processor.generate_enhanced_css()
        css_valid = ".word-keyword" in css and ".word-number" in css
        results.append(TestResult(
            name="EnrichedSubtitles CSS",
            success=css_valid,
            message=f"CSS généré ({len(css)} caractères)",
            details="Contient .word-keyword et .word-number"
        ))

    except Exception as e:
        results.append(TestResult(
            name="EnrichedSubtitles",
            success=False,
            message=str(e)
        ))

    return results


def test_adaptive_duration() -> List[TestResult]:
    """Teste le module de durées adaptatives"""
    results = []

    try:
        from src.adaptive_duration import (
            AdaptiveDurationManager, Platform,
            get_platform_spec, PLATFORM_SPECS
        )

        # Test des specs de plateforme
        for platform in [Platform.TIKTOK, Platform.REELS, Platform.SHORTS]:
            spec = get_platform_spec(platform)
            valid = spec.min_duration > 0 and spec.max_duration > spec.min_duration
            results.append(TestResult(
                name=f"Platform spec {platform.value}",
                success=valid,
                message=f"{spec.min_duration}-{spec.max_duration}s (ideal: {spec.ideal_duration}s)"
            ))

        # Test de détection de type de contenu
        manager = AdaptiveDurationManager()

        # Simuler un segment de podcast
        test_segments = [
            {"text": "Bienvenue dans ce podcast où nous allons parler de philosophie", "start": 0, "end": 5},
            {"text": "C'est une réflexion profonde sur le sens de la vie", "start": 5, "end": 10},
        ]

        content_type = manager.detect_content_type(test_segments)
        results.append(TestResult(
            name="Content type detection",
            success=content_type is not None,
            message=f"Type détecté: {content_type}"
        ))

    except Exception as e:
        results.append(TestResult(
            name="AdaptiveDuration",
            success=False,
            message=str(e)
        ))

    return results


def test_advanced_audio() -> List[TestResult]:
    """Teste le module d'analyse audio avancée"""
    results = []

    try:
        from src.advanced_audio_analyzer import (
            AdvancedAudioAnalyzer, AudioEvent, AudioEventType
        )

        # Test d'instanciation
        analyzer = AdvancedAudioAnalyzer()

        results.append(TestResult(
            name="AdvancedAudioAnalyzer init",
            success=True,
            message="Analyseur créé avec succès"
        ))

        # Vérifier les types d'événements
        event_types = [e for e in AudioEventType]
        results.append(TestResult(
            name="AudioEventType enum",
            success=len(event_types) >= 5,
            message=f"{len(event_types)} types d'événements définis",
            details=", ".join([e.value for e in event_types[:5]])
        ))

    except Exception as e:
        results.append(TestResult(
            name="AdvancedAudioAnalyzer",
            success=False,
            message=str(e)
        ))

    return results


def test_main_cli() -> List[TestResult]:
    """Teste que le CLI principal fonctionne"""
    results = []

    try:
        # Importer le module main
        import main

        # Vérifier que la fonction main existe et a les bons paramètres
        import inspect
        sig = inspect.signature(main.main)
        params = list(sig.parameters.keys())

        required_params = [
            "url", "input_file", "output_dir", "platform",
            "enriched_subtitles", "optimize_hooks", "advanced_audio", "adaptive_duration"
        ]

        missing = [p for p in required_params if p not in params]

        results.append(TestResult(
            name="CLI main function",
            success=len(missing) == 0,
            message=f"{len(params)} paramètres CLI",
            details=f"Manquants: {missing}" if missing else "Tous les params présents"
        ))

    except Exception as e:
        results.append(TestResult(
            name="CLI main",
            success=False,
            message=str(e)
        ))

    return results


def run_all_tests(quick: bool = False) -> Tuple[int, int]:
    """
    Exécute tous les tests

    Returns:
        Tuple (succès, échecs)
    """
    console.print(Panel("[bold cyan]ClipGenius beta - Test Suite[/bold cyan]",
                        border_style="cyan"))

    all_results = []

    # Tests d'imports (toujours exécutés)
    console.print("\n[bold]1. Tests d'imports[/bold]")
    all_results.extend(test_imports())

    if not quick:
        # Tests fonctionnels
        console.print("\n[bold]2. Tests du Hook Optimizer[/bold]")
        all_results.extend(test_hook_optimizer())

        console.print("\n[bold]3. Tests des Sous-titres Enrichis[/bold]")
        all_results.extend(test_enriched_subtitles())

        console.print("\n[bold]4. Tests des Durées Adaptatives[/bold]")
        all_results.extend(test_adaptive_duration())

        console.print("\n[bold]5. Tests de l'Analyse Audio[/bold]")
        all_results.extend(test_advanced_audio())

        console.print("\n[bold]6. Tests du CLI[/bold]")
        all_results.extend(test_main_cli())

    # Afficher les résultats
    console.print("\n")
    table = Table(title="Résultats des tests", show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", width=TEST_NAME_COLUMN_WIDTH)
    table.add_column("Status", justify="center", width=STATUS_COLUMN_WIDTH)
    table.add_column("Message", style="dim")

    success_count = 0
    fail_count = 0

    for result in all_results:
        if result.success:
            status = "[green]OK[/green]"
            success_count += 1
        else:
            status = "[red]FAIL[/red]"
            fail_count += 1

        table.add_row(result.name, status, result.message)

    console.print(table)

    # Résumé
    console.print("\n")
    if fail_count == 0:
        console.print(Panel(
            f"[bold green]Tous les tests passent! ({success_count}/{success_count})[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]{success_count} succès, {fail_count} échecs[/bold yellow]",
            border_style="yellow"
        ))

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description="Test du pipeline ClipGenius beta")
    parser.add_argument("--quick", action="store_true", help="Test rapide (imports seulement)")
    parser.add_argument("--with-video", type=str, help="Tester avec une vraie vidéo")

    args = parser.parse_args()

    success, fail = run_all_tests(quick=args.quick)

    if args.with_video:
        console.print("\n[bold]Test avec vidéo...[/bold]")
        console.print(f"[dim]Vidéo: {args.with_video}[/dim]")
        console.print("[yellow]Note: Ce test nécessite toutes les dépendances installées[/yellow]")
        # TODO: Implémenter le test avec vidéo réelle

    # Exit code
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
