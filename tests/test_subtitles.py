#!/usr/bin/env python3
"""
Test rapide du système de sous-titres ASS/pysubs2
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

console = Console()

def test_imports():
    """Test 1: Vérifier que tous les imports fonctionnent"""
    console.print("[cyan]Test 1: Imports...[/cyan]")
    
    try:
        from src.subtitles import SubtitleGenerator, TranscriptionResult
        console.print("  ✓ SubtitleGenerator importé")
    except Exception as e:
        console.print(f"  ✗ Erreur SubtitleGenerator: {e}")
        return False
    
    try:
        from src.tiktok_captions import generate_tiktok_ass
        console.print("  ✓ tiktok_captions importé")
    except Exception as e:
        console.print(f"  ✗ Erreur tiktok_captions: {e}")
        return False
    
    try:
        from src.clip_generator import ClipGenerator
        console.print("  ✓ ClipGenerator importé")
    except Exception as e:
        console.print(f"  ✗ Erreur ClipGenerator: {e}")
        return False
    
    return True


def test_ass_generation():
    """Test 2: Générer un fichier ASS de test"""
    console.print("\n[cyan]Test 2: Génération ASS...[/cyan]")
    
    try:
        from dataclasses import dataclass
        from src.tiktok_captions import generate_tiktok_ass
        
        @dataclass
        class TestWord:
            word: str
            start: float
            end: float
        
        test_words = [
            TestWord("INCROYABLE", 0.0, 0.5),
            TestWord("moment", 0.5, 0.8),
            TestWord("100%", 0.8, 1.2),
        ]
        
        test_colors = {
            'primary_color': '#FFFFFF',
            'keyword_color': '#FF00FF',
            'number_color': '#00FFFF',
        }
        
        output = "/tmp/clipgenius_test.ass"
        result = generate_tiktok_ass(
            words=test_words,
            preset_colors=test_colors,
            output_path=output,
            style_config={'theme': 'gaming', 'max_words': 3}
        )
        
        if result and Path(output).exists():
            console.print(f"  ✓ Fichier ASS généré: {output}")
            # Lire et afficher un extrait
            with open(output, 'r') as f:
                content = f.read()
                if 'Keyword' in content and 'Dialogue' in content:
                    console.print("  ✓ Contenu ASS valide (styles + events)")
                    return True
        
        console.print("  ✗ Fichier ASS non généré")
        return False
        
    except Exception as e:
        console.print(f"  ✗ Erreur génération ASS: {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def test_subtitle_config():
    """Test 3: Vérifier la configuration des sous-titres"""
    console.print("\n[cyan]Test 3: Configuration...[/cyan]")
    
    try:
        from src.clip_generator import ClipConfig
        
        config = ClipConfig(
            add_subtitles=True,
            subtitle_enriched=True,
            subtitle_theme='gaming',
            subtitle_custom_colors={
                'keyword_color': '#FF00FF',
                'number_color': '#00FFFF'
            }
        )
        
        console.print(f"  ✓ add_subtitles: {config.add_subtitles}")
        console.print(f"  ✓ subtitle_enriched: {config.subtitle_enriched}")
        console.print(f"  ✓ subtitle_theme: {config.subtitle_theme}")
        console.print(f"  ✓ custom_colors: {len(config.subtitle_custom_colors)} couleurs")
        
        return True
        
    except Exception as e:
        console.print(f"  ✗ Erreur configuration: {e}")
        return False


if __name__ == "__main__":
    console.print("[bold cyan]Test du système de sous-titres ClipGenius[/bold cyan]\n")
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Génération ASS", test_ass_generation()))
    results.append(("Configuration", test_subtitle_config()))
    
    console.print("\n[bold]Résumé:[/bold]")
    all_passed = True
    for name, passed in results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        console.print("\n[bold green]✓ Tous les tests passent ! Le système de sous-titres est prêt.[/bold green]")
        console.print("\n[cyan]Prochaine étape:[/cyan]")
        console.print("  1. Redémarrer l'application: [bold]python app.py[/bold]")
        console.print("  2. Générer des clips avec sous-titres activés")
        sys.exit(0)
    else:
        console.print("\n[bold red]✗ Certains tests ont échoué[/bold red]")
        sys.exit(1)
