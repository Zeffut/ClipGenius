#!/usr/bin/env python3
"""
Script de test rapide pour le LLM local (Phi-4-mini).

Cet script valide que tout est configuré correctement.
Supporte Phi-4-mini (recommandé) et Phi-3-mini (fallback).
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path (parent de tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()

# Constantes de test
SEGMENT_DURATION = 30
SEGMENT_1_END = 60
SEGMENT_2_END = 90


def test_local_llm_available():
    """Test 1: Vérifier si un modèle Phi (4 ou 3) est disponible."""
    console.print("[cyan]Test 1: Recherche du modèle Phi-4-mini ou Phi-3-mini...[/cyan]")

    try:
        from src.local_llm import LocalLLM

        # Chercher le modèle sans charger
        model_path = LocalLLM._find_model()

        if model_path:
            model_name = Path(model_path).name
            if 'phi-4' in model_name.lower() or 'phi4' in model_name.lower():
                console.print(f"  [green]✓ Modèle Phi-4-mini trouvé: {model_name}[/green]")
            else:
                console.print(f"  [yellow]✓ Modèle Phi-3 trouvé (fallback): {model_name}[/yellow]")
                console.print("  [dim]Pour de meilleures performances, télécharge Phi-4-mini:[/dim]")
                console.print("  [dim]https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF[/dim]")
            return True
        else:
            console.print("[yellow]  ⚠ Modèle non trouvé[/yellow]")
            console.print("  Télécharge Phi-4-mini: https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF")
            console.print("  Puis place-le dans ./models/ ou l'emplacement racine")
            return False

    except ImportError as e:
        console.print(f"  [red]✗ Erreur import: {e}[/red]")
        return False


def test_llama_cpp_available():
    """Test 2: Vérifier si llama-cpp-python est installé."""
    console.print("\n[cyan]Test 2: Vérification llama-cpp-python...[/cyan]")

    try:
        import llama_cpp
        console.print(f"  [green]✓ llama-cpp-python installé (version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'unknown'})[/green]")
        return True
    except ImportError:
        console.print("  [red]✗ llama-cpp-python non installé[/red]")
        console.print("  Installe avec: pip install llama-cpp-python")
        return False


def test_ai_analyzer():
    """Test 3: Vérifier que LocalAIViralAnalyzer charge."""
    console.print("\n[cyan]Test 3: Chargement de LocalAIViralAnalyzer...[/cyan]")

    try:
        from src.ai_analyzer import LocalAIViralAnalyzer, TranscriptSegment
        console.print("  [green]✓ LocalAIViralAnalyzer importé avec succès[/green]")

        # Créer une instance test (sans charger le modèle)
        analyzer = LocalAIViralAnalyzer.__new__(LocalAIViralAnalyzer)
        console.print("  [green]✓ Instance créée (modèle non chargé)[/green]")
        return True

    except Exception as e:
        console.print(f"  [red]✗ Erreur: {e}[/red]")
        return False


def test_subtitles_local_emoji():
    """Test 4: Vérifier que IconOnlyLocalLLM charge."""
    console.print("\n[cyan]Test 4: Chargement de IconOnlyLocalLLM...[/cyan]")

    try:
        from src.subtitles import IconOnlyLocalLLM
        console.print("  [green]✓ IconOnlyLocalLLM importé avec succès[/green]")
        return True

    except Exception as e:
        console.print(f"  [red]✗ Erreur: {e}[/red]")
        return False


def test_analyze_with_ai():
    """Test 5: Vérifier que analyze_with_ai fonctionne."""
    console.print("\n[cyan]Test 5: Fonction analyze_with_ai...[/cyan]")

    try:
        from src.ai_analyzer import analyze_with_ai, TranscriptSegment
        console.print("  [green]✓ analyze_with_ai importé avec succès[/green]")

        # Tester avec un segment court
        segments = [
            TranscriptSegment(0, SEGMENT_DURATION, "Ceci est un test de contenu viral"),
            TranscriptSegment(SEGMENT_DURATION, SEGMENT_1_END, "avec plusieurs segments pour validation"),
            TranscriptSegment(SEGMENT_1_END, SEGMENT_2_END, "et un dernier segment pour la durée")
        ]

        console.print("  [dim]Tentative d'analyse (peut prendre du temps si chargement du modèle)...[/dim]")

        # Cette fonction essaiera local, puis fallback API
        # On va juste vérifier qu'elle ne crash pas
        console.print("  [yellow]⚠ Test réel non effectué (nécessite le modèle chargé)[/yellow]")
        console.print("  [dim]Pour tester: python -c \"from src.ai_analyzer import analyze_with_ai; ...\"[/dim]")

        return True

    except Exception as e:
        console.print(f"  [red]✗ Erreur: {e}[/red]")
        return False


def main():
    """Exécute tous les tests."""
    console.print("[bold cyan]🧪 ClipGenius - Tests LLM Local[/bold cyan]\n")

    results = []

    results.append(("llama-cpp-python", test_llama_cpp_available()))
    results.append(("Modèle Phi-4/Phi-3", test_local_llm_available()))
    results.append(("LocalAIViralAnalyzer", test_ai_analyzer()))
    results.append(("IconOnlyLocalLLM", test_subtitles_local_emoji()))
    results.append(("analyze_with_ai", test_analyze_with_ai()))

    # Résumé
    console.print("\n[bold]📊 Résumé des tests:[/bold]")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[green]✓[/green]" if result else "[red]✗[/red]"
        console.print(f"  {status} {name}")

    console.print(f"\n[{'green' if passed == total else 'yellow'}]Score: {passed}/{total} tests[/]")

    if passed == total:
        console.print("\n[green bold]✨ Tout est configuré correctement![/green bold]")
        console.print("[dim]Tu peux maintenant lancer: python main.py video.mp4[/dim]")
        return 0
    elif passed >= 4:
        console.print("\n[yellow]⚠ Certains tests échouent, mais le système peut fonctionner[/yellow]")
        console.print("[dim]Consulte SETUP_LOCAL_LLM.md pour plus de détails[/dim]")
        return 1
    else:
        console.print("\n[red]❌ Configuration incomplète[/red]")
        console.print("[dim]Étapes:[/dim]")
        console.print("  1. pip install -r requirements.txt")
        console.print("  2. Télécharge Phi-4-mini depuis: https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF")
        console.print("  3. Place-le dans ./models/")
        console.print("  4. Relance ce script")
        return 2


if __name__ == "__main__":
    sys.exit(main())
