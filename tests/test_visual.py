#!/usr/bin/env python3
"""
Test visuel des améliorations ClipGenius

Ce script génère un court clip (20-30 secondes) pour valider visuellement:
- Le tracking facial amélioré (lissage temporel, ease-in-out)
- Le blur-fill avec dégradé de transition
- L'effet de zoom ease-out avec breathing
- L'encodage haute qualité (CRF 18, preset slow)

Usage:
    python test_visual.py                    # Utilise la vidéo de test par défaut
    python test_visual.py --video path.mp4   # Vidéo personnalisée
    python test_visual.py --start 60 --duration 25  # Segment personnalisé
"""

import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass
class TestConfig:
    """Configuration du test visuel"""
    video_path: str
    output_dir: str = "output_test"
    start_time: float = 30.0     # Début du segment (secondes)
    duration: float = 25.0       # Durée du clip test
    
    # Options de test
    test_blur_fill: bool = True    # Tester le blur-fill
    test_zoom: bool = True         # Tester le zoom ease-out
    test_smart_crop: bool = True   # Tester le tracking facial
    test_high_quality: bool = True # Tester l'encodage HQ (CRF 18)
    
    # Nouvelles options cinématiques
    test_color_grading: bool = True   # Tester le color grading
    color_style: str = "warm"          # Style: warm, cool, vibrant, cinematic
    test_sharpening: bool = True       # Tester le sharpening
    test_vignette: bool = False        # Tester la vignette
    zoom_style: str = "ease_out"       # Style: ease_out, breathing, pulse
    test_ken_burns: bool = False       # Tester l'effet Ken Burns
    ken_burns_intensity: float = 0.02  # Intensité du Ken Burns (0-0.1)


def generate_test_clip(config: TestConfig) -> Optional[str]:
    """
    Génère un clip de test pour validation visuelle
    
    Returns:
        Chemin du clip généré ou None si erreur
    """
    from src.clip_generator import ClipGenerator, ClipConfig
    from src.viral_detector import ViralMoment
    
    console.print(Panel(
        "[bold cyan]ClipGenius - Test Visuel[/bold cyan]\n\n"
        f"Vidéo: {Path(config.video_path).name}\n"
        f"Segment: {config.start_time:.1f}s - {config.start_time + config.duration:.1f}s\n"
        f"Durée: {config.duration:.1f}s",
        title="Configuration"
    ))
    
    # Vérifier que la vidéo existe
    if not Path(config.video_path).exists():
        console.print(f"[red]Erreur: Vidéo introuvable: {config.video_path}[/red]")
        return None
    
    # Créer le dossier de sortie
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration du générateur
    clip_config = ClipConfig(
        # Format 9:16 pour mobile
        output_width=1080,
        output_height=1920,
        output_fps=30,
        
        # Durées (on override avec notre segment de test)
        min_clip_duration=config.duration - 5,
        max_clip_duration=config.duration + 5,
        min_viral_score=0.0,  # Accepter tout (c'est un test)
        max_clips=1,
        
        # Options de test
        enable_blur_fill=config.test_blur_fill,
        blur_strength=51,
        enable_zoom_effect=config.test_zoom,
        zoom_factor=1.05,
        zoom_style=config.zoom_style,
        smart_crop=config.test_smart_crop,
        
        # Qualité d'encodage
        crf=18 if config.test_high_quality else 23,
        preset="slow" if config.test_high_quality else "medium",
        video_profile="high",
        video_level="4.1",
        use_lanczos=config.test_high_quality,
        
        # Effets cinématiques
        enable_color_grading=config.test_color_grading,
        color_grading_style=config.color_style,
        enable_sharpening=config.test_sharpening,
        sharpening_strength=0.3,
        enable_vignette=config.test_vignette,
        vignette_strength=0.15,
        
        # Ken Burns effect
        enable_ken_burns=config.test_ken_burns,
        ken_burns_intensity=config.ken_burns_intensity,
        
        # Pas de sous-titres pour ce test (plus rapide)
        add_subtitles=False,
    )
    
    # Créer un "faux" moment viral pour notre segment de test
    test_moment = ViralMoment(
        start_time=config.start_time,
        end_time=config.start_time + config.duration,
        score=1.0,
        reason="Test visuel des améliorations"
    )
    
    console.print("\n[bold]Fonctionnalites testees:[/bold]")
    features = [
        ("Tracking facial lisse", config.test_smart_crop, "Kalman + Catmull-Rom spline"),
        ("Blur-fill cinematique", config.test_blur_fill, "vignette + saturation boost"),
        ("Zoom dynamique", config.test_zoom, f"style: {config.zoom_style}"),
        ("Ken Burns effect", config.test_ken_burns, f"intensite: {config.ken_burns_intensity}"),
        ("Color grading", config.test_color_grading, f"style: {config.color_style}"),
        ("Sharpening intelligent", config.test_sharpening, "unsharp mask adaptatif"),
        ("Vignette", config.test_vignette, "focus central"),
        ("Encodage HQ", config.test_high_quality, "CRF 18, preset slow, LANCZOS4"),
    ]
    
    for name, enabled, desc in features:
        status = "[green]ON[/green]" if enabled else "[dim]OFF[/dim]"
        console.print(f"  {status} {name}: {desc}")
    
    console.print("")
    
    # Générer le clip
    generator = ClipGenerator(clip_config)
    
    try:
        clips = generator.generate_clips(
            video_path=config.video_path,
            output_dir=str(output_dir),
            moments=[test_moment]
        )
        
        if clips:
            clip_path = clips[0]
            file_size = Path(clip_path).stat().st_size / (1024 * 1024)  # Mo
            
            console.print(Panel(
                f"[bold green]Clip généré avec succès![/bold green]\n\n"
                f"Fichier: {clip_path}\n"
                f"Taille: {file_size:.1f} Mo",
                title="Résultat"
            ))
            
            return clip_path
        else:
            console.print("[red]Aucun clip généré[/red]")
            return None
            
    except Exception as e:
        console.print(f"[red]Erreur lors de la génération: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test visuel ClipGenius")
    parser.add_argument("--video", type=str, help="Chemin vers la vidéo source")
    parser.add_argument("--start", type=float, default=30.0, help="Temps de début (secondes)")
    parser.add_argument("--duration", type=float, default=25.0, help="Durée du clip test")
    parser.add_argument("--output", type=str, default="output_test", help="Dossier de sortie")
    
    # Options pour désactiver des fonctionnalités
    parser.add_argument("--no-blur-fill", action="store_true", help="Desactiver le blur-fill")
    parser.add_argument("--no-zoom", action="store_true", help="Desactiver le zoom")
    parser.add_argument("--no-smart-crop", action="store_true", help="Desactiver le tracking facial")
    parser.add_argument("--no-color-grading", action="store_true", help="Desactiver le color grading")
    parser.add_argument("--no-sharpening", action="store_true", help="Desactiver le sharpening")
    parser.add_argument("--vignette", action="store_true", help="Activer la vignette")
    parser.add_argument("--color-style", type=str, default="warm", 
                       choices=["warm", "cool", "vibrant", "cinematic", "none"],
                       help="Style de color grading")
    parser.add_argument("--zoom-style", type=str, default="ease_out",
                       choices=["ease_out", "ease_in_out", "breathing", "pulse"],
                       help="Style de zoom")
    parser.add_argument("--ken-burns", action="store_true", help="Activer l'effet Ken Burns")
    parser.add_argument("--ken-burns-intensity", type=float, default=0.02,
                       help="Intensite de l'effet Ken Burns (0-0.1, defaut: 0.02)")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (qualite reduite)")
    
    args = parser.parse_args()
    
    # Trouver une vidéo de test
    if args.video:
        video_path = args.video
    else:
        # Chercher la vidéo de test par défaut
        default_video = Path("downloads/7 Days Stranded On An Island.mp4")
        if default_video.exists():
            video_path = str(default_video)
        else:
            # Chercher n'importe quelle vidéo dans downloads
            downloads = Path("downloads")
            if downloads.exists():
                videos = list(downloads.glob("*.mp4"))
                if videos:
                    video_path = str(videos[0])
                else:
                    console.print("[red]Aucune vidéo trouvée dans downloads/[/red]")
                    console.print("Utilisez: python test_visual.py --video <chemin>")
                    sys.exit(1)
            else:
                console.print("[red]Dossier downloads/ non trouvé[/red]")
                sys.exit(1)
    
    config = TestConfig(
        video_path=video_path,
        output_dir=args.output,
        start_time=args.start,
        duration=args.duration,
        test_blur_fill=not args.no_blur_fill,
        test_zoom=not args.no_zoom,
        test_smart_crop=not args.no_smart_crop,
        test_high_quality=not args.fast,
        test_color_grading=not args.no_color_grading,
        color_style=args.color_style,
        test_sharpening=not args.no_sharpening,
        test_vignette=args.vignette,
        zoom_style=args.zoom_style,
        test_ken_burns=args.ken_burns,
        ken_burns_intensity=args.ken_burns_intensity,
    )
    
    result = generate_test_clip(config)
    
    if result:
        console.print(f"\n[dim]Pour ouvrir le clip: start {result}[/dim]")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
