"""
Module d'inférence LLM local avec Phi-4-mini et llama.cpp
100% offline, CPU-only, zéro dépendance API cloud

Phi-4-mini-instruct (3.8B params):
- Contexte: 128K tokens
- Vocabulaire: 200K tokens
- Format: <|system|>...<|end|><|user|>...<|end|><|assistant|>
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, cast
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from rich.console import Console

# Supprimer les warnings ggml_metal pour bf16 (non supporté sur certains GPU)
logging.getLogger("llama_cpp").setLevel(logging.ERROR)


@contextmanager
def suppress_stderr():
    """
    Supprime temporairement stderr pour éviter les warnings ggml_metal.
    Les warnings bf16 viennent du backend C++ Metal et ne peuvent pas
    être supprimés via Python logging.
    """
    # Sauvegarder le stderr original
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    
    try:
        # Rediriger stderr vers /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restaurer stderr
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)

console = Console()


@dataclass
class LLMResponse:
    """Réponse du LLM"""
    text: str
    tokens_used: int
    stop_reason: str


class LocalLLM:
    """
    Gestionnaire LLM local avec Phi-4-mini via llama.cpp.

    Charge le modèle une seule fois et le réutilise pour tous les appels.
    Supporte Phi-4-mini (recommandé) et Phi-3-mini (fallback).
    """

    # Singleton pattern: une seule instance du modèle
    _instance = None
    _llm = None

    def __new__(cls, model_path: Optional[str] = None, n_threads: Optional[int] = None):
        """Singleton: crée une seule instance du gestionnaire"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, n_threads: Optional[int] = None):
        """
        Initialise le LLM local.

        Args:
            model_path: Chemin vers le modèle GGUF (défaut: cherche automatiquement)
            n_threads: Nombre de threads CPU (défaut: auto-detect)
        """
        if self._initialized:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python n'est pas installé.\n"
                "Installe avec: pip install llama-cpp-python\n"
                "Voir: https://github.com/abetlen/llama-cpp-python"
            )

        # Déterminer le chemin du modèle
        if model_path is None:
            model_path = self._find_model()

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle Phi-4-mini GGUF introuvable à {model_path}\n"
                "Télécharge le modèle depuis:\n"
                "https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF\n"
                "Cherche: Phi-4-mini-instruct-Q4_K_M.gguf\n"
                "(Phi-3-mini est aussi supporté comme fallback)"
            )

        # Nombre de threads
        if n_threads is None:
            n_threads = min(8, (os.cpu_count() or 4) - 1)

        # Charger le modèle
        # Détecter le type de modèle pour afficher le bon message
        model_name = Path(model_path).name.lower()
        if 'phi-4' in model_name or 'phi4' in model_name:
            model_display = "Phi-4-mini"
            n_ctx = 4096  # Phi-4 supporte 128K mais on limite pour la RAM
        else:
            model_display = "Phi-3-mini"
            n_ctx = 2048  # Phi-3 a un contexte de 4K
        
        console.print(f"[cyan]Chargement de {model_display} local...[/cyan]")
        console.print(f"[dim]Modèle: {Path(model_path).name}[/dim]")
        console.print(f"[dim]Threads: {n_threads}, Contexte: {n_ctx}[/dim]")

        # Détecter si on est sur Mac (Apple Silicon) pour activer Metal
        import platform
        is_apple_silicon = (
            platform.system() == "Darwin" and 
            platform.machine() == "arm64"
        )
        
        # Configuration optimale selon la plateforme
        if is_apple_silicon:
            n_gpu_layers = -1  # -1 = charger TOUTES les couches sur Metal
            n_batch = 512      # Batch plus grand pour Metal (meilleure perf)
            use_mlock = True   # Verrouiller en RAM pour éviter le swap
            console.print(f"[green]Accélération Metal détectée (Apple Silicon)[/green]")
        else:
            n_gpu_layers = 0   # CPU uniquement sur les autres plateformes
            n_batch = 256
            use_mlock = False
        
        try:
            # Utiliser suppress_stderr pour masquer les warnings ggml_metal bf16
            # Ces warnings viennent du backend C++ et ne sont pas critiques
            with suppress_stderr():
                LocalLLM._llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,              # Contexte adapté au modèle
                    n_threads=n_threads,
                    n_batch=n_batch,          # Batch optimisé selon plateforme
                    verbose=False,
                    # GPU acceleration automatique sur Mac
                    n_gpu_layers=n_gpu_layers,
                    # Optimisations mémoire
                    use_mmap=True,            # Memory mapping pour chargement rapide
                    use_mlock=use_mlock,      # Verrouiller en RAM (Mac uniquement)
                )
            
            if is_apple_silicon:
                console.print(f"[green]✓ {model_display} chargé avec Metal (GPU)[/green]")
            else:
                console.print(f"[green]✓ {model_display} chargé avec succès[/green]")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")

        self._initialized = True

    @staticmethod
    def _find_model() -> Optional[str]:
        """Cherche le modèle GGUF dans les emplacements courants (Phi-4 prioritaire)"""
        # Noms possibles du modèle - Phi-4 en priorité, puis Phi-3 en fallback
        model_names = [
            # Phi-4-mini (recommandé)
            "Phi-4-mini-instruct-Q4_K_M.gguf",
            "Phi-4-mini-instruct-q4_k_m.gguf",
            "phi-4-mini-instruct-Q4_K_M.gguf",
            "phi-4-mini-instruct-q4_k_m.gguf",
            "Phi-4-mini-instruct.Q4_K_M.gguf",
            "Phi-4-mini-Q4_K_M.gguf",
            "phi4-mini-instruct-q4.gguf",
            # Phi-3-mini (fallback)
            "Phi-3-mini-4k-instruct-q4.gguf",
            "phi-3-mini-4k-instruct-q4.gguf",
            "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
            "phi-3-mini-4k-instruct-q4_k_m.gguf",
        ]
        
        # Dossiers de recherche
        search_dirs = [
            Path.cwd(),
            Path.cwd() / "models",
            Path.home() / ".cache" / "lm-studio" / "models",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "models",
            Path("/models"),
        ]
        
        for directory in search_dirs:
            for name in model_names:
                path = directory / name
                if path.exists():
                    console.print(f"[dim]Modèle trouvé: {path}[/dim]")
                    return str(path)

        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[list] = None,
    ) -> LLMResponse:
        """
        Génère une réponse à partir d'un prompt.

        Args:
            prompt: Texte du prompt
            max_tokens: Nombre max de tokens à générer
            temperature: Température pour la génération (0.0-1.0)
            top_p: Top-p sampling
            stop: Séquences d'arrêt

        Returns:
            LLMResponse avec le texte généré et les stats
        """
        if LocalLLM._llm is None:
            raise RuntimeError("LLM n'a pas été initialisé. Appelle LocalLLM() d'abord.")

        try:
            # Note: llama-cpp returns Union[CreateCompletionResponse, Iterator[...]]
            # We always get dict-like response since stream=False (default)
            raw_response = LocalLLM._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False,
            )
            response = cast(Dict[str, Any], raw_response)

            text: str = response["choices"][0]["text"].strip()
            tokens: int = response.get("usage", {}).get("completion_tokens", 0)
            stop_reason: str = str(response["choices"][0].get("finish_reason", "stop") or "stop")

            return LLMResponse(
                text=text,
                tokens_used=tokens,
                stop_reason=stop_reason
            )

        except Exception as e:
            console.print(f"[red]Erreur lors de la génération: {e}[/red]")
            raise

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.2,
        show_progress: bool = True
    ) -> list[LLMResponse]:
        """
        Génère des réponses pour plusieurs prompts.

        Args:
            prompts: Liste des prompts
            max_tokens: Nombre max de tokens par réponse
            temperature: Température pour la génération
            show_progress: Afficher la barre de progression

        Returns:
            Liste des réponses
        """
        responses = []

        if show_progress:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Inférence LLM...", total=len(prompts))
                for prompt in prompts:
                    response = self.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["\n\n", "---"]
                    )
                    responses.append(response)
                    progress.update(task, advance=1)
        else:
            for prompt in prompts:
                response = self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n\n", "---"]
                )
                responses.append(response)

        return responses

    @staticmethod
    def cleanup():
        """Libère les ressources du LLM"""
        if LocalLLM._llm is not None:
            try:
                del LocalLLM._llm
                LocalLLM._llm = None
                LocalLLM._instance = None
            except Exception:
                pass
