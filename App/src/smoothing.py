"""Fonctions de lissage et interpolation pour le tracking."""

import math
import numpy as np
from typing import Tuple


def ease_in_out_cubic(t: float) -> float:
    """
    Fonction d'interpolation ease-in-out cubique.
    Produit un mouvement plus naturel avec accélération/décélération.

    Args:
        t: Valeur entre 0 et 1

    Returns:
        Valeur transformée entre 0 et 1
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_quad(t: float) -> float:
    """
    Fonction d'interpolation ease-out quadratique.
    Mouvement rapide au début, ralentit à la fin.
    """
    return 1 - (1 - t) * (1 - t)


def ease_out_expo(t: float) -> float:
    """
    Fonction d'interpolation ease-out exponentielle.
    Décélération très prononcée, idéale pour les mouvements de caméra.
    """
    return 1 if t == 1 else 1 - pow(2, -10 * t)


def ease_in_out_sine(t: float) -> float:
    """
    Fonction d'interpolation ease-in-out sinusoïdale.
    Mouvement très doux et naturel.
    """
    return -(math.cos(math.pi * t) - 1) / 2


def smooth_value(current: float, target: float, smoothing: float = 0.15) -> float:
    """
    Lissage exponentiel pour des mouvements fluides.

    Args:
        current: Valeur actuelle
        target: Valeur cible
        smoothing: Facteur de lissage (0.1 = très lisse, 0.5 = réactif)

    Returns:
        Nouvelle valeur lissée
    """
    return current + (target - current) * smoothing


def critically_damped_spring(
    current: float,
    target: float,
    velocity: float,
    smoothness: float = 0.25,
    delta_time: float = 1/30
) -> Tuple[float, float]:
    """
    Simulation d'un ressort critiquement amorti pour des mouvements ultra-fluides.
    Utilisé par les systèmes de caméra professionnels.

    Args:
        current: Position actuelle
        target: Position cible
        velocity: Vélocité actuelle
        smoothness: Temps de lissage en secondes (plus grand = plus lent)
        delta_time: Intervalle de temps

    Returns:
        Tuple (nouvelle_position, nouvelle_vélocité)
    """
    omega = 2.0 / smoothness  # Fréquence naturelle

    x = current - target
    exp_term = np.exp(-omega * delta_time)

    new_pos = target + (x + (velocity + omega * x) * delta_time) * exp_term
    new_vel = (velocity - omega * (velocity + omega * x) * delta_time) * exp_term

    return new_pos, new_vel


class KalmanFilter1D:
    """
    Filtre de Kalman 1D pour un lissage prédictif optimal.
    Utilisé pour stabiliser les mouvements de caméra en réduisant le bruit
    tout en conservant la réactivité aux vrais mouvements.
    """

    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """
        Args:
            process_variance: Variance du processus (bruit du système)
            measurement_variance: Variance de mesure (bruit de détection)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0
        self.velocity = 0.0
        self.initialized = False

    def update(self, measurement: float, dt: float = 1/30) -> float:
        """
        Met à jour le filtre avec une nouvelle mesure.

        Args:
            measurement: Nouvelle mesure brute
            dt: Intervalle de temps depuis la dernière mesure

        Returns:
            Estimation filtrée
        """
        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
            return measurement

        # Prédiction
        predicted_estimate = self.estimate + self.velocity * dt
        predicted_error = self.estimate_error + self.process_variance

        # Mise à jour
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)

        # Nouvelle estimation
        new_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)

        # Estimer la vélocité
        self.velocity = (new_estimate - self.estimate) / dt if dt > 0 else 0

        self.estimate = new_estimate
        self.estimate_error = (1 - kalman_gain) * predicted_error

        return self.estimate

    def reset(self):
        """Réinitialise le filtre"""
        self.estimate = 0.0
        self.estimate_error = 1.0
        self.velocity = 0.0
        self.initialized = False
