"""Lissage des valeurs par moyenne mobile exponentielle (EMA)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ExponentialSmoother:
    """Moyenne mobile exponentielle vectorielle.

    valeur = alpha × cible + (1 − alpha) × valeur_précédente

    alpha proche de 1 : très réactif mais tremblant ;
    alpha proche de 0 : très lisse mais avec du retard.
    """

    def __init__(self, alpha: float, size: int, initial: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha doit être dans ]0, 1], reçu {alpha}")
        self.alpha = alpha
        self.value: FloatArray = np.full(size, initial, dtype=np.float64)

    def update(self, target: FloatArray, alpha: float | None = None) -> FloatArray:
        """Rapproche la valeur lissée de `target` et la renvoie (copie).

        `alpha` permet d'utiliser ponctuellement un autre coefficient (par ex.
        un retour plus lent vers la main ouverte quand la main est perdue).
        """
        a = self.alpha if alpha is None else alpha
        self.value = a * np.asarray(target, dtype=np.float64) + (1.0 - a) * self.value
        return self.value.copy()

    def reset(self, value: FloatArray | float) -> None:
        """Force la valeur courante (sans lissage)."""
        self.value = np.broadcast_to(np.asarray(value, dtype=np.float64), self.value.shape).copy()
