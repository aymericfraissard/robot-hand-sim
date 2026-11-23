"""Générateur de mains fictives (21 landmarks 3D) pour les tests.

Repère (proche des landmarks « monde » de MediaPipe, en mètres) :
  y = direction des doigts, x = côté du pouce, z = normale de la paume
  (les doigts se plient vers +z).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Base des doigts longs (MCP) et longueurs des 3 phalanges.
_MCP = {
    1: (0.030, 0.090, 0.0),
    2: (0.008, 0.095, 0.0),
    3: (-0.012, 0.088, 0.0),
    4: (-0.030, 0.078, 0.0),
}
_PHALANGES = {
    1: (0.040, 0.025, 0.022),
    2: (0.045, 0.030, 0.024),
    3: (0.042, 0.028, 0.022),
    4: (0.032, 0.020, 0.020),
}
# Pliage maximal (degrés) aux articulations MCP, PIP, DIP pour un poing.
_MAX_BEND = (85.0, 100.0, 70.0)


def _rotate(v: FloatArray, axis: FloatArray, angle_deg: float) -> FloatArray:
    """Rotation de Rodrigues du vecteur v autour de l'axe (unitaire)."""
    k = axis / np.linalg.norm(axis)
    t = np.radians(angle_deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1 - np.cos(t))


def _unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


def make_hand(curls: float | list[float]) -> FloatArray:
    """Construit une main dont chaque doigt est plié selon `curls` (0 → 1).

    `curls` peut être un seul nombre (tous les doigts) ou une liste de 5
    valeurs (pouce, index, majeur, annulaire, auriculaire).
    """
    c = [float(curls)] * 5 if np.isscalar(curls) else [float(x) for x in curls]  # type: ignore[arg-type]
    lm = np.zeros((21, 3), dtype=np.float64)
    palm_normal = np.array([0.0, 0.0, 1.0])

    # --- Doigts longs ---
    for finger, mcp in _MCP.items():
        mcp_v = np.array(mcp)
        direction = _unit(mcp_v)                       # prolongement du métacarpe
        axis = np.cross(direction, palm_normal)        # axe de flexion
        base = 4 * finger + 1                          # index du landmark MCP
        lm[base] = mcp_v
        point = mcp_v
        bend_total = 0.0
        for j, (length, max_bend) in enumerate(zip(_PHALANGES[finger], _MAX_BEND)):
            bend_total += c[finger] * max_bend
            point = point + length * _rotate(direction, axis, bend_total)
            lm[base + j + 1] = point

    # --- Pouce ---
    cmc = np.array([0.025, 0.025, 0.010])
    open_dir = _unit(np.array([0.75, 0.65, 0.0]))
    closed_dir = _unit(np.array([0.05, 0.60, 0.80]))
    meta_dir = _unit((1 - c[0]) * open_dir + c[0] * closed_dir)
    mcp = cmc + 0.040 * meta_dir
    toward_pinky = _unit(lm[17] - mcp)
    axis = np.cross(meta_dir, toward_pinky)
    ip = mcp + 0.032 * _rotate(meta_dir, axis, c[0] * 40.0)
    tip = ip + 0.028 * _rotate(meta_dir, axis, c[0] * 100.0)
    lm[1], lm[2], lm[3], lm[4] = cmc, mcp, ip, tip
    return lm
