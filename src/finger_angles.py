"""Calcul de la flexion des 5 doigts à partir des 21 landmarks MediaPipe.

Chaîne de calcul :
    landmarks (21 × 3) → flexion « brute » en degrés par doigt
                        → pourcentage 0–100 % (grâce à la calibration)
                        → angle de servo 0–180° (min/max + inversion par doigt)

On n'utilise QUE des angles entre vecteurs : le résultat ne dépend ni de la
taille de la main, ni de sa distance à la caméra, ni de son orientation.

Ordre des doigts partout dans le projet : pouce, index, majeur, annulaire, auriculaire.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

FINGER_NAMES: tuple[str, ...] = ("pouce", "index", "majeur", "annulaire", "auriculaire")
NUM_FINGERS = len(FINGER_NAMES)

# Indices des landmarks MediaPipe (voir la doc « hand landmarks »).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
PINKY_MCP = 17

# Pour les 4 doigts longs : poignet → MCP → PIP → DIP → bout.
# Le premier segment (poignet → MCP) sert de référence « métacarpe ».
FINGER_CHAINS: dict[int, tuple[int, int, int, int, int]] = {
    1: (WRIST, 5, 6, 7, 8),     # index
    2: (WRIST, 9, 10, 11, 12),  # majeur
    3: (WRIST, 13, 14, 15, 16), # annulaire
    4: (WRIST, 17, 18, 19, 20), # auriculaire
}

# Valeurs brutes par défaut (en degrés) utilisées tant qu'aucune calibration
# n'a été faite. Ordre : pouce, index, majeur, annulaire, auriculaire.
DEFAULT_OPEN_RAW: FloatArray = np.array([55.0, 15.0, 15.0, 15.0, 15.0])
DEFAULT_CLOSED_RAW: FloatArray = np.array([200.0, 230.0, 230.0, 230.0, 230.0])


def _angle_between(u: FloatArray, v: FloatArray) -> float:
    """Angle (en degrés) entre deux vecteurs ; 0° s'ils sont alignés."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos = float(np.dot(u, v)) / (nu * nv)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def bend_angle(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    """Angle de pliage à l'articulation b, entre les segments a→b et b→c.

    0° = segments alignés (doigt tendu), 90° = articulation pliée à angle droit.
    """
    return _angle_between(b - a, c - b)


def finger_flexion_raw(landmarks: FloatArray, finger: int) -> float:
    """Flexion brute (somme des pliages des 3 articulations) d'un doigt long."""
    if finger not in FINGER_CHAINS:
        raise ValueError(f"Doigt {finger} invalide : utiliser thumb_flexion_raw pour le pouce")
    p = [landmarks[i] for i in FINGER_CHAINS[finger]]
    # Pliage aux articulations MCP, PIP et DIP.
    return bend_angle(p[0], p[1], p[2]) + bend_angle(p[1], p[2], p[3]) + bend_angle(p[2], p[3], p[4])


def thumb_flexion_raw(landmarks: FloatArray) -> float:
    """Flexion brute du pouce, qui a une cinématique particulière.

    Le pouce plie peu ses phalanges : quand on ferme le poing, il pivote surtout
    vers la paume (opposition). On additionne donc :
      - le pliage à l'articulation MCP (CMC→MCP→IP) ;
      - le pliage à l'articulation IP (MCP→IP→bout) ;
      - un terme d'opposition : 180° moins l'angle entre la direction du pouce
        (MCP→bout) et la direction MCP du pouce → base de l'auriculaire.
        Pouce écarté : grand angle → terme faible.
        Pouce replié vers l'auriculaire : petit angle → terme fort.
    """
    cmc, mcp, ip, tip = (landmarks[i] for i in (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP))
    bends = bend_angle(cmc, mcp, ip) + bend_angle(mcp, ip, tip)
    opposition = 180.0 - _angle_between(tip - mcp, landmarks[PINKY_MCP] - mcp)
    return bends + opposition


def raw_flexions(landmarks: FloatArray | Sequence[Sequence[float]]) -> FloatArray:
    """Flexions brutes des 5 doigts (degrés), dans l'ordre pouce → auriculaire."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (21, 3):
        raise ValueError(f"21 landmarks 3D attendus, reçu la forme {lm.shape}")
    out = np.empty(NUM_FINGERS, dtype=np.float64)
    out[0] = thumb_flexion_raw(lm)
    for finger in FINGER_CHAINS:
        out[finger] = finger_flexion_raw(lm, finger)
    return out


def raw_to_percent(
    raw: FloatArray,
    open_raw: FloatArray = DEFAULT_OPEN_RAW,
    closed_raw: FloatArray = DEFAULT_CLOSED_RAW,
) -> FloatArray:
    """Convertit les flexions brutes en pourcentages 0 % (tendu) – 100 % (plié).

    Interpolation linéaire entre les valeurs « main ouverte » et « poing fermé »
    de la calibration, puis écrêtage dans [0, 100].
    """
    raw = np.asarray(raw, dtype=np.float64)
    span = np.asarray(closed_raw, dtype=np.float64) - np.asarray(open_raw, dtype=np.float64)
    # Évite la division par zéro si une calibration est dégénérée.
    span = np.where(np.abs(span) < 1e-6, 1e-6, span)
    pct = (raw - open_raw) / span * 100.0
    return np.clip(pct, 0.0, 100.0)


def percent_to_servo(
    percent: FloatArray,
    min_angles: FloatArray | Sequence[float],
    max_angles: FloatArray | Sequence[float],
    invert: Sequence[bool],
) -> FloatArray:
    """Convertit les pourcentages en angles de servo (degrés, 0–180).

    0 % → angle min, 100 % → angle max. Si `invert[i]` est vrai (servo monté à
    l'envers), le sens est inversé : 0 % → max, 100 % → min.
    """
    pct = np.clip(np.asarray(percent, dtype=np.float64), 0.0, 100.0) / 100.0
    lo = np.asarray(min_angles, dtype=np.float64)
    hi = np.asarray(max_angles, dtype=np.float64)
    inv = np.asarray(invert, dtype=bool)
    pct = np.where(inv, 1.0 - pct, pct)
    return np.clip(lo + pct * (hi - lo), 0.0, 180.0)
