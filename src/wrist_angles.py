"""Calcul de l'inclinaison latérale du poignet.

MediaPipe ne voit que la main, pas l'avant-bras : on mesure donc l'orientation
de la main par rapport à une POSE NEUTRE (enregistrée pendant la calibration),
en supposant que l'avant-bras reste à peu près immobile.

Repère de la main (construit avec les landmarks « monde ») :
  - axe long    : poignet (0) → base du majeur (9) ;
  - axe pouce   : base de l'auriculaire (17) → base de l'index (5), rendu
                  perpendiculaire à l'axe long : il pointe vers le pouce ;
  - normale     : perpendiculaire au plan de la paume.

Angle calculé (en degrés, par rapport à la pose neutre) :
  - inclinaison (déviation radiale / cubitale) : inclinaison de l'axe long dans
                  le plan de la paume ; positive vers le pouce.

La rotation de la main sur elle-même et la flexion du poignet (main vers le
haut / le bas) ne sont pas calculées : trop instables avec une seule webcam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17

# Axe long neutre par défaut, dans le repère de MediaPipe (x vers la droite de
# l'image, y vers le bas) : doigts vers le haut.
DEFAULT_NEUTRAL_LONG: FloatArray = np.array([0.0, -1.0, 0.0])


def _unit(v: FloatArray) -> FloatArray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def _project_unit(v: FloatArray, axis: FloatArray) -> FloatArray:
    """Composante de v perpendiculaire à `axis` (unitaire), normalisée."""
    return _unit(v - float(np.dot(v, axis)) * axis)


@dataclass(frozen=True)
class HandFrame:
    """Repère orthonormé de la main."""

    long: FloatArray       # poignet → base du majeur
    thumbward: FloatArray  # dans le plan de la paume, vers le pouce
    normal: FloatArray     # perpendiculaire à la paume


def hand_frame(landmarks: FloatArray) -> HandFrame:
    """Construit le repère de la main à partir des 21 landmarks."""
    lm = np.asarray(landmarks, dtype=np.float64)
    long = _unit(lm[MIDDLE_MCP] - lm[WRIST])
    thumbward = _project_unit(lm[INDEX_MCP] - lm[PINKY_MCP], long)
    normal = np.cross(long, thumbward)
    return HandFrame(long, thumbward, normal)


def wrist_deviation(frame: HandFrame, neutral_long: FloatArray) -> float:
    """Inclinaison (degrés) de la main par rapport à la pose neutre, positive vers le pouce.

    On cherche où se trouve l'axe long neutre dans le plan de la paume actuelle :
    si la main s'incline de +θ vers le pouce, l'axe neutre apparaît à −θ.
    """
    y0 = _project_unit(np.asarray(neutral_long, dtype=np.float64), frame.normal)
    return math.degrees(math.atan2(-float(np.dot(y0, frame.thumbward)), float(np.dot(y0, frame.long))))


def axis_to_servo(
    angle: float,
    range_deg: tuple[float, float],
    min_angle: float,
    max_angle: float,
    invert: bool,
) -> float:
    """Convertit un angle de poignet (degrés) en angle de servo.

    range_deg[0] → min_angle, range_deg[1] → max_angle (ou l'inverse si `invert`).
    """
    lo, hi = range_deg
    t = (angle - lo) / (hi - lo) if hi != lo else 0.5
    t = min(max(t, 0.0), 1.0)
    if invert:
        t = 1.0 - t
    return float(np.clip(min_angle + t * (max_angle - min_angle), 0.0, 180.0))


@dataclass(frozen=True)
class WristAxisConfig:
    """Réglages de l'inclinaison du poignet (section `wrist.deviation`)."""

    range_deg: tuple[float, float]
    min_angle: float
    max_angle: float
    invert: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WristAxisConfig":
        lo, hi = (float(v) for v in d["range_deg"])
        if hi <= lo:
            raise ValueError(f"wrist.deviation.range_deg : [{lo}, {hi}] doit être croissant")
        return cls((lo, hi), float(d["min_angle"]), float(d["max_angle"]), bool(d["invert"]))

    def clamp(self, angle: float) -> float:
        """Angle borné à la plage."""
        return min(max(angle, self.range_deg[0]), self.range_deg[1])

    def to_servo(self, angle: float) -> int:
        return int(round(axis_to_servo(self.clamp(angle), self.range_deg, self.min_angle, self.max_angle, self.invert)))
