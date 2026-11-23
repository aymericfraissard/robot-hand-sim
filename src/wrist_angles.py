"""Calcul de l'orientation du poignet : rotation et inclinaison latérale.

MediaPipe ne voit que la main, pas l'avant-bras : on mesure donc l'orientation
de la main par rapport à une POSE NEUTRE (enregistrée pendant la calibration),
en supposant que l'avant-bras reste à peu près immobile.

Repère de la main (construit avec les landmarks « monde ») :
  - axe long    : poignet (0) → base du majeur (9) ;
  - axe pouce   : base de l'auriculaire (17) → base de l'index (5), rendu
                  perpendiculaire à l'axe long : il pointe vers le pouce ;
  - normale     : sort de la paume, côté paume (dépend de la main gauche/droite).

Angles calculés (en degrés, par rapport à la pose neutre) :
  - rotation    (pronation / supination) : rotation de la paume autour de
                  l'axe long ; positive quand la paume tourne vers le pouce ;
  - inclinaison (déviation radiale / cubitale) : inclinaison de l'axe long dans
                  le plan de la paume ; positive vers le pouce.

La flexion du poignet (main vers le haut / le bas) n'est pas calculée : elle
dépend de la profondeur, que la webcam estime mal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17

# Pose neutre par défaut, dans le repère de MediaPipe (x vers la droite de
# l'image, y vers le bas, z qui s'éloigne de la caméra) : doigts vers le haut,
# paume face à la caméra.
DEFAULT_NEUTRAL_LONG: FloatArray = np.array([0.0, -1.0, 0.0])
DEFAULT_NEUTRAL_NORMAL: FloatArray = np.array([0.0, 0.0, -1.0])


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
    normal: FloatArray     # sort de la paume, côté paume


def hand_frame(landmarks: FloatArray, handedness_sign: float) -> HandFrame:
    """Construit le repère de la main.

    `handedness_sign` vaut +1 pour une main droite, −1 pour une main gauche
    (selon l'étiquette MediaPipe). Il fixe le sens de la normale : les mains
    gauche et droite sont symétriques l'une de l'autre, comme dans un miroir.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    long = _unit(lm[MIDDLE_MCP] - lm[WRIST])
    thumbward = _project_unit(lm[INDEX_MCP] - lm[PINKY_MCP], long)
    normal = handedness_sign * np.cross(long, thumbward)
    return HandFrame(long, thumbward, normal)


def wrist_angles(
    frame: HandFrame,
    neutral_long: FloatArray,
    neutral_normal: FloatArray,
    handedness_sign: float,
) -> tuple[float, float]:
    """Renvoie (rotation, inclinaison) en degrés par rapport à la pose neutre."""
    # Inclinaison : où se trouve l'axe long neutre dans le plan de la paume actuelle ?
    # Si la main s'incline de +θ vers le pouce, l'axe neutre apparaît à −θ.
    y0 = _project_unit(np.asarray(neutral_long), frame.normal)
    deviation = math.degrees(math.atan2(-float(np.dot(y0, frame.thumbward)), float(np.dot(y0, frame.long))))

    # Rotation : comme sur un vrai bras, elle se fait autour de l'AVANT-BRAS,
    # c'est-à-dire de l'axe long neutre (et non de l'axe actuel de la main, qui
    # est penché par l'inclinaison). On projette la normale actuelle sur le plan
    # perpendiculaire à l'avant-bras et on mesure son angle depuis la normale
    # neutre, positif vers le pouce.
    y_arm = _unit(np.asarray(neutral_long, dtype=np.float64))
    z0 = _project_unit(np.asarray(neutral_normal, dtype=np.float64), y_arm)
    t0 = handedness_sign * np.cross(z0, y_arm)  # direction « pouce » de la pose neutre
    zp = _project_unit(frame.normal, y_arm)
    rotation = math.degrees(math.atan2(float(np.dot(zp, t0)), float(np.dot(zp, z0))))
    return rotation, deviation


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
    """Réglages d'un axe du poignet (section `wrist.rotation` / `wrist.deviation`)."""

    enabled: bool
    range_deg: tuple[float, float]
    min_angle: float
    max_angle: float
    invert: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WristAxisConfig":
        lo, hi = (float(v) for v in d["range_deg"])
        if hi <= lo:
            raise ValueError(f"wrist.*.range_deg : [{lo}, {hi}] doit être croissant")
        return cls(bool(d["enabled"]), (lo, hi), float(d["min_angle"]), float(d["max_angle"]), bool(d["invert"]))

    def clamp(self, angle: float) -> float:
        """Angle borné à la plage ; 0 (neutre) si l'axe est désactivé."""
        return min(max(angle, self.range_deg[0]), self.range_deg[1]) if self.enabled else 0.0

    def to_servo(self, angle: float) -> int:
        return int(round(axis_to_servo(self.clamp(angle), self.range_deg, self.min_angle, self.max_angle, self.invert)))


class HandednessFilter:
    """Stabilise l'étiquette main gauche/droite de MediaPipe.

    L'étiquette peut basculer sur une image isolée, ce qui retournerait
    brutalement la normale de la paume. On la filtre avec une moyenne mobile.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self.value = 0.0

    def update(self, label: str) -> float:
        """Renvoie +1 (main droite) ou −1 (main gauche)."""
        s = 1.0 if label == "Right" else -1.0 if label == "Left" else 0.0
        self.value = s if self.value == 0.0 else (1 - self.alpha) * self.value + self.alpha * s
        return 1.0 if self.value >= 0.0 else -1.0
