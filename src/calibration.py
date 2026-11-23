"""Calibration « main ouverte » / « poing fermé ».

La calibration enregistre, pour chaque doigt, la flexion brute (degrés) quand
la main est grande ouverte et quand le poing est fermé. Ces valeurs servent de
bornes 0 % / 100 % (voir finger_angles.raw_to_percent).

Déroulement (touche `c`) :
    compte à rebours → capture « main ouverte » → compte à rebours
    → capture « poing fermé » → sauvegarde JSON.
Une nouvelle pression sur `c` pendant la procédure l'annule.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.finger_angles import DEFAULT_CLOSED_RAW, DEFAULT_OPEN_RAW, FINGER_NAMES, NUM_FINGERS
from src.wrist_angles import DEFAULT_NEUTRAL_LONG, DEFAULT_NEUTRAL_NORMAL

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

# Écart minimal (degrés) exigé entre ouvert et fermé pour accepter la calibration.
MIN_RANGE_DEG = 20.0


@dataclass
class Calibration:
    """Bornes brutes (degrés) par doigt (ordre pouce → auriculaire) et pose
    neutre du poignet (axe long et normale de la paume, vecteurs unitaires)."""

    open_raw: FloatArray = field(default_factory=lambda: DEFAULT_OPEN_RAW.copy())
    closed_raw: FloatArray = field(default_factory=lambda: DEFAULT_CLOSED_RAW.copy())
    is_default: bool = True
    wrist_long: FloatArray = field(default_factory=lambda: DEFAULT_NEUTRAL_LONG.copy())
    wrist_normal: FloatArray = field(default_factory=lambda: DEFAULT_NEUTRAL_NORMAL.copy())

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        """Charge le fichier JSON ; renvoie la calibration par défaut s'il est absent/invalide."""
        if not path.exists():
            log.info("Pas de calibration (%s) : valeurs par défaut utilisées.", path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            open_raw = np.array(data["open_raw"], dtype=np.float64)
            closed_raw = np.array(data["closed_raw"], dtype=np.float64)
            if open_raw.shape != (NUM_FINGERS,) or closed_raw.shape != (NUM_FINGERS,):
                raise ValueError("5 valeurs attendues par liste")
            # Pose neutre du poignet : optionnelle (anciens fichiers sans poignet).
            wrist = data.get("wrist_neutral")
            if wrist is not None:
                wrist_long, wrist_normal = _orthonormalize(
                    np.array(wrist["long"], dtype=np.float64), np.array(wrist["normal"], dtype=np.float64)
                )
            else:
                wrist_long, wrist_normal = DEFAULT_NEUTRAL_LONG.copy(), DEFAULT_NEUTRAL_NORMAL.copy()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Calibration illisible (%s) : %s. Valeurs par défaut utilisées.", path, exc)
            return cls()
        log.info("Calibration chargée depuis %s", path)
        return cls(open_raw, closed_raw, False, wrist_long, wrist_normal)

    def save(self, path: Path) -> None:
        """Sauvegarde la calibration en JSON lisible."""
        data = {
            "fingers": list(FINGER_NAMES),
            "open_raw": [round(float(v), 2) for v in self.open_raw],
            "closed_raw": [round(float(v), 2) for v in self.closed_raw],
            "wrist_neutral": {
                "long": [round(float(v), 4) for v in self.wrist_long],
                "normal": [round(float(v), 4) for v in self.wrist_normal],
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Calibration sauvegardée dans %s", path)


def _orthonormalize(long: FloatArray, normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Rend les deux vecteurs unitaires et perpendiculaires (normale corrigée)."""
    nl = float(np.linalg.norm(long))
    if nl < 1e-9:
        raise ValueError("axe long nul")
    long = long / nl
    normal = normal - float(np.dot(normal, long)) * long
    nn = float(np.linalg.norm(normal))
    if nn < 1e-9:
        raise ValueError("normale nulle")
    return long, normal / nn


class _Step(Enum):
    IDLE = auto()
    COUNTDOWN_OPEN = auto()
    CAPTURE_OPEN = auto()
    COUNTDOWN_CLOSED = auto()
    CAPTURE_CLOSED = auto()


class CalibrationProcedure:
    """Machine à états de la calibration, pilotée par `update()` à chaque image.

    Le temps (`now`, en secondes) est passé en paramètre pour rester testable.
    """

    def __init__(self, countdown_s: float = 3.0, capture_s: float = 1.0) -> None:
        self.countdown_s = countdown_s
        self.capture_s = capture_s
        self._step = _Step.IDLE
        self._step_start = 0.0
        self._samples: list[FloatArray] = []
        self._wrist_samples: list[FloatArray] = []
        self._open_raw: FloatArray | None = None
        self._wrist_neutral: tuple[FloatArray, FloatArray] | None = None
        self.message = ""  # Texte à afficher à l'écran

    @property
    def active(self) -> bool:
        return self._step is not _Step.IDLE

    def toggle(self, now: float) -> None:
        """Démarre la calibration, ou l'annule si elle est déjà en cours."""
        if self.active:
            self._step = _Step.IDLE
            self.message = "Calibration annulee"
        else:
            self._goto(_Step.COUNTDOWN_OPEN, now)

    def _goto(self, step: _Step, now: float) -> None:
        self._step = step
        self._step_start = now
        self._samples = []
        self._wrist_samples = []

    def update(
        self,
        now: float,
        raw: FloatArray | None,
        wrist: tuple[FloatArray, FloatArray] | None = None,
    ) -> Calibration | None:
        """Fait avancer la procédure. Renvoie la calibration quand elle est terminée.

        `raw` : flexions brutes des doigts ; `wrist` : (axe long, normale) de la
        main, enregistrés comme pose neutre du poignet pendant « main ouverte ».
        """
        elapsed = now - self._step_start
        step = self._step

        if step is _Step.COUNTDOWN_OPEN:
            self.message = f"OUVREZ la main, doigts en haut ({self.countdown_s - elapsed:.0f}s)"
            if elapsed >= self.countdown_s:
                self._goto(_Step.CAPTURE_OPEN, now)
        elif step is _Step.COUNTDOWN_CLOSED:
            self.message = f"Calibration : FERMEZ le poing ({self.countdown_s - elapsed:.0f}s)"
            if elapsed >= self.countdown_s:
                self._goto(_Step.CAPTURE_CLOSED, now)
        elif step in (_Step.CAPTURE_OPEN, _Step.CAPTURE_CLOSED):
            label = "main ouverte" if step is _Step.CAPTURE_OPEN else "poing ferme"
            self.message = f"Capture {label}... ne bougez plus"
            if raw is not None:
                self._samples.append(np.asarray(raw, dtype=np.float64))
            if wrist is not None and step is _Step.CAPTURE_OPEN:
                self._wrist_samples.append(np.concatenate(wrist).astype(np.float64))
            if elapsed >= self.capture_s:
                return self._finish_capture(now, step)
        return None

    def _finish_capture(self, now: float, step: _Step) -> Calibration | None:
        if len(self._samples) < 3:
            # Main non détectée pendant la capture : on recommence cette étape.
            self.message = "Main non detectee, on recommence..."
            self._goto(_Step.COUNTDOWN_OPEN if step is _Step.CAPTURE_OPEN else _Step.COUNTDOWN_CLOSED, now)
            return None
        # La médiane est robuste aux quelques images mal détectées.
        median = np.median(np.stack(self._samples), axis=0)
        if step is _Step.CAPTURE_OPEN:
            self._open_raw = median
            self._wrist_neutral = None
            if self._wrist_samples:
                w = np.median(np.stack(self._wrist_samples), axis=0)
                try:
                    self._wrist_neutral = _orthonormalize(w[:3], w[3:])
                except ValueError:
                    pass
            self._goto(_Step.COUNTDOWN_CLOSED, now)
            return None

        assert self._open_raw is not None
        self._step = _Step.IDLE
        span = median - self._open_raw
        bad = [FINGER_NAMES[i] for i in range(NUM_FINGERS) if span[i] < MIN_RANGE_DEG]
        if bad:
            self.message = "Calibration refusee (amplitude trop faible : " + ", ".join(bad) + ")"
            log.warning(self.message)
            return None
        self.message = "Calibration enregistree !"
        calib = Calibration(self._open_raw, median, is_default=False)
        if self._wrist_neutral is not None:
            calib.wrist_long, calib.wrist_normal = self._wrist_neutral
        return calib
