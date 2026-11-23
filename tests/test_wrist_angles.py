"""Tests du calcul de rotation / inclinaison du poignet."""

from __future__ import annotations

import numpy as np
import pytest

from src.wrist_angles import (
    DEFAULT_NEUTRAL_LONG,
    DEFAULT_NEUTRAL_NORMAL,
    HandednessFilter,
    axis_to_servo,
    hand_frame,
    wrist_angles,
)
from tests.synthetic_hand import _rotate, make_hand


def mp_hand(curl: float = 0.0, right: bool = True) -> np.ndarray:
    """Main fictive dans le repère MediaPipe : doigts vers le haut (y négatif),
    paume face à la caméra (z négatif), vue en miroir comme à l'écran."""
    lm = make_hand(curl)  # main « gauche » : pouce en +x, paume en +z
    if right:
        lm = lm * np.array([-1.0, 1.0, 1.0])  # symétrie → main droite
    return lm * np.array([1.0, -1.0, -1.0])   # rotation de 180° autour de x


def sign(right: bool) -> float:
    return 1.0 if right else -1.0


def rotated(lm: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    return np.array([_rotate(p, axis, deg) for p in lm])


def angles(lm: np.ndarray, right: bool) -> tuple[float, float]:
    """Angles par rapport à la pose neutre de la main fictive (comme après calibration)."""
    h = sign(right)
    neutral = hand_frame(mp_hand(right=right), h)
    return wrist_angles(hand_frame(lm, h), neutral.long, neutral.normal, h)


@pytest.mark.parametrize("right", [True, False])
def test_pose_neutre_par_defaut(right: bool) -> None:
    """Paume face caméra, doigts vers le haut ≈ pose neutre par défaut (à quelques degrés près :
    l'axe poignet → majeur de la main fictive n'est pas parfaitement vertical)."""
    h = sign(right)
    f = hand_frame(mp_hand(right=right), h)
    assert float(np.dot(f.normal, DEFAULT_NEUTRAL_NORMAL)) > 0.99
    rot, dev = wrist_angles(f, DEFAULT_NEUTRAL_LONG, DEFAULT_NEUTRAL_NORMAL, h)
    assert abs(rot) < 2.0
    assert abs(dev) < 6.0


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("deg", [-80.0, -30.0, 25.0, 70.0])
def test_rotation_seule(right: bool, deg: float) -> None:
    lm = mp_hand(right=right)
    f = hand_frame(lm, sign(right))
    # Axe qui fait tourner la normale vers le pouce = axe long de la main.
    rot, dev = angles(rotated(lm, np.cross(f.normal, f.thumbward), deg), right)
    assert rot == pytest.approx(deg, abs=1e-6)
    assert dev == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("deg", [-30.0, -10.0, 15.0, 30.0])
def test_inclinaison_seule(right: bool, deg: float) -> None:
    lm = mp_hand(right=right)
    f = hand_frame(lm, sign(right))
    # Axe qui fait pencher l'axe long vers le pouce = normale de la paume.
    rot, dev = angles(rotated(lm, np.cross(f.long, f.thumbward), deg), right)
    assert dev == pytest.approx(deg, abs=1e-6)
    assert rot == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("right", [True, False])
def test_rotation_et_inclinaison_combinees(right: bool) -> None:
    """Comme un vrai bras : inclinaison au poignet, puis rotation de tout l'ensemble
    autour de l'avant-bras (axe long neutre)."""
    lm = mp_hand(right=right)
    f0 = hand_frame(lm, sign(right))
    lm = rotated(lm, np.cross(f0.long, f0.thumbward), 20.0)   # inclinaison vers le pouce
    lm = rotated(lm, np.cross(f0.normal, f0.thumbward), 50.0)  # rotation autour de l'avant-bras
    rot, dev = angles(lm, right)
    assert dev == pytest.approx(20.0, abs=1e-6)
    assert rot == pytest.approx(50.0, abs=1e-6)


def test_independant_des_doigts_et_de_la_distance() -> None:
    lm = mp_hand(curl=0.0)
    f = hand_frame(lm, 1.0)
    ref = angles(rotated(lm, np.cross(f.normal, f.thumbward), 40.0), True)
    curled = mp_hand(curl=1.0)
    moved = rotated(curled, np.cross(f.normal, f.thumbward), 40.0) * 3.0 + 0.2
    np.testing.assert_allclose(angles(moved, True), ref, atol=1e-6)


def test_pose_neutre_calibree() -> None:
    """Si la pose neutre est la main tournée, cette pose donne 0."""
    lm = mp_hand()
    f = hand_frame(lm, 1.0)
    lm = rotated(lm, np.cross(f.normal, f.thumbward), 35.0)
    f = hand_frame(lm, 1.0)
    rot, dev = wrist_angles(f, f.long, f.normal, 1.0)
    assert rot == pytest.approx(0.0, abs=1e-6)
    assert dev == pytest.approx(0.0, abs=1e-6)


def test_axis_to_servo() -> None:
    assert axis_to_servo(0.0, (-90, 90), 0, 180, False) == pytest.approx(90.0)
    assert axis_to_servo(90.0, (-90, 90), 0, 180, False) == pytest.approx(180.0)
    assert axis_to_servo(90.0, (-90, 90), 0, 180, True) == pytest.approx(0.0)
    assert axis_to_servo(-200.0, (-30, 30), 45, 135, False) == pytest.approx(45.0)


def test_filtre_lateralite_ignore_un_basculement_isole() -> None:
    f = HandednessFilter()
    for _ in range(10):
        f.update("Right")
    assert f.update("Left") == 1.0
    for _ in range(30):
        last = f.update("Left")
    assert last == -1.0
