"""Tests du calcul de l'inclinaison du poignet."""

from __future__ import annotations

import numpy as np
import pytest

from src.wrist_angles import DEFAULT_NEUTRAL_LONG, WristAxisConfig, axis_to_servo, hand_frame, wrist_deviation
from tests.synthetic_hand import _rotate, make_hand


def mp_hand(curl: float = 0.0, right: bool = True) -> np.ndarray:
    """Main fictive dans le repère MediaPipe : doigts vers le haut (y négatif),
    paume face à la caméra (z négatif), vue en miroir comme à l'écran."""
    lm = make_hand(curl)  # main « gauche » : pouce en +x, paume en +z
    if right:
        lm = lm * np.array([-1.0, 1.0, 1.0])  # symétrie → main droite
    return lm * np.array([1.0, -1.0, -1.0])   # rotation de 180° autour de x


def rotated(lm: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    return np.array([_rotate(p, axis, deg) for p in lm])


def deviation(lm: np.ndarray, right: bool) -> float:
    """Inclinaison par rapport à la pose neutre de la main fictive (comme après calibration)."""
    neutral = hand_frame(mp_hand(right=right))
    return wrist_deviation(hand_frame(lm), neutral.long)


@pytest.mark.parametrize("right", [True, False])
def test_pose_neutre_par_defaut(right: bool) -> None:
    """Paume face caméra, doigts vers le haut ≈ pose neutre par défaut (à quelques degrés près :
    l'axe poignet → majeur de la main fictive n'est pas parfaitement vertical)."""
    assert abs(wrist_deviation(hand_frame(mp_hand(right=right)), DEFAULT_NEUTRAL_LONG)) < 6.0


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("deg", [-30.0, -10.0, 15.0, 30.0])
def test_inclinaison(right: bool, deg: float) -> None:
    lm = mp_hand(right=right)
    f = hand_frame(lm)
    # Axe qui fait pencher l'axe long vers le pouce.
    assert deviation(rotated(lm, np.cross(f.long, f.thumbward), deg), right) == pytest.approx(deg, abs=1e-6)


@pytest.mark.parametrize("right", [True, False])
def test_rotation_sur_elle_meme_ignoree(right: bool) -> None:
    """Tourner la main autour de l'avant-bras ne doit pas changer l'inclinaison."""
    lm = mp_hand(right=right)
    f0 = hand_frame(lm)
    lm = rotated(lm, np.cross(f0.long, f0.thumbward), 20.0)   # inclinaison vers le pouce
    lm = rotated(lm, np.cross(f0.normal, f0.thumbward), 50.0)  # rotation autour de l'avant-bras
    assert deviation(lm, right) == pytest.approx(20.0, abs=1e-6)


def test_independant_des_doigts_et_de_la_distance() -> None:
    lm = mp_hand(curl=0.0)
    f = hand_frame(lm)
    axis = np.cross(f.long, f.thumbward)
    ref = deviation(rotated(lm, axis, 15.0), True)
    moved = rotated(mp_hand(curl=1.0), axis, 15.0) * 3.0 + 0.2
    assert deviation(moved, True) == pytest.approx(ref, abs=1e-6)


def test_pose_neutre_calibree() -> None:
    """Si la pose neutre est la main inclinée, cette pose donne 0."""
    lm = mp_hand()
    f = hand_frame(lm)
    f = hand_frame(rotated(lm, np.cross(f.long, f.thumbward), 25.0))
    assert wrist_deviation(f, f.long) == pytest.approx(0.0, abs=1e-6)


def test_axis_to_servo() -> None:
    assert axis_to_servo(0.0, (-30, 30), 45, 135, False) == pytest.approx(90.0)
    assert axis_to_servo(30.0, (-30, 30), 45, 135, False) == pytest.approx(135.0)
    assert axis_to_servo(30.0, (-30, 30), 45, 135, True) == pytest.approx(45.0)
    assert axis_to_servo(-200.0, (-30, 30), 45, 135, False) == pytest.approx(45.0)


def test_config_axe() -> None:
    axis = WristAxisConfig.from_dict({"range_deg": [-30, 30], "min_angle": 45, "max_angle": 135, "invert": False})
    assert axis.to_servo(0.0) == 90
    assert axis.to_servo(100.0) == 135
    with pytest.raises(ValueError):
        WristAxisConfig.from_dict({"range_deg": [30, -30], "min_angle": 0, "max_angle": 180, "invert": False})
