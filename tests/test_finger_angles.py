"""Tests unitaires du calcul de flexion des doigts (landmarks fictifs)."""

from __future__ import annotations

import numpy as np
import pytest

from src.finger_angles import (
    bend_angle,
    percent_to_servo,
    raw_flexions,
    raw_to_percent,
)
from tests.synthetic_hand import make_hand


def percents(curls: float | list[float]) -> np.ndarray:
    return raw_to_percent(raw_flexions(make_hand(curls)))


def test_bend_angle_tendu_et_angle_droit() -> None:
    a, b = np.zeros(3), np.array([0.0, 1.0, 0.0])
    assert bend_angle(a, b, np.array([0.0, 2.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert bend_angle(a, b, np.array([1.0, 1.0, 0.0])) == pytest.approx(90.0)


def test_main_ouverte_proche_de_zero() -> None:
    assert np.all(percents(0.0) < 5.0)


def test_poing_ferme_proche_de_cent() -> None:
    assert np.all(percents(1.0) > 95.0)


def test_positions_intermediaires_monotones() -> None:
    """Le pourcentage doit croître continûment avec la flexion (pas de tout-ou-rien)."""
    values = np.array([percents(c) for c in np.linspace(0.0, 1.0, 11)])
    assert np.all(np.diff(values, axis=0) >= -1e-6)
    mid = percents(0.5)
    assert np.all((mid > 25.0) & (mid < 75.0))


def test_doigts_independants() -> None:
    """Plier seulement l'index ne doit pas faire bouger les autres doigts."""
    p = percents([0.0, 1.0, 0.0, 0.0, 0.0])
    assert p[1] > 95.0
    assert np.all(np.delete(p, 1) < 5.0)


def test_pouce_seul() -> None:
    p = percents([1.0, 0.0, 0.0, 0.0, 0.0])
    assert p[0] > 90.0
    assert np.all(p[1:] < 5.0)


@pytest.mark.parametrize("scale", [0.3, 1.0, 4.0])
def test_independant_de_la_distance(scale: float) -> None:
    """Une main plus petite/plus grande (plus loin/plus près) donne les mêmes valeurs."""
    hand = make_hand(0.4)
    ref = raw_flexions(hand)
    moved = raw_flexions(hand * scale + np.array([0.1, -0.2, 0.5]))
    np.testing.assert_allclose(moved, ref, atol=1e-6)


def test_independant_de_l_orientation() -> None:
    hand = make_hand(0.6)
    t = np.radians(35.0)
    rot = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(raw_flexions(hand @ rot.T), raw_flexions(hand), atol=1e-6)


def test_forme_invalide() -> None:
    with pytest.raises(ValueError):
        raw_flexions(np.zeros((20, 3)))


def test_calibration_personnalisee() -> None:
    raw = np.array([100.0] * 5)
    pct = raw_to_percent(raw, open_raw=np.full(5, 50.0), closed_raw=np.full(5, 150.0))
    np.testing.assert_allclose(pct, 50.0)


def test_percent_to_servo_min_max_inversion() -> None:
    pct = np.array([0.0, 100.0, 50.0, 0.0, 100.0])
    angles = percent_to_servo(pct, [0] * 5, [180] * 5, [False, False, False, True, True])
    np.testing.assert_allclose(angles, [0.0, 180.0, 90.0, 180.0, 0.0])


def test_percent_to_servo_bornes_par_doigt() -> None:
    angles = percent_to_servo(np.full(5, 100.0), [10] * 5, [160] * 5, [False] * 5)
    np.testing.assert_allclose(angles, 160.0)
    angles = percent_to_servo(np.full(5, 0.0), [10] * 5, [160] * 5, [False] * 5)
    np.testing.assert_allclose(angles, 10.0)
