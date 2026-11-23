"""Tests du lissage EMA et de la procédure de calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.calibration import Calibration, CalibrationProcedure
from src.finger_angles import raw_flexions
from src.smoothing import ExponentialSmoother
from tests.synthetic_hand import make_hand


def test_ema_converge_vers_la_cible() -> None:
    s = ExponentialSmoother(alpha=0.5, size=5)
    for _ in range(30):
        out = s.update(np.full(5, 100.0))
    np.testing.assert_allclose(out, 100.0, atol=1e-3)


def test_ema_lisse_un_saut() -> None:
    s = ExponentialSmoother(alpha=0.3, size=5)
    out = s.update(np.full(5, 100.0))
    np.testing.assert_allclose(out, 30.0)


def _run_procedure(proc: CalibrationProcedure, open_raw: np.ndarray, closed_raw: np.ndarray) -> Calibration | None:
    """Simule 30 images/s pendant toute la procédure."""
    proc.toggle(0.0)
    t, result = 0.0, None
    while t < 20.0 and result is None:
        t += 1 / 30
        raw = open_raw if t < proc.countdown_s + proc.capture_s else closed_raw
        result = proc.update(t, raw)
    return result


def test_procedure_complete() -> None:
    open_raw, closed_raw = raw_flexions(make_hand(0.0)), raw_flexions(make_hand(1.0))
    calib = _run_procedure(CalibrationProcedure(countdown_s=1.0, capture_s=0.5), open_raw, closed_raw)
    assert calib is not None
    np.testing.assert_allclose(calib.open_raw, open_raw)
    np.testing.assert_allclose(calib.closed_raw, closed_raw)


def test_procedure_refuse_amplitude_trop_faible() -> None:
    raw = raw_flexions(make_hand(0.2))
    proc = CalibrationProcedure(countdown_s=1.0, capture_s=0.5)
    assert _run_procedure(proc, raw, raw) is None
    assert not proc.active
    assert "refusee" in proc.message


def test_sauvegarde_et_rechargement(tmp_path: Path) -> None:
    path = tmp_path / "calib.json"
    Calibration(np.arange(5.0), np.arange(5.0) + 100, is_default=False).save(path)
    loaded = Calibration.load(path)
    assert not loaded.is_default
    np.testing.assert_allclose(loaded.closed_raw, np.arange(5.0) + 100)


def test_pose_neutre_poignet_capturee_et_sauvegardee(tmp_path: Path) -> None:
    long, normal = np.array([0.0, -0.8, -0.6]), np.array([0.0, 0.6, -0.8])
    open_raw, closed_raw = raw_flexions(make_hand(0.0)), raw_flexions(make_hand(1.0))
    proc = CalibrationProcedure(countdown_s=1.0, capture_s=0.5)
    proc.toggle(0.0)
    t, calib = 0.0, None
    while t < 20.0 and calib is None:
        t += 1 / 30
        raw = open_raw if t < 1.5 else closed_raw
        calib = proc.update(t, raw, (long, normal))
    assert calib is not None
    np.testing.assert_allclose(calib.wrist_long, long, atol=1e-9)
    np.testing.assert_allclose(calib.wrist_normal, normal, atol=1e-9)
    path = tmp_path / "c.json"
    calib.save(path)
    np.testing.assert_allclose(Calibration.load(path).wrist_normal, normal, atol=1e-4)


def test_ancien_fichier_sans_poignet(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text('{"open_raw": [1,2,3,4,5], "closed_raw": [101,102,103,104,105]}', encoding="utf-8")
    calib = Calibration.load(path)
    assert not calib.is_default
    np.testing.assert_allclose(calib.wrist_long, [0.0, -1.0, 0.0])


def test_fichier_absent_ou_invalide(tmp_path: Path) -> None:
    assert Calibration.load(tmp_path / "absent.json").is_default
    bad = tmp_path / "bad.json"
    bad.write_text("{pas du json", encoding="utf-8")
    assert Calibration.load(bad).is_default
