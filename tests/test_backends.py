"""Tests des sorties (sans matériel)."""

from __future__ import annotations

from src.backends import create_backend
from src.backends.base import HandState
from src.backends.serial_backend import SerialBackend, format_frame


def test_format_trame() -> None:
    assert format_frame([0, 45, 90, 135, 180]) == b"$0,45,90,135,180\n"


def test_format_trame_bornee_et_arrondie() -> None:
    assert format_frame([-5, 12.6, 200, 0, 0]) == b"$0,13,180,0,0\n"  # type: ignore[list-item]


def test_trame_avec_poignet() -> None:
    state = HandState((0.0,) * 5, (10, 20, 30, 40, 50), True, (12.0, -5.0), (100, 80))
    assert state.servo_frame == (10, 20, 30, 40, 50, 100, 80)
    assert format_frame(state.servo_frame) == b"$10,20,30,40,50,100,80\n"
    assert HandState((0.0,) * 5, (1, 2, 3, 4, 5), True).servo_frame == (1, 2, 3, 4, 5)


def test_port_absent_ne_plante_pas() -> None:
    """Sans Arduino branché, l'envoi doit être silencieux et non bloquant."""
    b = SerialBackend(port="COM_INEXISTANT", reconnect_interval_s=0.0)
    b.start()
    b.send(HandState((0.0,) * 5, (90,) * 5, True))
    assert "absent" in b.status()
    b.stop()


def test_mode_inconnu() -> None:
    import pytest

    with pytest.raises(ValueError):
        create_backend("laser", {}, web_dir=None)  # type: ignore[arg-type]
