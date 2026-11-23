"""Sorties interchangeables. `create_backend` construit celle demandée par la config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.backends.base import Backend, HandState, MultiBackend

__all__ = ["Backend", "HandState", "MultiBackend", "create_backend", "MODES"]

MODES = ("sim", "serial", "both")


def create_backend(mode: str, config: dict[str, Any], web_dir: Path, open_browser: bool = True) -> Backend:
    """Construit la sortie correspondant au mode : "sim", "serial" ou "both"."""
    # Imports locaux : pyserial n'est chargé que si on en a besoin.
    def make_sim() -> Backend:
        from src.backends.sim_backend import SimBackend

        sim = config["sim"]
        wrist = config.get("wrist", {})
        wrist_ranges = (
            [list(wrist["rotation"]["range_deg"]), list(wrist["deviation"]["range_deg"])]
            if wrist.get("enabled", False)
            else None
        )
        return SimBackend(
            web_dir=web_dir,
            host=sim["host"],
            http_port=int(sim["http_port"]),
            ws_port=int(sim["ws_port"]),
            open_browser=open_browser and bool(sim.get("open_browser", True)),
            wrist_ranges=wrist_ranges,
        )

    def make_serial() -> Backend:
        from src.backends.serial_backend import SerialBackend

        ser = config["serial"]
        return SerialBackend(
            port=str(ser["port"]),
            baudrate=int(ser["baudrate"]),
            send_rate_hz=float(ser["send_rate_hz"]),
            reconnect_interval_s=float(ser["reconnect_interval_s"]),
        )

    if mode == "sim":
        return make_sim()
    if mode == "serial":
        return make_serial()
    if mode == "both":
        return MultiBackend([make_sim(), make_serial()])
    raise ValueError(f"Mode inconnu « {mode} » (attendu : {', '.join(MODES)})")
