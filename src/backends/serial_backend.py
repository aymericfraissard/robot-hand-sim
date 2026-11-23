"""Sortie « serial » : envoie les angles à un Arduino par liaison série USB.

Trame texte (une ligne par envoi), angles entiers 0–180 :
    $a0,a1,a2,a3,a4\n          doigts seuls (poignet désactivé)
    $a0,a1,a2,a3,a4,a5,a6\n    avec le poignet
  a0–a4 : pouce, index, majeur, annulaire, auriculaire ;
  a5 : rotation du poignet, a6 : inclinaison du poignet.

Si le port est absent ou se déconnecte, on réessaie périodiquement sans
jamais bloquer la boucle vidéo.
"""

from __future__ import annotations

import logging
import time

import serial
from serial.tools import list_ports

from src.backends.base import Backend, HandState

log = logging.getLogger(__name__)

# L'ouverture du port redémarre la plupart des Arduino (signal DTR) :
# on attend la fin du bootloader avant d'envoyer des trames.
ARDUINO_BOOT_DELAY_S = 2.0


def format_frame(angles: tuple[int, ...] | list[int]) -> bytes:
    """Construit la trame « $a0,a1,...\\n » (angles bornés à 0–180)."""
    values = ",".join(str(max(0, min(180, int(round(a))))) for a in angles)
    return f"${values}\n".encode("ascii")


class SerialBackend(Backend):
    name = "serial"

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        send_rate_hz: float = 50.0,
        reconnect_interval_s: float = 2.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.min_period = 1.0 / send_rate_hz if send_rate_hz > 0 else 0.0
        self.reconnect_interval_s = reconnect_interval_s
        self._ser: serial.Serial | None = None
        self._ready_at = 0.0
        self._last_attempt = -1e9
        self._last_send = 0.0
        self._warned = False

    def start(self) -> None:
        self._try_connect(time.monotonic())

    def _try_connect(self, now: float) -> None:
        self._last_attempt = now
        try:
            # write_timeout=0 : écriture non bloquante (on préfère perdre une trame que ralentir la vidéo).
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0, write_timeout=0)
        except (serial.SerialException, OSError) as exc:
            self._ser = None
            if not self._warned:
                ports = ", ".join(p.device for p in list_ports.comports()) or "aucun"
                log.warning(
                    "Port série %s indisponible (%s). Ports détectés : %s. Nouvel essai toutes les %.0f s.",
                    self.port, exc, ports, self.reconnect_interval_s,
                )
                self._warned = True
            return
        self._ready_at = now + ARDUINO_BOOT_DELAY_S
        self._warned = False
        log.info("Port série %s ouvert à %d bauds", self.port, self.baudrate)

    def send(self, state: HandState) -> None:
        now = time.monotonic()
        if self._ser is None:
            if now - self._last_attempt >= self.reconnect_interval_s:
                self._try_connect(now)
            return
        if now < self._ready_at or now - self._last_send < self.min_period:
            return
        try:
            self._ser.write(format_frame(state.servo_frame))
            self._last_send = now
        except serial.SerialTimeoutException:
            pass  # tampon plein : trame ignorée, la suivante partira
        except (serial.SerialException, OSError) as exc:
            log.warning("Liaison série perdue (%s) : reconnexion...", exc)
            self._close()
            self._last_attempt = now

    def _close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except (serial.SerialException, OSError):
                pass
        self._ser = None

    def stop(self) -> None:
        self._close()

    def status(self) -> str:
        if self._ser is None:
            return f"serial: {self.port} absent"
        if time.monotonic() < self._ready_at:
            return f"serial: {self.port} (demarrage Arduino)"
        return f"serial: {self.port} OK"
