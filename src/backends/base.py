"""Interface commune à toutes les sorties (simulation, série, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class HandState:
    """État de la main envoyé aux sorties à chaque image.

    Ordre des doigts : pouce, index, majeur, annulaire, auriculaire.
    """

    percents: tuple[float, ...]       # Flexion logique 0 % (tendu) → 100 % (plié)
    finger_servos: tuple[int, ...]    # Angles servo des doigts 0–180° (min/max et inversion appliqués)
    hand_detected: bool
    wrist_deviation: float = 0.0      # Inclinaison logique du poignet (degrés, 0 = neutre, + = vers le pouce)
    wrist_servos: tuple[int, ...] = ()  # () si poignet désactivé, sinon (inclinaison,)

    @property
    def servo_frame(self) -> tuple[int, ...]:
        """Tous les angles servo dans l'ordre de la trame série (5 ou 6 valeurs)."""
        return self.finger_servos + self.wrist_servos


class Backend(ABC):
    """Classe de base d'une sortie. Pour ajouter une sortie, hériter de cette
    classe, implémenter les méthodes abstraites et l'enregistrer dans
    `src/backends/__init__.py` (fonction `create_backend`)."""

    name: str = "backend"

    @abstractmethod
    def start(self) -> None:
        """Ouvre les ressources (serveur, port série...). Ne doit pas bloquer longtemps."""

    @abstractmethod
    def send(self, state: HandState) -> None:
        """Envoie l'état courant. Appelée à chaque image : doit être rapide et non bloquante."""

    @abstractmethod
    def stop(self) -> None:
        """Libère les ressources."""

    def status(self) -> str:
        """Courte description de l'état (affichée dans la fenêtre OpenCV)."""
        return self.name

    def __enter__(self) -> "Backend":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class MultiBackend(Backend):
    """Envoie le même état à plusieurs sorties (mode « both »)."""

    name = "multi"

    def __init__(self, backends: list[Backend]) -> None:
        self.backends = backends

    def start(self) -> None:
        for b in self.backends:
            b.start()

    def send(self, state: HandState) -> None:
        for b in self.backends:
            b.send(state)

    def stop(self) -> None:
        for b in reversed(self.backends):
            b.stop()

    def status(self) -> str:
        return " | ".join(b.status() for b in self.backends)
