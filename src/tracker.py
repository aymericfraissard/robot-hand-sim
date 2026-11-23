"""Capture webcam (OpenCV) et détection de la main (MediaPipe Tasks HandLandmarker).

MediaPipe ≥ 0.10.x / 1.x : l'ancienne API `mp.solutions.hands` n'existe plus.
On utilise l'API Tasks en mode VIDEO (détection synchrone image par image,
avec suivi temporel entre les images → rapide et stable).
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from numpy.typing import NDArray

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Liste des segments du squelette (paires d'indices de landmarks).
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]

_CAMERA_APIS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


@dataclass
class HandDetection:
    """Résultat de détection pour une main."""

    image_landmarks: FloatArray  # 21 × 3, x/y normalisés [0, 1] dans l'image (pour le dessin)
    world_landmarks: FloatArray  # 21 × 3, en mètres, centrés sur la main (pour les angles)
    handedness: str              # "Left" / "Right"
    score: float


def ensure_model(model_path: Path, url: str) -> Path:
    """Télécharge le modèle `hand_landmarker.task` s'il est absent."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Téléchargement du modèle MediaPipe depuis %s ...", url)
    tmp = model_path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(model_path)
    log.info("Modèle enregistré : %s (%.1f Mo)", model_path, model_path.stat().st_size / 1e6)
    return model_path


class Camera:
    """Webcam OpenCV avec résolution/FPS demandés."""

    def __init__(
        self, index: int, width: int, height: int, fps: int, api: str = "any", exposure: float | str = "auto"
    ) -> None:
        backend = _CAMERA_APIS.get(api.lower(), cv2.CAP_ANY)
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la webcam n°{index} (api={api}). "
                "Vérifiez camera.index dans config.yaml et qu'aucune autre appli ne l'utilise."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Le MJPG permet 30 FPS en 640×480 (le YUY2 non compressé plafonne souvent à 15).
        # Avec DirectShow, il doit être demandé APRÈS la résolution, sinon il est ignoré.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limite la latence
        # Exposition fixe : évite que la webcam ne descende à 15 FPS quand la
        # pièce est peu éclairée (temps de pose automatique trop long).
        self._manual_exposure = str(exposure).lower() != "auto"
        if self._manual_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manuel (DirectShow)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        log.info(
            "Webcam ouverte : %dx%d @ %.0f FPS, exposition %s",
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.CAP_PROP_FPS),
            exposure,
        )

    def read(self) -> NDArray[np.uint8] | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        # Remet l'exposition automatique : certains pilotes conservent le réglage
        # pour les autres applications (Teams, Zoom...).
        if self._manual_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.cap.release()


class HandTracker:
    """Détecteur de main MediaPipe (1 main) en mode VIDEO."""

    def __init__(
        self,
        model_path: Path,
        model_url: str,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        ensure_model(model_path, model_url)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: NDArray[np.uint8]) -> HandDetection | None:
        """Détecte la main dans une image BGR (OpenCV). Renvoie None si aucune main."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Le mode VIDEO exige des horodatages strictement croissants.
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        img_lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float64)
        world_lm = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float64)
        cat = result.handedness[0][0] if result.handedness else None
        return HandDetection(
            image_landmarks=img_lm,
            world_landmarks=world_lm,
            handedness=cat.category_name if cat else "?",
            score=float(cat.score) if cat else 0.0,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def draw_hand(frame: NDArray[np.uint8], detection: HandDetection) -> None:
    """Dessine le squelette de la main sur l'image (en place)."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in detection.image_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        # Bouts des doigts en orange, autres articulations en vert.
        color = (0, 140, 255) if i in (4, 8, 12, 16, 20) else (80, 220, 80)
        cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
