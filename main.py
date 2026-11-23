"""Main robotique pilotée par webcam — point d'entrée.

Exemples :
    python main.py --mode sim        # main 3D dans le navigateur
    python main.py --mode serial     # vraie main (Arduino)
    python main.py --mode both       # les deux
    python main.py --demo            # sans webcam : mouvements automatiques

Touches (fenêtre vidéo) : q / Échap = quitter, c = calibrer, m = miroir.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from src.backends import MODES, Backend, HandState, create_backend
from src.calibration import Calibration, CalibrationProcedure
from src.finger_angles import FINGER_NAMES, NUM_FINGERS, percent_to_servo, raw_flexions, raw_to_percent
from src.smoothing import ExponentialSmoother
from src.wrist_angles import HandednessFilter, WristAxisConfig, hand_frame, wrist_angles

WRIST_NAMES = ("rotation", "inclin.")

ROOT = Path(__file__).resolve().parent
WINDOW = "Robot Hand - webcam"
log = logging.getLogger("main")

# Couleurs OpenCV (BGR).
GREEN, RED, WHITE, GREY, BEIGE, YELLOW = (80, 200, 80), (60, 60, 230), (255, 255, 255), (150, 150, 150), (162, 207, 227), (60, 200, 240)


# ---------------------------------------------------------------- Config
def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    for key in ("min_angles", "max_angles", "invert"):
        if len(cfg["angles"][key]) != NUM_FINGERS:
            raise ValueError(f"config.yaml : angles.{key} doit contenir {NUM_FINGERS} valeurs")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main robotique pilotée par la webcam (MediaPipe).")
    p.add_argument("--mode", choices=MODES, help="Sortie (remplace « mode » de config.yaml)")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="Fichier de configuration")
    p.add_argument("--demo", action="store_true", help="Sans webcam : mouvements de doigts automatiques")
    p.add_argument("--camera", type=int, help="Index de la webcam (remplace camera.index)")
    p.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir le navigateur")
    p.add_argument("--duration", type=float, default=0.0, help="Arrêt automatique après N secondes (tests)")
    return p.parse_args()


# ------------------------------------------------------------------ Démo
def demo_percents(t: float, speed: float) -> NDArray[np.float64]:
    """Mouvements automatiques : vague de doigts, puis poing / main ouverte."""
    t *= speed
    cycle = t % 12.0
    if cycle < 8.0:  # vague
        phases = np.arange(NUM_FINGERS) * 0.7
        return 50.0 - 50.0 * np.cos(1.6 * t - phases)
    # poing qui se ferme puis s'ouvre
    return np.full(NUM_FINGERS, 50.0 - 50.0 * math.cos((cycle - 8.0) * math.pi / 2.0))


def demo_wrist(t: float, speed: float) -> NDArray[np.float64]:
    """Poignet en démo : rotation lente ±60°, inclinaison ±20°."""
    t *= speed
    return np.array([60.0 * math.sin(0.5 * t), 20.0 * math.sin(0.9 * t + 1.0)])


# -------------------------------------------------------------- Affichage
def draw_overlay(
    frame: NDArray[np.uint8],
    percents: NDArray[np.float64],
    servos: NDArray[np.float64],
    detected: bool,
    fps: float,
    backend: Backend,
    calib: Calibration,
    message: str,
    demo: bool,
    wrist_deg: NDArray[np.float64] | None = None,
    wrist_servos: tuple[int, ...] = (),
    wrist_axes: list[WristAxisConfig] | None = None,
) -> None:
    """Panneau d'informations : angles par doigt et du poignet, FPS, état, aide."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = NUM_FINGERS + (len(WRIST_NAMES) if wrist_axes else 0)
    status_y = 52 + rows * 24 + 2
    # Fond semi-transparent du panneau.
    panel = frame[0 : status_y + 12, 0:300]
    panel[:] = (panel * 0.45).astype(np.uint8)

    state, color = ("MODE DEMO", YELLOW) if demo else (("Main detectee", GREEN) if detected else ("Aucune main", RED))
    cv2.putText(frame, state, (10, 24), font, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (200, 24), font, 0.6, WHITE, 1, cv2.LINE_AA)

    for i, name in enumerate(FINGER_NAMES):
        y = 52 + i * 24
        cv2.putText(frame, name, (10, y), font, 0.5, WHITE, 1, cv2.LINE_AA)
        x0, x1 = 110, 210
        cv2.rectangle(frame, (x0, y - 10), (x1, y), GREY, 1)
        cv2.rectangle(frame, (x0, y - 10), (x0 + int((x1 - x0) * percents[i] / 100.0), y), BEIGE, -1)
        cv2.putText(frame, f"{servos[i]:3.0f} deg", (220, y), font, 0.5, WHITE, 1, cv2.LINE_AA)

    # Poignet : jauge centrée sur la position neutre.
    if wrist_axes and wrist_deg is not None:
        for j, (name, axis) in enumerate(zip(WRIST_NAMES, wrist_axes)):
            y = 52 + (NUM_FINGERS + j) * 24
            cv2.putText(frame, name, (10, y), font, 0.5, WHITE if axis.enabled else GREY, 1, cv2.LINE_AA)
            x0, x1 = 110, 210
            xc = (x0 + x1) // 2
            cv2.rectangle(frame, (x0, y - 10), (x1, y), GREY, 1)
            cv2.line(frame, (xc, y - 12), (xc, y + 2), GREY, 1)
            if axis.enabled:
                lo, hi = axis.range_deg
                v = axis.clamp(float(wrist_deg[j]))
                tip = xc + int((x1 - xc) * v / hi) if v >= 0 else xc - int((xc - x0) * v / lo)
                cv2.rectangle(frame, (min(xc, tip), y - 10), (max(xc, tip), y), BEIGE, -1)
                cv2.putText(frame, f"{wrist_servos[j]:3d} deg", (220, y), font, 0.5, WHITE, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "off", (220, y), font, 0.5, GREY, 1, cv2.LINE_AA)

    cv2.putText(frame, backend.status(), (10, status_y), font, 0.45, GREY, 1, cv2.LINE_AA)
    if calib.is_default and not demo:
        cv2.putText(frame, "Calibration par defaut : appuyez sur c", (10, h - 36), font, 0.5, YELLOW, 1, cv2.LINE_AA)
    if message:
        (tw, _), _ = cv2.getTextSize(message, font, 0.8, 2)
        cv2.putText(frame, message, ((w - tw) // 2, h // 2), font, 0.8, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, "q: quitter   c: calibrer   m: miroir", (10, h - 12), font, 0.5, WHITE, 1, cv2.LINE_AA)


# -------------------------------------------------------------- Programme
def run(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    mode = args.mode or cfg["mode"]
    angles_cfg, cam_cfg, calib_cfg = cfg["angles"], cfg["camera"], cfg["calibration"]

    calib_path = ROOT / calib_cfg["file"]
    calibration = Calibration.load(calib_path)
    procedure = CalibrationProcedure(float(calib_cfg["countdown_s"]), float(calib_cfg["capture_s"]))
    smoother = ExponentialSmoother(float(angles_cfg["smoothing_alpha"]), NUM_FINGERS)
    return_alpha = float(angles_cfg["return_alpha"])
    mirror = bool(cam_cfg["mirror"])

    # Poignet (rotation, inclinaison).
    wrist_cfg = cfg.get("wrist", {})
    wrist_on = bool(wrist_cfg.get("enabled", False))
    wrist_axes = (
        [WristAxisConfig.from_dict(wrist_cfg["rotation"]), WristAxisConfig.from_dict(wrist_cfg["deviation"])]
        if wrist_on
        else None
    )
    wrist_smoother = ExponentialSmoother(float(wrist_cfg.get("smoothing_alpha", 0.35)), 2)
    handedness = HandednessFilter()
    demo_speed = float(cfg.get("demo", {}).get("speed", 1.0))

    # Source d'images : webcam + MediaPipe, ou démo.
    camera = tracker = None
    if not args.demo:
        from src.tracker import Camera, HandTracker, draw_hand

        trk = cfg["tracking"]
        tracker = HandTracker(
            ROOT / trk["model_path"],
            trk["model_url"],
            float(trk["min_detection_confidence"]),
            float(trk["min_presence_confidence"]),
            float(trk["min_tracking_confidence"]),
        )
        index = args.camera if args.camera is not None else int(cam_cfg["index"])
        camera = Camera(
            index, int(cam_cfg["width"]), int(cam_cfg["height"]), int(cam_cfg["fps"]),
            str(cam_cfg["api"]), cam_cfg.get("exposure", "auto"),
        )

    backend = create_backend(mode, cfg, ROOT / "web", open_browser=not args.no_browser)
    backend.start()
    log.info("Mode « %s »%s. Touches : q quitter, c calibrer, m miroir.", mode, " (démo)" if args.demo else "")
    log.info(
        "Poignet %s : trame série à %d valeurs.",
        "activé (rotation, inclinaison)" if wrist_on else "désactivé",
        NUM_FINGERS + (2 if wrist_on else 0),
    )

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    fps, last = 0.0, time.perf_counter()
    start = last
    message, message_until = "", 0.0
    try:
        while True:
            now = time.perf_counter()
            # --- 1. Image + détection
            detected = False
            raw: NDArray[np.float64] | None = None
            wrist_sample: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
            wrist_target: NDArray[np.float64] | None = None
            if camera is not None and tracker is not None:
                frame = camera.read()
                if frame is None:
                    log.error("Lecture webcam impossible (débranchée ?)")
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                detection = tracker.detect(frame)
                if detection is not None:
                    detected = True
                    raw = raw_flexions(detection.world_landmarks)
                    # Poignet : repère de la main comparé à la pose neutre calibrée.
                    h_sign = handedness.update(detection.handedness)
                    hf = hand_frame(detection.world_landmarks, h_sign)
                    wrist_sample = (hf.long, hf.normal)
                    rot, dev = wrist_angles(hf, calibration.wrist_long, calibration.wrist_normal, h_sign)
                    wrist_target = np.array([rot, dev])
                    draw_hand(frame, detection)
                target = raw_to_percent(raw, calibration.open_raw, calibration.closed_raw) if raw is not None else None
            else:
                frame = np.full((int(cam_cfg["height"]), int(cam_cfg["width"]), 3), 40, np.uint8)
                time.sleep(1 / 30)  # ~30 FPS comme une webcam
                detected = True
                target = demo_percents(now - start, demo_speed)
                wrist_target = demo_wrist(now - start, demo_speed)

            # --- 2. Calibration en cours ?
            if procedure.active:
                result = procedure.update(now, raw, wrist_sample)
                if result is not None:
                    calibration = result
                    calibration.save(calib_path)
                if not procedure.active:  # vient de se terminer : message affiché 3 s
                    message, message_until = procedure.message, now + 3.0

            # --- 3. Lissage (retour doux vers « main ouverte » sans main)
            if target is not None:
                percents = smoother.update(target)
            else:
                percents = smoother.update(np.zeros(NUM_FINGERS), alpha=return_alpha)
            servos = percent_to_servo(percents, angles_cfg["min_angles"], angles_cfg["max_angles"], angles_cfg["invert"])

            # Poignet : angles bornés à la plage AVANT lissage (évite qu'un saut
            # ±180° de la rotation ne traverse toute la plage), retour au neutre sans main.
            wrist_deg = np.zeros(2)
            wrist_servos: tuple[int, ...] = ()
            if wrist_axes is not None:
                if wrist_target is not None:
                    clamped = np.array([ax.clamp(float(v)) for ax, v in zip(wrist_axes, wrist_target)])
                    wrist_deg = wrist_smoother.update(clamped)
                else:
                    wrist_deg = wrist_smoother.update(np.zeros(2), alpha=return_alpha)
                wrist_servos = tuple(ax.to_servo(float(v)) for ax, v in zip(wrist_axes, wrist_deg))

            # --- 4. Envoi aux sorties
            backend.send(
                HandState(
                    percents=tuple(float(p) for p in percents),
                    finger_servos=tuple(int(round(a)) for a in servos),
                    hand_detected=detected,
                    wrist_degrees=(float(wrist_deg[0]), float(wrist_deg[1])),
                    wrist_servos=wrist_servos,
                )
            )

            # --- 5. Affichage
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            shown = procedure.message if procedure.active else (message if now < message_until else "")
            draw_overlay(
                frame, percents, servos, detected, fps, backend, calibration, shown, args.demo,
                wrist_deg, wrist_servos, wrist_axes,
            )
            cv2.imshow(WINDOW, frame)

            # --- 6. Clavier
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if args.demo:
                    message, message_until = "Calibration indisponible en mode demo", now + 2.0
                else:
                    procedure.toggle(now)
                    if not procedure.active:
                        message, message_until = procedure.message, now + 2.0
            elif key == ord("m"):
                mirror = not mirror
            # Fenêtre fermée avec la croix.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.duration and now - start > args.duration:
                log.info("Durée --duration atteinte (%.0f s), FPS moyen ≈ %.1f", args.duration, fps)
                break
    finally:
        backend.stop()
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        log.info("Arrêt propre.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s : %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("websockets").setLevel(logging.WARNING)  # nos propres messages suffisent
    args = parse_args()
    try:
        cfg = load_config(args.config)
        run(args, cfg)
    except (RuntimeError, ValueError, OSError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
