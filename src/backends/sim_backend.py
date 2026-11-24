"""Sortie « sim » : sert la page 3D (HTTP) et diffuse les angles (WebSocket).

Deux serveurs tournent dans des threads en arrière-plan pour ne jamais
ralentir la boucle vidéo :
  - un petit serveur HTTP qui sert le dossier `web/` ;
  - un serveur WebSocket (asyncio) qui diffuse un message JSON par image.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.asyncio.server import Server, ServerConnection, broadcast, serve

from src.backends.base import Backend, HandState

log = logging.getLogger(__name__)


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serveur de fichiers statiques sans journal de chaque requête."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def end_headers(self) -> None:
        # Pas de cache : les modifications de hand.js sont visibles en rechargeant.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class SimBackend(Backend):
    name = "sim"

    def __init__(
        self,
        web_dir: Path,
        host: str = "127.0.0.1",
        http_port: int = 8000,
        ws_port: int = 8765,
        open_browser: bool = True,
    ) -> None:
        self.web_dir = web_dir
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_browser = open_browser
        self._httpd: ThreadingHTTPServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server: Server | None = None
        self._ws_thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._last_message: str | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}/?ws={self.ws_port}"

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        # Serveur HTTP (fichiers statiques du dossier web/).
        handler = functools.partial(_QuietHandler, directory=str(self.web_dir))
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.http_port), handler)
        except OSError as exc:
            raise RuntimeError(
                f"Port HTTP {self.http_port} déjà utilisé : changez sim.http_port dans config.yaml"
            ) from exc
        threading.Thread(target=self._httpd.serve_forever, name="http", daemon=True).start()

        # Serveur WebSocket dans sa propre boucle asyncio.
        ready = threading.Event()
        errors: list[BaseException] = []
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, args=(ready, errors), name="websocket", daemon=True
        )
        self._ws_thread.start()
        if not ready.wait(timeout=5) or errors or self._ws_server is None:
            self.stop()
            raise RuntimeError(
                f"Serveur WebSocket non démarré sur le port {self.ws_port} "
                "(port déjà utilisé ? changez sim.ws_port dans config.yaml)"
            ) from (errors[0] if errors else None)

        log.info("Visualiseur 3D : %s", self.url)
        if self.open_browser:
            webbrowser.open(self.url)

    def _run_ws_loop(self, ready: threading.Event, errors: list[BaseException]) -> None:
        """Corps du thread WebSocket : exécute la boucle asyncio jusqu'à l'arrêt."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_main(loop, ready))
        except Exception as exc:  # noqa: BLE001 — remonté au thread principal par start()
            errors.append(exc)
        finally:
            ready.set()
            loop.close()

    async def _ws_main(self, loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        # websockets ≥ 14 : serve() doit être appelé dans une boucle asyncio en marche.
        self._stop_event = asyncio.Event()
        async with serve(self._on_client, self.host, self.ws_port) as server:
            self._ws_server = server
            self._loop = loop
            ready.set()
            await self._stop_event.wait()  # jusqu'à stop()
        self._ws_server = None

    async def _on_client(self, ws: ServerConnection) -> None:
        """Nouveau navigateur connecté : on lui envoie tout de suite le dernier état."""
        log.info("Visualiseur connecté (%s)", ws.remote_address)
        if self._last_message is not None:
            await ws.send(self._last_message)
        # On ignore les messages entrants ; on attend juste la déconnexion.
        async for _ in ws:
            pass
        log.info("Visualiseur déconnecté")

    # ------------------------------------------------------------------- send
    def send(self, state: HandState) -> None:
        # « flex » = flexion logique 0–180° qui anime la main 3D (non inversée).
        message = json.dumps(
            {
                "flex": [round(p * 1.8, 1) for p in state.percents],
                "percent": [round(p, 1) for p in state.percents],
                "servo": list(state.finger_servos),
                # Poignet : inclinaison logique (degrés, 0 = neutre, + = vers le pouce).
                "deviation": round(state.wrist_deviation, 1),
                "hand": state.hand_detected,
            }
        )
        self._last_message = message
        loop = self._loop
        if loop is not None and not loop.is_closed():
            # broadcast() n'est pas thread-safe : on le planifie dans la boucle asyncio.
            try:
                loop.call_soon_threadsafe(self._broadcast, message)
            except RuntimeError:
                pass  # boucle en cours d'arrêt

    def _broadcast(self, message: str) -> None:
        """Exécuté dans le thread asyncio : envoie à tous les navigateurs connectés."""
        if self._ws_server is not None:
            broadcast(self._ws_server.connections, message)

    def status(self) -> str:
        n = len(self._ws_server.connections) if self._ws_server is not None else 0
        return f"sim: {n} navigateur(s)"

    # ------------------------------------------------------------------- stop
    def stop(self) -> None:
        if self._loop is not None and self._stop_event is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3)
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self._loop = self._ws_server = self._httpd = self._ws_thread = None
