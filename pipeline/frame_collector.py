from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraSource:
    """Fuente de video para una cámara.

    Attributes:
        camera_id: Identificador único de la cámara.
        source: Path MP4 o URL RTSP.
        zone_id: ID de la zona minera asignada.
        fps: FPS objetivo de captura.
        resolution: Resolución de captura (width, height).
    """
    camera_id: str = field(compare=True)
    source: str = field(compare=False)
    zone_id: str = field(compare=False)
    fps: int = 25
    resolution: tuple[int, int] = (640, 480)


class FrameCollector:
    """Colector de frames multi-fuente con threads.

    Un thread daemon por cámara captura frames continuamente.
    collect() retorna el frame más reciente de cada cámara.

    Attributes:
        _sources: Lista de CameraSource.
        _threads: Dict de threads por camera_id.
        _frames: Dict de último frame por camera_id (con Lock).
        _running: Flag de control.
    """

    def __init__(self, sources: list[CameraSource]) -> None:
        """Inicializa el colector.

        Args:
            sources: Lista de fuentes de cámara.
        """
        self._sources = {s.camera_id: s for s in sources}
        self._threads: dict[str, threading.Thread] = {}
        self._frames: dict[str, np.ndarray | None] = {
            s.camera_id: None for s in sources
        }
        self._locks: dict[str, threading.Lock] = {
            s.camera_id: threading.Lock() for s in sources
        }
        self._running = False
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def num_sources(self) -> int:
        """Número de fuentes activas."""
        return len(self._sources)

    def start(self) -> None:
        """Inicia threads de captura para todas las cámaras."""
        if self._running:
            return
        self._running = True
        for camera_id, source in self._sources.items():
            thread = threading.Thread(
                target=self._capture_loop,
                args=(camera_id,),
                name=f"capture-{camera_id}",
                daemon=True,
            )
            self._threads[camera_id] = thread
            thread.start()
            self._logger.info("Capture thread started for %s", camera_id)

    def _capture_loop(self, camera_id: str) -> None:
        """Loop de captura para una cámara.

        Args:
            camera_id: ID de la cámara a capturar.
        """
        source = self._sources[camera_id]
        cap = cv.VideoCapture(source.source)

        if not cap.isOpened():
            self._logger.error(
                "Cannot open camera %s: %s", camera_id, source.source
            )
            return

        target_interval = 1.0 / source.fps if source.fps > 0 else 0
        last_capture = time.monotonic()

        while self._running:
            now = time.monotonic()
            if now - last_capture < target_interval:
                time.sleep(0.001)
                continue

            ret, frame = cap.read()
            if not ret:
                self._logger.warning("Frame read failed for %s", camera_id)
                cap.release()
                time.sleep(1)
                cap = cv.VideoCapture(source.source)
                continue

            last_capture = now
            with self._locks[camera_id]:
                self._frames[camera_id] = frame

        cap.release()

    def collect(self) -> dict[str, np.ndarray]:
        """Retorna el frame más reciente de cada cámara.

        Returns:
            Dict[camera_id, frame]. Solo cámaras con frame disponible.
        """
        result: dict[str, np.ndarray] = {}
        for camera_id in self._sources:
            with self._locks[camera_id]:
                frame = self._frames[camera_id]
                if frame is not None:
                    result[camera_id] = frame
        return result

    def release(self) -> None:
        """Detiene todos los threads y libera recursos."""
        self._running = False
        for thread in self._threads.values():
            thread.join(timeout=5)
        self._threads.clear()
        self._logger.info("FrameCollector released")


__all__ = ["CameraSource", "FrameCollector"]
