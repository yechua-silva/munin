from __future__ import annotations

import logging
from pathlib import Path

import cv2 as cv
import numpy as np

from munin.exceptions import VideoLoadError

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extractor de frames desde video MP4 usando OpenCV VideoCapture.

    Extrae frames a un framerate objetivo, saltando frames según sea
    necesario para igualar el framerate configurado.

    Attributes:
        _fps: Framerate objetivo de extracción (frames por segundo).
        _logger: Logger de la clase.
    """

    def __init__(self, fps: int = 25) -> None:
        """Inicializa FrameExtractor con framerate configurable.

        Args:
            fps: Framerate objetivo en frames por segundo (default: 25).
        """
        self._fps: int = fps
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract(self, video_path: str) -> list[np.ndarray]:
        """Extrae frames de un video MP4.

        Abre el video con OpenCV VideoCapture, lee todos los frames,
        y retorna solo aquellos que corresponden al framerate objetivo.
        El salteo de frames se calcula como: step = round(fps_original / fps_target),
        asegurando step >= 1.

        Args:
            video_path: Ruta al archivo MP4 a procesar.

        Returns:
            Lista de frames como np.ndarray con shape (H, W, 3),
            dtype uint8, canal BGR (formato OpenCV nativo).

        Raises:
            VideoLoadError: Si el archivo no existe, no puede ser abierto
                por OpenCV, el códec no es soportado, o el video
                produce 0 frames.
        """
        video_path_obj = Path(video_path)

        if not video_path_obj.exists():
            raise VideoLoadError(f"Video file not found: {video_path}")

        self._logger.info("Opening video: %s", video_path)
        cap = cv.VideoCapture(str(video_path_obj))

        if not cap.isOpened():
            raise VideoLoadError(
                f"Cannot open video file: {video_path}. "
                f"Check codec support and file integrity."
            )

        frames: list[np.ndarray] = []

        try:
            fps_original: float = cap.get(cv.CAP_PROP_FPS)
            total_frames: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            if fps_original <= 0:
                raise VideoLoadError(
                    f"Invalid original FPS ({fps_original}) for video: {video_path}"
                )

            step: int = max(1, round(fps_original / self._fps))

            self._logger.info(
                "Video: %s | Original FPS: %.2f | Target FPS: %d | "
                "Total frames: %d | Step: %d",
                video_path,
                fps_original,
                self._fps,
                total_frames,
                step,
            )

            frame_idx: int = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    frames.append(frame)

                frame_idx += 1

            self._logger.info(
                "Extracted %d frames from %s (step=%d)",
                len(frames),
                video_path,
                step,
            )

        except VideoLoadError:
            raise
        except Exception as e:
            raise VideoLoadError(
                f"Error processing video {video_path}: {e}"
            ) from e
        finally:
            cap.release()
            self._logger.debug("Released VideoCapture for: %s", video_path)

        if len(frames) == 0:
            raise VideoLoadError(
                f"No frames extracted from video: {video_path}. "
                f"The video may be corrupt or empty."
            )

        return frames
