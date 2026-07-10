"""Tests T15 — CameraSource + FrameCollector.

Mock de cv.VideoCapture para no requerir cámara real.
Verifica:
- CameraSource dataclass crea correctamente
- FrameCollector init con N sources
- num_sources property
- collect() retorna dict de frames disponibles
- release() detiene threads

Correr con: pytest tests/test_frame_collector.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from munin.pipeline.frame_collector import CameraSource, FrameCollector


class TestCameraSource:
    """Tests para CameraSource dataclass."""

    def test_minimal_creation(self) -> None:
        """CameraSource con solo campos requeridos."""
        cs = CameraSource(
            camera_id="cam01",
            source="/path/to/video.mp4",
            zone_id="extraccion",
        )
        assert cs.camera_id == "cam01"
        assert cs.source == "/path/to/video.mp4"
        assert cs.zone_id == "extraccion"
        assert cs.fps == 25  # default
        assert cs.resolution == (640, 480)  # default

    def test_full_creation(self) -> None:
        """CameraSource con todos los campos."""
        cs = CameraSource(
            camera_id="cam02",
            source="rtsp://cam2/stream",
            zone_id="procesamiento",
            fps=30,
            resolution=(1280, 720),
        )
        assert cs.camera_id == "cam02"
        assert cs.source == "rtsp://cam2/stream"
        assert cs.zone_id == "procesamiento"
        assert cs.fps == 30
        assert cs.resolution == (1280, 720)

    def test_immutable_like(self) -> None:
        """CameraSource es frozen-like (dataclass frozen por defecto con slots?)."""
        cs = CameraSource(camera_id="c", source="s", zone_id="z")
        # Puede mutar porque es dataclass estándar (no frozen)
        cs.fps = 15
        assert cs.fps == 15

    def test_equality_by_camera_id(self) -> None:
        """CameraSource compara por camera_id (field compare=True)."""
        cs1 = CameraSource(camera_id="cam01", source="a", zone_id="z")
        cs2 = CameraSource(camera_id="cam01", source="b", zone_id="z")
        cs3 = CameraSource(camera_id="cam02", source="a", zone_id="z")
        assert cs1 == cs2  # same camera_id
        assert cs1 != cs3  # different camera_id


class TestFrameCollector:
    """Tests para FrameCollector."""

    @pytest.fixture
    def mock_sources(self) -> list[CameraSource]:
        """2 fuentes de cámara mock."""
        return [
            CameraSource(camera_id="cam01", source="vid1.mp4", zone_id="z1"),
            CameraSource(camera_id="cam02", source="rtsp://cam2", zone_id="z2"),
        ]

    @pytest.fixture
    def mock_frame(self) -> np.ndarray:
        """Frame sintético de prueba."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_init_with_sources(self, mock_sources: list[CameraSource]) -> None:
        """FrameCollector init con lista de fuentes."""
        collector = FrameCollector(mock_sources)
        assert collector.num_sources == 2

    def test_init_empty_sources(self) -> None:
        """FrameCollector con lista vacía."""
        collector = FrameCollector([])
        assert collector.num_sources == 0

    def test_num_sources_property(
        self, mock_sources: list[CameraSource]
    ) -> None:
        """num_sources retorna cantidad correcta."""
        collector = FrameCollector(mock_sources)
        assert collector.num_sources == 2
        collector2 = FrameCollector([mock_sources[0]])
        assert collector2.num_sources == 1

    def test_collect_returns_available_frames(
        self, mock_sources: list[CameraSource]
    ) -> None:
        """collect() retorna frames disponibles de todas las cámaras."""
        collector = FrameCollector(mock_sources)
        # Sin start(), los frames son None → collect vacío
        result = collector.collect()
        assert isinstance(result, dict)
        assert len(result) == 0

    @patch("cv2.VideoCapture")
    def test_start_creates_threads(
        self,
        mock_vc: MagicMock,
        mock_sources: list[CameraSource],
    ) -> None:
        """start() inicia un thread por cámara."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        # Frame read retorna (True, frame) la primera vez, luego False
        mock_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_vc.return_value = mock_instance

        collector = FrameCollector(mock_sources)
        collector.start()

        # Verificar que se crearon los threads correctos
        for camera_id in ["cam01", "cam02"]:
            assert camera_id in collector._threads
            assert collector._threads[camera_id].is_alive()

        collector.release()

    @patch("cv2.VideoCapture")
    def test_collect_after_start_returns_frames(
        self,
        mock_vc: MagicMock,
        mock_sources: list[CameraSource],
    ) -> None:
        """collect() retorna frames después de start()."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, frame)
        mock_vc.return_value = mock_instance

        collector = FrameCollector(mock_sources)
        collector.start()

        # Pequeña pausa para que el thread capture
        import time
        time.sleep(0.05)

        result = collector.collect()
        assert len(result) > 0
        for cam_id in ["cam01", "cam02"]:
            assert cam_id in result

        collector.release()

    @patch("cv2.VideoCapture")
    def test_release_stops_threads(
        self,
        mock_vc: MagicMock,
        mock_sources: list[CameraSource],
    ) -> None:
        """release() detiene todos los threads."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_vc.return_value = mock_instance

        collector = FrameCollector(mock_sources)
        collector.start()
        collector.release()

        # Threads deben estar detenidos
        for thread in collector._threads.values():
            assert not thread.is_alive()

    @patch("cv2.VideoCapture")
    def test_start_idempotent(
        self,
        mock_vc: MagicMock,
        mock_sources: list[CameraSource],
    ) -> None:
        """start() llamado múltiples veces es seguro."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_vc.return_value = mock_instance

        collector = FrameCollector(mock_sources)
        collector.start()
        collector.start()  # Segundo start no debe crear más threads
        assert len(collector._threads) == 2
        collector.release()

    @patch("cv2.VideoCapture")
    def test_capture_loop_handles_read_failure(
        self,
        mock_vc: MagicMock,
        mock_sources: list[CameraSource],
    ) -> None:
        """Si cap.read() falla, el loop no crashea."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.side_effect = RuntimeError("read failed")
        mock_vc.return_value = mock_instance

        collector = FrameCollector(mock_sources)
        collector.start()
        import time
        time.sleep(0.05)

        # No debe crashear — collect simplemente vacío
        result = collector.collect()
        assert isinstance(result, dict)
        collector.release()
