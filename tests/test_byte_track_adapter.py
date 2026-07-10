"""Tests TDD para ByteTrackAdapter.

Mock de model.track() usando unittest.mock. NO requiere GPU ni modelo real.

Correr con: pytest munin/tests/test_byte_track_adapter.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from munin.exceptions import ConfigurationError, TrackingError
from munin.gate.schemas import TrackedPerson


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_yolo_model() -> MagicMock:
    """Mock de YOLO con track() configurable."""
    return MagicMock()


@pytest.fixture
def mock_track_results() -> MagicMock:
    """Mock de un objeto Results de ultralytics con 2 personas detectadas.

    El adaptador accede como results[0].boxes (results es lista real).
    """
    results = MagicMock()
    boxes = MagicMock()

    # __len__ en boxes: len(boxes) == 2
    boxes.__len__.return_value = 2

    # boxes.id[i] → [1, 2][i]
    boxes.id = MagicMock()
    boxes.id.__getitem__.side_effect = [1, 2]

    # boxes.conf[i] → [0.85, 0.72][i]
    boxes.conf = MagicMock()
    boxes.conf.__getitem__.side_effect = [0.85, 0.72]

    # boxes.xyxy[i] → arrays (con .tolist() para tuple conversion)
    boxes.xyxy = MagicMock()
    boxes.xyxy.__getitem__.side_effect = [
        np.array([0, 0, 100, 200], dtype=np.float32),
        np.array([150, 50, 300, 400], dtype=np.float32),
    ]

    # Atributo directo .boxes en el objeto Results
    results.boxes = boxes

    return results


@pytest.fixture
def adapter(mock_yolo_model: MagicMock) -> "ByteTrackAdapter":
    """ByteTrackAdapter con modelo mockeado.

    Parchea ultralytics.YOLO y Path.exists para evitar modelo real.
    """
    with patch("ultralytics.YOLO", return_value=mock_yolo_model), \
         patch("pathlib.Path.exists", return_value=True):
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter
        return ByteTrackAdapter(
            model_path="/fake/model.pt",
            confidence=0.6,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestByteTrackAdapter:
    """Suite TDD para ByteTrackAdapter."""

    def test_update_returns_tracked_persons(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
        mock_track_results: MagicMock,
    ) -> None:
        """update() debe retornar list[TrackedPerson] con IDs correctos."""
        mock_yolo_model.track.return_value = [mock_track_results]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        persons = adapter.update(frame)

        assert len(persons) == 2
        assert all(isinstance(p, TrackedPerson) for p in persons)
        assert persons[0].persona_id == 1
        assert persons[1].persona_id == 2

    def test_update_empty_frame_returns_empty(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
    ) -> None:
        """Sin detecciones (boxes is None) → retorna []."""
        empty_results = MagicMock()
        empty_results.boxes = None
        mock_yolo_model.track.return_value = [empty_results]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        persons = adapter.update(frame)

        assert persons == []

    def test_update_no_tracking_ids_returns_empty(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
    ) -> None:
        """boxes existe pero boxes.id is None → retorna []."""
        results = MagicMock()
        boxes = MagicMock()
        boxes.id = None
        results.boxes = boxes
        mock_yolo_model.track.return_value = [results]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        persons = adapter.update(frame)

        assert persons == []

    def test_update_epp_detectado_always_empty(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
        mock_track_results: MagicMock,
    ) -> None:
        """epp_detectado debe ser siempre set() vacío."""
        mock_yolo_model.track.return_value = [mock_track_results]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        persons = adapter.update(frame)

        assert all(p.epp_detectado == set() for p in persons)

    def test_update_filters_by_confidence(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
    ) -> None:
        """Detecciones con conf < threshold deben filtrarse."""
        results = MagicMock()
        boxes = MagicMock()

        boxes.__len__.return_value = 1

        boxes.id = MagicMock()
        boxes.id.__getitem__.side_effect = [1]

        boxes.xyxy = MagicMock()
        boxes.xyxy.__getitem__.side_effect = [
            np.array([0, 0, 100, 200], dtype=np.float32),
        ]

        boxes.conf = MagicMock()
        boxes.conf.__getitem__.side_effect = [0.3]  # menor que 0.6

        results.boxes = boxes
        mock_yolo_model.track.return_value = [results]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        persons = adapter.update(frame)

        assert len(persons) == 0

    def test_update_assigns_correct_ids(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
        mock_track_results: MagicMock,
    ) -> None:
        """track_id de boxes.id debe mapearse a persona_id."""
        mock_yolo_model.track.return_value = [mock_track_results]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        persons = adapter.update(frame)

        assert persons[0].persona_id == 1
        assert persons[1].persona_id == 2

    def test_configuration_error_if_model_not_found(self) -> None:
        """Model path inexistente → ConfigurationError."""
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter

        with pytest.raises(ConfigurationError):
            ByteTrackAdapter(
                model_path="/nonexistent/model.pt",
                confidence=0.6,
                device="cpu",
            )

    def test_tracking_error_if_inference_fails(
        self,
        adapter: "ByteTrackAdapter",
        mock_yolo_model: MagicMock,
    ) -> None:
        """model.track() falla → TrackingError."""
        mock_yolo_model.track.side_effect = RuntimeError("GPU OOM")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(TrackingError):
            adapter.update(frame)
