"""Tests T16 — ByteTrackAdapter refactor (sin modelo propio).

Verifica:
- update() recibe list[DetectionResult], no np.ndarray
- No se pasa model_path en __init__
- IoU matching con detecciones sintéticas
- Nuevo ID para persona sin match
- Lost track pruning
- Filtrado por class_name="person"

Correr con: pytest tests/test_byte_track_adapter.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from munin.gate.schemas import DetectionResult, TrackedPerson


@pytest.fixture
def adapter() -> ByteTrackAdapter:
    """ByteTrackAdapter con configuración por defecto."""
    from munin.pipeline.byte_track_adapter import ByteTrackAdapter
    return ByteTrackAdapter(
        confidence=0.6,
        iou_threshold=0.3,
        max_lost_frames=30,
    )


@pytest.fixture
def person_detections() -> list[DetectionResult]:
    """Detecciones sintéticas de 2 personas."""
    return [
        DetectionResult(
            class_name="person",
            bbox=(0.0, 0.0, 100.0, 200.0),
            confidence=0.85,
        ),
        DetectionResult(
            class_name="person",
            bbox=(150.0, 50.0, 300.0, 400.0),
            confidence=0.72,
        ),
    ]


class TestByteTrackAdapter:
    """Suite TDD para ByteTrackAdapter v4."""

    def test_init_no_model_path(self, adapter: ByteTrackAdapter) -> None:
        """__init__ ya no requiere model_path."""
        assert adapter is not None

    def test_update_returns_tracked_persons(
        self,
        adapter: ByteTrackAdapter,
        person_detections: list[DetectionResult],
    ) -> None:
        """update() retorna TrackedPerson con IDs."""
        persons = adapter.update(person_detections)
        assert len(persons) == 2
        assert all(isinstance(p, TrackedPerson) for p in persons)
        assert persons[0].persona_id == 0
        assert persons[1].persona_id == 1

    def test_update_filters_non_persons(self, adapter: ByteTrackAdapter) -> None:
        """Detecciones que no son 'person' se filtran."""
        detections = [
            DetectionResult(
                class_name="hardhat", bbox=(0, 0, 10, 10), confidence=0.9,
            ),
            DetectionResult(
                class_name="person", bbox=(0, 0, 100, 200), confidence=0.8,
            ),
        ]
        persons = adapter.update(detections)
        assert len(persons) == 1
        assert persons[0].persona_id == 0

    def test_update_empty_detections(
        self, adapter: ByteTrackAdapter,
    ) -> None:
        """Sin detecciones de persona → retorna []."""
        detections = [
            DetectionResult(
                class_name="hardhat", bbox=(0, 0, 10, 10), confidence=0.9,
            ),
        ]
        persons = adapter.update(detections)
        assert persons == []

    def test_iou_matching_same_person(
        self,
        adapter: ByteTrackAdapter,
        person_detections: list[DetectionResult],
    ) -> None:
        """Misma persona con bbox similar recibe mismo ID."""
        persons1 = adapter.update(person_detections)
        id0_first = persons1[0].persona_id

        # Mismas detecciones (mismos bboxes)
        persons2 = adapter.update(person_detections)
        id0_second = persons2[0].persona_id

        assert id0_first == id0_second

    def test_new_id_for_unmatched_person(
        self,
        adapter: ByteTrackAdapter,
        person_detections: list[DetectionResult],
    ) -> None:
        """Persona sin match IoU recibe nuevo ID."""
        persons1 = adapter.update(person_detections)

        # Nueva persona lejos de las anteriores
        new_detections = [
            DetectionResult(
                class_name="person",
                bbox=(500.0, 500.0, 600.0, 700.0),
                confidence=0.9,
            ),
        ]
        persons2 = adapter.update(new_detections)
        assert len(persons2) == 1
        assert persons2[0].persona_id == 2  # IDs: 0, 1, 2

    def test_lost_track_pruning(self, adapter: ByteTrackAdapter) -> None:
        """Track perdido por N frames se elimina."""
        det = [
            DetectionResult(
                class_name="person",
                bbox=(0, 0, 100, 200),
                confidence=0.9,
            ),
        ]
        persons1 = adapter.update(det)
        track_id = persons1[0].persona_id

        # Perder el track por max_lost_frames+1 frames
        for _ in range(31):
            adapter.update([])

        # Debe estar vacío (el track fue podado)
        persons2 = adapter.update(det)
        assert len(persons2) == 1
        assert persons2[0].persona_id != track_id  # Nuevo ID

    def test_epp_detectado_always_empty(
        self,
        adapter: ByteTrackAdapter,
        person_detections: list[DetectionResult],
    ) -> None:
        """epp_detectado debe ser set() vacío siempre."""
        persons = adapter.update(person_detections)
        assert all(p.epp_detectado == set() for p in persons)

    def test_confidence_not_filtered_by_tracker(self) -> None:
        """El tracker NO filtra por confianza (recibe detecciones pre-filtradas).

        v4: ByteTrackAdapter recibe detecciones ya filtradas por el detector.
        El tracker solo filtra por class_name='person' y hace IoU matching.
        La confianza se maneja en el detector, no en el tracker.
        """
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter
        high_conf_tracker = ByteTrackAdapter(confidence=0.9)
        detections = [
            DetectionResult(
                class_name="person",
                bbox=(0, 0, 100, 200),
                confidence=0.85,  # < 0.9, pero el tracker no filtra por esto
            ),
        ]
        persons = high_conf_tracker.update(detections)
        # El tracker NO filtra por confianza — retorna la persona igual
        assert len(persons) == 1

    def test_iou_zero_for_non_overlapping(
        self, adapter: ByteTrackAdapter,
    ) -> None:
        """IoU = 0 para bboxes que no se intersectan."""
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter as BTA

        a = (0.0, 0.0, 10.0, 10.0)
        b = (100.0, 100.0, 200.0, 200.0)
        iou = BTA._compute_iou(a, b)
        assert iou == 0.0

    def test_iou_perfect_overlap(
        self, adapter: ByteTrackAdapter,
    ) -> None:
        """IoU = 1.0 para bboxes idénticos."""
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter as BTA

        a = (0.0, 0.0, 100.0, 200.0)
        iou = BTA._compute_iou(a, a)
        assert iou == 1.0

    def test_iou_partial_overlap(
        self, adapter: ByteTrackAdapter,
    ) -> None:
        """IoU parcial calculado correctamente."""
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter as BTA

        a = (0.0, 0.0, 100.0, 100.0)
        b = (50.0, 0.0, 150.0, 100.0)  # 50% overlap
        iou = BTA._compute_iou(a, b)
        # intersection = 50*100 = 5000
        # area_a = 10000, area_b = 10000
        # union = 20000 - 5000 = 15000
        # iou = 5000 / 15000 = 0.333...
        assert iou == pytest.approx(0.3333, abs=0.01)
