"""Tests para SupervisionStandardizer (ADR-028).

Verifica el bridge entre tipos Munin y supervision sv.Detections.

Correr con: pytest munin/tests/test_supervision_standardizer.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from munin.gate.schemas import DetectionResult, TrackedPerson, Violation
from munin.gate.schemas import PPEMissing


class TestSupervisionStandardizer:
    """Tests para SupervisionStandardizer."""

    def test_from_detection_results_empty(self) -> None:
        """Lista vacía → sv.Detections.empty()."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        result = SupervisionStandardizer.from_detection_results([])
        assert len(result.xyxy) == 0
        assert len(result.confidence) == 0
        assert len(result.class_id) == 0

    def test_from_detection_results_with_detections(self) -> None:
        """Detecciones → xyxy, confidence, class_id correctos."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        detections = [
            DetectionResult(class_name="person", bbox=(10.0, 20.0, 30.0, 40.0), confidence=0.9),
            DetectionResult(class_name="hardhat", bbox=(50.0, 60.0, 70.0, 80.0), confidence=0.85),
        ]

        result = SupervisionStandardizer.from_detection_results(detections)

        assert len(result.xyxy) == 2
        assert np.array_equal(result.xyxy[0], [10.0, 20.0, 30.0, 40.0])
        assert np.array_equal(result.xyxy[1], [50.0, 60.0, 70.0, 80.0])
        assert np.allclose(result.confidence, [0.9, 0.85])
        assert np.array_equal(result.class_id, [0, 1])  # person=0, hardhat=1
        assert list(result.data["class_name"]) == ["person", "hardhat"]

    def test_from_detection_results_with_persons(self) -> None:
        """Persons → tracker_id asignado por IoU."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        detections = [
            DetectionResult(class_name="person", bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.9),
        ]
        persons = [
            TrackedPerson(
                persona_id=42,
                bbox=(10.0, 20.0, 100.0, 200.0),
                epp_detectado={"hardhat"},
            ),
        ]

        result = SupervisionStandardizer.from_detection_results(detections, persons)

        assert result.tracker_id is not None
        assert result.tracker_id[0] == 42

    def test_from_detection_results_with_persons_low_iou(self) -> None:
        """IoU bajo → tracker_id = -1."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        detections = [
            DetectionResult(class_name="person", bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.9),
        ]
        persons = [
            TrackedPerson(
                persona_id=42,
                bbox=(500.0, 500.0, 600.0, 600.0),  # lejano
                epp_detectado={"hardhat"},
            ),
        ]

        result = SupervisionStandardizer.from_detection_results(detections, persons)

        assert result.tracker_id is not None
        assert result.tracker_id[0] == -1

    def test_from_violations_with_violations(self) -> None:
        """Violaciones → solo personas violadas."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        violations = [
            Violation(
                persona_id=1,
                zona_id="extraccion",
                epp_faltantes=[PPEMissing(tipo="hardhat", descripcion="Casco", norma_chilena="NCh 1411")],
                frame_id=0,
            ),
        ]
        persons = [
            TrackedPerson(persona_id=1, bbox=(10.0, 20.0, 100.0, 200.0), epp_detectado=set()),
            TrackedPerson(persona_id=2, bbox=(200.0, 300.0, 300.0, 400.0), epp_detectado={"hardhat"}),
        ]

        result = SupervisionStandardizer.from_violations(violations, persons)

        assert len(result.xyxy) == 1
        assert np.array_equal(result.xyxy[0], [10.0, 20.0, 100.0, 200.0])
        assert result.tracker_id is not None
        assert result.tracker_id[0] == 1
        assert list(result.data["violation"]) == [True]

    def test_from_violations_empty(self) -> None:
        """Sin violaciones → sv.Detections.empty()."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        result = SupervisionStandardizer.from_violations([], [])
        assert len(result.xyxy) == 0

    def test_class_name_to_id_has_13_entries(self) -> None:
        """CLASS_NAME_TO_ID tiene 13 entries."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        assert len(SupervisionStandardizer.CLASS_NAME_TO_ID) == 13

    def test_id_to_class_name_is_inverse(self) -> None:
        """ID_TO_CLASS_NAME es el inverso de CLASS_NAME_TO_ID."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        for name, cid in SupervisionStandardizer.CLASS_NAME_TO_ID.items():
            assert SupervisionStandardizer.ID_TO_CLASS_NAME[cid] == name

    def test_compute_iou_no_overlap(self) -> None:
        """Sin solapamiento → IoU = 0.0."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        iou = SupervisionStandardizer._compute_iou(
            (0.0, 0.0, 10.0, 10.0),
            (20.0, 20.0, 30.0, 30.0),
        )
        assert iou == 0.0

    def test_compute_iou_perfect_overlap(self) -> None:
        """Mismo bbox → IoU = 1.0."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        iou = SupervisionStandardizer._compute_iou(
            (0.0, 0.0, 10.0, 10.0),
            (0.0, 0.0, 10.0, 10.0),
        )
        assert iou == 1.0

    def test_compute_iou_partial(self) -> None:
        """Solapamiento parcial → IoU > 0.0 y < 1.0."""
        from munin.pipeline.supervision_standardizer import SupervisionStandardizer

        iou = SupervisionStandardizer._compute_iou(
            (0.0, 0.0, 10.0, 10.0),
            (5.0, 5.0, 15.0, 15.0),
        )
        assert 0.0 < iou < 1.0
