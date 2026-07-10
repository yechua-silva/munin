from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from munin.gate.schemas import DetectionResult, TrackedPerson, Violation

logger = logging.getLogger(__name__)


class SupervisionStandardizer:
    """Bridge entre tipos Munin y supervision sv.Detections.

    Convierte list[DetectionResult] + list[TrackedPerson] opcional
    a sv.Detections para uso en dashboard y anotación visual.

    No toca IDetector ni detectores concretos (zero coupling).

    ADR-028: Bridge desde DetectionResult (Opción B).
    """

    CLASS_NAME_TO_ID: ClassVar[dict[str, int]] = {
        "person": 0, "hardhat": 1, "safety_vest": 2, "gloves": 3,
        "safety_glasses": 4, "safety_boots": 5, "harness": 6, "mask": 7,
        "no_hardhat": 8, "no_safety_vest": 9, "no_gloves": 10,
        "no_safety_boots": 11, "no_safety_glasses": 12,
    }
    ID_TO_CLASS_NAME: ClassVar[dict[int, str]] = {
        v: k for k, v in CLASS_NAME_TO_ID.items()
    }

    @staticmethod
    def from_detection_results(
        detections: list[DetectionResult],
        persons: list[TrackedPerson] | None = None,
    ) -> object:
        """Construye sv.Detections desde list[DetectionResult].

        Args:
            detections: Detecciones de Munin.
            persons: Personas trackeadas opcional (para tracker_id).

        Returns:
            sv.Detections con xyxy, confidence, class_id, y data.
        """
        import supervision as sv

        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([
            SupervisionStandardizer.CLASS_NAME_TO_ID.get(d.class_name, 0)
            for d in detections
        ], dtype=int)
        class_names = np.array([d.class_name for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
            data={"class_name": class_names},
        )

        if persons:
            tracker_ids = []
            for det in detections:
                best_id = -1
                best_iou = 0.0
                for person in persons:
                    iou = SupervisionStandardizer._compute_iou(det.bbox, person.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = person.persona_id
                tracker_ids.append(best_id if best_iou > 0.3 else -1)
            sv_detections.tracker_id = np.array(tracker_ids)

        return sv_detections

    @staticmethod
    def from_violations(
        violations: list[Violation],
        persons: list[TrackedPerson],
    ) -> object:
        """Construye sv.Detections desde violaciones (para highlight en dashboard).

        Args:
            violations: Violaciones detectadas.
            persons: Personas trackeadas con bboxes.

        Returns:
            sv.Detections con las personas que tienen violaciones.
        """
        import supervision as sv

        violated_ids = {v.persona_id for v in violations}
        violated_persons = [p for p in persons if p.persona_id in violated_ids]

        if not violated_persons:
            return sv.Detections.empty()

        xyxy = np.array([p.bbox for p in violated_persons], dtype=np.float32)
        tracker_ids = np.array([p.persona_id for p in violated_persons])

        return sv.Detections(
            xyxy=xyxy,
            tracker_id=tracker_ids,
            data={"violation": [True] * len(violated_persons)},
        )

    @staticmethod
    def _compute_iou(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        """Calcula IoU entre dos bboxes.

        Args:
            bbox_a: Primer bounding box (x1, y1, x2, y2).
            bbox_b: Segundo bounding box (x1, y1, x2, y2).

        Returns:
            IoU como float entre 0.0 y 1.0.
        """
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0


__all__ = ["SupervisionStandardizer"]
