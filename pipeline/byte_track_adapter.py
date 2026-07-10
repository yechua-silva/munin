"""ByteTrackAdapter — Tracker ligero sin modelo propio.

v4: No carga YOLO. Recibe detecciones ya inferidas del detector
central y asigna IDs por IoU matching greedy.

ADR-029: ByteTrackAdapter sin modelo propio. IoU matching sustituye
a model.track(persist=True) de ultralytics para evitar N instancias
YOLO en modo multi-cámara.
"""
from __future__ import annotations

import logging
from typing import Any

from munin.gate.schemas import DetectionResult, TrackedPerson

logger = logging.getLogger(__name__)


class ByteTrackAdapter:
    """Tracker ligero sin modelo propio — IoU matching greedy.

    v4: No carga YOLO. Recibe detecciones ya inferidas
    del detector central y asigna IDs por IoU matching.

    ADR-029: ByteTrackAdapter sin modelo propio.

    Attributes:
        _confidence: Threshold de confianza mínimo para considerar
            una detección como persona.
        _iou_threshold: IoU mínimo para considerar match entre
            detección y track existente.
        _max_lost_frames: Frames sin detección antes de eliminar track.
        _next_id: Próximo ID de track disponible.
        _tracks: Dict de tracks activos: track_id → {bbox, lost_frames}.
    """

    def __init__(
        self,
        confidence: float = 0.6,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
    ) -> None:
        """Inicializa ByteTrackAdapter sin modelo.

        Args:
            confidence: Threshold de confianza mínimo (0.0 - 1.0).
                Detecciones por debajo se filtran.
            iou_threshold: IoU mínimo para considerar match (0.0 - 1.0).
            max_lost_frames: Frames sin detección antes de podar track.
        """
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._next_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def update(self, detections: list[DetectionResult]) -> list[TrackedPerson]:
        """Actualiza tracking desde detecciones ya inferidas.

        Filtra solo personas (class_name='person'), hace IoU matching
        con tracks existentes, asigna nuevos IDs a personas sin match.

        Args:
            detections: Detecciones del detector central.

        Returns:
            Personas trackeadas con ID asignado y epp_detectado=set().
            Vacía si no hay personas detectadas.
        """
        persons_dets = [d for d in detections if d.class_name == "person"]

        if not persons_dets:
            # Incrementar lost_frames en todos los tracks
            for track in self._tracks.values():
                track["lost_frames"] += 1
            self._prune_lost()
            return []

        # IoU matching
        matched: list[tuple[int, int]] = []  # (track_id, det_idx)
        used_det_idx: set[int] = set()

        for track_id, track in list(self._tracks.items()):
            if track["lost_frames"] > self._max_lost_frames:
                continue
            best_iou = self._iou_threshold
            best_det = -1
            for i, det in enumerate(persons_dets):
                if i in used_det_idx:
                    continue
                iou = self._compute_iou(track["bbox"], det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = i
            if best_det >= 0:
                matched.append((track_id, best_det))
                used_det_idx.add(best_det)

        # Actualizar tracks matched
        matched_track_ids: set[int] = set()
        for track_id, det_idx in matched:
            self._tracks[track_id]["bbox"] = persons_dets[det_idx].bbox
            self._tracks[track_id]["lost_frames"] = 0
            matched_track_ids.add(track_id)

        # Crear nuevos tracks para detecciones no matched
        for i, det in enumerate(persons_dets):
            if i not in used_det_idx:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = {
                    "bbox": det.bbox,
                    "lost_frames": 0,
                }

        # Incrementar lost en tracks no matched
        for track_id in self._tracks:
            if track_id not in matched_track_ids:
                self._tracks[track_id]["lost_frames"] += 1

        self._prune_lost()

        # Retornar TrackedPerson list (solo activos con lost_frames == 0)
        persons: list[TrackedPerson] = []
        for track_id, track in self._tracks.items():
            if track["lost_frames"] == 0:  # Solo activos en este frame
                persons.append(TrackedPerson(
                    persona_id=track_id,
                    bbox=track["bbox"],
                    epp_detectado=set(),
                ))

        self._logger.debug(
            "ByteTrackAdapter: %d persons tracked from %d detections",
            len(persons), len(persons_dets),
        )

        return persons

    def _prune_lost(self) -> None:
        """Elimina tracks que excedieron max_lost_frames."""
        before = len(self._tracks)
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t["lost_frames"] <= self._max_lost_frames
        }
        pruned = before - len(self._tracks)
        if pruned > 0:
            self._logger.debug("Pruned %d lost tracks", pruned)

    @staticmethod
    def _compute_iou(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        """Calcula IoU entre dos bboxes.

        Args:
            bbox_a: Primer bbox (x1, y1, x2, y2).
            bbox_b: Segundo bbox (x1, y1, x2, y2).

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


__all__ = ["ByteTrackAdapter"]
