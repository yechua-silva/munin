from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from munin.exceptions import TrackingError
from munin.gate.schemas import DetectionResult, TrackedPerson

if TYPE_CHECKING:
    from munin.pipeline.interfaces import IDetector

logger = logging.getLogger(__name__)


def _compute_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Calcula el IoU (Intersection over Union) entre dos bounding boxes.

    Los bboxes están en formato (x1, y1, x2, y2) con coordenadas
    absolutas en píxeles.

    Args:
        bbox_a: Primer bounding box (x1, y1, x2, y2).
        bbox_b: Segundo bounding box (x1, y1, x2, y2).

    Returns:
        IoU como float entre 0.0 (sin solapamiento) y 1.0 (idénticos).

    Raises:
        TrackingError: Si las coordenadas son inválidas o el cálculo falla.
    """
    try:
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b

        # Validar coordenadas
        if x1_a >= x2_a or y1_a >= y2_a:
            raise TrackingError(
                f"Invalid bbox_a: ({x1_a}, {y1_a}, {x2_a}, {y2_a})"
            )
        if x1_b >= x2_b or y1_b >= y2_b:
            raise TrackingError(
                f"Invalid bbox_b: ({x1_b}, {y1_b}, {x2_b}, {y2_b})"
            )

        # Calcular intersección
        x_intersection = max(0.0, min(x2_a, x2_b) - max(x1_a, x1_b))
        y_intersection = max(0.0, min(y2_a, y2_b) - max(y1_a, y1_b))
        intersection_area = x_intersection * y_intersection

        # Calcular áreas individuales
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)

        # Calcular unión
        union_area = area_a + area_b - intersection_area

        if union_area <= 0.0:
            return 0.0

        iou = intersection_area / union_area
        return iou

    except (TypeError, ValueError, ZeroDivisionError) as e:
        raise TrackingError(f"Failed to compute IoU: {e}") from e


class PersonTracker:
    """Tracker de personas con IoU matching entre frames.

    Asigna IDs únicos a personas detectadas usando IoU para matchear
    entre frames consecutivos. Asocia EPP detectado a cada persona
    basado en proximidad de bounding boxes.

    Attributes:
        _iou_threshold: IoU mínimo para considerar mismo ID (default: 0.3).
        _max_lost: Frames máximos sin detección antes de descartar (default: 5).
        _next_id: Siguiente ID disponible para nuevas personas.
        _tracks: Diccionario de tracks activas (persona_id → TrackedPerson).
        _logger: Logger de la clase.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 5,
        detector: "IDetector | None" = None,
    ) -> None:
        """Inicializa PersonTracker.

        Args:
            iou_threshold: IoU mínimo para considerar mismo ID (default: 0.3).
            max_lost: Frames máximos sin detección antes de descartar
                una track (default: 5).
            detector: Detector opcional para soportar ITracker v3
                (update(frame: np.ndarray)). Si es None, usa
                update(detections: list) legacy.
        """
        self._iou_threshold: float = iou_threshold
        self._max_lost: int = max_lost
        self._detector: "IDetector | None" = detector
        self._next_id: int = 0
        self._tracks: dict[int, TrackedPerson] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def update(
        self, frame: "np.ndarray | list[DetectionResult]"
    ) -> list[TrackedPerson]:
        """Actualiza el tracking con un frame o lista de detecciones.

        v3: Si recibe np.ndarray (frame), usa el detector interno para
        obtener detecciones. Si recibe list[DetectionResult], mantiene
        comportamiento legacy (compatibilidad hacia atrás).

        Args:
            frame: Frame np.ndarray HWC BGR uint8, o lista de DetectionResult
                   (legacy compat).

        Returns:
            Lista de TrackedPerson activas.

        Raises:
            TrackingError: Si hay error en el cálculo de IoU.
        """
        if isinstance(frame, np.ndarray) and self._detector is not None:
            detections = self._detector.detect(frame)
        elif isinstance(frame, list):
            # Legacy: frame es en realidad list[DetectionResult]
            detections = frame
        else:
            self._logger.warning(
                "PersonTracker.update: no detector and frame is not list, "
                "returning empty tracks"
            )
            return []

        return self._update_from_detections(detections)

    def _update_from_detections(
        self, detections: list[DetectionResult]
    ) -> list[TrackedPerson]:
        """Lógica de tracking desde detecciones (migrado de update v2).

        Paso a paso:
        1. Separa detecciones en persons (person) y ppe_items (resto).
        2. Para cada track existente: busca mejor match por IoU con nuevas
           detecciones de personas.
        3. Si match > threshold: actualiza bbox, resetea lost_counter,
           asigna EPP detectado.
        4. Si no match: lost_counter += 1.
        5. Personas sin match: crea nuevo track con next_id++.
        6. Elimina tracks con lost_counter > max_lost.
        7. Asigna EPP detectado a cada persona según proximidad de bbox.

        Args:
            detections: Lista de DetectionResult del frame actual.

        Returns:
            Lista de TrackedPerson activas (lost_counter <= max_lost).

        Raises:
            TrackingError: Si hay error en el cálculo de IoU.
        """
        # 1. Separar detecciones
        persons_detected: list[DetectionResult] = []
        ppe_items: list[DetectionResult] = []

        for det in detections:
            if det.class_name == "person":
                persons_detected.append(det)
            else:
                ppe_items.append(det)

        self._logger.debug(
            "Frame: %d persons detected, %d PPE items",
            len(persons_detected),
            len(ppe_items),
        )

        # 2-4. Matchear tracks existentes con nuevas personas
        matched_track_ids: set[int] = set()
        used_person_indices: set[int] = set()

        for track_id, track in list(self._tracks.items()):
            best_iou = 0.0
            best_idx = -1

            for idx, person_det in enumerate(persons_detected):
                if idx in used_person_indices:
                    continue

                try:
                    iou = _compute_iou(track.bbox, person_det.bbox)
                except TrackingError:
                    self._logger.warning(
                        "IoU computation failed for track %d, skipping", track_id
                    )
                    continue

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou > self._iou_threshold and best_idx >= 0:
                # Match encontrado: actualizar track
                person_det = persons_detected[best_idx]
                track.bbox = person_det.bbox
                track.lost_counter = 0
                matched_track_ids.add(track_id)
                used_person_indices.add(best_idx)

                self._logger.debug(
                    "Track %d matched with IoU=%.3f", track_id, best_iou
                )
            else:
                # No match: incrementar lost_counter
                track.lost_counter += 1
                self._logger.debug(
                    "Track %d lost (lost_counter=%d/%d)",
                    track_id,
                    track.lost_counter,
                    self._max_lost,
                )

        # 5. Nuevas personas sin match: crear nuevas tracks
        for idx, person_det in enumerate(persons_detected):
            if idx in used_person_indices:
                continue

            new_id = self._next_id
            self._next_id += 1

            new_track = TrackedPerson(
                persona_id=new_id,
                bbox=person_det.bbox,
                epp_detectado=set(),
                lost_counter=0,
                consecutive_violations=0,
            )
            self._tracks[new_id] = new_track
            matched_track_ids.add(new_id)

            self._logger.debug(
                "New track %d created (IoU match not found)", new_id
            )

        # 6. Eliminar tracks con lost_counter > max_lost
        tracks_to_remove = [
            tid
            for tid, track in self._tracks.items()
            if track.lost_counter > self._max_lost
        ]
        for tid in tracks_to_remove:
            del self._tracks[tid]
            self._logger.debug("Track %d removed (lost_counter exceeded)", tid)

        # 7. Asignar EPP a cada persona activa
        # Un ítem de EPP se asigna si su bbox está dentro o se solapa
        # significativamente con el bbox de la persona
        for track in self._tracks.values():
            if track.lost_counter > self._max_lost:
                continue

            assigned_epp: set[str] = set()
            for ppe in ppe_items:
                if self._is_ppe_inside_person(ppe.bbox, track.bbox):
                    assigned_epp.add(ppe.class_name)

            # Mantener EPP previamente asignado + nuevas detecciones
            track.epp_detectado = assigned_epp

        # 8. Retornar tracks activas
        active_tracks = [
            t for t in self._tracks.values()
            if t.lost_counter <= self._max_lost
        ]

        self._logger.debug(
            "Tracker update: %d active tracks, %d total tracks",
            len(active_tracks),
            len(self._tracks),
        )

        return active_tracks

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _is_ppe_inside_person(
        ppe_bbox: tuple[float, float, float, float],
        person_bbox: tuple[float, float, float, float],
    ) -> bool:
        """Determina si un ítem de EPP está dentro o contiguo a una persona.

        Se considera que un EPP pertenece a una persona si el centroide
        del EPP está dentro del bbox de la persona, o si el IoU entre
        ambos bounding boxes es > 0.

        Args:
            ppe_bbox: Bounding box del EPP (x1, y1, x2, y2).
            person_bbox: Bounding box de la persona (x1, y1, x2, y2).

        Returns:
            True si el EPP pertenece a la persona.
        """
        # Calcular centroide del EPP
        ppe_cx = (ppe_bbox[0] + ppe_bbox[2]) / 2.0
        ppe_cy = (ppe_bbox[1] + ppe_bbox[3]) / 2.0

        # Verificar si el centroide está dentro del bbox de la persona
        inside = (
            person_bbox[0] <= ppe_cx <= person_bbox[2]
            and person_bbox[1] <= ppe_cy <= person_bbox[3]
        )

        if not inside:
            # Fallback: verificar si hay solapamiento (IoU > 0)
            try:
                iou = _compute_iou(ppe_bbox, person_bbox)
                inside = iou > 0.0
            except TrackingError:
                inside = False

        return inside


__all__ = ["PersonTracker"]
