"""ByteTrackAdapter — Adapter mínimo sobre model.track() de ultralytics.

Traduce Results de ultralytics → list[TrackedPerson].
Tracker PURO: solo asigna IDs y retorna bboxes. NO asigna EPP.
El PPEComplianceChecker asigna EPP desde detections.

ADR-017: usa model.track(persist=True) nativo de ultralytics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from munin.exceptions import ConfigurationError, TrackingError
from munin.gate.schemas import TrackedPerson

logger = logging.getLogger(__name__)


class ByteTrackAdapter:
    """Adapter mínimo sobre model.track() de ultralytics con ByteTrack.

    Delega en model.track(persist=True, tracker='bytetrack.yaml') de
    ultralytics. No implementa Kalman ni IoU manualmente — ultralytics
    maneja ByteTrack internamente.

    SRP: Solo traduce model.track() output → list[TrackedPerson].
    epp_detectado siempre es set() vacío — el checker asigna EPP.

    Attributes:
        _model: Modelo YOLO cargado para tracking.
        _confidence: Threshold de confianza (0.0 - 1.0).
        _device: Dispositivo de inferencia ("cuda:0", "cpu").
        _imgsz: Tamaño de entrada para YOLO.
        _tracker: Config de tracker ("bytetrack.yaml" | "botsort.yaml").
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.6,
        device: str = "cpu",
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        """Inicializa ByteTrackAdapter.

        Carga modelo YOLO y configura ByteTrack.

        Args:
            model_path: Ruta al modelo YOLO (.pt) para detección de personas.
            confidence: Threshold de confianza (0.0 - 1.0).
            device: Dispositivo ("cuda:0", "cpu").
            imgsz: Tamaño de entrada para YOLO (default: 640).
            tracker: Config de tracker (default: "bytetrack.yaml").

        Raises:
            ConfigurationError: Si model_path no existe.
            DetectionError: Si el modelo no puede cargarse.
        """
        from ultralytics import YOLO

        if not Path(model_path).exists():
            raise ConfigurationError(
                f"Model not found: {model_path}"
            )

        self._confidence = confidence
        self._device = device
        self._imgsz = imgsz
        self._tracker = tracker
        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            self._model = YOLO(model_path)
            self._logger.info(
                "ByteTrackAdapter: model loaded from %s", model_path
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load YOLO model: {e}"
            ) from e

    def update(self, frame: np.ndarray) -> list[TrackedPerson]:
        """Actualiza tracking con un frame.

        Llama model.track(persist=True) y traduce Results → TrackedPerson[].
        epp_detectado siempre es set() vacío — el checker asigna EPP
        desde detections.

        Args:
            frame: Imagen np.ndarray HWC BGR uint8.

        Returns:
            Lista de TrackedPerson con IDs asignados y epp_detectado=set()
            vacío. Vacía si no hay personas detectadas.

        Raises:
            TrackingError: Si model.track() falla.
        """
        try:
            results = self._model.track(
                frame,
                persist=True,
                tracker=self._tracker,
                conf=self._confidence,
                device=self._device,
                imgsz=self._imgsz,
                verbose=False,
            )
        except Exception as e:
            raise TrackingError(
                f"model.track() failed: {e}"
            ) from e

        # Sin detecciones
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        persons: list[TrackedPerson] = []
        boxes = results[0].boxes

        for i in range(len(boxes)):
            track_id = int(boxes.id[i])
            conf = float(boxes.conf[i])

            if conf < self._confidence:
                continue

            bbox = tuple(boxes.xyxy[i].tolist())

            persons.append(TrackedPerson(
                persona_id=track_id,
                bbox=bbox,
                epp_detectado=set(),  # Siempre vacío — checker asigna
            ))

        self._logger.debug(
            "ByteTrackAdapter: %d persons tracked", len(persons)
        )

        return persons


__all__ = ["ByteTrackAdapter"]
