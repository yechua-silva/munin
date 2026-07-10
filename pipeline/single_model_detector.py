"""SingleModelDetector v2 — Detector para Munin v4 fine-tuned (13 clases).

Un solo modelo YOLO produce personas + EPP positivo + clases negativas.
No requiere modelo COCO separado. Implementa IDetector (Protocol).

Schema v4 (13 clases):
  0: person, 1: hardhat, 2: safety_vest, 3: gloves, 4: safety_glasses,
  5: safety_boots, 6: harness, 7: mask,
  8: no_hardhat, 9: no_safety_vest, 10: no_gloves,
  11: no_safety_boots, 12: no_safety_glasses

ADR-018: IDetector abstraction para soportar LEGACY (TwoModelDetector)
y DUAL_CLASS (SingleModelDetector).

v2 (TRACK-A): Schema expandido de 11→13 clases. Se añaden harness y mask.
Clase "none" eliminada (no se ignora ninguna clase).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import numpy as np

from munin.exceptions import ConfigurationError, DetectionError
from munin.gate.schemas import DetectionResult

logger = logging.getLogger(__name__)

# Mapeo Munin v4 — 13 clases schema unificado
# 0:person, 1:hardhat, 2:safety_vest, 3:gloves, 4:safety_glasses,
# 5:safety_boots, 6:harness, 7:mask,
# 8:no_hardhat, 9:no_safety_vest, 10:no_gloves, 11:no_safety_boots,
# 12:no_safety_glasses
CONSTRUCTION_PPE_CLASS_MAP: dict[int, str] = {
    0: "person",
    1: "hardhat",
    2: "safety_vest",
    3: "gloves",
    4: "safety_glasses",
    5: "safety_boots",
    6: "harness",
    7: "mask",
    8: "no_hardhat",
    9: "no_safety_vest",
    10: "no_gloves",
    11: "no_safety_boots",
    12: "no_safety_glasses",
}

# Clases a ignorar (no producen DetectionResult)
# v4: todas las 13 clases producen detecciones válidas
_IGNORED_CLASSES: set[int] = set()


class SingleModelDetector:
    """Detector usando modelo único Munin v4 fine-tuned (13 clases).

    Un solo modelo produce personas, EPP positivo, y clases negativas
    (no_hardhat, no_safety_vest, etc.). No requiere modelo COCO separado.

    Schema v4 cubre 13 clases: person, hardhat, safety_vest, gloves,
    safety_glasses, safety_boots, harness, mask, y 5 clases negativas
    (no_*). Ninguna clase se ignora en v4.

    Attributes:
        _model: Modelo YOLO Munin v4 cargado.
        _confidence: Threshold de confianza (0.0 - 1.0).
        _device: Dispositivo de inferencia ("cuda:0", "cpu").
        _imgsz: Tamaño de entrada para YOLO.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.6,
        device: str = "cpu",
        imgsz: int = 640,
    ) -> None:
        """Inicializa SingleModelDetector v2 (13 clases).

        Args:
            model_path: Ruta a Munin v4 best.pt.
            confidence: Threshold de confianza.
            device: Dispositivo de inferencia.
            imgsz: Tamaño de entrada YOLO.

        Raises:
            ConfigurationError: Si model_path no existe.
            DetectionError: Si el modelo no puede cargarse.
        """
        from ultralytics import YOLO

        if not Path(model_path).exists():
            raise ConfigurationError(
                f"Munin v4 model not found: {model_path}"
            )

        self._confidence = confidence
        self._device = device
        self._imgsz = imgsz
        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            self._model = YOLO(model_path)
            self._logger.info(
                "SingleModelDetector v2: model loaded from %s", model_path
            )
        except Exception as e:
            raise DetectionError(
                f"Failed to load Munin v4 model: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detecta personas, EPP y clases negativas en un frame.

        Args:
            frame: Imagen np.ndarray HWC BGR uint8.

        Returns:
            Lista de DetectionResult con las 13 clases del schema v4:
            person, hardhat, safety_vest, gloves, safety_glasses,
            safety_boots, harness, mask y 5 clases negativas (no_*).
            Ordenado por confianza descendente.

        Raises:
            DetectionError: Si la inferencia falla.
        """
        try:
            results = self._model.predict(
                frame,
                conf=self._confidence,
                device=self._device,
                imgsz=self._imgsz,
                verbose=False,
            )
        except Exception as e:
            raise DetectionError(f"Inference failed: {e}") from e

        return self._parse_results(results[0] if results else None)

    def detect_stream(
        self, video_path: str
    ) -> Generator[list[DetectionResult], None, None]:
        """Detecta en modo streaming (YOLO gestiona frames internamente).

        Usa stream=True para no cargar todo el video en memoria.

        Args:
            video_path: Ruta al video MP4.

        Yields:
            Lista de DetectionResult por cada frame.

        Raises:
            DetectionError: Si la inferencia falla.
        """
        try:
            stream = self._model.predict(
                video_path,
                stream=True,
                conf=self._confidence,
                device=self._device,
                imgsz=self._imgsz,
                verbose=False,
            )
        except Exception as e:
            raise DetectionError(f"Stream inference failed: {e}") from e

        for result in stream:
            yield self._parse_results(result)

    def _parse_results(self, result: object | None) -> list[DetectionResult]:
        """Parsea Results de ultralytics → list[DetectionResult] (schema v4).

        Convierte class_id numérico a nombre de clase usando
        CONSTRUCTION_PPE_CLASS_MAP. Clases desconocidas generan warning.
        En v4 no se ignora ninguna clase.

        Args:
            result: Results object de ultralytics (o None).

        Returns:
            Lista de DetectionResult filtrada y mapeada (13 clases).
        """
        if result is None or result.boxes is None:
            return []

        detections: list[DetectionResult] = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < self._confidence:
                continue

            if class_id in _IGNORED_CLASSES:
                continue

            class_name = CONSTRUCTION_PPE_CLASS_MAP.get(class_id)
            if class_name is None:
                self._logger.warning(
                    "Unknown class_id %d in Munin v4 model, skipping",
                    class_id,
                )
                continue

            bbox = tuple(box.xyxy[0].tolist())
            detections.append(DetectionResult(
                class_name=class_name,
                bbox=bbox,
                confidence=conf,
            ))

        # Ordenar por confianza descendente
        detections.sort(key=lambda d: d.confidence, reverse=True)

        if detections:
            self._logger.debug(
                "SingleModelDetector v2: %d detections: %s",
                len(detections),
                [d.class_name for d in detections],
            )

        return detections


__all__ = ["SingleModelDetector", "CONSTRUCTION_PPE_CLASS_MAP"]
