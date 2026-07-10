"""SingleModelDetector — Detector para Construction-PPE fine-tuned (11 clases).

Un solo modelo YOLO produce personas + EPP positivo + clases negativas.
No requiere modelo COCO separado. Implementa IDetector (Protocol).

ADR-018: IDetector abstraction para soportar LEGACY (TwoModelDetector)
y DUAL_CLASS (SingleModelDetector).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import numpy as np

from munin.exceptions import ConfigurationError, DetectionError
from munin.gate.schemas import DetectionResult

logger = logging.getLogger(__name__)

# Mapeo Construction-PPE (11 clases duales)
# 0:helmet, 1:vest, 2:gloves, 3:glasses, 4:goggles, 5:none,
# 6:Person, 7:no_helmet, 8:no_vest, 9:no_gloves, 10:no_boots
CONSTRUCTION_PPE_CLASS_MAP: dict[int, str] = {
    0: "hardhat",
    1: "safety_vest",
    2: "gloves",
    3: "safety_glasses",
    4: "safety_glasses",  # goggles → safety_glasses (merge)
    # 5: "none" se ignora (clase genérica sin valor de EPP)
    6: "person",
    7: "no_helmet",
    8: "no_vest",
    9: "no_gloves",
    10: "no_boots",
}

# Clases a ignorar (no producen DetectionResult)
_IGNORED_CLASSES: set[int] = {5}  # "none"


class SingleModelDetector:
    """Detector usando modelo único Construction-PPE fine-tuned (11 clases).

    Un solo modelo produce personas, EPP positivo, y clases negativas
    (no_helmet, no_vest, etc.). No requiere modelo COCO separado.

    Attributes:
        _model: Modelo YOLO Construction-PPE cargado.
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
        """Inicializa SingleModelDetector.

        Args:
            model_path: Ruta a Construction-PPE best.pt.
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
                f"Construction-PPE model not found: {model_path}"
            )

        self._confidence = confidence
        self._device = device
        self._imgsz = imgsz
        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            self._model = YOLO(model_path)
            self._logger.info(
                "SingleModelDetector: model loaded from %s", model_path
            )
        except Exception as e:
            raise DetectionError(
                f"Failed to load Construction-PPE model: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detecta personas, EPP y clases negativas en un frame.

        Args:
            frame: Imagen np.ndarray HWC BGR uint8.

        Returns:
            Lista de DetectionResult. Puede incluir class_name="person",
            clases EPP positivas, y clases negativas (no_*).
            Clase 5 ("none") se ignora. Goggles (clase 4) mapea a
            safety_glasses. Ordenado por confianza descendente.

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
        """Parsea Results de ultralytics → list[DetectionResult].

        Args:
            result: Results object de ultralytics (o None).

        Returns:
            Lista de DetectionResult filtrada y mapeada.
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
                    "Unknown class_id %d in Construction-PPE model, skipping",
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
                "SingleModelDetector: %d detections: %s",
                len(detections),
                [d.class_name for d in detections],
            )

        return detections


__all__ = ["SingleModelDetector", "CONSTRUCTION_PPE_CLASS_MAP"]
