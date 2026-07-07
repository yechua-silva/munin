from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from munin.exceptions import ConfigurationError, DetectionError
from munin.gate.schemas import DetectionResult

logger = logging.getLogger(__name__)

# Mapeo de clases del modelo PPE (yolov8n_ppe_best.pt) → nuestros schema names
PPE_CLASS_MAP: dict[int, str] = {
    0: "gloves",          # Gloves → gloves
    1: "safety_vest",     # Vest → safety_vest
    2: "safety_glasses",  # goggles → safety_glasses
    3: "hardhat",         # helmet → hardhat
    4: "mask",            # mask (extra, no está en schema principal)
    5: "safety_boots",    # safety_shoe → safety_boots
}

# Mapeo de clases COCO → solo nos interesa "person" (id=0)
COCO_PERSON_ID: int = 0


class YOLODetector:
    """Detector de objetos usando doble modelo YOLOv8 en cascada.

    Usa dos modelos:
    1. YOLOv8n COCO base — detecta personas (class id 0)
    2. YOLOv8n PPE fine-tuned — detecta EPP (gloves, vest, helmet, etc.)

    Los resultados se mergean en una sola lista de DetectionResult.

    Attributes:
        _coco_model: Modelo YOLOv8n COCO para detección de personas.
        _ppe_model: Modelo YOLOv8n PPE para detección de EPP.
        _confidence: Threshold de confianza (0.0 - 1.0).
        _device: Dispositivo de inferencia ("cuda:0" o "cpu").
    """

    def __init__(
        self,
        coco_model_path: str,
        ppe_model_path: str,
        confidence: float = 0.6,
        device: str = "cpu",
    ) -> None:
        """Inicializa YOLODetector con doble modelo.

        Args:
            coco_model_path: Ruta al modelo YOLOv8n COCO (.pt).
            ppe_model_path: Ruta al modelo YOLOv8n PPE (.pt).
            confidence: Threshold de confianza (0.0 - 1.0).
            device: Dispositivo ("cuda:0", "cpu").

        Raises:
            ConfigurationError: Si los archivos no existen.
            DetectionError: Si los modelos no pueden cargarse.
        """
        from ultralytics import YOLO

        # Validar path PPE (el COCO puede descargarse automáticamente)
        if not Path(ppe_model_path).exists():
            raise ConfigurationError(
                f"PPE model not found: {ppe_model_path}"
            )

        self._confidence = confidence
        self._device = device
        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            self._coco_model = YOLO(coco_model_path)
            self._logger.info("COCO model loaded: %s", coco_model_path)
        except Exception as e:
            raise DetectionError(f"Failed to load COCO model: {e}") from e

        try:
            self._ppe_model = YOLO(ppe_model_path)
            self._logger.info("PPE model loaded: %s", ppe_model_path)
        except Exception as e:
            raise DetectionError(f"Failed to load PPE model: {e}") from e

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detecta personas y EPP en un frame usando doble modelo.

        Ejecuta COCO (personas) y PPE (EPP) en secuencia, merguea
        los resultados en una sola lista de DetectionResult.

        Args:
            frame: Imagen np.ndarray HWC BGR uint8.

        Returns:
            Lista de DetectionResult con personas y EPP detectados.
            Filtrado por confidence threshold.

        Raises:
            DetectionError: Si la inferencia falla.
        """
        results: list[DetectionResult] = []

        try:
            # 1. COCO — detectar personas
            coco_results = self._coco_model.predict(
                frame, conf=self._confidence, device=self._device,
                verbose=False, classes=[COCO_PERSON_ID],
            )
            if coco_results and coco_results[0].boxes is not None:
                for box in coco_results[0].boxes:
                    conf = float(box.conf[0])
                    if conf < self._confidence:
                        continue
                    bbox = tuple(box.xyxy[0].tolist())
                    results.append(DetectionResult(
                        class_name="person",
                        bbox=bbox,
                        confidence=conf,
                    ))

            # 2. PPE — detectar EPP
            ppe_results = self._ppe_model.predict(
                frame, conf=self._confidence, device=self._device,
                verbose=False,
            )
            if ppe_results and ppe_results[0].boxes is not None:
                for box in ppe_results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < self._confidence:
                        continue
                    class_name = PPE_CLASS_MAP.get(class_id)
                    if class_name:
                        bbox = tuple(box.xyxy[0].tolist())
                        results.append(DetectionResult(
                            class_name=class_name,
                            bbox=bbox,
                            confidence=conf,
                        ))

        except Exception as e:
            raise DetectionError(f"Inference failed: {e}") from e

        # Ordenar por confianza descendente
        results.sort(key=lambda d: d.confidence, reverse=True)

        if results:
            self._logger.debug(
                "Detected %d objects: %s",
                len(results),
                [d.class_name for d in results],
            )

        return results


__all__ = ["YOLODetector", "PPE_CLASS_MAP"]
