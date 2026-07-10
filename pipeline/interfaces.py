"""Interfaces (Protocol) para el pipeline de Munin v3.

Define los contratos que deben implementar los componentes del pipeline.
Usa Protocol con @runtime_checkable para duck typing.
"""
from __future__ import annotations

from typing import Generator, Protocol

import numpy as np

from munin.config import Zone
from munin.gate.schemas import DetectionResult, TrackedPerson, Violation


class IFrameExtractor(Protocol):
    """Interfaz para extracción de frames de video.

    Implementaciones deben extraer frames de un video MP4
    a un framerate configurable.
    """

    def extract(self, video_path: str) -> list[np.ndarray]:
        """Extrae frames de un video.

        Args:
            video_path: Ruta al archivo MP4.

        Returns:
            Lista de frames (np.ndarray HWC BGR uint8).

        Raises:
            VideoLoadError: Si el video no puede ser abierto.
        """
        ...


class IDetector(Protocol):
    """Interfaz para detección de objetos en frames (v3).

    Implementaciones deben detectar personas y EPP usando
    un modelo de detección de objetos. v3 añade detect_stream()
    para modo streaming con YOLO stream=True.
    """

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detecta objetos en un frame.

        Args:
            frame: Imagen np.ndarray HWC BGR uint8.

        Returns:
            Lista de detecciones con bbox, clase, confianza.

        Raises:
            DetectionError: Si la inferencia falla.
        """
        ...

    def detect_stream(
        self, video_path: str
    ) -> Generator[list[DetectionResult], None, None]:
        """Detecta objetos en modo streaming (YOLO gestiona frames).

        Args:
            video_path: Ruta al archivo MP4.

        Yields:
            Lista de DetectionResult por cada frame.

        Raises:
            DetectionError: Si la inferencia falla.
        """
        ...


class ITracker(Protocol):
    """Interfaz para tracking de personas entre frames (v3).

    v3: update(frame) recibe el frame completo, no las detecciones.
    El tracker internamente ejecuta model.track() para asignar IDs.
    NO asigna EPP — eso lo hace PPEComplianceChecker desde detections.
    """

    def update(self, frame: np.ndarray) -> list[TrackedPerson]:
        """Actualiza tracking con un nuevo frame.

        Args:
            frame: Imagen np.ndarray HWC BGR uint8.

        Returns:
            Personas trackeadas con ID asignado. epp_detectado=set()
            (siempre vacío — el checker asigna EPP desde detections).
        """
        ...


class IComplianceChecker(Protocol):
    """Interfaz para verificación de compliance EPP (v3).

    v3: check() recibe detections como tercer parámetro.
    El checker asigna EPP a personas desde detections (migrado de
    PersonTracker._is_ppe_inside_person).
    """

    def check(
        self,
        persons: list[TrackedPerson],
        detections: list[DetectionResult],
        zone: Zone,
    ) -> list[Violation]:
        """Verifica compliance de EPP por zona.

        Args:
            persons: Personas trackeadas en el frame.
            detections: Detecciones crudas del frame (personas + EPP).
            zone: Zona con requisitos de EPP.

        Returns:
            Lista de violaciones detectadas. Vacía si no hay.
        """
        ...


__all__ = [
    "IFrameExtractor",
    "IDetector",
    "ITracker",
    "IComplianceChecker",
]
