from __future__ import annotations

from typing import Protocol

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
    """Interfaz para detección de objetos en frames.

    Implementaciones deben detectar personas y EPP usando
    un modelo de detección de objetos.
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


class ITracker(Protocol):
    """Interfaz para tracking de personas entre frames.

    Implementaciones deben asignar IDs únicos a personas
    y asociar EPP detectado a cada persona.
    """

    def update(
        self, detections: list[DetectionResult]
    ) -> list[TrackedPerson]:
        """Actualiza tracking con nuevas detecciones.

        Args:
            detections: Detecciones del frame actual.

        Returns:
            Personas trackeadas con ID asignado y EPP asociado.
        """
        ...


class IComplianceChecker(Protocol):
    """Interfaz para verificación de compliance EPP.

    Implementaciones deben verificar que cada persona tenga
    el EPP requerido según la zona configurada.
    """

    def check(
        self, persons: list[TrackedPerson], zone: Zone
    ) -> list[Violation]:
        """Verifica compliance de EPP por zona.

        Args:
            persons: Personas trackeadas en el frame.
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
