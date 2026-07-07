from __future__ import annotations

import json
import logging
from pathlib import Path

from munin.config import Zone
from munin.exceptions import KnowledgeBaseError

logger = logging.getLogger(__name__)


class ZoneConfig:
    """Configuración de zonas mineras y EPP requerido.

    Carga la configuración de zonas desde un archivo JSON y
    provee acceso a cada zona por su ID.
    """

    def __init__(self, zones: list[Zone]) -> None:
        """Inicializa ZoneConfig con una lista de zonas.

        Args:
            zones: Lista de objetos Zone con configuración de cada zona.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._zones: dict[str, Zone] = {z.zone_id: z for z in zones}
        self._logger.info(
            "ZoneConfig initialized with %d zones", len(self._zones)
        )

    def get_zone(self, zona_id: str) -> Zone:
        """Obtiene la configuración de una zona por su ID.

        Args:
            zona_id: ID de la zona (extraccion, procesamiento, mantencion).

        Returns:
            Objeto Zone con la configuración de la zona.

        Raises:
            KnowledgeBaseError: Si la zona no existe.
        """
        zone = self._zones.get(zona_id)
        if zone is None:
            raise KnowledgeBaseError(f"Zone not found: {zona_id}")
        return zone

    @classmethod
    def from_json(cls, path: str) -> ZoneConfig:
        """Carga configuración de zonas desde un archivo JSON.

        Args:
            path: Ruta al archivo zones.json.

        Returns:
            Instancia de ZoneConfig con las zonas cargadas.

        Raises:
            KnowledgeBaseError: Si el archivo no existe o el JSON es inválido.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise KnowledgeBaseError(f"Zones config file not found: {path}")

        try:
            with open(path_obj, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise KnowledgeBaseError(
                f"Failed to load zones config from {path}: {e}"
            ) from e

        zonas_raw = data.get("zonas", [])
        zones = [Zone(**z) for z in zonas_raw]

        logger.info("Loaded %d zones from %s", len(zones), path)
        return cls(zones)


__all__ = ["ZoneConfig"]
