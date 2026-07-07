from __future__ import annotations

# DEPRECATED: This module is superseded by PydanticAI Agent + FireworksProvider.
# Kept for reference. New code should use VLMModelFactory from munin.vlm.factory.

from typing import Protocol

import numpy as np

from munin.exceptions import VLMError, VLMTimeoutError


class VLMClient(Protocol):
    """Interfaz para clientes VLM (Strategy Pattern).

    Define el contrato que deben implementar todos los backends de
    inferencia VLM. Cada backend se implementa como un Adapter
    que traduce esta interfaz al SDK específico del proveedor.

    Attributes:
        analyze: Método asíncrono para analizar un frame con un prompt.
    """

    async def analyze(self, frame: np.ndarray, prompt: str) -> str:
        """Analiza un frame con un prompt textual.

        Codifica el frame como imagen y lo envía al VLM junto con
        instrucciones textuales. El VLM retorna una respuesta JSON
        con el análisis solicitado.

        Args:
            frame: Imagen en formato np.ndarray (HWC, BGR, uint8).
            prompt: Instrucciones para el VLM (qué debe analizar).

        Returns:
            Respuesta del VLM como string JSON.

        Raises:
            VLMError: Si la inferencia falla o la API retorna error.
            VLMTimeoutError: Si la llamada excede el timeout configurado.
        """


__all__ = ["VLMClient"]
