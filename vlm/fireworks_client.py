from __future__ import annotations

# DEPRECATED: This module is superseded by PydanticAI FireworksProvider.
# Kept for reference. New code should use VLMModelFactory from munin.vlm.factory.

import asyncio
import base64
import logging

import cv2 as cv
import numpy as np

from munin.exceptions import VLMError, VLMTimeoutError
from munin.vlm.client import VLMClient

logger = logging.getLogger(__name__)


class FireworksVLMClient:
    """Cliente VLM que adapta el OpenAI SDK a Fireworks AI (Adapter Pattern).

    Codifica frames como JPEG base64 y los envía a la API de Fireworks AI
    usando el formato de mensajes multimodal de OpenAI. Las respuestas se
    exigen en formato JSON para facilitar el parseo posterior.

    Attributes:
        _client: Cliente OpenAI configurado para Fireworks endpoint.
        _model: Identificador del modelo VLM en Fireworks.
        _timeout: Timeout en segundos para cada llamada.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "accounts/fireworks/models/internvl3-8b",
        timeout: float = 30.0,
    ) -> None:
        """Inicializa FireworksVLMClient.

        Crea una instancia del cliente OpenAI apuntando al endpoint
        de inferencia de Fireworks AI.

        Args:
            api_key: API key de Fireworks AI.
            model: Identificador del modelo VLM en Fireworks.
            timeout: Timeout máximo por llamada en segundos.
        """
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model
        self._timeout = timeout
        self._logger = logging.getLogger(self.__class__.__name__)

    async def analyze(self, frame: np.ndarray, prompt: str) -> str:
        """Analiza un frame usando Fireworks AI.

        Codifica el frame como JPEG base64 y lo envía al modelo VLM
        junto con el prompt. La llamada se ejecuta en un hilo separado
        para no bloquear el event loop.

        Args:
            frame: Imagen np.ndarray (HWC, BGR, uint8).
            prompt: Instrucciones para el VLM.

        Returns:
            Respuesta del VLM como string JSON.

        Raises:
            VLMError: Si la API retorna error o content es None.
            VLMTimeoutError: Si la llamada excede el timeout.
        """
        # Codificar frame a JPEG base64
        success, buffer = cv.imencode(
            ".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            raise VLMError("Failed to encode frame as JPEG")

        b64_str = base64.b64encode(buffer).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=1024,
                    temperature=0.1,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            raise VLMTimeoutError(
                f"VLM call exceeded timeout of {self._timeout}s"
            )
        except Exception as e:
            raise VLMError(f"Fireworks API call failed: {e}") from e

        content = response.choices[0].message.content
        if content is None:
            raise VLMError("VLM returned empty response")

        self._logger.debug("VLM response received (%d chars)", len(content))
        return content


__all__ = ["FireworksVLMClient"]
