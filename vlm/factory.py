from __future__ import annotations

import logging
import os

from pydantic_ai.models.openai import OpenAIChatModel

from munin.config import AppSettings, VLMBackend
from munin.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class VLMModelFactory:
    """Factory para crear el modelo VLM según configuración.

    Usa PydanticAI con FireworksProvider o OpenAIProvider (para AMD vLLM).
    Strategy Pattern: el backend se selecciona via AppSettings.vlm_backend.

    Attributes:
        _settings: Configuración de la aplicación.
    """

    @staticmethod
    def create(settings: AppSettings) -> OpenAIChatModel:
        """Crea el modelo VLM apropiado.

        Args:
            settings: Configuración de la aplicación.

        Returns:
            OpenAIChatModel configurado con el provider correcto.

        Raises:
            ConfigurationError: Si el backend es desconocido o faltan credenciales.
        """
        if settings.vlm_backend == VLMBackend.FIREWORKS:
            if not settings.fireworks_api_key:
                raise ConfigurationError(
                    "FIREWORKS_API_KEY no configurada. "
                    "Set MUNIN_FIREWORKS_API_KEY in .env"
                )
            from pydantic_ai.providers.fireworks import FireworksProvider

            logger.info(
                "Creating Fireworks VLM model: %s",
                settings.fireworks_model,
            )
            return OpenAIChatModel(
                settings.fireworks_model,
                provider=FireworksProvider(
                    api_key=settings.fireworks_api_key,
                ),
            )

        elif settings.vlm_backend == VLMBackend.AMD:
            from pydantic_ai.providers.openai import OpenAIProvider
            from openai import AsyncOpenAI

            logger.info(
                "Creating AMD vLLM model: %s at %s",
                settings.amd_model,
                settings.amd_vllm_endpoint,
            )
            client = AsyncOpenAI(
                base_url=settings.amd_vllm_endpoint,
                api_key="dummy",
            )
            return OpenAIChatModel(
                settings.amd_model,
                provider=OpenAIProvider(openai_client=client),
            )

        raise ConfigurationError(
            f"VLM backend desconocido: {settings.vlm_backend}. "
            f"Supported: {list(VLMBackend)}"
        )


__all__ = ["VLMModelFactory"]
