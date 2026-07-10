from __future__ import annotations

import logging

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import ModelSettings

from munin.config import AppSettings, VLMBackend
from munin.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

SHARED_BASE_SYSTEM_PROMPT: str = (
    "Eres Munin, un asistente de seguridad industrial especializado en "
    "normativa chilena DS 132. Tu rol es analizar imágenes de trabajadores "
    "en faenas mineras y detectar incumplimientos de EPP (Elementos de "
    "Protección Personal).\n\n"
    "Debes responder ESTRICTAMENTE en el formato JSON solicitado. "
    "No agregues texto adicional fuera del JSON. "
    "Sé preciso y objetivo en tus evaluaciones."
)


class VLMModelFactory:
    """Factory para crear el modelo VLM según configuración.

    Usa PydanticAI con OpenAIProvider (para AMD vLLM on-premise) o
    FireworksProvider (cloud interim). Strategy Pattern: el backend
    se selecciona via AppSettings.vlm_backend.

    ADR-021: Default backend cambiado a AMD (vLLM on-premise MI300X).
    Fireworks permanece como fallback cloud.

    ADR-015: SHARED_BASE_SYSTEM_PROMPT se usa como prefijo en los prompts
    de cada agente. En Fireworks maximiza cache hits; en vLLM mejora
    consistencia de respuestas.

    Attributes:
        _settings: Configuración de la aplicación.
    """

    @staticmethod
    def create(settings: AppSettings) -> OpenAIChatModel:
        """Crea el modelo VLM apropiado con session affinity.

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
            from openai import AsyncOpenAI

            logger.info(
                "Creating Fireworks VLM model: %s (session: %s)",
                settings.fireworks_model,
                settings.prompt_cache_session_id,
            )
            client = AsyncOpenAI(
                api_key=settings.fireworks_api_key,
                timeout=settings.vlm_busy_timeout,
                default_headers={
                    "x-session-affinity": settings.prompt_cache_session_id,
                },
            )
            return OpenAIChatModel(
                settings.fireworks_model,
                provider=FireworksProvider(openai_client=client),
                settings=ModelSettings(max_tokens=settings.vlm_max_tokens),
            )

        elif settings.vlm_backend == VLMBackend.AMD:
            if not settings.amd_vllm_endpoint:
                raise ConfigurationError(
                    "AMD vLLM endpoint no configurado. "
                    "Set MUNIN_AMD_VLLM_ENDPOINT in .env"
                )
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
                timeout=settings.vlm_busy_timeout,
            )
            return OpenAIChatModel(
                settings.amd_model,
                provider=OpenAIProvider(openai_client=client),
                settings=ModelSettings(max_tokens=settings.vlm_max_tokens),
            )

        raise ConfigurationError(
            f"VLM backend desconocido: {settings.vlm_backend}. "
            f"Supported: {list(VLMBackend)}"
        )


__all__ = ["VLMModelFactory", "SHARED_BASE_SYSTEM_PROMPT"]
