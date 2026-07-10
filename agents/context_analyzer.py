from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Resultado del análisis de riesgo contextual.

    Attributes:
        nivel_riesgo_contextual: Nivel de riesgo contextual evaluado.
        factores_agravantes: Factores que agravan el riesgo.
        factores_mitigantes: Factores que mitigan el riesgo.
        requiere_accion_inmediata: Si se requiere acción inmediata.
        recomendacion: Recomendación de acción.
        confianza_analisis: Confianza del análisis (0.0 - 1.0).
    """

    nivel_riesgo_contextual: Literal["BAJO", "MEDIO", "ALTO", "CRITICO"] = Field(
        description="Nivel de riesgo contextual evaluado",
    )
    factores_agravantes: list[str] = Field(
        default_factory=list,
        description="Factores que agravan el riesgo",
    )
    factores_mitigantes: list[str] = Field(
        default_factory=list,
        description="Factores que mitigan el riesgo",
    )
    requiere_accion_inmediata: bool = Field(
        description="Si se requiere acción inmediata",
    )
    recomendacion: str = Field(
        description="Texto breve con recomendación de acción",
    )
    confianza_analisis: float = Field(
        ge=0.0,
        le=1.0,
        description="Confianza del análisis",
    )


PROMPT_CONTEXT_ANALYZER = """
Eres un analista de seguridad minera experto en DS 132 (Reglamento de Seguridad Minera de Chile).
Analiza el contexto de la siguiente violación de EPP detectada en una faena minera.

Contexto extraído por el sistema de detección:
{violation_json}

Análisis del extractor visual:
{extractor_output}

Evalúa:

1. NIVEL DE RIESGO CONTEXTUAL: Basado en:
   - Zona donde ocurre (extracción > procesamiento > mantención en riesgo)
   - Tipo de EPP faltante (arnés > casco > chaleco > lentes > guantes > botas)
   - Condiciones de la escena (iluminación, maquinaria operando)
   - Historial de la persona (reincidente o primera vez)

2. FACTORES AGRAVANTES:
   - Trabajo en altura sin arnés
   - Operación de maquinaria pesada sin EPP
   - Múltiples EPP faltantes simultáneamente
   - Mala iluminación que aumenta el riesgo

3. FACTORES MITIGANTES:
   - Persona en zona de descanso
   - Persona en tránsito (caminando entre zonas)
   - EPP parcialmente presente (ej: casco sí, chaleco no)

Responde SOLO con un JSON que cumpla este schema:
{{
  "nivel_riesgo_contextual": "BAJO | MEDIO | ALTO | CRITICO",
  "factores_agravantes": ["lista", "de", "factores"],
  "factores_mitigantes": ["lista", "de", "factores"],
  "requiere_accion_inmediata": true/false,
  "recomendacion": "Texto breve con recomendación de acción.",
  "confianza_analisis": 0.0-1.0
}}
"""


def create_context_analyzer_agent(
    model: OpenAIChatModel,
    max_tokens: int = 2048,
) -> Agent[None, AnalysisResult]:
    """Crea el agente analizador de contexto con PydanticAI.

    ADR-022: Usa PromptedOutput (JSON mode) para compatibilidad con vLLM.
    vLLM no soporta tool calling. PromptedOutput inyecta el schema
    en el prompt y usa response_format json_object.

    Args:
        model: Modelo VLM configurado (OpenAIChatModel con vLLM o Fireworks).
        max_tokens: Máximo de tokens en respuesta VLM (default 2048).

    Returns:
        Agent configurado con output_type=PromptedOutput(AnalysisResult).
    """
    return Agent(
        model,
        output_type=PromptedOutput(AnalysisResult),
        retries={"output": 3},
        model_settings=ModelSettings(temperature=0.1, max_tokens=max_tokens),
        system_prompt=PROMPT_CONTEXT_ANALYZER,
    )


__all__ = [
    "AnalysisResult",
    "create_context_analyzer_agent",
    "PROMPT_CONTEXT_ANALYZER",
]
