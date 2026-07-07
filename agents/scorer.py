from __future__ import annotations

import logging

from pydantic_ai import Agent, BinaryContent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel

from munin.gate.schemas import AgentDecision

logger = logging.getLogger(__name__)

PROMPT_SCORER = """
Eres un experto en normativa DS 132 (Chile) especializado en clasificación de infracciones de EPP.

Violación detectada:
{violation_json}

Contexto extraído:
{extractor_output}

Análisis contextual:
{analysis_output}

Basado en el DS 132 y los artículos aplicables, determina:

1. TIPO DE VIOLACIÓN:
   - "EPP_FALTANTE": Falta uno o más elementos de protección personal
   - "ZONA_NO_AUTORIZADA": Persona en zona para la que no tiene autorización
   - "SIN_VIOLACION": No hay violación (falso positivo del sistema de detección)

2. ARTÍCULO DS 132 APLICABLE:
   - Artículo 38: Uso obligatorio de EPP en faenas mineras
   - Artículo 42: Protección contra caídas (arnés en altura)
   - Artículo 45: EPP en zonas de extracción y procesamiento
   - Artículo 50: Protección ocular y respiratoria

3. NIVEL DE RIESGO FINAL (considerando contexto y normativa):
   - CRITICO: Vida en peligro inminente (altura sin arnés, zona de voladura)
   - ALTO: Riesgo serio de lesión grave
   - MEDIO: Riesgo moderado de lesión
   - BAJO: Riesgo menor o formalidad

4. CONFIANZA: ¿Qué tan seguro estás de esta evaluación?
   - 0.0-0.3: Baja (información insuficiente)
   - 0.3-0.7: Media (evidencia parcial)
   - 0.7-1.0: Alta (evidencia clara y completa)

Responde SOLO con un JSON que cumpla este schema:
{{
  "tipo_violacion": "EPP_FALTANTE | ZONA_NO_AUTORIZADA | SIN_VIOLACION",
  "articulo_ds132": "Art. 38 | Art. 42 | Art. 45 | Art. 50 | null",
  "nivel_riesgo": "CRITICO | ALTO | MEDIO | BAJO",
  "confianza": 0.0-1.0,
  "requiere_revision_humana": true/false,
  "razonamiento_vlm": "Explicación breve del razonamiento."
}}
"""


def create_scorer_agent(model: OpenAIChatModel) -> Agent[None, AgentDecision]:
    """Crea el agente scorer con PydanticAI.

    Args:
        model: Modelo VLM configurado (OpenAIChatModel con FireworksProvider).

    Returns:
        Agent configurado con output_type=AgentDecision.
    """
    return Agent(
        model,
        output_type=AgentDecision,
        retries={"output": 3},
        model_settings=ModelSettings(temperature=0.1, max_tokens=8192),
        system_prompt=PROMPT_SCORER,
    )


__all__ = ["create_scorer_agent", "PROMPT_SCORER"]
