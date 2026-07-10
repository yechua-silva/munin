from __future__ import annotations

import logging

from pydantic_ai import Agent, BinaryContent, ModelSettings, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from munin.gate.schemas import AgentDecision

logger = logging.getLogger(__name__)

PROMPT_SINGLE_PASS = """
Eres un asistente integral de seguridad minera especializado en DS 132 (Chile).
Actúa SIMULTÁNEAMENTE como 3 roles:

ROL 1 — EXTRACTOR VISUAL:
Extrae zona, personas y EPP observado de la imagen.

ROL 2 — ANALISTA DE CONTEXTO:
Evalúa factores de riesgo agravantes y mitigantes.

ROL 3 — SCORER DS 132:
Clasifica la infracción según normativa chilena.

Violación detectada por el sistema de detección:
{violation_json}

IMPORTANTE: Integra los 3 análisis en UNA sola respuesta JSON.
Si la información es insuficiente, usa confianza baja y requiere_revision_humana=true.
En caso de duda, prioriza la seguridad del trabajador.

Responde SOLO con un JSON que cumpla este schema:
{{
  "zona": "extraccion | procesamiento | mantencion | desconocida",
  "tipo_violacion": "EPP_FALTANTE | ZONA_NO_AUTORIZADA | SIN_VIOLACION",
  "epp_faltante": [
    {{
      "tipo": "hardhat | safety_vest | gloves | safety_glasses | safety_boots | harness",
      "descripcion": "Nombre en español del EPP faltante",
      "norma_chilena": "Norma chilena asociada, ej: NCh 1411"
    }}
  ],
  "nivel_riesgo": "CRITICO | ALTO | MEDIO | BAJO",
  "articulo_ds132": "Art. 38 | Art. 42 | Art. 45 | Art. 50 | null",
  "confianza": 0.0-1.0,
  "requiere_revision_humana": true/false,
  "razonamiento_vlm": "Explicación breve y en español del análisis integrado."
}}
"""


def create_single_pass_agent(model: OpenAIChatModel) -> Agent[None, AgentDecision]:
    """Crea el agente single-pass (fallback) con PydanticAI.

    ADR-022: Usa PromptedOutput (JSON mode) para compatibilidad con vLLM.
    vLLM no soporta tool calling. PromptedOutput inyecta el schema
    en el prompt y usa response_format json_object.

    Args:
        model: Modelo VLM configurado (OpenAIChatModel con vLLM o Fireworks).

    Returns:
        Agent configurado con output_type=PromptedOutput(AgentDecision).
    """
    return Agent(
        model,
        output_type=PromptedOutput(AgentDecision),
        retries={"output": 3},
        model_settings=ModelSettings(temperature=0.1, max_tokens=8192),
        system_prompt=PROMPT_SINGLE_PASS,
    )


__all__ = ["create_single_pass_agent", "PROMPT_SINGLE_PASS"]
