from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel

logger = logging.getLogger(__name__)


class EPPObservado(BaseModel):
    """EPP observado en una persona.

    Attributes:
        hardhat: Casco de seguridad.
        safety_vest: Chaleco reflectante.
        gloves: Guantes de seguridad.
        safety_glasses: Lentes de seguridad.
        safety_boots: Botas de seguridad.
        harness: Arnés de seguridad.
    """

    hardhat: bool = Field(description="¿Usa casco de seguridad?")
    safety_vest: bool = Field(description="¿Usa chaleco reflectante?")
    gloves: bool = Field(description="¿Usa guantes de seguridad?")
    safety_glasses: bool = Field(description="¿Usa lentes de seguridad?")
    safety_boots: bool = Field(description="¿Usa botas de seguridad?")
    harness: bool = Field(description="¿Usa arnés de seguridad?")


class PersonaExtraida(BaseModel):
    """Persona detectada con EPP observado.

    Attributes:
        id_persona: ID de la persona.
        epp_observado: EPP que se observa en la persona.
        bbox_aproximado: Bounding box aproximado.
    """

    id_persona: int = Field(description="ID de la persona")
    epp_observado: EPPObservado = Field(description="EPP observado")
    bbox_aproximado: list[float] = Field(
        default_factory=list,
        description="Bounding box aproximado [x1, y1, x2, y2]",
    )


class ExtractionResult(BaseModel):
    """Resultado de extracción de entidades del frame.

    Attributes:
        zona: Zona minera identificada.
        personas: Lista de personas con EPP visible.
        escena: Descripción breve de la escena.
        iluminacion: Condición de iluminación.
        maquinaria_visible: Maquinaria visible en la escena.
    """

    zona: Literal["extraccion", "procesamiento", "mantencion", "desconocida"] = Field(
        description="Zona minera detectada",
    )
    personas: list[PersonaExtraida] = Field(
        default_factory=list,
        description="Personas con EPP visible",
    )
    escena: str = Field(description="Descripción breve de la escena")
    iluminacion: Literal["buena", "regular", "mala"] = Field(
        description="Condición de iluminación",
    )
    maquinaria_visible: list[str] = Field(
        default_factory=list,
        description="Maquinaria visible en la escena",
    )


PROMPT_EXTRACTOR = """
Eres un asistente de visión computacional especializado en faenas mineras chilenas.
Analiza la imagen adjunta y extrae la siguiente información:

1. ZONA: ¿En qué zona de la faena se encuentra la persona?
   - "extraccion" (zona de extracción/minería)
   - "procesamiento" (zona de procesamiento/planta)
   - "mantencion" (zona de mantención/taller)
   - "desconocida"

2. PERSONAS: Para cada persona visible, describe:
   a) ¿Está usando casco? (hardhat: sí/no)
   b) ¿Está usando chaleco reflectante? (safety_vest: sí/no)
   c) ¿Está usando guantes? (gloves: sí/no)
   d) ¿Está usando lentes de seguridad? (safety_glasses: sí/no)
   e) ¿Está usando botas de seguridad? (safety_boots: sí/no)
   f) ¿Está usando arnés de seguridad? (harness: sí/no)

3. ESCENA: Describe brevemente la escena (iluminación, condiciones, maquinaria visible)

Contexto de la violación detectada por el sistema de detección:
{violation_json}

Responde SOLO con un JSON que cumpla este schema:
{{
  "zona": "extraccion | procesamiento | mantencion | desconocida",
  "personas": [
    {{
      "id_persona": 1,
      "epp_observado": {{
        "hardhat": true/false,
        "safety_vest": true/false,
        "gloves": true/false,
        "safety_glasses": true/false,
        "safety_boots": true/false,
        "harness": true/false
      }},
      "bbox_aproximado": [x1, y1, x2, y2]
    }}
  ],
  "escena": "Descripción breve de la escena.",
  "iluminacion": "buena | regular | mala",
  "maquinaria_visible": ["lista", "de", "maquinaria"]
}}
"""


def create_extractor_agent(model: OpenAIChatModel) -> Agent[None, ExtractionResult]:
    """Crea el agente extractor con PydanticAI.

    Args:
        model: Modelo VLM configurado (OpenAIChatModel con FireworksProvider).

    Returns:
        Agent configurado con output_type=ExtractionResult.
    """
    return Agent(
        model,
        output_type=ExtractionResult,
        retries={"output": 3},
        model_settings=ModelSettings(temperature=0.1, max_tokens=8192),
        system_prompt=PROMPT_EXTRACTOR,
    )


__all__ = [
    "ExtractionResult",
    "EPPObservado",
    "PersonaExtraida",
    "create_extractor_agent",
    "PROMPT_EXTRACTOR",
]
