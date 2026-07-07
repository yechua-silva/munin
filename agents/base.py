from __future__ import annotations

from pydantic import BaseModel, Field

from munin.gate.schemas import Violation


class AgentContext(BaseModel):
    """Contexto compartido entre agentes VLM.

    Se construye en el MuninOrchestrator y se pasa secuencialmente
    entre los agentes. Cada agente puede agregar su output
    al contexto para que el siguiente lo use.

    Attributes:
        violation: Violación detectada por el rule engine.
        extractor_output: Output del ExtractorAgent (JSON string).
        analysis_output: Output del ContextAnalyzerAgent (JSON string).
        frame_id: Número de frame que se está analizando.
    """

    violation: Violation = Field(description="Violación detectada")
    extractor_output: str | None = Field(
        default=None,
        description="Output del ExtractorAgent (JSON string)",
    )
    analysis_output: str | None = Field(
        default=None,
        description="Output del ContextAnalyzerAgent (JSON string)",
    )
    frame_id: int = Field(
        default=0,
        description="Número de frame que se analiza",
    )


__all__ = ["AgentContext"]
