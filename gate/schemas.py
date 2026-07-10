from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PPEMissing(BaseModel):
    """EPP faltante detectado en una persona.

    Attributes:
        tipo: Tipo de EPP faltante (Literal con 6 valores).
        descripcion: Nombre en español del EPP.
        norma_chilena: Norma chilena asociada (ej: NCh 1411).
    """

    tipo: Literal[
        "hardhat",
        "safety_vest",
        "gloves",
        "safety_glasses",
        "safety_boots",
        "harness",
        "mask",
    ] = Field(description="Tipo de EPP faltante")
    descripcion: str = Field(description="Nombre en español del EPP")
    norma_chilena: str = Field(description="Norma chilena asociada, ej: NCh 1411")


class DetectionResult(BaseModel):
    """Resultado de detección YOLO en un frame.

    Attributes:
        class_name: Clase del objeto detectado.
        bbox: Bounding box en formato (x1, y1, x2, y2).
        confidence: Confianza de la detección (0.0 - 1.0).
    """

    class_name: Literal[
        # Clases positivas
        "person",
        "hardhat",
        "safety_vest",
        "gloves",
        "safety_glasses",
        "safety_boots",
        "harness",
        "mask",
        # Clases negativas (SPEC-v3: dual-class mode)
        "no_helmet",
        "no_gloves",
        "no_vest",
        "no_boots",
        "no_goggle",
        "no_safety_glasses",
    ] = Field(description="Clase del objeto detectado")
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box (x1, y1, x2, y2)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confianza de la detección"
    )


class TrackedPerson(BaseModel):
    """Persona trackeada con EPP asignado.

    Attributes:
        persona_id: ID único de tracking.
        bbox: Bounding box actual de la persona.
        epp_detectado: Conjunto de EPP detectado en la persona.
        lost_counter: Frames consecutivos sin detectar a esta persona.
        consecutive_violations: Frames consecutivos con violación.
    """

    persona_id: int = Field(description="ID único de tracking")
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box actual (x1, y1, x2, y2)"
    )
    epp_detectado: set[str] = Field(
        default_factory=set, description="EPP detectado en la persona"
    )
    lost_counter: int = Field(
        default=0, description="Frames sin detección"
    )
    consecutive_violations: int = Field(
        default=0, description="Frames consecutivos con violación"
    )

    model_config = {"arbitrary_types_allowed": True}


class Violation(BaseModel):
    """Violación de EPP detectada por el rule engine.

    Attributes:
        persona_id: ID de la persona con violación.
        zona_id: ID de la zona donde ocurre la violación.
        epp_faltantes: Lista de EPP faltantes.
        frame_id: Número de frame donde se detectó.
        timestamp: Momento de la detección.
    """

    persona_id: int = Field(description="ID de la persona con violación")
    zona_id: str = Field(description="ID de la zona donde ocurre")
    epp_faltantes: list[PPEMissing] = Field(
        default_factory=list, description="EPP faltantes"
    )
    frame_id: int = Field(description="Número de frame")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Momento de la detección"
    )


class AgentDecision(BaseModel):
    """Schema final validado por Pydantic Gate.

    Este es el output universal del sistema. Toda salida del VLM
    debe ser validada contra este schema.

    Attributes:
        zona: ID de la zona minera donde ocurre la violación.
        tipo_violacion: Tipo de violación detectada.
        epp_faltante: Lista de EPP faltantes.
        nivel_riesgo: Nivel de riesgo asignado.
        timestamp: Momento de la decisión.
        articulo_ds132: Artículo DS 132 violado.
        confianza: Confianza de la decisión (0.0 - 1.0).
        requiere_revision_humana: Si requiere revisión humana.
        razonamiento_vlm: Razonamiento del VLM.
    """

    zona: str = Field(description="ID de la zona minera")
    tipo_violacion: Literal[
        "EPP_FALTANTE",
        "ZONA_NO_AUTORIZADA",
        "SIN_VIOLACION",
    ] = Field(description="Tipo de violación")
    epp_faltante: list[PPEMissing] = Field(
        default_factory=list, description="EPP faltantes"
    )
    nivel_riesgo: Literal["BAJO", "MEDIO", "ALTO", "CRITICO"] = Field(
        description="Nivel de riesgo"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Momento de la decisión"
    )
    articulo_ds132: str | None = Field(
        default=None, description="Artículo DS 132 violado, ej: 'Art. 38'"
    )
    confianza: float = Field(
        ge=0.0, le=1.0, description="Confianza de la decisión"
    )
    requiere_revision_humana: bool = Field(
        default=False, description="Si requiere revisión humana"
    )
    razonamiento_vlm: str | None = Field(
        default=None, description="Razonamiento del VLM"
    )


NEGATIVE_CLASS_MAP: dict[str, str] = {
    "no_helmet": "hardhat",
    "no_gloves": "gloves",
    "no_vest": "safety_vest",
    "no_boots": "safety_boots",
    "no_goggle": "safety_glasses",
    "no_safety_glasses": "safety_glasses",
}
"""Mapeo de clases negativas a EPP positivo (SPEC-v3 dual-class mode).

Ejemplo: si se detecta 'no_helmet', el EPP positivo correspondiente
es 'hardhat'. Se usa en PPEComplianceChecker modo DUAL_CLASS para
determinar qué EPP falta cuando el modelo detecta una clase negativa.
"""


__all__ = [
    "PPEMissing",
    "DetectionResult",
    "TrackedPerson",
    "Violation",
    "AgentDecision",
    "NEGATIVE_CLASS_MAP",
]
