from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class VLMBackend(Enum):
    """Backend de inferencia VLM.

    Attributes:
        FIREWORKS: Fireworks AI API (interim, cloud).
        AMD: AMD MI300X via vLLM ROCm (target, on-premise).
    """

    FIREWORKS = "fireworks"
    AMD = "amd"


class Zone(BaseModel):
    """Configuración de una zona minera.

    Define el EPP requerido y nivel de riesgo base por zona.

    Attributes:
        zone_id: Identificador único de la zona.
        nombre: Nombre descriptivo de la zona.
        required_epp: Lista de EPP obligatorios en la zona.
        riesgo_base: Nivel de riesgo base de la zona.
        min_confidence: Confianza mínima de YOLO para reportar.
        articulos_ds132: Artículos DS 132 aplicables a la zona.
    """

    zone_id: str = Field(description="Identificador único de la zona")
    nombre: str = Field(description="Nombre descriptivo de la zona")
    required_epp: list[str] = Field(description="EPP obligatorios en la zona")
    riesgo_base: Literal["bajo", "medio", "alto", "critico"] = Field(
        description="Nivel de riesgo base de la zona"
    )
    min_confidence: float = Field(
        ge=0.0, le=1.0, default=0.6,
        description="Confianza mínima de YOLO para reportar"
    )
    articulos_ds132: list[str] = Field(description="Artículos DS 132 aplicables")


class AgentConfig(BaseModel):
    """Configuración de agentes VLM.

    Attributes:
        timeout: Timeout por llamada VLM en segundos.
        max_retries: Intentos máximos de validación Pydantic Gate.
        temperature: Temperatura del LLM (baja = más determinístico).
    """

    timeout: float = Field(default=30.0, description="Timeout por llamada VLM")
    max_retries: int = Field(default=3, description="Intentos máximos del Gate")
    temperature: float = Field(default=0.1, description="Temperatura del LLM")


class AppSettings(BaseSettings):
    """Configuración centralizada de Munin.

    Todas las variables de entorno tienen el prefijo MUNIN_.
    Se cargan automáticamente desde un archivo .env si existe.

    Attributes:
        vlm_backend: Backend VLM a usar (fireworks | amd).
        fireworks_api_key: API key para Fireworks AI.
        fireworks_model: Modelo VLM en Fireworks.
        amd_vllm_endpoint: Endpoint de vLLM en AMD MI300X.
        amd_model: Modelo VLM en AMD vLLM.
        yolo_model_path: Ruta al modelo YOLO (.pt).
        yolo_confidence_threshold: Threshold de confianza YOLO.
        yolo_device: Dispositivo de inferencia YOLO.
        frame_rate: FPS objetivo de extracción de frames.
        min_consecutive_frames: Frames consecutivos con violación antes de VLM.
        vlm_busy_timeout: Timeout cuando VLM está ocupado.
        ds132_kb_path: Ruta al knowledge base DS 132.
        zones_config_path: Ruta al archivo de configuración de zonas.
        log_level: Nivel de logging.
    """

    # VLM Backend
    vlm_backend: VLMBackend = Field(default=VLMBackend.FIREWORKS)
    fireworks_api_key: str = Field(default="")
    fireworks_model: str = Field(
        default="accounts/fireworks/models/kimi-k2p6"
    )
    amd_vllm_endpoint: str = Field(default="http://localhost:8000/v1")
    amd_model: str = Field(default="InternVL2-8B")

    # YOLO
    yolo_model_path: str = Field(
        default="recursos/datasets/model-weights/yolov8n_ppe_best.pt"
    )
    yolo_coco_model_path: str = Field(
        default="yolov8n.pt",
        description="Ruta al modelo YOLOv8n COCO base para detección de personas",
    )
    yolo_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0
    )
    yolo_device: str = Field(default="cuda:0")

    # Pipeline
    frame_rate: int = Field(default=25)
    min_consecutive_frames: int = Field(default=3)
    vlm_busy_timeout: float = Field(default=30.0)

    # Knowledge
    ds132_kb_path: str = Field(default="./knowledge/ds132_kb.json")
    zones_config_path: str = Field(default="./knowledge/zones.json")

    # Logging
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_prefix="MUNIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


__all__ = ["VLMBackend", "Zone", "AgentConfig", "AppSettings"]
