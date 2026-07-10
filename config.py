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
        polygon: Polígono(s) de la zona en coords normalizadas 0-1.
            Lista de sub-polígonos (unión). None = sin filtro geométrico.
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
    polygon: list[list[list[float]]] | None = Field(
        default=None,
        description="Polígono(s) de la zona en coords normalizadas 0-1. "
        "Lista de sub-polígonos (unión). None = sin filtro geométrico.",
    )


class AgentConfig(BaseModel):
    """Configuración de agentes VLM.

    Attributes:
        timeout: Timeout por llamada VLM en segundos.
        max_retries: Intentos máximos de validación Pydantic Gate.
        temperature: Temperatura del LLM (baja = más determinístico).
    """

    timeout: float = Field(default=300.0, description="Timeout por llamada VLM (300s para reasoning extendido)")
    max_retries: int = Field(default=3, description="Intentos máximos del Gate")
    temperature: float = Field(default=0.1, description="Temperatura del LLM")


class AppSettings(BaseSettings):
    """Configuración centralizada de Munin.

    Todas las variables de entorno tienen el prefijo MUNIN_.
    Se cargan automáticamente desde un archivo .env si existe.

    Attributes:
        vlm_backend: Backend VLM a usar (amd | fireworks). Default: amd (on-premise).
        fireworks_api_key: API key para Fireworks AI.
        fireworks_model: Modelo VLM en Fireworks.
        amd_vllm_endpoint: Endpoint de vLLM en AMD MI300X.
        amd_model: Modelo VLM en AMD vLLM.
        vlm_max_tokens: Máximo de tokens en respuesta VLM.
        yolo_model_path: Ruta al modelo YOLO (.pt).
        yolo_confidence_threshold: Threshold de confianza YOLO.
        yolo_device: Dispositivo de inferencia YOLO.
        frame_rate: FPS objetivo de extracción de frames.
        min_consecutive_frames: Frames consecutivos con violación antes de VLM.
        vlm_busy_timeout: Timeout cuando VLM está ocupado.
        ds132_kb_path: Ruta al knowledge base DS 132.
        zones_config_path: Ruta al archivo de configuración de zonas.
        log_level: Nivel de logging.
        compliance_mode: Modo de compliance EPP (legacy | dual_class).
        yolo_stream_mode: Usar stream=True en YOLO.
        yolo_imgsz: Tamaño de entrada YOLO (320-1280).
        frame_resize_width: Ancho de resize para VLM.
        frame_resize_height: Alto de resize para VLM.
        prompt_cache_session_id: Session ID para prompt caching en Fireworks.
        yolo_ppe_model_path: Ruta al modelo Construction-PPE fine-tuned.
    """

    # VLM Backend
    vlm_backend: VLMBackend = Field(
        default=VLMBackend.AMD,
        description="Backend VLM: 'amd' (vLLM on-premise MI300X) o 'fireworks' (cloud interim)",
    )
    fireworks_api_key: str = Field(default="")
    fireworks_model: str = Field(
        default="accounts/fireworks/models/kimi-k2p6"
    )
    amd_vllm_endpoint: str = Field(default="http://localhost:8000/v1")
    amd_model: str = Field(
        default="InternVL2-8B",
        description="Modelo VLM para AMD vLLM (InternVL2-8B: 8B params, ~8GB VRAM)",
    )
    vlm_max_tokens: int = Field(
        default=2048,
        ge=256, le=32768,
        description="Max tokens para respuestas VLM (256-32768)",
    )

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
    vlm_busy_timeout: float = Field(
        default=300.0,
        description="Timeout VLM en segundos (300s = 5 min para reasoning extendido)",
    )

    # Compliance mode (SPEC-v3)
    compliance_mode: str = Field(
        default="legacy",
        description="Modo de compliance EPP: 'legacy' (6 clases) o 'dual_class' (11 clases)",
    )

    # YOLO stream mode (SPEC-v3)
    yolo_stream_mode: bool = Field(
        default=False,
        description="Usar stream=True en YOLO (gestión interna de frames)",
    )

    # YOLO image size (SPEC-v3)
    yolo_imgsz: int = Field(
        default=640,
        ge=320, le=1280,
        description="Tamaño de entrada YOLO (imgsz)",
    )

    # Frame resize para VLM (ADR-016)
    frame_resize_width: int = Field(
        default=640,
        description="Ancho de redimension para VLM",
    )
    frame_resize_height: int = Field(
        default=480,
        description="Alto de redimension para VLM",
    )

    # Prompt cache session (ADR-015)
    prompt_cache_session_id: str = Field(
        default="munin-session",
        description="Session ID para caching de prompts (x-session-affinity)",
    )

    # Construction-PPE model path (DUAL_CLASS mode)
    yolo_ppe_model_path: str = Field(
        default="/scratch/runs/detect/train/weights/best.pt",
        description="Ruta al modelo Construction-PPE fine-tuned (11 clases)",
    )

    # Knowledge
    ds132_kb_path: str = Field(default="./knowledge/ds132_kb.json")
    zones_config_path: str = Field(default="./knowledge/zones.json")

    # Multi-cámara (Track C)
    multi_camera_enabled: bool = Field(
        default=False,
        description="Habilitar pipeline multi-cámara",
    )
    multi_camera_sources: str = Field(
        default="",
        description="JSON con lista de CameraSource configs",
    )

    # VLM Queue (Track C)
    vlm_queue_max_size: int = Field(
        default=10,
        ge=1, le=100,
        description="Máximo de requests en cola VLM",
    )
    vlm_queue_workers: int = Field(
        default=1,
        ge=1, le=4,
        description="Número de worker threads VLM",
    )
    vlm_coalesce_frames: int = Field(
        default=30,
        ge=1, le=120,
        description="Coalescing: frames consecutivos misma violación para skip",
    )
    vlm_aging_seconds: int = Field(
        default=15,
        ge=1, le=120,
        description="Aging: segundos antes de subir prioridad",
    )

    # SinglePass default (CP4)
    use_single_pass: bool = Field(
        default=True,
        description="Usar SinglePassAgent como default (true) o 3-agent secuencial (false)",
    )

    # Logging
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_prefix="MUNIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


__all__ = ["VLMBackend", "Zone", "AgentConfig", "AppSettings"]
