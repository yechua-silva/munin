from __future__ import annotations

"""Pipeline layer — extracción, detección, tracking y compliance.

Componentes del pipeline de visión industrial:
- IFrameExtractor: extracción de frames desde video MP4
- IDetector: detección de objetos (personas + EPP)
- ITracker: tracking de personas entre frames
- IComplianceChecker: verificación de cumplimiento EPP por zona
- Pipeline: orquestador del flujo completo
- PipelineFactory: composition root para DI
- FrameCollector: colector multi-fuente con threads (Track C)
- VLMQueue: cola de prioridad VLM con backpressure (Track C)
- MultiCameraPipeline: pipeline multi-cámara (Track C)
"""

from munin.pipeline.frame_collector import CameraSource, FrameCollector
from munin.pipeline.multi_camera import MultiCameraPipeline
from munin.pipeline.pipeline import Pipeline, PipelineCallbacks
from munin.pipeline.factory import PipelineFactory
from munin.pipeline.vlm_queue import VLMQueue, VLMRequest, VLMResult

__all__ = [
    "Pipeline",
    "PipelineCallbacks",
    "PipelineFactory",
    "CameraSource",
    "FrameCollector",
    "MultiCameraPipeline",
    "VLMQueue",
    "VLMRequest",
    "VLMResult",
]
