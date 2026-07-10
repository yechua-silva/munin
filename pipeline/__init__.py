from __future__ import annotations

"""Pipeline layer — extracción, detección, tracking y compliance.

Componentes del pipeline de visión industrial:
- IFrameExtractor: extracción de frames desde video MP4
- IDetector: detección de objetos (personas + EPP)
- ITracker: tracking de personas entre frames
- IComplianceChecker: verificación de cumplimiento EPP por zona
- Pipeline: orquestador del flujo completo
- PipelineFactory: composition root para DI
"""

from munin.pipeline.pipeline import Pipeline, PipelineCallbacks
from munin.pipeline.factory import PipelineFactory

__all__ = [
    "Pipeline",
    "PipelineCallbacks",
    "PipelineFactory",
]
