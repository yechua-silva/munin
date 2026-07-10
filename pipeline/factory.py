"""PipelineFactory — Composition root para el pipeline de Munin v3.

Selecciona componentes según AppSettings: detector (LEGACY | DUAL_CLASS),
tracker (ByteTrackAdapter | PersonTracker fallback), checker (ComplianceMode),
orchestrator (resize + timeout).
"""
from __future__ import annotations

import logging

from munin.agents.orchestrator import MuninOrchestrator
from munin.config import AppSettings
from munin.exceptions import ConfigurationError
from munin.knowledge.ds132_kb import DS132KnowledgeBase
from munin.knowledge.zone_config import ZoneConfig
from munin.pipeline.frame_extractor import FrameExtractor
from munin.pipeline.pipeline import Pipeline, PipelineCallbacks
from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker
from munin.pipeline.two_model_detector import TwoModelDetector
from munin.vlm.factory import VLMModelFactory

logger = logging.getLogger(__name__)


class PipelineFactory:
    """Factory para crear el Pipeline completo con DI (composition root).

    Este es el punto de entrada único para construir el pipeline
    de visión industrial. Todas las dependencias se crean y se
    inyectan automáticamente a partir de AppSettings.

    v3: Selecciona detector según compliance_mode, usa ByteTrackAdapter
    como tracker primario con fallback PersonTracker, y configura
    orchestrator con resize y timeout desde settings.

    Uso típico::

        settings = AppSettings()
        pipeline = PipelineFactory.create(settings)
        decisions = await pipeline.process("video.mp4", "extraccion")
    """

    @staticmethod
    def create(settings: AppSettings) -> Pipeline:
        """Crea un Pipeline completamente configurado.

        Composition root que construye todas las dependencias según
        AppSettings. Selecciona detector (LEGACY vs DUAL_CLASS),
        tracker (ByteTrackAdapter vs PersonTracker), checker mode,
        y orchestrator con resize/timeout.

        Args:
            settings: Configuración global de la aplicación.

        Returns:
            Pipeline listo para procesar videos.

        Raises:
            ConfigurationError: Si alguna dependencia falla.
        """
        try:
            logger.info("Creating Pipeline from AppSettings")

            # 1. Crear modelo VLM
            model = VLMModelFactory.create(settings)

            # 2. Cargar knowledge base DS 132
            ds132_kb = DS132KnowledgeBase(settings.ds132_kb_path)

            # 3. Cargar configuración de zonas
            zone_config = ZoneConfig.from_json(settings.zones_config_path)

            # 4. Crear extractor de frames
            extractor = FrameExtractor(fps=settings.frame_rate)

            # 5. Seleccionar detector según compliance_mode (ADR-018)
            if settings.compliance_mode == "dual_class":
                from munin.pipeline.single_model_detector import (
                    SingleModelDetector,
                )
                detector = SingleModelDetector(
                    model_path=settings.yolo_ppe_model_path,
                    confidence=settings.yolo_confidence_threshold,
                    device=settings.yolo_device,
                    imgsz=settings.yolo_imgsz,
                )
                logger.info("Detector: SingleModelDetector (DUAL_CLASS mode)")
            else:
                detector = TwoModelDetector(
                    coco_model_path=settings.yolo_coco_model_path,
                    ppe_model_path=settings.yolo_model_path,
                    confidence=settings.yolo_confidence_threshold,
                    device=settings.yolo_device,
                    imgsz=settings.yolo_imgsz,
                )
                logger.info("Detector: TwoModelDetector (LEGACY mode)")

            # 6. Tracker: ByteTrackAdapter primario, PersonTracker fallback
            try:
                from munin.pipeline.byte_track_adapter import ByteTrackAdapter
                tracker = ByteTrackAdapter(
                    model_path=settings.yolo_coco_model_path,
                    confidence=settings.yolo_confidence_threshold,
                    device=settings.yolo_device,
                    imgsz=settings.yolo_imgsz,
                )
                logger.info("Tracker: ByteTrackAdapter created successfully")
            except Exception as e:
                from munin.pipeline.person_tracker import PersonTracker
                logger.warning(
                    "ByteTrackAdapter failed (%s), falling back to PersonTracker",
                    e,
                )
                tracker = PersonTracker(detector=detector)

            # 7. Checker con compliance_mode (ADR-014)
            mode = (
                ComplianceMode.DUAL_CLASS
                if settings.compliance_mode == "dual_class"
                else ComplianceMode.LEGACY
            )
            checker = PPEComplianceChecker(
                zone_config=zone_config,
                ds132_kb=ds132_kb,
                mode=mode,
            )

            # 8. Orchestrator con resize, timeout y max_tokens (ADR-016)
            orchestrator = MuninOrchestrator.from_model(
                model,
                timeout=settings.vlm_busy_timeout,
                resize_width=settings.frame_resize_width,
                resize_height=settings.frame_resize_height,
                max_tokens=settings.vlm_max_tokens,
            )

            # 9. Ensamblar pipeline
            pipeline = Pipeline(
                extractor=extractor,
                detector=detector,
                tracker=tracker,
                checker=checker,
                orchestrator=orchestrator,
                zone_config=zone_config,
                settings=settings,
            )

            logger.info(
                "Pipeline created successfully: zone_config=%d zones, "
                "frame_rate=%d, yolo_device=%s, vlm_timeout=%.1fs, "
                "compliance_mode=%s, tracker=%s",
                len(zone_config._zones) if hasattr(zone_config, '_zones') else 0,
                settings.frame_rate,
                settings.yolo_device,
                settings.vlm_busy_timeout,
                settings.compliance_mode,
                type(tracker).__name__,
            )
            return pipeline

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error("Failed to create Pipeline: %s", e)
            raise ConfigurationError(
                f"Failed to create Pipeline: {e}"
            ) from e


__all__ = ["PipelineFactory"]
