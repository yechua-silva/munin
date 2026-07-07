from __future__ import annotations

import logging

from munin.agents.orchestrator import MuninOrchestrator
from munin.config import AppSettings
from munin.exceptions import ConfigurationError
from munin.knowledge.ds132_kb import DS132KnowledgeBase
from munin.knowledge.zone_config import ZoneConfig
from munin.pipeline.frame_extractor import FrameExtractor
from munin.pipeline.person_tracker import PersonTracker
from munin.pipeline.pipeline import Pipeline, PipelineCallbacks
from munin.pipeline.ppe_checker import PPEComplianceChecker
from munin.pipeline.yolo_detector import YOLODetector
from munin.vlm.factory import VLMModelFactory

logger = logging.getLogger(__name__)


class PipelineFactory:
    """Factory para crear el Pipeline completo con DI (composition root).

    Este es el punto de entrada único para construir el pipeline
    de visión industrial. Todas las dependencias se crean y se
    inyectan automáticamente a partir de AppSettings.

    Uso típico::

        settings = AppSettings()
        pipeline = PipelineFactory.create(settings)
        decisions = await pipeline.process("video.mp4", "extraccion")
    """

    @staticmethod
    def create(settings: AppSettings) -> Pipeline:
        """Crea un Pipeline completamente configurado.

        Composition root que construye todas las dependencias:
        modelo VLM, knowledge base DS132, configuración de zonas,
        extractor de frames, detector YOLO, tracker, compliance
        checker, y orquestador de agents.

        Args:
            settings: Configuración global de la aplicación
                (AppSettings cargada desde variables de entorno).

        Returns:
            Pipeline listo para procesar videos.

        Raises:
            ConfigurationError: Si alguna dependencia falla al
                ser creada (archivos faltantes, modelos inválidos,
                errores de configuración, etc.).
        """
        try:
            logger.info("Creating Pipeline from AppSettings")

            # 1. Crear modelo VLM (OpenAIChatModel via VLMModelFactory)
            model = VLMModelFactory.create(settings)

            # 2. Cargar knowledge base DS 132
            ds132_kb = DS132KnowledgeBase(settings.ds132_kb_path)

            # 3. Cargar configuración de zonas desde JSON
            zone_config = ZoneConfig.from_json(settings.zones_config_path)

            # 4. Crear extractor de frames (OpenCV)
            extractor = FrameExtractor(fps=settings.frame_rate)

            # 5. Crear detector YOLO (doble modelo: COCO + PPE)
            detector = YOLODetector(
                coco_model_path=settings.yolo_coco_model_path,
                ppe_model_path=settings.yolo_model_path,
                confidence=settings.yolo_confidence_threshold,
                device=settings.yolo_device,
            )

            # 6. Crear tracker de personas (IoU-based)
            tracker = PersonTracker()

            # 7. Crear verificador de compliance EPP
            checker = PPEComplianceChecker(
                zone_config=zone_config,
                ds132_kb=ds132_kb,
            )

            # 8. Crear orquestador VLM con timeout configurable
            orchestrator = MuninOrchestrator.from_model(
                model,
                timeout=settings.vlm_busy_timeout,
            )

            # 9. Ensamblar pipeline completo
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
                "frame_rate=%d, yolo_device=%s, vlm_timeout=%.1fs",
                len(zone_config._zones) if hasattr(zone_config, '_zones') else 0,
                settings.frame_rate,
                settings.yolo_device,
                settings.vlm_busy_timeout,
            )
            return pipeline

        except ConfigurationError:
            # Re-lanzar ConfigurationError directamente
            raise
        except Exception as e:
            logger.error("Failed to create Pipeline: %s", e)
            raise ConfigurationError(
                f"Failed to create Pipeline: {e}"
            ) from e


__all__ = [
    "PipelineFactory",
]
