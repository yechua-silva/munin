"""MultiCameraPipeline — Pipeline multi-cámara con batch YOLO + VLM queue.

Arquitectura:
1. FrameCollector captura frames de N cámaras en threads
2. Main loop recolecta batch de frames
3. IDetector.detect() por cada frame (o batch si el detector lo soporta)
4. Por cada cámara: ITracker.update() + IComplianceChecker.check()
5. VLMQueue.enqueue() si hay violaciones (non-blocking)
6. drain_results() consolida decisiones

ADR-031: MultiCameraPipeline con tracker factory por cámara.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from munin.config import AppSettings, Zone
from munin.gate.schemas import AgentDecision, DetectionResult, Violation
from munin.knowledge.zone_config import ZoneConfig
from munin.pipeline.frame_collector import CameraSource, FrameCollector
from munin.pipeline.interfaces import IComplianceChecker, IDetector, ITracker
from munin.pipeline.vlm_queue import VLMQueue

logger = logging.getLogger(__name__)


class MultiCameraPipeline:
    """Pipeline multi-cámara con batch YOLO + N trackers + VLM queue.

    Arquitectura:
    1. FrameCollector captura frames de N cámaras en threads
    2. Main loop recolecta batch de frames
    3. IDetector.detect() por cada frame (o batch si el detector lo soporta)
    4. Por cada cámara: ITracker.update() + IComplianceChecker.check()
    5. VLMQueue.enqueue() si hay violaciones (non-blocking)
    6. drain_results() consolida decisiones

    Attributes:
        _detector: Detector central (compartido entre cámaras).
        _collector: FrameCollector con N cámaras.
        _tracker_factory: Factory que crea trackers ligeros.
        _checker: Compliance checker (compartido).
        _vlm_queue: Cola VLM con worker thread.
        _zone_config: Config de zonas mineras.
        _settings: Config global.
    """

    def __init__(
        self,
        detector: IDetector,
        sources: list[CameraSource],
        tracker_factory: callable,
        checker: IComplianceChecker,
        vlm_queue: VLMQueue,
        zone_config: ZoneConfig,
        settings: AppSettings,
    ) -> None:
        """Inicializa el pipeline multi-cámara.

        Args:
            detector: Detector central compartido entre cámaras.
            sources: Lista de fuentes de cámara.
            tracker_factory: Función que crea ITracker por cámara.
            checker: Compliance checker compartido.
            vlm_queue: Cola VLM con backpressure.
            zone_config: Config de zonas mineras.
            settings: Config global de la aplicación.
        """
        self._detector = detector
        self._collector = FrameCollector(sources)
        self._sources = {s.camera_id: s for s in sources}
        self._trackers: dict[str, ITracker] = {
            s.camera_id: tracker_factory() for s in sources
        }
        self._checker = checker
        self._vlm_queue = vlm_queue
        self._zone_config = zone_config
        self._settings = settings
        self._all_decisions: dict[str, list[AgentDecision]] = {
            s.camera_id: [] for s in sources
        }
        self._logger = logging.getLogger(self.__class__.__name__)

    async def process_all(
        self,
        frame_limit: int | None = None,
    ) -> dict[str, list[AgentDecision]]:
        """Procesa todas las cámaras en un bucle continuo.

        Args:
            frame_limit: Máximo número de frames a procesar
                por cámara (None = infinito, hasta KeyboardInterrupt).

        Returns:
            Dict[camera_id, list[AgentDecision]] con decisiones
            acumuladas de todas las cámaras.
        """
        self._collector.start()
        self._vlm_queue.start()

        frame_count = 0
        try:
            while True:
                if frame_limit and frame_count >= frame_limit:
                    break

                batch = self._collector.collect()
                if not batch:
                    # Sin frames disponibles, esperar breve
                    import asyncio
                    await asyncio.sleep(0.01)
                    continue

                frame_count += 1

                for camera_id, frame in batch.items():
                    try:
                        detections = self._detector.detect(frame)
                        tracker = self._trackers[camera_id]
                        persons = tracker.update(detections)

                        source = self._sources[camera_id]
                        zone = self._zone_config.get_zone(source.zone_id)

                        violations = self._checker.check(
                            persons, detections, zone,
                        )

                        if violations:
                            # Set camera_id en violations
                            for v in violations:
                                v.camera_id = camera_id

                            enqueued = self._vlm_queue.enqueue(
                                camera_id=camera_id,
                                frame=frame,
                                violations=violations,
                                frame_number=frame_count,
                            )
                            if not enqueued:
                                # Backpressure: decisión por defecto
                                for v in violations:
                                    self._all_decisions[camera_id].append(
                                        self._default_decision(v, camera_id)
                                    )
                    except Exception as e:
                        self._logger.warning(
                            "Error processing %s frame %d: %s",
                            camera_id, frame_count, e,
                        )

                # Drenar resultados VLM
                self._drain_results()

        except KeyboardInterrupt:
            self._logger.info("Pipeline interrupted by user")
        finally:
            self._vlm_queue.stop()
            self._collector.release()

        return self._all_decisions

    def _drain_results(self) -> None:
        """Drena resultados del VLM worker y los consolida."""
        for result in self._vlm_queue.drain_results():
            if result.error:
                self._logger.warning(
                    "VLM error for %s: %s", result.camera_id, result.error,
                )
                continue
            # Set camera_id en decisions
            for d in result.decisions:
                d.camera_id = result.camera_id
            self._all_decisions[result.camera_id].extend(result.decisions)

    @staticmethod
    def _default_decision(
        violation: Violation,
        camera_id: str,
    ) -> AgentDecision:
        """Decisión por defecto cuando VLM no disponible (backpressure).

        Args:
            violation: Violación que no pudo ser analizada por VLM.
            camera_id: ID de la cámara.

        Returns:
            AgentDecision con requiere_revision_humana=True y
            camera_id seteado.
        """
        return AgentDecision(
            zona=violation.zona_id,
            tipo_violacion="EPP_FALTANTE",
            epp_faltante=violation.epp_faltantes,
            nivel_riesgo="BAJO",
            timestamp=datetime.now(),
            confianza=0.0,
            requiere_revision_humana=True,
            razonamiento_vlm="VLM no disponible: backpressure o error",
            camera_id=camera_id,
        )


__all__ = ["MultiCameraPipeline"]
