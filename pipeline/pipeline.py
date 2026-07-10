from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from munin.agents.orchestrator import MuninOrchestrator
from munin.config import AppSettings, Zone
from munin.gate.schemas import AgentDecision, DetectionResult, Violation
from munin.knowledge.zone_config import ZoneConfig
from munin.pipeline.interfaces import IComplianceChecker, IDetector, IFrameExtractor, ITracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineCallbacks:
    """Callbacks opcionales para eventos del pipeline de visión.

    Cada callback se invoca en un punto específico del flujo de
    procesamiento por frame. Todos son opcionales (default None).

    Attributes:
        on_detection: Se invoca tras la detección YOLO con la
            lista de DetectionResult del frame actual.
        on_violation: Se invoca tras el compliance check con la
            lista de Violation detectadas.
        on_decision: Se invoca tras el análisis VLM (o decisión
            por defecto) con la lista acumulada de AgentDecision.
        on_progress: Se invoca al finalizar cada frame con el
            índice del frame actual y el total de frames.
    """

    on_detection: Callable[[list[DetectionResult]], None] | None = None
    on_violation: Callable[[list[Violation]], None] | None = None
    on_decision: Callable[[list[AgentDecision]], None] | None = None
    on_progress: Callable[[int, int], None] | None = None


class Pipeline:
    """Orquesta el flujo completo de video → decisiones.

    Pipeline end-to-end que coordina la ejecución secuencial de:
    extracción de frames → detección YOLO → tracking de personas →
    verificación de compliance EPP → análisis VLM (si hay violaciones).

    Attributes:
        _extractor: Extractor de frames de video (IFrameExtractor).
        _detector: Detector de objetos YOLO (IDetector).
        _tracker: Tracker de personas entre frames (ITracker).
        _checker: Verificador de compliance EPP (IComplianceChecker).
        _orchestrator: Orquestador de agents VLM (MuninOrchestrator).
        _zone_config: Configuración de zonas mineras (ZoneConfig).
        _settings: Configuración global de la aplicación (AppSettings).
        _callbacks: Callbacks opcionales para eventos del pipeline.
        _vlm_busy: Flag de control de concurrencia VLM (ADR-006).
    """

    def __init__(
        self,
        extractor: IFrameExtractor,
        detector: IDetector,
        tracker: ITracker,
        checker: IComplianceChecker,
        orchestrator: MuninOrchestrator,
        zone_config: ZoneConfig,
        settings: AppSettings,
        callbacks: PipelineCallbacks | None = None,
    ) -> None:
        """Inicializa el Pipeline con todas sus dependencias inyectadas.

        Args:
            extractor: Extractor de frames de video.
            detector: Detector de objetos (personas + EPP).
            tracker: Tracker de personas entre frames.
            checker: Verificador de compliance EPP.
            orchestrator: Orquestador de agents VLM.
            zone_config: Configuración de zonas mineras.
            settings: Configuración global de la aplicación.
            callbacks: Callbacks opcionales para eventos del pipeline.
        """
        self._extractor: IFrameExtractor = extractor
        self._detector: IDetector = detector
        self._tracker: ITracker = tracker
        self._checker: IComplianceChecker = checker
        self._orchestrator: MuninOrchestrator = orchestrator
        self._zone_config: ZoneConfig = zone_config
        self._settings: AppSettings = settings
        self._callbacks: PipelineCallbacks | None = callbacks
        self._vlm_busy: bool = False
        self._logger = logging.getLogger(self.__class__.__name__)

    async def process(
        self,
        video_path: str,
        zone_id: str = "extraccion",
    ) -> list[AgentDecision]:
        """Procesa un video completo y retorna todas las decisiones.

        Pipeline completo por frame:
        1. Obtiene configuración de la zona minera.
        2. Extrae todos los frames del video.
        3. Por cada frame:
           a. Detecta objetos (personas y EPP) con YOLO.
           b. Invoca callback on_detection si existe.
           c. Actualiza tracking de personas.
           d. Verifica compliance EPP contra la zona.
           e. Invoca callback on_violation si existe.
           f. Si hay violaciones y VLM está libre: analiza con VLM.
           g. Si hay violaciones y VLM ocupado: decisión por defecto.
           h. Invoca callback on_decision si existe.
           i. Invoca callback on_progress si existe.
        4. Retorna todas las decisiones acumuladas.

        Args:
            video_path: Ruta al archivo MP4 a procesar.
            zone_id: ID de la zona minera (default: "extraccion").

        Returns:
            Lista de AgentDecision acumuladas durante el procesamiento
            de todos los frames. Puede estar vacía si no hay violaciones.

        Raises:
            KnowledgeBaseError: Si la zona no existe en zone_config.
            VideoLoadError: Si el video no puede ser cargado.
        """
        # 1. Obtener configuración de zona
        zone: Zone = self._zone_config.get_zone(zone_id)
        self._logger.info(
            "Processing video: %s | zone: %s (%s)",
            video_path,
            zone_id,
            zone.nombre,
        )

        # Verificar si stream mode está activado
        if hasattr(self._settings, 'yolo_stream_mode') and self._settings.yolo_stream_mode:
            self._logger.info("Stream mode enabled, using _process_stream()")
            return await self._process_stream(video_path, zone_id)

        # 2. Extraer frames del video
        frames: list[np.ndarray] = self._extractor.extract(video_path)
        self._logger.info(
            "Extracted %d frames from %s", len(frames), video_path
        )

        all_decisions: list[AgentDecision] = []

        # 3. Procesar cada frame secuencialmente
        for i, frame in enumerate(frames):
            self._logger.debug(
                "Processing frame %d/%d", i + 1, len(frames)
            )

            try:
                # a. Detección de objetos en el frame
                detections: list[DetectionResult] = self._detector.detect(frame)

                # b. Callback on_detection
                if self._callbacks and self._callbacks.on_detection:
                    self._callbacks.on_detection(detections)

                # c. Actualizar tracking con el frame completo
                persons = self._tracker.update(frame)

                # d. Verificar compliance EPP contra la zona
                violations: list[Violation] = self._checker.check(persons, detections, zone)

                # e. Callback on_violation
                if self._callbacks and self._callbacks.on_violation:
                    self._callbacks.on_violation(violations)

                # f-g. Análisis VLM si hay violaciones (ADR-006)
                if violations:
                    await self._handle_violations(frame, violations, all_decisions, i + 1)

                # h. Callback on_decision
                if self._callbacks and self._callbacks.on_decision:
                    self._callbacks.on_decision(all_decisions)

                # i. Callback on_progress
                if self._callbacks and self._callbacks.on_progress:
                    self._callbacks.on_progress(i + 1, len(frames))

            except Exception as e:
                self._logger.warning(
                    "Error processing frame %d/%d: %s. Skipping frame.",
                    i + 1,
                    len(frames),
                    e,
                )
                continue

        self._logger.info(
            "Pipeline complete: %d frames processed, "
            "%d decisions generated for zone '%s'",
            len(frames),
            len(all_decisions),
            zone_id,
        )

        return all_decisions

    async def _process_stream(
        self,
        video_path: str,
        zone_id: str = "extraccion",
    ) -> list[AgentDecision]:
        """Procesa video en modo streaming (YOLO gestiona frames).

        Usa detector.detect_stream() para iterar frames. Por cada frame
        del stream, obtiene el frame crudo via extractor para tracker y VLM.

        Args:
            video_path: Ruta al video MP4.
            zone_id: ID de la zona minera.

        Returns:
            Lista de AgentDecision acumuladas.
        """
        zone: Zone = self._zone_config.get_zone(zone_id)
        self._logger.info(
            "Processing video (stream mode): %s | zone: %s",
            video_path,
            zone_id,
        )

        all_decisions: list[AgentDecision] = []
        frame_idx = 0

        for detections in self._detector.detect_stream(video_path):
            frame_idx += 1
            self._logger.debug(
                "Stream frame %d: %d detections", frame_idx, len(detections)
            )

            try:
                # Callback on_detection
                if self._callbacks and self._callbacks.on_detection:
                    self._callbacks.on_detection(detections)

                # Tracker usa frame — pero en stream mode no tenemos frame crudo
                # directamente. Usar tracker con detections (legacy compat)
                # PersonTracker.update acepta list[DetectionResult] como fallback
                persons = self._tracker.update(detections)

                # Checker con 3 params
                violations = self._checker.check(persons, detections, zone)

                if self._callbacks and self._callbacks.on_violation:
                    self._callbacks.on_violation(violations)

                if violations:
                    # En stream mode no tenemos frame crudo para VLM
                    # Crear decisiones por defecto
                    for violation in violations:
                        all_decisions.append(self._default_busy_decision(violation))

                if self._callbacks and self._callbacks.on_decision:
                    self._callbacks.on_decision(all_decisions)

            except Exception as e:
                self._logger.warning(
                    "Error processing stream frame %d: %s. Skipping.",
                    frame_idx,
                    e,
                )
                continue

        self._logger.info(
            "Stream pipeline complete: %d frames, %d decisions for zone '%s'",
            frame_idx,
            len(all_decisions),
            zone_id,
        )

        return all_decisions

    async def _handle_violations(
        self,
        frame: np.ndarray,
        violations: list[Violation],
        all_decisions: list[AgentDecision],
        frame_number: int,
    ) -> None:
        """Maneja violaciones con control de concurrencia VLM.

        Si el VLM está disponible (ADR-006), delega el análisis
        al orquestador. Si está ocupado, genera decisiones por
        defecto que requieren revisión humana.

        Args:
            frame: Frame actual del video (HWC, BGR, uint8).
            violations: Violaciones detectadas en el frame.
            all_decisions: Lista acumulada de decisiones (se modifica in-place).
            frame_number: Número de frame (1-indexed) para logging.
        """
        if not self._vlm_busy:
            # f. VLM libre: analizar con el orquestador
            self._vlm_busy = True
            try:
                self._logger.info(
                    "Frame %d: %d violation(s), analyzing with VLM",
                    frame_number,
                    len(violations),
                )
                decisions = await self._orchestrator.analyze(
                    frame, violations,
                    use_single_pass=self._settings.use_single_pass,
                )
                all_decisions.extend(decisions)
            finally:
                self._vlm_busy = False
        else:
            # g. VLM ocupado: decisiones por defecto (ADR-006)
            self._logger.warning(
                "Frame %d: VLM busy, creating default decisions "
                "for %d violation(s) (requires human review)",
                frame_number,
                len(violations),
            )
            for violation in violations:
                all_decisions.append(self._default_busy_decision(violation))

    def _default_busy_decision(self, violation: Violation) -> AgentDecision:
        """Crea una decisión por defecto cuando el VLM está ocupado.

        Args:
            violation: Violación que no pudo ser analizada por VLM.

        Returns:
            AgentDecision con requiere_revision_humana=True y
            confianza 0.0, indicando que se requiere intervención
            humana para validar la violación.
        """
        return AgentDecision(
            zona=violation.zona_id,
            tipo_violacion="EPP_FALTANTE",
            epp_faltante=violation.epp_faltantes,
            nivel_riesgo="BAJO",
            timestamp=datetime.now(),
            confianza=0.0,
            requiere_revision_humana=True,
            razonamiento_vlm="VLM ocupado: se requirió revisión humana",
        )


__all__ = [
    "Pipeline",
    "PipelineCallbacks",
]
