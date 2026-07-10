from __future__ import annotations

import asyncio
import logging

import cv2 as cv
import numpy as np
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel

from munin.agents.base import AgentContext
from munin.agents.context_analyzer import AnalysisResult, create_context_analyzer_agent
from munin.agents.extractor import ExtractionResult, create_extractor_agent
from munin.agents.scorer import create_scorer_agent
from munin.agents.single_pass import create_single_pass_agent
from munin.gate.schemas import AgentDecision, Violation
from munin.exceptions import VLMError

logger = logging.getLogger(__name__)


class MuninOrchestrator:
    """Coordina los 3 sub-agentes VLM usando programmatic hand-off.

    Modo óptimo: Extractor → ContextAnalyzer → Scorer (secuencia).
    Fallback: SinglePassAgent (1 solo Agent con output AgentDecision).

    Attributes:
        _extractor: Agente extractor.
        _analyzer: Agente analizador contextual.
        _scorer: Agente scorer con output AgentDecision.
        _single_pass: Agente fallback.
        _timeout: Timeout por violación en segundos.
        _resize_width: Ancho de redimension para VLM (default 640).
        _resize_height: Alto de redimension para VLM (default 480).
    """

    def __init__(
        self,
        extractor: Agent[None, ExtractionResult],
        analyzer: Agent[None, AnalysisResult],
        scorer: Agent[None, AgentDecision],
        single_pass: Agent[None, AgentDecision],
        timeout: float = 300.0,
        resize_width: int = 640,
        resize_height: int = 480,
    ) -> None:
        """Inicializa el orquestador con los 4 agentes.

        Args:
            extractor: Agente de extracción visual.
            analyzer: Agente de análisis contextual.
            scorer: Agente de scoring DS 132.
            single_pass: Agente de fallback (prompt fusionado).
            timeout: Timeout por violación en segundos.
            resize_width: Ancho de redimension para VLM (default 640).
            resize_height: Alto de redimension para VLM (default 480).
        """
        self._extractor = extractor
        self._analyzer = analyzer
        self._scorer = scorer
        self._single_pass = single_pass
        self._timeout = timeout
        self._resize_width = resize_width
        self._resize_height = resize_height

    @classmethod
    def from_model(
        cls,
        model: OpenAIChatModel,
        timeout: float = 300.0,
        resize_width: int = 640,
        resize_height: int = 480,
        max_tokens: int = 2048,
    ) -> MuninOrchestrator:
        """Crea el orquestador con todos los agentes desde un modelo.

        Args:
            model: Modelo VLM configurado.
            timeout: Timeout por violación.
            resize_width: Ancho de redimension para VLM (default 640).
            resize_height: Alto de redimension para VLM (default 480).
            max_tokens: Máximo de tokens en respuesta VLM (default 2048).

        Returns:
            MuninOrchestrator listo para usar.
        """
        return cls(
            extractor=create_extractor_agent(model, max_tokens=max_tokens),
            analyzer=create_context_analyzer_agent(model, max_tokens=max_tokens),
            scorer=create_scorer_agent(model, max_tokens=max_tokens),
            single_pass=create_single_pass_agent(model, max_tokens=max_tokens),
            timeout=timeout,
            resize_width=resize_width,
            resize_height=resize_height,
        )

    async def analyze(
        self,
        frame: np.ndarray,
        violations: list[Violation],
        use_single_pass: bool = True,
    ) -> list[AgentDecision]:
        """Analiza violaciones. SinglePass primero si use_single_pass=True.

        Cuando use_single_pass=True (default v4), intenta SinglePassAgent
        primero. Si falla, cae a 3-agent secuencial.
        Cuando use_single_pass=False (comportamiento v3), intenta 3-agent
        primero. Si falla, cae a SinglePassAgent.

        Args:
            frame: Frame del video (HWC, BGR, uint8).
            violations: Violaciones detectadas por rule engine.
            use_single_pass: True = SinglePass primero (default v4),
                False = 3-agent secuencial primero (v3 legacy).

        Returns:
            Lista de AgentDecision validados.

        Raises:
            VLMError: Si todos los métodos fallan para alguna violación.
        """
        if not violations:
            return []

        jpeg_bytes = self._encode_jpeg(frame)
        decisions: list[AgentDecision] = []

        for violation in violations:
            if use_single_pass:
                try:
                    decision = await asyncio.wait_for(
                        self._process_single_pass(jpeg_bytes, violation),
                        timeout=self._timeout,
                    )
                    decisions.append(decision)
                except Exception as e:
                    logger.warning(
                        "SinglePass failed for %s: %s. Trying 3-agent.",
                        violation.persona_id,
                        e,
                    )
                    try:
                        decision = await asyncio.wait_for(
                            self._process_violation(jpeg_bytes, violation),
                            timeout=self._timeout,
                        )
                        decisions.append(decision)
                    except Exception as e2:
                        logger.error(
                            "Both methods failed for %s: %s",
                            violation.persona_id,
                            e2,
                        )
                        decisions.append(self._default_decision(violation))
            else:
                # 3-agent sequential (comportamiento v3)
                try:
                    decision = await asyncio.wait_for(
                        self._process_violation(jpeg_bytes, violation),
                        timeout=self._timeout,
                    )
                    decisions.append(decision)
                except Exception as e:
                    logger.warning(
                        "3-agent failed for %s: %s. Trying SinglePass.",
                        violation.persona_id,
                        e,
                    )
                    try:
                        decision = await asyncio.wait_for(
                            self._process_single_pass(jpeg_bytes, violation),
                            timeout=self._timeout,
                        )
                        decisions.append(decision)
                    except Exception as e2:
                        logger.error(
                            "Both methods failed for %s: %s",
                            violation.persona_id,
                            e2,
                        )
                        decisions.append(self._default_decision(violation))

        return decisions

    async def _process_violation(
        self,
        jpeg_bytes: bytes,
        violation: Violation,
    ) -> AgentDecision:
        """Procesa una violación con 3 agentes secuenciales.

        Args:
            jpeg_bytes: Frame codificado en JPEG.
            violation: Violación a procesar.

        Returns:
            AgentDecision validado por PydanticAI.
        """
        # 1. Extractor — extrae contexto visual del frame
        logger.debug(
            "Violation %d: running ExtractorAgent",
            violation.persona_id,
        )
        extraction = await self._extractor.run(
            [
                f"Violación detectada: {violation.model_dump_json()}",
                BinaryContent(data=jpeg_bytes, media_type="image/jpeg"),
            ]
        )
        extraction_output = extraction.output

        # 2. ContextAnalyzer — analiza riesgo contextual
        logger.debug(
            "Violation %d: running ContextAnalyzerAgent",
            violation.persona_id,
        )
        analysis = await self._analyzer.run(
            [
                f"Violación: {violation.model_dump_json()}\n"
                f"Extracción: {extraction_output.model_dump_json()}",
                BinaryContent(data=jpeg_bytes, media_type="image/jpeg"),
            ]
        )
        analysis_output = analysis.output

        # 3. Scorer → AgentDecision (auto-validado por PydanticAI)
        logger.debug(
            "Violation %d: running ScorerAgent",
            violation.persona_id,
        )
        result = await self._scorer.run(
            [
                f"Violación: {violation.model_dump_json()}\n"
                f"Extracción: {extraction_output.model_dump_json()}\n"
                f"Análisis: {analysis_output.model_dump_json()}",
                BinaryContent(data=jpeg_bytes, media_type="image/jpeg"),
            ]
        )
        decision = result.output

        logger.info(
            "Violation %d processed: tipo=%s, riesgo=%s, confianza=%.2f",
            violation.persona_id,
            decision.tipo_violacion,
            decision.nivel_riesgo,
            decision.confianza,
        )
        return decision

    async def _process_single_pass(
        self,
        jpeg_bytes: bytes,
        violation: Violation,
    ) -> AgentDecision:
        """Fallback: SinglePassAgent con output AgentDecision directo.

        Args:
            jpeg_bytes: Frame codificado en JPEG.
            violation: Violación a procesar.

        Returns:
            AgentDecision validado por PydanticAI.
        """
        logger.debug(
            "Violation %d: running SinglePassAgent (fallback)",
            violation.persona_id,
        )
        result = await self._single_pass.run(
            [
                f"Violación detectada: {violation.model_dump_json()}",
                BinaryContent(data=jpeg_bytes, media_type="image/jpeg"),
            ]
        )
        decision = result.output

        logger.info(
            "Violation %d (single-pass): tipo=%s, riesgo=%s, confianza=%.2f",
            violation.persona_id,
            decision.tipo_violacion,
            decision.nivel_riesgo,
            decision.confianza,
        )
        return decision

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Redimensiona frame a configured size con INTER_AREA.

        ADR-016: resize 640×480 antes de VLM para reducir latencia
        y consumo de tokens. INTER_AREA es óptimo para downscaling.

        Args:
            frame: Frame original (HWC BGR uint8).

        Returns:
            Frame redimensionado a (resize_width × resize_height).
        """
        return cv.resize(
            frame,
            (self._resize_width, self._resize_height),
            interpolation=cv.INTER_AREA,
        )

    def _encode_jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """Redimensiona y codifica frame como JPEG.

        Combina resize + imencode en un solo paso.

        Args:
            frame: Frame original (HWC BGR uint8).
            quality: Calidad JPEG (1-100, default 85).

        Returns:
            JPEG bytes para BinaryContent.
        """
        resized = self._resize_frame(frame)
        _, buffer = cv.imencode(
            ".jpg", resized, [cv.IMWRITE_JPEG_QUALITY, quality]
        )
        return buffer.tobytes()

    def _default_decision(self, violation: Violation) -> AgentDecision:
        """Decisión por defecto cuando todo falla.

        Args:
            violation: Violación que no pudo procesarse.

        Returns:
            AgentDecision con requiere_revision_humana=True.
        """
        from datetime import datetime

        return AgentDecision(
            zona=violation.zona_id,
            tipo_violacion="EPP_FALTANTE",
            epp_faltante=violation.epp_faltantes,
            nivel_riesgo="BAJO",
            timestamp=datetime.now(),
            confianza=0.0,
            requiere_revision_humana=True,
            razonamiento_vlm="Fallback: VLM no disponible después de reintentos",
        )


__all__ = ["MuninOrchestrator"]
