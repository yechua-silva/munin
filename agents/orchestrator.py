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
    """

    def __init__(
        self,
        extractor: Agent[None, ExtractionResult],
        analyzer: Agent[None, AnalysisResult],
        scorer: Agent[None, AgentDecision],
        single_pass: Agent[None, AgentDecision],
        timeout: float = 30.0,
    ) -> None:
        """Inicializa el orquestador con los 4 agentes.

        Args:
            extractor: Agente de extracción visual.
            analyzer: Agente de análisis contextual.
            scorer: Agente de scoring DS 132.
            single_pass: Agente de fallback (prompt fusionado).
            timeout: Timeout por violación en segundos.
        """
        self._extractor = extractor
        self._analyzer = analyzer
        self._scorer = scorer
        self._single_pass = single_pass
        self._timeout = timeout

    @classmethod
    def from_model(
        cls,
        model: OpenAIChatModel,
        timeout: float = 30.0,
    ) -> MuninOrchestrator:
        """Crea el orquestador con todos los agentes desde un modelo.

        Args:
            model: Modelo VLM configurado.
            timeout: Timeout por violación.

        Returns:
            MuninOrchestrator listo para usar.
        """
        return cls(
            extractor=create_extractor_agent(model),
            analyzer=create_context_analyzer_agent(model),
            scorer=create_scorer_agent(model),
            single_pass=create_single_pass_agent(model),
            timeout=timeout,
        )

    async def analyze(
        self,
        frame: np.ndarray,
        violations: list[Violation],
    ) -> list[AgentDecision]:
        """Analiza violaciones con 3 agentes secuenciales.

        Por cada violación ejecuta el pipeline de 3 agentes
        (Extractor → ContextAnalyzer → Scorer). Si falla,
        ejecuta SinglePassAgent como fallback por violación.

        Args:
            frame: Frame del video (HWC, BGR, uint8).
            violations: Violaciones detectadas por rule engine.

        Returns:
            Lista de AgentDecision validados.

        Raises:
            VLMError: Si todos los métodos fallan para alguna violación.
        """
        if not violations:
            logger.info("No violations to analyze, returning empty list")
            return []

        logger.info(
            "Analyzing %d violation(s) with sequential 3-agent pipeline",
            len(violations),
        )

        # Codificar frame a JPEG bytes para BinaryContent
        _, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = buffer.tobytes()

        decisions: list[AgentDecision] = []
        for violation in violations:
            try:
                decision = await asyncio.wait_for(
                    self._process_violation(jpeg_bytes, violation),
                    timeout=self._timeout,
                )
                decisions.append(decision)
            except (TimeoutError, asyncio.TimeoutError) as e:
                logger.warning(
                    "Sequential analysis timed out for violation %s: %s. "
                    "Falling back to single-pass.",
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
                        "Single-pass also failed for violation %s: %s. "
                        "Using default decision.",
                        violation.persona_id,
                        e2,
                    )
                    decisions.append(self._default_decision(violation))
            except Exception as e:
                logger.warning(
                    "Sequential analysis failed for violation %s: %s. "
                    "Falling back to single-pass.",
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
                        "Single-pass also failed for violation %s: %s. "
                        "Using default decision.",
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
