"""Tests para MuninOrchestrator (post-migración PydanticAI).

Verifica:
- test_munin_orchestrator_no_violations_returns_empty
- test_agent_context_creation
- test_default_decision
- test_import_from_model

Los tests con VLM real se hacen en smoke test manual.
Usamos TestModel de pydantic_ai (no requiere API key ni conexión).

Correr con: pytest munin/tests/test_orchestrator.py -v
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from pydantic_ai.models.test import TestModel

from munin.agents.base import AgentContext
from munin.agents.context_analyzer import create_context_analyzer_agent
from munin.agents.extractor import create_extractor_agent
from munin.agents.orchestrator import MuninOrchestrator
from munin.agents.scorer import create_scorer_agent
from munin.agents.single_pass import create_single_pass_agent
from munin.gate.schemas import AgentDecision, PPEMissing, Violation

# ============================================================================
# HELPERS
# ============================================================================

# Frame dummy para pruebas (1x1 pixel BGR)
DUMMY_FRAME: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)


def _make_violation(
    zona_id: str = "extraccion",
    persona_id: int = 1,
) -> Violation:
    """Crea una violación de prueba."""
    return Violation(
        persona_id=persona_id,
        zona_id=zona_id,
        epp_faltantes=[
            PPEMissing(
                tipo="hardhat",
                descripcion="Casco de seguridad",
                norma_chilena="NCh 1411",
            ),
        ],
        frame_id=10,
    )


@pytest.fixture
def model() -> TestModel:
    """Fixture: TestModel de pydantic_ai (no requiere API key)."""
    return TestModel()


@pytest.fixture
def orchestrator(model: TestModel) -> MuninOrchestrator:
    """Fixture: MuninOrchestrator con TestModel (sin VLM real)."""
    return MuninOrchestrator(
        extractor=create_extractor_agent(model),
        analyzer=create_context_analyzer_agent(model),
        scorer=create_scorer_agent(model),
        single_pass=create_single_pass_agent(model),
        timeout=30.0,
    )


@pytest.fixture
def violation() -> Violation:
    """Fixture: violación de prueba."""
    return _make_violation()


# ============================================================================
# TESTS
# ============================================================================


class TestMuninOrchestrator:
    """Suite de tests para MuninOrchestrator."""

    @pytest.mark.asyncio
    async def test_no_violations_returns_empty(
        self,
        orchestrator: MuninOrchestrator,
    ) -> None:
        """0 violaciones → lista vacía (no llama VLM)."""
        # Act
        decisions = await orchestrator.analyze(DUMMY_FRAME, [])

        # Assert
        assert decisions == [], (
            f"Se esperaba lista vacía para 0 violaciones, "
            f"pero se obtuvo {len(decisions)} decisiones"
        )

    def test_default_decision(self, violation: Violation) -> None:
        """_default_decision retorna AgentDecision con requiere_revision_humana=True."""
        from pydantic_ai.models.test import TestModel
        from munin.agents.extractor import create_extractor_agent
        from munin.agents.context_analyzer import create_context_analyzer_agent
        from munin.agents.scorer import create_scorer_agent
        from munin.agents.single_pass import create_single_pass_agent

        model = TestModel()
        orch = MuninOrchestrator(
            extractor=create_extractor_agent(model),
            analyzer=create_context_analyzer_agent(model),
            scorer=create_scorer_agent(model),
            single_pass=create_single_pass_agent(model),
        )

        # Act
        decision = orch._default_decision(violation)

        # Assert
        assert isinstance(decision, AgentDecision), (
            f"Se esperaba AgentDecision, pero se obtuvo {type(decision)}"
        )
        assert decision.requiere_revision_humana is True, (
            "Default decision debe requerir revision humana"
        )
        assert decision.confianza == 0.0, (
            f"Default decision debe tener confianza 0.0, "
            f"pero tiene {decision.confianza}"
        )
        assert decision.tipo_violacion == "EPP_FALTANTE"
        assert decision.nivel_riesgo == "BAJO"
        assert decision.zona == violation.zona_id
        assert len(decision.epp_faltante) == 1
        assert decision.epp_faltante[0].tipo == "hardhat"
        assert decision.razonamiento_vlm is not None
        assert "Fallback" in decision.razonamiento_vlm

    def test_from_model_class_method(self) -> None:
        """from_model crea un MuninOrchestrator con todos los agentes."""
        model = TestModel()

        # Act
        orch = MuninOrchestrator.from_model(model, timeout=15.0)

        # Assert
        assert isinstance(orch, MuninOrchestrator)
        # Verify all agents were created (they are not None)
        assert orch._extractor is not None
        assert orch._analyzer is not None
        assert orch._scorer is not None
        assert orch._single_pass is not None
        assert orch._timeout == 15.0


class TestAgentContext:
    """Suite de tests para AgentContext."""

    def test_creation_with_violation(self) -> None:
        """AgentContext se crea correctamente con una Violation."""
        # Arrange
        violation = _make_violation()

        # Act
        ctx = AgentContext(violation=violation, frame_id=42)

        # Assert
        assert isinstance(ctx, AgentContext)
        assert ctx.violation.persona_id == 1
        assert ctx.violation.zona_id == "extraccion"
        assert len(ctx.violation.epp_faltantes) == 1
        assert ctx.violation.epp_faltantes[0].tipo == "hardhat"
        assert ctx.frame_id == 42
        assert ctx.extractor_output is None
        assert ctx.analysis_output is None

    def test_creation_with_minimal_violation(self) -> None:
        """AgentContext funciona con Violation mínima."""
        violation = Violation(
            persona_id=2,
            zona_id="procesamiento",
            epp_faltantes=[],
            frame_id=5,
        )
        ctx = AgentContext(violation=violation, frame_id=5)

        assert ctx.violation.persona_id == 2
        assert ctx.violation.zona_id == "procesamiento"
        assert ctx.violation.epp_faltantes == []
        assert ctx.frame_id == 5

    def test_serialization(self) -> None:
        """AgentContext se serializa/deserializa correctamente."""
        violation = _make_violation()
        ctx = AgentContext(
            violation=violation,
            extractor_output='{"zona": "extraccion"}',
            analysis_output='{"riesgo": "ALTO"}',
            frame_id=10,
        )

        # Serialize to dict
        data = ctx.model_dump()
        assert data["violation"]["persona_id"] == 1
        assert data["extractor_output"] == '{"zona": "extraccion"}'
        assert data["analysis_output"] == '{"riesgo": "ALTO"}'
        assert data["frame_id"] == 10

        # Deserialize back
        restored = AgentContext.model_validate(data)
        assert restored.frame_id == 10
        assert restored.extractor_output == '{"zona": "extraccion"}'
        assert restored.violation.persona_id == 1


class TestImports:
    """Verifica que los nuevos módulos importen correctamente."""

    def test_import_munin_orchestrator(self) -> None:
        """MuninOrchestrator importa desde orchestrator.py."""
        from munin.agents.orchestrator import MuninOrchestrator
        assert MuninOrchestrator is not None

    def test_import_agent_context(self) -> None:
        """AgentContext importa desde base.py."""
        from munin.agents.base import AgentContext
        assert AgentContext is not None

    def test_import_create_extractor_agent(self) -> None:
        """create_extractor_agent importa desde extractor.py."""
        from munin.agents.extractor import (
            create_extractor_agent,
            ExtractionResult,
        )
        assert create_extractor_agent is not None
        assert ExtractionResult is not None

    def test_import_create_scorer_agent(self) -> None:
        """create_scorer_agent importa desde scorer.py."""
        from munin.agents.scorer import create_scorer_agent
        assert create_scorer_agent is not None

    def test_import_create_context_analyzer_agent(self) -> None:
        """create_context_analyzer_agent importa desde context_analyzer.py."""
        from munin.agents.context_analyzer import (
            create_context_analyzer_agent,
            AnalysisResult,
        )
        assert create_context_analyzer_agent is not None
        assert AnalysisResult is not None

    def test_import_create_single_pass_agent(self) -> None:
        """create_single_pass_agent importa desde single_pass.py."""
        from munin.agents.single_pass import create_single_pass_agent
        assert create_single_pass_agent is not None

    def test_import_vlm_model_factory(self) -> None:
        """VLMModelFactory importa desde factory.py."""
        from munin.vlm.factory import VLMModelFactory
        assert VLMModelFactory is not None
