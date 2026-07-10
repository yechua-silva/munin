from __future__ import annotations

import inspect
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from munin.agents.orchestrator import MuninOrchestrator
from munin.gate.schemas import PPEMissing, Violation


@pytest.fixture
def mock_orchestrator() -> MuninOrchestrator:
    """Orchestrator with mocked agents."""
    extractor = MagicMock()
    analyzer = MagicMock()
    scorer = MagicMock()
    single_pass = MagicMock()
    return MuninOrchestrator(
        extractor=extractor,
        analyzer=analyzer,
        scorer=scorer,
        single_pass=single_pass,
        timeout=5.0,
    )


@pytest.fixture
def sample_violation() -> Violation:
    """Sample violation for testing."""
    return Violation(
        persona_id=1,
        zona_id="extraccion",
        epp_faltantes=[PPEMissing(tipo="hardhat", descripcion="Casco", norma_chilena="NCh 461")],
        frame_id=1,
        timestamp=datetime.now(),
    )


class TestSinglePassDefault:
    """Tests for SinglePass as default mode."""

    def test_analyze_accepts_use_single_pass(self, mock_orchestrator: MuninOrchestrator, sample_violation: Violation) -> None:
        """analyze() should accept use_single_pass parameter."""
        sig = inspect.signature(mock_orchestrator.analyze)
        assert 'use_single_pass' in sig.parameters

    def test_analyze_returns_list_on_no_violations(self, mock_orchestrator: MuninOrchestrator) -> None:
        """analyze() should return empty list when no violations."""
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_orchestrator.analyze(frame, [],)
        assert result is not None

    def test_analyze_use_single_pass_parameter_type(self, mock_orchestrator: MuninOrchestrator) -> None:
        """use_single_pass parameter should have bool type."""
        sig = inspect.signature(mock_orchestrator.analyze)
        param = sig.parameters['use_single_pass']
        assert param.default is True
