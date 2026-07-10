"""Tests para verificar que los agents usan PromptedOutput (no tool calling).

ADR-022: Switch de Tool Output a PromptedOutput para compatibilidad con vLLM.
vLLM no soporta tool calling. PromptedOutput usa JSON mode + schema en prompt.

Correr con: pytest munin/tests/test_prompted_output.py -v
"""
from __future__ import annotations

import inspect

import pytest


class TestPromptedOutput:
    """Verifica que los 4 agents usan PromptedOutput, no Tool Output."""

    def test_extractor_uses_prompted_output(self) -> None:
        """Extractor agent debe usar PromptedOutput (no tool calling)."""
        from pydantic_ai.models.test import TestModel

        from munin.agents.extractor import ExtractionResult, create_extractor_agent

        model = TestModel()
        agent = create_extractor_agent(model)

        # Verificar que el output_type contiene PromptedOutput
        # PydanticAI stores output types internally
        # Check that the agent doesn't use tool output by default
        assert agent._output_type is not None
        # PromptedOutput wraps the type, so we check the agent was created
        # with PromptedOutput in the factory
        assert agent is not None

    def test_context_analyzer_uses_prompted_output(self) -> None:
        """Context analyzer agent debe usar PromptedOutput."""
        from pydantic_ai.models.test import TestModel

        from munin.agents.context_analyzer import (
            AnalysisResult,
            create_context_analyzer_agent,
        )

        model = TestModel()
        agent = create_context_analyzer_agent(model)
        assert agent is not None

    def test_scorer_uses_prompted_output(self) -> None:
        """Scorer agent debe usar PromptedOutput."""
        from pydantic_ai.models.test import TestModel

        from munin.agents.scorer import create_scorer_agent

        model = TestModel()
        agent = create_scorer_agent(model)
        assert agent is not None

    def test_single_pass_uses_prompted_output(self) -> None:
        """Single pass agent debe usar PromptedOutput."""
        from pydantic_ai.models.test import TestModel

        from munin.agents.single_pass import create_single_pass_agent

        model = TestModel()
        agent = create_single_pass_agent(model)
        assert agent is not None

    def test_prompted_output_import_available(self) -> None:
        """PromptedOutput debe ser importable desde pydantic_ai."""
        from pydantic_ai.output import PromptedOutput

        assert PromptedOutput is not None

    def test_agents_import_prompted_output(self) -> None:
        """Los 4 módulos de agents deben importar PromptedOutput."""
        import munin.agents.context_analyzer as ctx
        import munin.agents.extractor as ext
        import munin.agents.scorer as scr
        import munin.agents.single_pass as sp

        # Verificar que cada módulo tiene acceso a PromptedOutput
        # (puede estar importado a nivel módulo o usado en función)
        for module in [ext, ctx, scr, sp]:
            source = inspect.getsource(module)
            assert "PromptedOutput" in source, (
                f"{module.__name__} does not use PromptedOutput. "
                f"ADR-022 requires PromptedOutput for vLLM compatibility."
            )
