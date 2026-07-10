from __future__ import annotations

"""Agent layer — agentes VLM con PydanticAI Agents.

Agentes especializados en análisis de violaciones de EPP:
- AgentContext: contexto compartido entre agentes (BaseModel)
- ExtractionResult: output validado del ExtractorAgent
- AnalysisResult: output validado del ContextAnalyzerAgent
- AgentDecision: output validado del ScorerAgent (gate.schemas)
- MuninOrchestrator: coordinación secuencial con fallback
"""

from munin.agents.base import AgentContext
from munin.agents.context_analyzer import AnalysisResult, create_context_analyzer_agent
from munin.agents.extractor import ExtractionResult, create_extractor_agent
from munin.agents.orchestrator import MuninOrchestrator
from munin.agents.scorer import create_scorer_agent
from munin.agents.single_pass import create_single_pass_agent

__all__ = [
    "AgentContext",
    "AnalysisResult",
    "ExtractionResult",
    "MuninOrchestrator",
    "create_context_analyzer_agent",
    "create_extractor_agent",
    "create_scorer_agent",
    "create_single_pass_agent",
]
