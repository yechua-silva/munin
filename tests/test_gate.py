"""Tests para PydanticGate (TASK-17).

Verifica que el gate valide correctamente el output JSON del VLM
contra el schema AgentDecision, con retry loop y errores controlados.

Correr con: pytest munin/tests/test_gate.py -v
"""
from __future__ import annotations

import json

import pytest

from munin.exceptions import GateValidationError
from munin.gate.schemas import AgentDecision


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_decision_json() -> str:
    """JSON válido que cumple AgentDecision schema."""
    return json.dumps({
        "zona": "extraccion",
        "tipo_violacion": "EPP_FALTANTE",
        "epp_faltante": [
            {
                "tipo": "hardhat",
                "descripcion": "Casco de seguridad",
                "norma_chilena": "NCh 1411",
            }
        ],
        "nivel_riesgo": "CRITICO",
        "timestamp": "2026-07-07T14:30:00",
        "articulo_ds132": "Art. 38",
        "confianza": 0.92,
        "requiere_revision_humana": False,
        "razonamiento_vlm": "Persona sin casco en zona de extracción.",
    })


@pytest.fixture
def critical_violation_json() -> str:
    """JSON con violación CRITICAL y múltiples EPP faltantes."""
    return json.dumps({
        "zona": "extraccion",
        "tipo_violacion": "EPP_FALTANTE",
        "epp_faltante": [
            {
                "tipo": "hardhat",
                "descripcion": "Casco de seguridad",
                "norma_chilena": "NCh 1411",
            },
            {
                "tipo": "safety_vest",
                "descripcion": "Chaleco reflectante",
                "norma_chilena": "NCh 461",
            },
        ],
        "nivel_riesgo": "CRITICO",
        "timestamp": "2026-07-07T14:30:05",
        "articulo_ds132": "Art. 45",
        "confianza": 0.85,
        "requiere_revision_humana": True,
        "razonamiento_vlm": "Persona sin casco ni chaleco en zona de extracción.",
    })


# ============================================================================
# TESTS
# ============================================================================


class TestPydanticGate:
    """Tests para PydanticGate.validate()."""

    def test_validate_valid_json(self, valid_decision_json: str) -> None:
        """JSON válido debe retornar un AgentDecision correctamente poblado."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=3)
        decision = gate.validate(valid_decision_json)

        assert isinstance(decision, AgentDecision)
        assert decision.zona == "extraccion"
        assert decision.tipo_violacion == "EPP_FALTANTE"
        assert decision.nivel_riesgo == "CRITICO"
        assert decision.confianza == 0.92
        assert decision.articulo_ds132 == "Art. 38"
        assert len(decision.epp_faltante) == 1
        assert decision.epp_faltante[0].tipo == "hardhat"
        assert decision.requiere_revision_humana is False

    def test_validate_invalid_json_raises(self) -> None:
        """JSON malformado debe lanzar GateValidationError después de retries."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=2)
        invalid_json = "{zona: extraccion,}"

        with pytest.raises(GateValidationError) as excinfo:
            gate.validate(invalid_json)

        assert "Failed to validate" in str(excinfo.value)

    def test_validate_missing_field_raises(self) -> None:
        """JSON sin campo requerido (confianza) debe lanzar GateValidationError."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=2)
        # Falta el campo 'confianza' que es requerido
        missing_field_json = json.dumps({
            "zona": "extraccion",
            "tipo_violacion": "EPP_FALTANTE",
            "epp_faltante": [],
            "nivel_riesgo": "ALTO",
            "timestamp": "2026-07-07T14:30:00",
        })

        with pytest.raises(GateValidationError) as excinfo:
            gate.validate(missing_field_json)

        assert "Failed to validate" in str(excinfo.value)

    def test_validate_empty_string_raises(self) -> None:
        """String vacío debe lanzar GateValidationError."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=2)

        with pytest.raises(GateValidationError) as excinfo:
            gate.validate("")

        assert "Failed to validate" in str(excinfo.value)

    def test_gate_with_valid_violation(
        self, critical_violation_json: str
    ) -> None:
        """JSON con violación CRITICAL debe producir AgentDecision correcto."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=3)
        decision = gate.validate(critical_violation_json)

        assert isinstance(decision, AgentDecision)
        assert decision.zona == "extraccion"
        assert decision.tipo_violacion == "EPP_FALTANTE"
        assert decision.nivel_riesgo == "CRITICO"
        assert decision.confianza == 0.85
        assert decision.articulo_ds132 == "Art. 45"
        assert len(decision.epp_faltante) == 2
        assert decision.epp_faltante[0].tipo == "hardhat"
        assert decision.epp_faltante[1].tipo == "safety_vest"
        assert decision.requiere_revision_humana is True
        assert decision.razonamiento_vlm is not None

    def test_validate_max_retries_exhausted_logs_warnings(self) -> None:
        """Verifica que el gate intente exactamente max_retries veces."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=3)

        with pytest.raises(GateValidationError):
            gate.validate("not even close to json")

    def test_invalid_tipo_violacion_raises(self) -> None:
        """JSON con tipo_violacion inválido debe lanzar GateValidationError."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=2)
        invalid_violation = json.dumps({
            "zona": "extraccion",
            "tipo_violacion": "INVALID_TYPE",
            "epp_faltante": [],
            "nivel_riesgo": "ALTO",
            "timestamp": "2026-07-07T14:30:00",
            "confianza": 0.9,
        })

        with pytest.raises(GateValidationError):
            gate.validate(invalid_violation)

    def test_invalid_nivel_riesgo_raises(self) -> None:
        """JSON con nivel_riesgo inválido debe lanzar GateValidationError."""
        from munin.gate.validator import PydanticGate

        gate = PydanticGate(max_retries=2)
        invalid_riesgo = json.dumps({
            "zona": "extraccion",
            "tipo_violacion": "SIN_VIOLACION",
            "epp_faltante": [],
            "nivel_riesgo": "SUPREMO",
            "timestamp": "2026-07-07T14:30:00",
            "confianza": 0.9,
        })

        with pytest.raises(GateValidationError):
            gate.validate(invalid_riesgo)
