"""Tests T14 — Schema extensions: camera_id in Violation and AgentDecision.

Verifica que:
- Violation tiene camera_id con default "default"
- Violation acepta camera_id custom
- AgentDecision tiene camera_id con default "default"
- AgentDecision acepta camera_id custom
- Backward compat: crear schemas sin camera_id no da error

Correr con: pytest tests/test_schemas_v4.py -v
"""
from __future__ import annotations

from datetime import datetime

import pytest

from munin.gate.schemas import AgentDecision, PPEMissing, Violation


class TestViolationCameraId:
    """Tests para camera_id en Violation."""

    def test_default_is_default(self) -> None:
        """Violation sin camera_id → default 'default'."""
        v = Violation(persona_id=1, zona_id="test", frame_id=0)
        assert v.camera_id == "default"

    def test_custom_camera_id(self) -> None:
        """Violation con camera_id custom."""
        v = Violation(
            persona_id=1,
            zona_id="test",
            frame_id=0,
            camera_id="cam01",
        )
        assert v.camera_id == "cam01"

    def test_backward_compat_no_error(self) -> None:
        """Crear Violation sin camera_id no da error."""
        v = Violation(persona_id=1, zona_id="zona_a", frame_id=10)
        assert isinstance(v, Violation)
        assert v.persona_id == 1

    def test_field_order_after_timestamp(self) -> None:
        """camera_id aparece después de timestamp en el modelo."""
        schema = Violation.model_fields
        field_names = list(schema.keys())
        ts_idx = field_names.index("timestamp")
        cam_idx = field_names.index("camera_id")
        assert cam_idx > ts_idx, (
            f"camera_id (idx={cam_idx}) debe estar después de "
            f"timestamp (idx={ts_idx})"
        )

    def test_multiple_violations_different_cameras(self) -> None:
        """Múltiples Violation con diferentes camera_id."""
        v1 = Violation(
            persona_id=1, zona_id="z1", frame_id=0, camera_id="cam01",
        )
        v2 = Violation(
            persona_id=2, zona_id="z2", frame_id=1, camera_id="cam02",
        )
        v3 = Violation(persona_id=3, zona_id="z3", frame_id=2)
        assert v1.camera_id == "cam01"
        assert v2.camera_id == "cam02"
        assert v3.camera_id == "default"

    def test_serialization_includes_camera_id(self) -> None:
        """model_dump incluye camera_id."""
        v = Violation(
            persona_id=1, zona_id="test", frame_id=0, camera_id="cam01",
        )
        dump = v.model_dump()
        assert "camera_id" in dump
        assert dump["camera_id"] == "cam01"


class TestAgentDecisionCameraId:
    """Tests para camera_id en AgentDecision."""

    def test_default_is_default(self) -> None:
        """AgentDecision sin camera_id → default 'default'."""
        d = AgentDecision(
            zona="test",
            tipo_violacion="SIN_VIOLACION",
            nivel_riesgo="BAJO",
            timestamp=datetime.now(),
            confianza=0.0,
        )
        assert d.camera_id == "default"

    def test_custom_camera_id(self) -> None:
        """AgentDecision con camera_id custom."""
        d = AgentDecision(
            zona="test",
            tipo_violacion="EPP_FALTANTE",
            nivel_riesgo="ALTO",
            timestamp=datetime.now(),
            confianza=0.85,
            camera_id="cam01",
        )
        assert d.camera_id == "cam01"

    def test_backward_compat_no_error(self) -> None:
        """Crear AgentDecision sin camera_id no da error."""
        d = AgentDecision(
            zona="extraccion",
            tipo_violacion="SIN_VIOLACION",
            nivel_riesgo="BAJO",
            timestamp=datetime.now(),
            confianza=0.0,
        )
        assert isinstance(d, AgentDecision)

    def test_field_order_after_razonamiento(self) -> None:
        """camera_id aparece después de razonamiento_vlm."""
        schema = AgentDecision.model_fields
        field_names = list(schema.keys())
        raz_idx = field_names.index("razonamiento_vlm")
        cam_idx = field_names.index("camera_id")
        assert cam_idx > raz_idx, (
            f"camera_id (idx={cam_idx}) debe estar después de "
            f"razonamiento_vlm (idx={raz_idx})"
        )

    def test_serialization_includes_camera_id(self) -> None:
        """model_dump incluye camera_id."""
        d = AgentDecision(
            zona="test",
            tipo_violacion="EPP_FALTANTE",
            nivel_riesgo="CRITICO",
            timestamp=datetime.now(),
            confianza=0.9,
            camera_id="cam02",
        )
        dump = d.model_dump()
        assert "camera_id" in dump
        assert dump["camera_id"] == "cam02"

    def test_full_decision_with_camera_id(self) -> None:
        """AgentDecision completo con todos los campos incluyendo camera_id."""
        d = AgentDecision(
            zona="extraccion",
            tipo_violacion="EPP_FALTANTE",
            epp_faltante=[
                PPEMissing(
                    tipo="hardhat",
                    descripcion="Casco de seguridad",
                    norma_chilena="NCh 1411",
                ),
            ],
            nivel_riesgo="CRITICO",
            timestamp=datetime.now(),
            articulo_ds132="Art. 38",
            confianza=0.95,
            requiere_revision_humana=False,
            razonamiento_vlm="Persona sin casco en zona de extracción.",
            camera_id="cam_main",
        )
        assert d.camera_id == "cam_main"
        assert d.zona == "extraccion"
        assert d.tipo_violacion == "EPP_FALTANTE"
        assert d.nivel_riesgo == "CRITICO"
        assert len(d.epp_faltante) == 1
