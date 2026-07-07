"""Tests para PPEComplianceChecker (TASK-16).

Verifica que el rule engine detecte correctamente violaciones de EPP
por zona, incluyendo la regla especial de arnés (ADR-008).

Correr con: pytest munin/tests/test_ppe_checker.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from munin.config import Zone
from munin.gate.schemas import PPEMissing, TrackedPerson, Violation


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_zone_config() -> Mock:
    """Mock de ZoneConfig."""
    return MagicMock()


@pytest.fixture
def mock_ds132_kb() -> Mock:
    """Mock de DS132KnowledgeBase."""
    kb = MagicMock()
    return kb


@pytest.fixture
def extraccion_zone() -> Zone:
    """Zona de extracción (riesgo_base='critico')."""
    return Zone(
        zone_id="extraccion",
        nombre="Zona de Extracción",
        required_epp=["hardhat", "safety_vest", "safety_glasses",
                       "gloves", "safety_boots"],
        riesgo_base="critico",
        min_confidence=0.7,
        articulos_ds132=["Art. 38", "Art. 42", "Art. 45"],
    )


@pytest.fixture
def alto_zone() -> Zone:
    """Zona con riesgo_base='alto' (para regla arnés ADR-008)."""
    return Zone(
        zone_id="procesamiento",
        nombre="Zona de Procesamiento",
        required_epp=["hardhat", "safety_vest", "safety_glasses",
                       "gloves", "safety_boots"],
        riesgo_base="alto",
        min_confidence=0.6,
        articulos_ds132=["Art. 38", "Art. 45"],
    )


@pytest.fixture
def person_without_hardhat() -> TrackedPerson:
    """Persona sin hardhat en epp_detectado."""
    return TrackedPerson(
        persona_id=1,
        bbox=(100.0, 200.0, 300.0, 500.0),
        epp_detectado={"safety_vest", "gloves", "safety_glasses",
                       "safety_boots"},
        lost_counter=0,
        consecutive_violations=0,
    )


@pytest.fixture
def person_without_harness() -> TrackedPerson:
    """Persona sin harness en epp_detectado (para regla arnés)."""
    return TrackedPerson(
        persona_id=2,
        bbox=(400.0, 200.0, 600.0, 500.0),
        epp_detectado={"hardhat", "safety_vest", "gloves",
                       "safety_glasses", "safety_boots"},
        lost_counter=0,
        consecutive_violations=0,
    )


@pytest.fixture
def person_with_full_epp() -> TrackedPerson:
    """Persona con todos los EPP requeridos."""
    return TrackedPerson(
        persona_id=3,
        bbox=(200.0, 300.0, 400.0, 600.0),
        epp_detectado={"hardhat", "safety_vest", "gloves",
                       "safety_glasses", "safety_boots", "harness"},
        lost_counter=0,
        consecutive_violations=0,
    )


# ============================================================================
# TESTS
# ============================================================================


class TestPPEComplianceChecker:
    """Tests para PPEComplianceChecker.check()."""

    def test_person_without_hardhat_in_extraccion_violation(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_without_hardhat: TrackedPerson,
    ) -> None:
        """Persona sin hardhat en zona extracción → violación con hardhat faltante."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
        )
        persons = [person_without_hardhat]

        violations = checker.check(persons, extraccion_zone)

        assert len(violations) == 1
        violation = violations[0]
        assert violation.persona_id == 1
        assert violation.zona_id == "extraccion"

        # Debe tener exactamente 1 EPP faltante (hardhat)
        # safety_vest, gloves, safety_glasses, safety_boots están presentes
        epp_faltante_tipos = {e.tipo for e in violation.epp_faltantes}
        assert epp_faltante_tipos == {"hardhat"}

        # Verificar detalles del PPEMissing
        hardhat_missing = next(
            e for e in violation.epp_faltantes if e.tipo == "hardhat"
        )
        assert hardhat_missing.descripcion == "Casco de seguridad"
        assert hardhat_missing.norma_chilena == "NCh 1411"

    def test_person_without_harness_in_alto_zone_violation(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        alto_zone: Zone,
        person_without_harness: TrackedPerson,
    ) -> None:
        """Persona sin harness en zona riesgo_base='alto' → violación arnés (ADR-008)."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
        )
        persons = [person_without_harness]

        violations = checker.check(persons, alto_zone)

        assert len(violations) == 1
        violation = violations[0]
        assert violation.persona_id == 2
        assert violation.zona_id == "procesamiento"

        # La persona tiene todos los EPP requeridos EXCEPTO harness
        # que se agrega por la regla de arnés (riesgo_base='alto')
        epp_faltante_tipos = {e.tipo for e in violation.epp_faltantes}
        assert "harness" in epp_faltante_tipos

        # Verificar detalle del arnés
        harness_missing = next(
            e for e in violation.epp_faltantes if e.tipo == "harness"
        )
        assert harness_missing.descripcion == "Arnés de cuerpo"
        assert harness_missing.norma_chilena == "NCh 1411"

    def test_person_with_full_epp_no_violation(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_with_full_epp: TrackedPerson,
    ) -> None:
        """Persona con todo el EPP requerido → lista vacía (compliance OK)."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
        )
        persons = [person_with_full_epp]

        violations = checker.check(persons, extraccion_zone)

        assert len(violations) == 0

    def test_empty_persons_no_violation(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
    ) -> None:
        """Sin personas trackeadas → lista vacía (no error)."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
        )

        violations = checker.check([], extraccion_zone)

        assert len(violations) == 0

    def test_multiple_persons_one_violation(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_without_hardhat: TrackedPerson,
        person_with_full_epp: TrackedPerson,
    ) -> None:
        """Múltiples personas: una sin hardhat, otra con todo → 1 violación."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
        )
        persons = [person_without_hardhat, person_with_full_epp]

        violations = checker.check(persons, extraccion_zone)

        assert len(violations) == 1
        assert violations[0].persona_id == 1

    def test_multiple_missing_epp_in_extraccion(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
    ) -> None:
        """Persona sin múltiples EPP en extracción → todas las faltantes."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker

        # Persona sin hardhat, safety_vest, ni safety_boots
        person = TrackedPerson(
            persona_id=4,
            bbox=(100.0, 100.0, 300.0, 400.0),
            epp_detectado={"gloves", "safety_glasses"},
            lost_counter=0,
            consecutive_violations=0,
        )

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
        )

        violations = checker.check([person], extraccion_zone)

        assert len(violations) == 1
        violation = violations[0]
        epp_faltante_tipos = {e.tipo for e in violation.epp_faltantes}

        # required_epp para extraccion: hardhat, safety_vest, safety_glasses,
        # gloves, safety_boots
        # person tiene: gloves, safety_glasses
        # faltan: hardhat, safety_vest, safety_boots
        assert epp_faltante_tipos == {"hardhat", "safety_vest", "safety_boots"}
        assert len(violation.epp_faltantes) == 3
