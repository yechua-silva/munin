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
# FIXTURES V3 (nueva signature + dual-class mode)
# ============================================================================


@pytest.fixture
def detections_with_epp() -> list:
    """Detecciones con 1 persona + hardhat dentro de su bbox."""
    from munin.gate.schemas import DetectionResult

    return [
        DetectionResult(class_name="person", bbox=(100.0, 200.0, 300.0, 500.0), confidence=0.9),
        DetectionResult(class_name="hardhat", bbox=(150.0, 180.0, 250.0, 250.0), confidence=0.85),
    ]


@pytest.fixture
def detections_with_negative_classes() -> list:
    """Detecciones con no_helmet y no_vest."""
    from munin.gate.schemas import DetectionResult

    return [
        DetectionResult(class_name="person", bbox=(100.0, 200.0, 300.0, 500.0), confidence=0.9),
        DetectionResult(class_name="no_helmet", bbox=(150.0, 180.0, 250.0, 250.0), confidence=0.80),
        DetectionResult(class_name="no_vest", bbox=(120.0, 260.0, 280.0, 350.0), confidence=0.75),
    ]


@pytest.fixture
def detections_no_ppe() -> list:
    """Solo detección de persona, sin EPP."""
    from munin.gate.schemas import DetectionResult

    return [
        DetectionResult(class_name="person", bbox=(100.0, 200.0, 300.0, 500.0), confidence=0.9),
    ]


@pytest.fixture
def person_no_epp() -> TrackedPerson:
    """Persona sin EPP detectado (epp_detectado vacío)."""
    return TrackedPerson(
        persona_id=1,
        bbox=(100.0, 200.0, 300.0, 500.0),
        epp_detectado=set(),
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
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )
        persons = [person_without_hardhat]

        violations = checker.check(persons, detections=[], zone=extraccion_zone)

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
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )
        persons = [person_without_harness]

        violations = checker.check(persons, detections=[], zone=alto_zone)

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
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )
        persons = [person_with_full_epp]

        violations = checker.check(persons, detections=[], zone=extraccion_zone)

        assert len(violations) == 0

    def test_empty_persons_no_violation(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
    ) -> None:
        """Sin personas trackeadas → lista vacía (no error)."""
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )

        violations = checker.check([], detections=[], zone=extraccion_zone)

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
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )
        persons = [person_without_hardhat, person_with_full_epp]

        violations = checker.check(persons, detections=[], zone=extraccion_zone)

        assert len(violations) == 1
        assert violations[0].persona_id == 1

    def test_multiple_missing_epp_in_extraccion(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
    ) -> None:
        """Persona sin múltiples EPP en extracción → todas las faltantes."""
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

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
            mode=ComplianceMode.LEGACY,
        )

        violations = checker.check([person], detections=[], zone=extraccion_zone)

        assert len(violations) == 1
        violation = violations[0]
        epp_faltante_tipos = {e.tipo for e in violation.epp_faltantes}

        # required_epp para extraccion: hardhat, safety_vest, safety_glasses,
        # gloves, safety_boots
        # person tiene: gloves, safety_glasses
        # faltan: hardhat, safety_vest, safety_boots
        assert epp_faltante_tipos == {"hardhat", "safety_vest", "safety_boots"}
        assert len(violation.epp_faltantes) == 3


class TestPPEComplianceCheckerV3:
    """Tests v3 para nueva signature + dual_class mode."""

    def test_check_new_signature_three_params(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_no_epp: TrackedPerson,
        detections_no_ppe: list,
    ) -> None:
        """check() acepta 3 parámetros: persons, detections, zone."""
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )
        # No debe lanzar error con 3 params
        violations = checker.check(
            [person_no_epp],
            detections=detections_no_ppe,
            zone=extraccion_zone,
        )
        assert isinstance(violations, list)

    def test_assign_epp_to_persons(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_no_epp: TrackedPerson,
        detections_with_epp: list,
    ) -> None:
        """_assign_epp_to_persons asigna EPP dentro de bbox de persona."""
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.LEGACY,
        )
        persons = [person_no_epp]
        checker.check(persons, detections=detections_with_epp, zone=extraccion_zone)

        # Después de check, person_no_epp debe tener hardhat en epp_detectado
        # porque el hardhat bbox está dentro del person bbox
        assert "hardhat" in persons[0].epp_detectado

    def test_dual_class_negative_detection(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_no_epp: TrackedPerson,
        detections_with_negative_classes: list,
    ) -> None:
        """DUAL_CLASS: no_helmet detectado → violation hardhat."""
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.DUAL_CLASS,
        )
        violations = checker.check(
            [person_no_epp],
            detections=detections_with_negative_classes,
            zone=extraccion_zone,
        )

        assert len(violations) >= 1
        # no_helmet → hardhat faltante
        all_faltantes = set()
        for v in violations:
            for e in v.epp_faltantes:
                all_faltantes.add(e.tipo)
        assert "hardhat" in all_faltantes

    def test_dual_class_no_negative_no_positive(
        self,
        mock_zone_config: Mock,
        mock_ds132_kb: Mock,
        extraccion_zone: Zone,
        person_no_epp: TrackedPerson,
        detections_no_ppe: list,
    ) -> None:
        """DUAL_CLASS: sin detección negativa ni positiva → violation legacy."""
        from munin.pipeline.ppe_checker import ComplianceMode, PPEComplianceChecker

        checker = PPEComplianceChecker(
            zone_config=mock_zone_config,
            ds132_kb=mock_ds132_kb,
            mode=ComplianceMode.DUAL_CLASS,
        )
        violations = checker.check(
            [person_no_epp],
            detections=detections_no_ppe,
            zone=extraccion_zone,
        )

        # Sin EPP detectado ni clases negativas → violación por todos los required
        assert len(violations) == 1
        epp_faltante_tipos = {e.tipo for e in violations[0].epp_faltantes}
        assert "hardhat" in epp_faltante_tipos
