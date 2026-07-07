from __future__ import annotations

import logging
from datetime import datetime

from munin.config import Zone
from munin.gate.schemas import PPEMissing, TrackedPerson, Violation
from munin.knowledge.ds132_kb import DS132KnowledgeBase
from munin.knowledge.zone_config import ZoneConfig

logger = logging.getLogger(__name__)

# Mapeo de tipo de EPP → norma chilena correspondiente
EPP_NORMA: dict[str, str] = {
    "hardhat": "NCh 1411",
    "safety_vest": "NCh 461",
    "gloves": "NCh 1432",
    "safety_glasses": "NCh 1332",
    "safety_boots": "NCh 746",
    "harness": "NCh 1411",
}

# Mapeo de tipo de EPP → descripción en español
EPP_DESC: dict[str, str] = {
    "hardhat": "Casco de seguridad",
    "safety_vest": "Chaleco reflectante",
    "gloves": "Guantes de trabajo",
    "safety_glasses": "Lentes de seguridad",
    "safety_boots": "Botas de seguridad",
    "harness": "Arnés de cuerpo",
}


class PPEComplianceChecker:
    """Verificador de compliance de EPP por zona (rule engine).

    Compara el EPP requerido por la zona contra el EPP detectado
    en cada persona trackeada. Aplica reglas especiales como la
    regla de arnés en zonas de riesgo alto (ADR-008).

    Attributes:
        _zone_config: Configuración de zonas mineras.
        _ds132_kb: Knowledge base de artículos DS 132.
        _logger: Logger de la clase.
    """

    def __init__(
        self,
        zone_config: ZoneConfig,
        ds132_kb: DS132KnowledgeBase,
    ) -> None:
        """Inicializa el checker con dependencias inyectadas.

        Args:
            zone_config: Configuración de zonas con EPP requerido.
            ds132_kb: Knowledge base de artículos DS 132.
        """
        self._zone_config: ZoneConfig = zone_config
        self._ds132_kb: DS132KnowledgeBase = ds132_kb
        self._logger = logging.getLogger(self.__class__.__name__)

    def check(
        self,
        persons: list[TrackedPerson],
        zone: Zone,
    ) -> list[Violation]:
        """Verifica compliance de EPP para cada persona en la zona.

        Para cada persona trackeada:
        1. Compara required_epp de la zona vs epp_detectado
        2. Aplica regla especial de arnés (ADR-008)
        3. Crea Violation con PPEMissing por cada EPP faltante

        Args:
            persons: Personas trackeadas en el frame actual.
            zone: Zona con requisitos de EPP (required_epp, riesgo_base).

        Returns:
            Lista de violaciones detectadas. Vacía si todas las personas
            cumplen con el EPP requerido o si no hay personas.
        """
        if not persons:
            self._logger.debug("No persons to check, returning empty violations")
            return []

        violations: list[Violation] = []

        for person in persons:
            epp_faltantes: list[PPEMissing] = []
            required_epp: list[str] = zone.required_epp

            # 1. Comparar required_epp vs epp_detectado
            for epp in required_epp:
                if epp not in person.epp_detectado:
                    epp_faltantes.append(self._build_ppemissing(epp))

            # 2. Regla especial: arnés en zonas de riesgo alto (ADR-008)
            if self._check_harness_rule(person, zone, epp_faltantes):
                self._logger.debug(
                    "Persona %d: adding harness violation via ADR-008 rule",
                    person.persona_id,
                )

            # 3. Si hay EPP faltantes, crear Violation
            if epp_faltantes:
                violation = Violation(
                    persona_id=person.persona_id,
                    zona_id=zone.zone_id,
                    epp_faltantes=epp_faltantes,
                    frame_id=0,
                    timestamp=datetime.now(),
                )
                violations.append(violation)

                self._logger.info(
                    "Violation detected: persona_id=%d, zona=%s, "
                    "epp_faltantes=%s",
                    person.persona_id,
                    zone.zone_id,
                    [e.tipo for e in epp_faltantes],
                )

        self._logger.debug(
            "Check complete: %d persons, %d violations",
            len(persons),
            len(violations),
        )

        return violations

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _build_ppemissing(self, epp_type: str) -> PPEMissing:
        """Construye un objeto PPEMissing para un tipo de EPP.

        Args:
            epp_type: Tipo de EPP (hardhat, safety_vest, etc.).

        Returns:
            PPEMissing con tipo, descripción y norma chilena.
        """
        return PPEMissing(
            tipo=epp_type,  # type: ignore[arg-type]
            descripcion=EPP_DESC.get(epp_type, epp_type),
            norma_chilena=EPP_NORMA.get(epp_type, "NCh -"),
        )

    def _check_harness_rule(
        self,
        person: TrackedPerson,
        zone: Zone,
        epp_faltantes: list[PPEMissing],
    ) -> bool:
        """Verifica la regla de arnés (ADR-008).

        Si la zona tiene riesgo_base='alto' y la persona no tiene
        'harness' detectado, se agrega harness a los EPP faltantes.

        Args:
            person: Persona trackeada.
            zone: Zona con configuración de riesgo.
            epp_faltantes: Lista actual de EPP faltantes (se modifica in-place).

        Returns:
            True si se agregó harness, False si no aplica.
        """
        if zone.riesgo_base == "alto" and "harness" not in person.epp_detectado:
            # Solo agregar si no está ya en la lista
            if not any(e.tipo == "harness" for e in epp_faltantes):
                epp_faltantes.append(self._build_ppemissing("harness"))
                return True
        return False


__all__ = ["PPEComplianceChecker"]
