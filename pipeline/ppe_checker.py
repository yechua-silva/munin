from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

from munin.config import Zone
from munin.gate.schemas import DetectionResult, PPEMissing, TrackedPerson, Violation
from munin.gate.schemas import NEGATIVE_CLASS_MAP
from munin.knowledge.ds132_kb import DS132KnowledgeBase
from munin.knowledge.zone_config import ZoneConfig

logger = logging.getLogger(__name__)


class ComplianceMode(str, Enum):
    """Modo de compliance EPP.

    Attributes:
        LEGACY: Comportamiento histórico (6 clases, sin clases negativas).
        DUAL_CLASS: Construction-PPE (11 clases, con clases negativas).
    """
    LEGACY = "legacy"
    DUAL_CLASS = "dual_class"


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
        _mode: Modo de compliance (LEGACY o DUAL_CLASS).
        _logger: Logger de la clase.
    """

    def __init__(
        self,
        zone_config: ZoneConfig,
        ds132_kb: DS132KnowledgeBase,
        mode: ComplianceMode = ComplianceMode.LEGACY,
    ) -> None:
        """Inicializa el checker con dependencias inyectadas.

        Args:
            zone_config: Configuración de zonas con EPP requerido.
            ds132_kb: Knowledge base de artículos DS 132.
            mode: Modo de compliance (LEGACY o DUAL_CLASS).
        """
        self._zone_config: ZoneConfig = zone_config
        self._ds132_kb: DS132KnowledgeBase = ds132_kb
        self._mode: ComplianceMode = mode
        self._logger = logging.getLogger(self.__class__.__name__)

    def check(
        self,
        persons: list[TrackedPerson],
        detections: list[DetectionResult],
        zone: Zone,
    ) -> list[Violation]:
        """Verifica compliance de EPP para cada persona en la zona.

        Paso a paso:
        0. Filtrar personas por polígono de zona (v4)
        1. Asignar EPP detectado desde las detecciones del frame
        2. Comparar required_epp de la zona vs epp_detectado
        3. En modo DUAL_CLASS, procesar clases negativas
        4. Aplicar regla especial de arnés (ADR-008)
        5. Crear Violation con PPEMissing por cada EPP faltante

        Args:
            persons: Personas trackeadas en el frame actual.
            detections: Detecciones YOLO del frame actual.
            zone: Zona con requisitos de EPP (required_epp, riesgo_base).

        Returns:
            Lista de violaciones detectadas. Vacía si todas las personas
            cumplen con el EPP requerido o si no hay personas.
        """
        if not persons:
            self._logger.debug("No persons to check, returning empty violations")
            return []

        # STEP 0: Filtrar personas por zona geométrica (v4)
        persons_in_zone = self._filter_by_zone(persons, zone)
        if not persons_in_zone:
            return []

        # 1. Asignar EPP a personas desde detections
        self._assign_epp_to_persons(persons_in_zone, detections)

        violations: list[Violation] = []

        for person in persons_in_zone:
            epp_faltantes: list[PPEMissing] = []
            required_epp: list[str] = zone.required_epp

            if self._mode == ComplianceMode.DUAL_CLASS:
                # DUAL_CLASS: procesar clases negativas
                negatives = [
                    d for d in detections
                    if d.class_name in NEGATIVE_CLASS_MAP
                    and self._is_ppe_inside_person(d.bbox, person.bbox)
                ]
                for neg in negatives:
                    mapped = NEGATIVE_CLASS_MAP[neg.class_name]
                    if mapped not in person.epp_detectado:
                        if not any(e.tipo == mapped for e in epp_faltantes):
                            epp_faltantes.append(self._build_ppemissing(mapped))

                # Fallback legacy: required EPP no cubierto ni por detección
                # positiva ni por clase negativa
                for epp in required_epp:
                    if epp not in person.epp_detectado:
                        if not any(e.tipo == epp for e in epp_faltantes):
                            epp_faltantes.append(self._build_ppemissing(epp))
            else:
                # LEGACY: comportamiento histórico
                for epp in required_epp:
                    if epp not in person.epp_detectado:
                        epp_faltantes.append(self._build_ppemissing(epp))

            # 2. Regla especial: arnés en zonas de riesgo alto (ADR-008)
            self._check_harness_rule(person, zone, epp_faltantes)

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
            len(persons_in_zone),
            len(violations),
        )

        return violations

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_polygon(
        point_x: float,
        point_y: float,
        polygon: list[list[float]],
    ) -> bool:
        """Determina si un punto está dentro de un polígono.

        Usa ray casting algorithm (even-odd rule).

        Args:
            point_x: Coordenada X del punto (normalizada 0-1).
            point_y: Coordenada Y del punto (normalizada 0-1).
            polygon: Lista de vértices [[x, y], ...] normalizados 0-1.

        Returns:
            True si el punto está dentro del polígono.
        """
        if not polygon or len(polygon) < 3:
            return False

        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > point_y) != (yj > point_y)):
                x_intersect = (xj - xi) * (point_y - yi) / (yj - yi) + xi
                if point_x < x_intersect:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _filter_by_zone(
        persons: list[TrackedPerson],
        zone: Zone,
    ) -> list[TrackedPerson]:
        """Filtra personas dentro del polígono de la zona.

        Usa BOTTOM_CENTER del bbox como anchor (posición de pies).
        Si zone.polygon es None, retorna todas las personas.

        Args:
            persons: Todas las personas trackeadas.
            zone: Zona con polígono opcional.

        Returns:
            Personas dentro del polígono (o todas si polygon=None).
        """
        if zone.polygon is None:
            return persons

        persons_in_zone: list[TrackedPerson] = []
        for person in persons:
            # BOTTOM_CENTER = ((x1+x2)/2, y2)
            x1, y1, x2, y2 = person.bbox
            anchor_x = (x1 + x2) / 2.0
            anchor_y = y2

            for sub_polygon in zone.polygon:
                if PPEComplianceChecker._point_in_polygon(anchor_x, anchor_y, sub_polygon):
                    persons_in_zone.append(person)
                    break

        logger.debug(
            "Zone filter: %d/%d persons in zone '%s'",
            len(persons_in_zone), len(persons), zone.zone_id,
        )
        return persons_in_zone

    def _assign_epp_to_persons(
        self,
        persons: list[TrackedPerson],
        detections: list[DetectionResult],
    ) -> None:
        """Asigna EPP detectado a cada persona basado en proximidad de bbox.

        Migrado de PersonTracker._is_ppe_inside_person.
        Un EPP se asigna si su centroide está dentro del bbox de la persona
        o si hay solapamiento (IoU > 0).

        Args:
            persons: Lista de personas trackeadas.
            detections: Detecciones YOLO del frame actual.
        """
        ppe_items = [d for d in detections if d.class_name != "person"]
        if not ppe_items:
            return

        for person in persons:
            assigned: set[str] = set()
            for ppe in ppe_items:
                if self._is_ppe_inside_person(ppe.bbox, person.bbox):
                    # Solo asignar EPP positivo (no clases negativas)
                    if ppe.class_name not in NEGATIVE_CLASS_MAP:
                        assigned.add(ppe.class_name)
            person.epp_detectado = assigned

    @staticmethod
    def _is_ppe_inside_person(
        ppe_bbox: tuple[float, float, float, float],
        person_bbox: tuple[float, float, float, float],
    ) -> bool:
        """Determina si un ítem de EPP está dentro o contiguo a una persona.

        Migrado de PersonTracker._is_ppe_inside_person.

        Args:
            ppe_bbox: Bounding box del ítem de EPP (x1, y1, x2, y2).
            person_bbox: Bounding box de la persona (x1, y1, x2, y2).

        Returns:
            True si el EPP está dentro de la persona o solapado.
        """
        ppe_cx = (ppe_bbox[0] + ppe_bbox[2]) / 2.0
        ppe_cy = (ppe_bbox[1] + ppe_bbox[3]) / 2.0

        inside = (
            person_bbox[0] <= ppe_cx <= person_bbox[2]
            and person_bbox[1] <= ppe_cy <= person_bbox[3]
        )

        if not inside:
            # Fallback: IoU > 0
            try:
                x1 = max(ppe_bbox[0], person_bbox[0])
                y1 = max(ppe_bbox[1], person_bbox[1])
                x2 = min(ppe_bbox[2], person_bbox[2])
                y2 = min(ppe_bbox[3], person_bbox[3])
                if x2 > x1 and y2 > y1:
                    inside = True
            except Exception:
                inside = False

        return inside

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


__all__ = [
    "ComplianceMode",
    "PPEComplianceChecker",
]
