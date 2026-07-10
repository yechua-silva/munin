"""Tests para point-in-polygon ray casting en PPEComplianceChecker (ADR-030).

Verifica que el filtro geométrico por polígono funciona correctamente.

Correr con: pytest munin/tests/test_point_in_polygon.py -v
"""
from __future__ import annotations

import pytest

from munin.config import Zone
from munin.gate.schemas import TrackedPerson
from munin.pipeline.ppe_checker import PPEComplianceChecker


class TestPointInPolygon:
    """Tests para _point_in_polygon (ray casting)."""

    def test_point_inside_convex_polygon(self) -> None:
        """Punto dentro de polígono convexo → True."""
        polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        assert PPEComplianceChecker._point_in_polygon(0.5, 0.5, polygon) is True

    def test_point_outside_convex_polygon(self) -> None:
        """Punto fuera de polígono convexo → False."""
        polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        assert PPEComplianceChecker._point_in_polygon(2.0, 2.0, polygon) is False

    def test_point_inside_concave_polygon(self) -> None:
        """Punto dentro de polígono cóncavo (L-shape) → True."""
        # L-shape polygon
        polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]]
        assert PPEComplianceChecker._point_in_polygon(0.25, 0.75, polygon) is True

    def test_point_outside_concave_polygon(self) -> None:
        """Punto fuera de polígono cóncavo (L-shape) → False."""
        polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]]
        # Punto en la esquina faltante de la L
        assert PPEComplianceChecker._point_in_polygon(0.75, 0.75, polygon) is False

    def test_point_on_edge(self) -> None:
        """Punto en vértice → consistente (puede ser True o False según precisión)."""
        polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        # El punto (0.0, 0.0) está justo en el vértice
        # Ray casting puede dar True o False según el algoritmo exacto
        result = PPEComplianceChecker._point_in_polygon(0.0, 0.0, polygon)
        # Debe ser booleano sin excepción
        assert isinstance(result, bool)

    def test_empty_polygon(self) -> None:
        """Polígono vacío → False."""
        assert PPEComplianceChecker._point_in_polygon(0.5, 0.5, []) is False

    def test_polygon_less_than_3_vertices(self) -> None:
        """Polígono < 3 vértices → False."""
        assert PPEComplianceChecker._point_in_polygon(0.5, 0.5, [[0.0, 0.0], [1.0, 0.0]]) is False

    def test_single_vertex_polygon(self) -> None:
        """1 vértice → False."""
        assert PPEComplianceChecker._point_in_polygon(0.5, 0.5, [[0.0, 0.0]]) is False

    def test_triangle_inside(self) -> None:
        """Punto dentro de triángulo → True."""
        polygon = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        assert PPEComplianceChecker._point_in_polygon(0.5, 0.25, polygon) is True

    def test_triangle_outside(self) -> None:
        """Punto fuera de triángulo → False."""
        polygon = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        assert PPEComplianceChecker._point_in_polygon(0.0, 1.0, polygon) is False


class TestFilterByZone:
    """Tests para _filter_by_zone."""

    def test_filter_by_zone_polygon_none(self) -> None:
        """polygon=None → todas las personas."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
            polygon=None,
        )
        persons = [
            TrackedPerson(persona_id=1, bbox=(0.1, 0.1, 0.2, 0.3), epp_detectado=set()),
            TrackedPerson(persona_id=2, bbox=(0.8, 0.8, 0.9, 0.95), epp_detectado=set()),
        ]
        result = PPEComplianceChecker._filter_by_zone(persons, zone)
        assert len(result) == 2

    def test_filter_by_zone_all_inside(self) -> None:
        """polygon con personas dentro."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
            polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        )
        persons = [
            TrackedPerson(persona_id=1, bbox=(0.1, 0.1, 0.2, 0.3), epp_detectado=set()),
            TrackedPerson(persona_id=2, bbox=(0.4, 0.4, 0.5, 0.6), epp_detectado=set()),
        ]
        result = PPEComplianceChecker._filter_by_zone(persons, zone)
        assert len(result) == 2

    def test_filter_by_zone_some_outside(self) -> None:
        """polygon con personas dentro y fuera."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
            polygon=[[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]],
        )
        persons = [
            TrackedPerson(persona_id=1, bbox=(0.1, 0.1, 0.2, 0.3), epp_detectado=set()),  # inside
            TrackedPerson(persona_id=2, bbox=(0.6, 0.1, 0.7, 0.3), epp_detectado=set()),  # outside (x > 0.5)
        ]
        result = PPEComplianceChecker._filter_by_zone(persons, zone)
        assert len(result) == 1
        assert result[0].persona_id == 1

    def test_filter_by_zone_all_outside(self) -> None:
        """polygon con todas las personas fuera."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
            polygon=[[[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]],
        )
        persons = [
            TrackedPerson(persona_id=1, bbox=(0.6, 0.1, 0.7, 0.3), epp_detectado=set()),
            TrackedPerson(persona_id=2, bbox=(0.8, 0.8, 0.9, 0.95), epp_detectado=set()),
        ]
        result = PPEComplianceChecker._filter_by_zone(persons, zone)
        assert len(result) == 0

    def test_filter_by_zone_multiple_sub_polygons(self) -> None:
        """Múltiples sub-polígonos — persona en cualquier sub-polígono se incluye."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
            polygon=[
                [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
                [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
            ],
        )
        persons = [
            TrackedPerson(persona_id=1, bbox=(0.1, 0.1, 0.2, 0.3), epp_detectado=set()),  # in sub1
            TrackedPerson(persona_id=2, bbox=(0.7, 0.7, 0.8, 0.85), epp_detectado=set()),  # in sub2
            TrackedPerson(persona_id=3, bbox=(0.4, 0.4, 0.5, 0.55), epp_detectado=set()),  # outside both
        ]
        result = PPEComplianceChecker._filter_by_zone(persons, zone)
        assert len(result) == 2
        result_ids = {p.persona_id for p in result}
        assert result_ids == {1, 2}
