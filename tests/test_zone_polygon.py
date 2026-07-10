"""Tests para Zone.polygon (ADR-029).

Verifica que el campo polygon en Zone es compatible con zones.json v1.0 y v2.0.

Correr con: pytest munin/tests/test_zone_polygon.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from munin.config import Zone
from munin.knowledge.zone_config import ZoneConfig


class TestZonePolygon:
    """Tests para el campo polygon en Zone."""

    def test_zone_with_polygon_none(self) -> None:
        """polygon=None → backward compatible."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
        )
        assert zone.polygon is None

    def test_zone_with_polygon(self) -> None:
        """polygon=[[...]] → parsea correctamente."""
        zone = Zone(
            zone_id="test",
            nombre="Test Zone",
            required_epp=["hardhat"],
            riesgo_base="bajo",
            articulos_ds132=["Art. 38"],
            polygon=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]],
        )
        assert zone.polygon is not None
        assert len(zone.polygon) == 1
        assert len(zone.polygon[0]) == 4
        assert zone.polygon[0][0] == [0.1, 0.1]
        assert zone.polygon[0][2] == [0.5, 0.5]

    def test_zone_with_multiple_sub_polygons(self) -> None:
        """Múltiples sub-polígonos."""
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
        assert zone.polygon is not None
        assert len(zone.polygon) == 2

    def test_v2_zones_json_loads_without_errors(self) -> None:
        """zones.json v2.0 carga sin errores."""
        config = ZoneConfig.from_json(
            str(Path(__file__).parent.parent / "knowledge" / "zones.json")
        )
        zone = config.get_zone("extraccion")
        assert zone.polygon is not None
        assert len(zone.polygon) == 1
        assert len(zone.polygon[0]) == 5

        zone = config.get_zone("mantencion")
        assert zone.polygon is None


class TestZonePolygonV1Compat:
    """Tests de compatibilidad con zones.json v1.0 (sin polygon)."""

    @pytest.fixture
    def v1_zones_json(self) -> str:
        """Crea un zones.json v1.0 temporal (sin polygon)."""
        data = {
            "metadata": {"faena": "Test", "version": "1.0.0"},
            "zonas": [
                {
                    "zone_id": "test_zone",
                    "nombre": "Zona de prueba",
                    "required_epp": ["hardhat"],
                    "riesgo_base": "medio",
                    "min_confidence": 0.5,
                    "articulos_ds132": ["Art. 38"],
                }
            ],
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp)
        tmp.close()
        yield tmp.name
        Path(tmp.name).unlink(missing_ok=True)

    def test_v1_json_loads_without_errors(self, v1_zones_json: str) -> None:
        """zones.json v1.0 (sin polygon) sigue funcionando."""
        config = ZoneConfig.from_json(v1_zones_json)
        zone = config.get_zone("test_zone")
        assert zone.zone_id == "test_zone"
        assert zone.polygon is None  # backward compatible

    def test_v1_json_polygon_default(self, v1_zones_json: str) -> None:
        """polygon default=None en v1.0."""
        config = ZoneConfig.from_json(v1_zones_json)
        zone = config.get_zone("test_zone")
        assert zone.polygon is None
