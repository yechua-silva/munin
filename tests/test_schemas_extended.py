"""Tests TDD para clases negativas en DetectionResult (SPEC-v3).

Verifica que DetectionResult acepte las 6 clases negativas nuevas
y que las clases positivas existentes sigan funcionando.
"""
from __future__ import annotations

from pydantic import ValidationError
import pytest


class TestDetectionResultExtended:
    """Suite TDD para DetectionResult con clases negativas."""

    def test_negative_class_no_helmet_valid(self) -> None:
        """DetectionResult acepta 'no_helmet' como class_name."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(
            class_name="no_helmet",
            bbox=(10, 20, 100, 200),
            confidence=0.85,
        )
        assert det.class_name == "no_helmet"

    def test_negative_class_no_gloves_valid(self) -> None:
        """DetectionResult acepta 'no_gloves' como class_name."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(
            class_name="no_gloves",
            bbox=(10, 20, 100, 200),
            confidence=0.75,
        )
        assert det.class_name == "no_gloves"

    def test_negative_class_no_vest_valid(self) -> None:
        """DetectionResult acepta 'no_vest' como class_name."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(
            class_name="no_vest",
            bbox=(10, 20, 100, 200),
            confidence=0.65,
        )
        assert det.class_name == "no_vest"

    def test_negative_class_no_boots_valid(self) -> None:
        """DetectionResult acepta 'no_boots' como class_name."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(
            class_name="no_boots",
            bbox=(10, 20, 100, 200),
            confidence=0.90,
        )
        assert det.class_name == "no_boots"

    def test_negative_class_no_goggle_valid(self) -> None:
        """DetectionResult acepta 'no_goggle' como class_name."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(
            class_name="no_goggle",
            bbox=(10, 20, 100, 200),
            confidence=0.80,
        )
        assert det.class_name == "no_goggle"

    def test_negative_class_no_safety_glasses_valid(self) -> None:
        """DetectionResult acepta 'no_safety_glasses' como class_name."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(
            class_name="no_safety_glasses",
            bbox=(10, 20, 100, 200),
            confidence=0.70,
        )
        assert det.class_name == "no_safety_glasses"

    def test_positive_classes_still_valid(self) -> None:
        """Las clases positivas existentes siguen funcionando."""
        from munin.gate.schemas import DetectionResult
        for cls in ["person", "hardhat", "safety_vest", "gloves",
                     "safety_glasses", "safety_boots", "harness", "mask"]:
            det = DetectionResult(
                class_name=cls,
                bbox=(0, 0, 10, 10),
                confidence=0.5,
            )
            assert det.class_name == cls

    def test_confidence_out_of_range_raises(self) -> None:
        """Confidence > 1.0 levanta ValidationError."""
        from munin.gate.schemas import DetectionResult
        with pytest.raises(ValidationError):
            DetectionResult(
                class_name="no_helmet",
                bbox=(0, 0, 1, 1),
                confidence=1.5,
            )

    def test_invalid_class_name_raises(self) -> None:
        """Un class_name que no está en el Literal levanta ValidationError."""
        from munin.gate.schemas import DetectionResult
        with pytest.raises(ValidationError):
            DetectionResult(
                class_name="invalid_class",
                bbox=(0, 0, 1, 1),
                confidence=0.5,
            )
