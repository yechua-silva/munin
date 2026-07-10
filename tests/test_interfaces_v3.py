"""Tests TDD para interfaces v3 (Protocol compliance).

Verifica que las implementaciones satisfagan las interfaces v3
usando duck typing con @runtime_checkable.

Correr con: pytest munin/tests/test_interfaces_v3.py -v
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

from munin.pipeline.interfaces import (
    IDetector,
    ITracker,
    IComplianceChecker,
)


class TestInterfacesV3:
    """Suite TDD para interfaces v3."""

    def test_itracker_update_takes_frame_not_detections(self) -> None:
        """ITracker.update debe aceptar np.ndarray (frame), no list[DetectionResult]."""
        sig = inspect.signature(ITracker.update)
        params = list(sig.parameters.keys())
        assert "frame" in params or params[1] == "frame"
        # El segundo parámetro (después de self) debe ser 'frame'
        assert params[1] == "frame"

    def test_icompliance_checker_check_takes_three_params(self) -> None:
        """IComplianceChecker.check debe aceptar persons, detections, zone."""
        sig = inspect.signature(IComplianceChecker.check)
        params = list(sig.parameters.keys())
        assert len(params) == 4  # self + persons + detections + zone
        assert "detections" in params

    def test_idetector_has_detect_stream(self) -> None:
        """IDetector debe tener método detect_stream."""
        assert hasattr(IDetector, "detect_stream")
        sig = inspect.signature(IDetector.detect_stream)
        params = list(sig.parameters.keys())
        assert "video_path" in params or params[1] == "video_path"

    def test_byte_track_adapter_satisfies_itracker(self) -> None:
        """ByteTrackAdapter satisface ITracker v3 (duck typing)."""
        from munin.pipeline.byte_track_adapter import ByteTrackAdapter
        # Verificar que tiene método update con la firma correcta
        sig = inspect.signature(ByteTrackAdapter.update)
        params = list(sig.parameters.keys())
        assert params[1] == "frame"

    def test_ppe_checker_satisfies_icompliance_checker_v3(self) -> None:
        """PPEComplianceChecker.check tiene 3 params (persons, detections, zone)."""
        from munin.pipeline.ppe_checker import PPEComplianceChecker
        sig = inspect.signature(PPEComplianceChecker.check)
        params = list(sig.parameters.keys())
        assert len(params) == 4  # self + persons + detections + zone
        assert "detections" in params

    def test_two_model_detector_has_detect_stream(self) -> None:
        """TwoModelDetector tiene detect_stream método."""
        from munin.pipeline.two_model_detector import TwoModelDetector
        assert hasattr(TwoModelDetector, "detect_stream")
        sig = inspect.signature(TwoModelDetector.detect_stream)
        params = list(sig.parameters.keys())
        assert "video_path" in params or params[1] == "video_path"

    def test_single_model_detector_satisfies_idetector(self) -> None:
        """SingleModelDetector tiene detect y detect_stream."""
        from munin.pipeline.single_model_detector import SingleModelDetector
        assert hasattr(SingleModelDetector, "detect")
        assert hasattr(SingleModelDetector, "detect_stream")
        sig = inspect.signature(SingleModelDetector.detect)
        params = list(sig.parameters.keys())
        assert params[1] == "frame"
