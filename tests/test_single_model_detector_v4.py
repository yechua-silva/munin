"""Tests for SingleModelDetector v2 — 13-class schema v4.

Validates class mapping, ignored classes (none in v4),
and unknown class handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from munin.exceptions import ConfigurationError, DetectionError
from munin.pipeline.single_model_detector import (
    CONSTRUCTION_PPE_CLASS_MAP,
    SingleModelDetector,
)


class TestConstructionPPEClassMapV4:
    """Verify the 13-class mapping is correct."""

    def test_map_has_13_entries(self) -> None:
        """CONSTRUCTION_PPE_CLASS_MAP must have exactly 13 entries."""
        assert len(CONSTRUCTION_PPE_CLASS_MAP) == 13

    def test_map_keys_are_0_to_12(self) -> None:
        """Map keys must cover all indices 0-12."""
        assert set(CONSTRUCTION_PPE_CLASS_MAP.keys()) == set(range(13))

    def test_map_positive_classes(self) -> None:
        """Verify positive class mappings."""
        assert CONSTRUCTION_PPE_CLASS_MAP[0] == "person"
        assert CONSTRUCTION_PPE_CLASS_MAP[1] == "hardhat"
        assert CONSTRUCTION_PPE_CLASS_MAP[2] == "safety_vest"
        assert CONSTRUCTION_PPE_CLASS_MAP[3] == "gloves"
        assert CONSTRUCTION_PPE_CLASS_MAP[4] == "safety_glasses"
        assert CONSTRUCTION_PPE_CLASS_MAP[5] == "safety_boots"
        assert CONSTRUCTION_PPE_CLASS_MAP[6] == "harness"
        assert CONSTRUCTION_PPE_CLASS_MAP[7] == "mask"

    def test_map_negative_classes(self) -> None:
        """Verify negative class mappings (v4 naming)."""
        assert CONSTRUCTION_PPE_CLASS_MAP[8] == "no_hardhat"
        assert CONSTRUCTION_PPE_CLASS_MAP[9] == "no_safety_vest"
        assert CONSTRUCTION_PPE_CLASS_MAP[10] == "no_gloves"
        assert CONSTRUCTION_PPE_CLASS_MAP[11] == "no_safety_boots"
        assert CONSTRUCTION_PPE_CLASS_MAP[12] == "no_safety_glasses"

    def test_all_class_names_are_strings(self) -> None:
        """All class map values must be non-empty strings."""
        for class_name in CONSTRUCTION_PPE_CLASS_MAP.values():
            assert isinstance(class_name, str)
            assert len(class_name) > 0


class TestIgnoredClassesV4:
    """Verify that v4 has no ignored classes."""

    def test_no_ignored_classes_in_v4(self) -> None:
        """In v4, _IGNORED_CLASSES must be empty."""
        from munin.pipeline.single_model_detector import _IGNORED_CLASSES

        assert len(_IGNORED_CLASSES) == 0

    def test_all_classes_produce_detections(self) -> None:
        """All 13 classes must be in the class map and not ignored."""
        from munin.pipeline.single_model_detector import _IGNORED_CLASSES

        for class_id in range(13):
            assert class_id not in _IGNORED_CLASSES
            assert class_id in CONSTRUCTION_PPE_CLASS_MAP


class TestInit:
    """Verify detector initialization."""

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_init_success(self, mock_yolo: MagicMock, tmp_path: MagicMock) -> None:
        """Detector initializes correctly with valid model path."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path), confidence=0.5)

        assert detector._confidence == 0.5
        assert detector._device == "cpu"
        assert detector._imgsz == 640

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_init_raises_on_missing_model(self, mock_yolo: MagicMock) -> None:
        """Detector must raise ConfigurationError if model not found."""
        with pytest.raises(ConfigurationError):
            SingleModelDetector("/nonexistent/model.pt")

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_init_raises_on_yolo_error(
        self, mock_yolo: MagicMock, tmp_path: MagicMock
    ) -> None:
        """Detector must raise DetectionError if YOLO fails to load."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        mock_yolo.side_effect = RuntimeError("CUDA OOM")

        with pytest.raises(DetectionError):
            SingleModelDetector(str(model_path))


class TestParseResultsV4:
    """Verify _parse_results handles all 13 v4 classes."""

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_all_13_classes_are_mapped(
        self, mock_yolo: MagicMock, tmp_path: MagicMock
    ) -> None:
        """All 13 class IDs must produce a DetectionResult."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path), confidence=0.0)

        # Mock YOLO result with all 13 classes
        mock_result = MagicMock()
        mock_result.boxes = []

        for class_id in range(13):
            mock_box = MagicMock()
            mock_box.cls = [class_id]
            mock_box.conf = [0.9]
            mock_box.xyxy = [[10, 20, 100, 200]]
            mock_result.boxes.append(mock_box)

        results = detector._parse_results(mock_result)

        assert len(results) == 13

        class_names_found = {d.class_name for d in results}
        expected_names = set(CONSTRUCTION_PPE_CLASS_MAP.values())
        assert class_names_found == expected_names

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_unknown_class_skipped_with_warning(
        self, mock_yolo: MagicMock, tmp_path: MagicMock, caplog: MagicMock
    ) -> None:
        """Unknown class IDs must be skipped with a warning."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path), confidence=0.0)

        mock_result = MagicMock()
        mock_result.boxes = []

        # One valid class, one unknown
        valid_box = MagicMock()
        valid_box.cls = [0]  # person
        valid_box.conf = [0.9]
        valid_box.xyxy = [[10, 20, 100, 200]]
        mock_result.boxes.append(valid_box)

        unknown_box = MagicMock()
        unknown_box.cls = [99]  # not in map
        unknown_box.conf = [0.9]
        unknown_box.xyxy = [[50, 60, 150, 250]]
        mock_result.boxes.append(unknown_box)

        import logging

        with caplog.at_level(logging.WARNING):
            results = detector._parse_results(mock_result)

        assert len(results) == 1
        assert results[0].class_name == "person"
        assert "Unknown class_id 99" in caplog.text

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_confidence_filter(
        self, mock_yolo: MagicMock, tmp_path: MagicMock
    ) -> None:
        """Detections below confidence threshold must be filtered out."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path), confidence=0.7)

        mock_result = MagicMock()
        mock_result.boxes = []

        # Above threshold
        high_conf_box = MagicMock()
        high_conf_box.cls = [0]
        high_conf_box.conf = [0.9]
        high_conf_box.xyxy = [[10, 20, 100, 200]]
        mock_result.boxes.append(high_conf_box)

        # Below threshold
        low_conf_box = MagicMock()
        low_conf_box.cls = [1]
        low_conf_box.conf = [0.4]
        low_conf_box.xyxy = [[50, 60, 150, 250]]
        mock_result.boxes.append(low_conf_box)

        results = detector._parse_results(mock_result)

        assert len(results) == 1
        assert results[0].class_name == "person"

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_no_boxes_returns_empty(
        self, mock_yolo: MagicMock, tmp_path: MagicMock
    ) -> None:
        """None result or result with no boxes returns empty list."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path))

        assert detector._parse_results(None) == []

        mock_empty = MagicMock()
        mock_empty.boxes = None
        assert detector._parse_results(mock_empty) == []

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_results_sorted_by_confidence(
        self, mock_yolo: MagicMock, tmp_path: MagicMock
    ) -> None:
        """Results must be sorted by confidence descending."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path), confidence=0.0)

        mock_result = MagicMock()
        mock_result.boxes = []

        confidences = [0.5, 0.9, 0.7]
        for i, conf in enumerate(confidences):
            box = MagicMock()
            box.cls = [i]  # 0, 1, 2
            box.conf = [conf]
            box.xyxy = [[10, 20, 100, 200]]
            mock_result.boxes.append(box)

        results = detector._parse_results(mock_result)

        assert len(results) == 3
        assert results[0].confidence >= results[1].confidence >= results[2].confidence


class TestDetectionResultV4:
    """Verify DetectionResult structure for v4 classes."""

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_detection_result_structure(
        self, mock_yolo: MagicMock, tmp_path: MagicMock
    ) -> None:
        """DetectionResult must have correct fields and types."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        detector = SingleModelDetector(str(model_path), confidence=0.0)

        mock_result = MagicMock()
        mock_result.boxes = []

        box = MagicMock()
        box.cls = [1]  # hardhat
        box.conf = [0.85]
        box.xyxy = [[10.5, 20.3, 100.7, 200.9]]
        mock_result.boxes.append(box)

        results = detector._parse_results(mock_result)

        assert len(results) == 1
        det = results[0]

        assert det.class_name == "hardhat"
        assert isinstance(det.bbox, tuple)
        assert len(det.bbox) == 4
        assert all(isinstance(v, float) for v in det.bbox)
        assert det.bbox == (10.5, 20.3, 100.7, 200.9)
        assert det.confidence == 0.85


class TestDetectMethodV4:
    """Verify detect() method integration."""

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_detect_calls_predict_and_parse(
        self, mock_yolo_class: MagicMock, tmp_path: MagicMock
    ) -> None:
        """detect() must call predict and _parse_results."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        detector = SingleModelDetector(str(model_path), confidence=0.5)

        mock_result = MagicMock()
        mock_result.boxes = []
        box = MagicMock()
        box.cls = [0]
        box.conf = [0.9]
        box.xyxy = [[10, 20, 100, 200]]
        mock_result.boxes.append(box)

        mock_model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert len(results) == 1
        assert results[0].class_name == "person"

        mock_model.predict.assert_called_once_with(
            frame,
            conf=0.5,
            device="cpu",
            imgsz=640,
            verbose=False,
        )

    @patch("munin.pipeline.single_model_detector.YOLO")
    def test_detect_raises_on_failure(
        self, mock_yolo_class: MagicMock, tmp_path: MagicMock
    ) -> None:
        """detect() must raise DetectionError on inference failure."""
        model_path = tmp_path / "best.pt"
        model_path.write_text("dummy")

        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_model.predict.side_effect = RuntimeError("Inference failed")

        detector = SingleModelDetector(str(model_path), confidence=0.5)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(DetectionError):
            detector.detect(frame)
