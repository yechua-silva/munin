"""Tests for DatasetBuilder — class mapping, dedup, split, YAML gen.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.dataset_builder import (
    DATASET_MAPPINGS,
    IGNORED_ORIGINAL_CLASSES,
    SCHEMA_V4,
    DatasetBuilder,
    _compute_dhash,
    _parse_yolo_label,
    _write_yolo_label,
)


class TestSchemaV4:
    """Verify SCHEMA_V4 has exactly 13 classes with correct names."""

    def test_schema_has_13_classes(self) -> None:
        """Schema v4 must have exactly 13 entries (0-12)."""
        assert len(SCHEMA_V4) == 13
        assert set(SCHEMA_V4.keys()) == set(range(13))

    def test_schema_class_names(self) -> None:
        """Verify specific class indices map to expected names."""
        assert SCHEMA_V4[0] == "person"
        assert SCHEMA_V4[1] == "hardhat"
        assert SCHEMA_V4[2] == "safety_vest"
        assert SCHEMA_V4[3] == "gloves"
        assert SCHEMA_V4[4] == "safety_glasses"
        assert SCHEMA_V4[5] == "safety_boots"
        assert SCHEMA_V4[6] == "harness"
        assert SCHEMA_V4[7] == "mask"
        assert SCHEMA_V4[8] == "no_hardhat"
        assert SCHEMA_V4[9] == "no_safety_vest"
        assert SCHEMA_V4[10] == "no_gloves"
        assert SCHEMA_V4[11] == "no_safety_boots"
        assert SCHEMA_V4[12] == "no_safety_glasses"


class TestDatasetMappings:
    """Verify class mappings for each dataset are valid."""

    def test_all_mappings_have_valid_targets(self) -> None:
        """All mapped class IDs must be in SCHEMA_V4 (0-12)."""
        for dataset_name, mapping in DATASET_MAPPINGS.items():
            for orig_id, target_id in mapping.items():
                assert 0 <= target_id <= 12, (
                    f"{dataset_name}: class {orig_id} → {target_id} out of range"
                )

    def test_ignored_classes_not_in_mapping(self) -> None:
        """Ignored classes should not appear in output of mapping."""
        for dataset_name, ignored in IGNORED_ORIGINAL_CLASSES.items():
            mapping = DATASET_MAPPINGS.get(dataset_name, {})
            for ign_id in ignored:
                assert ign_id not in mapping, (
                    f"{dataset_name}: ignored class {ign_id} should not be in mapping"
                )

    def test_construction_ppe_none_ignored(self) -> None:
        """Construction PPE class 5 ('none') must be in ignored set."""
        assert 5 in IGNORED_ORIGINAL_CLASSES["construction_ppe"]

    def test_shwd_mapping(self) -> None:
        """SHWD: hat→hardhat(1), person→person(0)."""
        mapping = DATASET_MAPPINGS["shwd"]
        assert mapping[0] == 1  # hat → hardhat
        assert mapping[1] == 0  # person → person


class TestDHash:
    """Verify perceptual hash computation."""

    def test_dhash_returns_python_int(self, tmp_path: Path) -> None:
        """dHash must return a Python int (not numpy int)."""
        from PIL import Image

        # Use a non-uniform image with variation in both dimensions
        img = Image.new("RGB", (32, 32), color="red")
        # Draw a simple pattern so hash is non-zero and meaningful
        for x in range(32):
            for y in range(32):
                if (x + y) % 3 == 0:
                    img.putpixel((x, y), (255, 255, 255))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        h = _compute_dhash(img_path)
        assert isinstance(h, int)

    def test_dhash_same_image_same_hash(self, tmp_path: Path) -> None:
        """Same image must produce the same dHash."""
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (32, 32), color="blue")
        for x in range(32):
            for y in range(32):
                if (x * y) % 5 == 0:
                    img.putpixel((x, y), (255, 0, 0))
        img.save(img_path)

        h1 = _compute_dhash(img_path)
        h2 = _compute_dhash(img_path)
        assert h1 == h2

    def test_dhash_different_images_different_hash(self, tmp_path: Path) -> None:
        """Different images should produce different hashes."""
        from PIL import Image

        red_path = tmp_path / "pattern_a.png"
        black_path = tmp_path / "pattern_b.png"

        img_a = Image.new("RGB", (32, 32), color="black")
        img_b = Image.new("RGB", (32, 32), color="black")

        # Draw different patterns
        for x in range(32):
            for y in range(32):
                if x < 16:
                    img_a.putpixel((x, y), (255, 0, 0))
                if y < 16:
                    img_b.putpixel((x, y), (0, 255, 0))

        img_a.save(red_path)
        img_b.save(black_path)

        h_a = _compute_dhash(red_path)
        h_b = _compute_dhash(black_path)

        # Different patterns should produce different hashes
        assert h_a != h_b


class TestLabelParsing:
    """Verify YOLO label parsing and writing."""

    def test_parse_valid_label(self, tmp_path: Path) -> None:
        """Parse a valid YOLO label file."""
        lbl = tmp_path / "test.txt"
        lbl.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.3 0.4\n")
        objects = _parse_yolo_label(lbl)
        assert len(objects) == 2
        assert objects[0] == (0, 0.5, 0.5, 0.2, 0.3)
        assert objects[1] == (1, 0.1, 0.2, 0.3, 0.4)

    def test_parse_empty_label(self, tmp_path: Path) -> None:
        """Parse an empty label file returns empty list."""
        lbl = tmp_path / "empty.txt"
        lbl.write_text("")
        assert _parse_yolo_label(lbl) == []

    def test_parse_missing_label(self, tmp_path: Path) -> None:
        """Parse a missing label file returns empty list."""
        lbl = tmp_path / "missing.txt"
        assert _parse_yolo_label(lbl) == []

    def test_parse_malformed_line(self, tmp_path: Path) -> None:
        """Malformed lines are skipped."""
        lbl = tmp_path / "bad.txt"
        lbl.write_text("0 0.5 0.5\n1 0.1 0.2 0.3 0.4\n")
        objects = _parse_yolo_label(lbl)
        assert len(objects) == 1
        assert objects[0] == (1, 0.1, 0.2, 0.3, 0.4)

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        """Writing then reading yields same data."""
        original = [(0, 0.5, 0.5, 0.2, 0.3), (2, 0.1, 0.2, 0.3, 0.4)]
        lbl = tmp_path / "roundtrip.txt"
        _write_yolo_label(lbl, original)
        parsed = _parse_yolo_label(lbl)
        assert parsed == original


class TestDeduplicate:
    """Verify deduplication by dHash."""

    def _make_pattern_image(self, path: Path, pattern: str) -> None:
        """Create a test image with a recognizable pattern."""
        from PIL import Image

        img = Image.new("RGB", (32, 32), color="black")
        for x in range(32):
            for y in range(32):
                if pattern == "red_left":
                    if x < 16:
                        img.putpixel((x, y), (255, 0, 0))
                elif pattern == "red_right":
                    if x >= 16:
                        img.putpixel((x, y), (255, 0, 0))
                elif pattern == "green_top":
                    if y < 16:
                        img.putpixel((x, y), (0, 255, 0))
                elif pattern == "blue_cross":
                    if x == y or x + y == 31:
                        img.putpixel((x, y), (0, 0, 255))
        img.save(path)

    def test_deduplicate_removes_duplicates(self, tmp_path: Path) -> None:
        """Identical images should be deduplicated."""
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        img3 = tmp_path / "img3.png"

        # img1 and img2 are identical (same pattern)
        self._make_pattern_image(img1, "red_left")
        self._make_pattern_image(img2, "red_left")  # same as img1
        self._make_pattern_image(img3, "green_top")  # different

        builder = DatasetBuilder(tmp_path)
        indices = builder._deduplicate([img1, img2, img3])
        # 2 unique images expected (img1 and img3)
        assert len(indices) == 2
        assert 0 in indices  # first occurrence kept
        assert 2 in indices  # different image kept
        # img2 (index 1) should be removed as duplicate

    def test_deduplicate_preserves_unique(self, tmp_path: Path) -> None:
        """All unique images are preserved."""
        paths = []
        for i, pattern in enumerate(["red_left", "green_top", "blue_cross"]):
            p = tmp_path / f"img_{i}.png"
            self._make_pattern_image(p, pattern)
            paths.append(p)

        builder = DatasetBuilder(tmp_path)
        indices = builder._deduplicate(paths)
        assert len(indices) == 3


class TestStratifiedSplit:
    """Verify stratified split proportions."""

    def test_split_proportions(self) -> None:
        """Split should be 80/10/10 approximately with enough samples."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Use enough images (260) so each class has ~20 samples
            # for the test split (10% of 260 = 26, covering 13 classes)
            n_samples = 260
            images = []
            labels = []
            for i in range(n_samples):
                img = tmp / f"img_{i:04d}.jpg"
                img.write_text("dummy")
                lbl = tmp / f"img_{i:04d}.txt"
                cls_id = i % 13  # distribute across all 13 classes
                lbl.write_text(f"{cls_id} 0.5 0.5 0.2 0.3\n")
                images.append(img)
                labels.append(lbl)

            builder = DatasetBuilder(tmp)
            train_imgs, val_imgs, test_imgs = builder._stratified_split(
                images, labels
            )

            total = len(images)
            # Allow +-5 tolerance for rounding
            assert len(train_imgs) == pytest.approx(int(total * 0.8), abs=5), (
                f"Train set size {len(train_imgs)} != ~{int(total * 0.8)}"
            )
            assert len(val_imgs) == pytest.approx(int(total * 0.1), abs=3), (
                f"Val set size {len(val_imgs)} != ~{int(total * 0.1)}"
            )
            assert len(test_imgs) == pytest.approx(int(total * 0.1), abs=3), (
                f"Test set size {len(test_imgs)} != ~{int(total * 0.1)}"
            )

    def test_split_sum_matches_total(self) -> None:
        """train + val + test should equal total images."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            n_samples = 260
            images = []
            labels = []
            for i in range(n_samples):
                img = tmp / f"img_{i:04d}.jpg"
                img.write_text("dummy")
                lbl = tmp / f"img_{i:04d}.txt"
                cls_id = i % 13
                lbl.write_text(f"{cls_id} 0.5 0.5 0.2 0.3\n")
                images.append(img)
                labels.append(lbl)

            builder = DatasetBuilder(tmp)
            train_imgs, val_imgs, test_imgs = builder._stratified_split(
                images, labels
            )

            total = len(train_imgs) + len(val_imgs) + len(test_imgs)
            assert total == len(images)


class TestYAMLGeneration:
    """Verify dataset-v4.yaml generation."""

    def test_yaml_format(self, tmp_path: Path) -> None:
        """Generated YAML must have correct structure."""
        builder = DatasetBuilder(tmp_path)
        builder._generate_yaml()

        yaml_path = tmp_path / "dataset-v4.yaml"
        assert yaml_path.exists()

        content = yaml_path.read_text()
        assert "path:" in content
        assert "train: images/train" in content
        assert "val: images/val" in content
        assert "test: images/test" in content
        assert "nc: 13" in content
        assert "person" in content
        assert "hardhat" in content
        assert "safety_vest" in content
        assert "no_hardhat" in content
        assert "no_safety_glasses" in content

    def test_yaml_nc_matches_schema(self, tmp_path: Path) -> None:
        """nc field must match number of classes in SCHEMA_V4."""
        builder = DatasetBuilder(tmp_path)
        builder._generate_yaml()

        yaml_path = tmp_path / "dataset-v4.yaml"
        content = yaml_path.read_text()
        assert "nc: 13" in content

        # Count class entries
        names_section = False
        names_count = 0
        for line in content.splitlines():
            if line.strip().startswith("names:"):
                names_section = True
                continue
            if names_section and line.strip() and ":" in line:
                names_count += 1
            elif names_section and not line.strip():
                names_section = False

        assert names_count == 13, f"YAML has {names_count} names, expected 13"


class TestRemapDataset:
    """Verify dataset remapping logic."""

    def test_remap_basic(self, tmp_path: Path) -> None:
        """Basic remapping converts class IDs correctly."""
        dataset_dir = tmp_path / "dataset"
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="gray")

        # Construction PPE: class 0 (helmet) → class 1 (hardhat) in v4
        img_path = img_dir / "test_img.jpg"
        img.save(img_path)
        lbl_path = lbl_dir / "test_img.txt"
        lbl_path.write_text("0 0.5 0.5 0.2 0.3\n")

        builder = DatasetBuilder(tmp_path)
        mapping = {0: 1}  # helmet → hardhat
        images, labels = builder._remap_dataset(dataset_dir, mapping, set())

        assert len(images) == 1
        assert len(labels) == 1

        # Check remapped label
        objects = _parse_yolo_label(labels[0])
        assert len(objects) == 1
        assert objects[0][0] == 1  # Should now be class 1 (hardhat)

    def test_remap_ignores_classes(self, tmp_path: Path) -> None:
        """Ignored classes are removed from labels."""
        dataset_dir = tmp_path / "dataset2"
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="gray")

        # Two objects: one valid, one ignored
        img_path = img_dir / "test.jpg"
        img.save(img_path)
        lbl_path = lbl_dir / "test.txt"
        lbl_path.write_text("0 0.5 0.5 0.2 0.3\n5 0.1 0.1 0.2 0.2\n")

        builder = DatasetBuilder(tmp_path)
        mapping = {0: 1}
        images, labels = builder._remap_dataset(dataset_dir, mapping, {5})

        assert len(images) == 1
        objects = _parse_yolo_label(labels[0])
        assert len(objects) == 1  # Only mapped class, not ignored
        assert objects[0][0] == 1  # hardhat

    def test_remap_unmapped_classes_skipped(self, tmp_path: Path) -> None:
        """Unmapped classes are skipped (no target in mapping)."""
        dataset_dir = tmp_path / "dataset3"
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="gray")

        img_path = img_dir / "test.jpg"
        img.save(img_path)
        lbl_path = lbl_dir / "test.txt"
        # Class 99 has no mapping
        lbl_path.write_text("99 0.5 0.5 0.2 0.3\n")

        builder = DatasetBuilder(tmp_path)
        mapping = {0: 1}
        images, labels = builder._remap_dataset(dataset_dir, mapping, set())

        # Should be excluded since the only class has no mapping
        assert len(images) == 0
