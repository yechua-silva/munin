"""Dataset Builder — Unifica 7 datasets PPE en schema v4 de 13 clases.

Pipeline completo:
  1. Descarga cada dataset desde Roboflow / HuggingFace / GitHub
  2. Remapea las clases originales al schema unificado de 13 clases
  3. Deduplica por hash perceptual (dHash)
  4. Split estratificado 80/10/10 (train/val/test)
  5. Exporta en formato YOLO (images/ + labels/) + dataset-v4.yaml

Usage:
    python scripts/dataset_builder.py --output /scratch/datasets/munin-v4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Schema unificado v4 — 13 clases
# ──────────────────────────────────────────────────────────────
SCHEMA_V4: dict[int, str] = {
    0: "person",
    1: "hardhat",
    2: "safety_vest",
    3: "gloves",
    4: "safety_glasses",
    5: "safety_boots",
    6: "harness",
    7: "mask",
    8: "no_hardhat",
    9: "no_safety_vest",
    10: "no_gloves",
    11: "no_safety_boots",
    12: "no_safety_glasses",
}

# ──────────────────────────────────────────────────────────────
# Mappings de clase original → schema v4 por dataset
# ──────────────────────────────────────────────────────────────
DATASET_MAPPINGS: dict[str, dict[int, int]] = {
    "construction_ppe": {
        0: 1,  # helmet → hardhat
        1: 2,  # vest → safety_vest
        2: 3,  # gloves → gloves
        3: 4,  # glasses → safety_glasses
        4: 4,  # goggles → safety_glasses (merge)
        # 5: "none" se ignora
        6: 0,  # Person → person
        7: 8,  # no_helmet → no_hardhat
        8: 9,  # no_vest → no_safety_vest
        9: 10,  # no_gloves → no_gloves
        10: 11,  # no_boots → no_safety_boots
    },
    "youcefs": {
        0: 1,  # helmet → hardhat
        1: 2,  # vest → safety_vest
        2: 0,  # person → person
        3: 4,  # goggles → safety_glasses
        4: 3,  # gloves → gloves
        5: 5,  # boots → safety_boots
        6: 6,  # harness → harness
        7: 8,  # no-helmet → no_hardhat
        8: 9,  # no-vest → no_safety_vest
    },
    "keremberke": {
        0: 0,  # Person → person
        1: 7,  # Mask → mask
        2: 1,  # Hardhat → hardhat
        3: 4,  # Safety_Glasses → safety_glasses
        4: 2,  # Safety_Vest → safety_vest
        5: 5,  # Safety_Boots → safety_boots
        6: 3,  # Gloves → gloves
        7: 8,  # No-Hardhat → no_hardhat
        8: 9,  # No-Safety_Vest → no_safety_vest
        9: 10,  # No-Gloves → no_gloves
    },
    "shwd": {
        0: 1,  # hat → hardhat
        1: 0,  # person → person
    },
    "skcet": {
        0: 0,  # person → person
        1: 1,  # hardhat → hardhat
        2: 2,  # safety_vest → safety_vest
        3: 3,  # gloves → gloves
        4: 4,  # safety_glasses → safety_glasses
        5: 5,  # safety_boots → safety_boots
    },
    "construction2": {
        0: 0,  # person → person
        1: 1,  # hardhat → hardhat
        2: 2,  # safety_vest → safety_vest
        3: 3,  # gloves → gloves
        4: 4,  # safety_glasses → safety_glasses
        5: 5,  # safety_boots → safety_boots
        6: 6,  # harness → harness
    },
    "voxdroid": {
        0: 0,  # person → person
        1: 1,  # hardhat → hardhat
        2: 2,  # safety_vest → safety_vest
    },
    "keremberke_small": {
        0: 0,  # Person → person
        1: 7,  # Mask → mask
        2: 1,  # Hardhat → hardhat
        3: 4,  # Safety_Glasses → safety_glasses
        4: 2,  # Safety_Vest → safety_vest
        5: 5,  # Safety_Boots → safety_boots
        6: 3,  # Gloves → gloves
    },
}

# Clases a ignorar (no se incluyen en el dataset unificado)
IGNORED_ORIGINAL_CLASSES: dict[str, set[int]] = {
    "construction_ppe": {5},  # "none" se ignora
    "youcefs": set(),
    "keremberke": set(),
    "shwd": set(),
    "skcet": set(),
    "construction2": set(),
    "voxdroid": set(),
    "keremberke_small": set(),
}


def _compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Compute perceptual dHash for an image.

    Args:
        image_path: Path to the image file.
        hash_size: Size of the hash (default 8 → 64-bit hash).

    Returns:
        Integer hash value.
    """
    from PIL import Image

    img = Image.open(image_path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    pixels = np.asarray(img, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_bits = diff.flatten()
    hash_int = sum(bit << i for i, bit in enumerate(hash_bits))
    return hash_int


def _parse_yolo_label(
    label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO format label file.

    Args:
        label_path: Path to .txt label file.

    Returns:
        List of (class_id, x_center, y_center, width, height) tuples.
    """
    objects: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return objects
    text = label_path.read_text().strip()
    if not text:
        return objects
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        objects.append((cls_id, xc, yc, w, h))
    return objects


def _write_yolo_label(
    label_path: Path,
    objects: list[tuple[int, float, float, float, float]],
) -> None:
    """Write objects to a YOLO format label file.

    Args:
        label_path: Output path for .txt label file.
        objects: List of (class_id, x_center, y_center, width, height).
    """
    lines = []
    for cls_id, xc, yc, w, h in objects:
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


class DatasetBuilder:
    """Builder para unificar datasets PPE en schema v4 (13 clases).

    Ejecuta el pipeline completo: download → map → dedup → split → export.
    Produce un dataset en formato YOLO listo para entrenar con ultralytics.

    Attributes:
        output_dir: Directorio raíz del dataset unificado.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize DatasetBuilder.

        Args:
            output_dir: Directorio donde se generará el dataset unificado.
        """
        self._output_dir = Path(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._raw_dir = self._output_dir / "raw"

    def build(self, datasets: list[dict[str, Any]]) -> None:
        """Run the complete pipeline: download → map → dedup → split → export.

        Args:
            datasets: List of dataset config dicts. Each config must have:
                - 'name': Dataset key in DATASET_MAPPINGS.
                - 'source': Dict with 'type' ('roboflow'|'huggingface'|'github')
                  and 'url' or 'path'.
                - 'mapping': Optional override mapping.
        """
        self._logger.info("Starting dataset build with %d sources", len(datasets))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        all_images: list[Path] = []
        all_labels: list[Path] = []

        for config in datasets:
            self._logger.info("Processing dataset: %s", config["name"])
            dataset_path = self._download_dataset(config)
            mapping = config.get("mapping") or DATASET_MAPPINGS.get(
                config["name"], {}
            )
            ignored = IGNORED_ORIGINAL_CLASSES.get(config["name"], set())

            images, labels = self._remap_dataset(
                dataset_path, mapping, ignored
            )
            all_images.extend(images)
            all_labels.extend(labels)
            self._logger.info(
                "  → %d images, %d labels after remap",
                len(images),
                len(labels),
            )

        # Deduplicate
        self._logger.info("Deduplicating %d images...", len(all_images))
        dedup_indices = self._deduplicate(all_images)
        all_images = [all_images[i] for i in dedup_indices]
        all_labels = [all_labels[i] for i in dedup_indices]
        self._logger.info("  → %d images after dedup", len(all_images))

        # Stratified split 80/10/10
        self._logger.info("Splitting dataset 80/10/10...")
        splits = self._stratified_split(all_images, all_labels)
        train_imgs, val_imgs, test_imgs = splits

        # Export YOLO
        self._logger.info("Exporting to YOLO format...")
        self._export_yolo(train_imgs, val_imgs, test_imgs, all_labels)

        # Generate dataset-v4.yaml
        self._generate_yaml()

        self._logger.info(
            "Dataset build complete. Output: %s", self._output_dir
        )

    def _download_dataset(self, config: dict[str, Any]) -> Path:
        """Download a dataset from its source.

        Supports Roboflow, HuggingFace, and GitHub sources.

        Args:
            config: Dataset configuration dict.

        Returns:
            Path to the downloaded dataset directory.
        """
        name = config["name"]
        source = config["source"]
        source_type = source.get("type", "roboflow")

        dest = self._raw_dir / name
        if dest.exists():
            self._logger.info("  Dataset %s already exists at %s, skipping download", name, dest)
            return dest

        dest.mkdir(parents=True, exist_ok=True)

        if source_type == "roboflow":
            self._download_roboflow(source, dest)
        elif source_type == "huggingface":
            self._download_huggingface(source, dest)
        elif source_type == "github":
            self._download_github(source, dest)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        return dest

    def _download_roboflow(self, source: dict[str, Any], dest: Path) -> None:
        """Download dataset from Roboflow.

        Uses roboflow Python SDK if available, otherwise falls back to
        direct download URL.

        Args:
            source: Dict with 'api_key', 'workspace', 'project', 'version',
                    and optionally 'format'.
            dest: Destination directory.
        """
        api_key = source.get("api_key", "")
        workspace = source.get("workspace", "")
        project = source.get("project", "")
        version = source.get("version", 1)
        fmt = source.get("format", "yolov8")

        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)
            rf_project = rf.workspace(workspace).project(project)
            dataset = rf_project.version(version).download(fmt, location=str(dest))
            self._logger.info("  Roboflow download complete: %s", dataset.location)
        except ImportError:
            # Fallback: direct download if a URL is provided
            url = source.get("url")
            if url:
                self._download_zip(url, dest)
            else:
                raise RuntimeError(
                    "Roboflow SDK not installed and no fallback URL provided. "
                    "Install with: pip install roboflow"
                )

    def _download_huggingface(self, source: dict[str, Any], dest: Path) -> None:
        """Download dataset from HuggingFace.

        Args:
            source: Dict with 'repo_id', 'subset', 'split'.
            dest: Destination directory.
        """
        try:
            from datasets import load_dataset

            repo_id = source["repo_id"]
            subset = source.get("subset")
            split = source.get("split", "train")

            hf_dataset = load_dataset(repo_id, subset, split=split, trust_remote_code=True)
            # Save images and labels
            img_dir = dest / "images"
            lbl_dir = dest / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for i, item in enumerate(tqdm(hf_dataset, desc=f"Downloading {repo_id}")):
                image = item.get("image")
                if image is None:
                    continue
                img_path = img_dir / f"img_{i:06d}.jpg"
                image.save(img_path, format="JPEG")

                # Save labels if present
                label_data = item.get("objects") or item.get("labels") or item.get("boxes")
                if label_data is not None:
                    lbl_path = lbl_dir / f"img_{i:06d}.txt"
                    self._write_hf_labels(lbl_path, label_data)

            self._logger.info("  HuggingFace download complete: %s", repo_id)
        except ImportError:
            raise RuntimeError(
                "HuggingFace datasets library not installed. "
                "Install with: pip install datasets"
            )

    def _download_github(self, source: dict[str, Any], dest: Path) -> None:
        """Download dataset from GitHub release or repo.

        Args:
            source: Dict with 'url' (to zip or git repo).
            dest: Destination directory.
        """
        url = source.get("url", "")
        if url.endswith(".zip"):
            self._download_zip(url, dest)
        else:
            # Assume git clone
            import subprocess
            subprocess.run(
                ["git", "clone", url, str(dest)],
                check=True,
                capture_output=True,
            )
            self._logger.info("  GitHub clone complete: %s", url)

    def _download_zip(self, url: str, dest: Path) -> None:
        """Download and extract a zip file.

        Args:
            url: Download URL for the zip file.
            dest: Destination directory.
        """
        import io
        import zipfile

        import requests

        self._logger.info("  Downloading zip from %s...", url)
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        z.extractall(path=dest)
        self._logger.info("  Zip extracted to %s", dest)

    def _write_hf_labels(
        self, label_path: Path, label_data: Any
    ) -> None:
        """Write HuggingFace label data to YOLO format.

        Args:
            label_path: Output label file path.
            label_data: Label data from HuggingFace dataset item.
                Can be a dict with 'category_id', 'bbox' or similar.
        """
        lines: list[str] = []
        if isinstance(label_data, list):
            for obj in label_data:
                if isinstance(obj, dict):
                    cls_id = obj.get("category_id", 0)
                    bbox = obj.get("bbox", [0, 0, 1, 1])
                    # Assume bbox is [x, y, w, h] normalized
                    xc, yc, w, h = bbox
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines))

    def _remap_dataset(
        self,
        dataset_path: Path,
        mapping: dict[int, int],
        ignored: set[int],
    ) -> tuple[list[Path], list[Path]]:
        """Remap class labels from a dataset to the unified schema v4.

        Scans for images and corresponding labels in the dataset directory,
        remaps class IDs, and returns aligned lists.

        Args:
            dataset_path: Path to the downloaded dataset.
            mapping: Dict mapping original class_id → schema_v4 class_id.
            ignored: Set of original class IDs to ignore.

        Returns:
            Tuple of (image_paths, label_paths) after remapping.
        """
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        # Find all images recursively
        image_paths: list[Path] = []
        label_paths: list[Path] = []

        # Common YOLO dataset structures:
        #   images/train/, images/val/, images/test/  OR
        #   train/images/, train/labels/  OR
        #   flat directory
        candidates = list(dataset_path.rglob("*"))
        img_dir_candidates: list[Path] = []
        lbl_dir_candidates: list[Path] = []

        for candidate in candidates:
            if candidate.is_dir():
                if candidate.name == "images":
                    img_dir_candidates.append(candidate)
                elif candidate.name == "labels":
                    lbl_dir_candidates.append(candidate)

        if not img_dir_candidates:
            # Flat structure — use dataset_path as image dir
            img_dir_candidates = [dataset_path]

        for img_dir in img_dir_candidates:
            # Find corresponding labels dir
            rel_to_dataset = img_dir.relative_to(dataset_path)
            # Try sibling labels dir
            lbl_dir = None
            parent = img_dir.parent
            lbl_candidate = parent / "labels"
            if lbl_candidate.exists():
                lbl_dir = lbl_candidate
            elif lbl_dir_candidates:
                # Match by relative structure
                for ldc in lbl_dir_candidates:
                    if ldc.parent == parent:
                        lbl_dir = ldc
                        break

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in supported_extensions:
                    continue

                # Find corresponding label
                if lbl_dir:
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                else:
                    # Check next to image
                    lbl_path = img_path.with_suffix(".txt")
                    if not lbl_path.exists():
                        # Check in a labels dir parallel to images dir
                        alt_lbl = (
                            img_path.parent.parent
                            / "labels"
                            / f"{img_path.stem}.txt"
                        )
                        if alt_lbl.exists():
                            lbl_path = alt_lbl

                objects = _parse_yolo_label(lbl_path)
                if not objects:
                    continue

                # Remap classes
                remapped: list[tuple[int, float, float, float, float]] = []
                skip = False
                for cls_id, xc, yc, w, h in objects:
                    if cls_id in ignored:
                        continue
                    new_id = mapping.get(cls_id)
                    if new_id is None:
                        self._logger.debug(
                            "  Skipping unmapped class %d in %s",
                            cls_id,
                            img_path.name,
                        )
                        continue
                    remapped.append((new_id, xc, yc, w, h))

                if not remapped:
                    continue

                # Write remapped label next to image (will be moved during export)
                remapped_lbl = img_path.with_suffix(".txt")
                _write_yolo_label(remapped_lbl, remapped)

                image_paths.append(img_path)
                label_paths.append(remapped_lbl)

        return image_paths, label_paths

    def _deduplicate(self, images: list[Path]) -> list[int]:
        """Deduplicate images by perceptual dHash.

        Computes dHash for each image and keeps only the first occurrence
        of each hash value.

        Args:
            images: List of image paths.

        Returns:
            List of indices to keep (first occurrence per hash).
        """
        seen_hashes: dict[int, int] = {}  # hash → first index
        keep_indices: list[int] = []

        for idx, img_path in enumerate(tqdm(images, desc="Deduplicating")):
            try:
                h = _compute_dhash(img_path)
                if h not in seen_hashes:
                    seen_hashes[h] = idx
                    keep_indices.append(idx)
                else:
                    self._logger.debug(
                        "  Duplicate (dHash): %s matches %s",
                        img_path.name,
                        images[seen_hashes[h]].name,
                    )
            except Exception as exc:
                self._logger.warning(
                    "  Could not hash %s: %s",
                    img_path.name,
                    exc,
                )
                keep_indices.append(idx)

        return keep_indices

    def _stratified_split(
        self,
        images: list[Path],
        labels: list[Path],
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Split dataset stratified by class distribution 80/10/10.

        Uses train_test_split twice to get train/val/test sets while
        maintaining class balance.

        Args:
            images: List of all image paths.
            labels: List of corresponding label paths.

        Returns:
            Tuple of (train_images, val_images, test_images).
            Labels follow the same split indices.
        """
        # Compute class distribution per image for stratification
        class_ids: list[int] = []
        for lbl_path in labels:
            objects = _parse_yolo_label(lbl_path)
            # Use first class for stratification
            if objects:
                class_ids.append(objects[0][0])
            else:
                class_ids.append(-1)

        # First split: train vs temp (val + test)
        train_idx, temp_idx = train_test_split(
            range(len(images)),
            test_size=0.2,
            random_state=42,
            stratify=class_ids,
        )

        # Second split: val vs test (50/50 of the 20%)
        temp_class_ids = [class_ids[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=42,
            stratify=temp_class_ids,
        )

        train_images = [images[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        test_images = [images[i] for i in test_idx]

        self._logger.info(
            "  Split: train=%d val=%d test=%d",
            len(train_images),
            len(val_images),
            len(test_images),
        )

        return train_images, val_images, test_images

    def _export_yolo(
        self,
        train_images: list[Path],
        val_images: list[Path],
        test_images: list[Path],
        all_labels: list[Path],
    ) -> None:
        """Export dataset in YOLO format.

        Creates the directory structure:
            images/train/
            images/val/
            images/test/
            labels/train/
            labels/val/
            labels/test/

        Images and labels are copied/renamed with a unified prefix to
        avoid name collisions between source datasets.

        Args:
            train_images: List of training image paths.
            val_images: List of validation image paths.
            test_images: List of test image paths.
            all_labels: List of all label paths (index-aligned with full set).
        """
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        # Build a mapping of original image path → label path
        # We need to reconstruct label paths from the images since
        # labels were written alongside images during remapping
        label_map: dict[str, Path] = {}
        for lbl_path in all_labels:
            label_map[lbl_path.stem] = lbl_path

        for split_name, split_images in splits.items():
            img_out = self._output_dir / "images" / split_name
            lbl_out = self._output_dir / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_images, desc=f"Exporting {split_name}"):
                # Unified filename to avoid collisions
                stem = f"{img_path.parent.parent.name}_{img_path.stem}"
                dst_img = img_out / f"{stem}{img_path.suffix}"
                shutil.copy2(img_path, dst_img)

                # Find corresponding label
                lbl_src = img_path.with_suffix(".txt")
                if lbl_src.exists():
                    dst_lbl = lbl_out / f"{stem}.txt"
                    shutil.copy2(lbl_src, dst_lbl)

    def _generate_yaml(self) -> None:
        """Generate dataset-v4.yaml configuration file.

        Creates a YAML file compatible with ultralytics YOLO training,
        pointing to the exported dataset structure.
        """
        yaml_path = self._output_dir / "dataset-v4.yaml"
        yaml_content = f"""# Munin v4 — Unified PPE Dataset (13 classes)
# Auto-generated by DatasetBuilder

path: {self._output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 13
names:
  0: person
  1: hardhat
  2: safety_vest
  3: gloves
  4: safety_glasses
  5: safety_boots
  6: harness
  7: mask
  8: no_hardhat
  9: no_safety_vest
  10: no_gloves
  11: no_safety_boots
  12: no_safety_glasses
"""
        yaml_path.write_text(yaml_content.lstrip())
        self._logger.info("  Generated dataset-v4.yaml at %s", yaml_path)

    def _validate_images_labels(
        self, images: list[Path], labels: list[Path]
    ) -> None:
        """Validate that images and labels are consistent.

        Checks that each image has a corresponding label and vice versa.

        Args:
            images: List of image paths.
            labels: List of label paths.
        """
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_labels = img_stems - lbl_stems
        if missing_labels:
            self._logger.warning(
                "  %d images without labels (will be skipped)",
                len(missing_labels),
            )


def default_dataset_configs() -> list[dict[str, Any]]:
    """Return the default list of 7 PPE dataset configurations.

    Returns:
        List of dataset config dicts for building Munin v4.
    """
    return [
        {
            "name": "youcefs",
            "source": {
                "type": "roboflow",
                "api_key": "${ROBOFLOW_API_KEY}",
                "workspace": "youcefs",
                "project": "construction-ppe",
                "version": 1,
                "format": "yolov8",
            },
            "mapping": DATASET_MAPPINGS["youcefs"],
        },
        {
            "name": "keremberke",
            "source": {
                "type": "huggingface",
                "repo_id": "keremberke/ppe-detection",
                "subset": "yolov8",
                "split": "train",
            },
            "mapping": DATASET_MAPPINGS["keremberke"],
        },
        {
            "name": "skcet",
            "source": {
                "type": "roboflow",
                "api_key": "${ROBOFLOW_API_KEY}",
                "workspace": "skcet",
                "project": "ppe-detection-v9rfk",
                "version": 1,
                "format": "yolov8",
            },
            "mapping": DATASET_MAPPINGS["skcet"],
        },
        {
            "name": "shwd",
            "source": {
                "type": "github",
                "url": "https://github.com/ggiscan/SHWD",
            },
            "mapping": DATASET_MAPPINGS["shwd"],
        },
        {
            "name": "construction2",
            "source": {
                "type": "roboflow",
                "api_key": "${ROBOFLOW_API_KEY}",
                "workspace": "construction",
                "project": "ppe-construction-2",
                "version": 1,
                "format": "yolov8",
            },
            "mapping": DATASET_MAPPINGS["construction2"],
        },
        {
            "name": "voxdroid",
            "source": {
                "type": "github",
                "url": "https://github.com/VoxDroid/PPE-Detection",
            },
            "mapping": DATASET_MAPPINGS["voxdroid"],
        },
        {
            "name": "keremberke_small",
            "source": {
                "type": "huggingface",
                "repo_id": "keremberke/ppe-detection-small",
                "subset": "yolov8",
                "split": "train",
            },
            "mapping": DATASET_MAPPINGS["keremberke_small"],
        },
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Munin v4 unified PPE dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch/datasets/munin-v4"),
        help="Output directory for the unified dataset",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download (use cached raw datasets)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    builder = DatasetBuilder(args.output)
    configs = default_dataset_configs()

    if args.no_download:
        for cfg in configs:
            cfg["source"]["type"] = "local"
            cfg["source"]["path"] = str(
                Path(args.output) / "raw" / cfg["name"]
            )

    builder.build(configs)
