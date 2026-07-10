"""Train YOLOv8x for Munin v4 — 13-class PPE detection.

Fine-tunes YOLOv8x on the unified Munin v4 dataset with
optimized hyperparameters for PPE detection in industrial settings.

Usage:
    python scripts/train_v4.py --data /scratch/datasets/munin-v4/dataset-v4.yaml
    python scripts/train_v4.py --data dataset-v4.yaml --epochs 200 --imgsz 1280
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train_v4(
    data: str,
    weights: str = "yolov8x.pt",
    epochs: int = 150,
    imgsz: int = 1280,
    batch: int = -1,
    device: str = "cuda:0",
    project: str = "/scratch/runs/train",
    name: str = "yolov8x-ppe-v4",
    resume: bool = False,
) -> None:
    """Fine-tune YOLOv8x for Munin v4 PPE detection (13 classes).

    Hyperparameters are optimized for:
    - Small/medium PPE objects (hardhats, glasses, gloves)
    - Dual-class mode (positive + negative classes)
    - Industrial/mining environments with varying lighting

    Args:
        data: Path to dataset-v4.yaml.
        weights: Pretrained weights (yolov8x.pt from COCO).
        epochs: Number of training epochs.
        imgsz: Image size for training (default 1280 for small object detection).
        batch: Batch size (-1 = auto-detect based on GPU memory).
        device: GPU device (e.g. "cuda:0", "cuda:0,1" for multi-GPU).
        project: Output project directory for runs.
        name: Experiment name (subdirectory under project).
        resume: Resume from last checkpoint if available.

    Raises:
        FileNotFoundError: If data yaml or weights path does not exist.
    """
    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    if not resume:
        weights_path = Path(weights)
        if not weights_path.exists() and not weights.startswith("yolov8"):
            raise FileNotFoundError(f"Weights not found: {weights}")

    from ultralytics import YOLO

    if resume:
        # Resume from last checkpoint in project/name
        last_ckpt = Path(project) / name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint to resume from: {last_ckpt}"
            )
        model = YOLO(str(last_ckpt))
        logger.info("Resuming training from checkpoint: %s", last_ckpt)
    else:
        model = YOLO(weights)
        logger.info("Starting training from weights: %s", weights)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        freeze=10,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        cls_pw=0.5,
        hsv_h=0.03,
        hsv_s=0.8,
        hsv_v=0.6,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        mixup=0.2,
        amp=True,
        workers=8,
        save_period=10,
        val=True,
        plots=True,
        patience=50,
        seed=42,
        deterministic=True,
        single_cls=False,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    logger.info(
        "Training complete. Best weights: %s",
        best_weights,
    )
    logger.info(
        "Results: %s/results.csv", Path(project) / name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8x for Munin v4 PPE detection (13 classes)"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset-v4.yaml",
    )
    parser.add_argument(
        "--weights",
        default="yolov8x.pt",
        help="Pretrained weights path or name (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size (default: 1280 for small PPE detection)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 = auto-detect, default: -1)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="GPU device(s) (default: cuda:0)",
    )
    parser.add_argument(
        "--project",
        default="/scratch/runs/train",
        help="Output project directory (default: /scratch/runs/train)",
    )
    parser.add_argument(
        "--name",
        default="yolov8x-ppe-v4",
        help="Experiment name (default: yolov8x-ppe-v4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_v4(
        data=args.data,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
