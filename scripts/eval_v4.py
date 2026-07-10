"""Evaluate Munin v4 — YOLOv8x 13-class PPE detection model.

Evaluates a fine-tuned model on the test split and optionally
compares against a baseline model to measure improvement.

Usage:
    # Single model evaluation
    python scripts/eval_v4.py --model runs/train/yolov8x-ppe-v4/weights/best.pt --data dataset-v4.yaml

    # Baseline comparison
    python scripts/eval_v4.py --model runs/train/yolov8x-ppe-v4/weights/best.pt --baseline yolov8x.pt --data dataset-v4.yaml

    # Save results to JSON
    python scripts/eval_v4.py --model best.pt --data dataset-v4.yaml --output results.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    data: str,
    conf: float = 0.5,
    iou: float = 0.5,
    device: str = "cuda:0",
    imgsz: int = 1280,
    half: bool = False,
) -> dict[str, Any]:
    """Evaluate a YOLO model and return detailed metrics.

    Runs validation on the test split, collecting mAP, precision,
    recall, and per-class metrics.

    Args:
        model_path: Path to .pt weights file.
        data: Path to dataset yaml configuration.
        conf: Confidence threshold for evaluation (default: 0.5).
        iou: IoU threshold for NMS (default: 0.5).
        device: GPU device for inference (default: cuda:0).
        imgsz: Image size for evaluation (default: 1280).
        half: Use FP16 half precision (default: False).

    Returns:
        Dict with overall mAP metrics and per-class breakdown:
        {
            "model": "path/to/model.pt",
            "mAP50": 0.85,
            "mAP50_95": 0.62,
            "precision": 0.88,
            "recall": 0.83,
            "f1_score": 0.85,
            "per_class": {
                "hardhat": {"mAP50": 0.92, "mAP50_95": 0.71},
                ...
            }
        }

    Raises:
        FileNotFoundError: If model_path or data path does not exist.
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path_obj = Path(data)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    from ultralytics import YOLO

    model = YOLO(str(model_path_obj))

    logger.info(
        "Evaluating model: %s on %s (conf=%.2f, iou=%.2f)",
        model_path,
        data,
        conf,
        iou,
    )

    results = model.val(
        data=data,
        conf=conf,
        iou=iou,
        split="test",
        plots=True,
        device=device,
        imgsz=imgsz,
        half=half,
        verbose=False,
    )

    f1 = (
        2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-12)
    )

    metrics: dict[str, Any] = {
        "model": str(model_path_obj.resolve()),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1_score": float(f1),
        "per_class": {},
    }

    names = model.names
    ap50_list = results.box.ap50
    ap_list = results.box.ap

    for i in range(len(ap50_list)):
        class_name = names.get(i, f"class_{i}")
        metrics["per_class"][class_name] = {
            "mAP50": float(ap50_list[i]),
            "mAP50_95": float(ap_list[i]) if i < len(ap_list) else 0.0,
        }

    logger.info("mAP@50: %.4f | mAP@50:95: %.4f", metrics["mAP50"], metrics["mAP50_95"])
    logger.info("Precision: %.4f | Recall: %.4f | F1: %.4f", metrics["precision"], metrics["recall"], metrics["f1_score"])

    return metrics


def compare_models(
    baseline_path: str,
    finetuned_path: str,
    data: str,
    conf: float = 0.5,
    iou: float = 0.5,
) -> dict[str, Any]:
    """Compare baseline vs fine-tuned model performance.

    Evaluates both models on the same dataset and computes
    per-metric improvements.

    Args:
        baseline_path: Path to baseline model .pt (e.g., COCO yolov8x.pt).
        finetuned_path: Path to fine-tuned model .pt.
        data: Path to dataset yaml.
        conf: Confidence threshold.
        iou: IoU threshold.

    Returns:
        Dict with baseline and fine-tuned metrics plus improvements:
        {
            "baseline": {...},
            "finetuned": {...},
            "improvement": {
                "mAP50": 0.45,
                "mAP50_95": 0.35,
                "precision": 0.20,
                "recall": 0.30,
            }
        }
    """
    logger.info("=== Evaluating BASELINE model ===")
    baseline = evaluate_model(baseline_path, data, conf, iou)

    logger.info("=== Evaluating FINE-TUNED model ===")
    finetuned = evaluate_model(finetuned_path, data, conf, iou)

    improvement = {
        "mAP50": finetuned["mAP50"] - baseline["mAP50"],
        "mAP50_95": finetuned["mAP50_95"] - baseline["mAP50_95"],
        "precision": finetuned["precision"] - baseline["precision"],
        "recall": finetuned["recall"] - baseline["recall"],
        "f1_score": finetuned["f1_score"] - baseline["f1_score"],
    }

    comparison: dict[str, Any] = {
        "baseline": baseline,
        "finetuned": finetuned,
        "improvement": improvement,
    }

    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "  %-20s %8s %8s %8s",
        "Metric", "Baseline", "Finetuned", "Delta",
    )
    logger.info("  " + "-" * 48)
    for metric in ["mAP50", "mAP50_95", "precision", "recall", "f1_score"]:
        b = baseline.get(metric, 0)
        f = finetuned.get(metric, 0)
        d = improvement.get(metric, 0)
        logger.info("  %-20s %8.4f %8.4f %+8.4f", metric, b, f, d)
    logger.info("=" * 60)

    return comparison


def _print_metrics(metrics: dict[str, Any]) -> None:
    """Print metrics table to stdout.

    Args:
        metrics: Metrics dict from evaluate_model().
    """
    print(f"\nModel: {metrics['model']}")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'mAP@50':<20} {metrics['mAP50']:<10.4f}")
    print(f"{'mAP@50:95':<20} {metrics['mAP50_95']:<10.4f}")
    print(f"{'Precision':<20} {metrics['precision']:<10.4f}")
    print(f"{'Recall':<20} {metrics['recall']:<10.4f}")
    print(f"{'F1 Score':<20} {metrics['f1_score']:<10.4f}")
    print(f"\n{'Class':<20} {'mAP@50':<10} {'mAP@50:95':<10}")
    print("-" * 40)
    for class_name, class_metrics in sorted(metrics["per_class"].items()):
        print(
            f"{class_name:<20} {class_metrics['mAP50']:<10.4f} "
            f"{class_metrics['mAP50_95']:<10.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Munin v4 — YOLOv8x 13-class PPE model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned .pt weights",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset-v4.yaml",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline model path for comparison (e.g., yolov8x.pt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="GPU device (default: cuda:0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for evaluation (default: 1280)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half precision",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.baseline:
        result = compare_models(
            args.baseline,
            args.model,
            args.data,
            conf=args.conf,
            iou=args.iou,
        )
    else:
        result = evaluate_model(
            args.model,
            args.data,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            half=args.half,
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Results saved to %s", output_path)
        print(f"\nResults saved to: {output_path}")
    else:
        print(json.dumps(result, indent=2))
