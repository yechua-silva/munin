from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patched: bool = False


def apply_nms_patch() -> None:
    """Apply NMS monkey-patch for ROCm compatibility.

    Replaces torchvision.ops.nms with TorchNMS.nms (pure PyTorch)
    to avoid missing HIP kernel in ROCm. Idempotent.
    """
    global _patched
    if _patched:
        logger.debug("NMS patch already applied, skipping")
        return

    try:
        import torch

        if not torch.cuda.is_available():
            logger.debug("No CUDA/HIP available, NMS patch not needed")
            return

        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if not is_rocm:
            logger.debug("Not ROCm, NMS patch not needed")
            return

        from ultralytics.utils.nms import TorchNMS
        import torchvision.ops

        original_nms = torchvision.ops.nms
        torchvision.ops.nms = TorchNMS.nms

        if hasattr(torchvision.ops, 'boxes') and hasattr(torchvision.ops.boxes, 'nms'):
            torchvision.ops.boxes.nms = TorchNMS.nms

        _patched = True
        logger.info("NMS patch applied: torchvision.ops.nms -> TorchNMS.nms (ROCm)")
    except ImportError as e:
        logger.warning("Cannot apply NMS patch (missing dependency): %s", e)
    except Exception as e:
        logger.error("Failed to apply NMS patch: %s", e)


__all__ = ["apply_nms_patch"]
