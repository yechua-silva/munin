"""Alias para compatibilidad — YOLODetector renombrado a TwoModelDetector.

Este archivo existe para mantener compatibilidad con imports legacy.
Nuevo código debe usar: from munin.pipeline.two_model_detector import TwoModelDetector

Deprecated desde v3. Ver ADR-018.
"""
from __future__ import annotations

from munin.pipeline.two_model_detector import (
    TwoModelDetector as YOLODetector,
    PPE_CLASS_MAP,
    COCO_PERSON_ID,
)

__all__ = ["YOLODetector", "PPE_CLASS_MAP", "COCO_PERSON_ID"]
