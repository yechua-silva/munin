"""Munin — Industrial Vision Agent.

Sistema de visión por computadora para detección de violaciones de EPP
en faenas mineras, utilizando YOLO + VLM + Pydantic Gate.

Este módulo contiene el pipeline de procesamiento de video, agentes VLM,
configuración, schemas de datos, y utilidades.
"""

from __future__ import annotations

from munin.rocm_patch import apply_nms_patch

apply_nms_patch()

__version__ = "1.0.0"
