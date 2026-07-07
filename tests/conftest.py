"""Configuración de pytest para Munin.

Asegura que el paquete `munin` sea importable desde los tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Añadir el directorio padre al sys.path para que `munin` sea importable
sys.path.insert(0, str(Path(__file__).parent.parent))
