"""Smoke test end-to-end para Munin.

Uso:
    1. Iniciar API: python main.py
    2. En otra terminal: python tests/smoke_test.py

Verifica el pipeline completo: video → API → pipeline → AgentDecision.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000/api/v1"
DEMO_VIDEO = Path(__file__).parent.parent / "demo" / "video.mp4"
TIMEOUT_SECONDS = 300  # 5 minutos
POLL_INTERVAL = 5  # segundos


def check_api_health() -> bool:
    """Verifica que la API está corriendo."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            logger.info("API healthy: %s", response.json())
            return True
    except requests.ConnectionError:
        pass
    logger.error("API no responde en %s. ¿Está corriendo?", API_BASE)
    return False


def ensure_demo_video() -> bool:
    """Asegura que existe video demo."""
    if DEMO_VIDEO.exists() and DEMO_VIDEO.stat().st_size > 0:
        logger.info("Video demo existe: %s (%d KB)", DEMO_VIDEO, DEMO_VIDEO.stat().st_size // 1024)
        return True
    logger.info("Generando video demo sintético...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from demo.download_video import generate_synthetic_video
    return generate_synthetic_video(str(DEMO_VIDEO))


def upload_video(video_path: Path) -> str | None:
    """Sube video a la API y retorna job_id."""
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        data = {"zone_id": "extraccion"}
        try:
            response = requests.post(f"{API_BASE}/analyze", files=files, data=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                logger.info("Video uploaded. Job ID: %s", result.get("job_id"))
                return result.get("job_id")
            else:
                logger.error("Upload failed: %d %s", response.status_code, response.text)
        except Exception as e:
            logger.error("Upload error: %s", e)
    return None


def poll_results(job_id: str) -> dict | None:
    """Hace polling hasta que el job complete."""
    start = time.time()
    while time.time() - start < TIMEOUT_SECONDS:
        try:
            response = requests.get(f"{API_BASE}/analyze/{job_id}", timeout=30)
            if response.status_code == 200:
                result = response.json()
                status = result.get("status")
                if status == "completed":
                    logger.info("Job completed in %.1fs", time.time() - start)
                    return result
                elif status == "failed":
                    logger.error("Job failed: %s", result.get("error"))
                    return result
                else:
                    logger.info("Job status: %s (%.0fs elapsed)", status, time.time() - start)
        except Exception as e:
            logger.warning("Poll error: %s", e)
        time.sleep(POLL_INTERVAL)
    logger.error("Timeout after %ds", TIMEOUT_SECONDS)
    return None


def validate_results(result: dict) -> bool:
    """Valida que los resultados sean correctos."""
    decisions = result.get("decisions", [])
    if not decisions:
        logger.error("No decisions returned")
        return False
    logger.info("Decisions received: %d", len(decisions))
    violations = [d for d in decisions if d.get("tipo_violacion") != "SIN_VIOLACION"]
    logger.info("Violations detected: %d", len(violations))
    for d in decisions[:5]:
        logger.info("  - zona=%s, tipo=%s, riesgo=%s, confianza=%.2f",
                     d.get("zona"), d.get("tipo_violacion"),
                     d.get("nivel_riesgo"), d.get("confianza", 0))
    if len(decisions) > 0:
        logger.info("SMOKE TEST PASSED ✅")
        return True
    logger.error("SMOKE TEST FAILED ❌")
    return False


def main() -> None:
    """Ejecuta smoke test end-to-end."""
    logger.info("=== Munin Smoke Test ===")
    if not check_api_health():
        sys.exit(1)
    if not ensure_demo_video():
        sys.exit(1)
    job_id = upload_video(DEMO_VIDEO)
    if not job_id:
        sys.exit(1)
    result = poll_results(job_id)
    if not result:
        sys.exit(1)
    if not validate_results(result):
        sys.exit(1)


if __name__ == "__main__":
    main()
