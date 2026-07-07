"""Script para obtener video demo para Munin.

Intenta descargar un video libre de derechos de Pexels.
Si no hay internet, genera un video sintético con OpenCV.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_from_pexels(output_path: str) -> bool:
    """Descarga video de minería desde Pexels (libre de derechos)."""
    try:
        import requests

        # Pexels API endpoint (sin API key, usar URL directa de descarga)
        # Video de construcción/minería libre: https://www.pexels.com/search/videos/mining/
        urls = [
            "https://videos.pexels.com/video-files/..."
        ]
        for url in urls:
            try:
                response = requests.get(url, timeout=30, stream=True)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info("Video descargado desde Pexels: %s", output_path)
                    return True
            except Exception:
                continue
    except ImportError:
        logger.warning("requests no instalado, saltando descarga")
    return False


def generate_synthetic_video(output_path: str) -> bool:
    """Genera video sintético con OpenCV simulando faena minera."""
    import cv2 as cv
    import numpy as np

    width, height = 1280, 720
    fps = 25
    duration_sec = 30
    total_frames = fps * duration_sec

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Fondo: gradiente marrón (tierra/minería)
        for y in range(height):
            frame[y, :] = [40 + y // 10, 60 + y // 8, 80 + y // 6]

        # Persona 1: CON casco (verde bbox) - izquierda
        x1, y1, x2, y2 = 200, 300, 350, 600
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.rectangle(frame, (x1, y1 - 40), (x2, y1), (255, 0, 0), -1)  # Casco azul
        cv.putText(frame, "Person 1 (EPP OK)", (x1, y1 - 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Persona 2: SIN casco (rojo bbox) - derecha, se mueve
        offset = int(50 * np.sin(i * 0.05))
        x1, y1, x2, y2 = 700 + offset, 280, 850 + offset, 580
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.putText(frame, "Person 2 (NO hardhat!)", (x1, y1 - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Zona label
        cv.putText(frame, "ZONA: EXTRACCION", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Timestamp
        cv.putText(frame, f"Frame {i}/{total_frames}", (10, height - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    logger.info("Video sintético generado: %s (%d frames, %ds)", output_path, total_frames, duration_sec)
    return True


def main() -> None:
    """Obtiene video demo para Munin."""
    output_dir = Path(__file__).parent
    output_path = str(output_dir / "video.mp4")

    if Path(output_path).exists():
        logger.info("Video ya existe: %s", output_path)
        return

    logger.info("Intentando descargar video de Pexels...")
    if download_from_pexels(output_path):
        return

    logger.info("Descarga falló. Generando video sintético...")
    if generate_synthetic_video(output_path):
        logger.info("Video demo listo: %s", output_path)
    else:
        logger.error("No se pudo obtener video demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
