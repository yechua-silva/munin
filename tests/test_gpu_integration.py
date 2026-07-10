"""Tests T21 — Integración GPU ROCm/CUDA.

Estos tests REQUIEREN GPU real (AMD MI300X con ROCm o NVIDIA con CUDA).
Se saltan automáticamente si no hay GPU disponible.

Correr con:
    pytest tests/test_gpu_integration.py -v -m gpu
    pytest tests/ -v -m "not gpu"  # skip GPU tests

Markers:
    - gpu: requiere GPU real
"""
from __future__ import annotations

import pytest

# Skip all tests in this module if no GPU available
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_GPU, reason="No GPU available"),
]


class TestGPUIntegration:
    """Tests que REQUIEREN GPU AMD MI300X o NVIDIA real.

    Todos los tests del módulo se skippean si no hay GPU.
    """

    def test_yolo_inference_cuda(self) -> None:
        """YOLO.predict() funciona en cuda:0 con NMS patch aplicado.

        Verifica que la integración ultralytics + ROCm/CUDA
        funcione correctamente con el NMS patch de munin.
        """
        from munin.rocm_patch import apply_nms_patch
        apply_nms_patch()

        from ultralytics import YOLO
        import numpy as np

        model = YOLO("yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model.predict(frame, device="cuda:0", verbose=False)
        assert results is not None
        assert len(results) > 0

    def test_vlm_amd_endpoint(self) -> None:
        """vLLM endpoint responde en localhost:8000.

        Verifica que el servidor vLLM on-premise esté operativo
        y que la API REST responda correctamente.
        """
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        models = client.models.list()
        assert len(models.data) > 0

    def test_vlm_json_mode_multimodal(self) -> None:
        """response_format json_object funciona con VLM multimodal.

        Verifica que el endpoint vLLM soporte el modo JSON structure
        output para respuestas estructuradas con AgentDecision.
        """
        from openai import OpenAI
        import base64
        import io
        from PIL import Image

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        models = client.models.list()
        model_id = models.data[0].id

        # Frame sintético rojo
        img = Image.new("RGB", (640, 480), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        'Respond as JSON: {"color": "red", '
                        '"detected": false}'
                    ),
                },
            ]}],
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        assert r.choices[0].message.content is not None
        import json
        parsed = json.loads(r.choices[0].message.content)
        assert "color" in parsed

    def test_yolo_batch_inference(self) -> None:
        """YOLO.predict() con batch de 4 frames en GPU.

        Verifica que el batch processing funcione correctamente
        en GPU para modo multi-cámara.
        """
        from ultralytics import YOLO
        import numpy as np

        model = YOLO("yolov8n.pt")
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        results = model.predict(
            frames, device="cuda:0", verbose=False,
        )
        assert results is not None
        assert len(results) == 4

    def test_torch_cuda_available(self) -> None:
        """torch.cuda.is_available() es True en GPU."""
        import torch
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= 1
        props = torch.cuda.get_device_properties(0)
        # AMD MI300X tiene 192GB HBM3
        assert props.total_memory > 0

    def test_torch_rocm_version(self) -> None:
        """Verifica ROCm version si está disponible."""
        import torch
        if hasattr(torch, "version") and hasattr(torch.version, "hip"):
            hip_ver = torch.version.hip
            assert hip_ver is not None
            assert "6." in hip_ver or "5." in hip_ver
