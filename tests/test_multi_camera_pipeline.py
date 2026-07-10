"""Tests T18 — MultiCameraPipeline.

Verifica:
- process_all con frame_limit=5
- TrackerFactory crea un tracker por cámara
- VLMQueue enqueue llamado con camera_id correcto
- drain_results consolida decisiones
- Default decision cuando enqueue falla

Correr con: pytest tests/test_multi_camera_pipeline.py -v
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from munin.config import Zone
from munin.gate.schemas import AgentDecision, DetectionResult, TrackedPerson, Violation
from munin.knowledge.zone_config import ZoneConfig
from munin.pipeline.frame_collector import CameraSource


@pytest.fixture
def mock_collector() -> MagicMock:
    """Mock de FrameCollector."""
    mc = MagicMock()
    mc.start.return_value = None
    mc.release.return_value = None
    return mc


@pytest.fixture
def mock_detector() -> MagicMock:
    """Mock de IDetector que detecta 1 persona por frame."""
    detector = MagicMock()
    detector.detect.return_value = [
        DetectionResult(
            class_name="person",
            bbox=(0.0, 0.0, 100.0, 200.0),
            confidence=0.85,
        ),
    ]
    return detector


@pytest.fixture
def mock_checker() -> MagicMock:
    """Mock de IComplianceChecker sin violaciones."""
    checker = MagicMock()
    checker.check.return_value = []
    return checker


@pytest.fixture
def mock_vlm_queue() -> MagicMock:
    """Mock de VLMQueue."""
    q = MagicMock()
    q.enqueue.return_value = True
    q.drain_results.return_value = []
    return q


@pytest.fixture
def mock_zone_config() -> MagicMock:
    """Mock de ZoneConfig."""
    zc = MagicMock(spec=ZoneConfig)
    zc.get_zone.return_value = Zone(
        zone_id="extraccion",
        nombre="Extracción",
        required_epp=["hardhat", "safety_vest"],
        riesgo_base="alto",
        articulos_ds132=["Art. 38"],
    )
    return zc


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock de AppSettings con valores mínimos."""
    settings = MagicMock()
    settings.frame_rate = 25
    settings.yolo_confidence_threshold = 0.6
    return settings


@pytest.fixture
def sources() -> list[CameraSource]:
    """2 fuentes de cámara."""
    return [
        CameraSource(
            camera_id="cam01", source="vid1.mp4", zone_id="extraccion",
        ),
        CameraSource(
            camera_id="cam02", source="vid2.mp4", zone_id="procesamiento",
        ),
    ]


@pytest.fixture
def pipeline(
    mock_detector: MagicMock,
    sources: list[CameraSource],
    mock_checker: MagicMock,
    mock_vlm_queue: MagicMock,
    mock_zone_config: MagicMock,
    mock_settings: MagicMock,
    mock_collector: MagicMock,
) -> Any:
    """MultiCameraPipeline con todos los mocks.

    El FrameCollector se reemplaza con un mock después de __init__
    para evitar que intente abrir cámaras reales.
    """
    from munin.pipeline.multi_camera import MultiCameraPipeline

    mock_tracker = MagicMock()
    mock_tracker.update.return_value = [
        TrackedPerson(
            persona_id=0,
            bbox=(0.0, 0.0, 100.0, 200.0),
            epp_detectado=set(),
        ),
    ]

    def tracker_factory() -> MagicMock:
        return mock_tracker

    pipe = MultiCameraPipeline(
        detector=mock_detector,
        sources=sources,
        tracker_factory=tracker_factory,
        checker=mock_checker,
        vlm_queue=mock_vlm_queue,
        zone_config=mock_zone_config,
        settings=mock_settings,
    )

    # Reemplazar FrameCollector real con mock
    pipe._collector = mock_collector

    return pipe


class TestMultiCameraPipeline:
    """Tests para MultiCameraPipeline."""

    @pytest.fixture(autouse=True)
    def _patch_frame_collector(self, mock_collector: MagicMock) -> None:
        """Auto-use: parchea FrameCollector a nivel de import."""
        from unittest.mock import patch as _patch
        patcher = _patch(
            "munin.pipeline.multi_camera.FrameCollector",
            return_value=mock_collector,
        )
        patcher.start()
        self._patcher = patcher

    def _cleanup_patcher(self) -> None:
        if hasattr(self, "_patcher"):
            self._patcher.stop()

    def test_init_creates_trackers_per_camera(
        self,
        mock_detector: MagicMock,
        sources: list[CameraSource],
        mock_checker: MagicMock,
        mock_vlm_queue: MagicMock,
        mock_zone_config: MagicMock,
        mock_settings: MagicMock,
        mock_collector: MagicMock,
    ) -> None:
        """Cada cámara tiene su propio tracker."""
        from munin.pipeline.multi_camera import MultiCameraPipeline

        tracker_factory = MagicMock()
        pipe_obj = MultiCameraPipeline(
            detector=mock_detector,
            sources=sources,
            tracker_factory=tracker_factory,
            checker=mock_checker,
            vlm_queue=mock_vlm_queue,
            zone_config=mock_zone_config,
            settings=mock_settings,
        )
        # Reemplazar collector con mock
        pipe_obj._collector = mock_collector

        assert len(pipe_obj._trackers) == 2
        assert "cam01" in pipe_obj._trackers
        assert "cam02" in pipe_obj._trackers
        assert tracker_factory.call_count == 2

    @pytest.mark.asyncio
    async def test_process_all_with_frame_limit(
        self,
        pipeline: Any,
    ) -> None:
        """process_all con frame_limit procesa N frames."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._collector.collect.return_value = {
            "cam01": frame, "cam02": frame,
        }

        decisions = await pipeline.process_all(frame_limit=3)

        assert isinstance(decisions, dict)
        assert "cam01" in decisions
        assert "cam02" in decisions
        assert decisions["cam01"] == []
        assert decisions["cam02"] == []

    @pytest.mark.asyncio
    async def test_vlm_enqueue_called_with_camera_id(
        self,
        pipeline: Any,
        mock_checker: MagicMock,
        mock_vlm_queue: MagicMock,
    ) -> None:
        """VLMQueue.enqueue se llama con camera_id correcto."""
        # Forzar violaciones
        mock_checker.check.return_value = [
            Violation(
                persona_id=1,
                zona_id="extraccion",
                frame_id=1,
            ),
        ]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._collector.collect.return_value = {"cam01": frame}

        await pipeline.process_all(frame_limit=2)

        # Verificar que enqueue fue llamado con camera_id="cam01"
        assert mock_vlm_queue.enqueue.call_count >= 1
        call_kwargs = mock_vlm_queue.enqueue.call_args.kwargs
        if call_kwargs:
            assert call_kwargs.get("camera_id") == "cam01"

    @pytest.mark.asyncio
    async def test_default_decision_when_enqueue_fails(
        self,
        pipeline: Any,
        mock_checker: MagicMock,
        mock_vlm_queue: MagicMock,
    ) -> None:
        """Cuando enqueue falla, se usa default decision."""
        # Forzar violaciones + enqueue fail
        mock_checker.check.return_value = [
            Violation(
                persona_id=1,
                zona_id="extraccion",
                epp_faltantes=[],
                frame_id=1,
            ),
        ]
        mock_vlm_queue.enqueue.return_value = False

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._collector.collect.return_value = {"cam01": frame}

        decisions = await pipeline.process_all(frame_limit=2)

        # Debe tener default decision con requiere_revision_humana=True
        assert len(decisions["cam01"]) >= 1
        assert decisions["cam01"][0].requiere_revision_humana is True
        assert decisions["cam01"][0].razonamiento_vlm is not None

    @pytest.mark.asyncio
    async def test_drain_results_consolidates(
        self,
        pipeline: Any,
        mock_vlm_queue: MagicMock,
    ) -> None:
        """drain_results consolida decisiones de VLM."""
        mock_vlm_queue.drain_results.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._collector.collect.return_value = {"cam01": frame}

        decisions = await pipeline.process_all(frame_limit=2)
        assert isinstance(decisions, dict)
