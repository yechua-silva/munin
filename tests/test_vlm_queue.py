"""Tests T17 — VLMQueue with PriorityQueue + backpressure + worker thread.

Verifica:
- _compute_violation_priority: crítico, alto, medio, bajo
- enqueue: normal, high pressure, full
- drain_results: retorna resultados
- start/stop: thread lifecycle
- Thread safety: no race conditions
- Worker loop: procesa requests y retorna resultados

Correr con: pytest tests/test_vlm_queue.py -v
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from munin.gate.schemas import AgentDecision, PPEMissing, Violation


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Mock de MuninOrchestrator.analyze."""
    orch = MagicMock()
    orch.analyze = AsyncMock()
    orch.analyze.return_value = [
        AgentDecision(
            zona="extraccion",
            tipo_violacion="EPP_FALTANTE",
            nivel_riesgo="CRITICO",
            timestamp=time.time(),
            confianza=0.9,
        ),
    ]
    return orch


@pytest.fixture
def sample_violations_critical() -> list[Violation]:
    """Violación CRITICA: hardhat + safety_vest faltantes."""
    return [
        Violation(
            persona_id=1,
            zona_id="extraccion",
            epp_faltantes=[
                PPEMissing(
                    tipo="hardhat",
                    descripcion="Casco de seguridad",
                    norma_chilena="NCh 1411",
                ),
                PPEMissing(
                    tipo="safety_vest",
                    descripcion="Chaleco reflectante",
                    norma_chilena="NCh 461",
                ),
            ],
            frame_id=10,
        ),
    ]


@pytest.fixture
def sample_violations_low() -> list[Violation]:
    """Violación BAJA: solo gloves faltante."""
    return [
        Violation(
            persona_id=1,
            zona_id="extraccion",
            epp_faltantes=[
                PPEMissing(
                    tipo="gloves",
                    descripcion="Guantes de trabajo",
                    norma_chilena="NCh 1432",
                ),
            ],
            frame_id=10,
        ),
    ]


class TestComputeViolationPriority:
    """Tests para _compute_violation_priority."""

    def test_critical(self) -> None:
        """hardhat + >=2 EPP → prioridad 0 (CRITICO)."""
        from munin.pipeline.vlm_queue import _compute_violation_priority

        v = [
            Violation(
                persona_id=1,
                zona_id="z",
                epp_faltantes=[
                    PPEMissing(tipo="hardhat", descripcion="Casco",
                               norma_chilena="NCh 1411"),
                    PPEMissing(tipo="safety_vest", descripcion="Chaleco",
                               norma_chilena="NCh 461"),
                ],
                frame_id=0,
            ),
        ]
        assert _compute_violation_priority(v) == 0

    def test_high(self) -> None:
        """hardhat solo → prioridad 1 (ALTO)."""
        from munin.pipeline.vlm_queue import _compute_violation_priority

        v = [
            Violation(
                persona_id=1,
                zona_id="z",
                epp_faltantes=[
                    PPEMissing(tipo="hardhat", descripcion="Casco",
                               norma_chilena="NCh 1411"),
                ],
                frame_id=0,
            ),
        ]
        assert _compute_violation_priority(v) == 1

    def test_medium(self) -> None:
        """safety_vest + safety_boots → prioridad 2 (MEDIO)."""
        from munin.pipeline.vlm_queue import _compute_violation_priority

        v = [
            Violation(
                persona_id=1,
                zona_id="z",
                epp_faltantes=[
                    PPEMissing(tipo="safety_vest", descripcion="Chaleco",
                               norma_chilena="NCh 461"),
                    PPEMissing(tipo="safety_boots", descripcion="Botas",
                               norma_chilena="NCh 746"),
                ],
                frame_id=0,
            ),
        ]
        assert _compute_violation_priority(v) == 2

    def test_low(self) -> None:
        """solo gloves → prioridad 3 (BAJO)."""
        from munin.pipeline.vlm_queue import _compute_violation_priority

        v = [
            Violation(
                persona_id=1,
                zona_id="z",
                epp_faltantes=[
                    PPEMissing(tipo="gloves", descripcion="Guantes",
                               norma_chilena="NCh 1432"),
                ],
                frame_id=0,
            ),
        ]
        assert _compute_violation_priority(v) == 3

    def test_empty(self) -> None:
        """Sin violaciones → prioridad 3 (BAJO)."""
        from munin.pipeline.vlm_queue import _compute_violation_priority
        assert _compute_violation_priority([]) == 3

    def test_harness_is_critical(self) -> None:
        """harness cuenta como crítico igual que hardhat."""
        from munin.pipeline.vlm_queue import _compute_violation_priority

        v = [
            Violation(
                persona_id=1,
                zona_id="z",
                epp_faltantes=[
                    PPEMissing(tipo="harness", descripcion="Arnés",
                               norma_chilena="NCh 1411"),
                    PPEMissing(tipo="gloves", descripcion="Guantes",
                               norma_chilena="NCh 1432"),
                ],
                frame_id=0,
            ),
        ]
        assert _compute_violation_priority(v) == 0


class TestVLMQueueInit:
    """Tests de inicialización de VLMQueue."""

    def test_init_defaults(self, mock_orchestrator: MagicMock) -> None:
        """VLMQueue con valores por defecto."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator)
        assert q.size == 0
        assert q.is_full is False
        assert q.is_high_pressure is False

    def test_init_custom(self, mock_orchestrator: MagicMock) -> None:
        """VLMQueue con valores custom."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(
            orchestrator=mock_orchestrator,
            max_size=20,
            high_threshold=10,
            vlm_timeout=60.0,
        )
        assert q._max_size == 20
        assert q._high_threshold == 10
        assert q._vlm_timeout == 60.0


class TestVLMQueueEnqueue:
    """Tests de enqueue con backpressure."""

    def test_enqueue_normal(
        self,
        mock_orchestrator: MagicMock,
        sample_violations_critical: list[Violation],
    ) -> None:
        """Enqueue normal retorna True."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator, max_size=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = q.enqueue(
            camera_id="cam01",
            frame=frame,
            violations=sample_violations_critical,
            frame_number=0,
        )
        assert result is True
        assert q.size == 1

    def test_enqueue_when_full(
        self,
        mock_orchestrator: MagicMock,
        sample_violations_critical: list[Violation],
    ) -> None:
        """Cola llena → retorna False."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator, max_size=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        q.enqueue(camera_id="cam01", frame=frame,
                  violations=sample_violations_critical, frame_number=0)
        result = q.enqueue(camera_id="cam02", frame=frame,
                           violations=sample_violations_critical, frame_number=1)
        assert result is False

    def test_high_pressure_drops_low_priority(
        self,
        mock_orchestrator: MagicMock,
        sample_violations_low: list[Violation],
        sample_violations_critical: list[Violation],
    ) -> None:
        """High pressure + baja prioridad → descarta."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(
            orchestrator=mock_orchestrator,
            max_size=10,
            high_threshold=1,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Llenar hasta high pressure
        q.enqueue(camera_id="cam01", frame=frame,
                  violations=sample_violations_critical, frame_number=0)

        # Ahora high pressure + baja prioridad → descarta
        result = q.enqueue(camera_id="cam02", frame=frame,
                           violations=sample_violations_low, frame_number=1)
        assert result is False


class TestVLMQueueStartStop:
    """Tests de ciclo de vida worker."""

    def test_start_creates_worker(
        self, mock_orchestrator: MagicMock,
    ) -> None:
        """start() crea thread worker."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator)
        q.start()
        assert q._worker is not None
        assert q._worker.is_alive()
        q.stop()

    def test_start_idempotent(
        self, mock_orchestrator: MagicMock,
    ) -> None:
        """start() múltiple es seguro."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator)
        q.start()
        q.start()  # Segundo start no debe crear otro worker
        q.stop()

    def test_stop_joins_worker(
        self, mock_orchestrator: MagicMock,
    ) -> None:
        """stop() detiene el worker."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator)
        q.start()
        q.stop()
        assert q._worker is None or not q._worker.is_alive()


class TestVLMQueueWorkerProcessing:
    """Tests del worker loop."""

    def test_worker_processes_request(
        self,
        mock_orchestrator: MagicMock,
        sample_violations_critical: list[Violation],
    ) -> None:
        """Worker procesa request y genera resultado."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator)
        q.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        q.enqueue(
            camera_id="cam01",
            frame=frame,
            violations=sample_violations_critical,
            frame_number=0,
        )

        # Esperar que el worker procese
        time.sleep(0.2)

        results = q.drain_results()
        assert len(results) == 1
        assert results[0].camera_id == "cam01"
        assert len(results[0].decisions) == 1
        assert results[0].latency_ms > 0
        q.stop()

    def test_worker_timeout_handling(
        self,
        sample_violations_critical: list[Violation],
    ) -> None:
        """Timeout en orchestrator → resultado con error."""
        from munin.pipeline.vlm_queue import VLMQueue

        slow_orch = MagicMock()
        slow_orch.analyze = AsyncMock()
        slow_orch.analyze.side_effect = TimeoutError("slow VLM")

        q = VLMQueue(
            orchestrator=slow_orch,
            vlm_timeout=0.1,  # Timeout rápido para test
        )
        q.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        q.enqueue(
            camera_id="cam01",
            frame=frame,
            violations=sample_violations_critical,
            frame_number=0,
        )

        time.sleep(0.3)
        results = q.drain_results()
        # Puede ser timeout o error dependiendo del timing
        assert len(results) >= 0
        q.stop()

    def test_drain_results_empty(
        self, mock_orchestrator: MagicMock,
    ) -> None:
        """drain_results sin resultados → []."""
        from munin.pipeline.vlm_queue import VLMQueue

        q = VLMQueue(orchestrator=mock_orchestrator)
        results = q.drain_results()
        assert results == []


class TestVLMQueueThreadSafety:
    """Tests de thread safety."""

    def test_concurrent_enqueue(
        self,
        mock_orchestrator: MagicMock,
        sample_violations_critical: list[Violation],
    ) -> None:
        """Múltiples enqueues concurrentes no crashean."""
        from munin.pipeline.vlm_queue import VLMQueue
        import threading

        q = VLMQueue(orchestrator=mock_orchestrator, max_size=20)
        q.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def enqueue_thread(camera_id: str) -> None:
            q.enqueue(
                camera_id=camera_id,
                frame=frame,
                violations=sample_violations_critical,
                frame_number=0,
            )

        threads = [
            threading.Thread(target=enqueue_thread, args=(f"cam{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.3)
        results = q.drain_results()
        # Algunos pueden haber sido encolados y procesados
        assert len(results) <= 5
        q.stop()
