"""VLMQueue — Cola de prioridad con backpressure para VLM.

Reemplaza el flag vlm_busy del Pipeline original con una cola
thread-safe basada en prioridad, con worker thread dedicado para
procesar requests VLM concurrentemente.

ADR-030: VLMQueue con PriorityQueue + backpressure.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from munin.gate.schemas import AgentDecision, Violation

logger = logging.getLogger(__name__)


@dataclass(order=True)
class VLMRequest:
    """Solicitud de análisis VLM con prioridad.

    El order=True permite usar con PriorityQueue.
    priority y timestamp son comparables (order=True).
    camera_id, frame, violations NO (compare=False).

    Attributes:
        priority: 0=CRITICO, 1=ALTO, 2=MEDIO, 3=BAJO.
        timestamp: Momento de creación (epoch float).
        camera_id: ID de la cámara.
        frame: Frame del video.
        violations: Violaciones a analizar.
        frame_number: Número de frame.
    """
    priority: int = field(compare=True)
    timestamp: float = field(compare=True)
    camera_id: str = field(compare=False)
    frame: np.ndarray = field(compare=False)
    violations: list[Violation] = field(compare=False)
    frame_number: int = field(compare=False)


@dataclass
class VLMResult:
    """Resultado del análisis VLM.

    Attributes:
        camera_id: ID de la cámara.
        decisions: Decisiones generadas.
        latency_ms: Latencia en milisegundos.
        error: Mensaje de error si falló.
    """
    camera_id: str
    decisions: list[AgentDecision]
    latency_ms: float = 0.0
    error: str | None = None


def _compute_violation_priority(violations: list[Violation]) -> int:
    """Computa prioridad 0-3 basada en severidad de violaciones.

    Rules:
        - 0 (CRITICO): hardhat o harness faltante + >=2 EPP totales.
        - 1 (ALTO): hardhat o harness faltante.
        - 2 (MEDIO): safety_vest o safety_boots faltante + >=2 EPP totales.
        - 3 (BAJO): otras combinaciones o sin violaciones.

    Args:
        violations: Lista de violaciones detectadas.

    Returns:
        Prioridad: 0=CRITICO, 1=ALTO, 2=MEDIO, 3=BAJO.
    """
    if not violations:
        return 3

    epp_faltantes: set[str] = set()
    for v in violations:
        for epp in v.epp_faltantes:
            epp_faltantes.add(epp.tipo)

    critical = {"hardhat", "harness"}
    high = {"safety_vest", "safety_boots"}

    has_critical = bool(epp_faltantes & critical)
    has_high = bool(epp_faltantes & high)
    count = len(epp_faltantes)

    if has_critical and count >= 2:
        return 0
    elif has_critical:
        return 1
    elif has_high and count >= 2:
        return 2
    else:
        return 3


class VLMQueue:
    """Cola de prioridad con backpressure para VLM.

    Reemplaza el flag vlm_busy del Pipeline original.
    Un worker thread consume requests de la cola y llama
    al orchestrator VLM de forma asíncrona.

    Attributes:
        _input_queue: PriorityQueue con VLMRequest.
        _result_queue: Queue simple con VLMResult.
        _orchestrator: MuninOrchestrator para análisis.
        _max_size: Tamaño máximo de la cola.
        _high_threshold: Threshold para backpressure alto.
        _running: Flag de control del worker.
    """

    def __init__(
        self,
        orchestrator: Any,
        max_size: int = 10,
        high_threshold: int = 5,
        vlm_timeout: float = 300.0,
    ) -> None:
        """Inicializa la cola VLM.

        Args:
            orchestrator: MuninOrchestrator para análisis VLM.
            max_size: Tamaño máximo de la cola (backpressure total).
            high_threshold: Threshold para backpressure alta
                (descarta requests de baja prioridad).
            vlm_timeout: Timeout por request VLM en segundos.
        """
        self._input_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max_size
        )
        self._result_queue: queue.Queue = queue.Queue()
        self._orchestrator = orchestrator
        self._max_size = max_size
        self._high_threshold = high_threshold
        self._vlm_timeout = vlm_timeout
        self._running = False
        self._worker: threading.Thread | None = None
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def size(self) -> int:
        """Profundidad actual de la cola de entrada."""
        return self._input_queue.qsize()

    @property
    def is_full(self) -> bool:
        """True si la cola ha alcanzado max_size."""
        return self.size >= self._max_size

    @property
    def is_high_pressure(self) -> bool:
        """True si la cola ha alcanzado high_threshold."""
        return self.size >= self._high_threshold

    def start(self) -> None:
        """Inicia el worker thread.

        El worker es daemon: se detiene automáticamente al salir.
        """
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="vlm-worker",
            daemon=True,
        )
        self._worker.start()
        self._logger.info(
            "VLM worker started (max=%d, threshold=%d)",
            self._max_size, self._high_threshold,
        )

    def stop(self) -> None:
        """Detiene el worker thread."""
        self._running = False
        try:
            self._input_queue.put_nowait(None)  # Sentinel para desbloquear
        except queue.Full:
            pass
        if self._worker:
            self._worker.join(timeout=10)
        self._logger.info("VLM worker stopped")

    def enqueue(
        self,
        camera_id: str,
        frame: np.ndarray,
        violations: list[Violation],
        frame_number: int,
    ) -> bool:
        """Encola un request VLM con backpressure.

        Si la cola está llena (>= max_size), descarta el request.
        Si la cola está en high pressure y la prioridad es baja (>= 3),
        también descarta.

        Args:
            camera_id: ID de la cámara.
            frame: Frame del video.
            violations: Violaciones detectadas.
            frame_number: Número de frame.

        Returns:
            True si se encoló correctamente.
            False si backpressure forzó descarte (debería usarse
            decisión por defecto).
        """
        if self.is_full:
            self._logger.warning(
                "VLM queue full (%d/%d), dropping %s frame %d",
                self.size, self._max_size, camera_id, frame_number,
            )
            return False

        priority = _compute_violation_priority(violations)

        if self.is_high_pressure and priority >= 3:
            self._logger.info(
                "VLM high pressure (%d >= %d), skipping low-priority "
                "request for %s frame %d",
                self.size, self._high_threshold, camera_id, frame_number,
            )
            return False

        request = VLMRequest(
            priority=priority,
            timestamp=time.time(),
            camera_id=camera_id,
            frame=frame,
            violations=violations,
            frame_number=frame_number,
        )

        try:
            self._input_queue.put_nowait(request)
            return True
        except queue.Full:
            self._logger.warning(
                "VLM queue full (race), dropping %s frame %d",
                camera_id, frame_number,
            )
            return False

    def drain_results(self) -> list[VLMResult]:
        """Drena todos los resultados disponibles de la cola de salida.

        Returns:
            Lista de VLMResult. Vacía si no hay resultados.
        """
        results: list[VLMResult] = []
        while not self._result_queue.empty():
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _worker_loop(self) -> None:
        """Loop principal del worker thread.

        Consume requests de la cola de prioridad y ejecuta
        el orchestrator VLM de forma asíncrona (event loop propio).
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            try:
                request = self._input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if request is None:
                break

            start_time = time.monotonic()
            try:
                decisions = loop.run_until_complete(
                    asyncio.wait_for(
                        self._orchestrator.analyze(
                            request.frame, request.violations,
                        ),
                        timeout=self._vlm_timeout,
                    )
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                self._result_queue.put(VLMResult(
                    camera_id=request.camera_id,
                    decisions=decisions,
                    latency_ms=elapsed_ms,
                ))
                self._logger.info(
                    "VLM processed: cam=%s, frame=%d, %.0fms, "
                    "%d decisions",
                    request.camera_id, request.frame_number,
                    elapsed_ms, len(decisions),
                )
            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                self._logger.error(
                    "VLM timeout for %s frame %d: %.0fms",
                    request.camera_id, request.frame_number, elapsed_ms,
                )
                self._result_queue.put(VLMResult(
                    camera_id=request.camera_id,
                    decisions=[],
                    error=f"Timeout after {elapsed_ms:.0f}ms",
                ))
            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                self._logger.error(
                    "VLM error for %s frame %d: %s",
                    request.camera_id, request.frame_number, e,
                )
                self._result_queue.put(VLMResult(
                    camera_id=request.camera_id,
                    decisions=[],
                    error=str(e),
                ))
            finally:
                self._input_queue.task_done()

        loop.close()
        self._logger.info("VLM worker loop ended")


__all__ = ["VLMQueue", "VLMRequest", "VLMResult", "_compute_violation_priority"]
