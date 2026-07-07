from __future__ import annotations

import logging
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from munin.config import AppSettings
from munin.gate.schemas import AgentDecision
from munin.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
_ALLOWED_EXTENSION = ".mp4"


class MuninAPI:
    """FastAPI wrapper que expone los endpoints REST de Munin.

    Attributes:
        app: Instancia de FastAPI lista para uvicorn.
    """

    def __init__(self, pipeline: Pipeline, settings: AppSettings) -> None:
        """Inicializa la API con el pipeline y configuración.

        Args:
            pipeline: Pipeline de visión industrial inyectado.
            settings: Configuración global de la aplicación.
        """
        self._pipeline = pipeline
        self._settings = settings
        self._results: dict[str, dict] = {}
        self._app = FastAPI(title="Munin API", version="1.0.0")
        self._register_routes()

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def app(self) -> FastAPI:
        """FastAPI app lista para ser montada por uvicorn."""
        return self._app

    # ------------------------------------------------------------------
    # Rutas privadas (registro)
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        """Vincula los handlers a sus rutas."""
        self._app.add_api_route(
            "/api/v1/health", self._health, methods=["GET"]
        )
        self._app.add_api_route(
            "/api/v1/analyze", self._analyze, methods=["POST"]
        )
        self._app.add_api_route(
            "/api/v1/analyze/{job_id}", self._get_result, methods=["GET"]
        )
        self._app.add_api_route(
            "/api/v1/analyze/single-pass/{job_id}",
            self._get_single_pass_result,
            methods=["GET"],
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _health(self) -> dict:
        """Health check endpoint.

        Returns:
            Dict con estado y versión de la API.
        """
        logger.debug("Health check solicitado")
        return {"status": "ok", "version": "1.0.0"}

    async def _analyze(
        self,
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks | None = None,
    ) -> dict:
        """Recibe un video MP4, inicia el análisis en background y retorna un job_id.

        Args:
            file: Archivo MP4 subido por el usuario.
            background_tasks: Gestor de tareas en background de FastAPI.

        Returns:
            Dict con job_id y estado inicial.

        Raises:
            HTTPException 400: Si el archivo no es .mp4 o supera 100 MB.
        """
        # Validar extensión
        if not file.filename or not file.filename.lower().endswith(_ALLOWED_EXTENSION):
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado. Solo se aceptan archivos {_ALLOWED_EXTENSION}.",
            )

        # Leer contenido
        content = await file.read()

        # Validar tamaño
        if len(content) > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="El archivo excede el tamaño máximo de 100 MB.",
            )

        # Guardar a archivo temporal
        suffix = Path(str(file.filename)).suffix if file.filename else ".mp4"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        # Generar job ID
        job_id = str(uuid.uuid4())
        self._results[job_id] = {"status": "processing"}

        logger.info("Job %s creado para archivo %s", job_id, file.filename)

        # Procesar en background
        if background_tasks is not None:
            background_tasks.add_task(
                self._process_video, job_id, temp_path, "extraccion"
            )

        return {"job_id": job_id, "status": "processing"}

    async def _get_result(self, job_id: str) -> dict:
        """Retorna el resultado de un job de análisis.

        Args:
            job_id: ID del job a consultar.

        Returns:
            Dict con estado y decisiones (si completó).

        Raises:
            HTTPException 404: Si el job_id no existe.
        """
        result = self._results.get(job_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} no encontrado.",
            )
        if result["status"] == "processing":
            return {"status": "processing"}
        return result

    async def _get_single_pass_result(self, job_id: str) -> dict:
        """Retorna el resultado de un job procesado en modo single-pass.

        Actualmente es un alias de _get_result; el flag de modo single-pass
        se definirá en versiones futuras del pipeline.

        Args:
            job_id: ID del job a consultar.

        Returns:
            Dict con estado y decisiones (si completó).

        Raises:
            HTTPException 404: Si el job_id no existe.
        """
        return await self._get_result(job_id)

    # ------------------------------------------------------------------
    # Procesamiento background
    # ------------------------------------------------------------------

    async def _process_video(
        self, job_id: str, video_path: str, zone_id: str
    ) -> None:
        """Ejecuta el pipeline en background y almacena el resultado.

        Args:
            job_id: ID del job asociado.
            video_path: Ruta al archivo de video temporal.
            zone_id: Zona de análisis (default: extraccion).
        """
        try:
            logger.info(
                "Procesando job %s (video: %s, zona: %s)",
                job_id,
                video_path,
                zone_id,
            )
            decisions: list[AgentDecision] = await self._pipeline.process(
                video_path, zone_id=zone_id
            )
            self._results[job_id] = {
                "status": "completed",
                "decisions": [d.model_dump() for d in decisions],
            }
            logger.info(
                "Job %s completado — %d decisiones generadas",
                job_id,
                len(decisions),
            )
        except Exception as exc:
            logger.exception("Job %s falló: %s", job_id, exc)
            self._results[job_id] = {"status": "failed", "error": str(exc)}
        finally:
            # Limpiar archivo temporal
            try:
                Path(video_path).unlink(missing_ok=True)
                logger.debug("Archivo temporal %s eliminado", video_path)
            except OSError:
                logger.warning(
                    "No se pudo eliminar el archivo temporal %s", video_path
                )
