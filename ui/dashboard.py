from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

import numpy as np

from munin.config import AppSettings, Zone
from munin.gate.schemas import AgentDecision, DetectionResult, TrackedPerson, Violation
from munin.pipeline.factory import PipelineFactory
from munin.pipeline.frame_collector import CameraSource
from munin.pipeline.supervision_standardizer import SupervisionStandardizer

logger = logging.getLogger(__name__)

try:
    import supervision as sv

    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger.warning("supervision not installed, annotation features disabled")


class StreamlitDashboard:
    """Dashboard Streamlit para Munin — monitoreo EPP DS 132.

    Permite subir un video MP4, seleccionar zona minera, ejecutar el
    pipeline de visión industrial y visualizar violaciones, métricas
    y alertas en tiempo real.

    Attributes:
        pipeline_factory: Factory class para construir el Pipeline.
        settings: Configuración global de la aplicación.
    """

    def __init__(
        self,
        pipeline_factory: type[PipelineFactory],
        settings: AppSettings,
    ) -> None:
        """Inicializa el dashboard.

        Args:
            pipeline_factory: Clase factory para crear el Pipeline.
            settings: Configuración global de la aplicación.
        """
        self._pipeline_factory = pipeline_factory
        self._settings = settings

    # ------------------------------------------------------------------
    # Render principal
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Renderiza el dashboard completo en Streamlit.

        Si multi_camera_enabled=True, muestra controles multi-cámara
        con tabs por cámara. Caso contrario, mantiene el
        comportamiento original de archivo único.
        """
        st.set_page_config(
            page_title="Munin — Industrial Vision Agent",
            layout="wide",
        )

        st.title("🦅 Munin — Monitoreo EPP DS 132")

        if self._settings.multi_camera_enabled:
            self._render_multi_camera()
        else:
            self._render_single_camera()

    def _render_single_camera(self) -> None:
        """Renderiza el dashboard modo single-cámara (v3 legacy)."""
        # Sidebar — controles
        with st.sidebar:
            st.header("Controles")
            uploaded_file = st.file_uploader(
                "Seleccionar video MP4",
                type=["mp4"],
            )
            zona = st.selectbox(
                "Zona minera",
                options=["extraccion", "procesamiento", "mantencion"],
                index=0,
            )
            analizar_btn = st.button(
                "Analizar", type="primary", use_container_width=True
            )

            st.divider()
            st.caption(
                f"**VLM:** {self._settings.vlm_backend.value} | "
                f"**Modelo:** {self._settings.fireworks_model}"
            )

        # Área principal — resultados
        if analizar_btn and uploaded_file is not None:
            with st.spinner("Procesando video..."):
                decisions = self._run_pipeline(uploaded_file, zona)

            if decisions is None:
                st.error("Error al procesar el video. Revisa los logs.")
                return

            if isinstance(decisions, dict):
                # Multi-cámara retorna dict, mostrar tabs
                self._render_results_dict(decisions, uploaded_file.name)
            else:
                self._render_results(decisions, uploaded_file.name, zona)

        elif analizar_btn and uploaded_file is None:
            st.warning("Debes seleccionar un archivo MP4 para analizar.")

    def _render_multi_camera(self) -> None:
        """Renderiza el dashboard modo multi-cámara con tabs."""
        st.info(
            "Modo multi-cámara activado. Las fuentes se configuran "
            "via MUNIN_MULTI_CAMERA_SOURCES en .env o la variable "
            "multi_camera_sources en AppSettings.",
        )

        with st.sidebar:
            st.header("Multi-Cámara")
            st.caption(
                f"Cámaras configuradas: "
                f"{self._settings.multi_camera_sources or 'N/A'}"
            )
            st.caption(
                f"Cola VLM máxima: {self._settings.vlm_queue_max_size}"
            )
            st.caption(
                f"Timeout VLM: {self._settings.vlm_busy_timeout}s"
            )
            frame_limit = st.number_input(
                "Límite de frames",
                min_value=1,
                max_value=10000,
                value=100,
            )
            iniciar_btn = st.button(
                "Iniciar monitoreo",
                type="primary",
                use_container_width=True,
            )

        if iniciar_btn:
            with st.spinner("Iniciando pipeline multi-cámara..."):
                decisions_dict = self._run_multi_camera(frame_limit)

            if decisions_dict is None:
                st.error("Error al iniciar pipeline multi-cámara.")
                return

            self._render_results_dict(decisions_dict, "multi-cam")

    # ------------------------------------------------------------------
    # Ejecución del pipeline
    # ------------------------------------------------------------------

    def _run_multi_camera(
        self,
        frame_limit: int = 100,
    ) -> dict[str, list[AgentDecision]] | None:
        """Ejecuta pipeline multi-cámara.

        Parsea las fuentes desde multi_camera_sources (JSON) y
        construye el pipeline multi-cámara.

        Args:
            frame_limit: Máximo de frames a procesar.

        Returns:
            Dict[camera_id, list[AgentDecision]] o None si falla.
        """
        try:
            import json

            sources_raw = self._settings.multi_camera_sources
            if not sources_raw:
                st.error(
                    "No hay fuentes configuradas. "
                    "Define MUNIN_MULTI_CAMERA_SOURCES en .env"
                )
                return None

            sources_list: list[dict] = json.loads(sources_raw)
            sources = [
                CameraSource(**s) for s in sources_list
            ]

            pipeline = self._pipeline_factory.create_multi_camera(
                self._settings, sources,
            )
            decisions: dict[str, list[AgentDecision]] = asyncio.run(
                pipeline.process_all(frame_limit=frame_limit)
            )
            logger.info(
                "Multi-camera pipeline completado — %d cámaras",
                len(decisions),
            )
            return decisions
        except Exception as exc:
            logger.exception("Error ejecutando multi-cámara: %s", exc)
            return None

    def _run_pipeline(
        self,
        uploaded_file,
        zone_id: str,
    ) -> list[AgentDecision] | None:
        """Guarda el upload a temporal, ejecuta el pipeline y retorna decisiones.

        Args:
            uploaded_file: Archivo subido desde Streamlit.
            zone_id: ID de la zona minera.

        Returns:
            Lista de decisiones del agente, o None si falla.
        """
        # Guardar archivo temporal
        suffix = Path(uploaded_file.name).suffix if uploaded_file.name else ".mp4"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:
            pipeline = self._pipeline_factory.create(self._settings)
            # asyncio.run() porque Streamlit no maneja async nativamente
            decisions: list[AgentDecision] = asyncio.run(
                pipeline.process(temp_path, zone_id=zone_id)
            )
            logger.info(
                "Pipeline completado — %d decisiones, archivo: %s",
                len(decisions),
                uploaded_file.name,
            )
            return decisions
        except Exception as exc:
            logger.exception("Error ejecutando pipeline: %s", exc)
            return None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Anotación de frames con supervision
    # ------------------------------------------------------------------

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        persons: list[TrackedPerson] | None = None,
        violations: list[Violation] | None = None,
        zone: Zone | None = None,
    ) -> np.ndarray:
        """Anota un frame con bounding boxes, labels, traces y zona usando supervision.

        ADR-019: Supervision se usa SOLO en dashboard, no en pipeline.
        ADR-028: SupervisionStandardizer bridge para conversión Detections→sv.Detections.

        Args:
            frame: Frame original (HWC BGR uint8).
            detections: Detecciones de YOLO en el frame.
            persons: Personas trackeadas con IDs (opcional).
            violations: Violaciones detectadas (opcional).
            zone: Zona configurada con polígono opcional (v4).

        Returns:
            Frame anotado con bounding boxes, labels, traces y zona.
        """
        if not SUPERVISION_AVAILABLE:
            return frame

        annotated = frame.copy()

        # Convertir DetectionResult → sv.Detections via SupervisionStandardizer
        sv_detections = SupervisionStandardizer.from_detection_results(
            detections, persons
        )

        if len(sv_detections) > 0:
            # BoxAnnotator
            box_annotator = sv.BoxAnnotator()
            annotated = box_annotator.annotate(annotated, sv_detections)

            # LabelAnnotator
            labels = [
                f"{d.class_name} {d.confidence:.2f}"
                for d in detections
            ]
            label_annotator = sv.LabelAnnotator()
            annotated = label_annotator.annotate(
                annotated, sv_detections, labels=labels
            )

            # TraceAnnotator si hay tracker_ids
            if persons and hasattr(sv, 'TraceAnnotator'):
                trace_annotator = sv.TraceAnnotator()
                annotated = trace_annotator.annotate(annotated, sv_detections)

        # PolygonZoneAnnotator si zone tiene polygon (v4)
        if zone is not None and zone.polygon is not None:
            h, w = frame.shape[:2]
            for sub_poly in zone.polygon:
                poly_px = np.array(
                    [[p[0] * w, p[1] * h] for p in sub_poly],
                    dtype=np.int64,
                )
                polygon_zone = sv.PolygonZone(polygon=poly_px)
                zone_annotator = sv.PolygonZoneAnnotator(
                    zone=polygon_zone,
                    color=sv.Color.RED,
                    opacity=0.15,
                )
                annotated = zone_annotator.annotate(annotated)

        return annotated

    # ------------------------------------------------------------------
    # Renderizado de resultados
    # ------------------------------------------------------------------

    def _render_results(
        self,
        decisions: list[AgentDecision],
        filename: str,
        zone_id: str,
    ) -> None:
        """Renderiza métricas, tabla de violaciones y alertas.

        Args:
            decisions: Decisiones generadas por el pipeline.
            filename: Nombre del archivo procesado.
            zone_id: Zona analizada.
        """
        st.success(f"Análisis completado — {filename} (zona: {zone_id})")

        # --- Métricas ---
        violaciones = [
            d for d in decisions if d.tipo_violacion != "SIN_VIOLACION"
        ]
        frames_procesados = len(decisions)
        violaciones_count = len(violaciones)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frames procesados", frames_procesados)
        with col2:
            st.metric("Violaciones detectadas", violaciones_count)
        with col3:
            criticas = sum(
                1 for d in violaciones if d.nivel_riesgo == "CRITICO"
            )
            st.metric("Críticas", criticas)

        # --- Tabla de violaciones ---
        if violaciones:
            st.subheader("Detalle de violaciones")
            rows = []
            for d in violaciones:
                epp_str = ", ".join(
                    f"{e.descripcion} ({e.tipo})" for e in d.epp_faltante
                )
                rows.append({
                    "Zona": d.zona,
                    "Tipo": d.tipo_violacion,
                    "EPP Faltante": epp_str,
                    "Nivel Riesgo": d.nivel_riesgo,
                    "Artículo DS 132": d.articulo_ds132 or "—",
                    "Confianza": f"{d.confianza:.2f}",
                })
            st.dataframe(rows, use_container_width=True)

        # --- Alertas por nivel de riesgo ---
        st.subheader("Alertas")
        if not violaciones:
            st.info("No se detectaron violaciones de EPP.")
        else:
            for d in violaciones:
                epp_str = ", ".join(
                    f"{e.descripcion}" for e in d.epp_faltante
                )
                mensaje = (
                    f"**{d.tipo_violacion}** — Zona: {d.zona} | "
                    f"EPP faltante: {epp_str} | "
                    f"Artículo: {d.articulo_ds132 or 'N/A'} | "
                    f"Confianza: {d.confianza:.2f}"
                )
                if d.nivel_riesgo == "CRITICO":
                    st.error(f"🔴 CRÍTICO — {mensaje}")
                elif d.nivel_riesgo == "ALTO":
                    st.warning(f"🟠 ALTO — {mensaje}")
                elif d.nivel_riesgo == "MEDIO":
                    st.info(f"🟡 MEDIO — {mensaje}")
                else:
                    st.success(f"🟢 BAJO — {mensaje}")

    def _render_results_dict(
        self,
        decisions_dict: dict[str, list[AgentDecision]],
        label: str,
    ) -> None:
        """Renderiza resultados multi-cámara con tabs.

        Args:
            decisions_dict: Dict[camera_id, list[AgentDecision]].
            label: Etiqueta descriptiva (nombre archivo o 'multi-cam').
        """
        st.success(f"Monitoreo completado — {label}")

        if not decisions_dict:
            st.warning("No se recibieron decisiones de ninguna cámara.")
            return

        camera_ids = list(decisions_dict.keys())
        tabs = st.tabs([f"📷 {cid}" for cid in camera_ids])

        for tab, camera_id in zip(tabs, camera_ids):
            with tab:
                decisions = decisions_dict[camera_id]
                violaciones = [
                    d for d in decisions
                    if d.tipo_violacion != "SIN_VIOLACION"
                ]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Frames procesados", len(decisions))
                with col2:
                    st.metric("Violaciones", len(violaciones))
                with col3:
                    criticas = sum(
                        1 for d in violaciones
                        if d.nivel_riesgo == "CRITICO"
                    )
                    st.metric("Críticas", criticas)

                if violaciones:
                    st.subheader("Detalle de violaciones")
                    rows = []
                    for d in violaciones:
                        epp_str = ", ".join(
                            f"{e.descripcion} ({e.tipo})"
                            for e in d.epp_faltante
                        )
                        rows.append({
                            "Zona": d.zona,
                            "Tipo": d.tipo_violacion,
                            "EPP Faltante": epp_str,
                            "Nivel Riesgo": d.nivel_riesgo,
                            "Artículo DS 132": d.articulo_ds132 or "—",
                            "Confianza": f"{d.confianza:.2f}",
                        })
                    st.dataframe(rows, use_container_width=True)

                    st.subheader("Alertas")
                    for d in violaciones:
                        epp_str = ", ".join(
                            f"{e.descripcion}" for e in d.epp_faltante
                        )
                        mensaje = (
                            f"**{d.tipo_violacion}** — "
                            f"Zona: {d.zona} | "
                            f"EPP faltante: {epp_str} | "
                            f"Artículo: {d.articulo_ds132 or 'N/A'} | "
                            f"Confianza: {d.confianza:.2f}"
                        )
                        if d.nivel_riesgo == "CRITICO":
                            st.error(f"🔴 CRÍTICO — {mensaje}")
                        elif d.nivel_riesgo == "ALTO":
                            st.warning(f"🟠 ALTO — {mensaje}")
                        elif d.nivel_riesgo == "MEDIO":
                            st.info(f"🟡 MEDIO — {mensaje}")
                        else:
                            st.success(f"🟢 BAJO — {mensaje}")
                else:
                    st.info(
                        f"No se detectaron violaciones en {camera_id}."
                    )

    def _render_results(
        self,
        decisions: list[AgentDecision],
        filename: str,
        zone_id: str,
    ) -> None:
        """Renderiza métricas, tabla de violaciones y alertas.

        Args:
            decisions: Decisiones generadas por el pipeline.
            filename: Nombre del archivo procesado.
            zone_id: Zona analizada.
        """
        st.success(f"Análisis completado — {filename} (zona: {zone_id})")

        # --- Métricas ---
        violaciones = [
            d for d in decisions if d.tipo_violacion != "SIN_VIOLACION"
        ]
        frames_procesados = len(decisions)
        violaciones_count = len(violaciones)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frames procesados", frames_procesados)
        with col2:
            st.metric("Violaciones detectadas", violaciones_count)
        with col3:
            criticas = sum(
                1 for d in violaciones if d.nivel_riesgo == "CRITICO"
            )
            st.metric("Críticas", criticas)

        # --- Tabla de violaciones ---
        if violaciones:
            st.subheader("Detalle de violaciones")
            rows = []
            for d in violaciones:
                epp_str = ", ".join(
                    f"{e.descripcion} ({e.tipo})" for e in d.epp_faltante
                )
                rows.append({
                    "Zona": d.zona,
                    "Tipo": d.tipo_violacion,
                    "EPP Faltante": epp_str,
                    "Nivel Riesgo": d.nivel_riesgo,
                    "Artículo DS 132": d.articulo_ds132 or "—",
                    "Confianza": f"{d.confianza:.2f}",
                })
            st.dataframe(rows, use_container_width=True)

        # --- Alertas por nivel de riesgo ---
        st.subheader("Alertas")
        if not violaciones:
            st.info("No se detectaron violaciones de EPP.")
        else:
            for d in violaciones:
                epp_str = ", ".join(
                    f"{e.descripcion}" for e in d.epp_faltante
                )
                mensaje = (
                    f"**{d.tipo_violacion}** — Zona: {d.zona} | "
                    f"EPP faltante: {epp_str} | "
                    f"Artículo: {d.articulo_ds132 or 'N/A'} | "
                    f"Confianza: {d.confianza:.2f}"
                )
                if d.nivel_riesgo == "CRITICO":
                    st.error(f"🔴 CRÍTICO — {mensaje}")
                elif d.nivel_riesgo == "ALTO":
                    st.warning(f"🟠 ALTO — {mensaje}")
                elif d.nivel_riesgo == "MEDIO":
                    st.info(f"🟡 MEDIO — {mensaje}")
                else:
                    st.success(f"🟢 BAJO — {mensaje}")

        # --- Frame anotado (si hay supervision) ---
        if SUPERVISION_AVAILABLE:
            st.subheader("Visualización con supervision")
            st.caption(
                "Los frames anotados se muestran cuando supervision está disponible. "
                "BoxAnnotator + LabelAnnotator + TraceAnnotator (ADR-019)."
            )
            st.info(
                "Para ver frames anotados en tiempo real, ejecuta el pipeline "
                "con el callback on_detection activado y frame anotado."
            )

        # --- Footer con info del modelo ---
        st.divider()
        st.caption(
            f"Munin — Backend VLM: {self._settings.vlm_backend.value} | "
            f"Modelo: {self._settings.fireworks_model} | "
            f"Confianza YOLO mínima: {self._settings.yolo_confidence_threshold}"
        )


def main() -> None:
    """Entry point para ``streamlit run dashboard.py``."""
    settings = AppSettings()
    dashboard = StreamlitDashboard(PipelineFactory, settings)
    dashboard.render()


if __name__ == "__main__":
    main()
