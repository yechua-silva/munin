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
        """Renderiza el dashboard completo en Streamlit."""
        st.set_page_config(
            page_title="Munin — Industrial Vision Agent",
            layout="wide",
        )

        st.title("🦅 Munin — Monitoreo EPP DS 132")

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

            self._render_results(decisions, uploaded_file.name, zona)

        elif analizar_btn and uploaded_file is None:
            st.warning("Debes seleccionar un archivo MP4 para analizar.")

    # ------------------------------------------------------------------
    # Ejecución del pipeline
    # ------------------------------------------------------------------

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
