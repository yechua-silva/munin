from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

import numpy as np

from munin.config import AppSettings
from munin.gate.schemas import AgentDecision, DetectionResult, TrackedPerson, Violation
from munin.pipeline.factory import PipelineFactory

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
    ) -> np.ndarray:
        """Anota un frame con bounding boxes, labels y traces usando supervision.

        ADR-019: Supervision se usa SOLO en dashboard, no en pipeline.

        Args:
            frame: Frame original (HWC BGR uint8).
            detections: Detecciones de YOLO en el frame.
            persons: Personas trackeadas con IDs (opcional).
            violations: Violaciones detectadas (opcional).

        Returns:
            Frame anotado con bounding boxes, labels y traces.
        """
        if not SUPERVISION_AVAILABLE:
            return frame

        # Convertir DetectionResult → sv.Detections
        if not detections:
            return frame

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_names = np.array([d.class_name for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=np.arange(len(detections)),
            data={"class_name": class_names},
        )

        # Añadir tracker_id si hay persons
        if persons:
            tracker_ids = []
            for det in detections:
                # Match por bbox closest person
                best_id = -1
                best_iou = 0.0
                for person in persons:
                    iou = self._compute_iou(det.bbox, person.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = person.persona_id
                tracker_ids.append(best_id if best_iou > 0.3 else -1)
            sv_detections.tracker_id = np.array(tracker_ids)

        # Annotators
        annotated = frame.copy()

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

        return annotated

    @staticmethod
    def _compute_iou(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        """Calcula IoU entre dos bboxes.

        Args:
            bbox_a: Primer bounding box (x1, y1, x2, y2).
            bbox_b: Segundo bounding box (x1, y1, x2, y2).

        Returns:
            IoU (Intersection over Union) como float entre 0.0 y 1.0.
        """
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

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
