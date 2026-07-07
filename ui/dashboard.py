from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from munin.config import AppSettings
from munin.gate.schemas import AgentDecision
from munin.pipeline.factory import PipelineFactory

logger = logging.getLogger(__name__)


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
