from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from munin.exceptions import KnowledgeBaseError

logger = logging.getLogger(__name__)


@dataclass
class DS132Article:
    """Artículo del DS 132 (Reglamento de Seguridad Minera de Chile).

    Attributes:
        id: Identificador del artículo (ej: "Art. 38").
        titulo: Título descriptivo del artículo.
        texto: Texto legal completo del artículo.
        zonas_aplicables: Zonas de la faena donde aplica este artículo.
        epp_requeridos: Lista de EPP requeridos por este artículo.
        nivel_incumplimiento: Nivel de gravedad del incumplimiento.
        referencias: Normas chilenas asociadas (ej: NCh 1411).
    """

    id: str = field(metadata={"description": "ID del artículo (ej: Art. 38)"})
    titulo: str = field(metadata={"description": "Título del artículo"})
    texto: str = field(metadata={"description": "Texto legal completo"})
    zonas_aplicables: list[str] = field(
        default_factory=list,
        metadata={"description": "Zonas donde aplica"},
    )
    epp_requeridos: list[str] = field(
        default_factory=list,
        metadata={"description": "EPP requeridos por este artículo"},
    )
    nivel_incumplimiento: str = field(
        default="ALTO",
        metadata={"description": "Nivel de gravedad del incumplimiento"},
    )
    referencias: list[str] = field(
        default_factory=list,
        metadata={"description": "Normas chilenas asociadas"},
    )


class DS132KnowledgeBase:
    """Knowledge base de artículos del DS 132.

    Carga artículos desde un archivo JSON y provee métodos de
    consulta por zona y por ID de artículo.
    """

    def __init__(self, kb_path: str) -> None:
        """Carga artículos desde archivo JSON.

        Args:
            kb_path: Ruta al archivo ds132_kb.json.

        Raises:
            KnowledgeBaseError: Si el archivo no existe o el JSON es inválido.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._articles: dict[str, DS132Article] = {}

        kb_path_obj = Path(kb_path)
        if not kb_path_obj.exists():
            raise KnowledgeBaseError(f"DS132 KB file not found: {kb_path}")

        try:
            with open(kb_path_obj, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise KnowledgeBaseError(
                f"Failed to load DS132 KB from {kb_path}: {e}"
            ) from e

        articulos_raw = data.get("articulos", [])
        for art in articulos_raw:
            article = DS132Article(
                id=art["id"],
                titulo=art["titulo"],
                texto=art["texto"],
                zonas_aplicables=art.get("zonas_aplicables", []),
                epp_requeridos=art.get("epp_requeridos", []),
                nivel_incumplimiento=art.get("nivel_incumplimiento", "ALTO"),
                referencias=art.get("referencias", []),
            )
            self._articles[article.id] = article

        self._logger.info(
            "Loaded %d DS132 articles from %s", len(self._articles), kb_path
        )

    def get_by_zone(self, zona_id: str) -> list[DS132Article]:
        """Obtiene artículos aplicables a una zona.

        Args:
            zona_id: ID de la zona (extraccion, procesamiento, mantencion).

        Returns:
            Lista de artículos aplicables a la zona.
        """
        return [
            a for a in self._articles.values() if zona_id in a.zonas_aplicables
        ]

    def get_article(self, article_id: str) -> DS132Article:
        """Obtiene un artículo por su ID.

        Args:
            article_id: ID del artículo (ej: "Art. 38").

        Returns:
            Artículo solicitado.

        Raises:
            KnowledgeBaseError: Si el artículo no existe.
        """
        if article_id not in self._articles:
            raise KnowledgeBaseError(f"Article not found: {article_id}")
        return self._articles[article_id]

    @property
    def articles(self) -> dict[str, DS132Article]:
        """Diccionario interno de artículos (ID → DS132Article)."""
        return dict(self._articles)


__all__ = ["DS132Article", "DS132KnowledgeBase"]
