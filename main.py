from __future__ import annotations

import logging

import uvicorn

from munin.api.routes import MuninAPI
from munin.config import AppSettings
from munin.pipeline.factory import PipelineFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

settings = AppSettings()
logger.info(
    "Munin starting — VLM backend: %s, model: %s",
    settings.vlm_backend.value,
    settings.fireworks_model,
)

pipeline = PipelineFactory.create(settings)
api = MuninAPI(pipeline, settings)
app = api.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
