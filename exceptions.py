from __future__ import annotations


class MuninError(Exception):
    """Excepción base para todos los errores de Munin.

    Todas las excepciones custom del sistema heredan de esta clase.
    """


class ConfigurationError(MuninError):
    """Error de configuración del sistema.

    Se lanza cuando las variables de entorno son inválidas,
    archivos de configuración no existen, o valores son inconsistentes.
    """


class VideoLoadError(MuninError):
    """Error al cargar o procesar video.

    Se lanza cuando un archivo MP4 no puede ser abierto,
    está corrupto, o no contiene frames.
    """


class DetectionError(MuninError):
    """Error durante la inferencia de YOLO.

    Se lanza cuando el modelo no puede cargar, la GPU falla,
    o la inferencia produce resultados inválidos.
    """


class TrackingError(MuninError):
    """Error durante el tracking de personas.

    Se lanza cuando hay errores en el matching IoU o
    índices inválidos en el tracker.
    """


class VLMError(MuninError):
    """Error genérico del VLM.

    Se lanza cuando la API del VLM retorna error, rate limit,
    o la respuesta es inválida.
    """


class VLMTimeoutError(VLMError):
    """El VLM excedió el timeout configurado.

    Se lanza cuando una llamada al VLM no responde en el tiempo
    configurado (default: 30 segundos).
    """


class VLMSchemaError(VLMError):
    """El VLM devolvió un response que no cumple el schema.

    Se lanza cuando el VLM retorna JSON que no matchea
    el schema AgentDecision esperado.
    """


class GateValidationError(MuninError):
    """El Pydantic Gate no pudo validar el output.

    Se lanza cuando el output del VLM no puede ser validado
    después de max_retries intentos.
    """


class KnowledgeBaseError(MuninError):
    """Error al cargar o consultar la knowledge base DS 132.

    Se lanza cuando el archivo JSON no existe, está corrupto,
    o un artículo/zona no se encuentra.
    """


__all__ = [
    "MuninError",
    "ConfigurationError",
    "VideoLoadError",
    "DetectionError",
    "TrackingError",
    "VLMError",
    "VLMTimeoutError",
    "VLMSchemaError",
    "GateValidationError",
    "KnowledgeBaseError",
]
