"""Smoke tests para validar la implementación del Día 2.

Verifica que los módulos básicos importen correctamente,
los schemas validen JSON, y la jerarquía de excepciones sea correcta.

Correr con: pytest munin/tests/test_smoke.py -v
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


# ============================================================================
# SECCIÓN 1: IMPORTS — Verificar que todos los módulos importan sin error
# ============================================================================

class TestImports:
    """Verifica que todos los módulos del Día 2 importen correctamente."""

    def test_import_config(self) -> None:
        """config.py importa sin error."""
        from munin.config import AppSettings, VLMBackend, Zone, AgentConfig
        assert AppSettings is not None
        assert VLMBackend is not None
        assert Zone is not None
        assert AgentConfig is not None

    def test_import_exceptions(self) -> None:
        """exceptions.py importa sin error."""
        from munin.exceptions import (
            MuninError, ConfigurationError, VideoLoadError,
            DetectionError, TrackingError, VLMError, VLMTimeoutError,
            VLMSchemaError, GateValidationError, KnowledgeBaseError,
        )
        assert MuninError is not None

    def test_import_schemas(self) -> None:
        """gate/schemas.py importa sin error."""
        from munin.gate.schemas import (
            AgentDecision, PPEMissing, DetectionResult,
            TrackedPerson, Violation,
        )
        assert AgentDecision is not None
        assert PPEMissing is not None

    def test_import_interfaces(self) -> None:
        """pipeline/interfaces.py importa sin error."""
        from munin.pipeline.interfaces import (
            IFrameExtractor, IDetector, ITracker, IComplianceChecker,
        )
        assert IFrameExtractor is not None

    def test_import_frame_extractor(self) -> None:
        """pipeline/frame_extractor.py importa sin error."""
        from munin.pipeline.frame_extractor import FrameExtractor
        assert FrameExtractor is not None

    def test_import_yolo_detector(self) -> None:
        """pipeline/yolo_detector.py importa sin error."""
        from munin.pipeline.yolo_detector import YOLODetector
        assert YOLODetector is not None

    def test_import_factory(self) -> None:
        """vlm/factory.py importa VLMModelFactory sin error."""
        from munin.vlm.factory import VLMModelFactory
        assert VLMModelFactory is not None


# ============================================================================
# SECCIÓN 2: EXCEPTIONS — Jerarquía correcta
# ============================================================================

class TestExceptions:
    """Verifica la jerarquía de excepciones custom."""

    def test_munin_error_is_base(self) -> None:
        """MuninError es la base de todas las excepciones."""
        from munin.exceptions import MuninError
        assert issubclass(MuninError, Exception)

    def test_all_inherit_from_munin_error(self) -> None:
        """Todas las excepciones heredan de MuninError."""
        from munin.exceptions import (
            MuninError, ConfigurationError, VideoLoadError,
            DetectionError, TrackingError, VLMError,
            GateValidationError, KnowledgeBaseError,
        )
        for exc in [ConfigurationError, VideoLoadError, DetectionError,
                     TrackingError, VLMError, GateValidationError,
                     KnowledgeBaseError]:
            assert issubclass(exc, MuninError), f"{exc.__name__} no hereda de MuninError"

    def test_vlm_timeout_inherits_vlm_error(self) -> None:
        """VLMTimeoutError hereda de VLMError."""
        from munin.exceptions import VLMTimeoutError, VLMError
        assert issubclass(VLMTimeoutError, VLMError)

    def test_vlm_schema_inherits_vlm_error(self) -> None:
        """VLMSchemaError hereda de VLMError."""
        from munin.exceptions import VLMSchemaError, VLMError
        assert issubclass(VLMSchemaError, VLMError)

    def test_exceptions_can_be_raised(self) -> None:
        """Las excepciones pueden ser instanciadas y lanzadas."""
        from munin.exceptions import MuninError, VLMError
        with pytest.raises(MuninError):
            raise VLMError("test error")
        with pytest.raises(VLMError):
            raise VLMError("test")


# ============================================================================
# SECCIÓN 3: SCHEMAS — Pydantic models validan correctamente
# ============================================================================

class TestSchemas:
    """Verifica que los schemas Pydantic validen JSON correctamente."""

    def test_ppemissing_valid(self) -> None:
        """PPEMissing valida un tipo correcto."""
        from munin.gate.schemas import PPEMissing
        item = PPEMissing(tipo="hardhat", descripcion="Casco de seguridad", norma_chilena="NCh 1411")
        assert item.tipo == "hardhat"
        assert item.descripcion == "Casco de seguridad"

    def test_ppemissing_invalid_tipo(self) -> None:
        """PPEMissing rechaza tipo inválido."""
        from munin.gate.schemas import PPEMissing
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PPEMissing(tipo="invalid", descripcion="test", norma_chilena="NCh")

    def test_detection_result_valid(self) -> None:
        """DetectionResult valida con todos los campos."""
        from munin.gate.schemas import DetectionResult
        det = DetectionResult(class_name="person", bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.85)
        assert det.class_name == "person"
        assert det.confidence == 0.85

    def test_detection_result_confidence_out_of_range(self) -> None:
        """DetectionResult rechaza confidence > 1.0."""
        from munin.gate.schemas import DetectionResult
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectionResult(class_name="person", bbox=(0, 0, 1, 1), confidence=1.5)

    def test_agent_decision_with_violation(self) -> None:
        """AgentDecision valida con violación CRITICAL."""
        from munin.gate.schemas import AgentDecision, PPEMissing
        decision = AgentDecision(
            zona="extraccion",
            tipo_violacion="EPP_FALTANTE",
            epp_faltante=[PPEMissing(tipo="hardhat", descripcion="Casco", norma_chilena="NCh 1411")],
            nivel_riesgo="CRITICO",
            timestamp=datetime.now(),
            articulo_ds132="Art. 38",
            confianza=0.92,
            requiere_revision_humana=False,
            razonamiento_vlm="Persona sin casco en zona de extracción",
        )
        assert decision.zona == "extraccion"
        assert decision.nivel_riesgo == "CRITICO"
        assert len(decision.epp_faltante) == 1

    def test_agent_decision_no_violation(self) -> None:
        """AgentDecision valida sin violación."""
        from munin.gate.schemas import AgentDecision
        decision = AgentDecision(
            zona="procesamiento",
            tipo_violacion="SIN_VIOLACION",
            epp_faltante=[],
            nivel_riesgo="BAJO",
            timestamp=datetime.now(),
            confianza=0.88,
        )
        assert decision.tipo_violacion == "SIN_VIOLACION"

    def test_agent_decision_from_json(self) -> None:
        """AgentDecision parsea JSON string correctamente."""
        from munin.gate.schemas import AgentDecision
        json_str = json.dumps({
            "zona": "extraccion",
            "tipo_violacion": "EPP_FALTANTE",
            "epp_faltante": [{"tipo": "hardhat", "descripcion": "Casco", "norma_chilena": "NCh 1411"}],
            "nivel_riesgo": "ALTO",
            "timestamp": "2026-07-07T14:30:00",
            "articulo_ds132": "Art. 38",
            "confianza": 0.90,
            "requiere_revision_humana": False,
            "razonamiento_vlm": "Falta casco",
        })
        decision = AgentDecision.model_validate_json(json_str)
        assert decision.nivel_riesgo == "ALTO"
        assert decision.epp_faltante[0].tipo == "hardhat"

    def test_agent_decision_invalid_riesgo(self) -> None:
        """AgentDecision rechaza nivel_riesgo inválido."""
        from munin.gate.schemas import AgentDecision
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentDecision(
                zona="test", tipo_violacion="SIN_VIOLACION",
                nivel_riesgo="INVALID", timestamp=datetime.now(), confianza=0.5,
            )


# ============================================================================
# SECCIÓN 4: CONFIG — AppSettings carga correctamente
# ============================================================================

class TestConfig:
    """Verifica que la configuración funcione."""

    def test_vlm_backend_enum(self) -> None:
        """VLMBackend enum tiene los valores correctos."""
        from munin.config import VLMBackend
        assert VLMBackend.FIREWORKS.value == "fireworks"
        assert VLMBackend.AMD.value == "amd"

    def test_zone_model_valid(self) -> None:
        """Zone model valida con campos correctos."""
        from munin.config import Zone
        zone = Zone(
            zone_id="extraccion",
            nombre="Zona de Extracción",
            required_epp=["hardhat", "safety_vest", "safety_boots", "harness"],
            riesgo_base="alto",
            min_confidence=0.6,
            articulos_ds132=["Art. 38", "Art. 42"],
        )
        assert zone.zone_id == "extraccion"
        assert zone.riesgo_base == "alto"

    def test_agent_config_defaults(self) -> None:
        """AgentConfig tiene defaults sensatos."""
        from munin.config import AgentConfig
        config = AgentConfig()
        assert config.timeout == 300.0
        assert config.max_retries == 3
        assert config.temperature == 0.1

    def test_app_settings_defaults(self) -> None:
        """AppSettings carga con defaults (sin .env)."""
        from munin.config import AppSettings, VLMBackend
        # AppSettings puede cargar sin .env porque todos los campos tienen defaults
        settings = AppSettings(_env_file=None)
        assert settings.vlm_backend == VLMBackend.FIREWORKS
        assert settings.frame_rate == 25
        assert settings.min_consecutive_frames == 3
        # Nuevos campos v3
        assert settings.compliance_mode == "legacy"
        assert settings.yolo_stream_mode is False
        assert settings.yolo_imgsz == 640
        assert settings.frame_resize_width == 640
        assert settings.frame_resize_height == 480
        assert settings.prompt_cache_session_id == "munin-session"
        assert settings.yolo_ppe_model_path == "/scratch/runs/detect/train/weights/best.pt"
        assert settings.vlm_busy_timeout == 300.0


# ============================================================================
# SECCIÓN 5: PPE_CLASS_MAP — YOLO class mapping correcto (modelo PPE)
# ============================================================================

class TestYOLOClassMap:
    """Verifica que PPE_CLASS_MAP tenga las clases correctas del modelo PPE."""

    def test_class_map_exists(self) -> None:
        """PPE_CLASS_MAP existe y tiene 6 clases EPP."""
        from munin.pipeline.yolo_detector import PPE_CLASS_MAP
        assert len(PPE_CLASS_MAP) == 6

    def test_class_map_has_hardhat(self) -> None:
        """PPE_CLASS_MAP mapea 3 → hardhat (helmet en modelo real)."""
        from munin.pipeline.yolo_detector import PPE_CLASS_MAP
        assert PPE_CLASS_MAP[3] == "hardhat"

    def test_class_map_has_gloves(self) -> None:
        """PPE_CLASS_MAP mapea 0 → gloves."""
        from munin.pipeline.yolo_detector import PPE_CLASS_MAP
        assert PPE_CLASS_MAP[0] == "gloves"


# SECCIÓN 6: FACTORY — VLMModelFactory lanza errores correctos
# ============================================================================

class TestVLMModelFactory:
    """Verifica VLMModelFactory."""

    def test_factory_missing_api_key_raises(self) -> None:
        """Factory lanza ConfigurationError si falta API key para Fireworks."""
        from munin.vlm.factory import VLMModelFactory
        from munin.config import AppSettings
        from munin.exceptions import ConfigurationError

        settings = AppSettings()
        settings.fireworks_api_key = ""
        # Fireworks sin API key debe lanzar ConfigurationError
        with pytest.raises(ConfigurationError):
            VLMModelFactory.create(settings)
    
    def test_factory_creates_fireworks_model(self) -> None:
        """Factory con API key crea un OpenAIChatModel (sin conexión real)."""
        from munin.vlm.factory import VLMModelFactory
        from munin.config import AppSettings
        from pydantic_ai.models.openai import OpenAIChatModel

        settings = AppSettings()
        settings.fireworks_api_key = "test-key-123"
        
        model = VLMModelFactory.create(settings)
        assert isinstance(model, OpenAIChatModel)
