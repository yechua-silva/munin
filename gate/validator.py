from __future__ import annotations

import logging
from json import JSONDecodeError

from pydantic import ValidationError

from munin.exceptions import GateValidationError
from munin.gate.schemas import AgentDecision

logger = logging.getLogger(__name__)


class PydanticGate:
    """Valida output VLM contra AgentDecision schema con retry.

    Implementa un retry loop que reintenta la validación del mismo
    output VLM hasta max_retries veces. Si todos fallan, lanza
    GateValidationError.

    Attributes:
        _max_retries: Intentos máximos de validación.
    """

    def __init__(self, max_retries: int = 3) -> None:
        """Inicializa el gate con número de reintentos.

        Args:
            max_retries: Intentos máximos de validación (default: 3).
        """
        self._max_retries = max_retries

    def validate(self, vlm_output: str) -> AgentDecision:
        """Valida y parsea output VLM contra AgentDecision schema.

        Args:
            vlm_output: String JSON proveniente del VLM.

        Returns:
            AgentDecision validado.

        Raises:
            GateValidationError: Si falla la validación después de
                max_retries intentos.
        """
        for attempt in range(self._max_retries):
            try:
                return AgentDecision.model_validate_json(vlm_output)
            except (ValidationError, JSONDecodeError) as e:
                if attempt == self._max_retries - 1:
                    raise GateValidationError(
                        f"Failed to validate VLM output after "
                        f"{self._max_retries} attempts: {e}"
                    ) from e
                logger.warning(
                    "Gate validation attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )

        raise GateValidationError(
            "Unexpected error validating VLM output"
        )


__all__ = ["PydanticGate"]
