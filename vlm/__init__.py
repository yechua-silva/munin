from __future__ import annotations

"""VLM layer — modelos de Vision Language Model via PydanticAI Agents.

Soporta múltiples backends:
- FireworksProvider: Fireworks AI API (interim, cloud)
- AMDvLLMProvider: AMD MI300X via vLLM ROCm (target, on-premise)
- VLMModelFactory: Factory para crear el backend correcto según config
"""
