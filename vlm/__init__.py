"""VLM layer — clientes de Vision Language Model (Strategy Pattern).

Soporta múltiples backends:
- FireworksVLMClient: Fireworks AI API (interim, cloud)
- AMDvLLMClient: AMD MI300X via vLLM ROCm (target, on-premise)
- VLMClientFactory: Factory para crear el backend correcto según config
"""
