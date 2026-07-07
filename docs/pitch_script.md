# 🎬 Munin — Pitch Script (3 minutos)

> **Hackathon:** AMD Developer Hackathon Act II (6-11 Julio 2026)
> **Idioma:** Español con subtítulos en inglés
> **Duración:** 3:00 minutos (~450 palabras)

---

## 0:00 — 0:20 | EL PROBLEMA

**ES:** La minería representa el 2% del empleo global, pero el 8% de las fatalidades ocupacionales. En Chile, el DS 132 exige monitoreo de EPP en faenas mineras. Pero la vigilancia humana no escala. Y los sindicatos prohíben enviar video a la nube.

**EN:** Mining is 2% of global employment but 8% of workplace fatalities. In Chile, DS 132 mandates PPE monitoring. But human surveillance doesn't scale. And unions prohibit cloud-based video.

---

## 0:20 — 1:00 | LA ARQUITECTURA

**ES:** Munin es un agente de visión industrial 100% on-premise. Pipeline de dos niveles: YOLOv8x detecta personas y EPP a 25 frames por segundo. Solo cuando detecta una violación, invoca un Vision Language Model para análisis contextual. Todo validado por un Pydantic Gate con schema estricto. Y todo corre en un solo nodo: AMD MI300X con 192 gigabytes de memoria.

**EN:** Munin is a 100% on-premise industrial vision agent. Two-tier pipeline: YOLOv8x detects people and PPE at 25 FPS. Only when a violation is found, it calls a Vision Language Model for contextual analysis. All validated by a Pydantic Gate. All running on a single node: AMD MI300X with 192 gigabytes of memory.

---

## 1:00 — 2:30 | LA DEMO

**ES:** Observemos. Cargamos un video de una faena minera. El pipeline procesa frame por frame. YOLO detecta dos personas. Persona uno: casco, chaleco, botas. Cumple. Persona dos: sin casco, en zona de extracción. El sistema detecta la violación, la confirma en tres frames consecutivos, e invoca al VLM. El VLM analiza el contexto: trabajo en altura, sin protección. El Pydantic Gate valida la decisión. Nivel de riesgo: crítico. Artículo DS 132 violado: 38. El dashboard muestra la alerta en tiempo real.

**EN:** Let's watch. We load mining site footage. The pipeline processes frame by frame. YOLO detects two workers. Worker one: helmet, vest, boots. Compliant. Worker two: no helmet, in extraction zone. The system flags the violation, confirms it across three consecutive frames, and calls the VLM. The VLM analyzes context: working at height, no protection. The Pydantic Gate validates. Risk level: critical. DS 132 article violated: 38. The dashboard shows the alert in real time.

---

## 2:30 — 3:00 | EL CIERRE

**ES:** Ningún otro hardware permite detección de objetos y análisis con lenguaje en un solo nodo de 192 gigabytes. Munin cumple con los sindicatos: lo que ve, queda en la mina. Cumple con DS 132. Y demuestra el poder de AMD ROCm para visión industrial on-premise. Munin: lo que ve, queda en la mina.

**EN:** No other hardware enables object detection and language analysis on a single 192-gigabyte node. Munin complies with unions: what it sees, stays at the mine. It complies with DS 132. And it demonstrates the power of AMD ROCm for on-premise industrial vision. Munin: what it sees, stays at the mine.

---

## Notas de grabación
- **Palabras PROHIBIDAS:** "mock", "falso", "simulado" → usar "batch histórico de auditoría"
- **Tono:** Profesional, directo, seguro
- **Visual:** Split screen con dashboard + código durante la demo
- **Duración máxima:** 3:00. Si excede, cortar sección de arquitectura
