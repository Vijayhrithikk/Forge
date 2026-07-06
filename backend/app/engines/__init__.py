"""
Forge Engines — the heart of the application.

Each engine owns one domain:
- Dataset Engine: upload, validation, analysis, quality scoring
- Training Engine: model loading, LoRA injection, training loop (Sprint 2)
- Inference Engine: model inference, playground (Sprint 5)

Every engine follows the same lifecycle:
    initialize() → validate() → run() → status() → cleanup()
"""
