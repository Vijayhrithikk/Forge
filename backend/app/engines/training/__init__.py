"""
Training Configuration Engine — owns all pre-training planning.

This sprint does NOT perform training. Instead it prepares a
complete, validated training plan that Sprint 3 will execute.

Modules:
- registry: Model registry loader
- config: LoRA and hyperparameter configuration engines
- compatibility: Dataset/Model/LoRA/hardware compatibility checks
- estimation: VRAM, training time, adapter size estimators
- scorer: Configuration quality scoring
- plan: Training plan generator and exporter
"""
