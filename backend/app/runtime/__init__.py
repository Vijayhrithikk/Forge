"""
Forge Execution Runtime — the heart of Forge.

The Runtime owns every operation from validating a Training Plan
until training completes. Trainers are implementation details.
Everything flows through the Runtime.

Mission A (Parts 1-4): Execution Foundation
- State machine, manifest, locks, workspace, environment, GPU
- Coordinator, events, logging, metrics, audit
- No model loading, no tokenizer, no PEFT, no training

Mission B: Model Runtime + Training Loop (future)
"""
