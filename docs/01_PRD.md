# LoRA Studio aka Forge — Engineering Implementation Specification (EIS) v1.0

## Mission

Build a production-inspired, deployable web application that demonstrates end-to-end LoRA fine-tuning of open-source language models.

The goal is engineering quality, not feature count.

Do not redesign the architecture.

Implement only what is specified.

---

# Success Criteria

The repository must satisfy all of the following:

* Builds successfully
* Runs locally with Docker
* Has a beautiful and responsive UI
* Fine-tunes a supported model with LoRA
* Streams progress and logs live
* Produces downloadable artifacts
* Supports inference inside the application
* Is fully documented

---

# Development Principles

1. Never sacrifice reliability for speed.
2. Never silently ignore errors.
3. Every public function must have one clear responsibility.
4. Every long-running task must report progress.
5. Every exception must be actionable.
6. Prefer readable code over clever code.
7. Keep architecture frozen.
8. Do not introduce new frameworks without necessity.

---

# Implementation Order

## Milestone 1 — Repository Bootstrap

Deliverables:

* Repository structure
* Docker setup
* Environment configuration
* Frontend shell
* Backend shell
* Shared configuration
* Health endpoint

Acceptance:

Application starts successfully.

---

## Milestone 2 — Dataset Engine

Deliverables:

* JSONL upload
* Schema validation
* Duplicate detection
* Empty field detection
* Token estimation
* Dataset statistics
* Dataset quality report

Acceptance:

Invalid datasets fail with actionable messages.

---

## Milestone 3 — Training Engine

Deliverables:

* Model loading
* LoRA injection
* Configuration validation
* Training pipeline
* Checkpoint handling
* Metrics collection

Acceptance:

A sample dataset trains successfully.

---

## Milestone 4 — Live Monitoring

Deliverables:

* Server Sent Events
* Live logs
* Live metrics
* Progress updates
* ETA estimation
* Stage visualization

Acceptance:

The UI reflects backend progress in real time.

---

## Milestone 5 — Evaluation

Deliverables:

* Training metrics
* Loss visualization
* Adapter verification
* Training summary

Acceptance:

Metrics are persisted and viewable.

---

## Milestone 6 — Playground

Deliverables:

* Base model inference
* Fine-tuned model inference
* Side-by-side comparison

Acceptance:

Users can compare outputs.

---

## Milestone 7 — Artifact Packaging

Deliverables:

* adapter.safetensors
* adapter_config.json
* metrics.json
* logs.jsonl
* training_report.pdf

Acceptance:

Artifacts download successfully.

---

## Milestone 8 — UI Polish

Deliverables:

* Loading states
* Empty states
* Error states
* Success states
* Responsive layout
* Animations
* Accessibility improvements

Acceptance:

Professional presentation.

---

## Milestone 9 — Deployment

Deliverables:

* Docker Compose
* Environment documentation
* Production build
* README
* Demo assets

Acceptance:

Fresh clone → one command → running application.

---

# Coding Standards

* Python: type hints everywhere practical.
* TypeScript: strict mode.
* No wildcard imports.
* Clear naming.
* Small functions.
* Modular components.
* Structured logging.
* Consistent formatting.

---

# Error Handling

Every error must include:

* Error code
* Human-readable title
* Description
* Recommendation
* Recoverable flag
* Optional technical details

Never expose raw stack traces to users.

---

# Logging

Every pipeline stage logs:

* Stage entered
* Stage completed
* Duration
* Warnings
* Errors
* Artifact creation

Logs must be streamable to the frontend.

---

# Progress

Each stage reports:

* Current stage
* Percentage
* Current step
* Total steps
* ETA
* Elapsed time

No indefinite spinners.

---

# Retry Policy

Automatically retry:

* Model downloads
* Temporary network failures
* Temporary file write failures

Do not automatically retry:

* Invalid datasets
* Unsupported models
* Configuration errors
* Out-of-memory conditions

Provide clear recovery guidance instead.

---

# Quality Gates

Before any milestone is considered complete:

* Code builds.
* Manual smoke test passes.
* Logs are meaningful.
* Errors are actionable.
* UI states are complete.
* No TODO placeholders remain.

---

# Scope Control

Any proposed feature must answer:

"Does this improve recruiter value for Version 1?"

If the answer is "No" or "Not directly," defer it to Version 2.

---

# Definition of Done

The project is complete only when a user can:

1. Launch the application.
2. Upload a JSONL dataset.
3. Receive a validation report.
4. Configure LoRA training.
5. Watch live progress.
6. Observe structured logs.
7. Download the trained adapter.
8. Compare the base and fine-tuned model.
9. Download the training report.
10. Follow the README to reproduce the workflow.
