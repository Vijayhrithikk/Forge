# Forge — TASKS.md

> **Execution Queue for Forge v1.0**

---

# Overview

This document defines every implementation task required to complete Forge Version 1.0.

Tasks are intentionally small, independently reviewable, and produce measurable progress.

A task is complete only when its acceptance criteria are satisfied.

No task may modify the architecture.

---

# Task Status

Each task must always be one of:

* TODO
* IN PROGRESS
* REVIEW
* DONE
* BLOCKED

---

# Sprint 0 — Repository Foundation

---

## TASK-001

### Title

Initialize Repository

### Status

TODO

### Priority

Critical

### Dependencies

None

### Deliverables

* Repository initialized
* Branch strategy documented
* README placeholder
* LICENSE
* `.gitignore`
* `docs/`
* `.claude/`
* Docker Compose placeholder

### Acceptance Criteria

* Repository structure matches Architecture.md
* Docker Compose validates
* README renders correctly

---

## TASK-002

### Title

Backend Bootstrap

### Status

TODO

### Dependencies

TASK-001

### Deliverables

* FastAPI project
* Health endpoint
* Environment loader
* Logging initialized

### Acceptance Criteria

GET /health returns success.

---

## TASK-003

### Title

Frontend Bootstrap

### Status

TODO

### Dependencies

TASK-001

### Deliverables

* Next.js
* Tailwind
* shadcn/ui
* Layout
* Theme

### Acceptance Criteria

Landing page loads successfully.

---

# Sprint 1 — Dataset Engine

---

## TASK-004

### Title

Dataset Upload

### Dependencies

TASK-002

### Deliverables

* Upload endpoint
* File validation
* Storage

### Acceptance Criteria

Valid JSONL uploads successfully.

---

## TASK-005

### Title

Dataset Validation

### Dependencies

TASK-004

### Deliverables

Validate:

* Empty prompts
* Empty responses
* Invalid schema
* Duplicate samples
* Encoding

### Acceptance Criteria

Validation report generated.

---

## TASK-006

### Title

Dataset Statistics

### Dependencies

TASK-005

### Deliverables

Generate:

* Sample count
* Token estimate
* Average sequence length
* Maximum length
* Quality score

### Acceptance Criteria

Statistics visible in UI.

---

# Sprint 2 — Training Engine

---

## TASK-007

### Title

Model Loader

### Dependencies

TASK-006

### Deliverables

* Load supported model
* Load tokenizer
* Configuration validation

### Acceptance Criteria

Model loads successfully.

---

## TASK-008

### Title

LoRA Injection

### Dependencies

TASK-007

### Deliverables

* Configure LoRA
* Validate parameters
* Report trainable parameters

### Acceptance Criteria

LoRA successfully attached.

---

## TASK-009

### Title

Training Pipeline

### Dependencies

TASK-008

### Deliverables

* Tokenization
* Trainer
* Checkpoints
* Metrics

### Acceptance Criteria

Reference dataset completes training.

---

# Sprint 3 — Observability

---

## TASK-010

### Title

Training State Machine

### Dependencies

TASK-009

### Deliverables

States:

* CREATED
* VALIDATING
* READY
* TRAINING
* EVALUATING
* PACKAGING
* COMPLETED
* FAILED
* CANCELLED

### Acceptance Criteria

Every transition logged.

---

## TASK-011

### Title

Server-Sent Events

### Dependencies

TASK-010

### Deliverables

Live stream:

* Logs
* Stage
* Progress
* ETA

### Acceptance Criteria

Dashboard updates without refresh.

---

## TASK-012

### Title

Structured Logging

### Dependencies

TASK-011

### Deliverables

Structured logs for:

* Validation
* Training
* Evaluation
* Packaging

### Acceptance Criteria

Logs visible in UI.

---

# Sprint 4 — Evaluation

---

## TASK-013

### Title

Metrics Engine

### Dependencies

TASK-009

### Deliverables

Generate:

* Loss history
* Duration
* Training summary

### Acceptance Criteria

Metrics persisted.

---

## TASK-014

### Title

Training Report

### Dependencies

TASK-013

### Deliverables

Generate:

* PDF report
* JSON metrics

### Acceptance Criteria

Reports downloadable.

---

# Sprint 5 — Playground

---

## TASK-015

### Title

Inference Engine

### Dependencies

TASK-014

### Deliverables

Load:

* Base model
* Adapter model

### Acceptance Criteria

Inference succeeds.

---

## TASK-016

### Title

Comparison Playground

### Dependencies

TASK-015

### Deliverables

Display:

* Base response
* Fine-tuned response

### Acceptance Criteria

Comparison page operational.

---

# Sprint 6 — Artifact Management

---

## TASK-017

### Title

Artifact Packaging

### Dependencies

TASK-016

### Deliverables

Export:

* adapter.safetensors
* adapter_config.json
* metrics.json
* logs.jsonl
* training_report.pdf

### Acceptance Criteria

All artifacts downloadable.

---

# Sprint 7 — UI Polish

---

## TASK-018

### Title

Dashboard Completion

### Dependencies

TASK-017

### Deliverables

* Progress pipeline
* Charts
* Empty states
* Loading states
* Error states

### Acceptance Criteria

Dashboard matches UI specification.

---

## TASK-019

### Title

Landing Page

### Dependencies

TASK-003

### Deliverables

* Hero
* Pipeline animation
* Features
* CTA

### Acceptance Criteria

Matches UI mockup.

---

# Sprint 8 — Release

---

## TASK-020

### Title

Production Release

### Dependencies

All previous tasks

### Deliverables

* README finalized
* Docker verified
* Screenshots
* Demo recording
* Environment documentation

### Acceptance Criteria

A new user can:

1. Clone the repository.
2. Launch the application.
3. Train a model.
4. Download artifacts.
5. Compare the fine-tuned model.

Without modifying the source code.

---

# Review Checklist (Run After Every Task)

* Builds successfully.
* Tests pass (where applicable).
* Logs are meaningful.
* Errors are actionable.
* No TODO placeholders.
* No unnecessary abstractions.
* Documentation updated.
* Acceptance criteria satisfied.

A task cannot move to **DONE** until every checklist item passes.

---

# Claude Code Execution Rules

For every implementation session:

1. Read the Engineering Constitution.
2. Read the Architecture document.
3. Read this TASKS document.
4. Implement **only the next unfinished task**.
5. Do not redesign the architecture.
6. Explain implementation decisions.
7. Stop after completing the assigned task.

Never implement future tasks preemptively.

---

# Project Principle

Forge is completed one verified task at a time.

No shortcuts.

No skipped milestones.

No unfinished features.

Every completed task should move the repository one step closer to a production-quality release.
