# LoRA Studio aks Forge — Software Architecture Specification v1.0

---

# Architecture Philosophy

LoRA Studio is built around one engineering principle:

> Every stage of the fine-tuning pipeline must be observable, deterministic, fault-tolerant, and independently testable.

The application consists of four logical layers only.

No microservices.

No event buses.

No unnecessary complexity.

---

# High-Level Architecture

```
                        Browser
                           │
                           │
             Next.js + Tailwind + shadcn/ui
                           │
                   REST + Server Sent Events
                           │
                    FastAPI Application
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
 Dataset Engine      Training Engine     Inference Engine
      │                    │                    │
      └────────────────────┼────────────────────┘
                           │
                Transformers + PEFT + TRL
                           │
                    HuggingFace Models
                           │
                Outputs / Models / Adapters
```

---

# Repository

```
lora-studio/

frontend/

backend/

README.md

docker-compose.yml

.env.example
```

No additional top-level folders.

---

# Frontend

```
frontend/

app/

components/

hooks/

lib/

types/

public/
```

---

## Responsibilities

Frontend owns:

* UI
* Forms
* Progress
* Charts
* Live logs
* Playground
* API communication

Frontend never performs training.

---

# Backend

```
backend/

app/

api/

core/

engines/

models/

schemas/

services/

storage/

utils/

main.py
```

---

# Folder Responsibilities

---

## api/

Only HTTP endpoints.

No business logic.

No ML code.

Only:

* receive request
* validate request
* call service
* return response

---

## core/

Application configuration.

Logging.

Environment.

Constants.

Startup.

Shutdown.

Nothing else.

---

## engines/

Heart of the application.

Contains:

Dataset Engine

Training Engine

Inference Engine

Evaluation Engine

Each engine exposes:

```
initialize()

validate()

run()

status()

cleanup()
```

Every engine follows identical lifecycle.

---

## models/

Internal Python models.

No API schemas.

Only internal state.

---

## schemas/

Pydantic request/response models.

Everything crossing HTTP lives here.

---

## services/

Application orchestration.

Example:

```
TrainingService

↓

Calls

Dataset Engine

↓

Training Engine

↓

Evaluation Engine

↓

Packaging
```

Services coordinate.

They never perform ML work.

---

## storage/

Handles:

Datasets

Outputs

Adapters

Reports

Temporary Files

Single responsibility:

File management.

---

## utils/

Pure helper functions.

No state.

No business logic.

---

# Training Pipeline

```
Dataset Upload

↓

Dataset Validation

↓

Dataset Statistics

↓

Formatting

↓

Tokenization

↓

LoRA Injection

↓

Training

↓

Evaluation

↓

Packaging

↓

Artifact Generation

↓

Completed
```

Every stage produces:

Status

Progress

Logs

Duration

Artifacts

---

# State Machine

Every training job exists in exactly one state.

```
CREATED

↓

VALIDATING

↓

READY

↓

TRAINING

↓

EVALUATING

↓

PACKAGING

↓

COMPLETED
```

Failure can occur from any state.

```
FAILED
```

Cancellation

```
CANCELLED
```

No hidden states.

---

# API Endpoints

Only five.

```
POST /projects

POST /datasets/upload

POST /training/start

GET /training/status

POST /playground/chat
```

Nothing else in Version 1.

---

# Frontend Pages

Landing

Dashboard

New Project

Training

Playground

Settings (read-only application information)

---

# Logging

Every log follows:

```
Timestamp

Level

Component

Stage

Message

Metadata
```

Example

```
INFO

TrainingEngine

Epoch 2

Loss improved to 1.42
```

---

# Progress Events

Every stage emits:

```
Current Stage

Percentage

Current Step

Total Steps

Elapsed

ETA
```

Frontend never estimates.

Backend owns progress.

---

# Error Contract

Every exception returns:

```
error_code

title

description

recommendation

recoverable

details
```

Never return plain strings.

---

# Retry Strategy

Retry only:

Model download

Network

Temporary IO

Never retry:

Bad datasets

Invalid configs

OOM

Unsupported model

---

# Artifact Structure

```
outputs/

project-id/

adapter/

metrics.json

report.pdf

logs.jsonl

adapter.safetensors

adapter_config.json
```

Every training run is isolated.

---

# Configuration

Single configuration object.

Contains:

Model

Training

LoRA

Output

Logging

Never scatter configuration.

---

# Communication

Training progress uses:

Server Sent Events (SSE)

Reason:

Simpler than WebSockets

Excellent for one-way streaming

Perfect for logs and progress

---

# UI Principles

Every page must support:

Loading

Empty

Success

Error

Completed

No blank screens.

---

# Engineering Rule

Every module must answer one question:

"If this file disappeared tomorrow, what responsibility would disappear?"

If the answer is

"Nothing"

Delete the file.

---

# Final Principle

The architecture is frozen.

No folder may be added without a documented reason.

No service may be created unless an existing service has multiple responsibilities.

No abstraction may be introduced before it solves an existing problem.
