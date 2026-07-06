# Forge — UI/UX Specification v1.0

## Design Philosophy

Forge should feel like a professional engineering tool rather than an academic dashboard.

Inspired by:

* Linear
* Vercel
* Cursor
* Weights & Biases

Core principles:

* Minimal
* Spacious
* Fast
* Informative
* Calm
* No visual clutter

Animations should communicate progress, not decorate the interface.

---

# Navigation

Top Navigation

---

Forge

Projects

Documentation

GitHub

Settings

Launch Studio

---

No sidebar until the user enters Studio.

---

# Landing Page

## Hero

Large headline

Train Open Models.
Not Your Patience.

Supporting text

Build, monitor and evaluate LoRA fine-tuning pipelines with complete observability.

Buttons

Launch Studio

View GitHub

---

## Hero Animation

Instead of decorative graphics...

Animate the actual ML pipeline.

Dataset

↓

Validation

↓

Tokenization

↓

LoRA Injection

↓

Training

↓

Evaluation

↓

Adapter

Each stage lights up sequentially.

This immediately explains what Forge does.

---

## Feature Section

Six cards.

Dataset Validation

↓

Training Engine

↓

Observability

↓

Evaluation

↓

Playground

↓

Artifact Export

No long paragraphs.

One sentence each.

---

## Engineering Principles

Display visually.

Reliable

Observable

Deterministic

Deployable

Simple

Production Inspired

---

## Footer

GitHub

Documentation

License

Acknowledgements

Version

---

# Studio Dashboard

Left Sidebar

Projects

Training

Playground

Artifacts

Logs

Settings

---

Main Dashboard

Header

Current Project

Model

Current Status

Start New Training

---

Cards

Training Status

Current Stage

Loss

ETA

Elapsed Time

GPU

Token Count

Trainable Parameters

---

Training Pipeline

Instead of one progress bar...

Represent the pipeline itself.

Dataset

✓

↓

Validation

✓

↓

Formatting

✓

↓

Tokenization

✓

↓

LoRA

Running

↓

Training

Running

↓

Evaluation

Waiting

↓

Packaging

Waiting

Every node has:

Pending

Running

Completed

Failed

Cancelled

---

# Logs Panel

Real-time streaming.

Example

[INFO]

Dataset uploaded.

---

[SUCCESS]

Schema validated.

---

[INFO]

Tokenization completed.

---

[INFO]

Training Epoch 2.

---

[WARNING]

Long sequence truncated.

---

[SUCCESS]

Adapter exported.

Logs are filterable.

Levels:

INFO

SUCCESS

WARNING

ERROR

---

# Metrics

Loss Curve

Learning Rate

Tokens Processed

Training Speed

GPU Memory

Elapsed Time

ETA

No unnecessary charts.

Every graph must answer a question.

---

# Dataset Upload

Large drag-and-drop area.

Supported

JSONL

Immediately after upload:

Dataset Summary

Examples

Token Estimate

Duplicate Count

Empty Samples

Sequence Length Distribution

Estimated Training Time

Quality Score (0–100)

The user should know whether the dataset is ready before training.

---

# Training Configuration

Simple.

Base Model

Dropdown

Epochs

Slider

Learning Rate

Input

Batch Size

Input

LoRA Rank

Slider

LoRA Alpha

Input

Dropout

Slider

Right Panel

Estimated Adapter Size

Estimated Training Time

Estimated GPU Usage

Trainable Parameters

---

# Training Screen

Top

Current Stage

Center

Pipeline

Right

Live Metrics

Bottom

Streaming Logs

Everything updates live.

No refresh button.

---

# Completion Screen

Large Success Card

Training Complete

Adapter Size

Training Duration

Final Loss

Model

Buttons

Download Adapter

Open Playground

Download Report

Start New Training

---

# Playground

Split Screen

Prompt

Input Box

Buttons

Compare

Clear

Responses

Base Model

Fine-Tuned Model

Differences should be visually highlighted.

---

# Artifacts

Each artifact appears as a card.

adapter.safetensors

Size

Download

---

metrics.json

Download

---

logs.jsonl

Download

---

training_report.pdf

Download

---

# Error Experience

Never show raw exceptions.

Every error card contains:

Title

Description

Reason

Recommendation

Retry (if applicable)

Example

Dataset Validation Failed

12 empty responses detected.

Recommendation:

Remove empty responses or enable automatic filtering.

---

# Loading States

Every page has skeleton loading.

Never blank.

Never spinner-only.

---

# Empty States

Projects

"No training projects yet."

Upload Dataset

"No dataset selected."

Playground

"Complete a training run to compare models."

Artifacts

"No artifacts generated."

---

# Notifications

Small toast notifications.

Examples

Dataset uploaded successfully.

Training started.

Evaluation complete.

Adapter ready for download.

Never interrupt the user.

---

# Motion System

Fade

Slide

Scale

Progress Fill

Skeleton Loading

Stage Transitions

No bounce.

No flip.

No excessive animation.

---

# Theme

Default:

Dark

Optional:

Light

Typography:

Geist

Icons:

Lucide

Corner Radius:

Large

Spacing:

Generous

---

# The Recruiter Journey

1. Open the landing page.
2. Understand Forge within 10 seconds.
3. Click Launch Studio.
4. Upload a dataset.
5. Watch a beautiful, transparent training pipeline.
6. See metrics improve live.
7. Download the adapter.
8. Compare the base and fine-tuned models.
9. Leave thinking:

"This feels like a real ML engineering product."

---

# Version 1 Freeze

No additional UI pages may be added.

No feature may be added unless it directly improves:

* clarity,
* engineering quality,
* recruiter experience,
* or usability.

Everything else belongs in Version 2.
