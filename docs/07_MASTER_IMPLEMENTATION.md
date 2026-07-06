# Forge — MASTER_IMPLEMENTATION.md

> **Implementation Guide for Claude Code**

Version 1.0

---

# Introduction

Welcome to the Forge engineering team.

Forge is a production-inspired AI engineering project focused on building a reliable, observable, and deployable LoRA fine-tuning platform.

Your responsibility is not to design the product.

Your responsibility is to implement the product exactly as specified.

Every architectural decision has already been made.

Do not redesign the system.

Do not introduce additional abstractions.

Do not change project scope.

Read the documentation before writing code.

---

# Required Reading Order

Before implementing anything, read the following documents in order:

1. Engineering Constitution
2. Product Requirements Document
3. Architecture Specification
4. UI/UX Specification
5. Implementation Plan
6. TASKS.md

Implementation begins only after these documents are understood.

---

# Primary Objective

Produce production-inspired software.

Not academic software.

Not prototype code.

Not demonstration code.

Every implementation should be maintainable, observable, and easy to review.

---

# Engineering Philosophy

Every implementation must reinforce the following values:

* Reliability
* Simplicity
* Determinism
* Observability
* Maintainability
* Readability
* Professional user experience

Never optimize for feature count.

Optimize for engineering quality.

---

# Scope Discipline

Version 1 includes only:

* JSONL datasets
* LoRA fine-tuning
* Live training dashboard
* Dataset validation
* Playground
* Artifact download

Everything else belongs to future versions.

If an implementation idea is outside Version 1 scope, stop and explain why instead of implementing it.

---

# Repository Rules

Follow the repository structure exactly.

Do not create new top-level folders.

Do not introduce new architectural layers.

Every new file must have one clear responsibility.

Before creating a file, ask:

"What responsibility does this file own?"

If the answer is unclear, do not create it.

---

# Task Execution

Always implement only one task from TASKS.md.

Never implement future tasks.

Never merge multiple milestones.

When a task is complete:

Stop.

Summarize:

* Files created
* Files modified
* Design decisions
* Remaining work

Do not continue automatically.

---

# Coding Standards

Python

* Strong typing where practical
* Clear function names
* Small functions
* Minimal nesting
* Meaningful docstrings for public APIs

TypeScript

* Strict mode
* Typed props
* Typed API responses
* No implicit any

General

* Avoid duplication
* Avoid premature abstraction
* Prefer composition over complexity
* Keep code readable

---

# Error Handling

Every user-facing error must include:

* Title
* Description
* Recommendation
* Recoverable state

Never expose raw stack traces in the interface.

Backend logs may contain technical details.

Frontend messages should remain understandable.

---

# Validation

Validate before executing.

Examples:

* Environment
* Dataset
* Configuration
* Model compatibility
* File uploads

Never begin expensive work with known invalid input.

---

# Logging

Use structured logging.

Every significant operation should emit:

* Timestamp
* Level
* Component
* Stage
* Message
* Metadata

Logging is a product feature.

Not debugging.

---

# Progress

Every long-running operation must expose:

* Stage
* Progress
* ETA
* Elapsed Time

Never leave the user waiting without feedback.

---

# Retry Strategy

Retry only transient failures.

Never retry deterministic failures.

Retries must:

* be bounded
* be logged
* expose retry count
* stop gracefully

---

# User Experience

Professional.

Minimal.

Calm.

Readable.

Every page supports:

* Loading
* Empty
* Success
* Error

Animations communicate state.

Never distract.

---

# Review Checklist

Before considering a task complete, verify:

✓ Builds successfully

✓ Architecture unchanged

✓ No unnecessary files

✓ Structured logging present

✓ Validation implemented

✓ Errors actionable

✓ UI follows specification

✓ Documentation updated if required

✓ Acceptance criteria satisfied

---

# Forbidden Practices

Do not:

* Redesign the architecture.
* Introduce databases.
* Add authentication.
* Add Redis.
* Add message queues.
* Add microservices.
* Add plugin systems.
* Add design patterns without necessity.
* Swallow exceptions.
* Ignore validation.
* Hardcode configuration.
* Add dependencies without clear benefit.
* Implement future milestones.

---

# Expected Response Format

After completing a task, provide:

## Summary

What was implemented.

## Files

Created

Modified

Deleted

## Design Decisions

Brief explanation.

## Testing

How the task was verified.

## Remaining Work

Next task according to TASKS.md.

Do not proceed automatically.

---

# Definition of Engineering Excellence

Forge is successful when:

* The repository is understandable.
* The architecture remains simple.
* The code is easy to review.
* The UI feels professional.
* The pipeline is observable.
* Errors are actionable.
* The project is deployable.

Not when it contains the most features.

---

# Guiding Principle

Every line of code should make a reviewer more confident that the engineer who wrote it understands how to build reliable AI software.

If a decision improves clarity, reliability, maintainability, or user experience, it is probably the correct decision.

If a decision only increases complexity without measurable benefit, it should be rejected.

---

# Final Instruction

You are not asked to invent Forge.

Forge already exists in these documents.

Your responsibility is to implement it faithfully, one milestone at a time, producing clean, production-inspired software that another engineer would be comfortable maintaining.

Do not rush.

Do not overengineer.

Finish one thing well before beginning the next.
