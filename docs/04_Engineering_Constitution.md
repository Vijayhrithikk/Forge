# Forge — Engineering Constitution v1.0

> **Build software that engineers trust.**

---

# Purpose

This document defines the engineering principles that govern every architectural, implementation, and product decision within Forge.

It exists to ensure that every contribution follows a consistent philosophy regardless of who writes the code.

When a decision is unclear, this document takes precedence over personal preference.

---

# Mission

Forge is not built to demonstrate how many technologies we know.

Forge is built to demonstrate how reliable, understandable, and production-inspired AI software should be engineered.

We optimize for engineering quality over engineering quantity.

---

# Core Values

## Reliability First

A system that works consistently is always more valuable than a system with more features.

Every implementation must prioritize correctness, stability, and predictability.

---

## Simplicity Wins

Every additional abstraction increases maintenance cost.

If a simpler solution solves the problem equally well, choose the simpler solution.

Complexity must always justify its existence.

---

## Deterministic Systems

The application should behave predictably.

Given the same configuration and dataset, users should understand exactly what the pipeline will do.

Avoid hidden behavior.

Avoid unexpected side effects.

---

## Observability by Default

Nothing should happen silently.

Every significant operation must expose:

* Current state
* Progress
* Duration
* Logs
* Metrics
* Result

Users should never wonder what the application is doing.

---

## Professional User Experience

Professional software reduces uncertainty.

Loading states, progress indicators, validation messages, and recovery guidance are part of the product—not optional enhancements.

---

# Product Philosophy

Forge solves one problem exceptionally well.

It intentionally avoids becoming a general AI platform.

Every feature must answer one question:

> Does this improve the Version 1 user experience and increase recruiter value?

If the answer is no, it belongs in a future version.

---

# Scope Discipline

Version 1 intentionally excludes:

* Authentication
* Databases
* OCR
* PDF ingestion
* DOCX ingestion
* CSV ingestion
* RAG
* Agents
* Vector databases
* Distributed training
* Multi-user support
* Billing
* Cloud orchestration

These exclusions are deliberate.

Removing scope is considered progress.

---

# Architecture Principles

## Single Responsibility

Every module should have one clear responsibility.

Every file should answer:

"What responsibility disappears if this file is removed?"

If the answer is "nothing," the file should not exist.

---

## Explicit Boundaries

API endpoints expose interfaces.

Services coordinate workflows.

Engines perform domain work.

Utilities contain reusable pure helpers.

Responsibilities must never overlap.

---

## Stable Architecture

Architecture changes are expensive.

Do not introduce additional services, folders, or abstractions without a documented reason.

Architecture should evolve only when existing structure demonstrably limits progress.

---

# Code Quality

Code should optimize for readability before cleverness.

Future contributors should understand code without extensive explanation.

Small functions.

Clear names.

Minimal nesting.

Meaningful comments only when necessary.

---

# Error Handling

Errors are part of the product.

Every error presented to a user must include:

* Title
* Description
* Root cause
* Recommendation
* Recoverability

Raw exceptions should never be shown directly in the user interface.

Errors should guide users toward recovery whenever possible.

---

# Validation

Validation occurs before expensive operations.

Examples include:

* Dataset validation
* Configuration validation
* File validation
* Environment validation

Reject invalid input early.

Never begin training with known invalid state.

---

# Retry Policy

Retries exist only for transient failures.

Suitable examples:

* Temporary network failures
* Model downloads
* File write interruptions

Do not automatically retry deterministic failures such as:

* Invalid datasets
* Invalid configuration
* Unsupported models
* Resource limitations

Retries must always be bounded.

Infinite retries are prohibited.

---

# Logging Standards

Logging is an engineering feature.

Every significant operation should emit structured logs containing:

* Timestamp
* Level
* Component
* Stage
* Message
* Metadata

Logging should explain what happened, not merely that something happened.

---

# Progress Reporting

Long-running operations must communicate progress continuously.

Every stage should expose:

* Current stage
* Percentage complete
* Current step
* Total steps
* Elapsed time
* Estimated remaining time

Infinite loading indicators are prohibited.

---

# Configuration

Configuration belongs in configuration files or environment variables.

Hardcoded values should be avoided unless they are true constants.

Behavior should be configurable without modifying source code.

---

# User Interface Philosophy

The interface should feel calm, focused, and professional.

Visual hierarchy should prioritize information rather than decoration.

Animations should communicate state transitions, progress, and completion.

Every page must include meaningful:

* Loading states
* Empty states
* Success states
* Error states

---

# Security

Never trust client input.

Validate every request.

Sanitize uploaded files.

Fail safely.

Protect filesystem operations.

Avoid exposing internal implementation details.

---

# Testing Philosophy

Every completed milestone should pass:

* Manual workflow verification
* Smoke testing
* Error scenario testing

A feature is not complete simply because it compiles.

---

# Documentation

Documentation is treated as part of the product.

Architecture, setup, and workflows should remain synchronized with implementation.

Outdated documentation is considered a defect.

---

# Performance

Optimize only after correctness.

Avoid premature optimization.

Measure before optimizing.

Simple and correct systems outperform complicated, fragile systems.

---

# Anti-Patterns

The following are prohibited unless explicitly justified.

Silently swallowing exceptions.

```python
except:
    pass
```

---

Generic debugging output.

```python
print(...)
```

Use structured logging instead.

---

Functions that perform multiple unrelated responsibilities.

---

Hidden global state.

---

Magic numbers without explanation.

---

Duplicate business logic.

---

Unnecessary abstractions introduced "for future scalability."

---

Architecture changes without documented reasoning.

---

Infinite retry loops.

---

User-facing error messages without recovery guidance.

---

# Engineering Oath

As contributors to Forge, we commit to the following principles.

We choose clarity over cleverness.

We choose reliability over feature count.

We validate before executing.

We make system behavior observable.

We fail loudly instead of silently.

We recover gracefully whenever possible.

We keep architecture simple.

We document important decisions.

We respect the reader of our code.

We optimize for maintainability.

We build software we would be proud to explain in an engineering interview.

We finish what we start.

---

# Final Principle

Forge is not an experiment.

Forge is a demonstration of engineering maturity.

Every line of code should reinforce that belief.
