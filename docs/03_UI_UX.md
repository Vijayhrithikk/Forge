# Forge — UI/UX Specification v1.0

## Design Philosophy

Forge should feel like a professional engineering tool rather than an academic dashboard.

Inspired by:

- Linear
- Vercel
- Cursor
- Weights & Biases

Core principles:

- Minimal
- Spacious
- Fast
- Informative
- Calm
- No visual clutter

Animations should communicate progress, not decorate the interface.

---

## Navigation

Top Navigation on landing page. Sidebar in Studio.

### Landing Nav

Forge | Documentation | GitHub | Launch Studio

### Studio Sidebar

Dashboard | Projects | Training | Playground | Artifacts | Logs | Settings

---

## Pages

### Landing Page

- **Hero**: Headline + subheadline + CTA buttons (Launch Studio, View GitHub)
- **Pipeline Animation**: Dataset → Validation → Tokenization → LoRA → Training → Evaluation → Adapter (sequential highlight animation)
- **Features**: 6 cards — Dataset Validation, Reliable Training, Live Observability, Evaluation, Artifact Export, Model Playground
- **Engineering Principles**: Reliable, Observable, Deterministic, Deployable, Simple, Production Inspired
- **Footer**: GitHub, Documentation, License, Version

### Studio Dashboard

- Stats cards: Current Project, Model, Status, GPU
- Recent Activity feed
- Quick Actions links
- Backend connectivity indicator

### Upload Experience

- Large drag-and-drop area
- Browse button, cancel, retry
- Upload progress with speed indicator
- Animated pipeline: Upload → Encoding → JSONL → Schema → Validation → Statistics → Quality → Ready

### Dataset Dashboard

- Project name, dataset name, status, file size, sample count, quality score
- Quality card with large score (color-coded: Excellent/Good/Needs Improvement/Poor)
- Statistics section: samples, avg/median/max/min tokens, duplicates, estimated training time
- Sequence distribution histogram
- Tokenizer preview (select sample → show tokens)
- Dataset inspector (browse samples with navigation)
- Warnings panel with actionable recommendations

---

## Theme

- **Default**: Dark
- **Optional**: Light
- **Typography**: Geist (sans + mono)
- **Icons**: Lucide
- **Accent**: Professional blue (#3b82f6)
- **Corners**: Large (0.75rem)
- **Spacing**: Generous

---

## Motion System

**Allowed**: Fade, Slide, Scale, Progress Fill, Skeleton Loading, Stage Transitions

**Forbidden**: Bounce, Flip, Spin Forever, Elastic, Decorative animations

---

## States

Every page must support:

- Loading (skeleton placeholders, never spinner-only)
- Empty (meaningful message + suggested action)
- Success (confirmation where appropriate)
- Error (title, description, recommendation, retry if applicable)

---

## Responsiveness

- Desktop: primary target
- Tablet: fully supported
- Mobile: usable, no horizontal scroll

---

## Accessibility

- Semantic HTML
- Keyboard navigation
- Visible focus states
- ARIA where appropriate
- Good contrast ratios
- Readable typography

---

## Recruiter Journey

1. Open landing page → understand Forge in 10 seconds
2. Click Launch Studio → upload a dataset
3. Watch a beautiful, transparent validation pipeline
4. See quality report and statistics
5. Leave thinking: "This feels like a real ML engineering product."
