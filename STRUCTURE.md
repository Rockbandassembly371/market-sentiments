# AION Project Structure

This document defines the boundary between open-source and internal systems.

---

## Repository Boundary

###  OSS Layer (Public)

**Repository:** [`market-sentiments`](https://github.com/AION-Analytics/market-sentiments)

**Purpose:** Developer toolkit for financial sentiment modeling

**Audience:**
- ML engineers
- Quant researchers
- Indie developers
- Students

**Scope:**
-  Data abstractions (sentiment, sectors, VIX)
-  Feature engineering utilities
-  Model utilities (transformers, lexicons)
-  Clean Python APIs
-  Experimentation/prototyping

**Exclusions:**
-  No Redis streams
-  No ClickHouse pipelines
-  No live tick ingestion
-  No execution engines
-  No system governance
-  No agent hierarchies

**Messaging:**
> "Build your own sentiment models with reusable components"

---

###  Intraday System (Internal)

**Repository:** `aion-intraday` (restricted access)

**Purpose:** Production-grade trading system

**Audience:**
- Infrastructure engineers
- System architects
- Production quants

**Scope:**
-  Live ingestion pipelines
-  Rust compute layer
-  Redis + ClickHouse infrastructure
-  Signal generation + execution
-  System governance + agents
-  Audit + observability

**Exclusions:**
-  No beginner-friendly abstractions
-  No library-style APIs
-  No simplified onboarding

**Messaging:**
> "Production trading system with low-latency compute and reliability"

---

## Linking Strategy

### One-Directional Authority Flow

**OSS → Intraday (Allowed):**
```markdown
These components are used in AION's production intraday system.
```

**Intraday → OSS (Allowed):**
```markdown
Some modules are open-sourced under market-sentiments.
```

**Bidirectional Blending (NOT Allowed):**
-  OSS README should NOT detail production infra
-  Intraday README should NOT position as reusable toolkit

---

## Failure Modes to Avoid

### 1. Feature Leakage
Putting advanced infra concepts (Redis streams, ClickHouse pipelines) into OSS → **overwhelms users**

### 2. Simplification Leakage
Dumbing down intraday system → **destroys credibility**

### 3. Identity Confusion
Users not knowing:
- "Is this a tool?" → **OSS**
- "Is this a system?" → **Intraday**
- "Can I use this?" → **OSS = yes, Intraday = restricted**

---

## Strategic Rationale

| Aspect | OSS (`market-sentiments`) | Intraday (`aion-intraday`) |
|--------|--------------------------|---------------------------|
| **Goal** | Distribution, adoption, visibility | Credibility, depth, moat |
| **Complexity** | Low (clean APIs) | High (production infra) |
| **Audience** | Broad (developers, researchers) | Narrow (infra engineers, serious quants) |
| **Access** | Public | Restricted |
| **Purpose** | Enable experimentation | Run production trading |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-15 | Separate OSS and Intraday at repo level | Avoid identity confusion, serve different audiences |
| 2026-03-15 | OSS README explicitly states what it is NOT | Prevent misuse, set clear expectations |
| 2026-03-15 | One-directional linking (OSS → Intraday) | Maintain authority flow without confusing users |

---

## Questions & Answers

**Q: Can I use OSS for production trading?**  
A: No. OSS is for prototyping and research. Use internal systems for production.

**Q: Why separate the repos?**  
A: Different audiences, different purposes. Combining them creates confusion and reduces adoption.

**Q: Can I contribute to OSS?**  
A: Yes! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Q: How do I access the intraday system?**  
A: Internal access only. Contact aionlabs@tutamail.com for inquiries.

---

*Last Updated: March 15, 2026*  
*AION Open-Source Project*
