# BAHA: Investment Thesis

> **Confidential Investment Memorandum**  
> *Prepared for prospective investors*  
> *Last Updated: January 2026*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem](#2-the-problem)
3. [Our Solution](#3-our-solution)
4. [Competitive Moat](#4-competitive-moat)
5. [Market Opportunity](#5-market-opportunity)
6. [Business Model](#6-business-model)
7. [Traction & Validation](#7-traction--validation)
8. [Financial Projections](#8-financial-projections)
9. [Use of Funds](#9-use-of-funds)
10. [Team](#10-team)
11. [Risk Factors](#11-risk-factors)
12. [Exit Landscape](#12-exit-landscape)
13. [The Ask](#13-the-ask)

---

## 1. Executive Summary

**BAHA** (Branch-Aware Heuristic Annealer) is an open-core optimization platform that solves hard combinatorial problems faster than traditional approaches by detecting "fractures" in the solution landscape and jumping directly to optimal branches.

**Investment Highlights:**

| Metric | Value |
|--------|-------|
| **Stage** | Seed / Pre-Series A |
| **Raise** | $2M – $3M |
| **Use** | Sales hire, BaaS infrastructure, community growth |
| **Target ARR (2027)** | $3M – $5M |
| **Gross Margin** | 70%+ |
| **Primary Moat** | Hardness observability + open-source data flywheel |

**Why Now:** Enterprises are hitting walls with legacy solvers (Gurobi, CPLEX) on emerging problem classes—ML hyperparameter search, logistics at scale, biotech assay design. BAHA's real-time hardness metrics let customers know *why* problems are hard and *when* to expect solutions.

---

## 2. The Problem

### The $64B Optimization Market Has a Visibility Problem

Enterprises spend billions on optimization software, yet:

- **Black-box solvers** provide no insight into *why* a problem is taking hours
- **Trial-and-error tuning** wastes engineering cycles
- **No standardized hardness metrics** exist—customers can't predict costs or timelines

> *"We threw 100 GPU-hours at a routing problem and it still didn't converge. We had no idea if we were 10% done or 90% done."*  
> — Logistics engineering lead, Fortune 500 retailer

### The Gap

| What Exists | What's Missing |
|-------------|----------------|
| Powerful solvers (Gurobi, OR-Tools) | Real-time hardness observability |
| Academic benchmarks | Production-grade "difficulty APIs" |
| Expensive consulting | Self-serve optimization intelligence |

---

## 3. Our Solution

### BAHA: The Fracture Hunter

BAHA introduces **hardness observability** to optimization:

```
┌─────────────────────────────────────────────────────────┐
│  Traditional Solver          │  BAHA                    │
├─────────────────────────────────────────────────────────│
│  "Running... please wait"    │  "β=2.3, ρ=4.7, FRACTURE │
│                              │   detected—jumping to    │
│                              │   branch k=0"            │
│  [no visibility]             │  [real-time metrics]     │
└─────────────────────────────────────────────────────────┘
```

**Core Innovation:**
- **Fracture Detection:** Monitors the derivative ρ = |d/dβ log Z| to detect phase transitions
- **Lambert-W Branch Enumeration:** Mathematically identifies all solution branches at a fracture point
- **Intelligent Jumping:** Skips slow annealing by jumping directly to the most promising branch

**Delivery Model:**
1. **Open-Core Library** (Apache 2.0) — free, drives adoption
2. **BaaS API** — managed, metered, SLA-backed
3. **Enterprise Add-ons** — priority support, custom tuning, on-prem deployment

---

## 4. Competitive Moat

### Our Moat Is Singular and Defensible

> **Primary Moat: Hardness Observability + Data Flywheel**

We are **not** claiming patents. Our defensibility comes from:

| Layer | What It Is | Why It's Defensible |
|-------|-----------|---------------------|
| **Hardness Metrics** | Real-time ρ (fracture rate), β-at-solution, branch counts | First-mover advantage; becomes the "industry standard" metric |
| **Open-Source Adoption** | Apache 2.0 core drives contributors and users | Community builds integrations we couldn't afford to build |
| **Benchmark Leadership** | Mega-Landscape, MILP, Ramsey test suites | Establishes BAHA as the reference implementation |
| **Data Flywheel** | Every API call improves our understanding of problem hardness | Proprietary corpus of hardness signatures across domains |
| **Service Expertise** | Consulting + training layer on top of the free core | High-margin revenue that competitors can't replicate without the community |

### What We Are NOT Claiming

- ❌ No patents filed (open-source philosophy)
- ❌ No proprietary algorithms locked away
- ❌ No vendor lock-in tactics

This is intentional. The open-core model builds trust and adoption; the service layer captures value.

---

## 5. Market Opportunity

### TAM → SAM → SOM (Conservative Estimates)

| Market Layer | Size | Rationale |
|--------------|------|-----------|
| **TAM** | $64B | Global optimization & decision-intelligence (Gartner 2024) |
| **SAM** | $6B | "High-hardness" segment: NP-hard problems in logistics, biotech, finance, ML |
| **Near-term SOM (2028)** | **$150M – $250M** | Three verticals with proven demand |

### Target Verticals (Ranked by Readiness)

| Vertical | Problem Type | Buyer | Why Now |
|----------|-------------|-------|---------|
| **1. Logistics** | Vehicle routing, warehouse layout | Ops Engineering | Rising fuel costs, same-day delivery pressure |
| **2. Biotech** | Assay design, DNA barcode optimization | R&D Leads | Combinatorial explosion in CRISPR/sequencing |
| **3. Cloud AI** | Hyperparameter search, model selection | ML Platform Teams | AutoML demand; GPU costs need optimization |

> **We are not boiling the ocean.** We focus on 3 verticals where hardness observability is acutely needed.

---

## 6. Business Model

### Revenue Streams (Ordered by Margin)

| Stream | % of Revenue (2027E) | Gross Margin | Notes |
|--------|---------------------|--------------|-------|
| **BaaS API** (usage-based) | 60% | 75% | $0.001–$0.01 per optimization call |
| **Enterprise Licenses** | 25% | 80% | Annual contracts, on-prem option |
| **Training & Consulting** | 15% | 65% | Masterclass, custom workshops |

### Pricing Philosophy

- **Free tier:** 10k API calls/month (drives adoption)
- **Pro tier:** $99/month + usage (self-serve scaling)
- **Enterprise:** Custom ACV ($50k–$200k/year)

### Unit Economics (Target)

| Metric | Value |
|--------|-------|
| **CAC** | ~$2,500 (sales-assisted) |
| **ACV** | $30,000 (blended) |
| **LTV** | $90,000 (3-year retention) |
| **LTV:CAC** | 36:1 |

---

## 7. Traction & Validation

### What We Have Today

| Metric | Status |
|--------|--------|
| **Open-source stars** | 150+ (growing 20%/month) |
| **Contributors** | 30 active |
| **Paid Pilots** | 2 ($15k each) |
| **BaaS Beta Users** | 45 (waitlist: 200+) |
| **Benchmark Suite** | 26+ test cases (Mega-Landscape, MILP, Ramsey, DNA) |

### Pilot Customers

| Customer | Vertical | Contract | Outcome |
|----------|----------|----------|---------|
| **[Biotech Startup]** | Assay Design | $15k pilot | 40% faster convergence vs. baseline |
| **[Telecom Provider]** | Network Planning | $15k pilot | Real-time hardness dashboards adopted |

### What This Proves

- **Technical validation:** BAHA works on real problems
- **Willingness to pay:** Customers will pay for hardness observability
- **Repeatable sales motion:** Pilot → POC → Contract

---

## 8. Financial Projections

### Revenue Trajectory (Tied to Concrete Levers)

| FY | ARR | Key Drivers |
|----|-----|-------------|
| **2025** | $0.3M | 2 pilots ($30k) + BaaS beta usage ($270k) |
| **2026** | $1.5M | • 5 enterprise pilots → 3 convert ($50k ACV) = $150k<br>• BaaS self-serve: 500 paying users × $200/mo × 12 = $1.2M<br>• Consulting: 2 workshops = $150k |
| **2027** | $3.5M | • 10 enterprise contracts ($100k ACV) = $1M<br>• BaaS: 2,000 users × $100/mo × 12 = $2.4M<br>• Consulting/Training: $100k |

> **Note:** We project 100–130% YoY growth, not 300%. Executable beats aspirational.

### Gross Margin Trajectory

| Year | Gross Margin | Driver |
|------|--------------|--------|
| 2025 | 55% | High pilot delivery costs |
| 2026 | 65% | BaaS automation kicks in |
| 2027 | 72% | Scale + self-serve dominates |

---

## 9. Use of Funds

### Raising: $2M – $3M (Seed / Pre-Series A)

| Allocation | Amount | Purpose |
|------------|--------|---------|
| **Sales & GTM** | $800k | Hire 2 SDRs + 1 enterprise AE |
| **Engineering** | $600k | 2 senior engineers (GPU, distributed systems) |
| **Infrastructure** | $300k | BaaS auto-scaling on AWS/GCP |
| **Community** | $150k | Hackathons, conference sponsorships, docs |
| **Legal & Ops** | $150k | Trademark, incorporation, compliance |

### Runway & Burn

| Metric | Current | Post-Raise |
|--------|---------|------------|
| **Monthly Burn** | $80k (R&D only) | $180k (adds GTM) |
| **Runway** | 8 months | 18 months |

> **Transparency:** Current burn is lean (founder + contractors). Post-raise burn reflects intentional GTM ramp.

---

## 10. Team

### Founding Team

| Role | Name | Background |
|------|------|------------|
| **CEO / Chief Scientist** | Seth Iyer | Researcher (optimization, physics-based methods); author of BAHA core algorithms |
| **CTO** (joining Q2 2026) | [To be announced] | Ex-[Cloud Provider], distributed systems |

### Advisors

- **[Academic Advisor]** — Professor, Operations Research
- **[Industry Advisor]** — Former VP Engineering, [Logistics Company]

### Hiring Plan (Post-Raise)

- Q2 2026: 2 SDRs, 1 Enterprise AE
- Q3 2026: 2 Senior Engineers (GPU, backend)
- Q4 2026: 1 Developer Advocate

---

## 11. Risk Factors

### We Are Honest About Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **Enterprise sales cycles are long** | High | Start with pilot-first motion; convert to annual contracts |
| **Incumbents (Gurobi, CPLEX) could add hardness metrics** | Medium | Our open-source community is a moat they can't easily replicate |
| **GPU costs could spike** | Medium | BaaS pricing includes usage-based pass-through; margins protected |
| **Key-person risk (founder)** | Medium | CTO joining; knowledge documented in open-source + courses |
| **Open-source doesn't convert to paid** | Low | 30%+ conversion in beta; service layer is differentiated |

---

## 12. Exit Landscape

### Acquisition-Focused (Not IPO Near-Term)

| Acquirer Type | Examples | Strategic Rationale |
|---------------|----------|---------------------|
| **Hyperscalers** | AWS, Google Cloud, Azure | "Optimization-as-a-Service" to compete with Vertex AI, SageMaker |
| **Hardware Vendors** | NVIDIA, AMD, Intel | FPGA/GPU co-design for hardness-aware accelerators |
| **Solver Incumbents** | Gurobi, FICO, IBM | Modernize with open-source community + hardness observability |
| **Logistics Platforms** | Flexport, Project44 | Vertical integration of routing optimization |

### Comparable Exits

| Company | Acquirer | Deal Size | Multiple |
|---------|----------|-----------|----------|
| Kiva Systems | Amazon | $775M | ~15× revenue |
| Optimal+ | National Instruments | $365M | ~8× revenue |
| DecisionBrain | (PE-backed growth) | — | 10× ARR |

> **Our target:** 6–10× ARR acquisition by Year 4–5, implying $20M–$50M exit on $3M–$5M ARR.

---

## 13. The Ask

### We Are Raising $2M – $3M

| Term | Proposed |
|------|----------|
| **Instrument** | SAFE (Post-money) or Priced Seed |
| **Valuation Cap** | $12M – $15M |
| **Target Close** | Q2 2026 |
| **Lead Investor** | Seeking one lead ($1M+) |
| **Use of Funds** | 60% Sales/Eng, 25% Infra, 15% Community/Legal |

### What We Offer Investors

- **Board seat** for lead investor
- **Quarterly updates** with audited financials
- **Direct access** to pilots and customer calls
- **Co-marketing** opportunities at conferences

### Why Now

1. **Pilots are converting** — we have paying customers
2. **Community momentum** — open-source adoption is accelerating
3. **Market timing** — enterprises are frustrated with legacy solvers
4. **Team ready to scale** — CTO joining, hiring pipeline in place

---

## Appendix: Supplementary Materials

Available upon request:
- BAHA technical whitepaper
- Pilot customer case studies
- Benchmark performance data
- Financial model (detailed)
- Cap table

---

> *"BAHA doesn't just solve problems—it tells you why they're hard."*

**Contact:**  
Seth Iyer, Founder & CEO  
[email] · [LinkedIn] · [GitHub: sethuiyer/baha]

---

*This document is confidential and intended solely for prospective investors. It contains forward-looking statements that involve risks and uncertainties.*
