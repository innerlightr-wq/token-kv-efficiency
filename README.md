# token-kv-efficiency

**Adaptive KV-cache efficiency under bounded downstream damage.**

The question this project asks: can exact local attention-deletion algebra
guide practical KV-cache eviction — removing or deprioritizing cache
content while keeping downstream model damage below a controlled level?

```
Local algebra
    ↓
engineering validation
    ↓
downstream-damage prediction
    ↓
KV eviction policy
    ↓
efficiency-vs-damage curve
```

Each level is a distinct epistemic claim and this repository does not
conflate them — see `docs/RESEARCH_STATUS.md` for the full breakdown
(EXACT ALGEBRA / ENGINEERING VERIFIED / PRELIMINARY EMPIRICAL RESULT /
EXPLORATORY RESULT / OPEN).

**Known limitation, stated up front:** a strong *pooled* association
between the local damage score and downstream KL divergence does not imply
strong *within-head* block ranking. The reproduced pilot shows pooled
Spearman ≈0.86 alongside a preregistered within-head statistic of ≈0.09 —
see `docs/RESEARCH_STATUS.md` for what that does and does not mean.

## Provenance

This repository migrates the mathematically load-bearing parts of an
earlier experiment at `~/Desktop/Token Research/files (29)/` (untouched;
see `docs/MIGRATION_NOTES.md` for the exact source → destination mapping).
`~/Desktop/Token Research/files (29) 2/` is separate protected reference
material and was never read for writing here.

## Layout

```
src/kv_efficiency/
    local_algebra.py            -- EXACT ALGEBRA, numpy, model-independent
    attention_reconstruction.py -- GPT-NeoX (Pythia) adapter
    eviction.py                 -- Tier A / Tier B / global joint eviction (torch)
    policies.py                 -- eviction ranking policies
    allocator.py                -- Milestone 2: head/layer budget allocator
    damage_metrics.py           -- KL / logit-L2 / hidden-state-L2
    hooks.py                    -- shared forward-hook utilities
    provenance.py               -- config/env snapshot for every result file
experiments/
    validate_local_identity.py  -- Milestone 1: engineering gate, real model
    reproduce_pilot.py          -- Milestone 1: pilot reproduction (provenance)
    coarse_vs_fine_analysis.py  -- Milestone 1: between- vs within-head signal
    eviction_curve.py           -- Milestone 1: single-head removal fraction vs. damage
    multistratum_eviction.py    -- Milestone 2: expanded grid, local vs. mass
    global_budget_curve.py      -- Milestone 2: budget allocator + global curve
tests/                          -- mostly synthetic; a few small real-model checks
configs/{pythia160m,milestone2_grid}.yaml
results/                        -- machine-readable outputs (gitignored)
docs/
    RESEARCH_STATUS.md           -- read this before quoting any number
    MIGRATION_NOTES.md
    EXPERIMENT_PROTOCOL.md       -- audited definitions (m_B, r_B, Tier A/B, F10)
```

## Environment

Isolated `venv`, **not** base conda (the old experiment's
`KMP_DUPLICATE_LIB_OK`/single-thread workaround is not used or needed
here — see `docs/RESEARCH_STATUS.md`):

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pytest
.venv/bin/python experiments/validate_local_identity.py
.venv/bin/python experiments/reproduce_pilot.py
.venv/bin/python experiments/coarse_vs_fine_analysis.py
.venv/bin/python experiments/eviction_curve.py
```

Exact installed versions: `requirements-lock.txt`.

## Status

See `docs/RESEARCH_STATUS.md` and `MILESTONE_STATUS.md`. Headline so far,
measured on **`EleutherAI/pythia-160m`, the layer/head/prompt/context
lengths actually tested in this repo's configs** (not claimed to
generalize beyond them), now checked at both single-head (Milestone 1)
and 20-stratum multi-head/layer scale (Milestone 2): local-damage and
attention-mass eviction both clearly beat random, but show **no measured
advantage of one over the other** — the honest reading is that attention
mass, far cheaper to compute, is currently the stronger practical
baseline. Sensitivity-based budget allocation across heads/layers likewise
showed no consistent advantage over a uniform budget. Both are real,
checked null results, not failures to find something that was assumed to
be there. No wall-clock speedup is measured or claimed, and no claim is
made that `d_B^local` predicts downstream damage universally.

```bash
# Milestone 2
.venv/bin/python experiments/multistratum_eviction.py
.venv/bin/python experiments/global_budget_curve.py
```
