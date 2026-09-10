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

## Citation

This project has three distinct things you might want to attribute:
the scientific paper, this software/repository, and the license terms that
govern reusing the code. They are not the same, and citing one does not
substitute for citing another.

**1. Scientific paper.** If you use the mathematical results, empirical
findings, methodology, interpretation, or figures, please cite the paper.
The Zenodo DOI is the preferred citation for the scientific results of
this project:

> De Jesús, E. (2026). *Exact Local Output Perturbation in KV-Cache
> Eviction: Ranking Inversions and Their Downstream Consequences*
> (Version v1) [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.22695261

**2. Software/repository.** If you use or substantially adapt the
implementation, experiment scripts, statistical-analysis code, or
figure-generation code, please cite the software using the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button
reads this file automatically), e.g.:

> De Jesús, E. (2026). *token-kv-efficiency* (Version 0.1.0) [Software].
> https://github.com/innerlightr-wq/token-kv-efficiency

**3. MIT License.** The [`LICENSE`](LICENSE) file governs reuse and
redistribution of the code — it is a legal permission, not a scholarly
citation. If you use the scientific results or figures, please cite the
Zenodo paper. If you use or adapt the implementation or experimental code,
please also cite the software repository. Code reuse is governed by the
MIT License. Researchers who rely materially on both the scientific
results and the software implementation are encouraged to cite both the
paper and the repository; this is a citation norm, not a licensing
requirement — the MIT License does not itself mandate citation.

## License

The software in this repository is released under the MIT License. See
[`LICENSE`](LICENSE).
