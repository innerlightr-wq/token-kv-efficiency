# Research status

Central question: **can we remove, compress, or deprioritize KV-cache
content while keeping downstream model damage below a controlled level?**
This document separates what is established at each level from what is
open. Do not quote a number from this repository without checking its
label here.

## EXACT ALGEBRA

The local delete-and-renormalize identity (`src/kv_efficiency/local_algebra.py`):

```
Δ_B = -r_B / (1 - m_B),   d_B^local = ‖Δ_B‖ = ‖r_B‖ / (1 - m_B)
```

Verified by direct substitution against independently-recomputed
delete-and-renormalize output, to float64 tolerance, on synthetic data
including edge cases (tiny removed mass, multi-token blocks, near-zero
residual, moderate `m_B`, and the `m_B → 1` singularity, which correctly
**raises** rather than returning a silently huge or NaN value) —
`tests/test_local_algebra.py`, 24/24 passing. This is algebra, not an
empirical claim about any model.

## ENGINEERING VERIFIED

Re-measured in this repository's isolated environment (`experiments/validate_local_identity.py`,
`EleutherAI/pythia-160m`, layer 3, head 2), not assumed from the old
project's report:

| Check | Old report | Measured here |
|---|---|---|
| Manual QKV/attention reconstruction vs. real model | ~7e-8 | **6.988e-8** |
| Delete-and-renormalize identity vs. direct recomputation | ~1e-16 | **1.110e-16** |
| Determinism (repeated identical forward passes) | pass | **0.0 exactly** |

Old numbers reproduced to within measurement noise — not hard-coded as
pass/fail thresholds; the script asserts against fixed tolerances
(`rel_err_A < 1e-3`, `err_B < 1e-9`, `det_err < 1e-6`) and reports the
actual values regardless.

## PRELIMINARY EMPIRICAL RESULT

Old pilot, reproduced in the isolated environment (`experiments/reproduce_pilot.py`,
same model/prompts/layers/heads/block construction/seed as the original):

| Statistic | Old report | Reproduced here |
|---|---|---|
| Pooled Spearman(d_B^local, Tier-B KL) | ≈0.857 | **0.857** |
| Pooled Spearman(Tier-A KL, Tier-B KL) | ≈0.838 | **0.838** |
| F10: median within-head Spearman(Tier-A KL, Tier-B KL) | ≈0.086 | **0.086** |

n=96 records, 4 layers × 4 heads × 2 prompts × ~3 blocks/stratum. **F10
fires** (0.086 < 0.7 threshold). Per the milestone framing: this is
**preliminary real-model evidence against the current F10 within-head
ranking hypothesis**, not "falsified," not "disproved," not "useless
metric" — and it is far below the preregistered minimum of 10
blocks/stratum, so it is not confirmatory in either direction.

## EXPLORATORY RESULT

`experiments/coarse_vs_fine_analysis.py`, computed from the same pilot
records, explicitly *not* pooled with F10's own statistic:

- **Between-head/layer** (coarse): Spearman(mean `d_B^local`, mean Tier-B
  KL) across the 16 (layer, head) strata = **0.77** (p=0.0004, n=16).
- **Within-head** (fine): median Spearman(`d_B^local`, Tier-B KL) within
  each stratum = **0.37** (n=15 strata with variance; range -0.89 to 1.00).

Note this is a *different* pair of quantities than F10 (which compares
Tier-A KL to Tier-B KL, not `d_B^local` to Tier-B KL directly) — both are
reported, neither substitutes for the other. The open hypothesis this
supports: **the local score may carry real coarse layer/head signal even
where fine within-head block ranking is weak or noisy at this sample
size.** Exploratory; no new preregistration has been frozen.

## First efficiency-vs-damage result (also EXPLORATORY, n=2 prompts)

`experiments/eviction_curve.py`, layer 3 / head 2 (same as the engineering
gate, not cherry-picked from the pilot's better-correlated strata), removal
fractions 0-50% of candidate key positions, mean KL at the final position
across 2 prompts (random averaged over 5 seeds):

| q | random | attention-mass | local-damage |
|---|---|---|---|
| 0.1 | 3.4e-3 | 6.5e-4 | 6.5e-4 |
| 0.3 | 8.5e-3 | 3.0e-3 | 3.0e-3 |
| 0.5 | 1.68e-2 | 6.3e-3 | 6.3e-3 |

**Finding, stated plainly:** both attention-mass and local-damage eviction
clearly beat random eviction at every removal level (~2.5-3x lower KL at
q=0.5). **Finding that should not be oversold:** in this single-head test,
local-damage ranking is nearly identical to plain attention-mass ranking —
no measured advantage of the exact algebra over the much cheaper
attention-weight heuristic. Whether that changes with more heads, layers,
or prompts is open.

## Milestone 2: multi-stratum comparison and global budget allocation

Primary question (Phase 1, stated before running anything): does
local-damage eviction outperform attention-mass eviction where value
geometry carries information mass alone does not? The null outcome
(attention mass sufficient, local damage adds little) was explicitly
accepted as a valid answer going in.

**EXPLORATORY RESULT — multi-stratum local-vs-mass**
(`experiments/multistratum_eviction.py`, `results/milestone2_multistratum/`):
5 layers x 4 heads = 20 strata, 3 evaluation prompts, 5 removal fractions,
268 paired observations. Pooled paired advantage (KL_mass − KL_local):
mean **−1.86e−5**, bootstrap 95% CI **[−5.67e−5, +2.01e−5]** — spans zero.
Strata: **local wins 6, mass wins 3, tied 11**. Rank-divergence
(Spearman(1−agreement(m_B, d_local), advantage)) = **0.071** (p=0.77,
n=20) — no detected relationship between how much a stratum's mass/local
rankings disagree and how much local-damage eviction helps there, at this
sample size. Best strata for local (layer 6/head 6, layer 3/head 3, layer
3/head 0) and worst (layer 0/head 6, layer 0/head 3, layer 0/head 0) are
recorded in full in `results/milestone2_multistratum/summary.json` —
reported without cherry-picking only the winners.

**Conclusion for the primary hypothesis: the null outcome is what the data
supports.** Attention mass and local-damage are practically
indistinguishable in this grid, pooled. This is one of the two outcomes
the milestone brief said was acceptable, not a failed experiment.

**EXPLORATORY RESULT — global budget allocator**
(`experiments/global_budget_curve.py`, `results/milestone2_global_budget/`):
sensitivity estimated from 3 calibration prompts (disjoint from the 3
evaluation prompts used for every reported number here — Phase 8 split,
`configs/milestone2_grid.yaml`). All four structured policies
(uniform/allocated x mass/local) clearly beat uniform-random eviction at
every global removal fraction (q=0.5: random 2.00e−2 vs. structured
1.45–1.70e−2 mean KL). **Sensitivity-allocated budgets showed no
consistent advantage over uniform per-stratum budgets** at matched global
q — the most important question Phase 7 posed, answered null in this data.
Efficiency frontier at damage tolerance 0.01: `uniform_mass` /
`allocated_mass` reach q=0.3; `uniform_local` / `allocated_local` /
`uniform_random` reach q=0.2 — the one place mass-ranking measurably beat
local-damage in this milestone. Tighter tolerances (1e-4 to 3e-3) were not
reached by any policy at the smallest tested q=0.1 — the tolerance grid
does not resolve below q=0.1 for this model/prompt combination.

## OPEN

- Whether local-damage ranking improves real KV-eviction efficiency over
  attention-mass ranking, tested now at both single-head (Milestone 1) and
  20-stratum multi-head/layer scale (Milestone 2): **no measurable
  advantage found at either scale, on this one model.** Whether that
  changes on a larger/different model remains open.
- Whether sensitivity-allocated budgets beat uniform budgets: tested this
  round, **no consistent advantage found**. Open whether a different
  sensitivity estimator, more calibration prompts, or a different model
  changes this.
- Uniqueness/generalization of the coarse-vs-fine split beyond this one
  pilot's 16 strata.
- Generalization of every Milestone 1/2 result beyond Pythia-160M.

## METHOD OBSTRUCTION

None repository-blocking. One real, reported failure mode (Phase 12): of
~2100 stratum-eviction attempts in the global budget experiment, **237
(~11%) hit the exact `m_B → 1` singularity guard and were silently-to-the-
policy-but-not-to-the-report skipped** — driven by the allocator's
`max_frac=0.95` bound combined with short prompts and concentrated
(sink-like) attention in some heads. `global_eviction` surfaces every skip
in its return value rather than absorbing it; `n_singularity_skips` is
recorded in every `results/milestone2_global_budget/summary.json` run.
This is exactly the numerical edge the `local_algebra.py` `m_B → 1` test
predicted would need handling, now observed for real. (The old base-conda
OpenMP conflict remains diagnosed, not an obstruction — see below.)

## Environment note

The old experiment's `KMP_DUPLICATE_LIB_OK=TRUE` / `OMP_NUM_THREADS=1` /
`torch.set_num_threads(1)` workaround was **not needed** in this
repository's isolated venv (`python3 -m venv`, pip-only torch/transformers,
no conda). `import torch`, model load, and a full forward pass all ran
cleanly at the default 4 threads with no crash. This strongly suggests the
original conflict was specific to base conda's MKL-linked numpy/scipy
colliding with pip-installed torch's bundled OpenMP runtime in the same
process — not an inherent torch/transformers issue — diagnosed rather than
masked, per the milestone's explicit instruction.

## Do not claim

"Token efficiency improved" — not established as a general claim.
"Local-damage eviction beats attention mass" — not established; Milestone
2's larger grid did not find this, and the honest reading is that
attention mass (far cheaper to compute) is the stronger practical
baseline in what's been tested so far. "Lean/Pythia verifies KV-cache
safety" is not a phrase used or implied anywhere in this project. Damage
tolerances (ε in the efficiency frontier) are exactly that — measured KL
budgets — never described as "safe."
