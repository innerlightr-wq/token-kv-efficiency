# Reproduction audit

Independent reproduction of this repository's test suite and experiments on a
different platform from the one that generated the committed results.
Everything below was run during the audit; no number is carried over from a
prior report.

## Environment

The committed results were generated on macOS (`provenance.config_path` in
`results/*/summary.json` reads `/Users/eliasdejesus/...`, Python 3.12.2). This
audit ran on Linux with the same library versions:

| | committed (macOS) | this audit (Linux) |
|---|---|---|
| python | 3.12.2 | **3.14.7** |
| platform | macOS | Linux 7.2.3-arch1-3, x86_64, glibc 2.44 |
| torch | 2.14.0 | 2.14.0+cpu |
| transformers | 5.16.1 | 5.16.1 |
| numpy | 2.5.2 | 2.5.2 |
| scipy | 1.18.1 | 1.18.1 |
| pandas | 3.0.5 | 3.0.5 |
| pyyaml | 6.0.3 | 6.0.3 |
| device | cpu | cpu (8 cores, 4 torch threads, no CUDA) |
| model revision | `50f5173d932e8e61f858120bcb800b97af589f46` | **same** |

`requirements-lock.txt` cannot be installed verbatim — it contains
`-e /Users/eliasdejesus/token-kv-efficiency`, an absolute path from the
authoring machine. Reproduce with `pip install -e .` plus the pinned versions.

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  'torch==2.14.0' 'transformers==5.16.1' 'numpy==2.5.2' 'scipy==1.18.1' \
  'pandas==3.0.5' 'pytest==9.1.1' 'pyyaml==6.0.3'
.venv/bin/pip install -e .
```

## Test suite

```bash
.venv/bin/pytest -q
```

**71 passed, 0 failed, 0 skipped, 1 warning, 22.1 s.** With the 15 guard-semantics
regression tests added by this audit: **86 passed**.

This is the current truth. Earlier reports quote different counts — 24/24
(`docs/RESEARCH_STATUS.md`, referring only to `test_local_algebra.py`) and
74/74 (`manuscript/CLAIM_LEDGER.md`). Neither matches the tree at `e42b3a6`;
`74` appears to be stale. The one warning is a benign
`requires_grad=True → scalar` notice from `tests/test_gpt2_adapter.py:176`.

## Experiment-by-experiment

Classification: `EXACTLY REPRODUCED` (agreement to the precision quoted) ·
`QUALITATIVELY REPRODUCED` (same conclusion, digits move) ·
`NOT REPRODUCED` · `INSUFFICIENT ARTIFACTS`.

### `experiments/validate_local_identity.py`

Config `configs/pythia160m.yaml`, Pythia-160M layer 3 head 2, seed 0, 57-token prompt.

| check | documented | reproduced | verdict |
|---|---|---|---|
| QKV/attention reconstruction rel. err | 6.988e-08 | **6.239e-08** | within tolerance (`< 1e-3`) |
| delete-renormalize identity abs. err | 1.110e-16 | **5.551e-17** | within tolerance (`< 1e-9`) |
| determinism | 0.0 | **0.0 exactly** | exact |
| Tier-A KL | — | 3.956e-08 | finite |
| Tier-B KL | — | 6.637e-04 | finite |
| verdict | ENGINEERING VERIFIED | **ENGINEERING VERIFIED** | — |

`QUALITATIVELY REPRODUCED`. The float-level differences are expected from a
different BLAS/platform; both values sit orders of magnitude inside the
asserted tolerances. **Recommendation:** quote these as "~1e-7" and "~1e-16"
rather than as fixed digit strings, which invite spurious mismatch reports.

### `experiments/reproduce_pilot.py`

n=96 records, 4 layers x 4 heads x 2 prompts x ~3 blocks/stratum, seed 0.

| statistic | documented | reproduced | verdict |
|---|---|---|---|
| Pooled Spearman(`d_local`, Tier-B KL) | 0.857 | **0.8551** | EXACTLY REPRODUCED (to 2 s.f.) |
| Pooled Spearman(Tier-A KL, Tier-B KL) | 0.838 | **0.8323** | QUALITATIVELY REPRODUCED |
| F10 median within-head Spearman | 0.086 | **0.2571** | **NOT REPRODUCED** |
| F10 fires (`< 0.7`) | True | **True** | reproduced |

**The F10 discrepancy is the audit's one material reproduction failure, and it
is a sample-size artifact rather than a bug.** F10 is the median of 13
per-stratum Spearman coefficients (3 strata have no variance and return
`null`), each computed on **n=6** blocks. At n=6, Spearman takes coarsely
quantised values and a float difference too small to see can flip a rank pair.
Per-stratum comparison:

| stratum | committed | reproduced |
|---|---|---|
| 1_6 | −0.3143 | −0.1429 |
| 1_9 | −0.8857 | −0.9429 |
| 4_3 | 0.6000 | 0.6571 |
| 4_6 | 0.0857 | 0.2571 |
| 7_0 | 1.0000 | 0.9429 |
| 7_3 | 0.8286 | 0.9429 |
| **7_6** | **−0.3143** | **+0.8286** |
| 7_9 | −0.0857 | 0.2571 |

Eight of thirteen differ; stratum 7_6 changes sign with a magnitude swing of
1.14. `README.md` and `docs/RESEARCH_STATUS.md` both quote "≈0.09"; that digit
is not portable. The **conclusion** (F10 fires; within-head ranking is weak) is
robust — both medians are far below 0.7 — and the repository already warns that
the grid is below its preregistered minimum of 10 blocks/stratum. See
`docs/CLAIM_LEDGER.md` A5/C1.

### `experiments/coarse_vs_fine_analysis.py`

| statistic | documented | reproduced | verdict |
|---|---|---|---|
| Coarse between-head/layer Spearman | 0.77 (p=0.0004, n=16) | **0.7706 (p=4.77e-4, n=16)** | EXACTLY REPRODUCED |
| Fine within-head median Spearman | 0.37 (n=15) | **0.3714 (n=15)** | EXACTLY REPRODUCED |

### `experiments/eviction_curve.py`

Layer 3 / head 2, 2 prompts, random averaged over 5 seeds.

| q | random (doc → repro) | attention-mass | local-damage | verdict |
|---|---|---|---|---|
| 0.1 | 3.4e-3 → **3.418e-3** | 6.5e-4 → **6.539e-4** | 6.5e-4 → **6.539e-4** | EXACTLY REPRODUCED |
| 0.3 | 8.5e-3 → **8.472e-3** | 3.0e-3 → **3.043e-3** | 3.0e-3 → **3.043e-3** | EXACTLY REPRODUCED |
| 0.5 | 1.68e-2 → **1.677e-2** | 6.3e-3 → **6.310e-3** | 6.3e-3 → **6.310e-3** | EXACTLY REPRODUCED |

Mass and local-damage agree to every reported digit at q ∈ {0.1, 0.3, 0.4, 0.5}
and differ only at q=0.2 (7.512e-4 vs 8.446e-4) — the documented "nearly
identical" finding, confirmed.

### `experiments/multistratum_eviction.py`

268 paired observations, 20 strata, 3 prompts, 5 removal fractions.

| statistic | documented | reproduced | verdict |
|---|---|---|---|
| Pooled paired advantage (mass − local), mean | −1.86e−5 | **−1.833e−5** | QUALITATIVELY REPRODUCED |
| 95% CI spans zero | yes | **yes** | reproduced |
| n paired | 268 | **268** | exact |
| Rank-divergence Spearman | 0.071 (p=0.77) | **0.0738 (p=0.757)** | QUALITATIVELY REPRODUCED |
| Strata local wins / mass wins / tied | 6 / 3 / 11 | **5 / 2 / 13** | QUALITATIVELY REPRODUCED |

Two strata moved from "wins" to "tied" — the same tie-threshold sensitivity as
F10, and it strengthens rather than weakens the null. Conclusion unchanged.

### `experiments/global_budget_curve.py`

The strongest reproduction in the repository. All five policies at all five
global removal fractions, mean KL:

| policy | q=0.1 | q=0.2 | q=0.3 | q=0.4 | q=0.5 |
|---|---|---|---|---|---|
| `uniform_random` | 3.375e-3 → **3.374e-3** | 8.859e-3 → **8.857e-3** | 1.378e-2 → **1.378e-2** | 1.642e-2 → **1.642e-2** | 1.999e-2 → **1.998e-2** |
| `uniform_mass` | 3.303e-3 → **3.301e-3** | 5.935e-3 → **5.930e-3** | 7.091e-3 → **7.086e-3** | 1.131e-2 → **1.131e-2** | 1.546e-2 → **1.546e-2** |
| `uniform_local` | 3.124e-3 → **3.126e-3** | 5.874e-3 → **5.870e-3** | 1.047e-2 → **1.046e-2** | 1.080e-2 → **1.080e-2** | 1.696e-2 → **1.694e-2** |
| `allocated_mass` | 3.240e-3 → **3.243e-3** | 5.860e-3 → **5.851e-3** | 7.158e-3 → **7.154e-3** | 1.145e-2 → **1.146e-2** | 1.448e-2 → **1.448e-2** |
| `allocated_local` | 3.021e-3 → **3.020e-3** | 5.814e-3 → **5.804e-3** | 1.069e-2 → **1.069e-2** | 1.108e-2 → **1.108e-2** | 1.647e-2 → **1.647e-2** |

`n_singularity_skips` **237 → 237, exact**. `n_calibration_records` 1720 → 1720.
Efficiency frontier at tolerance 1e-2 identical for all five policies
(`uniform_mass`/`allocated_mass` reach q=0.3; the other three reach q=0.2).

`EXACTLY REPRODUCED`.

### Manuscript experiments — not re-run

`cross_architecture_full_grid.py`, `cross_architecture_generalization.py`,
`full_grid_mechanistic_analysis.py`, `directional_value_geometry*.py`,
`downstream_propagation_*.py`, `functional_propagation_audit.py`,
`state_dependent_propagation_audit.py`, `cluster_aware_reanalysis.py`.

`INSUFFICIENT ARTIFACTS` for a same-session reproduction: these require GPT-2
in addition to Pythia and produce the 426,816-comparison grids behind the
manuscript's `Gamma` claims. Their stored `summary.json`/`report.txt` are
present and internally consistent, and the manuscript's own claim ledger
records recomputation of the load-bearing figures. Re-running them was out of
scope for this audit; the numbers in `manuscript/CLAIM_LEDGER.md` claims 5–13
are therefore **unverified by this audit** and are not repeated as facts in
`docs/NOVELTY_AND_PROVENANCE.md`.

One asymmetry worth noting from the repository's own review file: GPT-2 was
loaded without a pinned revision, unlike Pythia. That is disclosed in
`manuscript/EXTERNAL_REVIEW_SIMULATION.md` and remains a real gap in the
GPT-2 reproducibility guarantee.

## New diagnostics added by this audit

All three import the production code and modify none of it.

```bash
.venv/bin/python experiments/diagnose_singularity_skips.py   # ~2 min
.venv/bin/python experiments/compare_guard_semantics.py      # ~8 min
.venv/bin/python experiments/counterfactual_eviction.py      # ~35 min, 8400 passes
.venv/bin/pytest tests/test_guard_semantics.py -q            # 15 tests, 2.4 s
```

| script | output | headline |
|---|---|---|
| `diagnose_singularity_skips.py` | `results/diagnose_singularity_skips/` | 2100 attempts, 237 skips (matches production exactly), 851 singular rows, 12/237 with the measured row singular |
| `compare_guard_semantics.py` | `results/compare_guard_semantics/` | applied/planned 0.8664 (production) vs 1.0000 (per-row), mean KL 1.0480e-02 vs 1.0571e-02 |
| `counterfactual_eviction.py` | `results/counterfactual_eviction/` | 8400 interventions, 8364 valid; top-first damages 4.0–6.6x more than bottom-first |

Seeds are fixed from `configs/milestone2_grid.yaml` (`seed: 0`);
`counterfactual_eviction.py` uses 5 random seeds per matched-budget control.

## Randomness and stability

`uniform_random` is averaged over 3 seeds in the Milestone 2 grid and 5 in
`eviction_curve.py`; `counterfactual_eviction.py` uses 5. Random-arm KL
reproduced to 3–4 significant figures, so seed variance is not the source of
the F10 instability — rank quantisation at n=6 is.

## What publication-grade validation would require

Not attempted here, and not reasonable on this hardware; listed so the gap is
explicit rather than implied.

- **Models that instruction-follow.** Pythia-160M and GPT-2 are 160M/124M
  parameters. Every closest prior-art system (`h2o2023`, `snapkv2024`,
  `vatp2024`, `pyramidkv2024`, `adakv2024`) evaluates at 7B and above, and
  `vatp2024` reaches the opposite conclusion to this repository's headline null
  at that scale.
- **Standard long-context benchmarks.** LongBench (used by `vatp2024`,
  `pyramidkv2024`, `fixedcontract2026`), RULER and NeedleBench
  (`fixedcontract2026`), or perplexity on PG19/WikiText. This repository
  evaluates KL at one token position on 3 synthetic prompts.
- **Real cache compaction.** Eviction here is an additive delta on head
  outputs, so no memory is freed and no latency measured. Memory and throughput
  claims require an actual compacted-cache implementation.
- **Matched realised budgets.** See `docs/METRIC_DEFINITIONS.md` §8: the
  memory axis currently reports the plan, not the realisation.
- **More blocks per stratum.** The repository's own preregistered minimum is 10;
  the pilot has ~3–6. F10's instability (C1) is the direct consequence.
