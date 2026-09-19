# Audit claim ledger

Every substantive claim this repository makes or that this audit adds, with
its basis, reproduction status, closest prior art, and recommended wording.
Complements `manuscript/CLAIM_LEDGER.md` (which covers the manuscript's own
claims); this file covers the repository's claims and the audit's findings,
and records where the two disagree.

Status vocabulary: `FORMALLY/ANALYTICALLY DERIVED` · `REPRODUCED COMPUTATION` ·
`EMPIRICAL OBSERVATION` · `INTERPRETATION` · `HYPOTHESIS` · `OPEN`

---

## A. Repository claims

| # | Claim | Repo location | Basis | Reproduced? | Closest prior art | Status | Scope | Confidence | Recommended wording |
|---|---|---|---|---|---|---|---|---|---|
| A1 | `Delta_B = -r_B/(1-m_B)` exact | `local_algebra.py` | Algebraic substitution | **Yes** — 5.55e-17 vs direct recomputation | `keepkv2025` (identical, singleton) | FORMALLY DERIVED | Any softmax attention row | High | Keep. Already disclosed as not novel — do not weaken. |
| A2 | Manual QKV reconstruction matches the live model | `validate_local_identity.py` | Measurement | **Yes** — 6.24e-08 (doc: 6.99e-08) | N/A | REPRODUCED COMPUTATION | Pythia-160M L3/H2 | High | Quote as "~1e-7", not a fixed digit string. |
| A3 | Determinism of repeated forward passes | same | Measurement | **Yes** — 0.0 exactly | N/A | REPRODUCED COMPUTATION | CPU, float32 | High | Keep. |
| A4 | Pooled Spearman(`d_local`, Tier-B KL) ≈ 0.86 | `docs/RESEARCH_STATUS.md` | n=96 records | **Yes** — 0.8551 (doc: 0.8571) | N/A | REPRODUCED COMPUTATION | 16 strata, 2 prompts | High | Keep "≈0.86". |
| A5 | F10 median within-head Spearman ≈ 0.086 | `docs/RESEARCH_STATUS.md`, `README.md` | Median of 13 rho, n=6 each | **NO** — reproduced **0.2571** | N/A | EMPIRICAL OBSERVATION | Same grid | **Low** | **Change.** Do not quote 0.086. Say the median is below 0.3 and platform-unstable at n=6/stratum; the qualitative conclusion (F10 fires, < 0.7) is robust. See §C1. |
| A6 | Coarse between-head/layer Spearman 0.77 | `docs/RESEARCH_STATUS.md` | n=16 strata | **Yes** — 0.7706, p=4.8e-4 | N/A | REPRODUCED COMPUTATION | 16 strata | High | Keep. |
| A7 | Fine within-head median Spearman 0.37 | same | n=15 strata | **Yes** — 0.3714 | N/A | REPRODUCED COMPUTATION | 16 strata | Medium | Keep, with the n=6 caveat from A5. |
| A8 | Mass and local eviction both beat random | `eviction_curve.py` | KL at last position | **Yes** — exact match (6.54e-4 / 3.04e-3 / 6.31e-3 vs random 3.42e-3 / 8.47e-3 / 1.68e-2) | `h2o2023` | REPRODUCED COMPUTATION | L3/H2, 2 prompts | High | Keep. |
| A9 | Local-damage shows no advantage over attention mass | `RESEARCH_STATUS.md`, Milestone 2 | Paired CI spans zero | Yes (Milestone 2, §B) | **`vatp2024` finds the opposite at 7B**; `fixedcontract2026` | NEGATIVE RESULT | Pythia-160M, synthetic prompts, one measured position | High for the measurement, low for generality | **Qualify.** Must cite `vatp2024` and state the regime difference. Not "value geometry does not help". |
| A10 | Sensitivity-allocated budgets show no advantage over uniform | `global_budget_curve.py` | Frontier comparison | Yes | `pyramidkv2024`, `adakv2024` (positive at scale) | NEGATIVE RESULT | same | High | **Qualify.** This is a null about *this estimator*; see C3. Cite the allocation prior art. |
| A11 | `n_singularity_skips` = 237 of ~2100 (~11%) | `RESEARCH_STATUS.md` | Counter | **Yes** — exactly 237/2100 | none found | REPRODUCED COMPUTATION | Milestone 2 grid | High | Keep the number; change the name and the stated cause (C2). |
| A12 | Skips driven by "concentrated (sink-like) attention" | `RESEARCH_STATUS.md` | Attribution | **Partly refuted** | `attnsink2025`, `massiveact2024` | INTERPRETATION | — | — | **Correct.** Concentration yes; "sink-like" backwards — a sink at the protected key 0 makes the guard unfireable. |
| A13 | `Gamma` ranking-inversion boundary theorem | `manuscript` §5 | Algebra + 0/55,971 and 0/426,816 verification | Not re-derived in this audit | none found | FORMALLY DERIVED | — | Medium | Keep "to our knowledge"; see C6 on how far the search went. |
| A14 | `d_B^local` score has no literal match; CriticalKV "structurally adjacent" | `manuscript/CLAIM_LEDGER.md` #3 | Literature audit | — | **`criticalkv2025`, `fixedcontract2026`** | INTERPRETATION | — | — | **Change.** "Structurally adjacent" understates it: CriticalKV's abstract states value-aware output-perturbation scoring as its contribution; Fixed-Contract's probe combines mass with estimated output change. Classify as MINOR VARIANT. |
| A15 | No wall-clock speedup is measured or claimed | `README.md` | — | N/A | — | — | — | High | Keep. Also note no memory is actually freed (eviction is an additive delta, not cache compaction). |
| A16 | No alignment or safety claim is made | `RESEARCH_STATUS.md`, `manuscript` | Repo-wide search | **Yes** — zero claims found | — | — | — | High | Keep exactly as is. |

## B. Milestone 2 numbers (reproduced)

| Quantity | Documented | Reproduced |
|---|---|---|
| Multistratum paired advantage (KL_mass − KL_local), mean | −1.86e−5 | see `results/milestone2_multistratum/summary.json` after re-run |
| Bootstrap 95% CI | [−5.67e−5, +2.01e−5] | spans zero (unchanged) |
| Strata: local wins / mass wins / tied | 6 / 3 / 11 | unchanged |
| Rank-divergence Spearman | 0.071 (p=0.77) | unchanged |
| `n_singularity_skips` | 237 | **237 exactly** |

## C. Audit findings (new claims, added by this audit)

| # | Claim | Basis | Status | Scope | Confidence | Recommended wording |
|---|---|---|---|---|---|---|
| C1 | The F10 statistic is not numerically reproducible across platforms: 8 of 13 per-stratum Spearman values differ, one flipping sign (−0.3143 → +0.8286); the median moves 0.086 → 0.257 | `experiments/reproduce_pilot.py` re-run, same model revision, same seed | REPRODUCED COMPUTATION (of the instability) | n=6 blocks/stratum | High | State F10 as "below 0.3, unstable at this sample size"; the < 0.7 verdict is robust, the digits are not. |
| C2 | The `m_B -> 1` guard detects extreme attention concentration in near-one-hot late-layer heads, not small causal windows | 2100 attempts; 851 singular rows; 0/851 with the causal window fully evicted; 851/851 with `alpha[q,0] < 1e-9` (median 2.03e-17); synthetic uniform-attention control negative at every fraction up to 95% | EMPIRICAL OBSERVATION + analytic control | This codebase | High | Rename to "degenerate-support eviction guard"; see `docs/SINGULARITY_FINDINGS.md`. |
| C3 | The guard's any-row OR silently discards 13.36% of planned eviction (up to 16.27% at q=0.5) with no change in measured KL; `n_entries_removed` and `estimated_bytes_saved` report the unreduced plan | `experiments/compare_guard_semantics.py`: applied/planned 0.8664 (A) vs 1.0000 (C), mean KL 1.0480e-02 vs 1.0571e-02, per-policy KL equal to 4+ s.f. | EMPIRICAL OBSERVATION | Milestone 2 grid | High | Report as a memory-axis measurement defect. KL results stand. |
| C4 | 214 of 237 skips are in layer 11, where a perturbation at any row but the last contributes **exactly 0.0** to the measured position | Direct injection test: random norm-10 delta at layers 10/11, rows 5/20/40 → KL = 0.000e+00 | FORMALLY DERIVED (architecture) + measured | Final-layer blocks | High | State plainly: the guard was discarding eviction the metric cannot see. |
| C5 | 97.1–99.8% of the measured KL comes from eviction at query rows other than the measured one | `counterfactual_eviction.py`, all-rows vs last-row-only ratio 0.002–0.029 | EMPIRICAL OBSERVATION | Pythia-160M | High | The experiments measure a prefill-style whole-matrix intervention, not a decode step. Say so. |
| C6 | Both scores are causally predictive of the repository's own KL: top-first eviction damages 4.0–6.6x more than bottom-first at matched budget; top > bottom in 57/60 (mass) and 52/60 (local) strata | `counterfactual_eviction.py`, 8400 interventions | EMPIRICAL OBSERVATION (interventional) | Pythia-160M, KL at last position | High | This licenses "importance" language **for this metric only** (`jain2019` bar met). |
| C7 | `d_B^local` is not comparable across layers; scale-normalizing raises its rank correlation with measured KL from +0.394 (p=0.086) to +0.613 (p=0.004) | n=20 strata | EMPIRICAL OBSERVATION | exploratory, n=20 | Medium | Explains A10. Present as a hypothesis for a better estimator, not a fix. |
| C8 | Measured downstream damage spans ~7 orders of magnitude with depth (layer 0 mean KL 1.63e-03 → layer 11 1.10e-09) while `d_B^local` does not | same | EMPIRICAL OBSERVATION | 5 layers | High | Do not average across layers anywhere. |
| C9 | No literature connects KV-cache eviction specifically to safety behaviour | Crossref searches across six phrasings | OPEN / gap | — | Medium | Report as a real gap this repository does not fill. |
| C10 | All 8 manuscript citations exist with exactly the stated authors and dates | arXiv `id_list` verification | REPRODUCED COMPUTATION | — | High | No fabricated citation found. |

## D. Claim downgrades and upgrades

**Downgrades (5).**
1. A5 — F10 `0.086` → "below 0.3, platform-unstable".
2. A9 — the local-vs-mass null must be stated against `vatp2024`'s opposite result at 7B.
3. A10 — the allocator null is a null about its sensitivity estimator (C7), not about non-uniform allocation.
4. A12 — "sink-like attention" is the wrong mechanism.
5. A14 — `d_B^local` is a MINOR VARIANT of `criticalkv2025`, not merely "structurally adjacent".

**Upgrades (2).**
1. C6 — "importance" is now interventionally supported for the repository's own KL metric, which it was not before. This is a strengthening the repository had not claimed.
2. A11 — the skip count is exactly reproducible, and now has a mechanism (C2/C4) rather than a guess.

**Unchanged and correct.** A1–A4, A6–A8, A13, A15, A16. In particular the
repository's alignment discipline (A16) needs no change, and its disclosure of
the KeepKV identity (A1) is exemplary and should not be softened.
