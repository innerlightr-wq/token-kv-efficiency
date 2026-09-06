# Token KV Efficiency — Checkpoint

## Research goal

Adaptive KV-cache efficiency under bounded downstream damage.

## Exact local algebra

For attention output

    o = Σ_i α_i v_i

and block B,

    m_B = Σ_{i∈B} α_i

    o_B = Σ_{i∈B} α_i v_i

    r_B = o_B - m_B o

the exact delete-and-renormalize change is

    Δ_B = -r_B/(1-m_B)

and

    d_B^local = ||Δ_B||.

**Label: EXACT ALGEBRA.** Verified by direct substitution against
independently-recomputed delete-and-renormalize output on synthetic data,
including the `m_B → 1` edge case (correctly raises rather than returning
a silently huge or NaN value) — `tests/test_local_algebra.py`, 24/24
passing.

## Engineering validation

Reproduced values (`results/validate_local_identity.json`,
`EleutherAI/pythia-160m`, layer 3, head 2):

    QKV/attention reconstruction relative error ≈ 6.988e-8

    delete/renormalize identity absolute error ≈ 1.110e-16

    deterministic repeat error = 0

**Label: ENGINEERING VERIFIED.**

## Original pilot

    Pythia-160M
    n = 96

    pooled Spearman(d_local, Tier-B KL) ≈ 0.857

    pooled Spearman(Tier-A, Tier-B) ≈ 0.838

    F10 median within-head Spearman(Tier-A,Tier-B) ≈ 0.086

**F10 fires because 0.086 < 0.7.**

**Label: PRELIMINARY EMPIRICAL RESULT.**

F10 is not called disproved or falsified in a universal sense: the sample
is far below the preregistered minimum of 10 blocks/stratum, so this is
evidence against the within-head ranking hypothesis at this sample size,
not a confirmatory refutation.

## Milestone 1 eviction result

Both attention-mass eviction and local-damage eviction beat random
eviction in the initial single-head experiment
(`results/eviction_curve.json`, layer 3/head 2, 2 prompts). Local-damage
did not demonstrate an advantage over attention mass in that experiment.

**Label: EXPLORATORY RESULT.**

## Milestone 2 multistratum result

    20 layer/head strata
    3 evaluation prompts
    5 removal fractions
    268 paired observations

    mean paired advantage (KL_mass - KL_local) ≈ -1.86e-5
    bootstrap 95% CI ≈ [-5.67e-5, +2.01e-5]

    local wins = 6 strata
    mass wins = 3 strata
    ties = 11 strata

    Spearman(1 - rankAgreement(m_B, d_local), local advantage) ≈ 0.071
    p ≈ 0.77, n = 20

**Interpretation:** no detected evidence in this experiment that greater
divergence between attention-mass ranking and local-damage ranking
predicts a practical advantage for local-damage eviction.

**Label: EXPLORATORY RESULT.**

## Global budget result

At damage tolerance ε = 0.01, maximum tested removal fractions were:

    uniform_mass       0.30
    allocated_mass     0.30
    uniform_local      0.20
    allocated_local    0.20
    uniform_random     0.20

Sensitivity-based budget allocation did not consistently improve over
uniform allocation. Attention-mass is currently the stronger practical
baseline.

This does not generalize beyond the tested model/context regime
(Pythia-160M, 5 layers, 4 heads, 3 evaluation prompts, contexts of a few
dozen tokens).

**Label: EXPLORATORY RESULT.**

## Important failure / limitation

Approximately 11% of aggressive stratum-eviction attempts (237 of ~2100,
in the global budget experiment) encountered the `m_B → 1` singularity
guard and were skipped. This remains visible as a methodological
limitation, not hidden inside an averaged number — see
`results/milestone2_global_budget/summary.json`'s `n_singularity_skips`
field and `docs/RESEARCH_STATUS.md`.

## Current conclusion

Structured KV eviction is empirically more effective than random eviction
in the tested Pythia-160M experiments.

However, the exact local-damage score has not demonstrated an efficiency
advantage over the simpler attention-mass baseline.

The result is model- and context-specific.

## Next milestone

Recorded, **not started**:

Test whether attention mass remains the stronger baseline on a different
architecture / larger model / longer context.

Potential secondary question: whether local residual geometry becomes
useful as a *correction* to attention mass rather than as a *replacement*
for it.
