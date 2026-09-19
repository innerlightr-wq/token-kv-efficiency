# Novelty and provenance

Feature-by-feature provenance for everything this repository implements,
against literature verified live during the audit (arXiv export API and
Crossref; see `references.bib`, 32 entries, and the Zotero tree
`Token KV Efficiency`). Classifications used:

`CLASSICAL` · `KNOWN KV METHOD` · `MINOR VARIANT` · `IMPLEMENTATION DIFFERENCE`
`DIAGNOSTIC CONTRIBUTION` · `APPARENTLY DISTINCT` · `NEGATIVE RESULT`
`ARTIFACT / CONFOUND` · `UNCERTAIN — MORE SEARCH NEEDED`

No feature is labelled bare "novel".

---

## Summary table

| # | Feature | Closest prior art | Classification | Confidence |
|---|---|---|---|---|
| 1 | Exact delete-and-renormalize identity `Delta_B = -r_B/(1-m_B)` | `keepkv2025` (singleton, identical) | KNOWN KV METHOD | High |
| 2 | Block generalization of that identity | `keepkv2025` | MINOR VARIANT | High |
| 3 | Scoring by exact output perturbation, value-aware (`d_B^local`) | `criticalkv2025`; `fixedcontract2026`; `vatp2024` | MINOR VARIANT | Medium-High |
| 4 | Attention-mass ranking baseline | `h2o2023`, `scissorhands2023` | KNOWN KV METHOD | High |
| 5 | Scoring from the last query row (`alpha_last`) | `snapkv2024` (observation window) | KNOWN KV METHOD | High |
| 6 | Per-stratum budget allocation from calibration sensitivity | `pyramidkv2024`, `adakv2024`, `headkv2024`, `fastgen2024` | KNOWN KV METHOD | High |
| 7 | Multi-stratum (20 layer x head cell) comparison grid | `fixedcontract2026` | IMPLEMENTATION DIFFERENCE | Medium |
| 8 | `Gamma` pairwise ranking-inversion boundary theorem | none found | APPARENTLY DISTINCT | Medium |
| 9 | Local-vs-mass null result at 160M | `vatp2024`, `fixedcontract2026` (opposite sign, larger scale) | NEGATIVE RESULT | High |
| 10 | Sensitivity-allocation null result | `pyramidkv2024` (positive at scale) | NEGATIVE RESULT, explained | High |
| 11 | `m_B -> 1` guard semantics and its 13.36% budget loss | none found | DIAGNOSTIC CONTRIBUTION | Medium-High |
| 12 | `d_B^local` cross-layer non-comparability | `criticalkv2025` (normalizes differently) | DIAGNOSTIC CONTRIBUTION | Medium |
| 13 | GPT-NeoX / GPT-2 adapters, provenance capture, test suite | N/A | IMPLEMENTATION DIFFERENCE | High |

---

## 1–2. The exact delete-and-renormalize identity

**Repo implementation.** `src/kv_efficiency/local_algebra.py` (numpy) and
`src/kv_efficiency/eviction.py` (torch):
`Delta_B = (m_B o - o_B)/(1 - m_B) = -r_B/(1 - m_B)`.

**Standard terminology.** Exact output perturbation under
delete-and-renormalize; equivalently the leave-block-out update of a convex
combination.

**Closest source.** KeepKV (`keepkv2025`, arXiv:2504.09936) gives, for a single
deleted entry, `o' = (o - alpha_e v_e)/(1 - alpha_e)`, hence
`o' - o = alpha_e(o - v_e)/(1 - alpha_e)` — term for term, sign included, the
repository's singleton specialization.

**Identical?** Yes for the singleton. The block form follows by reading
`alpha_e, v_e` as `m_B, o_B/m_B`; a direct and minor extension.

**Surviving difference.** Only the generalization and the machine-checked
implementation. **The repository already states this**, in its abstract,
introduction and §4.2, and in `manuscript/CLAIM_LEDGER.md`. That disclosure is
correct and should not be weakened.

`KNOWN KV METHOD` (identity) / `MINOR VARIANT` (block form). High confidence.

## 3. `d_B^local` as a token-importance score

**Repo implementation.** `policies.local_damage_scores` — per-position
`||Delta_i||` at the last query row, evicting smallest first.

**Closest sources, all three closer than the manuscript currently allows:**

- **CriticalKV** (`criticalkv2025`, arXiv:2502.03805) is "a formal study on
  identifying critical KV cache entries by analyzing **attention output
  perturbation**", stating that "beyond attention weights, the **value
  states** within KV entries and pretrained parameter matrices are also
  crucial", and optimizes the **worst-case** output perturbation. The
  governing idea — value-aware scoring by output perturbation — is CriticalKV's
  stated contribution. The repository's score is the *exact* perturbation for
  the specific block rather than a worst-case bound.
- **Fixed-Contract Diagnostic** (`fixedcontract2026`, arXiv:2605.08234): "For
  value ranking, the probe **combines a block's attention mass with the
  estimated output change from removing it**." That is precisely the
  repository's `(m_B, d_B^local)` pair, in a diagnostic framing, evaluated on
  LongBench across three models plus NeedleBench and RULER.
- **VATP** (`vatp2024`, EMNLP 2024, doi:10.18653/v1/2024.emnlp-main.1178)
  scores tokens by attention score together with the L1 norm of value vectors.

**Identical?** No formula is literally identical. But the *principle* the
repository treats as its scoring insight is published, formally in CriticalKV
and diagnostically in Fixed-Contract.

**Assessment.** `manuscript/CLAIM_LEDGER.md` claim 3 records "no literal match
found (CriticalKV/Fixed-Contract structurally adjacent, not equal — different
norm/reference point)". On the fetched abstracts that **understates the
overlap**: "structurally adjacent" is too weak for a paper whose abstract
states the same principle. Recommended wording change in
`docs/CLAIM_LEDGER.md`.

`MINOR VARIANT`. Medium-high confidence (based on abstracts, not full texts —
a full-text read of CriticalKV is the one remaining check).

## 4–5. Attention-mass ranking, and reading it at the last row

`h2o2023` (H2O, NeurIPS 2023) established heavy-hitter eviction by
**accumulated** attention; `scissorhands2023` established persistence of
importance across steps. The repository's `attention_mass_ranking` is the
same signal read at a **single** query row.

Reading importance from recent query rows is **SnapKV**'s observation window
(`snapkv2024`, arXiv:2404.14469); the repository's `alpha_last` is that window
at size 1. This is currently **uncited** and should be cited where
`alpha_last` is introduced, with the window-size-1 relationship stated.

The repository correctly treats attention mass as a baseline, not a
contribution. `KNOWN KV METHOD`, high confidence.

## 6. Budget allocation across strata

`src/kv_efficiency/allocator.py` allocates removal fraction inversely to
calibration-estimated mean `d_B^local`, water-filled to conserve the global
budget exactly.

Non-uniform KV budgets are established: **PyramidKV** (`pyramidkv2024`)
allocates by layer following pyramidal information funnelling; **Ada-KV**
(`adakv2024`) allocates adaptively across heads with a loss-reduction
argument; **HeadKV** (`headkv2024`) allocates at head level; **FastGen**
(`fastgen2024`) profiles attention structure to pick per-head policies. All
four are currently **uncited**.

The novelty question is not the variable name but whether anyone already
allocates KV budget from equivalent information. They do. `KNOWN KV METHOD`,
high confidence. Exact global-budget conservation by water-filling is an
implementation detail of an existing idea.

## 7. The multi-stratum grid

20 `(layer, head)` cells x 3 prompts x 5 removal fractions, with a
calibration/evaluation prompt split fixed in advance. Fixed-Contract
(`fixedcontract2026`) runs the same genre of experiment (hold the selector
fixed, vary one decision slot) at substantially larger scale with a 264-cell
sign evaluation. `IMPLEMENTATION DIFFERENCE`; the repository's grid is smaller
and on smaller models. Medium confidence.

## 8. The `Gamma` ranking-inversion boundary theorem

`manuscript/token_kv_eviction.tex` §5 derives the exact boundary at which the
`d_B^local` ranking of two candidate positions inverts relative to the
attention-mass ranking, with equality case and edge cases, verified against
direct recomputation with 0/55,971 mismatches (Pythia) and 0/426,816 (GPT-2).

Searched against `keepkv2025`, `criticalkv2025`, `fixedcontract2026`,
`vatp2024`, `h2o2023`, `scissorhands2023`, `snapkv2024`, `pyramidkv2024`,
`adakv2024`, `headkv2024`, `keyformer2024`, `quest2024`,
`duoattention2024`, `razorattention2024` and three surveys
(`kvsurvey2024`, `effinfsurvey2024`, `keepcostdown2024`). None derives a
pairwise inversion boundary between a mass ranking and a perturbation
ranking. Fixed-Contract's "leverage effects near the boundary" is the nearest
published notion and is not the same object.

`APPARENTLY DISTINCT` — stated as "no prior statement found", not as a new
theorem of consequence. It is an algebraic reformulation of when two scores
disagree; its value is that it makes the empirical inversion study possible.
Medium confidence (abstract-level search of 26 arXiv works and three
surveys; not an exhaustive full-text search of the field).

## 9. Local-vs-mass: the null result

The repository finds no measurable advantage of `d_B^local` over attention
mass at either single-head or 20-stratum scale on Pythia-160M (pooled paired
advantage −1.86e−5, bootstrap 95% CI [−5.67e−5, +2.01e−5], spans zero;
local wins 6 strata, mass 3, tied 11).

**This must be read against `vatp2024`, which reports the opposite at larger
scale**: value-awareness beats attention-only on more than 12 of 16 LongBench
tasks on LLaMA2-7B-chat and Vicuna-v1.5-7B. And `fixedcontract2026` finds the
mass-plus-output-change probe positive on 72.6% of positive-margin cells.

These are not contradictions — they are different regimes (160M vs 7B,
synthetic prompts and one-position KL vs real long-context tasks) — but the
repository's evidence is the weaker of the two, and the null **cannot** be
stated as "value geometry does not help" without citing the published
positive results. `NEGATIVE RESULT`, scope-limited. High confidence in the
measurement, low in any generalization.

## 10. Sensitivity allocation: the null result, now explained

The allocator showed no consistent advantage over uniform budgets. This audit
found a direct explanation: **its sensitivity proxy is nearly uninformative
about the damage it is trying to avoid.**

Across the 20 strata, mean `d_B^local` vs measured downstream KL gives
Spearman **+0.394 (p=0.086, n=20)** — not significant. Measured damage is
dominated by **layer depth**, spanning about seven orders of magnitude
(layer 0 mean KL 1.63e-03 down to layer 11 1.10e-09), and layer 11 has the
*second highest* mean `d_B^local`. Scale-normalizing the score
(`d_B^local/||o||`) raises the correlation to **+0.613 (p=0.004)**.

So the null is a null about **this sensitivity estimator**, not about
non-uniform allocation — and the structure the estimator misses (monotone
decrease of influence with depth) is exactly what `pyramidkv2024` exploits
successfully. `NEGATIVE RESULT`, with a mechanism. High confidence in the
measurement; the n=20 correlations are exploratory.

## 11. The `m_B -> 1` guard as a diagnostic finding

Full treatment in `docs/SINGULARITY_FINDINGS.md`. In provenance terms:

- Searched for prior reports that KV/attention concentration metrics become
  spuriously extreme because early causal query rows have mechanically short
  support. **No such prior statement found** — and this audit's synthetic
  control shows the repository's guard is *not* that phenomenon, so the
  question is moot for this codebase. Recorded as
  `APPARENTLY DISTINCT DIAGNOSTIC OBSERVATION — NO PRIOR STATEMENT FOUND`,
  not as a theorem.
- The finding that *does* survive is different and more concrete: a guard
  quantified with `any()` over all query rows, combined with rankings built
  from one row and damage measured at one position, silently discards 13.36%
  of planned eviction while the memory accounting reports the full plan. No
  prior art found for this specific implementation hazard, which is
  unsurprising — it is a property of this codebase, not of the field.
- The attention geometry behind it is published: near-one-hot late-layer heads
  and massive activations (`massiveact2024`), and sink emergence
  (`attnsink2025`).

`DIAGNOSTIC CONTRIBUTION`. Medium-high confidence.

## 12. `d_B^local` is not comparable across layers

`d_B^local` is an unnormalized L2 norm in a `d_model` space whose scale grows
with depth, carrying no weighting by downstream influence. CriticalKV takes
its bound over query states rather than leaving the norm raw. Documented in
`docs/METRIC_DEFINITIONS.md` §5. `DIAGNOSTIC CONTRIBUTION`, medium confidence
(n=20, exploratory).

## 13. Engineering

GPT-NeoX and GPT-2 adapters, forward-hook utilities, provenance capture
(`capture_environment` records config snapshot, model revision, package
versions in every result file), 71-test suite. No novelty claimed; this is
where much of the repository's actual value sits. `IMPLEMENTATION DIFFERENCE`.

---

## Collections deliberately left empty

Three Zotero subcollections hold nothing, honestly rather than padded:

- **10 — Low-Rank / Spectral KV**: the repository computes no singular values,
  effective rank, or low-rank factorization anywhere. The word "singularity"
  here refers to the pole of a scalar rational function at `m_B = 1`, **not**
  to matrix singularity or spectral structure. Nothing to compare.
- **12 — Long-Context Evaluation**: the repository runs no LongBench, RULER,
  NeedleBench, PG19 or perplexity evaluation. What publication-grade
  validation would require is listed in `docs/REPRODUCTION_AUDIT.md`.
- **15 — RoPE / Position Handling**: the repository never re-indexes or
  compacts a cache. Eviction is implemented as an additive delta on the head
  *output*, so no retained key is ever reused at a new position and no RoPE
  phase is recomputed. Position remapping — the thing RoPE-aware eviction work
  addresses — does not arise. This is a genuine scope limit, not an oversight:
  it also means the repository does not measure any real cache-compaction cost.

## Missing citations, ranked

1. **`vatp2024`** — published (EMNLP 2024), reports the opposite of the
   repository's headline null at 7B scale. Highest priority.
2. **`snapkv2024`** — `alpha_last` is its observation window at size 1.
3. **`pyramidkv2024`**, **`adakv2024`**, **`headkv2024`**, **`fastgen2024`** —
   the allocator's prior art.
4. **`jain2019`**, **`wiegreffe2019`** — required before any "importance"
   language; now *satisfied* by this audit's counterfactual test.
5. **`massiveact2024`**, **`attnsink2025`** — the mechanism behind the guard
   events, and the correction to the "sink-like" attribution.
6. **`beyondperplexity2024`** — why a small KL is not a safety statement.
7. Surveys **`kvsurvey2024`**, **`keepcostdown2024`**, **`effinfsurvey2024`**
   for orientation.

The eight works the manuscript already cites were all verified to exist with
exactly the stated author lists and dates. No fabricated citation was found.
