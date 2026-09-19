# The `m_B -> 1` events: what they actually are

Produced by the literature/reproduction audit. Reproduction commands and
environment in `docs/REPRODUCTION_AUDIT.md`; the diagnostic scripts are
`experiments/diagnose_singularity_skips.py` and
`experiments/compare_guard_semantics.py`, both of which import the production
code and change nothing in it.

## Original observation

`docs/RESEARCH_STATUS.md` (METHOD OBSTRUCTION) reports that of ~2100
stratum-eviction attempts in the Milestone 2 global-budget experiment, **237
(~11%) hit the exact `m_B -> 1` singularity guard** and were skipped, and
attributes this to "the allocator's `max_frac=0.95` bound combined with short
prompts and concentrated (sink-like) attention in some heads."

The skip count reproduces exactly: **2100 attempts, 237 skipped (11.3%)**,
same model revision `50f5173d932e8e61f858120bcb800b97af589f46`. The stated
*cause* is partly wrong, and the consequence is larger than reported.

## Exact metric

`eviction.delta_from_block` is handed the **full** `(seq, seq)` attention
matrix, so

```
m[q] = sum_{k in B} alpha[q, k]        for every query row q
if torch.any(m >= 1.0 - 1e-9):  raise
```

and `global_eviction` catches that and drops the whole `(layer, head)`
stratum. Three different query rows are involved in one decision:

| role | which row |
|---|---|
| builds the eviction set `B` | the **last** row (`alpha_full[qp]`) |
| triggers the guard | **any** row, via the OR |
| reported damage (KL) | the **last** position only |

## Why it triggered

Not because of short causal windows. The measured mechanism, over all 851
singular query rows:

1. The head is **numerically one-hot**. Every skipping stratum has median
   per-row max attention weight `>= 0.9992`, median entropy `<= 0.006` nats,
   and median effective support `1.00`.
2. The eviction set is chosen from the **last** row's ranking, which puts the
   many `~1e-20`-weight keys first. For a *different* row `q`, the single key
   that row actually attends to is among them, so `B` captures that row's
   entire attended mass.
3. The protected sink key 0 carries **numerically zero** weight at that row.
   All 851 singular rows have `alpha[q, 0] < 1e-9`, median **2.03e-17**,
   minimum measured `6.76e-31`.

Hence `1 - m[q]` collapses: **217 of 851** are exactly `0.0` in float64, 265
are below machine epsilon, median `1 - m = 3.9e-13`. The guard's `1e-9`
threshold is met with about four orders of magnitude to spare, so these are
not borderline cases and not a threshold-tuning question.

Crucially, **0 of 851** singular rows had their whole causal window evicted.
The key-0 protection guarantees at least one visible unmasked key always
exists; the mass on it is simply negligible.

## Early-query contribution

Real but not the driver. Singular rows have causal window `w_q` between 3 and
54, median **17**, mean 19.0, quartiles 10 / 17 / 25. Median relative position
in the sequence is **0.357** — middle, not the beginning. Excluding short
windows removes almost nothing:

| exclude rows with | skips remaining (of 237) |
|---|---|
| `w_q < 4` | 237 (100%) |
| `w_q < 8` | 230 (97%) |
| `w_q < 16` | 178 (75%) |
| `w_q < 32` | 84 (35%) |

A pure small-window artifact would have been eliminated by the first row of
that table. It was not.

## Early-key contribution

This is the operative axis, with the **opposite** sign to the usual
attention-sink story. Pythia-160m does have a strong position-0 sink: mean
`alpha[.,0]` over 12 heads rises from 0.048 (layer 0) to 0.42–0.55 (layers
4–11). Because key 0 is *protected* in every experiment, a head with a
healthy sink can never trigger the guard — the retained mass is bounded below
by the sink weight. The synthetic test confirms it: with every row placing 0.9
on key 0, evicting 95% of the remaining keys gives `max m_B = 0.100` and the
guard is unfireable.

The guard fires in heads where the sink weight **collapses at particular
rows** while the row is one-hot elsewhere. Every skipping stratum satisfies
`min_q alpha[q, 0] < 1e-9`; no stratum without that property skips at all.

So `docs/RESEARCH_STATUS.md`'s "concentrated (sink-like) attention" is half
right and half backwards: concentration yes, sink no. Sink presence
*suppresses* the event.

## Causal-window dependence

`m_B` genuinely is window-size-dependent — under perfectly uniform causal
attention and a fixed 50% eviction it is 0.969 at row 31 and 0.484 at row 63.
But that dependence never approaches the threshold: under uniform causal
attention every row retains key 0 with weight `1/(q+1) >= 1/64`, so
`max m_B = 0.983` even at 95% eviction. The window dependence is real and
worth documenting (it is in `docs/METRIC_DEFINITIONS.md`), and it is not what
produced these events.

## Per-query normalization result

Replacing the OR over rows with a per-row decision changes the realised
budget substantially and the measured damage not at all
(`experiments/compare_guard_semantics.py`, 315 configurations):

| guard semantics | strata skipped | eviction applied / planned | mean KL |
|---|---|---|---|
| A — any-row skip (**production**) | 237 | **0.8664** | 1.0480e-02 |
| B — guard the measured row only | 12 | 0.9944 | 1.0571e-02 |
| C — per-row exclusion | **0** | **1.0000** | 1.0571e-02 |

For each individual policy and `q` the three KL values agree to four or more
significant figures (e.g. `uniform_mass` at q=0.5: 1.546e-02 under all
three), while the realised removal fraction moves from 0.433 to 0.500.
Semantics A reproduces the committed production numbers (median relative
deviation 1.1e-03 against `results/milestone2_global_budget/rows.json`).

The reason the damage does not move is structural: **214 of the 237 skips are
in layer 11**, the final block, where a perturbation at any row other than
the last contributes *exactly* 0.0 to the last position's logits (measured:
a random norm-10 delta at layer 10 or 11, rows 5/20/40, gives KL = 0.000e+00).
The guard was discarding eviction that could not have changed the reported
metric.

**Net effect of the production guard: it silently reduces realised eviction by
13.36% overall (up to 16.3% at q=0.5) without changing measured damage, while
`n_entries_removed` and `estimated_bytes_saved` continue to report the
unreduced plan.** That is a measurement-integrity issue on the memory axis of
the quality-memory curve, not a damage-side error.

## Synthetic uniform-attention result

`A[q,k] = 1/(q+1)` for `k <= q`, n=64, protecting `{0, last}`:

| eviction fraction | guard fires | max `m_B` | at row |
|---|---|---|---|
| 10% | no | 0.857 | 6 |
| 30% | no | 0.950 | 19 |
| 50% | no | 0.969 | 31 |
| 90% | no | 0.982 | 56 |
| 95% | no | 0.983 | 59 |

**The metric does not declare pathology under perfectly uniform causal
attention.** This is the single most important control in this document and it
comes out negative: the guard is not structurally confounded by causal-window
size. (Removing the key-0 protection *does* make it fire immediately, with
`m[row 0] = 1.0` exactly — that is the textbook small-window failure mode, and
the repository's protect set prevents it.)

## Synthetic concentrated-attention result

Each row placing `1 - eps` on its diagonal key, eps spread over the rest:

| off-diagonal weight | guard fires | singular rows |
|---|---|---|
| ~1e-2 | no | 0 |
| ~1e-6 | no | 0 |
| ~1e-10 | **yes** | 31 |
| ~1e-17 | **yes** | 31 |

The guard distinguishes concentration from uniformity, which is what a
concentration detector should do — but its trigger point is set by float
underflow (`~1e-9`), not by any property of the model. It cannot distinguish
"very concentrated" (1e-6) from "uniform" (it passes both); it only separates
"concentrated past the reciprocal of the epsilon".

## Attention-sink synthetic result

Every row `q >= 1` placing 0.9 on key 0: guard never fires at any eviction
fraction, `max m_B = 0.100`. The metric therefore does **not** recognise an
attention sink as pathological, and correctly so given key-0 protection —
but this also means the guard cannot be read as a sink detector.

## Layer effects

| layer | skips | mean measured KL (bottom-first, q=0.3) | mean `d_B^local` |
|---|---|---|---|
| 0 | 0 | 1.63e-03 | 0.0420 |
| 3 | 0 | 6.10e-04 | 0.0118 |
| 6 | 0 | 1.22e-04 | 0.0155 |
| 9 | 23 | 1.47e-05 | 0.0055 |
| 11 | 214 | **1.10e-09** | 0.0291 |

Measured damage falls by about seven orders of magnitude with depth while
`d_B^local` does not — layer 11 has the second-highest mean local score and
essentially no downstream effect. Do not average these layers together.

## Policy effects

Skips per policy across all prompts and `q`: `uniform_mass` 32,
`uniform_local` 32, `allocated_mass` 32, `allocated_local` 32,
`uniform_random` 109 (over 3 seeds, ~36/seed). Mass-ranked and local-ranked
policies skip **identically**, because in a one-hot head both rankings put the
same near-zero-weight keys first. Skips grow with budget: 20 at q=0.1 rising
to 67 at q=0.5.

## What remains after controls

- The events are **real**: `1 - m_B` genuinely reaches 0.0 in float64 for 217
  rows. Nothing is being miscomputed; the algebra is correctly reporting that
  the caller asked to delete a row's entire effective attention mass.
- They are **not** a causal-window artifact. The uniform-attention control is
  negative and window exclusion does not remove them.
- They are **not** primarily a numerical-precision artifact either. The
  underlying `alpha` values (1e-17 to 1e-31) are representable; float32 vs
  float64 changes the threshold's exact crossing point but not the
  phenomenon, which is genuine one-hot concentration.
- The **name is wrong**, and the **response is wrong**. "Singularity" suggests
  instability of the identity; what is detected is extreme attention
  concentration in near-deterministic late-layer heads. And skipping the whole
  stratum discards eviction the metric was provably insensitive to, while the
  memory accounting keeps counting it.

**Proposed rename.** `m_B -> 1` guard → **degenerate-support eviction guard**,
and `n_singularity_skips` → `n_degenerate_support_skips`. What it detects, said
plainly: *the requested eviction removes every key that this query row places
non-negligible attention on.* This audit does not rename the production
symbols (that is a code change beyond a provenance audit); the recommendation
is recorded here and in `docs/CLAIM_LEDGER.md`.

## Final interpretation

The events are a correct response to a real degeneracy, arising from
near-one-hot late-layer attention heads rather than from causal-window
geometry, mis-described by their current name and mis-handled by an
all-or-nothing per-stratum skip whose only measured consequence is a 13.36%
silent overstatement of eviction on the memory axis.

`REAL BUT MISNAMED`
