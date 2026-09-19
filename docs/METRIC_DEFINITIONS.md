# Metric definitions

Every quantity this repository computes, with its exact formula, domain, and
the dependencies that matter when the number is compared across layers,
heads, positions, or policies. Produced by the literature/reproduction audit;
the empirical claims here were measured in this repository's own environment
(see `docs/REPRODUCTION_AUDIT.md`).

Classification tags used below:

- `RAW` — an unnormalized magnitude; not comparable across contexts without
  further scaling.
- `NORMALIZED` — dimensionless or bounded by construction.
- `POSITION-DEPENDENT` — its value depends on which query row it is read at.
- `WINDOW-SIZE-DEPENDENT` — its value depends on `w_q`, the number of keys
  query row `q` can legally attend to.
- `LAYER-DEPENDENT` — its distribution changes materially with depth.
- `POLICY-DEPENDENT` — defined only relative to an eviction selection.

Notation follows `src/kv_efficiency/local_algebra.py`: `alpha` is one
softmax-normalized attention row over keys, `values[i] = W_O^h v_i` is the
per-key value vector already mapped into `d_model`, and `B` is the evicted
block.

---

## 1. `alpha` — attention weights

```
alpha[q, k] = softmax_k( q_q . k_k / sqrt(d_head) + causal mask )
```

`NORMALIZED`, `POSITION-DEPENDENT`, `WINDOW-SIZE-DEPENDENT`, `LAYER-DEPENDENT`.

Domain `[0, 1]`, rows sum to 1. Because the mask is causal, row `q` has
support only on keys `0..q`, so

```
w_q = q + 1
```

is the row's causal window. Every quantity built from `alpha` inherits this:
row 0 is a point mass by construction, and early rows are mechanically more
concentrated than late rows at equal "true" sharpness. `require_normalized`
rejects rows that do not sum to 1 within `1e-12` rather than silently
repairing them.

Measured on `EleutherAI/pythia-160m` (12 layers x 12 heads, evaluation prompt
1, 57 tokens): median per-row max weight ranges from **0.186** (layer 0,
head 6) to **1.0000** (layer 10-11, several heads), and median per-row
entropy from **2.80 nats** down to **0.000**. Median effective support
`1/sum_k alpha^2` falls to exactly **1.00** in several layer-9/10/11 heads —
those heads are numerically one-hot.

## 2. `m_B` — block attention mass

```
m_B[q] = sum_{k in B} alpha[q, k]
```

`NORMALIZED`, `POSITION-DEPENDENT`, `WINDOW-SIZE-DEPENDENT`, `POLICY-DEPENDENT`.

Domain `[0, 1]`. **This is a per-query-row vector, not a scalar**, whenever it
is computed from a full `(seq, seq)` attention matrix — which is what
`eviction.delta_from_block` and `eviction.global_eviction` do. That fact is
load-bearing and is the subject of `docs/SINGULARITY_FINDINGS.md`.

`m_B` is strongly window-size-dependent even under ideal conditions. Under
*perfectly uniform* causal attention (`alpha[q,k] = 1/(q+1)`), evicting the
same 50% of candidate keys gives `m_B = 0.969` at row 31 but `m_B = 0.484` at
row 63. Comparing raw `m_B` across query rows therefore compares different
things. It does **not** follow that the guard below is a window artifact; see
§6 and `docs/SINGULARITY_FINDINGS.md`.

## 3. `o`, `o_B`, `r_B` — head output, block contribution, residual

```
o     = sum_k alpha_k values_k          (head output, d_model)
o_B   = sum_{k in B} alpha_k values_k   (block's raw contribution)
r_B   = o_B - m_B * o                   (residual)
```

`RAW`, `LAYER-DEPENDENT`. Unbounded vectors in `d_model` space.

Mean `||o||` measured across the audited strata: **0.576** (layer 0), 0.979
(layer 3), 0.943 (layer 6), 0.736 (layer 9), **2.535** (layer 11). The
residual stream itself grows much faster with depth — median `||h||` is 0.8
at layer 0, 17.3 at layer 6, 31.4 at layer 11, and 270.8 after the final
layer norm.

## 4. `Delta_B` — exact perturbation from evicting `B`

```
Delta_B[q] = ( m_B[q] * o[q] - o_B[q] ) / ( 1 - m_B[q] )   =  -r_B[q] / (1 - m_B[q])
```

`RAW`, `POSITION-DEPENDENT`, `LAYER-DEPENDENT`, `POLICY-DEPENDENT`.

Exact, not an approximation: deleting `B` and renormalizing the surviving
weights gives `o' = (o - o_B)/(1 - m_B)`, so `o' - o = Delta_B` by
substitution. Verified against direct recomputation to float64 tolerance
(`tests/test_local_algebra.py`) and, on the real model, to `1.110e-16`
(`experiments/validate_local_identity.py`).

**Edge case — the pole at `m_B = 1`.** The expression is singular exactly when
the evicted block carries all of a row's attention mass. This is a genuine
pole of the identity, not a numerical accident: if every key a row attends to
is removed, the renormalized output is `0/0`. The implementation raises at
`m_B >= 1 - 1e-9`. What that threshold *empirically* detects in this
repository is documented in `docs/SINGULARITY_FINDINGS.md` — it is not what
the name "singularity" suggests.

## 5. `d_B^local` — local damage score

```
d_B^local = || Delta_B ||_2 = || r_B || / (1 - m_B)
```

`RAW`, `POSITION-DEPENDENT`, `LAYER-DEPENDENT`, `POLICY-DEPENDENT`.

Domain `[0, inf)`. This is the repository's token-importance score, computed
per singleton position at the last query row in every experiment
(`policies.local_damage_scores`).

**`d_B^local` is not comparable across layers.** It is an unnormalized L2
norm in a `d_model` space whose scale changes with depth, and it carries no
weighting by how much that head's output actually influences the model's
output. Measured across the 20 audited `(layer, head)` strata:

| relationship | Spearman | p | n |
|---|---|---|---|
| mean `d_B^local` vs measured downstream KL | **+0.394** | 0.086 | 20 |
| `||o||` vs measured downstream KL | −0.347 | 0.134 | 20 |
| mean `d_B^local` / `||o||` vs measured downstream KL | **+0.613** | **0.004** | 20 |

Dividing by the head-output scale — i.e. making the score a *relative*
perturbation — moves it from non-significant to significant. The raw score is
largest in layer 11 (mean 0.0291, second highest of the five layers) where
measured downstream damage is smallest (mean KL 1.10e-9, seven orders of
magnitude below layer 0). This is the direct explanation for the
budget-allocator null result in `docs/RESEARCH_STATUS.md`; see
`docs/NOVELTY_AND_PROVENANCE.md`. Exploratory, n=20.

`d_B^local` *is* causally predictive of the repository's own KL metric within
a stratum: see §8.

## 6. The `m_B -> 1` guard

```
if any(m_B[q] >= 1 - 1e-9 for q in all query rows):  raise
```

`POLICY-DEPENDENT`, `LAYER-DEPENDENT`. Not window-size-dependent in practice.

The quantifier is an OR over **every** query row of the full attention
matrix, while the rankings that build `B` are computed from the **last** row
only and the reported KL is read at the **last** position only. What it
actually fires on, measured over all 2100 stratum-eviction attempts of the
Milestone 2 global-budget grid:

- 237 attempts skipped (11.3%), containing 851 singular query rows;
- **12 of 237** have the measured (last) row singular;
- **0 of 851** singular rows have their entire causal window evicted;
- **851 of 851** have `alpha[q, 0] < 1e-9` (median **2.03e-17**) — the
  protected sink key carries numerically zero weight at that row;
- all 237 skips are in layers 9 and 11 (214 in layer 11); layers 0, 3, 6
  produce **zero**.

Under perfectly uniform causal attention the guard **never** fires, even at
95% eviction (max `m_B = 0.983`). It fires only once off-diagonal attention
underflows below about `1e-9`. It is therefore an **extreme-concentration
detector**, not a small-causal-window detector, and `docs/SINGULARITY_FINDINGS.md`
proposes renaming it accordingly.

## 7. Damage metrics

```
KL(p_full || p_compressed) = sum_v p_full(v) [ log p_full(v) - log p_comp(v) ]
logit_l2                   = || logits_full - logits_compressed ||_2
hidden_state_l2            = || h_full - h_compressed ||_2
```

`RAW` for the L2 variants, `NORMALIZED`-ish for KL (nats, `[0, inf)`),
`POSITION-DEPENDENT`. Computed in float64; KL's direction is fixed with the
full model as reference.

**Scope that is easy to miss:** every reported KL in this repository is read
at **one token position** (the last). Two consequences, both measured here:

1. **97.1–99.8% of the measured KL comes from eviction applied at query rows
   other than the one being measured.** Applying the identical delta at the
   final row alone instead of all rows reduces mean KL by a factor of 34–500
   (ratio 0.029 at q=0.5 down to 0.002 at q=0.1). The experiments therefore
   measure a prefill-style whole-matrix intervention, not a decode step.
2. **In the final layers, non-final rows cannot contribute at all.** A random
   norm-10 perturbation injected at layer 10 or 11 at any row other than the
   last produces *exactly* 0.0 KL at the last position, because layer 11's
   row-`q` output feeds only `logits[q]`. For layer 11 the all-rows and
   last-row-only measurements coincide (ratio 1.000).

## 8. Retention / budget quantities

```
q          = requested global removal fraction
k          = round(q_h * |candidates|)     per stratum
candidates = all positions except protect = {0, last}
```

`POLICY-DEPENDENT`. Note that position 0 and the final position are protected
in every experiment, so the sink key is never evicted.

**`n_entries_removed` and `estimated_bytes_saved` in
`results/milestone2_global_budget/summary.json` are computed from the
eviction *plan*, before `global_eviction` reports which strata it skipped.**
They therefore overstate realised eviction whenever the guard fires:

| requested q | planned entries | not applied | overstatement |
|---|---|---|---|
| 0.1 | 1942 | 92 | 4.74% |
| 0.2 | 3638 | 307 | 8.44% |
| 0.3 | 5490 | 711 | 12.95% |
| 0.4 | 7312 | 1078 | 14.74% |
| 0.5 | 9244 | 1504 | 16.27% |
| **all** | **27626** | **3692** | **13.36%** |

The KL values are unaffected (they measure whatever was actually applied);
the **memory axis** of the quality-memory curve is what is overstated. At
q=0.5 the realised fractions are 0.433 for the four structured policies and
0.401 for `uniform_random`, so the policies are *not* compared at matched
realised budget. Random received *less* eviction and still showed higher KL,
so the "structured beats random" conclusion survives and is conservative at
that point; the absolute `estimated_bytes_saved` figures are not.

## 9. Causality of the importance scores

Attention weight alone is not explanation (`jain2019`, `wiegreffe2019`), so
"importance" requires an intervention. Measured by
`experiments/counterfactual_eviction.py` (8400 interventions, 8364 valid):
evicting the **top**-scored positions instead of the bottom-scored, at
matched budget, increases mean KL by

| scorer | q=0.1 | q=0.2 | q=0.3 | q=0.4 | q=0.5 |
|---|---|---|---|---|---|
| attention mass | 5.9x | 5.9x | 6.6x | 5.6x | 5.2x |
| local damage | 5.7x | 4.0x | 5.2x | 4.4x | 4.5x |

with matched-size random eviction in between (1.1–1.4x the bottom-first
baseline). Per-stratum sign test at q=0.3: top damages more than bottom in
**57/60** strata for attention mass and **52/60** for local damage.

Both scores are therefore causally predictive **of this repository's own KL
metric**. Note the asymmetry: the gap between bottom-first and random is
small (1.1–1.4x) while the gap to top-first is large (4–6.6x), i.e. the
scores identify the few decisive positions well and barely separate the
unimportant majority — consistent with the heavy-hitter picture
(`h2o2023`, `scissorhands2023`).
