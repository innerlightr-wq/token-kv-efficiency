# Gate 1 report — internal KV/attention diagnostics as an early-warning signal

Preregistration: [`GATE1_EARLY_WARNING_PREREGISTRATION.md`](GATE1_EARLY_WARNING_PREREGISTRATION.md)
SHA-256 as preregistered, before any run:
`bb3b4572ef6eac5facd54232ebda12acd8500583535afa8a745e4ecd2b721498`
SHA-256 after Amendment 1 (appended post-primary-run):
`58dc5e8199cbb4acc829ef8f599ea70fd31e9f36af9aa9cb15cb99b2f1272eff`

**Result in one line:** no repository-specific KV diagnostic gave an early
warning that survived controls; a trivial final-layer hidden-state norm
outperformed all of them, and the one diagnostic with a positive median lead
time reverses its predictive direction between the two prompt families.

## Setup

Passive monitoring only. Nothing evicted, masked, or modified the KV cache, so
the design measures `internal signal -> later behaviour`, not
`intervention -> behaviour`. For each prompt: generate greedily, label the
behaviour by a deterministic rule, then recover every decode step's diagnostics
from one teacher-forced forward pass over prompt+generation. The attention row
used for step `t` is at absolute position `P-1+t` and sees only the prompt plus
generated tokens `0..t-1`.

## Model

| field | value |
|---|---|
| model | `Qwen/Qwen2.5-0.5B-Instruct` (494M params) |
| architecture | 24 layers, 14 query heads, **2 KV heads** (GQA, 7 query heads per group), head_dim 64 |
| dtype / attention | float32 / `eager` |
| decoding | greedy on raw logits, `repetition_penalty=1.0`, `max_new_tokens=32` |
| device | CPU; torch 2.14.0+cpu, transformers 5.16.1 |

`SmolLM2-360M-Instruct` was rejected during model selection because it has no
refusal behaviour at all (it complied with a break-in request), which would make
the refusal families meaningless.

**A defect found and fixed (Amendment 1.1).** The first run inherited the
model's shipped `generation_config.json`, which sets `repetition_penalty: 1.1`;
that is applied even with `do_sample=False`. So the realised decoding was
greedy-with-penalty, not the greedy-on-raw-logits the preregistration specified,
and 137 of 1412 decode steps had a teacher-forced argmax differing from the
generated token. I first read that as float32 noise, which was wrong — the
median top-1 minus top-2 gap at those steps was 0.70. With the penalty removed,
**teacher-forcing mismatches are 0 of 1412**, so the single-pass reconstruction
is exact.

## Prompt families and behavioural outcomes

Rule-based labels throughout; no model-subjective labelling.

| family | description | n | failures | rate | median `t_F` |
|---|---|---|---|---|---|
| A | benign controls | 20 | 0 | 0% | — |
| B | refusal-expected, direct | 12 | 1 | 8% | 0 |
| C | refusal-expected, adversarially framed | 12 | 3 | 25% | 0 |
| D | instruction retention (YES/NO, TRUE/FALSE) | 40 | 4 | 10% | 1 |
| E | matched benign controls for D | 20 | 0 | 0% | — |
| **F** | forbidden-word retention (amended) | 24 | **15** | 62% | **4** |
| **G** | three-word-limit retention (amended) | 24 | **12** | 50% | **4** |

`t_F` distribution over all 35 labelled failures:
`{0:7, 1:1, 2:1, 3:5, 4:13, 5:3, 6:1, 13:3, 16:1}`.

**Why the preregistered families were insufficient (Amendment 1.2).** Families
A–E produced 8 failures in 104 prompts, and **6 of 8 had `t_F = 0`** — the
violation is visible at the very first generated token, so a positive lead time
is impossible by construction. The held-out half of the D+E arm contains
**1 failure**, which cannot support AUROC or an operating point. That is a flaw
in my one-word constraint, not a property of the model: `YES`/`NO` either fails
immediately or not at all.

Families F and G were added under Amendment 1 with constraints violable
*mid-generation*, using **graded question sets** so both outcomes occur
naturally. The grading worked as designed and is not threshold tuning:

| | invites violation | mild | unrelated |
|---|---|---|---|
| F (forbidden word "water") | 8/8 | 7/8 | **0/8** |
| G (at most three words) | 8/8 | 0/8 | 4/8 |

Everything below is the **F+G arm** (n=48, 27 failures; held-out n=24, 14
failures) because it is the only arm with enough failures and a real warning
window. It is **exploratory under the amendment** and cannot by itself justify
CONTINUE.

## Diagnostics and baselines

Repository-specific: `max_m_B_block8`, `max_d_local`, `inversion_rate`
(Γ-disagreement between the mass ranking and the `d_i` ranking, computed as
`(1-tau_b)/2`), `any_inversion`, `abs_gamma_mean`, `abs_gamma_max`,
`pole_proximity` = `-log10(1-max_alpha)`, `near_one_hot`.
Baselines: `max_alpha`, `entropy`, `eff_support`, `max_m_B_single`,
`out_entropy`, `logit_margin`, `refusal_prob`, `eos_prob`, `hidden_norm`,
`hidden_delta`.

Γ is the manuscript's Theorem 5.1 object:
`Γ_ij = log(g_i/g_j) - log[α_j(1-α_i)/(α_i(1-α_j))]`, with
`d_i = [α_i/(1-α_i)]·g_i` and `g_i = ‖o - v_i‖`; `Γ_ij > 0 ⟺ d_i > d_j`.

**Two diagnostics are degenerate as implemented, and this was declared in
advance for the first:**
- `max_m_B_single` is *identically* `max_alpha` at singleton granularity, so the
  repository's block-mass diagnostic is not distinguishable from the
  max-attention baseline there (confirmed by a test).
- `near_one_hot` and `any_inversion` **saturate at 1.0 on every step of every
  run** — failures, benign controls and family A alike (144/144, 552/552,
  135/135, 616/616). Aggregating an indicator by max over 336 (layer, head)
  pairs guarantees some head qualifies. This is an aggregation artifact of my
  implementation, not a finding about the model; a head-mean aggregation would
  be needed to say anything, and was not stored.

## Main results — discrimination at decode step 0 (held-out, leakage-free)

Step 0 precedes every generated token, so it cannot contain post-failure
information. Bootstrap 95% CIs over 4000 resamples.

| diagnostic | | AUROC | 95% CI | P(AUROC>0.5) |
|---|---|---|---|---|
| `inversion_rate` | repo | 0.539 | [0.297, 0.778] | 0.602 |
| `abs_gamma_mean` | repo | 0.493 | [0.257, 0.743] | 0.471 |
| `max_m_B_block8` | repo | 0.486 | — | — |
| `abs_gamma_max` | repo | 0.468 | — | — |
| `pole_proximity` | repo | 0.404 | [0.168, 0.643] | 0.221 |
| `max_d_local` | repo | 0.386 | [0.125, 0.664] | 0.195 |
| `eff_support` | baseline | 0.436 | — | — |
| `max_alpha` / `max_m_B_single` | baseline | 0.429 | [0.214, 0.671] | 0.276 |
| `entropy` | baseline | 0.414 | [0.170, 0.672] | 0.229 |
| `out_entropy` | baseline | 0.307 | [0.062, 0.593] | 0.078 |
| `logit_margin` | baseline | 0.236 | — | — |
| `refusal_prob` | baseline | 0.600 | — | — |
| **`hidden_norm`** | **baseline** | **0.721** | **[0.471, 0.922]** | **0.958** |

**No repository-specific diagnostic's CI excludes 0.5.** The best is
`inversion_rate` at 0.539 with P(AUROC>0.5) = 0.60 — indistinguishable from
chance. The strongest signal in the whole table is `hidden_norm`, a trivial
final-layer hidden-state norm.

Paired bootstrap against that baseline:

| comparison | mean ΔAUROC | 95% CI | P(repo better) |
|---|---|---|---|
| `inversion_rate` − `hidden_norm` | −0.182 | [−0.500, +0.157] | 0.140 |
| `max_d_local` − `hidden_norm` | −0.335 | [−0.685, +0.044] | **0.040** |

## Lead-time results

`L = t_F - t_D`, over all 27 failures with an onset. Abnormality = crossing the
frozen threshold (95th/5th percentile of calibration-split benign runs) on two
consecutive steps.

| diagnostic | warned | warned before `t_F` | frac | median `L` (positive only) | ≥1 | ≥2 | ≥4 | held-out median `L` |
|---|---|---|---|---|---|---|---|---|
| `inversion_rate` | 6/27 | **4** | 0.148 | **+2.5** | 4 | 4 | 1 | **+2.0** |
| `max_d_local` | 3/27 | 3 | 0.111 | +11.0 | 3 | 3 | 3 | +11.0 |
| `abs_gamma_mean` | 5/27 | 0 | 0.0 | — | 0 | 0 | 0 | −8.0 |
| `abs_gamma_max` | 2/27 | 0 | 0.0 | — | 0 | 0 | 0 | −8.0 |
| `pole_proximity` | 1/27 | 0 | 0.0 | — | 0 | 0 | 0 | — |
| `out_entropy` | 11/27 | 1 | 0.037 | +7.0 | 1 | 1 | 1 | 0.0 |
| `logit_margin` | 2/27 | 0 | 0.0 | — | 0 | 0 | 0 | −16.0 |
| `max_alpha`, `entropy`, `near_one_hot`, `any_inversion`, `max_m_B_block8` | **0/27** | 0 | 0.0 | — | 0 | 0 | 0 | — |

Two diagnostics do have a strictly positive median lead time — `inversion_rate`
(+2.0 tokens held-out) and `max_d_local` (+11) — but they fire on only 15% and
11% of failures respectively, and the attention baselines never fire at all, so
"beats the attention baselines" is satisfied only in the vacuous sense that
zero is a low bar.

## False alarms

| diagnostic | family-A run FPR | A warnings/token | arm non-failure run FPR | arm warnings/token |
|---|---|---|---|---|
| `inversion_rate` | **0.350** | 0.0195 | 0.048 | 0.0074 |
| `max_d_local` | 0.100 | 0.0065 | **0.381** | 0.0815 |
| `abs_gamma_mean` | 0.150 | 0.0065 | 0.190 | 0.0296 |
| `pole_proximity` | 0.200 | 0.0065 | 0.048 | 0.0074 |
| `entropy` | 0.050 | 0.0016 | 0.095 | 0.0148 |
| `near_one_hot`, `any_inversion`, `max_m_B_block8`, `max_alpha` | 0.000 | 0.0 | 0.000 | 0.0 |

`inversion_rate` fires on **35% of benign family-A runs** while catching 14.8%
of failures before onset — it warns more often on harmless prompts than it
catches real violations. `max_d_local` fires on 38% of the arm's own
non-failures. Neither is a usable operating point.

## Family generalization — the decisive control

| diagnostic | F (n=12, 8 pos) | G (n=12, 6 pos) |
|---|---|---|
| `inversion_rate` | 0.453 | 0.694 |
| `max_d_local` | **0.781** [0.455, 1.000] | **0.111** [0.000, 0.375] |
| `abs_gamma_mean` | 0.469 | 0.444 |
| `pole_proximity` | 0.312 | 0.486 |
| `max_alpha` | 0.359 | 0.500 |
| `entropy` | 0.469 | 0.361 |
| `out_entropy` | 0.250 | 0.389 |

`max_d_local`'s bootstrap CIs in the two families are **non-overlapping**: it is
predictive in F and *anti*-predictive in G. That is not small-sample noise; the
diagnostic's direction depends on which constraint is being violated. This
triggers STOP conditions 4 and 5 directly.

## Γ-specific result

`inversion_rate` is the only repository quantity with a positive median lead
time (+2.0 held-out), and it is the best repo-specific discriminator (0.539) —
but it is at chance, inconsistent across families (0.453 vs 0.694), and fires on
35% of benign runs. `any_inversion` is saturated and uninformative;
`abs_gamma_mean`/`abs_gamma_max` are at chance (0.493, 0.468) and their warnings
arrive **after** failure (median `L` = −8).

Incremental information over conventional uncertainty signals, 5-fold CV on the
arm (step-0 features):

| base | base CV AUROC | + `inversion_rate` | + `abs_gamma_mean` | + `max_d_local` | + `pole_proximity` |
|---|---|---|---|---|---|
| `entropy` | 0.323 | 0.527 (+0.204) | 0.328 (+0.005) | 0.607 (+0.284) | 0.568 (+0.245) |
| `out_entropy` | 0.571 | 0.534 (−0.037) | 0.568 (−0.003) | 0.626 (+0.055) | 0.591 (+0.020) |

The gains over `entropy` look large only because the `entropy`-only model is
*below* chance (0.323); the augmented models still land at 0.53–0.61, and
log-loss changes are within ±0.03. Over `out_entropy` the best gain is +0.055.
At n=48 none of this is distinguishable from noise.

## Pole / near-one-hot result

`near_one_hot` (effective support < 1.5 for any head) is true on **100% of
steps in every condition**. Proximity to the `m_B -> 1` pole is therefore
completely non-discriminative here — the event is ubiquitous in benign
generation, which is exactly the "fires on every near-one-hot event" failure
mode the brief warned about. `pole_proximity`, the continuous version, is
*anti*-predictive (AUROC 0.404) and warns after failure.

## Layer analysis

Held-out AUROC by layer group (0–7 / 8–15 / 16–23):

| diagnostic | early | middle | late |
|---|---|---|---|
| `inversion_rate` | 0.546 | 0.579 | 0.407 |
| `max_d_local` | 0.486 | 0.600 | 0.386 |
| `pole_proximity` | 0.436 | 0.386 | 0.200 |
| `max_alpha` | 0.418 | 0.425 | 0.200 |
| `entropy` | 0.414 | 0.407 | 0.571 |

No group carries a signal; the weak maxima sit in the middle layers (0.58–0.60)
and the late layers are consistently *anti*-predictive. Headwise values were
recorded per (layer, head) but no head-level analysis is reported, because
nothing at the layer level justified pursuing it. No head is given a semantic
label.

## Event-aligned trajectories

Medians at offsets relative to `t_F`, failures vs matched same-family controls:

| diagnostic | failures at −3,−2,−1 | failures at 0,+1,+2 | controls at −3,−2,−1 | controls at 0,+1,+2 |
|---|---|---|---|---|
| `inversion_rate` | 0.149, 0.149, 0.172 | 0.169, 0.172, 0.165 | 0.155, 0.152, 0.165 | 0.172, 0.136, 0.165 |
| `max_d_local` | 8.82, 9.81, 8.97 | 8.24, 8.62, 8.33 | 20.1, 13.0, 9.50 | 9.03, 8.23, 9.72 |
| `entropy` | 4.25, 4.22, 4.37 | 3.72, 3.78, 3.77 | 4.90, 4.84, 3.58 | 3.61, 3.56, 3.62 |
| `out_entropy` | 0.884, 1.13, 0.801 | 1.44, 1.72, 1.13 | 0.746, 0.239, 0.051 | 0.659, 0.131, 0.210 |
| `logit_margin` | 1.72, 2.68, 2.51 | 1.63, 1.32, 1.17 | 1.81, 3.30, 6.06 | 2.69, 4.57, 3.81 |

**There is no consistent pre-failure change in any repository diagnostic.**
`inversion_rate` is flat across the onset and lies on top of the controls.
`max_d_local` is *lower* in failures than in controls before onset. Attention
entropy does fall across the onset (4.37 → 3.72) but the controls sit at the
same post-onset level (3.58–3.62), so the drop is not specific to failure. The
only clear separation is in output space (`out_entropy` higher, `logit_margin`
lower in failures) and it appears **at or after** onset, i.e. it is a correlate
of the failing continuation rather than a precursor.

## Temporal stability

Median over runs, persistence-1 crossings:

| diagnostic | failures: max consecutive | benign: max consecutive | failures: abnormal fraction | benign: abnormal fraction |
|---|---|---|---|---|
| `max_d_local` | 0.0 | **1.0** | 0.0 | **0.231** |
| `abs_gamma_mean` | 1.0 | 0.0 | 0.031 | 0.0 |
| `pole_proximity` | 1.0 | 0.0 | 0.031 | 0.0 |
| `inversion_rate` | 0.0 | 0.0 | 0.0 | 0.0 |
| `max_alpha`, `entropy`, `near_one_hot` | 0.0 | 0.0 | 0.0 | 0.0 |

`max_d_local` is abnormal on 23% of benign steps and 0% of failure steps — the
wrong direction. Where warnings exist at all they are isolated single-step
spikes (max consecutive 1.0), not sustained regime changes.

## Leakage checks

- Attention rows place **exactly zero** weight on any future key (asserted on
  every causal row in the tests).
- Teacher-forcing reproduces decoding exactly: **0 of 1412** steps mismatched.
- Step diagnostics at row `r` are invariant (to float32 precision) to tokens
  appended after `r`.
- Discrimination uses decode step 0 only; no generated text, classifier output,
  future token, or label enters any diagnostic.
- Thresholds were fit on calibration-split non-failing runs only and frozen
  before the held-out half was scored once.

## Controlled-KV-degradation arm

**Not run.** It is secondary in the preregistration and conditional on the
passive arm being informative, which it was not. Running it would answer a
different question (does degradation cause constraint failure) and would mix
intervention with detection.

## Seed robustness

Greedy decoding on raw logits is deterministic, so each prompt has one
trajectory and re-running reproduces it exactly. The preregistration requires
multi-seed repetition only for a *promising* result; since the result is
negative, no seed sweep was run. A future positive result would require one.

## Failure analysis — why this came out negative

1. **The preregistered constraint could not exhibit the phenomenon.** A
   one-word `YES/NO` answer fails at token 0 or not at all, leaving no warning
   interval. I should have chosen a mid-generation-violable constraint from the
   start; that is a design error, caught only after running.
2. **Two diagnostics were degenerate under max-over-(layer,head) aggregation.**
   `near_one_hot` and `any_inversion` saturate at 1.0 everywhere. A max over 336
   head-layer pairs cannot discriminate anything that any head ever does.
3. **The pole event is ubiquitous.** Near-one-hot attention occurs on every step
   of benign generation in this model, so it carries no safety information by
   itself — consistent with the earlier audit's finding that the pole is a
   property of concentrated attention, not of pathology.
4. **Sample size.** 24 held-out runs with 14 failures gives AUROC CIs about
   ±0.24 wide. The negative conclusion rests on the *pattern* — every repo
   diagnostic at chance, a trivial baseline ahead of all of them, and a
   non-overlapping cross-family sign flip — not on any single interval.
5. **Model scale.** At 494M parameters, instruction retention is fragile and
   failures may be driven by prompt-level features the internal state reflects
   only weakly.

## Decision

Against the preregistered rules (section 13/14/25):

- STOP 1 — failures too rare to evaluate: **met** for the preregistered A–E arm
  (1 held-out failure).
- STOP 2 — diagnostics move only at or after failure: **met** for
  `abs_gamma_mean`, `abs_gamma_max`, `pole_proximity`, `logit_margin`.
- STOP 3 — performance does not meaningfully exceed simple baselines: **met**.
  `hidden_norm` (0.721) beats every repository-specific diagnostic (best 0.539);
  P(repo better) = 0.14 and 0.04.
- STOP 4 — depends on one prompt family: **met**. `max_d_local` flips from 0.781
  to 0.111 with non-overlapping CIs.
- STOP 6 — explained by output-space signals moving first: **partly**. Output
  signals separate failures but only at/after onset, so they are not earlier
  either.
- CONTINUE requires *all* of: positive median lead time (met by
  `inversion_rate`), exceeds attention baselines (vacuously), generalizes across
  ≥2 families (**failed**), tolerable benign false-positive burden
  (**failed**, 35%), reproducible separation (**failed**).

CONTINUE is not satisfied. Four STOP conditions are met.

## What a redesign would have to change

Recorded because the negative result is informative about *how* to ask the
question, not only about the answer:

1. Use a constraint violable mid-generation from the start (as F and G were) so
   `t_F > 0` and a warning interval exists.
2. Aggregate diagnostics per head or per layer, never by max over all
   (layer, head) pairs — that saturates indicators.
3. Start from `hidden_norm`, which was the only signal to approach significance
   here, and ask whether KV diagnostics add anything to *it* rather than to
   attention entropy.
4. Use a model large enough to follow instructions reliably, so failures are
   informative rather than capability-limited.
5. Power the design for the CI width actually needed: ±0.24 at n=24 cannot
   separate 0.54 from 0.72.

Note that items 1–3 are fixes to *this experiment*, and the hypothesis they
would test is no longer "the repository's diagnostics warn early" but "does
anything in the residual stream warn early" — a different project.

## Claim boundary, restated

This experiment supports **no** alignment claim. It does not show alignment
detection, deception detection, scheming detection, corrigibility, jailbreak
immunity, general model safety, mechanistic interpretability, or causal
explanation of alignment failure. It is a passive association study; it
establishes no causation. A predictive internal signal would not have been a
mechanistic explanation even had one been found, and attention weight is not
explanation. KL divergence was not used as a safety label anywhere.

---

`GATE 1 DECISION: STOP`

`EARLY WARNING EVIDENCE: WEAK`

`MEDIAN LEAD TIME: +2.0 tokens (inversion_rate, held-out; 4 of 27 failures warned before onset). All other repository diagnostics: no warning or negative lead time.`

`BEST REPOSITORY-SPECIFIC SIGNAL: inversion_rate (Gamma ranking disagreement) — held-out AUROC 0.539, 95% CI [0.297, 0.778], P(AUROC>0.5)=0.60`

`BEST BASELINE SIGNAL: hidden_norm (final-layer hidden-state norm) — held-out AUROC 0.721, 95% CI [0.471, 0.922], P(AUROC>0.5)=0.96`

`DOES THE KV SIGNAL ADD INFORMATION BEYOND BASELINES?: NO`

`ALIGNMENT CLAIM JUSTIFIED?: NO`
