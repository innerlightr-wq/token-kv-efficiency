# Alignment relevance

Written as a separate audit section because efficiency results and safety
results are different kinds of claim and the distance between them is easy to
elide. The short answer is that this repository makes no alignment claim, and
should not.

## What the repository currently claims

A repository-wide search for `alignment`, `safety`, `safe`, `robust`,
`reliable`, `interpretab*`, `hallucinat*`, `anomal*`, `monitor*`, `decept*`,
`oversight`, `faithful`, `verifier`, `risk` returns **no alignment or safety
claim anywhere**. Every hit is one of:

- an explicit **denial**: `docs/RESEARCH_STATUS.md` — "'Lean/Pythia verifies
  KV-cache safety' is not a phrase used or implied anywhere in this project.
  Damage tolerances (epsilon in the efficiency frontier) are exactly that —
  measured KL budgets — never described as 'safe.'"; and
  `manuscript/token_kv_eviction.tex` — "a universal theorem about transformers
  in general, not an AI-safety [claim]";
- an unrelated technical use of "alignment" meaning **vector cosine**
  (`experiments/directional_value_geometry.py`: `cos_vi_o`, `delta_cos_o`);
- `hallucinat*`, `anomal*`, `monitor*`, `decept*`, `oversight`, `verifier`:
  **zero occurrences**.

This is the correct state and the audit recommends no change to it. The
discipline is already in the repository; it does not need to be added.

## The three questions

### 1. Does this repository establish an alignment result?

**No.** It measures KL divergence at one token position on Pythia-160M and
GPT-2 under KV eviction. No safety-relevant property is defined, measured, or
compared.

### 2. Does it establish a safety result?

**No**, and there is a specific reason beyond "it was not attempted". The only
damage measure is aggregate KL over the vocabulary at the final position.
`beyondperplexity2024` (Findings of EMNLP 2024) shows that compression can
degrade safety-relevant behaviour in ways aggregate perplexity-style measures
do not reveal. A small KL is therefore not weak evidence of preserved safety
behaviour — it is *no* evidence about it, because the two are measured on
different axes. The repository's own refusal to call a KL tolerance "safe" is
supported by that literature.

### 3. Could it be useful to alignment researchers?

**Possibly, as infrastructure**, with two concrete pieces:

- The **exact** perturbation identity means an intervention's effect on a head
  output can be computed in closed form rather than estimated, and verified
  against direct recomputation to `1.1e-16`. That is a clean primitive for
  causal-intervention work of the kind `rome2022` and `tunedlens2023` do.
- The **counterfactual eviction harness** added by this audit
  (`experiments/counterfactual_eviction.py`) tests whether a token-importance
  score is causally predictive rather than merely correlated — the bar
  `jain2019` sets. On this repository's own metric both scores pass it
  (top-first eviction damages 4.0–6.6x more than bottom-first at matched
  budget).

Neither makes the repository an alignment contribution. Both are tools that
alignment work could use.

## Classification

### DIRECT ALIGNMENT RESULT — no

Would require evidence that the mechanism improves or measures a
safety-relevant property. None exists here.

### ALIGNMENT-RELEVANT INFRASTRUCTURE — partially, as potential

The exact-perturbation primitive and the counterfactual harness are usable by
safety research. No safety effect is demonstrated, so this is potential, not
achievement.

### PURE EFFICIENCY RESULT — this, with a caveat

Even "efficiency" overstates what is measured. The repository measures a
**quality-vs-retained-fraction tradeoff**, not efficiency: no wall-clock
latency, throughput, or measured memory is reported, `estimated_bytes_saved`
is an arithmetic projection from entry counts (and, per
`docs/METRIC_DEFINITIONS.md` §8, overstates realised eviction by 13.36%), and
the eviction is implemented as an additive delta on head outputs rather than
by compacting a cache — so no real memory is freed in any experiment.

**The repository is best classified as a DIAGNOSTIC / MECHANISTIC study of KV
eviction scoring, with PURE EFFICIENCY as the closest of the three offered
categories and no alignment content.**

## What the smallest genuine alignment experiment would be

Not invented here, and not run: the repository's architecture does not make it
lightweight, and `docs/RESEARCH_STATUS.md` correctly leaves it open. Stated so
that nobody has to guess what would count.

**Minimal design.** Take an instruction-following setup with a system prompt
carrying a constraint the model demonstrably obeys (a refusal, a format
requirement, or a negation). Then:

1. Establish the baseline behaviour rate on held-out prompts with the full
   cache — the constraint must be obeyed at a high, measured rate, or there is
   nothing to preserve.
2. Apply each eviction policy at matched **realised** budget (see
   `docs/METRIC_DEFINITIONS.md` §8 — the plan is not the realisation).
3. Measure **behavioural** compliance, not KL: does the model still refuse,
   still honour the negation, still respect the instruction hierarchy.
4. Report compliance against retained fraction per policy, and separately
   report whether system-prompt token positions were disproportionately
   evicted. Position 0 is protected here; positions 1..k of a system prompt
   are not.
5. Include a matched-budget random-eviction control and a top-first
   counterfactual arm, so a compliance drop can be attributed to *which*
   tokens were removed rather than to how many.

**What would make it a result.** A policy that preserves KL equally well but
compliance differently would be a genuine finding: it would show that the
KL-based damage tolerance the repository measures is not a proxy for
safety-relevant retention. A null — compliance tracks KL — would also be
worth reporting, and would justify a much narrower claim than "safe".

**Prerequisites this repository does not yet have:** a model that
instruction-follows reliably (Pythia-160M does not), a behavioural evaluation
harness, and real long-context tasks. This is a different project, not a
follow-up experiment.

## Claims that must not be made

- That a measured KL tolerance is a safety, robustness, or reliability
  guarantee. It is a divergence budget on one token's next-token distribution.
- That `d_B^local` measures the *importance* of a token to the model's
  behaviour. This audit establishes it is causally predictive of **this
  repository's own KL metric** within a stratum; that is the whole scope.
- That attention mass or local damage identifies safety-critical tokens.
  Untested.
- That preserving KL preserves instruction hierarchy, refusals, negations, or
  long-range constraints. Untested, and `beyondperplexity2024` gives positive
  reason to doubt it.
- That the work is interpretability. It exposes token-importance scores; it
  runs no attribution against a behavioural target. `jain2019` and
  `wiegreffe2019` set the bar, and the counterfactual test in this audit meets
  it only for the KL metric.
- That the efficiency result generalizes. Two small models, synthetic prompts,
  one measured position.

## Literature gap, stated honestly

A Crossref search across KV-cache compression combined with safety,
alignment, jailbreak, refusal, instruction fidelity, and prompt-injection
resistance returned **no work connecting KV-cache eviction specifically to
safety behaviour**. The nearest items are `beyondperplexity2024` (safety
evaluation of compressed *models*, not caches) and `promptinjrobust2024`
(instruction-following robustness under prompt injection, not compression).

That gap is real and is worth someone's attention. This repository does not
fill it, and the audit recommends against framing it as if it did.
