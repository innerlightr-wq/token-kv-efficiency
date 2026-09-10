---
name: final-review-changes
description: Every substantive change made to token_kv_eviction.tex during the final external-style review round, and why.
---

# Final External-Style Review — Changes Made

Scope: this round performed a skeptical external-reviewer audit (novelty, statistics,
reproducibility, causal language, theory edge cases, architecture-generalization framing,
figures, abstract/title/conclusion) against `manuscript/token_kv_eviction.tex`,
`manuscript/CLAIM_LEDGER.md`, and `manuscript/figures/FIGURE_PROVENANCE.md`. Per the
round's instruction, the manuscript was modified **only** where the audit exposed a real,
addressable gap — not stylistically rewritten. Three such gaps were found; all three are
listed below. No new figures were added (none were needed), no experiments were run, no
existing numerical claim, theorem statement, or interpretation was changed.

## Change 1 — Added a Reproducibility paragraph (end of Section 5 / `sec:design`)

**Gap found (Part IV, reproducibility attack):** the manuscript cited `results/*/`
directories throughout but never stated (a) which exact HF checkpoints/revisions were
loaded, (b) the seed used for every stratified draw and bootstrap resample, or (c) how a
reader maps a cited `results/` directory to the script that produced it. A reviewer
cloning the repository would have had to explore ~14 experiment scripts and several stored
JSON provenance blocks to reconstruct this by hand.

**Fix:** added a `\paragraph{Reproducibility.}` block (set in `\small`, after the existing
adapter-validation/grid paragraphs) stating: Pythia's exact HF id and a 12-character git
revision prefix (verified against the full 40-character SHA recorded in
`results/full_grid_mechanistic_analysis/summary.json`'s `provenance.model_revision`
field); that GPT-2 was loaded unpinned (`gpt2`, verified from
`experiments/cross_architecture_full_grid.py`'s `MODEL_ID` constant) — stated honestly as
an asymmetry versus Pythia, not concealed; the fixed seed `20,260,910`, verified present in
`experiments/downstream_propagation_sample.py`, `experiments/cluster_aware_reanalysis.py`,
and `experiments/state_dependent_propagation_audit.py`; the fact (verified directly, not
assumed) that every `results/<name>/` directory cited in the paper has a correspondingly
named `experiments/<name>.py` script; and a pointer to `configs/milestone2_grid.yaml` for
exact prompt texts and grid configuration. No new experiment was run to produce this
paragraph — every fact in it was read from already-existing code/result files.

## Change 2 — Added a "not distributional evidence" clause to Figure 3's caption

**Gap found (Part VIII, figure attack):** the round's brief explicitly required ensuring
"Figure 3's representative examples cannot be mistaken for distributional evidence." The
existing caption described the selection rule precisely but never stated outright that the
four plotted trajectories are illustrative single cases rather than a distributional claim.

**Fix:** appended one sentence to the Figure 3 caption: "These four trajectories are
illustrative single examples ($n=1$ per category), not distributional evidence; the
distributional claims (flip rates and their uncertainty) are Table 1, not this figure." No
change to the figure image itself (`fig3_downstream_trajectories.pdf/png` were not
regenerated — the caption is separate LaTeX text, not baked into the image).

## Change 3 — Added a "on the simplicity of Theorem 5.1's proof" paragraph (opening of Discussion)

**Gap found (Part II, novelty attack — "could a reviewer argue $\Gamma$ is algebraically
trivial?"):** this is a real, credible objection (the proof genuinely is two lines of
algebra), and the manuscript had no place where it explicitly owned this rather than
leaving a skeptical reader to raise it unprompted.

**Fix:** added a short paragraph at the start of Section 9 (Discussion) that concedes the
proof's simplicity directly, states that depth of derivation is not the claimed
contribution, and relocates the contribution explicitly to "converting a question that
could otherwise only be answered empirically... into a closed-form condition that can be
checked exactly and at scale," citing the combined $482{,}787$-pair verification count
(computed as $55{,}971 + 426{,}816$, both figures already stated elsewhere in the paper —
no new number introduced) as the concrete payoff of that conversion.

## Non-changes (audited, found already compliant — no edit made)

- **Stratified-sample-vs-population-prevalence wording (Part III):** searched every
  occurrence of "prevalence," every DPH/FLH rate, Table 1, the abstract, Discussion, and
  Conclusion. The word "prevalence" is never applied to a stratified-sample rate; every
  DPH/FLH rate is scoped to "the stratified sample" / "of 250 such cases" either in the
  same sentence or in the immediately preceding paragraph (§7.2), which already states the
  mandatory distinction verbatim. No change needed.
- **Causal-overclaim language (Part V):** re-grepped the full manuscript, including all
  newly-added figure captions and appendix text from the prior figure-generation round, for
  "optimal," "universal," "chaotic," "unpredictable," "explains the flip/reversal,"
  "practically significant," "causal importance," etc. Every hit is inside an explicit
  negation. No change needed.
- **Theory edge cases (Part VI):** re-checked Theorem 4.1's $m_B < 1$ hypothesis and
  Theorem 5.1's $0 < \alpha_i < \alpha_j < 1$, $g_i, g_j > 0$ hypotheses against every place
  either theorem is invoked in the empirical sections; found no invocation outside the
  stated hypotheses, and the existing edge-case paragraph (equal weights, $\alpha_i \to
  0,1$, $g_i=0$) already covers the degenerate cases correctly. No counterexample to any
  theorem statement as worded was found. No change needed.
- **"Cross-architecture" framing (Part VII):** the phrase appears exactly once in the
  manuscript (introduction, describing the *evidence*, not the *claim*), and the
  "What this paper is not" paragraph already explicitly disclaims "a universal theorem
  about transformers in general." No change needed.
- **Title/abstract/conclusion strength (Part IX):** compared the current title, abstract,
  and conclusion against this round's prescribed ceiling ("exact local deletion effects can
  be derived analytically... but the eventual winner is not determined by the local ranking
  alone"). The existing text matches this ceiling — it does not exceed it anywhere and does
  not fall short of stating the finding plainly. No change needed.
- **`manuscript/CLAIM_LEDGER.md`:** reviewed against the current manuscript. No new
  empirical or theoretical claim was introduced this round (all three changes above are
  documentation/framing additions, not new evidence), and no `\section`/`\subsection` was
  added or removed (only `\paragraph`-level insertions), so every section-number reference
  in the ledger remains accurate. **No changes made to the claim ledger.**

## What was deliberately left alone

Per this round's explicit instruction ("do not stylistically rewrite a stable manuscript
merely because alternative wording is possible"), several points raised in the simulated
review (`EXTERNAL_REVIEW_SIMULATION.md`) were recorded as reviewer concerns but **not**
acted on, because they are matters of emphasis or scope rather than defects: whether
Theorem 5.1 should be downgraded from "Theorem" to "Proposition" given its simple proof;
whether the DVG negative result should carry an explicit "does not rule out other
featurizations" caveat; whether DVG's AUC point estimates should carry confidence
intervals; and whether exact library versions should be surfaced in the main text rather
than left in the repository's provenance files. These are noted for a future round, not
acted on here.
