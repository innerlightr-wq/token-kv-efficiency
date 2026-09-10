---
name: draft1-audit-report
description: Compilation and publication audit of manuscript/token_kv_eviction.tex Draft 1 — every substantive change made, and the reviewer-style coherence audit findings.
---

# Draft 1 — Compilation and Publication Audit Report

Scope: compile the manuscript, fix LaTeX/table/reference/notation defects, run a
reviewer-style coherence audit, make only necessary revisions, and determine
what figures still need generating. No new model experiments, no new
hypotheses, no alteration of existing experimental results. Nothing
committed or pushed.

## Part I — Freeze and inspect

- `git status`: working tree had pre-existing modifications from before this
  session's manuscript work (`MILESTONE_STATUS.md`, `docs/RESEARCH_STATUS.md`,
  `experiments/multistratum_eviction.py`, `results/milestone2_multistratum/summary.json`,
  `src/kv_efficiency/eviction.py`, `tests/test_eviction.py` — all pre-existing,
  not touched this round) plus a large set of untracked files from prior
  rounds (experiments, results directories, `manuscript/`). None of these
  were altered in this round except inside `manuscript/`.
- Test suite before editing: **74/74 passed**.
- LaTeX toolchain: no `pdflatex`/`latexmk`/`xelatex` on `PATH`, but
  **`tectonic` 0.17.0 was already installed via Homebrew** — used for all
  compilation in this round.
- Read the complete manuscript (`token_kv_eviction.tex`, 776 lines) and the
  complete claim ledger (`CLAIM_LEDGER.md`) before making any change.

## Part II — Compilation

Compiled with `tectonic --keep-logs -o build token_kv_eviction.tex`.

- **Result: succeeds cleanly.** No LaTeX errors, no undefined references, no
  duplicate labels, no bibliography errors, no missing packages (`tectonic`
  fetches packages on demand from its bundle).
- First compile: several overfull `\hbox` warnings up to **66.9pt** (design
  section) and **60.5pt** (falsification-summary table + caption), caused by
  long unbroken `\texttt{path/to/file}` strings (no internal break points)
  and a table column layout too wide for its content.
- **Fixes applied** (Part III/XV below has the itemized list): switched all
  long file-path references from `\texttt{}` to `\url{}` (via
  `\usepackage[hyphens]{url}` + `\urlstyle{tt}`, which permits line breaks at
  `/` and `_`), added `\usepackage{microtype}`, converted the
  falsification-summary table to `p{}`-width columns, wrapped the two dense
  appendix sections in `\small`, and shortened a few appendix sentences that
  stacked 3–4 long paths in one clause.
- **Final compile**: worst remaining overfull box is **14.9pt** (~0.2in,
  appendix figure-2 source list) and one harmless underfull-badness table
  cell; one benign `` `h' float specifier changed to `ht' `` notice (normal
  LaTeX behavior, not touched, per the instruction not to change content to
  silence harmless warnings).
- **Output: `manuscript/token_kv_eviction.pdf`, 13 pages**, copied from
  `manuscript/build/token_kv_eviction.pdf`.

## Part III — Equation and notation consistency

Audited every occurrence of $o, v_i, \alpha_i, d_i, g_i, G_{ij}, M_{ij},
\Gamma_{ij}, \Delta_B, r_B, m_B$.

- All symbols are defined before their first substantive use ($o, v_i,
  \alpha_i, m_B, o_B, r_B$ in §4.1's Setup before Theorem 4.1; $d_i$ in
  §4.3's Definition; $g_i, G_{ij}, M_{ij}, \Gamma_{ij}$ in §5 immediately
  before/within Theorem 5.1).
- Norm choice ($\|\cdot\|$ = Euclidean/$L_2$) is stated explicitly where
  $d_i$ is defined ("exact Euclidean norm of the change...").
- "Exact local output perturbation" is used consistently for $d_i$
  throughout (verified by search — no place uses a different term for the
  same object).
- KeepKV's sign convention was independently re-derived and shown identical
  to ours in §4.2, not merely asserted.
- Block deletion (Theorem 4.1, general $B$) and singleton deletion (§4.2's
  specialization) are kept visibly distinct: the singleton case is
  introduced explicitly as "For a single evicted token $i$ ($B=\{i\}$)...",
  never silently substituted for the general case.
- **No notation defect found requiring correction.**

## Part IV — Reviewer-style claim audit

Read end-to-end as a skeptical claim-by-claim reviewer. Findings:

- Every strong claim in the main text is traceable to a row in
  `CLAIM_LEDGER.md` with a matching status (THEOREM / COMPUTATIONAL /
  REPLICATED COMPUTATIONAL / NEGATIVE RESULT / INTERPRETATION / LIMITATION).
- Every conditional claim (architecture, sample, numerical threshold, policy
  comparison) already states its condition in the same sentence or the
  immediately following one — e.g. "measurable... means only 'above the
  configured numerical threshold,' not necessarily practically large" and
  "this paper does not evaluate [task-quality]" recur at every point where a
  reader could over-generalize.
- **No claim required strengthening or weakening.** Two clarity gaps were
  found and fixed (not evidence problems, prose-completeness problems — see
  Part IX).

## Part V — Central distinction check

"Exact immediate perturbation $\neq$ guaranteed downstream consequence" is
preserved in the abstract, introduction (the boxed question/answer), every
results section, the falsification section, the discussion, and the
conclusion. Specifically verified absent: "the local score is globally
wrong," "attention mass is invalid," "the downstream winner is
unpredictable," "state dependence explains the reversals," "local damage is
optimal." (Full grep results in Part XV below.)

## Part VI — SDPH correction audit

Re-read every SDPH paragraph (§9.4, the Discussion paragraph referencing it,
the falsification summary table row). **All three already state the correct
interpretation**: state-dependence is described as "a confirmed, causally
demonstrated phenomenon, but a falsified candidate explanation for winner
reversal," with the disconfirming numbers (flip/stable median-$S$ ratio
0.2856, architecture-inconsistent pattern) given explicitly and the sentence
"This finding must not be read as an explanation for the DPH/FLH reversals"
stated before the numbers, not after. **No correction was needed here** —
this was verified, not assumed, by re-reading the full subsection this
round.

## Part VII — Statistical audit

Confirmed present and correct in §8: primary cluster =
architecture$\times$prompt$\times$layer$\times$head; 10,000 bootstrap
replicates; DPH pooled flip 0.324 CI [0.261,0.392]; Pythia 0.380 CI
[0.264,0.522]; GPT-2 0.287 CI [0.216,0.360]; architecture-difference CI
[-0.045,0.252] (crosses zero, stated as such); paired FLH-vs-DPH differences
explicitly described as using "the same resampled cells applied to both
analyses within each bootstrap replicate." The pair-count-vs-cluster
distinction is stated verbatim in §8.2. **No correction needed.**

## Part VIII — Abstract audit (revised)

The abstract was missing an explicit **limitation** clause (required
element 7) and carried more granular numbers than necessary (element list
asked to avoid excessive numbers). **Changed**: removed the inline
$>99\%$/$>98\%$ set-change/measurable-KL statistics from the abstract
(these remain in §7's body, which is the right place for them), kept only
the two truly headline numbers (3.55%/5.38% inversion prevalence, 32.4%
downstream reversal with its CI), and appended one sentence stating the
two-architecture/stratified-sample scope limitation and the no-deployment-claim
disclaimer. This is a wording change only — no evidentiary content changed.

## Part IX — Introduction audit (revised)

Two of the five required questions (Part IX's list) were answered only
implicitly. **Changed**: added one sentence explaining concretely *why*
attention-mass ranking can disagree with exact local perturbation (a token
can carry mass but sit close to the output, or carry little mass but sit
far from it), and one sentence stating explicitly what $\Gamma$ adds (it
makes the disagreement condition precise rather than leaving it as an
empirical observation). The KeepKV attribution paragraph ("Prior-art
correction stated up front") was already positioned before the Contributions
list — confirmed, not changed.

## Part X — Related work / bibliography audit

Re-verified against the existing `CLAIM_LEDGER.md` bibliographic-verification
table (all 8 entries were fetched live from arXiv abstract pages in the
prior manuscript-drafting round). **No changes needed**: KeepKV is correctly
cited as an arXiv preprint with an explicit "no venue acceptance stated"
note (a correction already made in the prior round, re-verified here);
CriticalKV and the Fixed-Contract Diagnostic paper are correctly cited as
arXiv preprints with no invented venue; H$_2$O, Scissorhands, StreamingLLM,
ROME, and Tuned Lens carry their verified venues (NeurIPS 2023 $\times$2,
ICLR 2024, NeurIPS 2022, arXiv preprint respectively).

## Part XI — Figure audit (see manuscript Appendix A)

All four figures can be generated from already-stored results with **no new
forward passes**. Appendix A was rewritten this round to specify, per
figure: source result files, panel count, axes/quantities, the exact
caption claim, and whether the panel is analytically schematic or
data-derived. Summary:

| # | Figure | Schematic or data-derived | Source files |
|---|---|---|---|
| 1 | Exact eviction geometry + $\Gamma=0$ boundary | Schematic (optional data inset) | None required; optional inset from `results/cross_architecture_full_grid/report.txt`'s strongest-pair record |
| 2 | Inversion→set-change→measurable-KL funnel | Data-derived | `results/full_grid_mechanistic_analysis/summary.json`, `results/cross_architecture_full_grid/summary.json` |
| 3 | Representative downstream trajectories | Data-derived | `results/downstream_propagation_audit/cases.json`, `results/downstream_propagation_sample/cases.json` |
| 4 | Falsification summary (DVG→DPH→FLH→SDPH) | Data-derived, summary-level | As in Table~\ref{tab:falsification}'s caption |

None were rendered in this round (no plotting was performed); this was a
specification pass only, as instructed.

## Part XII — Appendix depth

Added a new **Appendix B** ("Supplementary Material Not Reproduced in the
Main Text") explicitly listing what stays out of the main text and its
exact source file: full adapter validation detail, full architecture grids,
per-stratum heterogeneity, the 36-cell pilot grid (with its own prevalence
figure explicitly flagged as not a headline result), extended DVG feature
detail, FLH cross-tabs, SDPH linearity controls, and cluster-bootstrap
sensitivity analyses. The main text itself was not shortened (it was
already appropriately terse and pointer-based) — this is a new roadmap, not
a content migration.

## Part XIII — Pilot-grid audit

Confirmed the 36-cell GPT-2 pilot was **already** used only for adapter
validation (row-sum deviation, causal masking, QK recomputation, $c\_proj$
reconstruction, delete-renorm identity) and never as a headline
prevalence estimate — this was compliant before this round. **Added**, for
extra robustness against misreading: one explicit sentence in §6 stating
"This pilot grid is used exclusively for adapter validation in this paper;
its own inversion-prevalence figures are not used as a headline empirical
result anywhere in this manuscript," plus the pilot's actual number (6.62%)
recorded only in the new Appendix B, explicitly labeled as not a headline
result.

## Part XIV — Title and thesis check

Working title retained unchanged: *"Exact Local Output Perturbation in
KV-Cache Eviction: Ranking Inversions and Their Downstream Consequences."*
No clearly better conservative alternative was found; the title accurately
names the paper's two central objects (exact local perturbation, ranking
inversions) and its scope-limiting qualifier (downstream consequences,
not downstream guarantees).

## Part XV — Full change list (files modified/created this round)

**Modified:**
- `manuscript/token_kv_eviction.tex` — LaTeX/packaging fixes (`url`,
  `microtype`, `\urlstyle{tt}`); grammar fix ("not an implementation
  defect"); table restructured to `p{}` columns; long file paths converted
  `\texttt{}`→`\url{}` throughout; abstract rewritten (added limitation
  clause, trimmed granular numbers); introduction extended (two clarifying
  sentences); §6 pilot-grid disclaimer sentence added; Appendix A rewritten
  with full per-figure specification; new Appendix B added; added
  `\label{tab:falsification}` and replaced two hardcoded "Table~3"/"Figure
  3" references with proper `\ref{}` / descriptive text.
- `manuscript/CLAIM_LEDGER.md` — updated "Open items" section to record
  successful compilation and the new Appendix A/B content; no claim rows
  changed (no evidentiary content was altered).

**Created:**
- `manuscript/DRAFT1_AUDIT_REPORT.md` (this file).
- `manuscript/token_kv_eviction.pdf` (13 pages, compiled output).
- `manuscript/build/` (tectonic build artifacts: `.log`, `.pdf`; intermediate
  `.aux`/`.out` not kept per `tectonic`'s default behavior).

**Not modified:** no file outside `manuscript/` was touched. No experimental
result, no stored JSON/report, no source code, no test file was altered.

### Full prohibited-language grep (re-run after all edits)

```
grep -ni -E "proved experimentally|optimal[^i]|universal|architecture-independent downstream|explains the (flip|reversal)|attention is useless|better than attention mass|practically important|we solve|solves kv-cache|chaotic|unpredictable" token_kv_eviction.tex
```
All 6 hits are inside explicit negations ("not... optimal," "not a universal
theorem," "non-universal reversal" $\times$2, "not... chaotic or
unpredictable"). **Zero violations.**

## Part XVI — Final verdict

**Compile status:** succeeds cleanly with `tectonic`; 13 pages; 0 errors; 0
undefined references; 0 duplicate labels; 0 bibliography errors; remaining
warnings are cosmetic (max 14.9pt overfull hbox in an appendix file-path
list, one harmless float-placement notice).

**Files changed:** 2 modified (`token_kv_eviction.tex`, `CLAIM_LEDGER.md`),
3 created (this report, the PDF, the `build/` directory).

**Scientific corrections:** none — no claim's evidentiary content was
altered. All changes were compilation fixes, notation/cross-reference
repairs, and prose completeness additions (abstract limitation clause,
introduction clarifications, pilot-grid disclaimer, appendix detail).

**Citation corrections:** none new this round; the KeepKV
no-venue-acceptance correction was already made in the prior round and was
re-verified, not re-done.

**Figure-generation requirements:** all four figures can be built from
existing stored results with no new forward passes; Appendix A now
specifies exactly what each one plots. None were rendered this round.

**Appendix recommendations:** Appendix B's roadmap is in place; a future
round should decide how much of it to expand into rendered tables/prose
rather than file pointers, but the main text should stay as terse as it
currently is.

### PUB-A — Manuscript is structurally ready; next round should generate figures.

The manuscript compiles cleanly, every claim traces to the ledger, the
central local-vs-downstream distinction and the SDPH falsified-explanation
framing both survive the reviewer-style audit intact, all bibliographic
metadata is verified, and no evidentiary correction was required. The only
remaining substantive work before submission is generating the four planned
figures from already-stored data (Appendix A) and deciding how much of
Appendix B to expand.

74/74 tests pass (re-run at the end of this round, unaffected by
manuscript-only changes). Nothing committed, nothing pushed.
