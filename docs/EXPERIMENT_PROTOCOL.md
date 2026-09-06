# Experiment protocol (audited from source, not memory)

Recovered from `~/Desktop/Token Research/files (29)/README.md`,
`ablate.py`, `scratch/gate_test.py`, `scratch/run_pilot.py`,
`falisification.yaml`, `preregistration.yaml` before writing any new code.
Where code and any remembered description disagreed, the code won; no
disagreements were found this round.

## The exact local identity

Verbatim from `files (29)/README.md` lines 9-11:

```
o = Σᵢ αᵢ vᵢ           (full head output)
m_B = Σ_{i∈B} αᵢ       (attention mass on block B)
o_B = Σ_{i∈B} αᵢ vᵢ    (block's raw contribution to head output)
r_B = o_B − m_B·o      (residual)
Δ_B = −r_B / (1 − m_B) (exact perturbation from evicting B and renormalizing)
d_B^local = ‖Δ_B‖ = ‖r_B‖ / (1 − m_B)
```

Implemented in `ablate.py::delta_from_block` (lines 127-145) as
`(m_B*o - o_B)/(1-m_B)`, algebraically identical to `-r_B/(1-m_B)`. Raises
`ValueError` when `m_B >= 1 - 1e-9` (eviction identity: the deleted-mass
"eviction identity, direct vs closed form"). **Epistemic status: EXACT
ALGEBRA.**

## Tier A / Tier B (verbatim from `ablate.py` module docstring)

- **Tier A**: perturb the head output at a single query position (the
  last), then rerun the whole forward pass (the docstring notes this is
  "correct but wasteful" -- a suffix-only optimization was never built).
  **Not eviction.**
- **Tier B**: true eviction -- remove the block from the head's KV for
  every query position and rerun. Implemented in
  `scratch/gate_test.py::tier_b_eviction`, new code (no generic Tier B
  routine existed elsewhere in the file set), derived exactly from GPT-NeoX's
  parallel-residual structure (`hidden_out = hidden_in + attn_output +
  mlp_output`, MLP independent of attention) -- see that file's docstring
  for the derivation.

## F10 (from `falsification.yaml`)

```
id: F10
name: tier_a_not_a_valid_proxy
statistic: median within-head spearman(tier_a_kl, tier_b_kl)
fires_when: < 0.7
```

Matches `preregistration.yaml`'s `gate.tier_a_is_valid_proxy`, threshold
0.7. **This gate is about Tier A KL vs. Tier B KL, not about `d_B^local`
directly** -- a distinction preserved throughout this repository (see
`experiments/coarse_vs_fine_analysis.py`, which separately tests
`d_B^local` vs. Tier B KL).

## KL damage computation

`ablate.py::kl_divergence`: `KL(p_full || p_compressed)` in nats, computed
in float64 from log-softmax, full model as the reference distribution.

## Model / hook points

`EleutherAI/pythia-160m` (GPT-NeoX architecture: 12 layers, 12 heads,
hidden=768, `use_parallel_residual=True`), loaded with
`dtype=torch.float32, attn_implementation="eager"` (eager required for
`output_attentions=True` to return faithful post-softmax weights). Hooks
attach to `model.gpt_neox.layers[l]` (whole decoder block, for Tier A/B
residual injection) and `model.gpt_neox.layers[l].attention.dense`
(forward-pre-hook, for validating manual QKV extraction against the real
per-head context).

**Local attention output vs. downstream output**: `d_B^local` is computed
from one layer's one head's `(alpha, values)` pair -- it never looks past
that layer. Tier A/B KL is measured at the model's final logits, after
every subsequent layer. The gap between these two measurement points is
exactly what F10 (and this repo's `coarse_vs_fine_analysis.py`) tests.

## Block selection (pilot)

`run_pilot.py::phase4_f10_pilot`: layers `[1,4,7,10]`, heads `[0,3,6,9]`,
`block_size=4`, block start positions via `np.linspace(1, n-block_size-1,
n_blocks_per_head=3)` (rounded to unique ints), always `>= 1` to avoid a
block that would consume all causal attention mass at position 0. Blocks
with `m_B >= 1-1e-6` or `<= 0` are skipped.

## Prompt / data handling

Two hand-written English paragraphs (Roman Empire, photosynthesis), no
corpus, no held-out set. `torch.manual_seed(0)`; single-threaded was used
in the old base-conda environment only as an OpenMP-conflict workaround
(see `docs/RESEARCH_STATUS.md`) -- not needed in this repository's isolated
venv.

## Reproducibility controls (as audited, not assumed)

- `torch.manual_seed(0)` before model load.
- `attn_implementation="eager"`, `dtype=torch.float32` fixed.
- No dropout (`model.eval()`).
- Determinism was itself a checked property (`8_determinism_err` in the
  old smoke test), not merely assumed -- reproduced in this repo's
  `experiments/validate_local_identity.py` (check C).

## Milestone 2 addition: `global_eviction` exactness scope

New code, no old-project counterpart. Combining evictions from multiple
`(layer, head)` strata in one forward pass is **exact** when the strata
share a layer (GPT-NeoX's parallel residual makes per-head contributions
additive and independent -- proved by test in
`tests/test_global_eviction.py::test_global_eviction_same_layer_two_heads_is_additive`)
and an **approximation** across layers (each layer's delta is computed
from its own baseline attention/hidden state; a later layer's hook does
not account for an earlier layer's eviction having already changed its
input). Documented in `eviction.py`'s module docstring; not silently
assumed exact.

## Result-artifact provenance map (checkpoint audit)

Milestone 2's `results/milestone2_*/summary.json` files embed a full
`provenance` block (`provenance.py`: config snapshot, seed, model,
package versions, timestamp). Milestone 1's four result files predate
that utility and do **not** embed seed/layer/head/model as JSON fields;
that information is fixed in code, not lost -- recorded here rather than
re-run to add it:

| Result file | Producing script | Model | Seed | Layer/head | Notes |
|---|---|---|---|---|---|
| `results/validate_local_identity.json` | `experiments/validate_local_identity.py` | `EleutherAI/pythia-160m` (in JSON as `model_id`) | `torch.manual_seed(0)` (line 52) | `layer, head = 3, 2` (line 67) | — |
| `results/pilot_reproduction/{summary,records}.json` | `experiments/reproduce_pilot.py` | `EleutherAI/pythia-160m` (`MODEL_ID`, line 31; in JSON as `model_id`) | `torch.manual_seed(0)` (line 62) | `layers`/`heads` recorded in JSON | 2 prompts hardcoded in script, not duplicated into JSON |
| `results/eviction_curve.json` | `experiments/eviction_curve.py` | `EleutherAI/pythia-160m` (`MODEL_ID`, line 32) -- **not recorded in the JSON itself** | `torch.manual_seed(0)` (line 69) | `LAYER, HEAD = 3, 2` (line 46, in JSON as `layer`/`head`) | model id recoverable only from script, not the artifact |
| `results/milestone2_multistratum/*.json` | `experiments/multistratum_eviction.py` | via `provenance` block | via `provenance` block | `configs/milestone2_grid.yaml` | complete |
| `results/milestone2_global_budget/*.json` | `experiments/global_budget_curve.py` | via `provenance` block | via `provenance` block | `configs/milestone2_grid.yaml` | complete |
