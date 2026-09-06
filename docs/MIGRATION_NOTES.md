# Migration notes

Source directory (read-only, untouched):
`~/Desktop/Token Research/files (29)/` — **not** `files (29) 2/`, which is
protected reference material and was never opened for writing.

No original file was modified, renamed, or deleted. Everything below is a
new file in this repository; the old files remain in place as provenance.

| Source | Destination | How | Semantic change |
|---|---|---|---|
| `ablate.py::delta_from_block`, `require_normalized`, `InputNormalizationError` | `src/kv_efficiency/local_algebra.py` | Reimplemented in numpy (source was torch-only) | None to the algebra; numpy chosen so the module is model-independent per the milestone spec ("the test should not depend on Pythia") |
| `ablate.py::delta_from_block`, `require_normalized`, `kl_divergence`, `tier_a_perturbation` | `src/kv_efficiency/eviction.py`, `src/kv_efficiency/damage_metrics.py` | Copied near-verbatim (torch) | `tier_a_perturbation`'s hook logic factored into `hooks.py::additive_hook`; no formula changed |
| `scratch/gate_test.py::GPTNeoXAdapter`, `head_values_dmodel` | `src/kv_efficiency/attention_reconstruction.py` | Copied near-verbatim | None |
| `scratch/gate_test.py::tier_b_eviction` | `src/kv_efficiency/eviction.py::tier_b_eviction` | Copied, generalized `block: slice` to accept an arbitrary boolean mask (needed for Phase 10's multi-position eviction; slices still work identically -- see `tests/test_eviction.py::test_delta_from_block_accepts_boolean_mask`) | Interface generalization only; the identity itself is untouched and re-verified |
| `scratch/run_pilot.py::phase3_smoke_test` | `experiments/validate_local_identity.py` | Restructured, same checks, numbers re-measured (not hard-coded) | None to the checks themselves |
| `scratch/run_pilot.py::phase4_f10_pilot` | `experiments/reproduce_pilot.py` | Same model/prompts/layers/heads/block construction/seed | None -- reproduction, not redesign |
| `falsification.yaml` (F10 definition) | referenced, not copied | `reproduce_pilot.py` reports the exact F10 statistic and `fires_when` string as data, does not reimplement the YAML | None |

New code with no old-project counterpart (did not exist anywhere in
`files (29)/`, including as an unimplemented reference): `policies.py`
(eviction ranking policies), `eviction_curve.py` (the removal-fraction-vs-
damage experiment), `coarse_vs_fine_analysis.py`.

No silent rewrites: every formula in `local_algebra.py` and `eviction.py`
is covered by a test that cross-checks it against either direct
recomputation (`test_local_algebra.py`) or the sibling implementation
(`test_eviction.py`).
