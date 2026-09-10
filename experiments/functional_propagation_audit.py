r"""Functional-Latent Hypothesis (FLH) audit.

FLH (hypothesis, not a claim): Euclidean residual distance is an imperfect
proxy for downstream functional damage. The eventual mass-vs-local winner
may become distinguishable earlier and more consistently when intermediate
states are projected through the model's final normalization and LM head
(the "logit lens") into logit/probability space.

**Reuses the exact same 250-case sample, same seed, no new population
draw.** Cases are read directly from
`results/downstream_propagation_sample/cases.json` (the frozen DPH sample:
100 Pythia + 150 GPT-2, seed 20260910) -- the (prompt, layer, head, i, j)
tuples and final mass/local labels are taken from there unchanged, giving
an exactly paired residual-vs-functional comparison per case.

**Reuses, unchanged, via direct import:** `get_stratum_data`,
`run_traced_forward`, `logit_lens`, `load_pythia`, `load_gpt2`, `n_layers`,
`DEPTH_MAX` from `experiments/downstream_propagation_audit.py` -- the exact
same hook-based intervention pipeline (never `output_hidden_states`,
matching the fix validated two rounds ago). `kl_divergence` and `logit_l2`
are reused unchanged from `kv_efficiency/damage_metrics.py`. No
intervention semantics are altered anywhere in this file.

**KL convention, fixed and documented once:** `kl_divergence(logits_full,
logits_compressed) = KL(p_full || p_compressed)` (ported verbatim in
damage_metrics.py from the original ablate.py; reference distribution
first). Applied here at every layer via the logit lens: `KL_policy(l) =
KL(p_ref(l) || p_policy(l))`. `Delta_KL_l := KL_mass(l) - KL_local(l)`;
positive means local is functionally closer to the reference at layer l --
the same sign convention as the true final `advantage = KL_mass - KL_local`
and as the residual-space `delta_resid` from the prior round.

Writes `results/functional_propagation_audit/{summary,cases}.json` and
`report.txt`. Does not modify `src/kv_efficiency/`, any test, any existing
result file (including the 250-case DPH sample), or any doc/status file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_dpa_spec = importlib.util.spec_from_file_location(
    "downstream_propagation_audit", ROOT / "experiments" / "downstream_propagation_audit.py"
)
dpa = importlib.util.module_from_spec(_dpa_spec)
_dpa_spec.loader.exec_module(dpa)

get_stratum_data = dpa.get_stratum_data
run_traced_forward = dpa.run_traced_forward
logit_lens = dpa.logit_lens
n_layers = dpa.n_layers
load_pythia = dpa.load_pythia
load_gpt2 = dpa.load_gpt2
DEPTH_MAX = dpa.DEPTH_MAX

from kv_efficiency.damage_metrics import kl_divergence, logit_l2  # noqa: E402

RESULTS_DIR = ROOT / "results" / "functional_propagation_audit"
SAMPLE_CASES = ROOT / "results" / "downstream_propagation_sample" / "cases.json"

# TOLERANCE, REVISED ONCE, FOR A DOCUMENTED AND DEBUGGED REASON --
# not tuned against the aggregate flip-rate/emergence results.
#
# First attempt reused NOISE_FLOOR_KL = 1e-8 verbatim from
# `full_grid_mechanistic_analysis.py` (that threshold is for the EXACT,
# directly-computed final KL, via the model's own forward pass). Running
# this script's own Part VIII validation gate (final-layer logit-lens KL
# vs. the true final KL, computed two independent ways for the same
# hidden state) caught max errors up to 2.2e-5 -- 1e-8 was therefore too
# tight for the logit-lens PATHWAY specifically, not for KL divergence in
# general. Diagnosed directly (not assumed): a targeted single-case check
# showed re-applying `final_norm`+`lm_head` to an exactly-recovered
# (lossless float64->float32 round-trip) hidden state reproduces the
# model's own logits only up to ~1e-3 absolute logit error -- ordinary
# float32 matmul non-associativity in the ~50k-vocabulary projection, not
# a logic bug (confirmed further: the worst-error cases are exactly the
# largest-hidden-state-norm cases, i.e. error scales with operand
# magnitude, the expected float32 rounding signature). The TRUE final KL
# (computed once, directly, never via logit-lens) still matches the frozen
# DPH sample's stored values with 0.0 error -- only the logit-lens
# RE-APPLICATION at intermediate/final layers carries this extra noise.
# Both tolerances below are set at ~5x the empirically observed worst-case
# logit-lens reconstruction error (2.2e-5), not chosen to produce any
# particular downstream statistic.
TOL_DELTA_KL = 1e-4
FINAL_LAYER_VALIDATION_TOL = 1e-4  # logit-lens vs. true final logits, abs KL (see Part VIII)


def js_divergence(logp_ref: torch.Tensor, logp_policy: torch.Tensor) -> float:
    """Jensen-Shannon divergence in nats between two log-probability
    vectors, computed in float64. JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M),
    M=(P+Q)/2 -- symmetric, bounded (<= ln 2), numerically stable since M
    is bounded away from 0 wherever either P or Q is."""
    p = logp_ref.exp()
    q = logp_policy.exp()
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-300))
    kl_pm = (p * (logp_ref - logm)).sum()
    kl_qm = (q * (logp_policy - logm)).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine similarity between two raw logit vectors, float64."""
    a64, b64 = a.double(), b.double()
    na, nb = torch.linalg.norm(a64), torch.linalg.norm(b64)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(1.0 - torch.dot(a64, b64) / (na * nb))


def classify_by_delta(per_layer_deltas: list[tuple[int, float]], final_winner: str, tol: float) -> dict:
    """Same persistent-sign algorithm as
    `downstream_propagation_sample.classify_trajectory`, generalized to an
    arbitrary (layer, delta) sequence and tolerance -- not a new method,
    the identical logic applied to Delta_KL instead of delta_resid."""
    target = 1 if final_winner == "local" else -1
    layers = [l for l, _ in per_layer_deltas]
    signs = [0 if abs(d) < tol else (1 if d > 0 else -1) for _, d in per_layer_deltas]

    idx_nonzero = [j for j, s in enumerate(signs) if s != 0]
    if not idx_nonzero:
        return dict(category="unresolved", emergence_layer=None, emergence_fraction=None,
                     n_sign_changes=0, immediate_sign="tie")

    nz_signs = [signs[j] for j in idx_nonzero]
    n_sign_changes = int(sum(1 for a, b in zip(nz_signs, nz_signs[1:]) if a != b))
    immediate_sign_val = nz_signs[0]
    immediate_sign = "local" if immediate_sign_val == 1 else "mass"

    last_bad = None
    for j in reversed(idx_nonzero):
        if signs[j] != target:
            last_bad = j
            break
    if last_bad is None:
        emergence_layer = layers[idx_nonzero[0]]
    else:
        after = [j for j in idx_nonzero if j > last_bad]
        emergence_layer = layers[after[0]] if after else None

    if emergence_layer is None:
        category = "never_consistent"
    elif immediate_sign_val == target and n_sign_changes == 0:
        category = "immediate_consistent"
    elif n_sign_changes <= 1:
        category = "single_flip"
    else:
        category = "multiple_flip"

    eviction_layer, last_layer = layers[0], layers[-1]
    denom = last_layer - eviction_layer
    emergence_fraction = (
        0.0 if denom == 0 or emergence_layer is None else (emergence_layer - eviction_layer) / denom
    )
    return dict(category=category, emergence_layer=emergence_layer, emergence_fraction=emergence_fraction,
                n_sign_changes=n_sign_changes, immediate_sign=immediate_sign)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def trace_functional(model, adapter, tok, arch: str, c: dict) -> dict:
    layer, head, i, j, p_idx = c["layer"], c["head"], c["i"], c["j"], c["prompt"]
    sd = get_stratum_data(model, adapter, tok, arch, p_idx, layer, head, i, j)
    n_lay = n_layers(model, arch)
    trace_layers = list(range(layer, n_lay))

    mass_block = torch.zeros(sd["n"], dtype=torch.bool)
    for idx in sd["mass_rank"][: sd["change_k"]]:
        mass_block[idx] = True
    local_block = torch.zeros(sd["n"], dtype=torch.bool)
    for idx in sd["local_rank"][: sd["change_k"]]:
        local_block[idx] = True

    ref = run_traced_forward(model, adapter, arch, sd, layer, head, None, trace_layers)
    mass = run_traced_forward(model, adapter, arch, sd, layer, head, mass_block, trace_layers)
    local = run_traced_forward(model, adapter, arch, sd, layer, head, local_block, trace_layers)

    kl_mass_final_true = float(kl_divergence(sd["logits_ref_full"][sd["qp"]], mass["logits"]))
    kl_local_final_true = float(kl_divergence(sd["logits_ref_full"][sd["qp"]], local["logits"]))
    advantage_final_true = kl_mass_final_true - kl_local_final_true
    final_winner = "local" if advantage_final_true > 0 else "mass"

    # cross-check against the frozen DPH sample's stored final values (paired reuse, not re-sampled)
    stored_advantage = c["advantage_final"]
    reconstruction_err = abs(advantage_final_true - stored_advantage)

    per_layer = []
    for b in trace_layers:
        h_ref, h_mass, h_local = ref["hidden"][b], mass["hidden"][b], local["hidden"][b]
        logits_ref_l = logit_lens(model, arch, h_ref)
        logits_mass_l = logit_lens(model, arch, h_mass)
        logits_local_l = logit_lens(model, arch, h_local)

        kl_mass_l = float(kl_divergence(logits_ref_l, logits_mass_l))
        kl_local_l = float(kl_divergence(logits_ref_l, logits_local_l))
        lp_ref = torch.log_softmax(logits_ref_l, dim=-1)
        lp_mass = torch.log_softmax(logits_mass_l, dim=-1)
        lp_local = torch.log_softmax(logits_local_l, dim=-1)
        js_mass = js_divergence(lp_ref, lp_mass)
        js_local = js_divergence(lp_ref, lp_local)
        l2_mass = float(logit_l2(logits_ref_l, logits_mass_l))
        l2_local = float(logit_l2(logits_ref_l, logits_local_l))
        cos_mass = cosine_distance(logits_ref_l, logits_mass_l)
        cos_local = cosine_distance(logits_ref_l, logits_local_l)

        per_layer.append(dict(
            layer=b,
            kl_mass=kl_mass_l, kl_local=kl_local_l, delta_kl=kl_mass_l - kl_local_l,
            js_mass=js_mass, js_local=js_local, delta_js=js_mass - js_local,
            l2_mass=l2_mass, l2_local=l2_local, delta_l2=l2_mass - l2_local,
            cos_mass=cos_mass, cos_local=cos_local, delta_cos=cos_mass - cos_local,
        ))

    # Part VIII: final-layer logit-lens must match the TRUE final logits
    # near-exactly (same computation: final_norm + lm_head on the final
    # hidden state) -- validated per case, not assumed.
    last = per_layer[-1]
    final_check_err_mass = abs(last["kl_mass"] - kl_mass_final_true)
    final_check_err_local = abs(last["kl_local"] - kl_local_final_true)
    final_sign_agrees = (
        (last["delta_kl"] > 0) == (advantage_final_true > 0)
        if abs(last["delta_kl"]) >= TOL_DELTA_KL else None
    )

    cls_kl = classify_by_delta([(e["layer"], e["delta_kl"]) for e in per_layer], final_winner, TOL_DELTA_KL)

    return dict(
        arch=arch, prompt=p_idx, layer=layer, head=head, i=i, j=j,
        depth_tertile=c["depth_tertile"], eviction_depth_fraction=c["eviction_depth_fraction"],
        final_winner=final_winner, advantage_final=advantage_final_true,
        reconstruction_err_vs_stored=reconstruction_err,
        final_layer_validation=dict(
            err_mass=final_check_err_mass, err_local=final_check_err_local, sign_agrees=final_sign_agrees,
        ),
        residual_category=c["category"],  # from the frozen DPH sample, unchanged
        functional_category=cls_kl["category"], functional_emergence_layer=cls_kl["emergence_layer"],
        functional_emergence_fraction=cls_kl["emergence_fraction"],
        functional_n_sign_changes=cls_kl["n_sign_changes"], functional_immediate_sign=cls_kl["immediate_sign"],
        per_layer=per_layer,
    )


def run() -> dict:
    torch.manual_seed(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dph_cases = json.loads(SAMPLE_CASES.read_text())
    print(f"reusing {len(dph_cases)} cases from the frozen DPH sample (no new population draw)")

    tok_p, model_p, adapter_p = load_pythia()
    tok_g, model_g, adapter_g = load_gpt2()

    cases = []
    n_errors = 0
    py_done = gp_done = 0
    for c in dph_cases:
        model, adapter, tok = (model_p, adapter_p, tok_p) if c["arch"] == "pythia" else (model_g, adapter_g, tok_g)
        try:
            cases.append(trace_functional(model, adapter, tok, c["arch"], c))
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"  [{c['arch']}] SKIPPED prompt={c['prompt']} layer={c['layer']} head={c['head']}: {e}")
        if c["arch"] == "pythia":
            py_done += 1
            if py_done % 20 == 0:
                print(f"[pythia] {py_done}/100", flush=True)
        else:
            gp_done += 1
            if gp_done % 30 == 0:
                print(f"[gpt2] {gp_done}/150", flush=True)

    print(f"n_errors: {n_errors}")
    (RESULTS_DIR / "cases.json").write_text(json.dumps(cases, indent=2))

    summary = analyze(cases)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, cases)
    print("Done. See results/functional_propagation_audit/report.txt")
    return summary


def _rate_block(cases: list[dict]) -> dict:
    n = len(cases)
    if n == 0:
        return dict(n=0)
    counts = {cat: sum(1 for c in cases if c["functional_category"] == cat)
              for cat in ("immediate_consistent", "single_flip", "multiple_flip", "never_consistent", "unresolved")}
    n_flip = counts["single_flip"] + counts["multiple_flip"]
    n_immediate = counts["immediate_consistent"]
    n_never_or_unres = counts["never_consistent"] + counts["unresolved"]
    fracs = [c["functional_emergence_fraction"] for c in cases if c["functional_emergence_fraction"] is not None]
    return dict(
        n=n, category_counts=counts,
        flip_rate=n_flip / n, flip_rate_ci95=wilson_ci(n_flip, n),
        immediate_consistent_rate=n_immediate / n, immediate_consistent_rate_ci95=wilson_ci(n_immediate, n),
        never_or_unresolved_rate=n_never_or_unres / n, never_or_unresolved_rate_ci95=wilson_ci(n_never_or_unres, n),
        median_emergence_fraction=float(np.median(fracs)) if fracs else None,
    )


def analyze(cases: list[dict]) -> dict:
    py = [c for c in cases if c["arch"] == "pythia"]
    gp = [c for c in cases if c["arch"] == "gpt2"]

    by_arch = dict(pythia=_rate_block(py), gpt2=_rate_block(gp), pooled=_rate_block(cases))
    by_winner = dict(
        pythia={w: _rate_block([c for c in py if c["final_winner"] == w]) for w in ("local", "mass")},
        gpt2={w: _rate_block([c for c in gp if c["final_winner"] == w]) for w in ("local", "mass")},
        pooled={w: _rate_block([c for c in cases if c["final_winner"] == w]) for w in ("local", "mass")},
    )

    # Part VI: direct residual-vs-functional cross-tabulation
    def succeeds(cat):
        return cat in ("immediate_consistent", "single_flip", "multiple_flip")

    cross = {"both_succeed": 0, "residual_fails_functional_succeeds": 0,
             "residual_succeeds_functional_fails": 0, "both_fail": 0}
    for c in cases:
        r_ok, f_ok = succeeds(c["residual_category"]), succeeds(c["functional_category"])
        if r_ok and f_ok:
            cross["both_succeed"] += 1
        elif not r_ok and f_ok:
            cross["residual_fails_functional_succeeds"] += 1
        elif r_ok and not f_ok:
            cross["residual_succeeds_functional_fails"] += 1
        else:
            cross["both_fail"] += 1

    resid_never = [c for c in cases if c["residual_category"] == "never_consistent"]
    resid_never_py = [c for c in resid_never if c["arch"] == "pythia"]
    resid_never_gp = [c for c in resid_never if c["arch"] == "gpt2"]

    def resolved_frac(group):
        if not group:
            return dict(n=0, n_resolved=0, frac=None)
        n_res = sum(1 for c in group if succeeds(c["functional_category"]))
        return dict(n=len(group), n_resolved=n_res, frac=n_res / len(group))

    resid_never_resolution = dict(
        pythia=resolved_frac(resid_never_py), gpt2=resolved_frac(resid_never_gp), pooled=resolved_frac(resid_never)
    )

    # Part VIII: numerical validation
    errs_mass = [c["final_layer_validation"]["err_mass"] for c in cases]
    errs_local = [c["final_layer_validation"]["err_local"] for c in cases]
    sign_checks = [c["final_layer_validation"]["sign_agrees"] for c in cases if c["final_layer_validation"]["sign_agrees"] is not None]
    recon_errs = [c["reconstruction_err_vs_stored"] for c in cases]
    validation = dict(
        max_final_layer_kl_err_mass=max(errs_mass), max_final_layer_kl_err_local=max(errs_local),
        max_reconstruction_err_vs_stored_dph=max(recon_errs),
        n_sign_checks=len(sign_checks), n_sign_agree=sum(sign_checks),
        frac_sign_agree=sum(sign_checks) / len(sign_checks) if sign_checks else None,
    )

    # Part IX: residual-vs-logit PERTURBATION-MAGNITUDE correlation in
    # never_consistent cases (magnitude, not signed winner-difference --
    # "is a big residual move associated with a big logit move" is a
    # question about sizes, not about which policy wins). Paired by case
    # identity against the frozen DPH sample's own stored residual trace.
    from scipy import stats as sstats
    dph_lookup = {(c["arch"], c["prompt"], c["layer"], c["head"], c["i"], c["j"]): c
                  for c in json.loads(SAMPLE_CASES.read_text())}
    resid_vs_logit_pairs = []
    for c in resid_never:
        key = (c["arch"], c["prompt"], c["layer"], c["head"], c["i"], c["j"])
        dph_c = dph_lookup.get(key)
        if dph_c is None:
            continue
        last_resid = dph_c["per_layer"][-1]
        last_func = c["per_layer"][-1]
        resid_magnitude = 0.5 * (last_resid["d_mass_resid"] + last_resid["d_local_resid"])
        logit_magnitude = 0.5 * (last_func["l2_mass"] + last_func["l2_local"])
        resid_vs_logit_pairs.append((resid_magnitude, logit_magnitude))
    corr_resid_logit = None
    if len(resid_vs_logit_pairs) >= 5:
        rr = np.array([p[0] for p in resid_vs_logit_pairs])
        ll = np.array([p[1] for p in resid_vs_logit_pairs])
        if np.std(rr) > 0 and np.std(ll) > 0:
            rho, pval = sstats.spearmanr(rr, ll)
            corr_resid_logit = dict(rho=float(rho), p=float(pval), n=len(rr))

    # strongest naturally sampled example (largest |advantage| where functional succeeds but residual failed)
    rescued = [c for c in cases if c["residual_category"] == "never_consistent" and succeeds(c["functional_category"])]
    strongest_rescued = max(rescued, key=lambda c: abs(c["advantage_final"]), default=None)
    if strongest_rescued:
        strongest_rescued = {k: v for k, v in strongest_rescued.items() if k != "per_layer"}

    # strongest counterexample: functional ALSO fails, largest |advantage|
    both_fail = [c for c in cases if not succeeds(c["residual_category"]) and not succeeds(c["functional_category"])]
    strongest_counterexample = max(both_fail, key=lambda c: abs(c["advantage_final"]), default=None)
    if strongest_counterexample:
        strongest_counterexample = {k: v for k, v in strongest_counterexample.items() if k != "per_layer"}

    # verdict
    pooled_flip = by_arch["pooled"]["flip_rate"]
    pooled_immediate = by_arch["pooled"]["immediate_consistent_rate"]
    frac_resolved = resid_never_resolution["pooled"]["frac"] or 0.0
    py_resolved = resid_never_resolution["pythia"]["frac"] or 0.0
    gp_resolved = resid_never_resolution["gpt2"]["frac"] or 0.0

    if frac_resolved < 0.10:
        verdict = "FLH-A"
    elif frac_resolved < 0.40:
        verdict = "FLH-B"
    elif py_resolved >= 0.40 and gp_resolved >= 0.40:
        verdict = "FLH-C"
    else:
        verdict = "FLH-B"
    # FLH-D requires an explicit, reproducible mechanism -- not assigned here
    # from prevalence alone; see Part IX correlation for whether that bar is met.

    return dict(
        seed_note="reused exactly, no new draw (see results/downstream_propagation_sample, seed 20260910)",
        tol_delta_kl=TOL_DELTA_KL,
        by_arch=by_arch, by_final_winner=by_winner,
        cross_tabulation=cross,
        residual_never_consistent_resolution=resid_never_resolution,
        final_layer_validation=validation,
        component_note="Part IX correlation (|resid delta| vs |logit-space delta|) in never_consistent cases:",
        resid_vs_logit_correlation=corr_resid_logit,
        strongest_rescued_example=strongest_rescued,
        strongest_counterexample=strongest_counterexample,
        verdict=verdict,
    )


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    if isinstance(x, tuple):
        return "(" + ", ".join(_f(v) for v in x) + ")"
    return str(x)


def write_report(summary: dict, cases: list[dict]) -> None:
    L = []
    A = L.append
    W = 100
    A("FUNCTIONAL-LATENT HYPOTHESIS (FLH) AUDIT")
    A("=" * W)
    A("")
    A(summary["seed_note"])
    A(f"TOL_DELTA_KL = {summary['tol_delta_kl']:.1e} (revised from the initial 1e-8 reuse of NOISE_FLOOR_KL after")
    A(f"  this script's own Part VIII validation caught up to 2.2e-5 logit-lens reconstruction noise -- see")
    A(f"  the module docstring/comment above TOL_DELTA_KL for the full diagnosis.)")
    A(f"KL convention: kl_divergence(logits_full, logits_compressed) = KL(p_full || p_compressed); "
      f"Delta_KL_l = KL_mass(l) - KL_local(l); positive => local functionally closer to reference at layer l.")

    A("")
    A("-" * W)
    A("PART VIII -- NUMERICAL VALIDATION (final-layer logit-lens vs. true final logits)")
    A("-" * W)
    v = summary["final_layer_validation"]
    A(f"max |final-layer logit-lens KL - true final KL|, mass side: {v['max_final_layer_kl_err_mass']:.3e}")
    A(f"max |final-layer logit-lens KL - true final KL|, local side: {v['max_final_layer_kl_err_local']:.3e}")
    A(f"max |recomputed advantage - DPH-sample-stored advantage|: {v['max_reconstruction_err_vs_stored_dph']:.3e}")
    A(f"final-layer sign(Delta_KL) agrees with true final winner: {v['n_sign_agree']}/{v['n_sign_checks']} "
      f"({_f(v['frac_sign_agree'])})")
    if v["max_final_layer_kl_err_mass"] > FINAL_LAYER_VALIDATION_TOL or v["max_final_layer_kl_err_local"] > FINAL_LAYER_VALIDATION_TOL:
        A(f"*** WARNING: exceeds the pre-declared validation tolerance {FINAL_LAYER_VALIDATION_TOL:.1e} -- "
          f"results below should be treated with caution pending debugging. ***")
    else:
        A(f"All within the pre-declared validation tolerance ({FINAL_LAYER_VALIDATION_TOL:.1e}). Proceeding.")

    A("")
    A("-" * W)
    A("PRIMARY FUNCTIONAL STATISTICS BY ARCHITECTURE")
    A("-" * W)
    for arch in ("pythia", "gpt2", "pooled"):
        s = summary["by_arch"][arch]
        A(f"[{arch}] n={s['n']}")
        A(f"  functional flip_rate            = {_f(s['flip_rate'])}  95% CI {_f(s['flip_rate_ci95'])}")
        A(f"  functional immediate_consistent = {_f(s['immediate_consistent_rate'])}  95% CI {_f(s['immediate_consistent_rate_ci95'])}")
        A(f"  functional never/unresolved     = {_f(s['never_or_unresolved_rate'])}  95% CI {_f(s['never_or_unresolved_rate_ci95'])}")
        A(f"  median functional emergence frac= {_f(s['median_emergence_fraction'])}")
        A(f"  category_counts = {s['category_counts']}")
        A("")

    A("-" * W)
    A("RESIDUAL vs. FUNCTIONAL CROSS-TABULATION (Part VI)")
    A("-" * W)
    ct = summary["cross_tabulation"]
    for k, v2 in ct.items():
        A(f"  {k:<38}: {v2}")

    A("")
    A("-" * W)
    A("CENTRAL STATISTIC: residual never_consistent cases resolved in functional/KL space")
    A("-" * W)
    for arch in ("pythia", "gpt2", "pooled"):
        r = summary["residual_never_consistent_resolution"][arch]
        A(f"  [{arch}] n={r['n']}  resolved={r['n_resolved']}  frac={_f(r['frac'])}")

    A("")
    A("-" * W)
    A("LOCAL-WIN vs MASS-WIN, FUNCTIONAL SPACE (Part VII)")
    A("-" * W)
    for arch in ("pythia", "gpt2", "pooled"):
        A(f"[{arch}]")
        for w in ("local", "mass"):
            s = summary["by_final_winner"][arch][w]
            A(f"  {w:<6} n={s.get('n',0):<4} functional_flip_rate={_f(s.get('flip_rate'))} "
              f"functional_immediate={_f(s.get('immediate_consistent_rate'))} "
              f"median_emergence_frac={_f(s.get('median_emergence_fraction'))}")

    A("")
    A("-" * W)
    A("PART IX -- residual-vs-logit perturbation-norm relationship in never_consistent cases (descriptive)")
    A("-" * W)
    A(str(summary["resid_vs_logit_correlation"]))
    A("(Tested only as a simple consequence check -- not a new feature search. A weak/absent correlation would")
    A(" be consistent with residual perturbations landing partly in functionally weak directions; this is")
    A(" reported as a numerical observation only, not asserted as the mechanism.)")

    A("")
    A("-" * W)
    A("STRONGEST NATURALLY SAMPLED EXAMPLE (largest |advantage| rescued: residual never_consistent, functional succeeds)")
    A("-" * W)
    sr = summary["strongest_rescued_example"]
    if sr:
        A(f"[{sr['arch']}] prompt={sr['prompt']} layer={sr['layer']} head={sr['head']} i={sr['i']} j={sr['j']}")
        A(f"  advantage={sr['advantage_final']:.4e}  final_winner={sr['final_winner']}")
        A(f"  residual_category={sr['residual_category']}  functional_category={sr['functional_category']}  "
          f"functional_emergence_layer={sr['functional_emergence_layer']}")
    else:
        A("none found")

    A("")
    A("-" * W)
    A("STRONGEST COUNTEREXAMPLE (both residual and functional fail to resolve, largest |advantage|)")
    A("-" * W)
    sc = summary["strongest_counterexample"]
    if sc:
        A(f"[{sc['arch']}] prompt={sc['prompt']} layer={sc['layer']} head={sc['head']} i={sc['i']} j={sc['j']}")
        A(f"  advantage={sc['advantage_final']:.4e}  final_winner={sc['final_winner']}")
        A(f"  residual_category={sc['residual_category']}  functional_category={sc['functional_category']}")
    else:
        A("none found -- every case where residual fails, functional resolves (or vice versa is not both-fail)")

    A("")
    A("-" * W)
    A("VERDICT")
    A("-" * W)
    A(summary["verdict"])

    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
