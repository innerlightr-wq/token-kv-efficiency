r"""Moderate-scale stratified random replication of the Downstream
Propagation Hypothesis (DPH) case-study audit.

The 14-case hand-picked panel (`results/downstream_propagation_audit/`)
found a 43% winner-flip rate and DPH-C. This script asks whether that rate
survives UNBIASED, stratified random sampling at ~10x the panel size, or
was itself a selection artifact of hand-picking examples.

**Reuses, unchanged, via direct import (not re-transcribed):** every
tracing primitive from `experiments/downstream_propagation_audit.py` --
`get_stratum_data`, `run_traced_forward`, `get_block`, `get_attn_mlp`,
`get_final_norm_and_head`, `n_layers`, `capture_output_hook`,
`multi_capture`, `logit_lens`, `load_pythia`, `load_gpt2`, `DEPTH_MAX`,
`MASS_EPSILON`, `EVALUATION_PROMPTS`. In particular, this script does NOT
revert to `output_hidden_states=True` -- the fresh-read-only-hook fix for
Pythia's residual-stream capture (verified last round: the installed
transformers version's `@capture_outputs`-decorated `GPTNeoXModel.forward`
does not reflect a forward hook's modification in its `hidden_states`
tuple) is preserved exactly, because `run_traced_forward` is imported
unchanged, not reimplemented.

No new feature engineering: sampling stratifies only on already-computed
quantities (`layer`, `advantage_mass_minus_local_at_change`'s sign,
`prompt`) already present in the frozen full-grid `pairs.json` files.

Writes `results/downstream_propagation_sample/{summary,cases}.json` and
`report.txt`. Does not modify `src/kv_efficiency/`, any test, any existing
result file (including the 14-case panel), or any doc/status file.
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

# ---------------------------------------------------------------------------
# Import the validated trace pipeline UNCHANGED from the case-study script.
# ---------------------------------------------------------------------------
_dpa_spec = importlib.util.spec_from_file_location(
    "downstream_propagation_audit", ROOT / "experiments" / "downstream_propagation_audit.py"
)
dpa = importlib.util.module_from_spec(_dpa_spec)
_dpa_spec.loader.exec_module(dpa)

get_stratum_data = dpa.get_stratum_data
run_traced_forward = dpa.run_traced_forward
n_layers = dpa.n_layers
load_pythia = dpa.load_pythia
load_gpt2 = dpa.load_gpt2
DEPTH_MAX = dpa.DEPTH_MAX

RESULTS_DIR = ROOT / "results" / "downstream_propagation_sample"
PYTHIA_PAIRS = ROOT / "results" / "full_grid_mechanistic_analysis" / "pairs.json"
GPT2_PAIRS = ROOT / "results" / "cross_architecture_full_grid" / "pairs.json"

SEED = 20260910  # fixed, recorded
N_TARGET_PYTHIA = 100
N_TARGET_GPT2 = 150
MIN_PER_ARCH = 75

# Near-zero tolerance for sign(Delta_l), fixed BEFORE looking at aggregate
# results. Justification: the model forward pass itself runs in float32
# (dtype=torch.float32); float32 relative precision is ~1.19e-7. Residual-
# stream norms observed in the 14-case panel ranged from O(1e-3) (near-tie
# cases) to O(1e2) (deep layers, large perturbations); an absolute
# reconstruction/rounding noise floor of eps_rel * typical_norm therefore
# spans roughly 1e-7 to 1e-5 across the panel's own observed scale. We set
# TOL_DELTA = 1e-4 -- one to two orders of magnitude above that estimated
# float32 noise floor at the scales actually observed -- so that a
# near-zero classification reflects genuine floating-point/reconstruction
# indistinguishability, not real (if small) signal.
TOL_DELTA = 1e-4

DEPTH_TERTILE_EDGES = (1 / 3, 2 / 3)  # depth_fraction cutoffs for early/mid/late


def measurable_pairs(path: Path) -> list[dict]:
    pairs = json.loads(path.read_text())
    return [
        p for p in pairs
        if p.get("changes_set_within_kmax") and p.get("above_noise_floor")
        and p.get("advantage_mass_minus_local_at_change") is not None
    ]


def depth_tertile(layer: int) -> str:
    f = layer / DEPTH_MAX
    if f < DEPTH_TERTILE_EDGES[0]:
        return "early"
    if f < DEPTH_TERTILE_EDGES[1]:
        return "mid"
    return "late"


def stratified_sample(pairs: list[dict], n_target: int, rng: np.random.Generator) -> list[dict]:
    """Stratify by (depth_tertile, final_winner); allocate ~n_target/6 to
    each of the 6 cells, backfilling shortfall from other cells
    proportionally to their remaining availability. Fixed seed, recorded."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for p in pairs:
        winner = "local" if p["advantage_mass_minus_local_at_change"] > 0 else "mass"
        key = (depth_tertile(p["layer"]), winner)
        cells.setdefault(key, []).append(p)
    for k in cells:
        rng.shuffle(cells[k])

    n_cells = len(cells)
    base = n_target // n_cells if n_cells else 0
    allocation = {k: min(base, len(v)) for k, v in cells.items()}
    shortfall = n_target - sum(allocation.values())

    # backfill shortfall from cells with remaining capacity, round-robin
    remaining = {k: len(cells[k]) - allocation[k] for k in cells}
    keys_cycle = list(cells.keys())
    i = 0
    guard = 0
    while shortfall > 0 and any(remaining[k] > 0 for k in keys_cycle) and guard < 100000:
        k = keys_cycle[i % len(keys_cycle)]
        if remaining[k] > 0:
            allocation[k] += 1
            remaining[k] -= 1
            shortfall -= 1
        i += 1
        guard += 1

    sample = []
    for k, n in allocation.items():
        sample.extend(cells[k][:n])
    rng.shuffle(sample)
    return sample, {f"{k[0]}_{k[1]}": dict(available=len(cells[k]), sampled=allocation[k]) for k in cells}


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def classify_trajectory(per_layer: list[dict], final_winner: str) -> dict:
    target = 1 if final_winner == "local" else -1
    signs = []
    for e in per_layer:
        d = e["delta_resid"]
        signs.append(0 if abs(d) < TOL_DELTA else (1 if d > 0 else -1))

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
        emergence_layer = per_layer[idx_nonzero[0]]["layer"]
    else:
        after = [j for j in idx_nonzero if j > last_bad]
        emergence_layer = per_layer[after[0]]["layer"] if after else None

    if emergence_layer is None:
        category = "never_consistent"
    elif immediate_sign_val == target and n_sign_changes == 0:
        category = "immediate_consistent"
    elif n_sign_changes <= 1:
        category = "single_flip"
    else:
        category = "multiple_flip"

    eviction_layer = per_layer[0]["layer"]
    last_layer = per_layer[-1]["layer"]
    denom = last_layer - eviction_layer
    emergence_fraction = (
        0.0 if denom == 0 or emergence_layer is None
        else (emergence_layer - eviction_layer) / denom
    )

    return dict(
        category=category, emergence_layer=emergence_layer, emergence_fraction=emergence_fraction,
        n_sign_changes=n_sign_changes, immediate_sign=immediate_sign,
    )


def trace_one(model, adapter, tok, arch: str, p: dict) -> dict:
    layer, head, i, j, p_idx = p["layer"], p["head"], p["i"], p["j"], p["prompt"]
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

    from kv_efficiency.damage_metrics import kl_divergence
    kl_mass_final = float(kl_divergence(sd["logits_ref_full"][sd["qp"]], mass["logits"]))
    kl_local_final = float(kl_divergence(sd["logits_ref_full"][sd["qp"]], local["logits"]))
    advantage_final = kl_mass_final - kl_local_final
    final_winner = "local" if advantage_final > 0 else "mass"

    per_layer = []
    for b in trace_layers:
        h_ref, h_mass, h_local = ref["hidden"][b], mass["hidden"][b], local["hidden"][b]
        d_mass_resid = float(torch.linalg.norm(h_mass - h_ref))
        d_local_resid = float(torch.linalg.norm(h_local - h_ref))
        attn_ref, mlp_ref = ref["attn"][b], ref["mlp"][b]
        d_attn_mass = float(torch.linalg.norm(mass["attn"][b] - attn_ref))
        d_attn_local = float(torch.linalg.norm(local["attn"][b] - attn_ref))
        d_mlp_mass = float(torch.linalg.norm(mass["mlp"][b] - mlp_ref))
        d_mlp_local = float(torch.linalg.norm(local["mlp"][b] - mlp_ref))
        per_layer.append(dict(
            layer=b, d_mass_resid=d_mass_resid, d_local_resid=d_local_resid,
            delta_resid=d_mass_resid - d_local_resid,
            delta_attn=d_attn_mass - d_attn_local, delta_mlp=d_mlp_mass - d_mlp_local,
        ))

    cls = classify_trajectory(per_layer, final_winner)
    n_lay_total = n_layers(model, arch)

    return dict(
        arch=arch, prompt=p_idx, layer=layer, head=head, i=i, j=j,
        depth_tertile=depth_tertile(layer), eviction_depth_fraction=layer / DEPTH_MAX,
        n_downstream_layers=n_lay_total - layer,
        change_k=sd["change_k"], n_candidates=sd["n"] - 2,
        kl_mass_final=kl_mass_final, kl_local_final=kl_local_final, advantage_final=advantage_final,
        final_winner=final_winner,
        category=cls["category"], emergence_layer=cls["emergence_layer"],
        emergence_fraction=cls["emergence_fraction"], n_sign_changes=cls["n_sign_changes"],
        immediate_sign=cls["immediate_sign"],
        per_layer=per_layer,
    )


def run() -> dict:
    torch.manual_seed(0)
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    py_pool = measurable_pairs(PYTHIA_PAIRS)
    gp_pool = measurable_pairs(GPT2_PAIRS)
    print(f"measurable pool: pythia={len(py_pool)} gpt2={len(gp_pool)}")

    py_sample, py_strata = stratified_sample(py_pool, N_TARGET_PYTHIA, rng)
    gp_sample, gp_strata = stratified_sample(gp_pool, N_TARGET_GPT2, rng)
    print(f"sampled: pythia={len(py_sample)} gpt2={len(gp_sample)}")
    print("pythia strata:", py_strata)
    print("gpt2 strata:", gp_strata)

    tok_p, model_p, adapter_p = load_pythia()
    tok_g, model_g, adapter_g = load_gpt2()

    cases = []
    n_errors = 0
    for idx, p in enumerate(py_sample):
        try:
            cases.append(trace_one(model_p, adapter_p, tok_p, "pythia", p))
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"  [pythia #{idx}] SKIPPED due to error: {e}")
        if (idx + 1) % 20 == 0:
            print(f"[pythia] traced {idx+1}/{len(py_sample)}", flush=True)
    for idx, p in enumerate(gp_sample):
        try:
            cases.append(trace_one(model_g, adapter_g, tok_g, "gpt2", p))
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"  [gpt2 #{idx}] SKIPPED due to error: {e}")
        if (idx + 1) % 30 == 0:
            print(f"[gpt2] traced {idx+1}/{len(gp_sample)}", flush=True)

    print(f"n_errors (skipped): {n_errors}")
    (RESULTS_DIR / "cases.json").write_text(json.dumps(cases, indent=2))

    summary = analyze(cases, py_strata, gp_strata, len(py_pool), len(gp_pool))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, cases)
    print("Done. See results/downstream_propagation_sample/report.txt")
    return summary


def group_stats(cases: list[dict]) -> dict:
    n = len(cases)
    if n == 0:
        return dict(n=0)
    n_flip = sum(1 for c in cases if c["category"] in ("single_flip", "multiple_flip"))
    n_immediate = sum(1 for c in cases if c["category"] == "immediate_consistent")
    n_multi = sum(1 for c in cases if c["category"] == "multiple_flip")
    n_unresolved = sum(1 for c in cases if c["category"] in ("unresolved", "never_consistent"))
    depths = [c["emergence_fraction"] for c in cases if c["emergence_fraction"] is not None]
    layers = [c["emergence_layer"] for c in cases if c["emergence_layer"] is not None]
    n_final_third = sum(1 for d in depths if d >= 2 / 3)
    n_immediate_stable = sum(1 for d in depths if d == 0.0)
    return dict(
        n=n,
        flip_rate=n_flip / n, flip_rate_ci95=wilson_ci(n_flip, n),
        immediate_consistent_rate=n_immediate / n, immediate_consistent_rate_ci95=wilson_ci(n_immediate, n),
        multiple_flip_rate=n_multi / n, multiple_flip_rate_ci95=wilson_ci(n_multi, n),
        unresolved_rate=n_unresolved / n, unresolved_rate_ci95=wilson_ci(n_unresolved, n),
        median_emergence_fraction=float(np.median(depths)) if depths else None,
        emergence_fraction_quartiles=(
            [float(np.percentile(depths, 25)), float(np.percentile(depths, 75))] if depths else None
        ),
        median_emergence_layer=float(np.median(layers)) if layers else None,
        frac_emerges_in_final_third=n_final_third / len(depths) if depths else None,
        frac_immediately_stable=n_immediate_stable / len(depths) if depths else None,
        category_counts={
            cat: sum(1 for c in cases if c["category"] == cat)
            for cat in ("immediate_consistent", "single_flip", "multiple_flip", "never_consistent", "unresolved")
        },
    )


def analyze(cases: list[dict], py_strata: dict, gp_strata: dict, py_pool_n: int, gp_pool_n: int) -> dict:
    py_cases = [c for c in cases if c["arch"] == "pythia"]
    gp_cases = [c for c in cases if c["arch"] == "gpt2"]

    by_arch = dict(pythia=group_stats(py_cases), gpt2=group_stats(gp_cases), pooled=group_stats(cases))

    by_depth = {}
    for arch, arch_cases in (("pythia", py_cases), ("gpt2", gp_cases), ("pooled", cases)):
        by_depth[arch] = {t: group_stats([c for c in arch_cases if c["depth_tertile"] == t]) for t in ("early", "mid", "late")}

    by_winner = {}
    for arch, arch_cases in (("pythia", py_cases), ("gpt2", gp_cases), ("pooled", cases)):
        by_winner[arch] = {w: group_stats([c for c in arch_cases if c["final_winner"] == w]) for w in ("local", "mass")}

    # architecture comparison: simple effect size (risk difference) + CI on the difference (normal approx)
    def risk_diff(k1, n1, k2, n2):
        if n1 == 0 or n2 == 0:
            return dict(diff=None, ci95=None)
        p1, p2 = k1 / n1, k2 / n2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        diff = p1 - p2
        return dict(diff=float(diff), ci95=(float(diff - 1.96 * se), float(diff + 1.96 * se)))

    n_flip_py = sum(1 for c in py_cases if c["category"] in ("single_flip", "multiple_flip"))
    n_flip_gp = sum(1 for c in gp_cases if c["category"] in ("single_flip", "multiple_flip"))
    n_multi_py = sum(1 for c in py_cases if c["category"] == "multiple_flip")
    n_multi_gp = sum(1 for c in gp_cases if c["category"] == "multiple_flip")
    arch_comparison = dict(
        flip_rate_diff_pythia_minus_gpt2=risk_diff(n_flip_py, len(py_cases), n_flip_gp, len(gp_cases)),
        multiple_flip_rate_diff_pythia_minus_gpt2=risk_diff(n_multi_py, len(py_cases), n_multi_gp, len(gp_cases)),
    )

    # component audit: at the emergence layer, is delta_attn or delta_mlp larger in magnitude?
    comp = dict(attn_larger=0, mlp_larger=0, comparable=0, n_evaluated=0)
    for c in cases:
        if c["emergence_layer"] is None:
            continue
        entry = next((e for e in c["per_layer"] if e["layer"] == c["emergence_layer"]), None)
        if entry is None:
            continue
        a, m = abs(entry["delta_attn"]), abs(entry["delta_mlp"])
        comp["n_evaluated"] += 1
        if a > 1.5 * m:
            comp["attn_larger"] += 1
        elif m > 1.5 * a:
            comp["mlp_larger"] += 1
        else:
            comp["comparable"] += 1

    # strongest naturally sampled counterexample: largest |advantage| among flipped cases
    flipped = [c for c in cases if c["category"] in ("single_flip", "multiple_flip", "never_consistent")]
    strongest_counterexample = max(flipped, key=lambda c: abs(c["advantage_final"]), default=None)
    if strongest_counterexample is not None:
        strongest_counterexample = {k: v for k, v in strongest_counterexample.items() if k != "per_layer"}

    # falsification check
    pooled = by_arch["pooled"]
    falsification = dict(
        flip_rate_near_zero=pooled["flip_rate"] < 0.05,
        almost_all_immediately_stable=pooled["immediate_consistent_rate"] > 0.90,
    )

    if falsification["flip_rate_near_zero"] or falsification["almost_all_immediately_stable"]:
        verdict = "DPH-A"
    else:
        py_flip = by_arch["pythia"]["flip_rate"]
        gp_flip = by_arch["gpt2"]["flip_rate"]
        both_nontrivial = py_flip >= 0.10 and gp_flip >= 0.10
        comp_dominant = comp["n_evaluated"] > 0 and max(comp["attn_larger"], comp["mlp_larger"]) / comp["n_evaluated"] >= 0.75
        if both_nontrivial and comp_dominant:
            verdict = "DPH-D"
        elif both_nontrivial:
            verdict = "DPH-C"
        else:
            verdict = "DPH-B"

    return dict(
        seed=SEED, tol_delta=TOL_DELTA,
        n_pythia_pool=py_pool_n, n_gpt2_pool=gp_pool_n,
        pythia_strata_allocation=py_strata, gpt2_strata_allocation=gp_strata,
        by_arch=by_arch, by_depth_tertile=by_depth, by_final_winner=by_winner,
        architecture_comparison=arch_comparison,
        component_audit=comp,
        strongest_counterexample=strongest_counterexample,
        falsification_checks=falsification,
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
    A("DOWNSTREAM PROPAGATION HYPOTHESIS -- MODERATE-SCALE STRATIFIED RANDOM REPLICATION")
    A("=" * W)
    A("")
    A(f"seed={summary['seed']}  TOL_DELTA={summary['tol_delta']:.1e}")
    A(f"measurable pools: pythia={summary['n_pythia_pool']}  gpt2={summary['n_gpt2_pool']}")
    A(f"sampled: pythia={summary['by_arch']['pythia']['n']}  gpt2={summary['by_arch']['gpt2']['n']}  "
      f"pooled={summary['by_arch']['pooled']['n']}")
    A("")
    A("Stratum allocation (available / sampled), by (depth_tertile, final_winner):")
    A(f"  pythia: {summary['pythia_strata_allocation']}")
    A(f"  gpt2:   {summary['gpt2_strata_allocation']}")

    A("")
    A("-" * W)
    A("PRIMARY STATISTICS BY ARCHITECTURE")
    A("-" * W)
    for arch in ("pythia", "gpt2", "pooled"):
        s = summary["by_arch"][arch]
        A(f"[{arch}] n={s['n']}")
        A(f"  flip_rate               = {_f(s['flip_rate'])}  95% CI {_f(s['flip_rate_ci95'])}")
        A(f"  immediate_consistent    = {_f(s['immediate_consistent_rate'])}  95% CI {_f(s['immediate_consistent_rate_ci95'])}")
        A(f"  multiple_flip_rate      = {_f(s['multiple_flip_rate'])}  95% CI {_f(s['multiple_flip_rate_ci95'])}")
        A(f"  unresolved_rate         = {_f(s['unresolved_rate'])}  95% CI {_f(s['unresolved_rate_ci95'])}")
        A(f"  median_emergence_frac   = {_f(s['median_emergence_fraction'])}  IQR={_f(s['emergence_fraction_quartiles'])}")
        A(f"  median_emergence_layer  = {_f(s['median_emergence_layer'])}")
        A(f"  frac_emerges_final_third= {_f(s['frac_emerges_in_final_third'])}")
        A(f"  frac_immediately_stable = {_f(s['frac_immediately_stable'])}")
        A(f"  category_counts         = {s['category_counts']}")
        A("")

    A("-" * W)
    A("EVICTION-DEPTH EFFECT (early/mid/late, remaining-depth-normalized emergence fraction)")
    A("-" * W)
    for arch in ("pythia", "gpt2", "pooled"):
        A(f"[{arch}]")
        for tert in ("early", "mid", "late"):
            s = summary["by_depth_tertile"][arch][tert]
            A(f"  {tert:<6} n={s.get('n',0):<4} flip_rate={_f(s.get('flip_rate'))} "
              f"median_emergence_frac={_f(s.get('median_emergence_fraction'))}")

    A("")
    A("-" * W)
    A("FINAL-WINNER EFFECT (local-win vs mass-win cases)")
    A("-" * W)
    for arch in ("pythia", "gpt2", "pooled"):
        A(f"[{arch}]")
        for w in ("local", "mass"):
            s = summary["by_final_winner"][arch][w]
            A(f"  {w:<6} n={s.get('n',0):<4} flip_rate={_f(s.get('flip_rate'))} "
              f"immediate_consistent={_f(s.get('immediate_consistent_rate'))} "
              f"median_emergence_frac={_f(s.get('median_emergence_fraction'))}")

    A("")
    A("-" * W)
    A("ARCHITECTURE COMPARISON (simple risk difference, normal-approx 95% CI)")
    A("-" * W)
    ac = summary["architecture_comparison"]
    A(f"flip_rate difference (pythia - gpt2): {_f(ac['flip_rate_diff_pythia_minus_gpt2']['diff'])}  "
      f"95% CI {_f(ac['flip_rate_diff_pythia_minus_gpt2']['ci95'])}")
    A(f"multiple_flip_rate difference (pythia - gpt2): {_f(ac['multiple_flip_rate_diff_pythia_minus_gpt2']['diff'])}  "
      f"95% CI {_f(ac['multiple_flip_rate_diff_pythia_minus_gpt2']['ci95'])}")
    A("(A CI crossing zero means the architectures are not distinguishable at this sample size --")
    A(" not promoted to an architectural mechanism regardless of the point estimate.)")

    A("")
    A("-" * W)
    A("COMPONENT AUDIT (descriptive only, at the emergence layer)")
    A("-" * W)
    ca = summary["component_audit"]
    A(f"n_evaluated={ca['n_evaluated']}  attn_larger={ca['attn_larger']} ({_f(ca['attn_larger']/max(1,ca['n_evaluated']))})  "
      f"mlp_larger={ca['mlp_larger']} ({_f(ca['mlp_larger']/max(1,ca['n_evaluated']))})  "
      f"comparable={ca['comparable']} ({_f(ca['comparable']/max(1,ca['n_evaluated']))})")
    A("('larger' = >1.5x the other component's |delta| at the emergence layer)")

    A("")
    A("-" * W)
    A("STRONGEST NATURALLY SAMPLED COUNTEREXAMPLE (largest |advantage| among flipped/never-consistent cases)")
    A("-" * W)
    sc = summary["strongest_counterexample"]
    if sc:
        A(f"[{sc['arch']}] prompt={sc['prompt']} layer={sc['layer']} head={sc['head']} i={sc['i']} j={sc['j']}")
        A(f"  advantage={sc['advantage_final']:.4e}  final_winner={sc['final_winner']}  category={sc['category']}")
        A(f"  immediate_sign={sc['immediate_sign']}  emergence_layer={sc['emergence_layer']}  "
          f"emergence_fraction={_f(sc['emergence_fraction'])}  n_sign_changes={sc['n_sign_changes']}")
    else:
        A("none found (no flipped/never-consistent cases in the sample)")

    A("")
    A("-" * W)
    A("FALSIFICATION CHECKS")
    A("-" * W)
    fc = summary["falsification_checks"]
    A(f"flip_rate_near_zero (<5%, pooled): {fc['flip_rate_near_zero']}")
    A(f"almost_all_immediately_stable (>90%, pooled): {fc['almost_all_immediately_stable']}")

    A("")
    A("-" * W)
    A("VERDICT")
    A("-" * W)
    A(summary["verdict"])

    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
