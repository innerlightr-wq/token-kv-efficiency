r"""State-Dependent Propagation Hypothesis (SDPH) audit.

SDPH (hypothesis, not a claim): the eventual winner is hard to predict from
static perturbation measures because eviction changes the hidden state
itself, and the SAME downstream block responds differently to the SAME
perturbation depending on the state it is evaluated at -- i.e. the local
linear response J_l(h) genuinely differs between J_l(h_mass) and
J_l(h_local), in a way that matters for the outcome.

**Cheapest direct test, not a full Jacobian.** For a single chosen block
F_l, this script evaluates finite differences F_l(base + delta) - F_l(base)
for various (base, delta, scale) combinations -- never a materialized
Jacobian, never claimed as curvature/Lyapunov/Hessian.

**How "F_l evaluated at an arbitrary state" is made well-defined (Part IV):**
a forward PRE-hook on the target block replaces ONLY the hidden-state row
at the measured query position with a custom vector, leaving every other
position and every other forward argument (causal mask, position
embeddings/rotary, layer_past, etc.) exactly as the model's own reference
forward pass already supplies them. This sidesteps hand-reconstructing
architecture-specific auxiliary state (which would be a much more invasive,
error-prone approach) while still answering exactly the intended question:
"if this vector were the hidden state at this one position, what would this
block compute?" Validated directly below (not assumed): substituting the
position's OWN natural value back in must reproduce the already-captured
reference output exactly.

**Reuses, unchanged, via direct import:** `get_stratum_data`,
`run_traced_forward`, `get_block`, `get_attn_mlp`, `n_layers`, `DEPTH_MAX`,
`load_pythia`, `load_gpt2`, `capture_output_hook`, `multi_capture` from
`experiments/downstream_propagation_audit.py`. Case selection uses only
fields already present in `results/downstream_propagation_sample/cases.json`
(arch, depth_tertile, final_winner, category, emergence_layer,
eviction_depth_fraction) -- no new metric.

Writes `results/state_dependent_propagation_audit/{summary,cases}.json` and
`report.txt`. Does not modify `src/kv_efficiency/`, any test, any existing
result file, or any doc/status file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import ExitStack, contextmanager
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
get_block = dpa.get_block
get_attn_mlp = dpa.get_attn_mlp
n_layers = dpa.n_layers
DEPTH_MAX = dpa.DEPTH_MAX
load_pythia = dpa.load_pythia
load_gpt2 = dpa.load_gpt2
capture_output_hook = dpa.capture_output_hook

RESULTS_DIR = ROOT / "results" / "state_dependent_propagation_audit"
DPH_SAMPLE = ROOT / "results" / "downstream_propagation_sample" / "cases.json"

SELECTION_SEED = 20260910
N_PER_CELL = 2  # arch x category_group x depth_tertile(early/mid) -> 2*3*2 = 12 cells
SCALES = (0.25, 0.5, 1.0)
EPS_NORM = 1e-12
VALIDATION_TOL = 1e-4  # block-output reconstruction, same order as prior FLH logit-lens tolerance


# ---------------------------------------------------------------------------
# Part IV -- the swap-input mechanism
# ---------------------------------------------------------------------------

@contextmanager
def override_position_hook(module: torch.nn.Module, position: int, new_vector: torch.Tensor):
    """Forward PRE-hook: replaces `hidden_states[:, position, :]` (the
    module's first positional OR keyword arg) with `new_vector`. Every
    other position and every other argument passes through unmodified --
    the model's own forward supplies causal mask / position embeddings /
    cache exactly as it would for an ordinary reference pass."""
    def hook(_module, args, kwargs):
        if args:
            hs = args[0]
            rest_args = args[1:]
            hs = hs.clone()
            hs[:, position, :] = new_vector.to(hs.dtype).to(hs.device)
            return (hs, *rest_args), kwargs
        hs = kwargs["hidden_states"].clone()
        hs[:, position, :] = new_vector.to(hs.dtype).to(hs.device)
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = hs
        return args, new_kwargs

    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def eval_block_at_state(model, arch: str, ids: torch.Tensor, qp: int, layer_idx: int,
                         custom_vector: torch.Tensor) -> dict:
    """F_l(custom_vector) at position qp: run the REFERENCE forward pass
    (no eviction hook) with the block's input at qp overridden, capturing
    the block's own output plus (secondary) its internal attn/mlp outputs
    at that same position, all at query position qp."""
    block = get_block(model, arch, layer_idx)
    attn_m, mlp_m = get_attn_mlp(model, arch, layer_idx)
    capture: dict = {}
    with override_position_hook(block, qp, custom_vector):
        with capture_output_hook(block, capture, "out"), \
             capture_output_hook(attn_m, capture, "attn"), \
             capture_output_hook(mlp_m, capture, "mlp"):
            with torch.no_grad():
                model(ids)
    return dict(
        out=capture["out"][0, qp, :].double().clone(),
        attn=capture["attn"][0, qp, :].double().clone(),
        mlp=capture["mlp"][0, qp, :].double().clone(),
    )


# ---------------------------------------------------------------------------
# Part III -- case selection, from already-labeled DPH categories only
# ---------------------------------------------------------------------------

def category_group(cat: str) -> str:
    if cat == "immediate_consistent":
        return "stable"
    if cat in ("single_flip", "multiple_flip"):
        return "flip"
    return "unresolved"  # never_consistent, unresolved


def select_cases(dph_cases: list[dict], rng: np.random.Generator) -> list[dict]:
    cells: dict[tuple, list[dict]] = {}
    for c in dph_cases:
        if c["depth_tertile"] not in ("early", "mid"):
            continue  # Part III: early/mid depths only
        if c["n_downstream_layers"] < 2:
            continue  # need l* strictly beyond the eviction layer
        key = (c["arch"], category_group(c["category"]), c["depth_tertile"])
        cells.setdefault(key, []).append(c)
    for k in cells:
        rng.shuffle(cells[k])

    selected = []
    selection_log = {}
    for key, pool in cells.items():
        take = pool[:N_PER_CELL]
        selected.extend(take)
        selection_log["_".join(key)] = dict(available=len(pool), selected=len(take))
    rng.shuffle(selected)
    return selected, selection_log


def choose_measurement_layer(c: dict, n_lay: int) -> int:
    """Part X: block immediately before the sign reversal (flip cases);
    matched downstream depth ~30% into the remaining depth (stable/
    unresolved cases). Uses only already-computed fields (eviction layer,
    n_downstream_layers, emergence_layer) -- no new metric."""
    layer = c["layer"]
    group = category_group(c["category"])
    if group == "flip" and c["emergence_layer"] is not None and c["emergence_layer"] > layer:
        l_star = c["emergence_layer"] - 1
    else:
        l_star = layer + max(1, round(0.3 * c["n_downstream_layers"]))
    l_star = max(layer + 1, min(l_star, n_lay - 1))
    return l_star


# ---------------------------------------------------------------------------
# Part V/VI/VII -- per-case computation
# ---------------------------------------------------------------------------

def run_case(model, adapter, tok, arch: str, c: dict) -> dict:
    layer, head, i, j, p_idx = c["layer"], c["head"], c["i"], c["j"], c["prompt"]
    sd = get_stratum_data(model, adapter, tok, arch, p_idx, layer, head, i, j)
    n_lay = n_layers(model, arch)
    trace_layers = list(range(layer, n_lay))
    qp = sd["qp"]

    mass_block = torch.zeros(sd["n"], dtype=torch.bool)
    for idx in sd["mass_rank"][: sd["change_k"]]:
        mass_block[idx] = True
    local_block = torch.zeros(sd["n"], dtype=torch.bool)
    for idx in sd["local_rank"][: sd["change_k"]]:
        local_block[idx] = True

    ref = run_traced_forward(model, adapter, arch, sd, layer, head, None, trace_layers)
    mass = run_traced_forward(model, adapter, arch, sd, layer, head, mass_block, trace_layers)
    local = run_traced_forward(model, adapter, arch, sd, layer, head, local_block, trace_layers)

    l_star = choose_measurement_layer(c, n_lay)
    # input to block l_star = output of block l_star - 1 (captured, since l_star-1 >= layer)
    h_ref = ref["hidden"][l_star - 1]
    h_mass = mass["hidden"][l_star - 1]
    h_local = local["hidden"][l_star - 1]
    delta_m = h_mass - h_ref
    delta_l = h_local - h_ref

    # --- validation: swapping a state's OWN natural value back in must
    # reproduce the already-captured natural output of block l_star exactly.
    # h_ref is the clean, decisive case: the reference forward pass supplies
    # EVERY position (including qp) from the reference trajectory itself, so
    # substituting h_ref back in at qp changes nothing and the block's own
    # output must match exactly. (h_mass/h_local are NOT expected to
    # reproduce mass["hidden"][l_star]/local["hidden"][l_star] this way --
    # eval_block_at_state always runs on the REFERENCE forward pass, so
    # positions other than qp still carry reference values, whereas the
    # mass/local trajectories differ at those other positions too; that
    # mismatch is expected and is not what this validation checks.)
    def response(base_vec, delta_vec, scale):
        return eval_block_at_state(model, arch, sd["ids"], qp, l_star, base_vec + scale * delta_vec)

    base_resp = dict(ref=response(h_ref, h_ref * 0, 0.0), mass=response(h_mass, h_mass * 0, 0.0),
                      local=response(h_local, h_local * 0, 0.0))
    val_err_ref = float(torch.linalg.norm(base_resp["ref"]["out"] - ref["hidden"][l_star]))

    scale_results = {}
    for delta_name, delta_vec, own_base_name, own_base_vec, cross_base_name, cross_base_vec in (
        ("delta_m", delta_m, "ref", h_ref, "local", h_local),
        ("delta_l", delta_l, "ref", h_ref, "mass", h_mass),
    ):
        for scale in SCALES:
            own_r = response(own_base_vec, delta_vec, scale)
            cross_r = response(cross_base_vec, delta_vec, scale)
            own_resp_diff = own_r["out"] - base_resp[own_base_name]["out"]
            cross_resp_diff = cross_r["out"] - base_resp[cross_base_name]["out"]
            own_attn_diff = own_r["attn"] - base_resp[own_base_name]["attn"]
            cross_attn_diff = cross_r["attn"] - base_resp[cross_base_name]["attn"]
            own_mlp_diff = own_r["mlp"] - base_resp[own_base_name]["mlp"]
            cross_mlp_diff = cross_r["mlp"] - base_resp[cross_base_name]["mlp"]

            numerator = float(torch.linalg.norm(own_resp_diff - cross_resp_diff))
            delta_norm = float(torch.linalg.norm(scale * delta_vec))
            S = numerator / (delta_norm + EPS_NORM)

            numerator_attn = float(torch.linalg.norm(own_attn_diff - cross_attn_diff))
            numerator_mlp = float(torch.linalg.norm(own_mlp_diff - cross_mlp_diff))

            # linearity control: c*[F(base+delta)-F(base)] at scale=1.0 vs actual response at this scale
            scale_results[f"{delta_name}_c{scale}"] = dict(
                delta_name=delta_name, scale=scale,
                own_base=own_base_name, cross_base=cross_base_name,
                numerator=numerator, delta_norm=delta_norm, S=S,
                numerator_attn=numerator_attn, numerator_mlp=numerator_mlp,
                own_resp_norm=float(torch.linalg.norm(own_resp_diff)),
                cross_resp_norm=float(torch.linalg.norm(cross_resp_diff)),
            )

    # linearity check: compare scale=1.0 own-response, scaled by c, to actual response at scale c
    linearity = {}
    for delta_name in ("delta_m", "delta_l"):
        r1 = scale_results[f"{delta_name}_c1.0"]["own_resp_norm"]
        for scale in (0.25, 0.5):
            r_c = scale_results[f"{delta_name}_c{scale}"]["own_resp_norm"]
            predicted_linear = scale * r1
            linearity[f"{delta_name}_c{scale}"] = dict(
                actual=r_c, predicted_linear=predicted_linear,
                rel_nonlinearity=abs(r_c - predicted_linear) / (predicted_linear + EPS_NORM),
            )

    return dict(
        arch=arch, prompt=p_idx, layer=layer, head=head, i=i, j=j,
        eviction_depth_fraction=c["eviction_depth_fraction"], depth_tertile=c["depth_tertile"],
        final_winner=c["final_winner"], advantage_final=c["advantage_final"],
        residual_category=c["category"], category_group=category_group(c["category"]),
        l_star=l_star, n_downstream_layers=c["n_downstream_layers"],
        validation_err_ref=val_err_ref,
        scale_results=scale_results, linearity=linearity,
    )


def run() -> dict:
    torch.manual_seed(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SELECTION_SEED)

    dph_cases = json.loads(DPH_SAMPLE.read_text())
    selected, selection_log = select_cases(dph_cases, rng)
    print(f"selected {len(selected)} cases (seed={SELECTION_SEED})")
    print("selection cells:", selection_log)

    tok_p, model_p, adapter_p = load_pythia()
    tok_g, model_g, adapter_g = load_gpt2()

    cases = []
    n_errors = 0
    for idx, c in enumerate(selected):
        model, adapter, tok = (model_p, adapter_p, tok_p) if c["arch"] == "pythia" else (model_g, adapter_g, tok_g)
        try:
            cases.append(run_case(model, adapter, tok, c["arch"], c))
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"  [{c['arch']}] SKIPPED prompt={c['prompt']} layer={c['layer']} head={c['head']}: {e}")
        print(f"[{idx+1}/{len(selected)}] done", flush=True)

    print(f"n_errors: {n_errors}")
    (RESULTS_DIR / "cases.json").write_text(json.dumps(cases, indent=2))

    summary = analyze(cases, selection_log)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, cases)
    print("Done. See results/state_dependent_propagation_audit/report.txt")
    return summary


def analyze(cases: list[dict], selection_log: dict) -> dict:
    max_val_err = max((c["validation_err_ref"] for c in cases), default=0.0)

    # primary S at scale=1.0, per case (average of delta_m and delta_l)
    def case_S1(c):
        sm = c["scale_results"]["delta_m_c1.0"]["S"]
        sl = c["scale_results"]["delta_l_c1.0"]["S"]
        return 0.5 * (sm + sl)

    def case_S1_num(c):
        sm = c["scale_results"]["delta_m_c1.0"]["numerator"]
        sl = c["scale_results"]["delta_l_c1.0"]["numerator"]
        return 0.5 * (sm + sl)

    def case_own_resp_norm(c):
        return 0.5 * (c["scale_results"]["delta_m_c1.0"]["own_resp_norm"] + c["scale_results"]["delta_l_c1.0"]["own_resp_norm"])

    groups = {}
    for grp in ("stable", "flip", "unresolved"):
        gcases = [c for c in cases if c["category_group"] == grp]
        S_vals = [case_S1(c) for c in gcases]
        groups[grp] = dict(
            n=len(gcases),
            median_S=float(np.median(S_vals)) if S_vals else None,
            mean_S=float(np.mean(S_vals)) if S_vals else None,
            iqr_S=[float(np.percentile(S_vals, 25)), float(np.percentile(S_vals, 75))] if S_vals else None,
            values=S_vals,
        )

    by_arch_group = {}
    for arch in ("pythia", "gpt2"):
        by_arch_group[arch] = {}
        for grp in ("stable", "flip", "unresolved"):
            gcases = [c for c in cases if c["arch"] == arch and c["category_group"] == grp]
            S_vals = [case_S1(c) for c in gcases]
            by_arch_group[arch][grp] = dict(
                n=len(gcases), median_S=float(np.median(S_vals)) if S_vals else None, values=S_vals,
            )

    # effect size: simple rank-biserial-ish via median ratio, flip vs stable
    def median_ratio(a_vals, b_vals):
        if not a_vals or not b_vals:
            return None
        ma, mb = np.median(a_vals), np.median(b_vals)
        return float(ma / mb) if mb != 0 else None

    flip_vs_stable_ratio = median_ratio(groups["flip"]["values"], groups["stable"]["values"])

    # linearity summary
    rel_nonlin = {"delta_m_c0.25": [], "delta_m_c0.5": [], "delta_l_c0.25": [], "delta_l_c0.5": []}
    for c in cases:
        for k in rel_nonlin:
            rel_nonlin[k].append(c["linearity"][k]["rel_nonlinearity"])
    linearity_summary = {k: dict(median=float(np.median(v)), mean=float(np.mean(v))) for k, v in rel_nonlin.items()}

    # component: attn vs mlp share of the state-dependence numerator, at scale 1.0
    comp = dict(attn_larger=0, mlp_larger=0, comparable=0, n=0)
    for c in cases:
        for key in ("delta_m_c1.0", "delta_l_c1.0"):
            r = c["scale_results"][key]
            a, m = r["numerator_attn"], r["numerator_mlp"]
            comp["n"] += 1
            if a > 1.5 * m:
                comp["attn_larger"] += 1
            elif m > 1.5 * a:
                comp["mlp_larger"] += 1
            else:
                comp["comparable"] += 1

    strongest_support = max(cases, key=case_S1, default=None)
    strongest_counter = min(cases, key=case_S1, default=None)

    # verdict
    architectures_present = {c["arch"] for c in cases}
    flip_gt_stable_both_arch = True
    for arch in architectures_present:
        f = by_arch_group.get(arch, {}).get("flip", {}).get("median_S")
        s = by_arch_group.get(arch, {}).get("stable", {}).get("median_S")
        if f is None or s is None or not (f > s):
            flip_gt_stable_both_arch = False

    avg_rel_nonlin_at_1 = None  # scale=1.0 nonlinearity not directly comparable (predicted=actual by construction);
    # use the shrink-with-c pattern instead: does S shrink as c decreases?
    def median_S_at_scale(grp, scale):
        vals = []
        for c in cases:
            if c["category_group"] != grp:
                continue
            vals.append(0.5 * (c["scale_results"][f"delta_m_c{scale}"]["S"] + c["scale_results"][f"delta_l_c{scale}"]["S"]))
        return float(np.median(vals)) if vals else None

    S_shrinks_with_scale = {}
    for grp in ("stable", "flip", "unresolved"):
        S_shrinks_with_scale[grp] = dict(
            c0_25=median_S_at_scale(grp, 0.25), c0_5=median_S_at_scale(grp, 0.5), c1_0=median_S_at_scale(grp, 1.0),
        )

    if max_val_err > VALIDATION_TOL:
        verdict = "INVALID -- validation failed, see report"
    elif flip_vs_stable_ratio is not None and flip_vs_stable_ratio > 1.5 and flip_gt_stable_both_arch:
        # check it's not just large-scale nonlinearity
        flip_shrink = S_shrinks_with_scale["flip"]
        shrinks_a_lot = (flip_shrink["c0_25"] is not None and flip_shrink["c1_0"] is not None
                          and flip_shrink["c0_25"] < 0.3 * flip_shrink["c1_0"])
        verdict = "SDPH-B" if shrinks_a_lot else "SDPH-C"
    else:
        verdict = "SDPH-A" if (flip_vs_stable_ratio is not None and flip_vs_stable_ratio < 1.2) else "SDPH-B"

    return dict(
        seed=SELECTION_SEED, n_cases=len(cases), selection_log=selection_log,
        max_validation_err=max_val_err,
        groups=groups, by_arch_group=by_arch_group,
        flip_vs_stable_median_ratio=flip_vs_stable_ratio,
        linearity_summary=linearity_summary,
        S_shrinks_with_scale=S_shrinks_with_scale,
        component_audit=comp,
        strongest_support_case={k: v for k, v in strongest_support.items() if k not in ("scale_results",)} if strongest_support else None,
        strongest_counterexample_case={k: v for k, v in strongest_counter.items() if k not in ("scale_results",)} if strongest_counter else None,
        verdict=verdict,
    )


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v) for v in x) + "]"
    return str(x)


def write_report(summary: dict, cases: list[dict]) -> None:
    L = []
    A = L.append
    W = 100
    A("STATE-DEPENDENT PROPAGATION HYPOTHESIS (SDPH) AUDIT")
    A("=" * W)
    A("")
    A(f"seed={summary['seed']}  n_cases={summary['n_cases']}")
    A("selection cells (arch_categorygroup_depthtertile: available/selected):")
    for k, v in summary["selection_log"].items():
        A(f"  {k}: {v}")

    A("")
    A("-" * W)
    A("VALIDATION (Part IV): swapping a state's OWN natural value back in must reproduce")
    A("the already-captured block output exactly")
    A("-" * W)
    A(f"max validation error across all {summary['n_cases']} cases: {summary['max_validation_err']:.3e}")
    A(f"(tolerance {VALIDATION_TOL:.1e}; {'PASSED' if summary['max_validation_err'] <= VALIDATION_TOL else 'FAILED -- see verdict'})")

    A("")
    A("-" * W)
    A("PRIMARY STATE-DEPENDENCE DIAGNOSTIC S (scale=1.0), BY CASE CLASS")
    A("-" * W)
    for grp in ("stable", "flip", "unresolved"):
        g = summary["groups"][grp]
        A(f"[{grp}] n={g['n']}  median_S={_f(g['median_S'])}  mean_S={_f(g['mean_S'])}  IQR={_f(g['iqr_S'])}")
        A(f"    values={_f(g['values'])}")

    A("")
    A(f"flip / stable median-S ratio: {_f(summary['flip_vs_stable_median_ratio'])}")

    A("")
    A("-" * W)
    A("ARCHITECTURE-SEPARATED")
    A("-" * W)
    for arch in ("pythia", "gpt2"):
        A(f"[{arch}]")
        for grp in ("stable", "flip", "unresolved"):
            g = summary["by_arch_group"][arch][grp]
            A(f"    {grp:<11} n={g['n']}  median_S={_f(g['median_S'])}")

    A("")
    A("-" * W)
    A("LINEARITY CONTROL (Part VII): does S shrink toward zero as c -> 0?")
    A("-" * W)
    for grp in ("stable", "flip", "unresolved"):
        s = summary["S_shrinks_with_scale"][grp]
        A(f"  [{grp}] median_S at c=0.25: {_f(s['c0_25'])}   c=0.5: {_f(s['c0_5'])}   c=1.0: {_f(s['c1_0'])}")
    A("")
    A("Relative nonlinearity of the OWN-state response (actual vs. c times the full-scale response):")
    for k, v in summary["linearity_summary"].items():
        A(f"  {k}: median rel. nonlinearity = {_f(v['median'])}")

    A("")
    A("-" * W)
    A("COMPONENT AUDIT (secondary, descriptive): attention vs MLP share of the state-dependence numerator")
    A("-" * W)
    ca = summary["component_audit"]
    A(f"n={ca['n']}  attn_larger={ca['attn_larger']} ({_f(ca['attn_larger']/max(1,ca['n']))})  "
      f"mlp_larger={ca['mlp_larger']} ({_f(ca['mlp_larger']/max(1,ca['n']))})  "
      f"comparable={ca['comparable']} ({_f(ca['comparable']/max(1,ca['n']))})")

    A("")
    A("-" * W)
    A("STRONGEST EVIDENCE FOR SDPH (largest state-dependence S)")
    A("-" * W)
    s = summary["strongest_support_case"]
    if s:
        A(f"[{s['arch']}] prompt={s['prompt']} layer={s['layer']} head={s['head']} l_star={s['l_star']} "
          f"category_group={s['category_group']}  final_winner={s['final_winner']}  advantage={s['advantage_final']:.4e}")

    A("")
    A("-" * W)
    A("STRONGEST COUNTEREXAMPLE (smallest state-dependence S)")
    A("-" * W)
    s = summary["strongest_counterexample_case"]
    if s:
        A(f"[{s['arch']}] prompt={s['prompt']} layer={s['layer']} head={s['head']} l_star={s['l_star']} "
          f"category_group={s['category_group']}  final_winner={s['final_winner']}  advantage={s['advantage_final']:.4e}")

    A("")
    A("-" * W)
    A("VERDICT")
    A("-" * W)
    A(summary["verdict"])

    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
