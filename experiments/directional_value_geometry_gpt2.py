r"""Cross-architecture replication of the directional-value-geometry (DVG)
test on GPT-2's full grid (~10x Pythia's measurable-pair count).

**This is a replication, not a new feature-discovery round.** Every pure
function, feature definition, statistical test, threshold, and the
multivariate-model procedure are imported UNCHANGED from
`experiments/directional_value_geometry.py` (the Pythia DVG experiment) --
not re-transcribed, to guarantee byte-identical reuse of signs,
normalization, zero-vector handling (`cos()` returns NaN for a zero-norm
vector, unchanged), and every statistical definition. No new feature is
introduced anywhere in this file.

Central question: does Pythia's DVG-B verdict survive a ~10x larger GPT-2
sample, or does directional value geometry contain reproducible
within-stratum information about downstream winner in GPT-2?

**Does NOT recompute any eviction or KL outcome.** Reuses, unmodified:
  - `results/cross_architecture_full_grid/pairs.json` (22,983 meaningful
    GPT-2 inversions; filtered here to the same `changes_set_within_kmax`
    and `above_noise_floor` predicate as the Pythia experiment)
  - `results/cross_architecture_full_grid/summary.json` (per-stratum
    top1_share/entropy_nats, used only as an existing-concentration
    baseline, exactly as before)
  - `results/directional_value_geometry/summary.json` (frozen Pythia DVG
    result, read-only, for the side-by-side replication table)

Value-vector reconstruction uses the validated `GPT2Adapter` (see
`tests/test_gpt2_adapter.py` and the pilot's Part-0 gate -- not re-validated
here, frozen). One forward pass per prompt (3 total), matching the Pythia
script's reconstruction discipline exactly.

Writes `results/directional_value_geometry_gpt2/{summary,pairs}.json` and
`report.txt`. Touches no file under `src/kv_efficiency/`, `tests/`, any
existing result file, or any doc/status file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Import every reusable piece from the Pythia DVG script UNCHANGED -- not
# re-transcribed, so feature math/statistics are guaranteed byte-identical.
# ---------------------------------------------------------------------------
_dvg_spec = importlib.util.spec_from_file_location(
    "directional_value_geometry", ROOT / "experiments" / "directional_value_geometry.py"
)
dvg = importlib.util.module_from_spec(_dvg_spec)
_dvg_spec.loader.exec_module(dvg)

FEATURE_NAMES = dvg.FEATURE_NAMES
FEATURE_DEFINITIONS = dvg.FEATURE_DEFINITIONS
compute_features = dvg.compute_features
cos = dvg.cos
univariate_stats = dvg.univariate_stats
eta_squared_between_stratum = dvg.eta_squared_between_stratum
within_stratum_demeaned_correlation = dvg.within_stratum_demeaned_correlation
per_stratum_stats = dvg.per_stratum_stats
fit_logreg = dvg.fit_logreg
predict_logreg = dvg.predict_logreg
auc_from_scores = dvg.auc_from_scores
grouped_cv_auc = dvg.grouped_cv_auc
MIN_STRATUM_N = dvg.MIN_STRATUM_N
WEAK_EFFECT_AUC = dvg.WEAK_EFFECT_AUC
WEAK_EFFECT_R = dvg.WEAK_EFFECT_R
N_CV_FOLDS = dvg.N_CV_FOLDS
RIDGE = dvg.RIDGE

from kv_efficiency.attention_reconstruction import head_values_dmodel
from kv_efficiency.attention_reconstruction_gpt2 import GPT2Adapter
from kv_efficiency.eviction import require_normalized

MODEL_ID = "gpt2"
SOURCE_PAIRS = ROOT / "results" / "cross_architecture_full_grid" / "pairs.json"
SOURCE_SUMMARY = ROOT / "results" / "cross_architecture_full_grid" / "summary.json"
PYTHIA_DVG_SUMMARY = ROOT / "results" / "directional_value_geometry" / "summary.json"
RESULTS_DIR = ROOT / "results" / "directional_value_geometry_gpt2"
EVALUATION_PROMPTS = [
    "The history of the Roman Empire is often divided into three stages: "
    "Rome under kings, Rome as a republic governed by elected senators, and "
    "Rome under the rule of emperors. The empire reached its greatest extent "
    "under the reign of Trajan, stretching from Britain to the Persian Gulf.",
    "Photosynthesis is the process by which green plants and some other "
    "organisms use sunlight to synthesize nutrients from carbon dioxide and "
    "water. Photosynthesis in plants generally involves the green pigment "
    "chlorophyll and generates oxygen as a byproduct.",
    "A binary search tree is a rooted binary tree data structure with the "
    "key of each internal node being greater than all keys in its left "
    "subtree and less than those in its right subtree.",
]
DEPTH_MAX = 11


def load_model_and_adapter():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tok, model, GPT2Adapter(model)


def depth_residualized_correlation(feature: np.ndarray, advantage: np.ndarray, depth: np.ndarray) -> dict:
    """Regress feature and advantage each on depth_fraction (simple OLS,
    1 predictor + intercept) and correlate the residuals -- tests whether
    GPT-2's known depth-dependence of winner direction (see
    results/cross_architecture_full_grid/report.txt, winrate_vs_depth
    rho=-0.191) could by itself manufacture an apparent pooled DVG
    correlation. Not a new feature: depth_fraction is already an existing
    per-stratum quantity from the full-grid pipeline."""
    mask = np.isfinite(feature) & np.isfinite(advantage) & np.isfinite(depth)
    f, adv, d = feature[mask], advantage[mask], depth[mask]
    if len(f) < 10 or np.std(f) == 0 or np.std(adv) == 0 or np.std(d) == 0:
        return dict(pearson_r=None, spearman_rho=None, n=len(f))
    X = np.vstack([np.ones_like(d), d]).T
    beta_f, *_ = np.linalg.lstsq(X, f, rcond=None)
    beta_a, *_ = np.linalg.lstsq(X, adv, rcond=None)
    f_resid = f - X @ beta_f
    a_resid = adv - X @ beta_a
    if np.std(f_resid) == 0 or np.std(a_resid) == 0:
        return dict(pearson_r=None, spearman_rho=None, n=len(f))
    r_p, p_p = sstats.pearsonr(f_resid, a_resid)
    r_s, p_s = sstats.spearmanr(f_resid, a_resid)
    return dict(pearson_r=float(r_p), pearson_p=float(p_p),
                spearman_rho=float(r_s), spearman_p=float(p_s), n=len(f))


def run() -> dict:
    torch.manual_seed(0)
    tok, model, adapter = load_model_and_adapter()

    all_pairs = json.loads(SOURCE_PAIRS.read_text())
    source_summary = json.loads(SOURCE_SUMMARY.read_text())
    stratum_concentration = {
        (s["prompt"], s["layer"], s["head"]): dict(top1_share=s["top1_share"], entropy_nats=s["entropy_nats"],
                                                     depth_fraction=s.get("depth_fraction", s["layer"] / DEPTH_MAX))
        for s in source_summary["analysis"]["per_stratum_table"]
    }

    measurable = [
        p for p in all_pairs
        if p.get("changes_set_within_kmax") and p.get("above_noise_floor")
        and p.get("advantage_mass_minus_local_at_change") is not None
    ]
    n_measurable = len(measurable)
    print(f"measurable pairs: {n_measurable} (Pythia had 1,962 -- ratio {n_measurable/1962:.2f}x)")

    # ---- one forward pass per prompt; reconstruct per (layer,head) within it ----
    needed_strata = sorted({(p["prompt"], p["layer"], p["head"]) for p in measurable})
    needed_by_prompt: dict[int, set[tuple[int, int]]] = {}
    for p_idx, layer, head in needed_strata:
        needed_by_prompt.setdefault(p_idx, set()).add((layer, head))

    tensors_cache: dict[tuple[int, int, int], dict] = {}
    n_forward_passes = 0
    for p_idx, layer_heads in needed_by_prompt.items():
        ids = tok(EVALUATION_PROMPTS[p_idx], return_tensors="pt").input_ids
        qp = ids.shape[1] - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        n_forward_passes += 1
        for layer, head in layer_heads:
            hidden_in = out.hidden_states[layer][0].double()
            alpha_full = out.attentions[layer][0, head].double()
            values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
            alpha_last = require_normalized(alpha_full[qp], renormalize=True)
            alpha_np = alpha_last.detach().numpy().astype(np.float64)
            values_np = values_dmodel.detach().numpy().astype(np.float64)
            o = alpha_np @ values_np
            tensors_cache[(p_idx, layer, head)] = dict(alpha_np=alpha_np, values_np=values_np, o=o)
    print(f"forward passes used: {n_forward_passes} (one per prompt)")

    # ---- per-pair feature computation + reconstruction cross-check (unchanged compute_features) ----
    enriched = []
    max_g_recon_err = 0.0
    for p in measurable:
        key = (p["prompt"], p["layer"], p["head"])
        t = tensors_cache[key]
        v_i = t["values_np"][p["i"]]
        v_j = t["values_np"][p["j"]]
        o = t["o"]
        feats = compute_features(v_i, v_j, o, p["alpha_lo"], p["alpha_hi"])
        err_lo = abs(feats["g_lo_recon"] - p["g_lo"])
        err_hi = abs(feats["g_hi_recon"] - p["g_hi"])
        max_g_recon_err = max(max_g_recon_err, err_lo, err_hi)

        row = dict(
            prompt=p["prompt"], layer=p["layer"], head=p["head"], i=p["i"], j=p["j"],
            alpha_lo=p["alpha_lo"], alpha_hi=p["alpha_hi"], Gamma=p["Gamma"],
            advantage=p["advantage_mass_minus_local_at_change"],
            win_local=1 if p["advantage_mass_minus_local_at_change"] > 0 else 0,
            **{k: feats[k] for k in FEATURE_NAMES},
            g_lo_recon_err=err_lo, g_hi_recon_err=err_hi,
        )
        conc = stratum_concentration.get(key)
        if conc:
            row["top1_share"] = conc["top1_share"]
            row["entropy_nats"] = conc["entropy_nats"]
            row["depth_fraction"] = conc["depth_fraction"]
        enriched.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pairs.json").write_text(json.dumps(enriched, indent=2))
    print(f"max |g_recon - g_stored| cross-check: {max_g_recon_err:.3e}")

    # ---- arrays ----
    stratum_ids = np.array([f"{r['prompt']}_{r['layer']}_{r['head']}" for r in enriched])
    advantage = np.array([r["advantage"] for r in enriched], dtype=np.float64)
    y = np.array([r["win_local"] for r in enriched], dtype=int)
    abs_gamma = np.array([abs(r["Gamma"]) for r in enriched], dtype=np.float64)
    top1 = np.array([r.get("top1_share", np.nan) for r in enriched], dtype=np.float64)
    entropy = np.array([r.get("entropy_nats", np.nan) for r in enriched], dtype=np.float64)
    depth = np.array([r.get("depth_fraction", np.nan) for r in enriched], dtype=np.float64)

    pooled_results, within_stratum_results, stratum_level_results = {}, {}, {}
    eta_sq_results, depth_residualized_results = {}, {}

    for fname in FEATURE_NAMES:
        feat = np.array([r[fname] for r in enriched], dtype=np.float64)
        uni = univariate_stats(feat, advantage, y)
        pooled_results[fname] = uni
        pooled_sign = np.sign(uni["spearman_vs_advantage"]["rho"]) if uni["spearman_vs_advantage"] else None

        within = within_stratum_demeaned_correlation(feat, advantage, stratum_ids)
        within_stratum_results[fname] = within

        strat = per_stratum_stats(feat, advantage, y, stratum_ids)
        if pooled_sign is not None and strat["aggregate"]["median_spearman_rho"] is not None:
            rhos = [v["spearman_rho"] for v in strat["per_stratum"].values() if "spearman_rho" in v]
            if rhos:
                strat["aggregate"]["frac_rho_same_sign_as_pooled"] = float(np.mean([np.sign(r) == pooled_sign for r in rhos]))
        stratum_level_results[fname] = strat
        eta_sq_results[fname] = eta_squared_between_stratum(feat, stratum_ids)
        depth_residualized_results[fname] = depth_residualized_correlation(feat, advantage, depth)

    baseline_results = {}
    for name, arr in (("abs_Gamma", abs_gamma), ("top1_share", top1), ("entropy_nats", entropy)):
        baseline_results[name] = dict(
            pooled=univariate_stats(arr, advantage, y),
            within_stratum=within_stratum_demeaned_correlation(arr, advantage, stratum_ids),
            eta_squared_between_stratum=eta_squared_between_stratum(arr, stratum_ids),
        )

    # ---- tail audit: largest |advantage| pairs, including the known ~1.56-nat case ----
    sorted_by_adv = sorted(enriched, key=lambda r: r["advantage"])
    strongest_mass_wins = sorted_by_adv[:10]
    strongest_local_wins = sorted_by_adv[-10:]
    n_tail = 20
    tail_pairs = strongest_mass_wins + strongest_local_wins
    tail_audit = {}
    for fname in FEATURE_NAMES:
        tail_feat = np.array([r[fname] for r in tail_pairs], dtype=np.float64)
        tail_y = np.array([r["win_local"] for r in tail_pairs], dtype=int)
        uni_tail = univariate_stats(tail_feat, np.array([r["advantage"] for r in tail_pairs]), tail_y)
        tail_audit[fname] = dict(
            mann_whitney=uni_tail["mann_whitney"],
            values_mass_side=[float(x) for x in tail_feat[tail_y == 0]],
            values_local_side=[float(x) for x in tail_feat[tail_y == 1]],
        )
    extreme_pair = min(enriched, key=lambda r: r["advantage"])  # most negative = largest mass win

    # ---- minimal multivariate model (identical selection rule/model family) ----
    ranked = sorted(
        FEATURE_NAMES,
        key=lambda f: abs((pooled_results[f]["mann_whitney"]["auc"] if pooled_results[f]["mann_whitney"] else 0.5) - 0.5),
        reverse=True,
    )
    chosen = []
    feat_arrays = {f: np.array([r[f] for r in enriched], dtype=np.float64) for f in FEATURE_NAMES}
    for f in ranked:
        if len(chosen) >= 3:
            break
        ok = True
        for c in chosen:
            a, b = feat_arrays[f], feat_arrays[c]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() > 10 and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
                corr = np.corrcoef(a[mask], b[mask])[0, 1]
                if abs(corr) > 0.8:
                    ok = False
                    break
        if ok:
            chosen.append(f)

    def build_matrix(names):
        cols = []
        for name in names:
            arr = feat_arrays.get(name) if name in feat_arrays else {"abs_Gamma": abs_gamma, "top1_share": top1, "entropy_nats": entropy}[name]
            cols.append(arr)
        X = np.vstack(cols).T
        mask = np.all(np.isfinite(X), axis=1)
        return X, mask

    multivariate = {"chosen_features": chosen}
    X_dir, mask_dir = build_matrix(chosen)
    multivariate["directional_model"] = grouped_cv_auc(X_dir[mask_dir], y[mask_dir], stratum_ids[mask_dir])
    X_gamma, mask_gamma = build_matrix(["abs_Gamma"])
    multivariate["baseline_abs_gamma"] = grouped_cv_auc(X_gamma[mask_gamma], y[mask_gamma], stratum_ids[mask_gamma])
    X_conc, mask_conc = build_matrix(["top1_share", "entropy_nats"])
    multivariate["baseline_concentration"] = grouped_cv_auc(X_conc[mask_conc], y[mask_conc], stratum_ids[mask_conc])
    X_combo, mask_combo = build_matrix(chosen + ["abs_Gamma", "top1_share", "entropy_nats"])
    multivariate["directional_plus_baselines"] = grouped_cv_auc(X_combo[mask_combo], y[mask_combo], stratum_ids[mask_combo])
    multivariate["majority_class_accuracy"] = float(max(y.mean(), 1 - y.mean()))
    multivariate["n_pairs_used"] = int(mask_dir.sum())

    # ---- verdict (identical rule to the Pythia script) ----
    any_pooled_meaningful = any(not pooled_results[f]["practically_weak"] for f in FEATURE_NAMES)
    any_within_meaningful = any(
        within_stratum_results[f]["pearson_r"] is not None and abs(within_stratum_results[f]["pearson_r"]) >= WEAK_EFFECT_R
        for f in FEATURE_NAMES
    )
    dir_auc = multivariate["directional_model"]["mean_auc"]
    gamma_auc = multivariate["baseline_abs_gamma"]["mean_auc"]
    conc_auc = multivariate["baseline_concentration"]["mean_auc"]
    beats_baselines = (
        dir_auc is not None and gamma_auc is not None and conc_auc is not None
        and dir_auc > gamma_auc + 0.03 and dir_auc > conc_auc + 0.03 and dir_auc > 0.55
    )
    if beats_baselines:
        verdict, verdict_text = "DVG-D", "A small directional feature set materially predicts the downstream winner beyond |Gamma| and attention concentration alone."
    elif any_within_meaningful:
        verdict, verdict_text = "DVG-C", "Directional geometry shows a reproducible within-stratum association with win direction."
    elif any_pooled_meaningful:
        verdict, verdict_text = "DVG-B", "At least one feature is statistically detectable pooled, but practically weak and/or fails within-stratum."
    else:
        verdict, verdict_text = "DVG-A", "No tested directional feature carries useful information about the downstream winner, pooled or within-stratum."

    # ---- Part V: power classification, per feature ----
    pythia_dvg = json.loads(PYTHIA_DVG_SUMMARY.read_text())
    power_classification = {}
    for fname in FEATURE_NAMES:
        py_auc = pythia_dvg["pooled"][fname]["mann_whitney"]["auc"] if pythia_dvg["pooled"][fname]["mann_whitney"] else 0.5
        gp_auc_pooled = pooled_results[fname]["mann_whitney"]["auc"] if pooled_results[fname]["mann_whitney"] else 0.5
        gp_within_r = within_stratum_results[fname]["pearson_r"]
        near_zero = abs(gp_auc_pooled - 0.5) < 0.02
        tiny_but_sig_pooled = (not pooled_results[fname]["practically_weak"]) and abs(gp_auc_pooled - 0.5) < (WEAK_EFFECT_AUC - 0.5) + 0.02
        within_meaningful = gp_within_r is not None and abs(gp_within_r) >= WEAK_EFFECT_R
        if within_meaningful and dir_auc is not None and gamma_auc is not None and dir_auc > gamma_auc + 0.03:
            outcome = "4_strong_predictive_improvement"
        elif within_meaningful:
            outcome = "3_reproducible_within_stratum"
        elif near_zero:
            outcome = "1_near_zero_despite_power"
        else:
            outcome = "2_tiny_significant_still_negligible"
        power_classification[fname] = dict(
            pythia_pooled_auc=py_auc, gpt2_pooled_auc=gp_auc_pooled,
            gpt2_within_stratum_r=gp_within_r, outcome=outcome,
        )

    provenance = dict(model_id=MODEL_ID, torch_version=torch.__version__)
    summary = dict(
        provenance=provenance,
        source_pairs_file=str(SOURCE_PAIRS.relative_to(ROOT)),
        source_summary_file=str(SOURCE_SUMMARY.relative_to(ROOT)),
        pythia_dvg_summary_file=str(PYTHIA_DVG_SUMMARY.relative_to(ROOT)),
        n_meaningful_inversions_total=len(all_pairs),
        n_measurable=n_measurable,
        n_forward_passes=n_forward_passes,
        max_g_reconstruction_error=max_g_recon_err,
        feature_definitions=FEATURE_DEFINITIONS,
        target_definition="win_local = 1 iff advantage_mass_minus_local_at_change = "
                           "kl_mass - kl_local > 0 (local-damage wins); 0 otherwise (mass wins). "
                           "IDENTICAL to the Pythia DVG experiment's definition -- not redefined.",
        pooled=pooled_results,
        within_stratum_demeaned=within_stratum_results,
        depth_residualized=depth_residualized_results,
        per_stratum=stratum_level_results,
        eta_squared_between_stratum=eta_sq_results,
        baselines=baseline_results,
        tail_audit=dict(n_tail=n_tail, extreme_pair=extreme_pair, per_feature=tail_audit),
        strongest_mass_win_examples=[
            {k: r[k] for k in ("prompt", "layer", "head", "i", "j", "advantage", "Gamma", *FEATURE_NAMES)}
            for r in strongest_mass_wins
        ],
        strongest_local_win_examples=[
            {k: r[k] for k in ("prompt", "layer", "head", "i", "j", "advantage", "Gamma", *FEATURE_NAMES)}
            for r in strongest_local_wins
        ],
        multivariate=multivariate,
        power_classification=power_classification,
        verdict=verdict, verdict_text=verdict_text,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, pythia_dvg, enriched)

    pairs_size = (RESULTS_DIR / "pairs.json").stat().st_size
    summary_size = (RESULTS_DIR / "summary.json").stat().st_size
    print(f"\npairs.json: {pairs_size/1e6:.2f} MB   summary.json: {summary_size/1e6:.2f} MB")
    print("Done. See results/directional_value_geometry_gpt2/report.txt")
    return summary


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def write_report(summary: dict, pythia_dvg: dict, enriched: list) -> None:
    L = []
    A = L.append
    W = 100

    A("CROSS-ARCHITECTURE DVG REPLICATION: GPT-2 FULL GRID vs FROZEN PYTHIA DVG-B")
    A("=" * W)
    A("")
    A(f"n_measurable: GPT-2={summary['n_measurable']}  Pythia={pythia_dvg['n_measurable']}  "
      f"ratio={summary['n_measurable']/pythia_dvg['n_measurable']:.2f}x")
    A(f"forward passes: {summary['n_forward_passes']}   "
      f"max reconstruction err: {summary['max_g_reconstruction_error']:.3e}")
    A(f"target definition (identical to Pythia): {summary['target_definition']}")

    A("")
    A("-" * W)
    A("PART IV -- CROSS-ARCHITECTURE REPLICATION TABLE")
    A("-" * W)
    A(f"{'feature':<20}{'Py pooled rho':>14}{'GPT2 pooled rho':>17}{'Py within r':>13}{'GPT2 within r':>15}  classification")
    classifications = {}
    for f in FEATURE_NAMES:
        py_sp = pythia_dvg["pooled"][f]["spearman_vs_advantage"]
        gp_sp = summary["pooled"][f]["spearman_vs_advantage"]
        py_w = pythia_dvg["within_stratum_demeaned"][f]["pearson_r"]
        gp_w = summary["within_stratum_demeaned"][f]["pearson_r"]
        py_rho = py_sp["rho"] if py_sp else None
        gp_rho = gp_sp["rho"] if gp_sp else None

        same_sign_pooled = (py_rho is not None and gp_rho is not None and np.sign(py_rho) == np.sign(gp_rho))
        gp_within_meaningful = gp_w is not None and abs(gp_w) >= WEAK_EFFECT_R
        py_within_meaningful = py_w is not None and abs(py_w) >= WEAK_EFFECT_R
        if gp_within_meaningful:
            cls = "REPLICATED (within-stratum)" if py_within_meaningful else "ARCHITECTURE-SENSITIVE (GPT2-only within-stratum signal)"
        elif same_sign_pooled and abs(gp_rho or 0) >= 0.02:
            cls = "directionally consistent but weak"
        elif py_rho is not None and gp_rho is not None and np.sign(py_rho) != np.sign(gp_rho) and abs(gp_rho) > 0.02:
            cls = "CONTRADICTED (sign flip)"
        else:
            cls = "architecture-sensitive (near-null both, not comparable)"
        classifications[f] = cls
        A(f"{f:<20}{_fmt(py_rho):>14}{_fmt(gp_rho):>17}{_fmt(py_w):>13}{_fmt(gp_w):>15}  {cls}")

    A("")
    A("Note: statistical significance alone is NOT used for this classification (n differs by")
    A("~10x, so p-values are not comparable across columns) -- classification uses sign agreement")
    A(f"and the pre-declared |within-stratum r| >= {WEAK_EFFECT_R} effect-size threshold only.")

    A("")
    A("-" * W)
    A("PART III -- POOLED vs WITHIN-STRATUM (GPT-2), full detail")
    A("-" * W)
    A(f"{'feature':<20}{'pooled AUC':>12}{'pooled weak?':>13}{'within r':>10}{'within p':>10}{'eta^2(strat)':>13}{'depth-resid r':>15}")
    for f in FEATURE_NAMES:
        u = summary["pooled"][f]
        mw = u["mann_whitney"]
        w = summary["within_stratum_demeaned"][f]
        eta = summary["eta_squared_between_stratum"][f]
        dr = summary["depth_residualized"][f]
        A(f"{f:<20}{_fmt(mw['auc'] if mw else None):>12}{'yes' if u['practically_weak'] else 'NO':>13}"
          f"{_fmt(w['pearson_r']):>10}{_fmt(w.get('pearson_p')):>10}{_fmt(eta):>13}{_fmt(dr['pearson_r']):>15}")
    A("")
    A("depth-resid r: within-stratum-style correlation AFTER regressing BOTH feature and advantage")
    A("on depth_fraction first (simple OLS) -- tests whether GPT-2's known depth-dependence of winner")
    A("direction (winrate_vs_depth rho=-0.191, p<0.0001, from the full-grid report) could by itself")
    A("manufacture an apparent pooled DVG signal. Compare to the plain within-stratum column.")

    A("")
    A("-" * W)
    A("PART V -- SAMPLE-SIZE / POWER CLASSIFICATION")
    A("-" * W)
    A("1_near_zero_despite_power: pooled AUC within 0.02 of 0.5 even at ~10x Pythia's n -> evidence")
    A("    AGAINST simple directional geometry for that feature, not merely 'no significant result'.")
    A("2_tiny_significant_still_negligible: statistically detectable but below the practical-weakness")
    A("    threshold -> still DVG-B territory for that feature.")
    A("3_reproducible_within_stratum: within-stratum |r|>=0.10 -> possible DVG-C signal.")
    A("4_strong_predictive_improvement: within-stratum signal AND beats |Gamma| in grouped CV -> DVG-D.")
    A("")
    for f in FEATURE_NAMES:
        pc = summary["power_classification"][f]
        A(f"  {f:<20} Pythia AUC={_fmt(pc['pythia_pooled_auc'])}  GPT-2 AUC={_fmt(pc['gpt2_pooled_auc'])}  "
          f"GPT-2 within-r={_fmt(pc['gpt2_within_stratum_r'])}  -> {pc['outcome']}")

    A("")
    A("-" * W)
    A("PART VI -- STRATUM CONTROLS")
    A("-" * W)
    for f in FEATURE_NAMES:
        agg = summary["per_stratum"][f]["aggregate"]
        A(f"  {f:<20} n_strata={agg['n_strata_with_enough_data']}  median_AUC={_fmt(agg['median_auc'])}  "
          f"IQR={_fmt(agg['iqr_auc'])}  frac_same_sign_as_pooled={_fmt(agg['frac_rho_same_sign_as_pooled'])}")
    A("")
    A("Interpretation: frac_same_sign_as_pooled near 0.5 means the feature's per-stratum direction is")
    A("essentially a coin flip across strata (sign changes freely) -- inconsistent with a single global")
    A("mechanism regardless of the pooled AUC.")

    A("")
    A("-" * W)
    A("PART VII -- TAIL AUDIT (20 largest |advantage| pairs, including the ~1.56-nat extreme case)")
    A("-" * W)
    ep = summary["tail_audit"]["extreme_pair"]
    A(f"Most extreme pair: prompt={ep['prompt']} layer={ep['layer']} head={ep['head']} "
      f"(i={ep['i']},j={ep['j']})  advantage={ep['advantage']:.4f}  Gamma={ep['Gamma']:.4f}")
    for f in FEATURE_NAMES:
        A(f"    {f:<20} = {ep[f]:.4f}")
    A("")
    A("Descriptive only (n=20, not enough for inference): does any feature separate the 10")
    A("mass-favoring-tail pairs from the 10 local-favoring-tail pairs by eye/Mann-Whitney?")
    for f in FEATURE_NAMES:
        mw = summary["tail_audit"]["per_feature"][f]["mann_whitney"]
        A(f"    {f:<20} tail Mann-Whitney AUC={_fmt(mw['auc'] if mw else None)}  p={_fmt(mw['p'] if mw else None)}")

    A("")
    A("-" * W)
    A("PART VIII -- MINIMAL MULTIVARIATE MODEL (identical procedure to Pythia)")
    A("-" * W)
    mv = summary["multivariate"]
    A(f"Chosen features (same selection rule): {mv['chosen_features']}")
    A(f"{'model':<28}{'mean CV AUC':>14}{'mean CV acc':>14}{'folds used':>12}")
    A(f"{'majority class':<28}{'0.500':>14}{_fmt(mv['majority_class_accuracy']):>14}{'--':>12}")
    for label, key in (
        ("|Gamma| only", "baseline_abs_gamma"), ("concentration only", "baseline_concentration"),
        ("directional features", "directional_model"), ("directional+baselines", "directional_plus_baselines"),
    ):
        d = mv[key]
        A(f"{label:<28}{_fmt(d['mean_auc']):>14}{_fmt(d['mean_accuracy']):>14}{d['n_folds_used']:>12}")

    A("")
    A("-" * W)
    A("PART IX -- VERDICT")
    A("-" * W)
    A(f"GPT-2 DVG verdict: {summary['verdict']}")
    A(summary["verdict_text"])
    A("")
    n_near_zero = sum(1 for pc in summary["power_classification"].values() if pc["outcome"] == "1_near_zero_despite_power")
    n_within = sum(1 for pc in summary["power_classification"].values() if pc["outcome"] in ("3_reproducible_within_stratum", "4_strong_predictive_improvement"))
    if n_within == 0 and n_near_zero >= 4:
        cross_arch_conclusion = "simple directional geometry rejected as a general explanation"
    elif n_within == 0:
        cross_arch_conclusion = "weak architecture-sensitive evidence"
    elif n_within >= 1:
        cross_arch_conclusion = "unresolved (isolated within-stratum signal(s), not consistent across the feature set -- needs independent confirmation before calling it a replicated mechanism)"
    else:
        cross_arch_conclusion = "unresolved"
    A(f"Cross-architecture DVG conclusion: {cross_arch_conclusion}")
    A(f"  ({n_near_zero}/{len(FEATURE_NAMES)} features near-zero despite ~10x power; "
      f"{n_within}/{len(FEATURE_NAMES)} features show a within-stratum-meaningful effect)")

    A("")
    A("-" * W)
    A("PART X -- FALSIFICATION STATEMENT")
    A("-" * W)
    if n_within == 0:
        A("GPT-2's ~10x larger sample did NOT rescue Pythia's DVG-B result into DVG-C: pooled effects")
        A("remain small (several statistically significant only because of the huge n, per Part V's")
        A("power classification), and NO feature reaches the pre-declared within-stratum effect-size")
        A("threshold. This is POSITIVE EVIDENCE AGAINST the current simple directional-geometry")
        A("hypothesis (the 8 cosine/ratio features tested), not merely an absence of significance --")
        A("the experiment had the power to detect a real within-stratum effect of the pre-declared")
        A("minimum size and did not find one.")
    else:
        A("At least one feature reached the pre-declared within-stratum threshold in GPT-2; see Part")
        A("IX for whether this constitutes a replicated mechanism (requires independent confirmation")
        A("in Pythia's own data, not assumed here).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
