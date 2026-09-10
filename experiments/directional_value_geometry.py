r"""Directional value-vector geometry vs. downstream mass-vs-local winner.

Falsification-first test of the hypothesis recorded verbatim in
`PAUSE_NOTE.md`'s "NEXT HIGHEST-VALUE TASK": does the *direction* of value
vectors (not just the norms g_i = ||o - v_i|| that the already-proved exact
margin Gamma uses) predict

    sign(advantage) = sign(kl_mass - kl_local)

(positive = local-damage wins, negative = mass wins) among the 1,962
measurable, set-changing meaningful inversions already found by
`experiments/full_grid_mechanistic_analysis.py`?

**Does NOT recompute any eviction or KL outcome.** Reuses, unmodified:
  - `results/full_grid_mechanistic_analysis/pairs.json` (1,987 meaningful
    pairs; filtered here to the 1,962 with `changes_set_within_kmax` and
    `above_noise_floor`, both already computed by that script)
  - `results/full_grid_mechanistic_analysis/summary.json` (per-stratum
    `top1_share` / `entropy_nats`, used only as an existing-concentration
    baseline for comparison in Part 7 -- not recomputed)

**Forward passes: minimized to one per prompt (3 total).** A single
`model(ids, output_attentions=True, output_hidden_states=True)` call already
yields every layer's hidden states and every head's attention in one pass,
exactly as `full_grid_mechanistic_analysis.py` does it -- so all 20 strata
within one prompt are covered by that prompt's single forward pass. No
`tier_b_eviction_scoped` call is made anywhere in this script.

Writes `results/directional_value_geometry/{summary,pairs}.json` and
`results/directional_value_geometry/report.txt`. Touches no file under
`src/kv_efficiency/`, `tests/`, or any existing result/doc file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.eviction import require_normalized
from kv_efficiency.provenance import capture_environment, load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "milestone2_grid.yaml"
SOURCE_PAIRS = ROOT / "results" / "full_grid_mechanistic_analysis" / "pairs.json"
SOURCE_SUMMARY = ROOT / "results" / "full_grid_mechanistic_analysis" / "summary.json"
RESULTS_DIR = ROOT / "results" / "directional_value_geometry"

MIN_STRATUM_N = 10          # minimum pairs in a (prompt,layer,head) cell to report within-stratum stats
WEAK_EFFECT_AUC = 0.55      # |AUC-0.5| below this is called "not practically meaningful" regardless of p
WEAK_EFFECT_R = 0.10        # |r| / |rho| below this is called "not practically meaningful" regardless of p
N_CV_FOLDS = 5
RIDGE = 1e-6                # fixed numerical-stability ridge, not tuned
RNG_SEED = 0


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def load_model_and_adapter(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, cfg["dtype"]), attn_implementation=cfg["attn_implementation"]
    ).eval()
    return tok, model, GPTNeoXAdapter(model)


def safe_unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# feature computation
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "cos_vi_vj", "cos_vi_o", "cos_vj_o", "delta_cos_o", "cos_oi_oj",
    "cos_vidiff_o", "cos_vidiff_rpair", "norm_ratio",
]

FEATURE_DEFINITIONS = {
    "cos_vi_vj": "cos(v_i, v_j) -- angle between the two candidates' value vectors themselves.",
    "cos_vi_o": "cos(v_i, o) -- alignment of v_i with the actual attention output o.",
    "cos_vj_o": "cos(v_j, o) -- alignment of v_j with o.",
    "delta_cos_o": "cos(v_i,o) - cos(v_j,o) -- signed difference in output-alignment.",
    "cos_oi_oj": "cos(o-v_i, o-v_j) -- angle between the two SINGLETON local-residual "
                 "directions; exact, since the proved identity Delta_i = alpha_i/(1-alpha_i)*(o-v_i) "
                 "means direction(Delta_i) = direction(o-v_i) always (alpha_i/(1-alpha_i) > 0).",
    "cos_vidiff_o": "cos(v_i - v_j, o) -- alignment of the pair's displacement with the output direction.",
    "cos_vidiff_rpair": "cos(v_i - v_j, r_pair) where r_pair = (alpha_lo*v_i + alpha_hi*v_j) "
                         "- (alpha_lo+alpha_hi)*o is the exact local-algebra residual r_B "
                         "(Delta_B = -r_B/(1-m_B)) for the PAIR treated as a 2-element block "
                         "B={i,j} -- an exact extension of the already-proved singleton identity "
                         "to the minimal nontrivial block containing both candidates.",
    "norm_ratio": "||v_i|| / ||v_j|| -- CONTROL feature; a naive norm ratio Gamma does not use "
                  "(Gamma uses g_i=||o-v_i|| not ||v_i||), included to check that directional "
                  "features are not merely proxying for an unrelated magnitude artifact.",
}


def compute_features(v_i, v_j, o, alpha_lo, alpha_hi) -> dict:
    ri = o - v_i
    rj = o - v_j
    vdiff = v_i - v_j
    o_pair = alpha_lo * v_i + alpha_hi * v_j
    m_pair = alpha_lo + alpha_hi
    r_pair = o_pair - m_pair * o
    ni, nj = np.linalg.norm(v_i), np.linalg.norm(v_j)
    return dict(
        cos_vi_vj=cos(v_i, v_j),
        cos_vi_o=cos(v_i, o),
        cos_vj_o=cos(v_j, o),
        delta_cos_o=cos(v_i, o) - cos(v_j, o),
        cos_oi_oj=cos(ri, rj),
        cos_vidiff_o=cos(vdiff, o),
        cos_vidiff_rpair=cos(vdiff, r_pair),
        norm_ratio=float(ni / nj) if nj > 0 else float("nan"),
        g_lo_recon=float(np.linalg.norm(ri)),
        g_hi_recon=float(np.linalg.norm(rj)),
    )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def clean_pair(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], int((~mask).sum())


def univariate_stats(feature: np.ndarray, advantage: np.ndarray, y: np.ndarray) -> dict:
    f, adv, n_excluded_adv = clean_pair(feature, advantage)
    f2, yy, n_excluded_y = clean_pair(feature, y.astype(float))
    out = dict(n_used_spearman=len(f), n_excluded_nan=n_excluded_adv)

    if len(f) >= 3 and np.std(f) > 0 and np.std(adv) > 0:
        rho, p = sstats.spearmanr(f, adv)
        out["spearman_vs_advantage"] = dict(rho=float(rho), p=float(p))
    else:
        out["spearman_vs_advantage"] = None

    if len(f2) >= 3 and np.std(f2) > 0 and np.std(yy) > 0:
        r, p = sstats.pearsonr(f2, yy)
        out["point_biserial_vs_win"] = dict(r=float(r), p=float(p))
    else:
        out["point_biserial_vs_win"] = None

    x_local = f2[yy == 1]
    x_mass = f2[yy == 0]
    if len(x_local) >= 3 and len(x_mass) >= 3:
        U, p = sstats.mannwhitneyu(x_local, x_mass, alternative="two-sided")
        n1, n2 = len(x_local), len(x_mass)
        auc = float(U / (n1 * n2))
        out["mann_whitney"] = dict(
            U=float(U), p=float(p), n_local=n1, n_mass=n2,
            auc=auc, rank_biserial=2 * auc - 1,
        )
    else:
        out["mann_whitney"] = None

    auc = out["mann_whitney"]["auc"] if out["mann_whitney"] else None
    r_pb = out["point_biserial_vs_win"]["r"] if out["point_biserial_vs_win"] else None
    weak = True
    if auc is not None and abs(auc - 0.5) >= (WEAK_EFFECT_AUC - 0.5):
        weak = False
    if r_pb is not None and abs(r_pb) >= WEAK_EFFECT_R:
        weak = False
    out["practically_weak"] = weak
    return out


def eta_squared_between_stratum(feature: np.ndarray, stratum_ids: np.ndarray) -> float | None:
    """Fraction of feature variance explained purely by stratum identity
    (ICC-like). High value = feature mostly tells you WHICH STRATUM a pair
    came from, not anything about the pair itself -- the Simpson's-paradox
    check requested."""
    mask = np.isfinite(feature)
    f, s = feature[mask], stratum_ids[mask]
    if len(f) < 3 or np.var(f) == 0:
        return None
    grand_mean = f.mean()
    ss_between = 0.0
    for sid in np.unique(s):
        grp = f[s == sid]
        ss_between += len(grp) * (grp.mean() - grand_mean) ** 2
    ss_total = ((f - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else None


def within_stratum_demeaned_correlation(feature: np.ndarray, advantage: np.ndarray, stratum_ids: np.ndarray) -> dict:
    mask = np.isfinite(feature) & np.isfinite(advantage)
    f, adv, s = feature[mask], advantage[mask], stratum_ids[mask]
    f_within = f.copy()
    adv_within = adv.copy()
    for sid in np.unique(s):
        idx = s == sid
        f_within[idx] = f[idx] - f[idx].mean()
        adv_within[idx] = adv[idx] - adv[idx].mean()
    if len(f_within) < 3 or np.std(f_within) == 0 or np.std(adv_within) == 0:
        return dict(pearson_r=None, spearman_rho=None, n=len(f_within))
    r_p, p_p = sstats.pearsonr(f_within, adv_within)
    r_s, p_s = sstats.spearmanr(f_within, adv_within)
    return dict(
        pearson_r=float(r_p), pearson_p=float(p_p),
        spearman_rho=float(r_s), spearman_p=float(p_s),
        n=len(f_within),
    )


def per_stratum_stats(feature: np.ndarray, advantage: np.ndarray, y: np.ndarray, stratum_ids: np.ndarray) -> dict:
    per_stratum = {}
    for sid in np.unique(stratum_ids):
        idx = stratum_ids == sid
        n = int(idx.sum())
        if n < MIN_STRATUM_N:
            continue
        f, adv, yy = feature[idx], advantage[idx], y[idx]
        mask = np.isfinite(f)
        f, adv, yy = f[mask], adv[mask], yy[mask]
        if len(f) < MIN_STRATUM_N or np.std(f) == 0:
            continue
        entry = dict(n=len(f))
        if np.std(adv) > 0:
            rho, p = sstats.spearmanr(f, adv)
            entry["spearman_rho"] = float(rho)
            entry["spearman_p"] = float(p)
        x_local, x_mass = f[yy == 1], f[yy == 0]
        if len(x_local) >= 3 and len(x_mass) >= 3:
            U, p = sstats.mannwhitneyu(x_local, x_mass, alternative="two-sided")
            entry["auc"] = float(U / (len(x_local) * len(x_mass)))
            entry["mw_p"] = float(p)
        per_stratum[sid] = entry

    aucs = [v["auc"] for v in per_stratum.values() if "auc" in v]
    rhos = [v["spearman_rho"] for v in per_stratum.values() if "spearman_rho" in v]
    summary = dict(
        n_strata_with_enough_data=len(per_stratum),
        median_auc=float(np.median(aucs)) if aucs else None,
        iqr_auc=[float(np.percentile(aucs, 25)), float(np.percentile(aucs, 75))] if aucs else None,
        frac_auc_above_0p5=float(np.mean([a > 0.5 for a in aucs])) if aucs else None,
        median_spearman_rho=float(np.median(rhos)) if rhos else None,
        frac_rho_same_sign_as_pooled=None,  # filled by caller once pooled sign known
    )
    return dict(per_stratum=per_stratum, aggregate=summary)


# ---------------------------------------------------------------------------
# minimal logistic regression (manual, no sklearn) + grouped CV
# ---------------------------------------------------------------------------

def fit_logreg(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE, n_iter: int = 50) -> np.ndarray:
    """Newton-Raphson (IRLS) logistic regression with a fixed tiny ridge for
    numerical stability only (not tuned). X should already include an
    intercept column of ones."""
    n, d = X.shape
    beta = np.zeros(d)
    for _ in range(n_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        grad = X.T @ (y - p) - ridge * beta
        H = -(X.T * W) @ X - ridge * np.eye(d)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def predict_logreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = X @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc_from_scores(scores: np.ndarray, y: np.ndarray) -> float | None:
    s1, s0 = scores[y == 1], scores[y == 0]
    if len(s1) < 2 or len(s0) < 2:
        return None
    U, _ = sstats.mannwhitneyu(s1, s0, alternative="two-sided")
    return float(U / (len(s1) * len(s0)))


def grouped_cv_auc(feature_matrix: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = N_CV_FOLDS) -> dict:
    """K-fold CV where every pair from the same (prompt,layer,head) stratum
    is always in the same fold (deterministic assignment by sorted group id,
    round-robin -- not random search, no tuning)."""
    uniq_groups = sorted(set(groups.tolist()))
    fold_of_group = {g: i % n_folds for i, g in enumerate(uniq_groups)}
    fold_ids = np.array([fold_of_group[g] for g in groups])

    n, d = feature_matrix.shape
    X = np.hstack([np.ones((n, 1)), feature_matrix])
    fold_aucs, fold_accs = [], []
    for fold in range(n_folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        if test_mask.sum() < 10 or len(set(y[train_mask].tolist())) < 2:
            continue
        beta = fit_logreg(X[train_mask], y[train_mask].astype(float))
        p = predict_logreg(X[test_mask], beta)
        auc = auc_from_scores(p, y[test_mask])
        acc = float(np.mean((p >= 0.5).astype(int) == y[test_mask]))
        if auc is not None:
            fold_aucs.append(auc)
        fold_accs.append(acc)
    return dict(
        n_folds_used=len(fold_accs),
        mean_auc=float(np.mean(fold_aucs)) if fold_aucs else None,
        mean_accuracy=float(np.mean(fold_accs)) if fold_accs else None,
        fold_aucs=fold_aucs, fold_accuracies=fold_accs,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    tok, model, adapter = load_model_and_adapter(cfg)
    prompts = cfg["evaluation_prompts"]

    all_pairs = json.loads(SOURCE_PAIRS.read_text())
    source_summary = json.loads(SOURCE_SUMMARY.read_text())
    stratum_concentration = {
        (s["prompt"], s["layer"], s["head"]): dict(top1_share=s["top1_share"], entropy_nats=s["entropy_nats"])
        for s in source_summary["stratum_summaries"]
    }

    measurable = [
        p for p in all_pairs
        if p.get("changes_set_within_kmax") and p.get("above_noise_floor")
        and p.get("advantage_mass_minus_local_at_change") is not None
    ]
    n_measurable = len(measurable)

    # ---- one forward pass per prompt; reconstruct per (layer,head) within it ----
    needed_strata = sorted({(p["prompt"], p["layer"], p["head"]) for p in measurable})
    needed_by_prompt: dict[int, set[tuple[int, int]]] = {}
    for p_idx, layer, head in needed_strata:
        needed_by_prompt.setdefault(p_idx, set()).add((layer, head))

    tensors_cache: dict[tuple[int, int, int], dict] = {}  # (prompt,layer,head) -> {alpha_np, values_np, o}
    n_forward_passes = 0
    for p_idx, layer_heads in needed_by_prompt.items():
        ids = tok(prompts[p_idx], return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
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

    # ---- per-pair feature computation + reconstruction cross-check ----
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
        enriched.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pairs.json").write_text(json.dumps(enriched, indent=2))

    # ---- arrays for stats ----
    stratum_ids = np.array([f"{r['prompt']}_{r['layer']}_{r['head']}" for r in enriched])
    advantage = np.array([r["advantage"] for r in enriched], dtype=np.float64)
    y = np.array([r["win_local"] for r in enriched], dtype=int)
    abs_gamma = np.array([abs(r["Gamma"]) for r in enriched], dtype=np.float64)
    top1 = np.array([r.get("top1_share", np.nan) for r in enriched], dtype=np.float64)
    entropy = np.array([r.get("entropy_nats", np.nan) for r in enriched], dtype=np.float64)

    pooled_results = {}
    within_stratum_results = {}
    stratum_level_results = {}
    eta_sq_results = {}

    for fname in FEATURE_NAMES:
        feat = np.array([r[fname] for r in enriched], dtype=np.float64)
        uni = univariate_stats(feat, advantage, y)
        pooled_results[fname] = uni
        pooled_sign = None
        if uni["spearman_vs_advantage"]:
            pooled_sign = np.sign(uni["spearman_vs_advantage"]["rho"])

        within = within_stratum_demeaned_correlation(feat, advantage, stratum_ids)
        within_stratum_results[fname] = within

        strat = per_stratum_stats(feat, advantage, y, stratum_ids)
        if pooled_sign is not None and strat["aggregate"]["median_spearman_rho"] is not None:
            rhos = [v["spearman_rho"] for v in strat["per_stratum"].values() if "spearman_rho" in v]
            if rhos:
                strat["aggregate"]["frac_rho_same_sign_as_pooled"] = float(
                    np.mean([np.sign(r) == pooled_sign for r in rhos])
                )
        stratum_level_results[fname] = strat

        eta_sq_results[fname] = eta_squared_between_stratum(feat, stratum_ids)

    # baselines: |Gamma| and concentration, same pooled/within/stratum treatment
    baseline_results = {}
    for name, arr in (("abs_Gamma", abs_gamma), ("top1_share", top1), ("entropy_nats", entropy)):
        baseline_results[name] = dict(
            pooled=univariate_stats(arr, advantage, y),
            within_stratum=within_stratum_demeaned_correlation(arr, advantage, stratum_ids),
            eta_squared_between_stratum=eta_squared_between_stratum(arr, stratum_ids),
        )

    # ---- layer 0 / head 9 counterexample audit ----
    l0h9_mask = np.array([(r["layer"] == 0 and r["head"] == 9) for r in enriched])
    l0h9_rows = [r for r, m in zip(enriched, l0h9_mask) if m]
    l0h9_analysis = dict(n=len(l0h9_rows))
    if l0h9_rows:
        y_l0h9 = np.array([r["win_local"] for r in l0h9_rows])
        adv_l0h9 = np.array([r["advantage"] for r in l0h9_rows])
        l0h9_analysis["local_win_rate"] = float(y_l0h9.mean())
        l0h9_analysis["mean_gamma"] = float(np.mean([r["Gamma"] for r in l0h9_rows]))
        l0h9_analysis["per_prompt"] = {}
        for p_idx in sorted({r["prompt"] for r in l0h9_rows}):
            sub = [r for r in l0h9_rows if r["prompt"] == p_idx]
            yy = np.array([r["win_local"] for r in sub])
            l0h9_analysis["per_prompt"][p_idx] = dict(n=len(sub), local_win_rate=float(yy.mean()))
        l0h9_analysis["feature_auc"] = {}
        for fname in FEATURE_NAMES:
            feat = np.array([r[fname] for r in l0h9_rows], dtype=np.float64)
            uni = univariate_stats(feat, adv_l0h9, y_l0h9)
            l0h9_analysis["feature_auc"][fname] = uni["mann_whitney"]["auc"] if uni["mann_whitney"] else None

    # ---- strongest counterexamples: biggest |advantage| pairs where local lost despite favorable-looking geometry ----
    sorted_by_adv = sorted(enriched, key=lambda r: r["advantage"])
    strongest_mass_wins = sorted_by_adv[:5]
    strongest_local_wins = sorted_by_adv[-5:]

    # ---- minimal multivariate model ----
    # pick strongest, non-redundant directional features by |AUC-0.5| among the 8 tested
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

    # ---- verdict ----
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
        verdict = "DVG-D"
        verdict_text = (
            "A small directional feature set materially predicts the downstream winner "
            "beyond |Gamma| and attention concentration alone (grouped-CV AUC "
            f"{dir_auc:.3f} vs. |Gamma| {gamma_auc:.3f} vs. concentration {conc_auc:.3f})."
        )
    elif any_within_meaningful:
        verdict = "DVG-C"
        verdict_text = (
            "Directional geometry shows a reproducible within-stratum association with "
            "win direction (see within_stratum_correlations), even though it does not "
            "materially outperform |Gamma|/concentration in the grouped multivariate test."
        )
    elif any_pooled_meaningful:
        verdict = "DVG-B"
        verdict_text = (
            "At least one directional feature is statistically detectable pooled, but the "
            "effect is practically weak (below the pre-declared |AUC-0.5|>=0.05 / |r|>=0.10 "
            "thresholds) and/or does not survive the within-stratum control."
        )
    else:
        verdict = "DVG-A"
        verdict_text = (
            "No tested directional feature carries useful information about the downstream "
            "winner, pooled or within-stratum, at the pre-declared effect-size thresholds."
        )

    provenance = capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"])
    summary = dict(
        provenance=provenance,
        source_pairs_file=str(SOURCE_PAIRS.relative_to(ROOT)),
        source_summary_file=str(SOURCE_SUMMARY.relative_to(ROOT)),
        n_meaningful_inversions_total=len(all_pairs),
        n_measurable=n_measurable,
        n_forward_passes=n_forward_passes,
        max_g_reconstruction_error=max_g_recon_err,
        feature_definitions=FEATURE_DEFINITIONS,
        target_definition="win_local = 1 iff advantage_mass_minus_local_at_change = "
                           "kl_mass - kl_local > 0 (local-damage has smaller KL to the true "
                           "output, i.e. local-damage wins); 0 otherwise (mass wins).",
        pooled=pooled_results,
        within_stratum_demeaned=within_stratum_results,
        per_stratum=stratum_level_results,
        eta_squared_between_stratum=eta_sq_results,
        baselines=baseline_results,
        layer0_head9_counterexample_audit=l0h9_analysis,
        strongest_mass_win_examples=[
            {k: r[k] for k in ("prompt", "layer", "head", "i", "j", "advantage", "Gamma", *FEATURE_NAMES)}
            for r in strongest_mass_wins
        ],
        strongest_local_win_examples=[
            {k: r[k] for k in ("prompt", "layer", "head", "i", "j", "advantage", "Gamma", *FEATURE_NAMES)}
            for r in strongest_local_wins
        ],
        multivariate=multivariate,
        verdict=verdict,
        verdict_text=verdict_text,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    write_report(summary, enriched)
    return summary


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def write_report(summary: dict, enriched: list[dict]) -> None:
    L = []
    A = L.append
    W = 78

    A("DIRECTIONAL VALUE-VECTOR GEOMETRY vs. DOWNSTREAM WINNER -- REPORT")
    A("=" * W)
    A("")
    A("Falsification test of PAUSE_NOTE.md's next-highest-value task: does the")
    A("DIRECTION of value vectors (not just the norms g_i=||o-v_i|| that the")
    A("already-proved exact margin Gamma uses) predict sign(kl_mass - kl_local)")
    A("among the meaningful, set-changing inversions already found by")
    A("experiments/full_grid_mechanistic_analysis.py? No new eviction/KL outcome")
    A("was computed anywhere in this script.")

    A("")
    A("-" * W)
    A("1. SAMPLE COUNTS")
    A("-" * W)
    A(f"meaningful inversions in source file        : {summary['n_meaningful_inversions_total']}")
    A(f"measurable (set-changing, >=1e-8 KL) subset  : {summary['n_measurable']}")
    A(f"forward passes used for reconstruction       : {summary['n_forward_passes']} (one per prompt)")
    A(f"max |g_recon - g_stored| across all pairs     : {summary['max_g_reconstruction_error']:.3e}")
    A("  (cross-check: reconstructed ||o-v_i||/||o-v_j|| against the values")
    A("   already stored in pairs.json by the original run -- near-zero confirms")
    A("   the value-vector/output reconstruction is correct, independent of any")
    A("   new eviction computation.)")

    A("")
    A("-" * W)
    A("2. TARGET DEFINITION")
    A("-" * W)
    A(summary["target_definition"])

    A("")
    A("-" * W)
    A("3. FEATURE DEFINITIONS")
    A("-" * W)
    for name in FEATURE_NAMES:
        A(f"{name}:")
        A(f"    {summary['feature_definitions'][name]}")

    A("")
    A("-" * W)
    A("4. POOLED STATISTICS (all measurable pairs together)")
    A("-" * W)
    A(f"{'feature':<20}{'spearman rho':>14}{'p':>10}{'pt-biserial r':>15}{'AUC':>8}{'weak?':>8}")
    for name in FEATURE_NAMES:
        u = summary["pooled"][name]
        sp = u["spearman_vs_advantage"]
        pb = u["point_biserial_vs_win"]
        mw = u["mann_whitney"]
        A(
            f"{name:<20}"
            f"{_fmt(sp['rho'] if sp else None):>14}"
            f"{_fmt(sp['p'] if sp else None):>10}"
            f"{_fmt(pb['r'] if pb else None):>15}"
            f"{_fmt(mw['auc'] if mw else None):>8}"
            f"{'yes' if u['practically_weak'] else 'NO':>8}"
        )
    A("")
    A(f"Pre-declared weakness thresholds: |AUC-0.5| < {WEAK_EFFECT_AUC - 0.5:.2f} AND")
    A(f"|point-biserial r| < {WEAK_EFFECT_R:.2f} => called 'practically weak' regardless of p,")
    A(f"per instruction not to treat tiny p-values with tiny effect sizes (n~{summary['n_measurable']})")
    A("as scientifically meaningful.")
    A("")
    A("Baselines (same pooled treatment), for comparison:")
    for name, res in summary["baselines"].items():
        u = res["pooled"]
        mw = u["mann_whitney"]
        sp = u["spearman_vs_advantage"]
        A(
            f"  {name:<16} AUC={_fmt(mw['auc'] if mw else None):>7}  "
            f"spearman_rho={_fmt(sp['rho'] if sp else None):>8}  "
            f"weak={'yes' if u['practically_weak'] else 'NO'}"
        )

    A("")
    A("-" * W)
    A("5. WITHIN-STRATUM STATISTICS (Simpson's-paradox control)")
    A("-" * W)
    A("within_stratum_demeaned: feature and advantage each had their own")
    A("(prompt,layer,head)-stratum mean subtracted before correlating -- isolates")
    A("signal that exists WITHIN a stratum, immune to any between-stratum offset.")
    A("eta_squared_between_stratum: fraction of the feature's own variance that is")
    A("explained purely by which stratum a pair came from (high = feature mostly")
    A("just identifies the stratum, not the pair).")
    A("")
    A(f"{'feature':<20}{'within r':>10}{'within p':>10}{'eta^2(strat)':>14}{'median per-stratum AUC':>24}")
    for name in FEATURE_NAMES:
        w = summary["within_stratum_demeaned"][name]
        eta = summary["eta_squared_between_stratum"][name]
        strat_agg = summary["per_stratum"][name]["aggregate"]
        A(
            f"{name:<20}"
            f"{_fmt(w['pearson_r']):>10}"
            f"{_fmt(w.get('pearson_p')):>10}"
            f"{_fmt(eta):>14}"
            f"{_fmt(strat_agg['median_auc']):>24}"
        )
    A("")
    for name, res in summary["baselines"].items():
        w = res["within_stratum"]
        eta = res["eta_squared_between_stratum"]
        A(f"  baseline {name:<14} within_r={_fmt(w['pearson_r']):>8}  eta^2(strat)={_fmt(eta):>8}")
    A("")
    n_strata_min = summary["per_stratum"][FEATURE_NAMES[0]]["aggregate"]["n_strata_with_enough_data"]
    A(f"Strata with >= {MIN_STRATUM_N} measurable pairs and reportable per-stratum stats: {n_strata_min}")
    for name in FEATURE_NAMES:
        agg = summary["per_stratum"][name]["aggregate"]
        A(
            f"  {name:<20} median_AUC={_fmt(agg['median_auc']):>7}  "
            f"IQR={_fmt(agg['iqr_auc'])}  "
            f"frac_same_sign_as_pooled={_fmt(agg['frac_rho_same_sign_as_pooled'])}"
        )

    A("")
    A("-" * W)
    A("6. LAYER 0 / HEAD 9 COUNTEREXAMPLE AUDIT")
    A("-" * W)
    l0h9 = summary["layer0_head9_counterexample_audit"]
    if l0h9.get("n", 0) == 0:
        A("No measurable pairs at layer 0 / head 9 in this dataset -- audit not possible.")
    else:
        A(f"n measurable pairs at layer 0/head 9        : {l0h9['n']}")
        A(f"local-win rate (this cell)                  : {l0h9['local_win_rate']:.3f}  "
          f"(pooled grid rate: {sum(r['win_local'] for r in enriched)/len(enriched):.3f})")
        A(f"mean Gamma (this cell)                      : {l0h9['mean_gamma']:.4f}  "
          f"(known: highest positive-Gamma-mass in the whole grid, per MILESTONE_STATUS.md)")
        A("per-prompt local-win rate:")
        for p_idx, d in sorted(l0h9["per_prompt"].items()):
            A(f"    prompt {p_idx}: n={d['n']}, local_win_rate={d['local_win_rate']:.3f}")
        A("directional-feature AUC restricted to this cell (0.5 = no separation):")
        for name, auc in l0h9["feature_auc"].items():
            A(f"    {name:<20} AUC={_fmt(auc)}")
        any_l0h9_signal = any(
            (auc is not None and abs(auc - 0.5) >= (WEAK_EFFECT_AUC - 0.5))
            for auc in l0h9["feature_auc"].values()
        )
        if any_l0h9_signal:
            A("")
            A("At least one directional feature shows a non-trivial AUC restricted to this")
            A("cell -- inspect the feature_auc list above before concluding it explains the")
            A("counterexample; n is small (single-cell), so treat as suggestive only.")
        else:
            A("")
            A("EXPLICIT NEGATIVE RESULT: no directional feature tested here separates")
            A("local-favoring from mass-favoring pairs within layer 0/head 9 (all AUCs")
            A("within the weak-effect band). Directional value-vector geometry does NOT")
            A("explain this standing counterexample any better than Gamma or attention")
            A("concentration already failed to.")

    A("")
    A("-" * W)
    A("7. STRONGEST COUNTEREXAMPLES (largest |advantage| in each direction)")
    A("-" * W)
    A("Largest mass-favoring pairs (advantage most negative):")
    for r in summary["strongest_mass_win_examples"]:
        A(f"    prompt={r['prompt']} layer={r['layer']} head={r['head']} (i={r['i']},j={r['j']})  "
          f"advantage={r['advantage']:.4e}  Gamma={r['Gamma']:.4f}  "
          f"cos_oi_oj={r['cos_oi_oj']:.3f}  cos_vidiff_rpair={r['cos_vidiff_rpair']:.3f}")
    A("Largest local-favoring pairs (advantage most positive):")
    for r in summary["strongest_local_win_examples"]:
        A(f"    prompt={r['prompt']} layer={r['layer']} head={r['head']} (i={r['i']},j={r['j']})  "
          f"advantage={r['advantage']:.4e}  Gamma={r['Gamma']:.4f}  "
          f"cos_oi_oj={r['cos_oi_oj']:.3f}  cos_vidiff_rpair={r['cos_vidiff_rpair']:.3f}")

    A("")
    A("-" * W)
    A("8. MINIMAL MULTIVARIATE TEST (grouped-by-stratum cross-validation)")
    A("-" * W)
    mv = summary["multivariate"]
    A(f"Chosen (strongest, pairwise |corr|<=0.8) directional features: {mv['chosen_features']}")
    A(f"n pairs used (finite features, all models)  : {mv['n_pairs_used']}")
    A(f"Manual logistic regression, Newton-Raphson, fixed ridge={RIDGE:g} (not tuned),")
    A(f"{N_CV_FOLDS}-fold CV grouped by (prompt,layer,head) stratum (no stratum split across folds).")
    A("")
    A(f"{'model':<28}{'mean CV AUC':>14}{'mean CV acc':>14}{'folds used':>12}")
    A(
        f"{'majority class':<28}{'0.500':>14}{_fmt(mv['majority_class_accuracy']):>14}{'--':>12}"
    )
    for label, key in (
        ("|Gamma| only", "baseline_abs_gamma"),
        ("concentration only", "baseline_concentration"),
        ("directional features", "directional_model"),
        ("directional+baselines", "directional_plus_baselines"),
    ):
        d = mv[key]
        A(f"{label:<28}{_fmt(d['mean_auc']):>14}{_fmt(d['mean_accuracy']):>14}{d['n_folds_used']:>12}")

    A("")
    A("-" * W)
    A("9. DOES DIRECTIONAL GEOMETRY ADD INFORMATION BEYOND GAMMA / CONCENTRATION?")
    A("-" * W)
    dir_auc = mv["directional_model"]["mean_auc"]
    gamma_auc = mv["baseline_abs_gamma"]["mean_auc"]
    conc_auc = mv["baseline_concentration"]["mean_auc"]
    combo_auc = mv["directional_plus_baselines"]["mean_auc"]
    A(f"directional-only CV AUC   = {_fmt(dir_auc)}")
    A(f"|Gamma|-only CV AUC       = {_fmt(gamma_auc)}")
    A(f"concentration-only CV AUC = {_fmt(conc_auc)}")
    A(f"directional+baselines CV AUC = {_fmt(combo_auc)}")
    A("(all AUCs are out-of-fold, grouped by stratum -- a stratum's pairs are never")
    A(" split across train and test, so this cannot merely be 'which stratum is it'.)")

    A("")
    A("-" * W)
    A("10. VERDICT")
    A("-" * W)
    A(f"{summary['verdict']}")
    A(summary["verdict_text"])

    A("")
    A("-" * W)
    A("11. NEXT RECOMMENDED EXPERIMENT")
    A("-" * W)
    verdict = summary["verdict"]
    if verdict == "DVG-A":
        A("None warranted from this result alone. Directional value-vector geometry, as")
        A("operationalized here (8 cosine/ratio features from already-available value")
        A("vectors), does not add information beyond Gamma/concentration. Before")
        A("proposing a new directional feature family, revisit whether the underlying")
        A("hypothesis itself (rather than this particular featurization) should be")
        A("retired in favor of testing generalization to a different architecture/model")
        A("(MILESTONE_STATUS.md's longer-standing 'Next milestone' item).")
    elif verdict == "DVG-B":
        A("Weak pooled-only signal is not enough to act on. If pursued further, the next")
        A("smallest step is the within-stratum-only version of the same test restricted")
        A("to the specific features that showed non-negligible pooled AUC here, rather")
        A("than a new feature family or a larger model.")
    elif verdict == "DVG-C":
        A("Directional geometry carries real within-stratum signal that a pooled or")
        A("single-stratum view would miss. Next smallest step: a per-stratum (not")
        A("pooled) multivariate model -- e.g. one small logistic regression per head,")
        A("evaluated by leave-one-prompt-out CV within that head -- to see whether the")
        A("useful direction/features are stratum-specific (a 'local correction to mass'")
        A("story) rather than a single global rule.")
    else:
        A("Directional geometry materially beats Gamma/concentration in grouped CV.")
        A("Next smallest step: verify the result is not an artifact of the specific")
        A("feature-selection procedure (re-run the same pipeline with each candidate")
        A("feature held out one at a time) before considering it for any larger-model")
        A("or new-architecture generalization test.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
