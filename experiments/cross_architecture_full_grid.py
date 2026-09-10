r"""Full GPT-2 (124M) grid: all 12 layers x all 12 heads x the same 3
evaluation prompts as the pilot and as Pythia. The bounded pilot
(`results/cross_architecture_generalization/`) already passed adapter
validation (8/8 tests + inline Part-0 gate) and the pre-registered
expansion gate; this script does NOT re-validate the adapter (that gate is
frozen, not re-run) and does NOT modify `GPT2Adapter`.

Core per-stratum loop is the SAME algebra/definitions as
`experiments/cross_architecture_generalization.py` and, before that,
`experiments/full_grid_mechanistic_analysis.py` (Pythia) --
MEANINGFUL_ALPHA_FLOOR=1e-6, NOISE_FLOOR_KL=1e-8, Q_MAX=0.95, identical
candidate-pair / meaningful-inversion / set-change / KL / advantage
definitions, not re-derived or altered. Written as a fresh, self-contained
script (not importing the pilot script) so the already-validated pilot
script is never at risk of being changed.

sign(Gamma) <-> ranking inversion remains an algebraic identity (see the
Part II derivation from the previous round); this script records mismatch
counts as an implementation sanity check only, never as evidence.

Adds, on top of the pilot's analysis: depth-fraction normalization
(layer/11), Gini/top-10% concentration of inversions across strata,
within-stratum-residualized G7 correlation with a bootstrap CI, per-stratum
local-win-rate heterogeneity by depth, cross-architecture predictability of
stratum-level prevalence from top1_share/entropy (Pearson r^2, both models,
using only already-defined quantities), and an explicit falsification-
target search. All are read from data this script itself collects, plus a
read-only comparison against the frozen Pythia baseline JSON.

Writes results/cross_architecture_full_grid/{summary,pairs,rows}.json and
report.txt. Does not modify results/cross_architecture_generalization/,
results/full_grid_mechanistic_analysis/, any src/ file, any test, or any
doc/status file.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.attention_reconstruction import head_values_dmodel
from kv_efficiency.attention_reconstruction_gpt2 import GPT2Adapter
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import require_normalized, tier_b_eviction_scoped
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, local_damage_scores

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "cross_architecture_full_grid"
PILOT_SUMMARY = ROOT / "results" / "cross_architecture_generalization" / "summary.json"
PYTHIA_BASELINE_SUMMARY = ROOT / "results" / "full_grid_mechanistic_analysis" / "summary.json"

MODEL_ID = "gpt2"
LAYERS = list(range(12))
HEADS = list(range(12))
DEPTH_MAX = 11  # depth_fraction = layer / DEPTH_MAX
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
Q_MAX = 0.95
MASS_EPSILON = 1e-9
MEANINGFUL_ALPHA_FLOOR = 1e-6
NOISE_FLOOR_KL = 1e-8
N_BOOTSTRAP = 2000
BOOT_SEED = 0


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tok, model


def mask_from_indices(n: int, indices) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return mask


def entropy_nats(alpha: np.ndarray) -> float:
    a = alpha[alpha > 0]
    return float(-(a * np.log(a)).sum())


def first_k_where_pair_changes_set(mass_rank, local_rank, i, j, k_max):
    mass_set, local_set = set(), set()
    for k in range(1, k_max + 1):
        mass_set.add(mass_rank[k - 1])
        local_set.add(local_rank[k - 1])
        if (i in mass_set) != (i in local_set) or (j in mass_set) != (j in local_set):
            return k
    return None


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = N_BOOTSTRAP, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return None
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(values[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


# ---------------------------------------------------------------------------
# PART 1 -- full-grid data collection (same algebra as the pilot/Pythia scripts)
# ---------------------------------------------------------------------------

def run_grid(tok, model, adapter) -> dict:
    all_rows = []
    all_meaningful_pairs = []
    stratum_summaries = []
    verification_mismatches = []
    n_pairs_grand_total = 0
    n_verification_checks = 0
    n_cells_total = len(EVALUATION_PROMPTS) * len(LAYERS) * len(HEADS)
    n_cells_done = 0

    for p_idx, prompt in enumerate(EVALUATION_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        logits_full = out.logits[0]
        protect = {0, qp}

        for layer in LAYERS:
            hidden_in = out.hidden_states[layer][0].double()
            for head in HEADS:
                alpha_full = out.attentions[layer][0, head].double()
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                alpha_last = require_normalized(alpha_full[qp], renormalize=True)
                alpha_np = alpha_last.detach().numpy().astype(np.float64)
                values_np = values_dmodel.detach().numpy().astype(np.float64)

                d_scores = local_damage_scores(alpha_last, values_dmodel, protect=protect)
                mass_rank = attention_mass_ranking(alpha_last, protect=protect)
                local_rank = local_damage_ranking(alpha_last, values_dmodel, protect=protect)
                n_cand = len(mass_rank)
                k_max = round(Q_MAX * n_cand)

                o = alpha_np @ values_np
                candidates = [i for i in range(n) if i not in protect]
                g = {i: float(np.linalg.norm(o - values_np[i])) for i in candidates}
                scored = [i for i in candidates if i in d_scores]

                top1_share = float(alpha_np.max())
                ent = entropy_nats(alpha_np)
                mean_mass = float(alpha_np[candidates].mean()) if candidates else None
                median_mass = float(np.median(alpha_np[candidates])) if candidates else None
                mass_sorted_alpha = [float(alpha_np[i]) for i in mass_rank]
                gaps = np.diff(mass_sorted_alpha)
                median_adjacent_gap = float(np.median(gaps)) if len(gaps) else None
                min_adjacent_gap = float(np.min(gaps)) if len(gaps) else None

                n_pairs = 0
                n_meaningful_inverted = 0
                n_trivial_inverted = 0
                gammas_meaningful = []
                involved_in_inversion = set()
                stratum_pairs_this = []
                for a in range(len(scored)):
                    for b in range(a + 1, len(scored)):
                        x, y = scored[a], scored[b]
                        ax, ay = alpha_np[x], alpha_np[y]
                        if ax == ay:
                            continue
                        lo, hi = (x, y) if ax < ay else (y, x)
                        a_lo, a_hi = alpha_np[lo], alpha_np[hi]
                        g_lo, g_hi = g[lo], g[hi]
                        n_pairs += 1
                        n_pairs_grand_total += 1

                        M = (a_hi * (1.0 - a_lo)) / (a_lo * (1.0 - a_hi))
                        G = g_lo / g_hi if g_hi > 0 else float("inf")
                        Gamma = math.log(G) - math.log(M) if G > 0 and M > 0 else float("nan")

                        actual_inverted = d_scores[lo] > d_scores[hi]
                        predicted_inverted = Gamma > 0
                        n_verification_checks += 1
                        if predicted_inverted != actual_inverted:
                            verification_mismatches.append(dict(
                                prompt=p_idx, layer=layer, head=head, i=lo, j=hi,
                                alpha_lo=float(a_lo), alpha_hi=float(a_hi), Gamma=Gamma,
                                d_lo=d_scores[lo], d_hi=d_scores[hi],
                            ))

                        if not actual_inverted:
                            continue
                        meaningful = max(a_lo, a_hi) >= MEANINGFUL_ALPHA_FLOOR
                        if not meaningful:
                            n_trivial_inverted += 1
                            continue
                        n_meaningful_inverted += 1
                        gammas_meaningful.append(Gamma)
                        involved_in_inversion.add(lo)
                        involved_in_inversion.add(hi)
                        stratum_pairs_this.append(dict(
                            prompt=p_idx, layer=layer, head=head, i=lo, j=hi,
                            alpha_lo=float(a_lo), alpha_hi=float(a_hi),
                            g_lo=g_lo, g_hi=g_hi, M=M, G=G, Gamma=Gamma,
                            d_lo=d_scores[lo], d_hi=d_scores[hi],
                            n_candidates=n_cand, k_max=k_max,
                        ))

                positive_gamma_mass = float(sum(alpha_np[i] for i in involved_in_inversion))

                stratum_summaries.append(dict(
                    prompt=p_idx, layer=layer, head=head, depth_fraction=layer / DEPTH_MAX,
                    n_candidates=n_cand, k_max=k_max,
                    top1_share=top1_share, entropy_nats=ent,
                    mean_mass=mean_mass, median_mass=median_mass,
                    median_adjacent_mass_gap=median_adjacent_gap, min_adjacent_mass_gap=min_adjacent_gap,
                    n_pairs=n_pairs, n_meaningful_inverted=n_meaningful_inverted,
                    n_trivial_inverted=n_trivial_inverted,
                    frac_meaningful_inverted=(n_meaningful_inverted / n_pairs) if n_pairs else None,
                    max_gamma=max(gammas_meaningful) if gammas_meaningful else None,
                    median_gamma_meaningful=float(np.median(gammas_meaningful)) if gammas_meaningful else None,
                    positive_gamma_mass=positive_gamma_mass,
                    n_changed_eviction_set=0, n_beyond_kmax=0,
                    n_measurable_kl=0, n_local_wins_measurable=0, n_mass_wins_measurable=0,
                ))
                all_meaningful_pairs.extend(stratum_pairs_this)

                if stratum_pairs_this:
                    for k in range(1, k_max + 1):
                        for policy_name, ranking in (("attention_mass", mass_rank), ("local_damage", local_rank)):
                            evict_idx = ranking[:k]
                            mask = mask_from_indices(n, evict_idx)
                            kl = None
                            skipped = False
                            try:
                                logits_b, _, invalid = tier_b_eviction_scoped(
                                    model, adapter, ids, alpha_full, hidden_in,
                                    layer=layer, head=head, block=mask,
                                    measured_positions={qp}, mass_epsilon=MASS_EPSILON,
                                )
                                kl = float(kl_divergence(logits_full[qp], logits_b[qp]))
                            except ValueError:
                                skipped = True
                            all_rows.append(dict(
                                prompt=p_idx, layer=layer, head=head, k=k, q=k / n_cand,
                                policy=policy_name, kl=kl, skipped_singular_at_measured=skipped,
                            ))
                n_cells_done += 1

        print(f"prompt {p_idx} done ({n} tokens); cells so far {n_cells_done}/{n_cells_total}; "
              f"meaningful pairs so far {len(all_meaningful_pairs)}", flush=True)

    kl_lookup = {(r["prompt"], r["layer"], r["head"], r["k"], r["policy"]): r["kl"] for r in all_rows}
    strata_index = {(s["prompt"], s["layer"], s["head"]): s for s in stratum_summaries}
    rank_cache = {}

    def get_ranks(p_idx, layer, head):
        key = (p_idx, layer, head)
        if key in rank_cache:
            return rank_cache[key]
        ids = tok(EVALUATION_PROMPTS[p_idx], return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        hidden_in = out.hidden_states[layer][0].double()
        alpha_full = out.attentions[layer][0, head].double()
        values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
        alpha_last = require_normalized(alpha_full[ids.shape[1] - 1], renormalize=True)
        protect = {0, ids.shape[1] - 1}
        mass_rank = attention_mass_ranking(alpha_last, protect=protect)
        local_rank = local_damage_ranking(alpha_last, values_dmodel, protect=protect)
        rank_cache[key] = (mass_rank, local_rank)
        return mass_rank, local_rank

    for pair in all_meaningful_pairs:
        p_idx, layer, head = pair["prompt"], pair["layer"], pair["head"]
        mass_rank, local_rank = get_ranks(p_idx, layer, head)
        change_k = first_k_where_pair_changes_set(mass_rank, local_rank, pair["i"], pair["j"], pair["n_candidates"])
        changes_within_kmax = change_k is not None and change_k <= pair["k_max"]
        pair["change_k"] = change_k
        pair["q_at_change"] = (change_k / pair["n_candidates"]) if change_k else None
        pair["changes_set_within_kmax"] = bool(changes_within_kmax)

        kl_mass = kl_lookup.get((p_idx, layer, head, change_k, "attention_mass")) if changes_within_kmax else None
        kl_local = kl_lookup.get((p_idx, layer, head, change_k, "local_damage")) if changes_within_kmax else None
        advantage = (kl_mass - kl_local) if (kl_mass is not None and kl_local is not None) else None
        pair["kl_mass_at_change"] = kl_mass
        pair["kl_local_at_change"] = kl_local
        pair["advantage_mass_minus_local_at_change"] = advantage
        pair["above_noise_floor"] = bool(advantage is not None and abs(advantage) >= NOISE_FLOOR_KL)

        s = strata_index[(p_idx, layer, head)]
        if change_k is None:
            continue
        if changes_within_kmax:
            s["n_changed_eviction_set"] += 1
            if advantage is not None and abs(advantage) >= NOISE_FLOOR_KL:
                s["n_measurable_kl"] += 1
                if advantage > 0:
                    s["n_local_wins_measurable"] += 1
                else:
                    s["n_mass_wins_measurable"] += 1
        else:
            s["n_beyond_kmax"] += 1

    return dict(
        all_rows=all_rows, all_meaningful_pairs=all_meaningful_pairs,
        stratum_summaries=stratum_summaries, verification_mismatches=verification_mismatches,
        n_pairs_grand_total=n_pairs_grand_total, n_verification_checks=n_verification_checks,
    )


# ---------------------------------------------------------------------------
# PART 2 -- analysis (G1-G7 + the additional requested statistics)
# ---------------------------------------------------------------------------

def analyze(grid: dict, pythia_baseline: dict, pilot_summary: dict) -> dict:
    pairs = grid["all_meaningful_pairs"]
    strata = grid["stratum_summaries"]
    n_total_pairs = grid["n_pairs_grand_total"]
    n_meaningful = len(pairs)
    n_mismatches = len(grid["verification_mismatches"])

    measurable = [
        p for p in pairs
        if p.get("changes_set_within_kmax") and p.get("above_noise_floor")
        and p.get("advantage_mass_minus_local_at_change") is not None
    ]
    set_changing = [p for p in pairs if p.get("changes_set_within_kmax")]
    n_set_changing = len(set_changing)
    n_measurable = len(measurable)

    prevalence = n_meaningful / n_total_pairs if n_total_pairs else None
    pythia_prevalence = pythia_baseline["n_meaningful_inversions_total"] / pythia_baseline["n_pairs_grand_total"]
    frac_set_changing = n_set_changing / n_meaningful if n_meaningful else None
    frac_measurable_of_set_changing = n_measurable / n_set_changing if n_set_changing else None

    local_wins = [p for p in measurable if p["advantage_mass_minus_local_at_change"] > 0]
    mass_wins = [p for p in measurable if p["advantage_mass_minus_local_at_change"] < 0]
    n_local, n_mass = len(local_wins), len(mass_wins)
    local_win_rate = n_local / n_measurable if n_measurable else None
    adv_arr = np.array([p["advantage_mass_minus_local_at_change"] for p in measurable]) if measurable else np.array([])
    sum_advantage = float(adv_arr.sum()) if len(adv_arr) else None
    mean_advantage = float(adv_arr.mean()) if len(adv_arr) else None
    median_advantage = float(np.median(adv_arr)) if len(adv_arr) else None
    advantage_percentiles = (
        {str(q): float(np.percentile(adv_arr, q)) for q in (1, 5, 25, 50, 75, 95, 99)} if len(adv_arr) else None
    )
    local_win_rate_ci = bootstrap_ci(
        np.array([1.0 if p["advantage_mass_minus_local_at_change"] > 0 else 0.0 for p in measurable]),
        statistic=np.mean,
    ) if measurable else None
    sum_advantage_ci = bootstrap_ci(adv_arr, statistic=np.sum) if len(adv_arr) else None

    top1 = np.array([s["top1_share"] for s in strata])
    ent = np.array([s["entropy_nats"] for s in strata])
    depth = np.array([s["depth_fraction"] for s in strata])
    frac_inv = np.array([s["frac_meaningful_inverted"] if s["frac_meaningful_inverted"] is not None else 0.0 for s in strata])
    rho_top1, p_top1 = sstats.spearmanr(top1, frac_inv)
    rho_ent, p_ent = sstats.spearmanr(ent, frac_inv)
    rho_depth, p_depth = sstats.spearmanr(depth, frac_inv)

    # cross-architecture predictability of stratum-level prevalence (Pearson r^2)
    r_top1, _ = sstats.pearsonr(top1, frac_inv)
    r_ent, _ = sstats.pearsonr(ent, frac_inv)
    gpt2_r2_top1, gpt2_r2_ent = r_top1 ** 2, r_ent ** 2

    py_top1 = np.array([s["top1_share"] for s in pythia_baseline["stratum_summaries"]])
    py_ent = np.array([s["entropy_nats"] for s in pythia_baseline["stratum_summaries"]])
    py_frac = np.array([
        s["frac_meaningful_inverted"] if s["frac_meaningful_inverted"] is not None else 0.0
        for s in pythia_baseline["stratum_summaries"]
    ])
    py_r_top1, _ = sstats.pearsonr(py_top1, py_frac)
    py_r_ent, _ = sstats.pearsonr(py_ent, py_frac)
    pythia_r2_top1, pythia_r2_ent = py_r_top1 ** 2, py_r_ent ** 2

    # G7: |Gamma| vs |advantage| and vs signed advantage -- pooled + within-stratum residualized
    adv_pairs = [p for p in pairs if p.get("advantage_mass_minus_local_at_change") is not None]
    g7 = dict(pooled=None, within_stratum=None)
    if len(adv_pairs) >= 5:
        gam = np.array([abs(p["Gamma"]) for p in adv_pairs])
        adv = np.array([p["advantage_mass_minus_local_at_change"] for p in adv_pairs])
        cell_ids = np.array([f"{p['prompt']}_{p['layer']}_{p['head']}" for p in adv_pairs])
        rho_mag, p_mag = sstats.spearmanr(gam, np.abs(adv))
        rho_sign, p_sign = sstats.spearmanr(gam, adv)
        ci_mag = bootstrap_ci(np.arange(len(gam)), statistic=lambda idx: sstats.spearmanr(gam[idx], np.abs(adv)[idx])[0])
        ci_sign = bootstrap_ci(np.arange(len(gam)), statistic=lambda idx: sstats.spearmanr(gam[idx], adv[idx])[0])
        g7["pooled"] = dict(
            rho_gamma_vs_abs_advantage=float(rho_mag), p_mag=float(p_mag), ci_mag=ci_mag,
            rho_gamma_vs_signed_advantage=float(rho_sign), p_sign=float(p_sign), ci_sign=ci_sign,
            n=len(gam),
        )
        # within-stratum residualized: demean gamma and advantage by cell before correlating
        gam_w, adv_w = gam.copy(), adv.copy()
        for cid in np.unique(cell_ids):
            idx = cell_ids == cid
            gam_w[idx] = gam[idx] - gam[idx].mean()
            adv_w[idx] = adv[idx] - adv[idx].mean()
        if np.std(gam_w) > 0 and np.std(adv_w) > 0:
            rho_w_mag, p_w_mag = sstats.spearmanr(gam_w, np.abs(adv_w))
            rho_w_sign, p_w_sign = sstats.spearmanr(gam_w, adv_w)
            g7["within_stratum"] = dict(
                rho_gamma_vs_abs_advantage=float(rho_w_mag), p=float(p_w_mag),
                rho_gamma_vs_signed_advantage=float(rho_w_sign), p_sign=float(p_w_sign),
                n=len(gam_w),
            )

    # concentration of inversions across strata: Gini + top-10% share
    inv_counts = np.array([s["n_meaningful_inverted"] for s in strata])
    gini_coef = gini(inv_counts)
    n_top = max(1, round(0.10 * len(inv_counts)))
    top10_share = float(np.sort(inv_counts)[::-1][:n_top].sum() / inv_counts.sum()) if inv_counts.sum() > 0 else None

    # per-stratum table + heterogeneity
    per_stratum_table = []
    for s in strata:
        per_stratum_table.append(dict(
            prompt=s["prompt"], layer=s["layer"], head=s["head"], depth_fraction=s["depth_fraction"],
            n_candidates=s["n_candidates"], top1_share=s["top1_share"], entropy_nats=s["entropy_nats"],
            n_pairs=s["n_pairs"], n_meaningful_inverted=s["n_meaningful_inverted"],
            frac_meaningful_inverted=s["frac_meaningful_inverted"],
            n_changed_eviction_set=s["n_changed_eviction_set"], n_measurable_kl=s["n_measurable_kl"],
            n_local_wins_measurable=s["n_local_wins_measurable"], n_mass_wins_measurable=s["n_mass_wins_measurable"],
        ))

    zero_inversion = [s for s in per_stratum_table if s["n_meaningful_inverted"] == 0]
    high_prevalence = sorted(
        [s for s in per_stratum_table if s["frac_meaningful_inverted"]],
        key=lambda s: s["frac_meaningful_inverted"], reverse=True
    )[:10]
    strongly_mass = [s for s in per_stratum_table if s["n_measurable_kl"] >= 5 and s["n_mass_wins_measurable"] / s["n_measurable_kl"] >= 0.7]
    strongly_local = [s for s in per_stratum_table if s["n_measurable_kl"] >= 5 and s["n_local_wins_measurable"] / s["n_measurable_kl"] >= 0.7]

    win_rate_strata = [
        (s["n_local_wins_measurable"] / s["n_measurable_kl"], s["depth_fraction"])
        for s in per_stratum_table if s["n_measurable_kl"] >= 5
    ]
    if len(win_rate_strata) >= 5:
        wr = np.array([w for w, _ in win_rate_strata])
        dep = np.array([d for _, d in win_rate_strata])
        rho_winrate_depth, p_winrate_depth = sstats.spearmanr(dep, wr)
    else:
        rho_winrate_depth = p_winrate_depth = None

    # falsification-target search
    high_ent_low_prev = sorted(
        [s for s in per_stratum_table if s["entropy_nats"] > np.median(ent)],
        key=lambda s: (s["frac_meaningful_inverted"] or 0.0)
    )[:5]
    concentrated_high_prev = sorted(
        [s for s in per_stratum_table if s["top1_share"] > np.median(top1)],
        key=lambda s: (s["frac_meaningful_inverted"] or 0.0), reverse=True
    )[:5]
    many_inv_negligible_downstream = sorted(
        [s for s in per_stratum_table if s["n_meaningful_inverted"] >= 10 and s["n_measurable_kl"] <= max(1, round(0.05 * s["n_meaningful_inverted"]))],
        key=lambda s: s["n_meaningful_inverted"], reverse=True
    )[:5]
    mass_favoring_despite_global_local_lean = strongly_mass  # already computed, re-labeled for clarity below

    n_prompts_with_inversions = len({p["prompt"] for p in pairs})
    n_heads_with_inversions = len({(p["layer"], p["head"]) for p in pairs})
    n_cells_with_inversions = sum(1 for s in per_stratum_table if s["n_meaningful_inverted"] > 0)
    n_cells_total = len(per_stratum_table)

    strongest_pair = max(pairs, key=lambda p: abs(p.get("advantage_mass_minus_local_at_change") or 0.0), default=None)

    # ---- CAG verdict, mechanical, pre-declared rule matching the pilot's own rule exactly ----
    if n_meaningful == 0:
        verdict, verdict_text = "CAG-A", "No meaningful inversions occurred in the full grid."
    elif n_cells_with_inversions < 0.5 * n_cells_total:
        verdict = "CAG-B"
        verdict_text = "Basic inversion phenomenon replicates but is not pervasive across the full grid; principal structure differs from Pythia."
    else:
        prevalence_ratio = prevalence / pythia_prevalence if pythia_prevalence else None
        prevalence_close = prevalence_ratio is not None and 0.3 <= prevalence_ratio <= 3.0
        set_change_close = frac_set_changing is not None and frac_set_changing >= 0.8
        measurable_close = frac_measurable_of_set_changing is not None and frac_measurable_of_set_changing >= 0.8
        concentration_match = (rho_top1 < 0) and (rho_ent > 0)
        g7_null_in_both = (
            g7["pooled"] is not None and abs(g7["pooled"]["rho_gamma_vs_signed_advantage"]) < 0.05
        )
        if prevalence_close and set_change_close and measurable_close and concentration_match:
            verdict = "CAG-D"
            verdict_text = "Strong replication across principal empirical signatures on both complete grids."
        elif set_change_close and measurable_close and concentration_match:
            verdict = "CAG-C"
            verdict_text = "Qualitative cross-architecture replication with meaningful architectural differences (see prevalence/localization/G7 detail)."
        else:
            verdict = "CAG-B"
            verdict_text = "Basic inversion phenomenon replicates but principal downstream/concentration structure differs substantially."

    return dict(
        n_total_pairs=n_total_pairs, n_meaningful=n_meaningful, n_verification_mismatches=n_mismatches,
        prevalence=prevalence, pythia_prevalence=pythia_prevalence,
        n_set_changing=n_set_changing, frac_set_changing=frac_set_changing,
        n_measurable=n_measurable, frac_measurable_of_set_changing=frac_measurable_of_set_changing,
        n_local_wins=n_local, n_mass_wins=n_mass, local_win_rate=local_win_rate, local_win_rate_ci95=local_win_rate_ci,
        sum_advantage=sum_advantage, sum_advantage_ci95=sum_advantage_ci,
        mean_advantage=mean_advantage, median_advantage=median_advantage, advantage_percentiles=advantage_percentiles,
        g6_spearman_top1_vs_prevalence=dict(rho=float(rho_top1), p=float(p_top1)),
        g6_spearman_entropy_vs_prevalence=dict(rho=float(rho_ent), p=float(p_ent)),
        g6_spearman_depth_vs_prevalence=dict(rho=float(rho_depth) if not math.isnan(rho_depth) else None,
                                              p=float(p_depth) if not math.isnan(p_depth) else None),
        cross_arch_predictability=dict(
            gpt2_r2_top1=float(gpt2_r2_top1), gpt2_r2_entropy=float(gpt2_r2_ent),
            pythia_r2_top1=float(pythia_r2_top1), pythia_r2_entropy=float(pythia_r2_ent),
        ),
        g7=g7,
        concentration_across_strata=dict(gini=gini_coef, top10pct_share_of_inversions=top10_share),
        per_stratum_table=per_stratum_table,
        zero_inversion_strata=zero_inversion,
        high_prevalence_strata=high_prevalence,
        strongly_mass_favoring_strata=strongly_mass,
        strongly_local_favoring_strata=strongly_local,
        winrate_vs_depth=dict(rho=float(rho_winrate_depth) if rho_winrate_depth is not None else None,
                               p=float(p_winrate_depth) if p_winrate_depth is not None else None),
        falsification_high_entropy_low_prevalence=high_ent_low_prev,
        falsification_concentrated_high_prevalence=concentrated_high_prev,
        falsification_many_inversions_negligible_downstream=many_inv_negligible_downstream,
        falsification_mass_favoring_despite_global_local_lean=mass_favoring_despite_global_local_lean,
        n_prompts_with_inversions=n_prompts_with_inversions,
        n_heads_with_inversions=n_heads_with_inversions,
        n_cells_with_inversions=n_cells_with_inversions, n_cells_total=n_cells_total,
        strongest_pair=strongest_pair,
        verdict=verdict, verdict_text=verdict_text,
    )


def run() -> None:
    torch.manual_seed(0)
    tok, model = load_model()
    adapter = GPT2Adapter(model)
    print(f"Full GPT-2 grid: {len(LAYERS)} layers x {len(HEADS)} heads x {len(EVALUATION_PROMPTS)} prompts "
          f"= {len(LAYERS)*len(HEADS)*len(EVALUATION_PROMPTS)} cells. Adapter validation NOT re-run "
          f"(frozen from the pilot gate).", flush=True)

    grid = run_grid(tok, model, adapter)

    pythia_baseline = json.loads(PYTHIA_BASELINE_SUMMARY.read_text())
    pilot_summary = json.loads(PILOT_SUMMARY.read_text())
    analysis = analyze(grid, pythia_baseline, pilot_summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pairs.json").write_text(json.dumps(grid["all_meaningful_pairs"], indent=2))
    (RESULTS_DIR / "rows.json").write_text(json.dumps(grid["all_rows"], indent=2))
    summary = dict(
        model_id=MODEL_ID, layers=LAYERS, heads=HEADS, q_max=Q_MAX,
        meaningful_alpha_floor=MEANINGFUL_ALPHA_FLOOR, noise_floor_kl=NOISE_FLOOR_KL,
        n_layer_head_combos=len(LAYERS) * len(HEADS), n_prompts=len(EVALUATION_PROMPTS),
        n_prompt_layer_head_cells=len(LAYERS) * len(HEADS) * len(EVALUATION_PROMPTS),
        n_verification_checks=grid["n_verification_checks"],
        n_verification_mismatches=len(grid["verification_mismatches"]),
        verification_mismatches=grid["verification_mismatches"],
        analysis=analysis,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)

    pairs_size = (RESULTS_DIR / "pairs.json").stat().st_size
    rows_size = (RESULTS_DIR / "rows.json").stat().st_size
    print(f"\npairs.json: {pairs_size/1e6:.2f} MB   rows.json: {rows_size/1e6:.2f} MB")
    print("Done. See results/cross_architecture_full_grid/report.txt")


def write_report(summary: dict) -> None:
    a = summary["analysis"]
    L = []
    A = L.append
    A("FULL GPT-2 GRID (12 layers x 12 heads x 3 prompts) vs Pythia-160M FROZEN BASELINE")
    A("=" * 78)
    A(f"model={summary['model_id']}  n_prompts={summary['n_prompts']}  "
      f"n_layer_head_combos={summary['n_layer_head_combos']}  "
      f"n_prompt_layer_head_cells={summary['n_prompt_layer_head_cells']}")
    A("Adapter validation: NOT re-run this script -- frozen from the pilot's Part-0 gate "
      "(8/8 tests + inline checks, all far inside tolerance).")
    A("")
    A("-" * 78)
    A("GAMMA IMPLEMENTATION SANITY CHECK (architecture-independent identity, not evidence)")
    A("-" * 78)
    A(f"  n_verification_checks={summary['n_verification_checks']}  n_mismatches={summary['n_verification_mismatches']}")
    A("")
    A("-" * 78)
    A("PRIMARY MEASUREMENTS")
    A("-" * 78)
    A(f"total candidate pairs: {a['n_total_pairs']}")
    A(f"meaningful inversions: {a['n_meaningful']}")
    A(f"prevalence: GPT-2={a['prevalence']:.4%}  Pythia={a['pythia_prevalence']:.4%}  "
      f"(relative {(a['prevalence']/a['pythia_prevalence']-1):+.1%})")
    A(f"set-change rate: {a['frac_set_changing']}  (Pythia 0.996)")
    A(f"measurable-KL rate: {a['frac_measurable_of_set_changing']}  (Pythia 0.991)")
    A(f"local wins={a['n_local_wins']}  mass wins={a['n_mass_wins']}  "
      f"rate={a['local_win_rate']}  95% CI={a['local_win_rate_ci95']}  (Pythia 0.563)")
    A(f"sum advantage={a['sum_advantage']}  95% CI={a['sum_advantage_ci95']}")
    A(f"mean advantage={a['mean_advantage']}  median={a['median_advantage']}")
    A(f"advantage percentiles: {a['advantage_percentiles']}")
    A("")
    A("-" * 78)
    A("G6 -- CONCENTRATION VS PREVALENCE")
    A("-" * 78)
    A(f"top1_share vs prevalence: {a['g6_spearman_top1_vs_prevalence']}  (Pythia rho=-0.650)")
    A(f"entropy vs prevalence: {a['g6_spearman_entropy_vs_prevalence']}  (Pythia rho=+0.759)")
    A(f"depth_fraction vs prevalence: {a['g6_spearman_depth_vs_prevalence']}")
    A("")
    A("-" * 78)
    A("CROSS-ARCHITECTURE PREDICTABILITY OF STRATUM-LEVEL PREVALENCE (Pearson r^2)")
    A("-" * 78)
    ca = a["cross_arch_predictability"]
    A(f"GPT-2:   r^2(top1_share)={ca['gpt2_r2_top1']:.4f}   r^2(entropy)={ca['gpt2_r2_entropy']:.4f}")
    A(f"Pythia:  r^2(top1_share)={ca['pythia_r2_top1']:.4f}   r^2(entropy)={ca['pythia_r2_entropy']:.4f}")
    A("")
    A("-" * 78)
    A("G7 -- |GAMMA| VS ADVANTAGE, POOLED AND WITHIN-STRATUM")
    A("-" * 78)
    A(f"pooled: {a['g7']['pooled']}")
    A(f"within-stratum (demeaned by cell): {a['g7']['within_stratum']}")
    A("")
    A("-" * 78)
    A("CONCENTRATION OF INVERSIONS ACROSS STRATA")
    A("-" * 78)
    A(f"Gini coefficient (inversion counts across {a['n_cells_total']} cells): {a['concentration_across_strata']['gini']:.4f}")
    A(f"top-10% of cells' share of all inversions: {a['concentration_across_strata']['top10pct_share_of_inversions']}")
    A("")
    A("-" * 78)
    A("HETEROGENEITY")
    A("-" * 78)
    A(f"cells with zero inversions: {len(a['zero_inversion_strata'])} / {a['n_cells_total']}")
    A("top-10 highest-prevalence cells:")
    for s in a["high_prevalence_strata"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']} depth={s['depth_fraction']:.2f}  "
          f"frac={s['frac_meaningful_inverted']:.4f}  n_inv={s['n_meaningful_inverted']}  top1={s['top1_share']:.3f}")
    A(f"strongly mass-favoring cells (>=70%, n>=5): {len(a['strongly_mass_favoring_strata'])}")
    for s in a["strongly_mass_favoring_strata"][:15]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']} depth={s['depth_fraction']:.2f}  "
          f"mass_rate={s['n_mass_wins_measurable']/s['n_measurable_kl']:.3f}  n={s['n_measurable_kl']}")
    A(f"strongly local-favoring cells (>=70%, n>=5): {len(a['strongly_local_favoring_strata'])}")
    for s in a["strongly_local_favoring_strata"][:15]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']} depth={s['depth_fraction']:.2f}  "
          f"local_rate={s['n_local_wins_measurable']/s['n_measurable_kl']:.3f}  n={s['n_measurable_kl']}")
    A(f"win-rate vs depth: {a['winrate_vs_depth']}")
    A("")
    A(f"cells with >=1 inversion: {a['n_cells_with_inversions']} / {a['n_cells_total']}")
    A(f"prompts with >=1 inversion: {a['n_prompts_with_inversions']} / {summary['n_prompts']}")
    A(f"(layer,head) combos with >=1 inversion: {a['n_heads_with_inversions']} / {summary['n_layer_head_combos']}")
    A("")
    A("-" * 78)
    A("FALSIFICATION-TARGET SEARCH")
    A("-" * 78)
    A("High-entropy cells with LOWEST prevalence (should be rare if the relationship is robust):")
    for s in a["falsification_high_entropy_low_prevalence"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  entropy={s['entropy_nats']:.3f}  "
          f"frac_inv={s['frac_meaningful_inverted']}")
    A("Concentrated (top1_share > median) cells with HIGHEST prevalence (should be rare):")
    for s in a["falsification_concentrated_high_prevalence"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  top1={s['top1_share']:.3f}  "
          f"frac_inv={s['frac_meaningful_inverted']}")
    A("Cells with many inversions but negligible downstream (measurable) effect:")
    for s in a["falsification_many_inversions_negligible_downstream"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  n_inv={s['n_meaningful_inverted']}  "
          f"n_measurable={s['n_measurable_kl']}")
    A("Strongly mass-favoring cells despite the positive global local-win lean:")
    for s in a["falsification_mass_favoring_despite_global_local_lean"][:10]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  "
          f"mass_rate={s['n_mass_wins_measurable']/s['n_measurable_kl']:.3f}  n={s['n_measurable_kl']}")
    A("")
    A("-" * 78)
    A("STRONGEST PAIR (largest |advantage|)")
    A("-" * 78)
    A(json.dumps(a["strongest_pair"], indent=2) if a["strongest_pair"] else "none")
    A("")
    A("-" * 78)
    A("VERDICT")
    A("-" * 78)
    A(a["verdict"])
    A(a["verdict_text"])

    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
