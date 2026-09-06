r"""Milestone 2, Phases 2-6: expanded layer/head/prompt grid, local-damage
vs. attention-mass vs. random, per-stratum summary, rank-divergence
analysis.

Primary hypothesis (Phase 1, NOT assumed true): local-damage eviction can
outperform attention-mass eviction where value-vector geometry carries
information attention mass alone does not. Null outcome (equally
acceptable): attention mass is sufficient, local damage adds little.

Grid, prompts, and removal fractions are fixed in
configs/milestone2_grid.yaml BEFORE this script is run -- not chosen after
seeing results. Uses evaluation_prompts only (never calibration_prompts,
which are reserved for the budget allocator in global_budget_curve.py).

Writes to results/milestone2_multistratum/ (new directory; Milestone 1's
results/*.json are never modified).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.damage_metrics import kl_divergence, logit_l2
from kv_efficiency.eviction import require_normalized, tier_b_eviction
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, local_damage_scores, random_ranking
from kv_efficiency.provenance import capture_environment, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "milestone2_multistratum"

TIE_THRESHOLD_RELATIVE = 0.05  # |advantage| / max(mean_kl_mass, mean_kl_local) below this => "tied"


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, cfg["dtype"]), attn_implementation=cfg["attn_implementation"]
    ).eval()
    return tok, model


def mask_from_indices(n: int, indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return mask


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    tok, model = load_model(cfg)
    adapter = GPTNeoXAdapter(model)

    layers = cfg["layers"]
    heads = cfg["heads"]
    prompts = cfg["evaluation_prompts"]
    qs = cfg["removal_fractions"]
    n_seeds = cfg["n_random_seeds"]

    rows = []  # one row per (prompt, layer, head, q, policy)
    rank_divergence = []  # one row per (prompt, layer, head): Spearman(mass, local) at candidate level

    for p_idx, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        logits_full = out.logits[0]

        for layer in layers:
            hidden_in = out.hidden_states[layer][0].double()
            for head in heads:
                alpha_full = out.attentions[layer][0, head].double()
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                alpha_last = require_normalized(alpha_full[qp], renormalize=True)
                protect = {0, qp}

                mass_rank = attention_mass_ranking(alpha_last, protect=protect)
                local_rank = local_damage_ranking(alpha_last, values_dmodel, protect=protect)
                local_scores = local_damage_scores(alpha_last, values_dmodel, protect=protect)

                # rank-divergence: Spearman(m_B, d_B_local) over the candidate set with a defined score
                cand = sorted(local_scores.keys())
                if len(cand) >= 3:
                    m_vals = np.array([float(alpha_last[i]) for i in cand])
                    d_vals = np.array([local_scores[i] for i in cand])
                    rho_md, _ = spearmanr(m_vals, d_vals)
                else:
                    rho_md = float("nan")
                rank_divergence.append(dict(
                    prompt=p_idx, layer=layer, head=head, n_candidates=len(cand),
                    spearman_mB_dlocal=None if np.isnan(rho_md) else float(rho_md),
                ))

                n_cand = len(mass_rank)
                for q in qs:
                    k = round(q * n_cand)

                    for policy_name, ranking in (("attention_mass", mass_rank), ("local_damage", local_rank)):
                        evict_idx = ranking[:k]
                        mask = mask_from_indices(n, evict_idx)
                        if k == 0:
                            kl, l2 = 0.0, 0.0
                        else:
                            try:
                                logits_b, _ = tier_b_eviction(
                                    model, adapter, ids, alpha_full, hidden_in,
                                    layer=layer, head=head, block=mask,
                                )
                                kl = float(kl_divergence(logits_full[qp], logits_b[qp]))
                                l2 = float(logit_l2(logits_full[qp], logits_b[qp]))
                            except ValueError:
                                kl, l2 = None, None
                        rows.append(dict(prompt=p_idx, layer=layer, head=head, q=q, k_evicted=k,
                                          n_candidates=n_cand, policy=policy_name, kl=kl, logit_l2=l2))

                    random_kls, random_l2s = [], []
                    for seed in range(n_seeds):
                        ranking = random_ranking(n, protect=protect, seed=seed)
                        evict_idx = ranking[:k]
                        mask = mask_from_indices(n, evict_idx)
                        if k == 0:
                            random_kls.append(0.0)
                            random_l2s.append(0.0)
                            continue
                        try:
                            logits_b, _ = tier_b_eviction(
                                model, adapter, ids, alpha_full, hidden_in,
                                layer=layer, head=head, block=mask,
                            )
                            random_kls.append(float(kl_divergence(logits_full[qp], logits_b[qp])))
                            random_l2s.append(float(logit_l2(logits_full[qp], logits_b[qp])))
                        except ValueError:
                            continue
                    if random_kls:
                        rows.append(dict(prompt=p_idx, layer=layer, head=head, q=q, k_evicted=k,
                                          n_candidates=n_cand, policy="random", kl=float(np.mean(random_kls)),
                                          logit_l2=float(np.mean(random_l2s)), kl_std=float(np.std(random_kls)),
                                          n_seeds=len(random_kls)))

    # ---- Phase 6: per-stratum summary ----
    strata = sorted(set((r["layer"], r["head"]) for r in rows))
    stratum_summaries = []
    paired_advantages = []  # per (prompt, layer, head, q): kl_mass - kl_local
    for (layer, head) in strata:
        sub = [r for r in rows if r["layer"] == layer and r["head"] == head]
        by_policy = {p: [r["kl"] for r in sub if r["policy"] == p and r["kl"] is not None] for p in
                     ("random", "attention_mass", "local_damage")}

        # paired per (prompt, q)
        stratum_pairs = []
        for p_idx in range(len(prompts)):
            for q in qs:
                mass_vals = [r["kl"] for r in sub if r["policy"] == "attention_mass" and r["prompt"] == p_idx and r["q"] == q and r["kl"] is not None]
                local_vals = [r["kl"] for r in sub if r["policy"] == "local_damage" and r["prompt"] == p_idx and r["q"] == q and r["kl"] is not None]
                if mass_vals and local_vals:
                    diff = mass_vals[0] - local_vals[0]
                    stratum_pairs.append(diff)
                    paired_advantages.append(dict(layer=layer, head=head, prompt=p_idx, q=q, advantage=diff))

        mean_advantage = float(np.mean(stratum_pairs)) if stratum_pairs else None
        rho_entries = [r["spearman_mB_dlocal"] for r in rank_divergence
                       if r["layer"] == layer and r["head"] == head and r["spearman_mB_dlocal"] is not None]

        stratum_summaries.append(dict(
            layer=layer, head=head, n_observations=len(sub),
            mean_kl_random=float(np.mean(by_policy["random"])) if by_policy["random"] else None,
            median_kl_random=float(np.median(by_policy["random"])) if by_policy["random"] else None,
            mean_kl_mass=float(np.mean(by_policy["attention_mass"])) if by_policy["attention_mass"] else None,
            median_kl_mass=float(np.median(by_policy["attention_mass"])) if by_policy["attention_mass"] else None,
            mean_kl_local=float(np.mean(by_policy["local_damage"])) if by_policy["local_damage"] else None,
            median_kl_local=float(np.median(by_policy["local_damage"])) if by_policy["local_damage"] else None,
            mean_advantage_mass_minus_local=mean_advantage,
            mean_spearman_mB_dlocal=float(np.mean(rho_entries)) if rho_entries else None,
        ))

    # win/lose/tie count across strata
    n_local_wins = n_mass_wins = n_tied = 0
    for s in stratum_summaries:
        adv = s["mean_advantage_mass_minus_local"]
        scale = max(s["mean_kl_mass"] or 0.0, s["mean_kl_local"] or 0.0, 1e-12)
        if adv is None:
            continue
        rel = adv / scale
        if abs(rel) < TIE_THRESHOLD_RELATIVE:
            n_tied += 1
        elif adv > 0:
            n_local_wins += 1
        else:
            n_mass_wins += 1

    # rank-divergence vs advantage, across strata
    strata_with_both = [s for s in stratum_summaries
                         if s["mean_spearman_mB_dlocal"] is not None and s["mean_advantage_mass_minus_local"] is not None]
    if len(strata_with_both) >= 3:
        rho_vec = np.array([s["mean_spearman_mB_dlocal"] for s in strata_with_both])
        adv_vec = np.array([s["mean_advantage_mass_minus_local"] for s in strata_with_both])
        rho_divergence_vs_advantage, p_divergence = spearmanr(1 - rho_vec, adv_vec)
    else:
        rho_divergence_vs_advantage, p_divergence = float("nan"), float("nan")

    # best / worst strata by advantage
    ranked = sorted(strata_with_both, key=lambda s: s["mean_advantage_mass_minus_local"] or 0.0)
    worst_for_local = ranked[:3]
    best_for_local = ranked[-3:][::-1]

    all_advantages = np.array([p["advantage"] for p in paired_advantages])
    ci_lo, ci_hi = bootstrap_ci(all_advantages)

    summary = dict(
        provenance=capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"]),
        n_strata=len(strata), n_rows=len(rows), n_prompts=len(prompts),
        tie_threshold_relative=TIE_THRESHOLD_RELATIVE,
        pooled_paired_advantage_mass_minus_local=dict(
            mean=float(all_advantages.mean()) if len(all_advantages) else None,
            median=float(np.median(all_advantages)) if len(all_advantages) else None,
            bootstrap_95ci=[ci_lo, ci_hi],
            n=len(all_advantages),
        ),
        strata_counts=dict(local_wins=n_local_wins, mass_wins=n_mass_wins, tied=n_tied),
        rank_divergence_vs_advantage_spearman=dict(
            statistic="spearman(1 - Spearman(m_B, d_local), advantage)",
            rho=None if np.isnan(rho_divergence_vs_advantage) else float(rho_divergence_vs_advantage),
            p=None if np.isnan(p_divergence) else float(p_divergence),
            n_strata=len(strata_with_both),
        ),
        best_strata_for_local=best_for_local,
        worst_strata_for_local=worst_for_local,
        stratum_summaries=stratum_summaries,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "rows.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "rank_divergence.json").write_text(json.dumps(rank_divergence, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "stratum_summaries"}, indent=2))
    return summary


if __name__ == "__main__":
    run()
