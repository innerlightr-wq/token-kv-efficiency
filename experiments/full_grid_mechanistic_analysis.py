r"""Full 20-stratum extension of `experiments/mechanistic_inversion_analysis.py`.

That script derived, from the already-proved exact singleton identity
(`d_i^local = alpha_i/(1-alpha_i) * ||o-v_i||`), an EXACT pairwise
inversion condition and confirmed it on layer 11, heads 3/6/9 only. This
script asks whether the same condition identifies a reproducible
"intermediate ambiguity" regime across the FULL Milestone-2 grid (5 layers
x 4 heads = 20 strata, all 3 evaluation prompts), and whether it predicts
downstream KL consequences beyond what attention concentration alone
would tell you.

**The competition margin (exact, not approximate).** For a pair with
alpha_i < alpha_j (mass ranks i more evictable):

    M_ij = alpha_j*(1-alpha_i) / (alpha_i*(1-alpha_j))     -- mass's own boundary
    G_ij = g_i / g_j                                        -- geometry's competing ratio
    Gamma_ij = log(G_ij) - log(M_ij)

Gamma_ij > 0 iff d_i^local > d_j^local iff local-damage reverses the mass
ordering. This is an EXACT rewrite of the singleton formula (no small-
alpha approximation this time -- the (1-alpha) terms are kept), so
sign(Gamma) must equal the actual inversion label everywhere, up to
floating-point rounding. Verified computationally below, not re-proved.

**What's new here vs. the 3-head study:** every one of the 20 strata, at
every one of the 3 prompts, gets (a) the same fine per-k eviction-set /
downstream-KL sweep up to q=0.95 (reusing `tier_b_eviction_scoped`
unchanged -- the exact `m_B -> 1` guard is preserved exactly, never
divided through), and (b) concentration statistics (top-1 share, entropy)
and mass-gap statistics, so cross-stratum regime questions (does
concentration alone predict inversions? does Gamma add information beyond
concentration?) can be tested rather than assumed from 2 data points.

Writes results/full_grid_mechanistic_analysis/{summary,pairs,rows}.json.
Does not modify the local-damage metric, does not extend q past 0.95,
does not touch layer/head/prompt/removal-fraction definitions elsewhere
in the repo.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import require_normalized, tier_b_eviction_scoped
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, local_damage_scores
from kv_efficiency.provenance import capture_environment, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "full_grid_mechanistic_analysis"
Q_MAX = 0.95
MASS_EPSILON = 1e-9
MEANINGFUL_ALPHA_FLOOR = 1e-6  # excludes subnormal-tie noise among candidates that carry no real weight
# "Clearly above the float64 KL-computation noise floor" -- justified empirically below
# (see summary["noise_floor_justification"]), not chosen to manufacture a result.
NOISE_FLOOR_KL = 1e-8


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


def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    tok, model = load_model(cfg)
    adapter = GPTNeoXAdapter(model)

    layers = cfg["layers"]
    heads = cfg["heads"]
    prompts = cfg["evaluation_prompts"]

    all_rows = []          # per (prompt, layer, head, k, policy): kl, m_at_qp, etc.
    all_meaningful_pairs = []
    stratum_summaries = []
    verification_mismatches = []
    n_pairs_grand_total = 0
    n_verification_checks = 0

    for p_idx, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        logits_full = out.logits[0]
        protect = {0, qp}

        for layer in layers:
            hidden_in = out.hidden_states[layer][0].double()
            for head in heads:
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

                # mass-gap statistic: consecutive gaps in mass_rank order (candidates only)
                mass_sorted_alpha = [float(alpha_np[i]) for i in mass_rank]
                gaps = np.diff(mass_sorted_alpha)
                median_adjacent_gap = float(np.median(gaps)) if len(gaps) else None
                min_adjacent_gap = float(np.min(gaps)) if len(gaps) else None

                # ---- Part I: exact Gamma vs actual inversion, verified over EVERY pair ----
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
                    prompt=p_idx, layer=layer, head=head, n_candidates=n_cand, k_max=k_max,
                    top1_share=top1_share, entropy_nats=ent,
                    median_adjacent_mass_gap=median_adjacent_gap, min_adjacent_mass_gap=min_adjacent_gap,
                    n_pairs=n_pairs, n_meaningful_inverted=n_meaningful_inverted,
                    n_trivial_inverted=n_trivial_inverted,
                    frac_meaningful_inverted=(n_meaningful_inverted / n_pairs) if n_pairs else None,
                    max_gamma=max(gammas_meaningful) if gammas_meaningful else None,
                    median_gamma_meaningful=float(np.median(gammas_meaningful)) if gammas_meaningful else None,
                    positive_gamma_mass=positive_gamma_mass,
                    # filled in below once the per-k sweep is available:
                    n_changed_eviction_set=0, n_beyond_kmax=0,
                    n_measurable_kl=0, n_local_wins_measurable=0, n_mass_wins_measurable=0,
                ))

                all_meaningful_pairs.extend(stratum_pairs_this)

                # ---- per-k sweep for THIS stratum (needed for set-change + KL lookup) ----
                if stratum_pairs_this:  # only spend forward passes where there's something to explain
                    for k in range(1, k_max + 1):
                        for policy_name, ranking in (("attention_mass", mass_rank), ("local_damage", local_rank)):
                            evict_idx = ranking[:k]
                            mask = mask_from_indices(n, evict_idx)
                            m_at_qp = float(alpha_last[mask].sum())
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
                                policy=policy_name, m_at_measured_position=m_at_qp,
                                kl=kl, skipped_singular_at_measured=skipped,
                            ))

    # ---- resolve set-change + KL lookup for every meaningful pair, now that rows exist ----
    kl_lookup = {(r["prompt"], r["layer"], r["head"], r["k"], r["policy"]): r["kl"] for r in all_rows}
    strata_index = {(s["prompt"], s["layer"], s["head"]): s for s in stratum_summaries}
    rank_cache = {}  # (prompt, layer, head) -> (mass_rank, local_rank, k_max)

    # rebuild rank lists cheaply is wasteful; instead recompute change_k directly from stored pairs
    # by replaying the ranking arithmetic -- but we didn't persist mass_rank/local_rank globally.
    # Re-derive change_k using the pair's own (prompt, layer, head) by recomputing rankings once,
    # cached per stratum to avoid recomputation across many pairs in the same stratum.
    def get_ranks(p_idx, layer, head):
        key = (p_idx, layer, head)
        if key in rank_cache:
            return rank_cache[key]
        ids = tok(prompts[p_idx], return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        hidden_in = out.hidden_states[layer][0].double()
        alpha_full = out.attentions[layer][0, head].double()
        values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
        alpha_last = require_normalized(alpha_full[qp], renormalize=True)
        protect = {0, qp}
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

    # ---- empirical justification for the noise-floor threshold ----
    all_advantages_when_sets_differ = [
        p["advantage_mass_minus_local_at_change"] for p in all_meaningful_pairs
        if p["advantage_mass_minus_local_at_change"] is not None
    ]
    noise_floor_justification = dict(
        threshold_used=NOISE_FLOOR_KL,
        note=(
            "All |advantage| values at set-changing k, sorted, to check for a natural gap "
            "separating a noise cluster from a signal cluster (not chosen to manufacture a split)."
        ),
        sorted_abs_advantages=sorted(abs(a) for a in all_advantages_when_sets_differ),
    )

    provenance = capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"])
    summary = dict(
        provenance=provenance,
        layers=layers, heads=heads, q_max=Q_MAX,
        meaningful_alpha_floor=MEANINGFUL_ALPHA_FLOOR,
        noise_floor_kl=NOISE_FLOOR_KL,
        noise_floor_justification=noise_floor_justification,
        n_strata=len(layers) * len(heads), n_prompts=len(prompts),
        n_pairs_grand_total=n_pairs_grand_total,
        n_verification_checks=n_verification_checks,
        n_verification_mismatches=len(verification_mismatches),
        verification_mismatches=verification_mismatches,
        n_meaningful_inversions_total=len(all_meaningful_pairs),
        n_rows_kl_sweep=len(all_rows),
        stratum_summaries=stratum_summaries,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "pairs.json").write_text(json.dumps(all_meaningful_pairs, indent=2))
    (RESULTS_DIR / "rows.json").write_text(json.dumps(all_rows, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k not in ("stratum_summaries", "noise_floor_justification")}, indent=2))
    return summary


if __name__ == "__main__":
    run()
