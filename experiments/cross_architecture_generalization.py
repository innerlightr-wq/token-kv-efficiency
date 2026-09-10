r"""Cross-architecture generalization pilot: GPT-2 (124M) vs. the frozen
Pythia-160M baseline (`results/full_grid_mechanistic_analysis/summary.json`).

**Pre-registered scope (frozen before this script saw any GPT-2 result):**
model=gpt2; same 3 evaluation-prompt STRINGS as the Pythia grid (reused
verbatim; tokenization will differ, reported not normalized away); layers
{0,6,11}; heads {0,3,6,9}; MEANINGFUL_ALPHA_FLOOR=1e-6; NOISE_FLOOR_KL=1e-8;
Q_MAX=0.95; identical candidate-pair, meaningful-inversion, mass-score, and
local-damage definitions as `full_grid_mechanistic_analysis.py`. Gates
G1-G7 are defined in that pre-registration, not after seeing results, and
are evaluated in Part 3 below unchanged from that definition.

**Structure:**
  Part 0 -- MANDATORY adapter validation, using the actual pilot prompts.
            Hard-stops (raises) if any tolerance is violated. No research
            result is computed before this passes.
  Part 1 -- pilot data collection, structurally identical to
            `full_grid_mechanistic_analysis.py` (same Gamma/inversion/KL
            logic, reused conceptually, not copy-pasted verbatim, since the
            per-stratum loop is now over GPT-2's config, and delete/eviction
            calls go through `GPT2Adapter` -- verified in Part 0 and by
            `tests/test_gpt2_adapter.py` to give correct results via the
            SAME `eviction.py` functions, unmodified).
  Part 2 -- Gamma-vs-inversion agreement, labeled strictly as an
            implementation sanity check (the identity is architecture-
            independent by derivation, not evidence from this run).
  Part 3 -- G1-G7 empirical measurements, per-stratum heterogeneity,
            comparison against the frozen Pythia baseline (read-only).
  Part 4 -- CAG verdict and full-grid-expansion recommendation.

Does not modify `attention_reconstruction.py`, `eviction.py`,
`local_algebra.py`, `policies.py`, any existing test, or any existing result
file. Writes only `results/cross_architecture_generalization/*`.
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
from kv_efficiency.hooks import residual_add_hook
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, local_damage_scores

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "cross_architecture_generalization"
PYTHIA_BASELINE_SUMMARY = ROOT / "results" / "full_grid_mechanistic_analysis" / "summary.json"

# ---------------------------------------------------------------------------
# PRE-REGISTERED PILOT CONFIG -- frozen before any GPT-2 result was seen.
# ---------------------------------------------------------------------------
MODEL_ID = "gpt2"
LAYERS = [0, 6, 11]
HEADS = [0, 3, 6, 9]
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

# ---------------------------------------------------------------------------
# Part-0 validation tolerances -- declared BEFORE running, not tuned after.
# ---------------------------------------------------------------------------
TOL_ROW_SUM = 1e-6           # attention row sums to 1
TOL_CAUSAL_ZERO = 0.0        # strictly zero mass on non-causal positions
TOL_QK_RECOMPUTE = 1e-4      # recomputed-vs-real attention weights, abs err
TOL_CPROJ_RECON_REL = 1e-3   # per-head context reconstruction, relative err
TOL_DELETE_RENORM = 1e-8     # delete-and-renormalize closed-form vs direct, abs err


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


# ---------------------------------------------------------------------------
# PART 0 -- mandatory adapter validation on the ACTUAL pilot prompts
# ---------------------------------------------------------------------------

def validate_adapter(tok, model, adapter) -> dict:
    print("=" * 78)
    print("PART 0 -- ADAPTER VALIDATION (mandatory gate; aborts on failure)")
    print("=" * 78)
    errors = dict(
        row_sum_max_dev=0.0, causal_leak_max=0.0, qk_recompute_max_abs_err=0.0,
        cproj_recon_max_rel_err=0.0, delete_renorm_max_abs_err=0.0,
        g_consistency_max_abs_err=0.0,
    )
    n_checks = dict(row_sum=0, causal=0, qk_recompute=0, cproj_recon=0, delete_renorm=0)

    for p_idx, prompt in enumerate(EVALUATION_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)

        for layer in LAYERS:
            hidden_in = out.hidden_states[layer][0].double()
            for head in HEADS:
                alpha_full = out.attentions[layer][0, head].double()

                # -- attention dims, causal masking, row sums (cheap, every stratum) --
                assert out.attentions[layer].shape == (1, adapter.n_heads, n, n)
                row_sums = alpha_full.sum(dim=-1)
                dev = float((row_sums - 1.0).abs().max())
                errors["row_sum_max_dev"] = max(errors["row_sum_max_dev"], dev)
                n_checks["row_sum"] += 1
                # strict upper-triangular zero-mass check (causal masking)
                triu_mass = float(torch.triu(alpha_full, diagonal=1).abs().max())
                errors["causal_leak_max"] = max(errors["causal_leak_max"], triu_mass)
                n_checks["causal"] += 1

                # -- Q/K/V split + head reshape: recompute attention, compare to real --
                with torch.no_grad():
                    q, k, _ = adapter.head_context_and_value(layer, head, hidden_in)
                scale = adapter.head_size ** -0.5
                scores = (q @ k.T) * scale
                causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
                scores = scores.masked_fill(~causal, float("-inf"))
                recomputed = torch.softmax(scores, dim=-1)
                err_qk = float((recomputed - alpha_full).abs().max())
                errors["qk_recompute_max_abs_err"] = max(errors["qk_recompute_max_abs_err"], err_qk)
                n_checks["qk_recompute"] += 1

                # -- delete-one-renormalize local-damage vs direct recomputation,
                #    several (up to 5) candidate positions, this stratum --
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                a_last = require_normalized(alpha_full[n - 1], renormalize=True)
                rng = np.random.default_rng(1000 * p_idx + 10 * layer + head)
                cand_pool = [i for i in range(n - 1) if i not in (0,)]
                sample = rng.choice(cand_pool, size=min(5, len(cand_pool)), replace=False)
                o_full = a_last @ values_dmodel
                for i in sample:
                    if float(a_last[i]) >= 1.0 - MASS_EPSILON:
                        continue
                    block = torch.zeros(n, dtype=torch.bool)
                    block[int(i)] = True
                    from kv_efficiency.eviction import delta_from_block
                    delta = delta_from_block(a_last, values_dmodel, block, renormalize=False)
                    predicted = o_full + delta
                    a_masked = a_last.clone()
                    a_masked[block] = 0.0
                    a_masked = a_masked / a_masked.sum()
                    o_direct = a_masked @ values_dmodel
                    err = float((predicted - o_direct).abs().max())
                    errors["delete_renorm_max_abs_err"] = max(errors["delete_renorm_max_abs_err"], err)
                    n_checks["delete_renorm"] += 1
                    g_a = float(torch.linalg.norm(o_full - values_dmodel[int(i)]))
                    g_b = float(torch.linalg.norm(a_last @ values_dmodel - values_dmodel[int(i)]))
                    errors["g_consistency_max_abs_err"] = max(errors["g_consistency_max_abs_err"], abs(g_a - g_b))

        # -- per-head output projection, one representative head per layer, this prompt --
        for layer in LAYERS:
            head = HEADS[0]
            capture: dict = {}
            with residual_add_hook(model.transformer.h[layer].attn.c_proj, capture):
                with torch.no_grad():
                    out2 = model(ids, output_attentions=True, output_hidden_states=True)
            hidden_in2 = out2.hidden_states[layer][0].double()
            alpha2 = out2.attentions[layer][0, head].double()
            hs = adapter.head_size
            real_context_h = capture["x"][0, :, head * hs : (head + 1) * hs].double()
            with torch.no_grad():
                _, _, value_h = adapter.head_context_and_value(layer, head, hidden_in2)
            manual_context_h = alpha2 @ value_h
            rel_err = float((real_context_h - manual_context_h).abs().max() / real_context_h.abs().max())
            errors["cproj_recon_max_rel_err"] = max(errors["cproj_recon_max_rel_err"], rel_err)
            n_checks["cproj_recon"] += 1

    print(f"attention dims OK for every (prompt,layer,head) checked: {sum(n_checks.values())} total checks")
    print(f"row-sum max deviation from 1     : {errors['row_sum_max_dev']:.3e}  (tol {TOL_ROW_SUM:.1e}), n={n_checks['row_sum']}")
    print(f"causal-mask max leaked mass      : {errors['causal_leak_max']:.3e}  (tol {TOL_CAUSAL_ZERO:.1e}), n={n_checks['causal']}")
    print(f"Q/K recompute max abs err        : {errors['qk_recompute_max_abs_err']:.3e}  (tol {TOL_QK_RECOMPUTE:.1e}), n={n_checks['qk_recompute']}")
    print(f"c_proj reconstruction max rel err: {errors['cproj_recon_max_rel_err']:.3e}  (tol {TOL_CPROJ_RECON_REL:.1e}), n={n_checks['cproj_recon']}")
    print(f"delete-renormalize max abs err   : {errors['delete_renorm_max_abs_err']:.3e}  (tol {TOL_DELETE_RENORM:.1e}), n={n_checks['delete_renorm']}")
    print(f"g_i two-ways-computed max abs err: {errors['g_consistency_max_abs_err']:.3e}")

    failures = []
    if errors["row_sum_max_dev"] > TOL_ROW_SUM:
        failures.append("row_sum_max_dev")
    if errors["causal_leak_max"] > TOL_CAUSAL_ZERO:
        failures.append("causal_leak_max")
    if errors["qk_recompute_max_abs_err"] > TOL_QK_RECOMPUTE:
        failures.append("qk_recompute_max_abs_err")
    if errors["cproj_recon_max_rel_err"] > TOL_CPROJ_RECON_REL:
        failures.append("cproj_recon_max_rel_err")
    if errors["delete_renorm_max_abs_err"] > TOL_DELETE_RENORM:
        failures.append("delete_renorm_max_abs_err")

    if failures:
        raise RuntimeError(
            f"ADAPTER VALIDATION FAILED on: {failures}. Errors: {errors}. "
            "Per instructions: STOP and debug the adapter. Pilot not run."
        )
    print("\nALL VALIDATION CHECKS PASSED. Proceeding to pilot data collection.\n")
    errors["n_checks"] = n_checks
    errors["tolerances"] = dict(
        row_sum=TOL_ROW_SUM, causal=TOL_CAUSAL_ZERO, qk_recompute=TOL_QK_RECOMPUTE,
        cproj_recon_rel=TOL_CPROJ_RECON_REL, delete_renorm=TOL_DELETE_RENORM,
    )
    return errors


# ---------------------------------------------------------------------------
# PART 1 -- pilot data collection (structurally mirrors full_grid_mechanistic_analysis.py)
# ---------------------------------------------------------------------------

def run_pilot(tok, model, adapter) -> dict:
    print("=" * 78)
    print("PART 1 -- PILOT DATA COLLECTION")
    print("=" * 78)

    all_rows = []
    all_meaningful_pairs = []
    stratum_summaries = []
    verification_mismatches = []
    n_pairs_grand_total = 0
    n_verification_checks = 0

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
                    prompt=p_idx, layer=layer, head=head, n_candidates=n_cand, k_max=k_max,
                    top1_share=top1_share, entropy_nats=ent,
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

        print(f"prompt {p_idx} done: n_tokens={n}")

    # ---- resolve set-change + KL lookup ----
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
        alpha_last = require_normalized(alpha_full[out.hidden_states[layer].shape[1] - 1], renormalize=True)
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
# PART 3/4 -- G1-G7, heterogeneity, comparison, verdict
# ---------------------------------------------------------------------------

def analyze(pilot: dict, pythia_baseline: dict) -> dict:
    print("=" * 78)
    print("PART 2/3 -- GAMMA SANITY CHECK, G1-G7, HETEROGENEITY, COMPARISON")
    print("=" * 78)

    pairs = pilot["all_meaningful_pairs"]
    n_total_pairs = pilot["n_pairs_grand_total"]
    n_meaningful = len(pairs)
    n_mismatches = len(pilot["verification_mismatches"])

    print(f"\n[Part 2] Gamma sign vs actual inversion -- IMPLEMENTATION SANITY CHECK ONLY")
    print(f"  (this identity is architecture-independent by derivation; a nonzero mismatch")
    print(f"   count here would indicate an ADAPTER BUG, not new cross-architecture evidence)")
    print(f"  n_verification_checks={pilot['n_verification_checks']}  n_mismatches={n_mismatches}")

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
    sum_advantage = float(sum(p["advantage_mass_minus_local_at_change"] for p in measurable)) if measurable else None
    median_advantage = float(np.median([p["advantage_mass_minus_local_at_change"] for p in measurable])) if measurable else None

    # G6: concentration vs prevalence, across the 36 strata
    strata = pilot["stratum_summaries"]
    top1 = np.array([s["top1_share"] for s in strata])
    ent = np.array([s["entropy_nats"] for s in strata])
    frac_inv = np.array([s["frac_meaningful_inverted"] if s["frac_meaningful_inverted"] is not None else 0.0 for s in strata])
    rho_top1, p_top1 = sstats.spearmanr(top1, frac_inv)
    rho_ent, p_ent = sstats.spearmanr(ent, frac_inv)

    # G7: |Gamma| vs |advantage| (magnitude) and vs sign(advantage)
    adv_pairs = [p for p in pairs if p.get("advantage_mass_minus_local_at_change") is not None]
    if len(adv_pairs) >= 5:
        gam = np.array([abs(p["Gamma"]) for p in adv_pairs])
        adv = np.array([p["advantage_mass_minus_local_at_change"] for p in adv_pairs])
        rho_gamma_mag, p_gamma_mag = sstats.spearmanr(gam, np.abs(adv))
        rho_gamma_sign, p_gamma_sign = sstats.spearmanr(gam, adv)
    else:
        rho_gamma_mag = p_gamma_mag = rho_gamma_sign = p_gamma_sign = None

    print(f"\n[G1] meaningful inversions occur: {n_meaningful > 0}  (n={n_meaningful})")
    print(f"[G2] prevalence: GPT-2={prevalence:.4%}  Pythia={pythia_prevalence:.4%}  "
          f"abs diff={prevalence - pythia_prevalence:+.4%}  "
          f"relative={(prevalence / pythia_prevalence - 1):+.1%}" if prevalence else "[G2] n/a")
    print(f"[G3] fraction of meaningful inversions that change the eviction set: "
          f"{frac_set_changing:.4%}  (Pythia 99.6%)" if frac_set_changing is not None else "[G3] n/a (no meaningful inversions)")
    print(f"[G4] fraction of set-changing clearing noise floor: "
          f"{frac_measurable_of_set_changing:.4%}  (Pythia 99.1%)" if frac_measurable_of_set_changing is not None else "[G4] n/a")
    print(f"[G5] local wins={n_local} mass wins={n_mass} "
          f"local_win_rate={local_win_rate:.4%}  (Pythia 56.3%)" if local_win_rate is not None else "[G5] n/a")
    print(f"[G6] Spearman(top1_share, prevalence)={rho_top1:.4f} p={p_top1:.4f}  (Pythia -0.650)")
    print(f"     Spearman(entropy, prevalence)={rho_ent:.4f} p={p_ent:.4f}  (Pythia +0.759)")
    print(f"[G7] Spearman(|Gamma|,|advantage|)={rho_gamma_mag}  Spearman(|Gamma|,advantage sign/mag)={rho_gamma_sign}"
          f"  (Pythia +0.159 / +0.015)")

    # per-stratum heterogeneity, all 36 (or fewer if some skipped) strata
    per_stratum_table = []
    for s in strata:
        per_stratum_table.append(dict(
            prompt=s["prompt"], layer=s["layer"], head=s["head"],
            n_candidates=s["n_candidates"], top1_share=s["top1_share"], entropy_nats=s["entropy_nats"],
            n_pairs=s["n_pairs"], n_meaningful_inverted=s["n_meaningful_inverted"],
            frac_meaningful_inverted=s["frac_meaningful_inverted"],
            n_changed_eviction_set=s["n_changed_eviction_set"], n_measurable_kl=s["n_measurable_kl"],
            n_local_wins_measurable=s["n_local_wins_measurable"], n_mass_wins_measurable=s["n_mass_wins_measurable"],
        ))
    zero_inversion_strata = [s for s in per_stratum_table if s["n_meaningful_inverted"] == 0]
    high_prevalence_strata = sorted(
        [s for s in per_stratum_table if s["frac_meaningful_inverted"]],
        key=lambda s: s["frac_meaningful_inverted"], reverse=True
    )[:5]
    strongly_mass_favoring = [
        s for s in per_stratum_table
        if s["n_measurable_kl"] >= 5 and s["n_mass_wins_measurable"] / s["n_measurable_kl"] >= 0.7
    ]
    strongly_local_favoring = [
        s for s in per_stratum_table
        if s["n_measurable_kl"] >= 5 and s["n_local_wins_measurable"] / s["n_measurable_kl"] >= 0.7
    ]

    n_independent_strata_with_inversions = sum(1 for s in per_stratum_table if s["n_meaningful_inverted"] > 0)
    n_independent_strata_with_measurable = sum(1 for s in per_stratum_table if s["n_measurable_kl"] > 0)

    strongest_pair = max(pairs, key=lambda p: abs(p.get("advantage_mass_minus_local_at_change") or 0.0), default=None)

    # ---- decision gate for full-grid expansion ----
    n_prompts_with_inversions = len({p["prompt"] for p in pairs})
    n_heads_with_inversions = len({(p["layer"], p["head"]) for p in pairs})
    expansion_conditions = dict(
        multiple_independent_strata=n_independent_strata_with_inversions >= 3,
        no_adapter_concerns=(n_mismatches == 0),
        enough_measurable_cases=(n_measurable >= 20),
        not_confined_to_one_prompt_or_head=(n_prompts_with_inversions >= 2 and n_heads_with_inversions >= 2),
    )
    expansion_justified = all(expansion_conditions.values())

    # ---- CAG verdict ----
    if n_meaningful == 0:
        verdict = "CAG-A"
        verdict_text = "No meaningful inversions occurred in the pilot after implementation controls."
    elif n_independent_strata_with_inversions < 2 or n_prompts_with_inversions < 2:
        verdict = "CAG-A"
        verdict_text = "Inversions occur but are confined to essentially one stratum/prompt -- not treated as reproduction."
    else:
        prevalence_ratio = prevalence / pythia_prevalence if pythia_prevalence else None
        prevalence_close = prevalence_ratio is not None and 0.3 <= prevalence_ratio <= 3.0
        set_change_close = frac_set_changing is not None and frac_set_changing >= 0.8
        measurable_close = frac_measurable_of_set_changing is not None and frac_measurable_of_set_changing >= 0.8
        concentration_qualitative_match = (rho_top1 < 0) and (rho_ent > 0)
        if prevalence_close and set_change_close and measurable_close and concentration_qualitative_match:
            verdict = "CAG-D"
            verdict_text = "Strong cross-architecture replication: prevalence, set-change rate, KL-measurability, and the concentration relationship all match Pythia qualitatively and are numerically close."
        elif set_change_close and measurable_close and concentration_qualitative_match:
            verdict = "CAG-C"
            verdict_text = "Qualitative cross-architecture replication: inversions, the concentration relationship, and downstream relevance (set-change + measurable-KL rates) all reproduce, though numerical prevalence differs from Pythia."
        else:
            verdict = "CAG-B"
            verdict_text = "Inversions reproduce, but prevalence and/or downstream behavior (set-change rate, measurable-KL rate, or the concentration relationship) differ substantially from Pythia."

    result = dict(
        n_total_pairs=n_total_pairs, n_meaningful=n_meaningful, n_verification_mismatches=n_mismatches,
        prevalence=prevalence, pythia_prevalence=pythia_prevalence,
        n_set_changing=n_set_changing, frac_set_changing=frac_set_changing,
        n_measurable=n_measurable, frac_measurable_of_set_changing=frac_measurable_of_set_changing,
        n_local_wins=n_local, n_mass_wins=n_mass, local_win_rate=local_win_rate,
        sum_advantage=sum_advantage, median_advantage=median_advantage,
        g6_spearman_top1_vs_prevalence=dict(rho=float(rho_top1), p=float(p_top1)),
        g6_spearman_entropy_vs_prevalence=dict(rho=float(rho_ent), p=float(p_ent)),
        g7_spearman_gamma_vs_abs_advantage=dict(rho=float(rho_gamma_mag) if rho_gamma_mag is not None else None,
                                                 p=float(p_gamma_mag) if p_gamma_mag is not None else None),
        g7_spearman_gamma_vs_signed_advantage=dict(rho=float(rho_gamma_sign) if rho_gamma_sign is not None else None,
                                                    p=float(p_gamma_sign) if p_gamma_sign is not None else None),
        per_stratum_table=per_stratum_table,
        zero_inversion_strata=zero_inversion_strata,
        high_prevalence_strata=high_prevalence_strata,
        strongly_mass_favoring_strata=strongly_mass_favoring,
        strongly_local_favoring_strata=strongly_local_favoring,
        n_independent_strata_with_inversions=n_independent_strata_with_inversions,
        n_independent_strata_with_measurable=n_independent_strata_with_measurable,
        n_prompts_with_inversions=n_prompts_with_inversions,
        n_heads_with_inversions=n_heads_with_inversions,
        strongest_pair=strongest_pair,
        expansion_conditions=expansion_conditions,
        expansion_justified=expansion_justified,
        verdict=verdict, verdict_text=verdict_text,
    )
    return result


def run() -> None:
    torch.manual_seed(0)
    tok, model = load_model()
    adapter = GPT2Adapter(model)

    validation = validate_adapter(tok, model, adapter)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "adapter_validation.json").write_text(json.dumps(validation, indent=2))

    pilot = run_pilot(tok, model, adapter)

    pythia_baseline = json.loads(PYTHIA_BASELINE_SUMMARY.read_text())
    analysis = analyze(pilot, pythia_baseline)

    (RESULTS_DIR / "pairs.json").write_text(json.dumps(pilot["all_meaningful_pairs"], indent=2))
    summary = dict(
        model_id=MODEL_ID, layers=LAYERS, heads=HEADS, q_max=Q_MAX,
        meaningful_alpha_floor=MEANINGFUL_ALPHA_FLOOR, noise_floor_kl=NOISE_FLOOR_KL,
        n_layer_head_combos=len(LAYERS) * len(HEADS), n_prompts=len(EVALUATION_PROMPTS),
        n_prompt_layer_head_cells=len(LAYERS) * len(HEADS) * len(EVALUATION_PROMPTS),
        adapter_validation=validation,
        n_verification_checks=pilot["n_verification_checks"],
        n_verification_mismatches=len(pilot["verification_mismatches"]),
        verification_mismatches=pilot["verification_mismatches"],
        analysis=analysis,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print("\nDone. See results/cross_architecture_generalization/report.txt")


def write_report(summary: dict) -> None:
    a = summary["analysis"]
    L = []
    A = L.append
    A("CROSS-ARCHITECTURE GENERALIZATION PILOT -- GPT-2 (124M) vs Pythia-160M")
    A("=" * 78)
    A(f"model={summary['model_id']}  layers={summary['layers']}  heads={summary['heads']}  "
      f"n_prompts={summary['n_prompts']}  n_layer_head_combos={summary['n_layer_head_combos']}  "
      f"n_prompt_layer_head_cells={summary['n_prompt_layer_head_cells']}")
    A("")
    A("-" * 78)
    A("ADAPTER VALIDATION")
    A("-" * 78)
    v = summary["adapter_validation"]
    for k in ("row_sum_max_dev", "causal_leak_max", "qk_recompute_max_abs_err",
              "cproj_recon_max_rel_err", "delete_renorm_max_abs_err", "g_consistency_max_abs_err"):
        A(f"  {k:<32}: {v[k]:.3e}")
    A(f"  checks run: {v['n_checks']}")
    A("")
    A("-" * 78)
    A("GAMMA IMPLEMENTATION SANITY CHECK (architecture-independent identity)")
    A("-" * 78)
    A(f"  n_verification_checks={summary['n_verification_checks']}  "
      f"n_mismatches={summary['n_verification_mismatches']}")
    A("")
    A("-" * 78)
    A("G1-G7")
    A("-" * 78)
    A(f"G1 inversions occur: {a['n_meaningful'] > 0}  (n={a['n_meaningful']} of {a['n_total_pairs']} pairs)")
    A(f"G2 prevalence: GPT-2={a['prevalence']:.4%}  Pythia={a['pythia_prevalence']:.4%}")
    A(f"G3 set-change rate: {a['frac_set_changing']}  (Pythia 0.996)")
    A(f"G4 measurable-KL rate: {a['frac_measurable_of_set_changing']}  (Pythia 0.991)")
    A(f"G5 local wins={a['n_local_wins']} mass wins={a['n_mass_wins']} rate={a['local_win_rate']}  (Pythia 0.563)")
    A(f"   sum advantage={a['sum_advantage']}  median={a['median_advantage']}")
    A(f"G6 top1_share vs prevalence: {a['g6_spearman_top1_vs_prevalence']}  (Pythia rho=-0.650)")
    A(f"   entropy vs prevalence: {a['g6_spearman_entropy_vs_prevalence']}  (Pythia rho=+0.759)")
    A(f"G7 |Gamma| vs |advantage|: {a['g7_spearman_gamma_vs_abs_advantage']}  (Pythia rho=+0.159)")
    A(f"   |Gamma| vs signed advantage: {a['g7_spearman_gamma_vs_signed_advantage']}  (Pythia rho=+0.015)")
    A("")
    A("-" * 78)
    A("PER-STRATUM HETEROGENEITY")
    A("-" * 78)
    A(f"(prompt,layer,head) cells with zero inversions: {len(a['zero_inversion_strata'])} / {summary['n_prompt_layer_head_cells']}")
    for s in a["zero_inversion_strata"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}")
    A("top-5 highest inversion-prevalence strata:")
    for s in a["high_prevalence_strata"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  "
          f"frac={s['frac_meaningful_inverted']:.4f}  n_inv={s['n_meaningful_inverted']}  top1={s['top1_share']:.3f}")
    A(f"strongly mass-favoring strata (>=70% mass wins, n>=5 measurable): {len(a['strongly_mass_favoring_strata'])}")
    for s in a["strongly_mass_favoring_strata"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  "
          f"mass_win_rate={s['n_mass_wins_measurable']/s['n_measurable_kl']:.3f}  n={s['n_measurable_kl']}")
    A(f"strongly local-favoring strata (>=70% local wins, n>=5 measurable): {len(a['strongly_local_favoring_strata'])}")
    for s in a["strongly_local_favoring_strata"]:
        A(f"    prompt={s['prompt']} layer={s['layer']} head={s['head']}  "
          f"local_win_rate={s['n_local_wins_measurable']/s['n_measurable_kl']:.3f}  n={s['n_measurable_kl']}")
    A("")
    A(f"(prompt,layer,head) cells with >=1 inversion: {a['n_independent_strata_with_inversions']} / {summary['n_prompt_layer_head_cells']}")
    A(f"(prompt,layer,head) cells with >=1 measurable KL pair: {a['n_independent_strata_with_measurable']} / {summary['n_prompt_layer_head_cells']}")
    A(f"prompts with >=1 inversion: {a['n_prompts_with_inversions']} / {summary['n_prompts']}")
    A(f"(layer,head) combos with >=1 inversion: {a['n_heads_with_inversions']} / {len(summary['layers'])*len(summary['heads'])}")
    A("")
    A("-" * 78)
    A("STRONGEST PAIR (largest |advantage|)")
    A("-" * 78)
    A(json.dumps(a["strongest_pair"], indent=2) if a["strongest_pair"] else "none")
    A("")
    A("-" * 78)
    A("FULL-GRID EXPANSION DECISION GATE")
    A("-" * 78)
    for k, v2 in a["expansion_conditions"].items():
        A(f"  {k}: {v2}")
    A(f"EXPANSION JUSTIFIED: {a['expansion_justified']}")
    A("")
    A("-" * 78)
    A("VERDICT")
    A("-" * 78)
    A(a["verdict"])
    A(a["verdict_text"])

    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
