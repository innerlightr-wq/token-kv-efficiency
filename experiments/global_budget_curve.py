r"""Milestone 2, Phases 7-10: head/layer budget allocator and the global
eviction-efficiency curve.

Five policies (Phase 7):
  1. uniform_random        -- uniform per-stratum budget, random ranking
  2. uniform_mass          -- uniform per-stratum budget, attention-mass ranking
  3. uniform_local         -- uniform per-stratum budget, local-damage ranking
  4. allocated_mass        -- sensitivity-allocated budget, attention-mass ranking
  5. allocated_local       -- sensitivity-allocated budget, local-damage ranking

Sensitivity for policies 4/5 is estimated ONLY from `calibration_prompts`
(Phase 8); all reported damage is measured ONLY on `evaluation_prompts`.
Eviction is joint across every (layer, head) in the grid in one forward
pass per prompt (`eviction.global_eviction`) -- exact within a layer,
an approximation across layers (see `eviction.py`'s module docstring).

Writes to results/milestone2_global_budget/ (new directory).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.allocator import allocate_budget, estimate_sensitivity
from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import global_eviction, require_normalized
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, random_ranking
from kv_efficiency.provenance import capture_environment, load_config, memory_bytes_saved

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "milestone2_global_budget"

GLOBAL_REMOVAL_TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5]
MIN_FRAC, MAX_FRAC = 0.05, 0.95  # allocator bounds: never fully protect or fully evict a stratum
DAMAGE_TOLERANCES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


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


def forward_prompt(tok, model, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    return ids, out


def collect_calibration_records(tok, model, adapter, cfg) -> list[dict]:
    """d_local for a sample of singleton positions, calibration prompts only."""
    from kv_efficiency.policies import local_damage_scores

    records = []
    for prompt in cfg["calibration_prompts"]:
        ids, out = forward_prompt(tok, model, prompt)
        n = ids.shape[1]
        qp = n - 1
        for layer in cfg["layers"]:
            hidden_in = out.hidden_states[layer][0].double()
            for head in cfg["heads"]:
                alpha_full = out.attentions[layer][0, head].double()
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                alpha_last = require_normalized(alpha_full[qp], renormalize=True)
                scores = local_damage_scores(alpha_last, values_dmodel, protect={0, qp})
                for i, s in scores.items():
                    records.append(dict(layer=layer, head=head, position=i, d_local=s))
    return records


def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    tok, model = load_model(cfg)
    adapter = GPTNeoXAdapter(model)

    # ---- Phase 8: calibration ----
    calibration_records = collect_calibration_records(tok, model, adapter, cfg)
    sensitivity = estimate_sensitivity(calibration_records)

    # ---- Phase 9/10: evaluation ----
    n_seeds = cfg["n_random_seeds"]
    rows = []

    for p_idx, prompt in enumerate(cfg["evaluation_prompts"]):
        ids, out = forward_prompt(tok, model, prompt)
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            baseline_logits = model(ids).logits[0]

        per_stratum_data = {}
        cand_by_stratum = {}
        rankings_by_stratum = {}
        for layer in cfg["layers"]:
            hidden_in = out.hidden_states[layer][0].double()
            for head in cfg["heads"]:
                alpha_full = out.attentions[layer][0, head].double()
                per_stratum_data[(layer, head)] = (alpha_full, hidden_in)
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                alpha_last = require_normalized(alpha_full[qp], renormalize=True)
                protect = {0, qp}
                mass_rank = attention_mass_ranking(alpha_last, protect=protect)
                local_rank = local_damage_ranking(alpha_last, values_dmodel, protect=protect)
                cand_by_stratum[(layer, head)] = mass_rank  # candidate set (any ranking has same set)
                rankings_by_stratum[(layer, head)] = dict(mass=mass_rank, local=local_rank)

        for q_global in GLOBAL_REMOVAL_TARGETS:
            uniform_budget = {k: q_global for k in per_stratum_data}
            allocated_budget = allocate_budget(sensitivity, q_global, min_frac=MIN_FRAC, max_frac=MAX_FRAC) \
                if sensitivity else uniform_budget

            policies = {
                "uniform_mass": (uniform_budget, "mass"),
                "uniform_local": (uniform_budget, "local"),
                "allocated_mass": (allocated_budget, "mass"),
                "allocated_local": (allocated_budget, "local"),
            }

            for policy_name, (budget, ranking_key) in policies.items():
                selections = {}
                n_removed_total = 0
                for stratum, cand in cand_by_stratum.items():
                    q_h = budget.get(stratum, q_global)
                    k = round(q_h * len(cand))
                    n_removed_total += k
                    idx = rankings_by_stratum[stratum][ranking_key][:k]
                    selections[stratum] = mask_from_indices(n, idx)
                logits, skipped = global_eviction(model, adapter, ids, per_stratum_data, selections)
                kl = float(kl_divergence(baseline_logits[qp], logits[qp]))
                rows.append(dict(prompt=p_idx, q_global=q_global, policy=policy_name, kl=kl,
                                  n_entries_removed=n_removed_total, skipped_strata=skipped))

            # uniform_random, averaged over seeds
            random_kls, random_removed, random_skipped = [], [], []
            for seed in range(n_seeds):
                selections = {}
                n_removed_total = 0
                for stratum, cand in cand_by_stratum.items():
                    k = round(q_global * len(cand))
                    n_removed_total += k
                    order = random_ranking(n, protect={0, qp}, seed=seed)
                    idx = order[:k]
                    selections[stratum] = mask_from_indices(n, idx)
                logits, skipped = global_eviction(model, adapter, ids, per_stratum_data, selections)
                random_kls.append(float(kl_divergence(baseline_logits[qp], logits[qp])))
                random_removed.append(n_removed_total)
                if skipped:
                    random_skipped.append(skipped)
            rows.append(dict(prompt=p_idx, q_global=q_global, policy="uniform_random",
                              kl=float(np.mean(random_kls)), kl_std=float(np.std(random_kls)),
                              n_entries_removed=float(np.mean(random_removed)), n_seeds=n_seeds,
                              skipped_strata=random_skipped))

    # ---- summaries ----
    policies_all = sorted(set(r["policy"] for r in rows))
    curve = []
    for policy in policies_all:
        for q in GLOBAL_REMOVAL_TARGETS:
            sub = [r for r in rows if r["policy"] == policy and r["q_global"] == q]
            if not sub:
                continue
            kls = [r["kl"] for r in sub]
            n_removed = float(np.mean([r["n_entries_removed"] for r in sub]))
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            bytes_saved = memory_bytes_saved(n_entries_removed=int(round(n_removed)), head_dim=head_dim)
            curve.append(dict(policy=policy, q_global=q, mean_kl=float(np.mean(kls)),
                               median_kl=float(np.median(kls)), n=len(kls),
                               mean_entries_removed=n_removed, estimated_bytes_saved=bytes_saved))

    # efficiency frontier: max tested q with mean_kl <= tolerance, per policy/tolerance
    frontier = []
    for policy in policies_all:
        policy_curve = sorted([c for c in curve if c["policy"] == policy], key=lambda c: c["q_global"])
        for tol in DAMAGE_TOLERANCES:
            feasible = [c["q_global"] for c in policy_curve if c["mean_kl"] <= tol]
            frontier.append(dict(policy=policy, damage_tolerance=tol,
                                  max_tested_removal_fraction=max(feasible) if feasible else None))

    n_singularity_skips = sum(
        len(r.get("skipped_strata") or []) if r["policy"] != "uniform_random"
        else sum(len(s) for s in (r.get("skipped_strata") or []))
        for r in rows
    )

    summary = dict(
        provenance=capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"]),
        allocator_bounds=dict(min_frac=MIN_FRAC, max_frac=MAX_FRAC),
        n_calibration_records=len(calibration_records),
        n_strata=len(sensitivity),
        sensitivity_estimate={f"{k[0]}_{k[1]}": v for k, v in sensitivity.items()},
        n_singularity_skips=n_singularity_skips,
        curve=curve,
        efficiency_frontier=frontier,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "rows.json").write_text(json.dumps(rows, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k not in ("sensitivity_estimate",)}, indent=2))
    return summary


if __name__ == "__main__":
    run()
