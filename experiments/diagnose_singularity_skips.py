r"""Diagnostic: what exactly triggers the `m_B -> 1` guard in `global_eviction`?

DIAGNOSTIC ONLY. This script does not change any policy or production code
path. It replicates `global_budget_curve.py`'s selection logic exactly, then
-- instead of letting `delta_from_block` raise -- recomputes the per-query-row
mass vector

    m[q] = sum_{i in mask, i <= q} alpha[q, i]

itself and records which query rows are singular, how large each row's causal
support is, and whether the stratum would still be skipped under alternative
guard definitions.

The guard in `eviction.delta_from_block` is

    if torch.any(m >= 1.0 - 1e-9): raise

i.e. an OR over *every* query row of the full (seq, seq) attention matrix.
A single pathological row cancels the whole stratum's eviction. The rankings
that build `mask`, by contrast, are computed from the LAST row only
(`alpha_full[qp]`), and the reported damage (KL) is read at the last position
only. This script measures whether the rows that trigger the guard are the
rows anyone is actually measuring.

Writes results/diagnose_singularity_skips/.
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
from kv_efficiency.eviction import require_normalized
from kv_efficiency.policies import (
    attention_mass_ranking,
    local_damage_ranking,
    local_damage_scores,
    random_ranking,
)
from kv_efficiency.provenance import capture_environment, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "diagnose_singularity_skips"

GLOBAL_REMOVAL_TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5]
MIN_FRAC, MAX_FRAC = 0.05, 0.95
MASS_EPS = 1e-9
EXCLUSION_THRESHOLDS = [4, 8, 16, 32]


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, cfg["dtype"]),
        attn_implementation=cfg["attn_implementation"],
    ).eval()
    return tok, model


def mask_from_indices(n: int, indices) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return mask


def forward_prompt(tok, model, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    return ids, out


def collect_calibration(tok, model, adapter, cfg):
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
                for i, s in local_damage_scores(alpha_last, values_dmodel, protect={0, qp}).items():
                    records.append(dict(layer=layer, head=head, position=i, d_local=s))
    return records


def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    tok, model = load_model(cfg)
    adapter = GPTNeoXAdapter(model)

    sensitivity = estimate_sensitivity(collect_calibration(tok, model, adapter, cfg))
    n_seeds = cfg["n_random_seeds"]

    events = []      # one record per (prompt, q_global, policy, stratum)
    row_events = []  # one record per singular query row

    for p_idx, prompt in enumerate(cfg["evaluation_prompts"]):
        ids, out = forward_prompt(tok, model, prompt)
        n = ids.shape[1]
        qp = n - 1

        alpha_by_stratum, cand_by_stratum, rank_by_stratum = {}, {}, {}
        for layer in cfg["layers"]:
            hidden_in = out.hidden_states[layer][0].double()
            for head in cfg["heads"]:
                alpha_full = out.attentions[layer][0, head].double()
                a = require_normalized(alpha_full, renormalize=True)
                alpha_by_stratum[(layer, head)] = a
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                alpha_last = require_normalized(alpha_full[qp], renormalize=True)
                protect = {0, qp}
                mass_rank = attention_mass_ranking(alpha_last, protect=protect)
                local_rank = local_damage_ranking(alpha_last, values_dmodel, protect=protect)
                cand_by_stratum[(layer, head)] = mass_rank
                rank_by_stratum[(layer, head)] = dict(mass=mass_rank, local=local_rank)

        for q_global in GLOBAL_REMOVAL_TARGETS:
            uniform_budget = {k: q_global for k in alpha_by_stratum}
            allocated_budget = (
                allocate_budget(sensitivity, q_global, min_frac=MIN_FRAC, max_frac=MAX_FRAC)
                if sensitivity else uniform_budget
            )
            plans = {
                "uniform_mass": (uniform_budget, "mass", None),
                "uniform_local": (uniform_budget, "local", None),
                "allocated_mass": (allocated_budget, "mass", None),
                "allocated_local": (allocated_budget, "local", None),
            }
            for seed in range(n_seeds):
                plans[f"uniform_random_seed{seed}"] = (uniform_budget, "random", seed)

            for policy_name, (budget, ranking_key, seed) in plans.items():
                for stratum, cand in cand_by_stratum.items():
                    q_h = budget.get(stratum, q_global)
                    k = round(q_h * len(cand))
                    if ranking_key == "random":
                        idx = random_ranking(n, protect={0, qp}, seed=seed)[:k]
                    else:
                        idx = rank_by_stratum[stratum][ranking_key][:k]
                    if k == 0:
                        continue
                    mask = mask_from_indices(n, idx)
                    a = alpha_by_stratum[stratum]
                    m = a[:, mask].sum(dim=-1).numpy()          # per-query-row mass
                    sing = np.flatnonzero(m >= 1.0 - MASS_EPS)  # singular rows
                    w = np.arange(1, n + 1)                     # w_q = q + 1

                    ev = dict(
                        prompt=p_idx, seq_len=n, q_global=q_global, policy=policy_name,
                        layer=stratum[0], head=stratum[1], k_evicted=k,
                        skipped=bool(sing.size > 0),
                        n_singular_rows=int(sing.size),
                        singular_rows=[int(x) for x in sing],
                        min_singular_row=int(sing.min()) if sing.size else None,
                        min_singular_w=int(w[sing.min()]) if sing.size else None,
                        max_singular_row=int(sing.max()) if sing.size else None,
                        m_last=float(m[qp]),
                        m_max=float(m.max()), m_argmax=int(m.argmax()),
                        m_median=float(np.median(m)), m_mean=float(m.mean()),
                        last_row_singular=bool(m[qp] >= 1.0 - MASS_EPS),
                        alpha_last_on_key0=float(a[qp, 0]),
                        alpha_row1_on_key0=float(a[1, 0]) if n > 1 else None,
                    )
                    # alternative guards: exclude short-causal-window rows
                    for thr in EXCLUSION_THRESHOLDS:
                        keep = w >= thr
                        ev[f"skipped_excl_w_lt_{thr}"] = bool(np.any(m[keep] >= 1.0 - MASS_EPS))
                    events.append(ev)

                    for q in sing:
                        row_events.append(dict(
                            prompt=p_idx, q_global=q_global, policy=policy_name,
                            layer=stratum[0], head=stratum[1],
                            query_row=int(q), w_q=int(w[q]), seq_len=n,
                            m=float(m[q]),
                            n_masked_visible=int(mask[: q + 1].sum()),
                            alpha_on_key0=float(a[q, 0]),
                        ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = dict(
        provenance=capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"]),
        mass_epsilon=MASS_EPS,
        n_stratum_attempts=len(events),
        n_skipped=sum(1 for e in events if e["skipped"]),
        n_singular_rows_total=len(row_events),
        n_skipped_last_row=sum(1 for e in events if e["last_row_singular"]),
        exclusion_survivors={
            f"w_lt_{thr}": sum(1 for e in events if e[f"skipped_excl_w_lt_{thr}"])
            for thr in EXCLUSION_THRESHOLDS
        },
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "events.json").write_text(json.dumps(events, indent=2))
    (RESULTS_DIR / "singular_rows.json").write_text(json.dumps(row_events, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
