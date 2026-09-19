r"""Diagnostic: three semantics for the `m_B -> 1` guard, compared.

DIAGNOSTIC ONLY -- imports the production code and does not modify it.

`eviction.global_eviction` currently uses semantics (A):

  (A) ANY-ROW SKIP (production). `delta_from_block` raises if ANY query row
      of the full (seq, seq) attention matrix has m[q] >= 1 - 1e-9, and
      `global_eviction` then drops that (layer, head) stratum entirely.

Two alternatives that are well defined on exactly the same inputs:

  (B) MEASURED-ROW GUARD. Guard only the query row whose output is actually
      read (the last position -- the row the rankings come from and the row
      KL is computed at).

  (C) PER-ROW EXCLUSION. Apply the exact eviction delta at every query row
      whose denominator is well conditioned, and apply nothing at the rows
      that are singular. The identity is per-row, so this is exact where it
      is applied; no row gets a fabricated value.

For each we record the KL at the last position and how much eviction was
actually applied, so damage can be compared at MATCHED realised budget.

Writes results/compare_guard_semantics/.
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.allocator import allocate_budget, estimate_sensitivity
from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import require_normalized
from kv_efficiency.hooks import additive_hook
from kv_efficiency.policies import (
    attention_mass_ranking, local_damage_ranking, local_damage_scores, random_ranking,
)
from kv_efficiency.provenance import capture_environment, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "compare_guard_semantics"
GLOBAL_REMOVAL_TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5]
MIN_FRAC, MAX_FRAC = 0.05, 0.95
EPS = 1e-9


def deltas_for_semantics(a, value_h, wo_h, mask, qp, semantics):
    """Return (delta_dmodel or None, n_rows_applied, n_rows_singular).

    `a` is the full renormalized (seq, seq) attention matrix.
    """
    m = a[:, mask].sum(dim=-1)                      # (seq,)
    sing = m >= 1.0 - EPS
    o = a @ value_h
    o_b = a[:, mask] @ value_h[mask]

    if semantics == "A_any_row_skip":
        if bool(sing.any()):
            return None, 0, int(sing.sum())
        rows = torch.ones_like(sing)
    elif semantics == "B_measured_row_guard":
        if bool(sing[qp]):
            return None, 0, int(sing.sum())
        rows = ~sing                                # still cannot divide by 0 elsewhere
    elif semantics == "C_per_row_exclusion":
        rows = ~sing
        if not bool(rows.any()):
            return None, 0, int(sing.sum())
    else:
        raise ValueError(semantics)

    denom = (1.0 - m).clone()
    denom[~rows] = 1.0                              # placeholder; zeroed below
    delta_context = (m.unsqueeze(-1) * o - o_b) / denom.unsqueeze(-1)
    delta_context[~rows] = 0.0
    return delta_context @ wo_h.T, int(rows.sum()), int(sing.sum())


def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, cfg["dtype"]),
        attn_implementation=cfg["attn_implementation"]).eval()
    adapter = GPTNeoXAdapter(model)

    def fwd(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            return ids, model(ids, output_attentions=True, output_hidden_states=True)

    # calibration (identical to global_budget_curve)
    calib = []
    for prompt in cfg["calibration_prompts"]:
        ids, out = fwd(prompt); n = ids.shape[1]; qp = n - 1
        for L in cfg["layers"]:
            hid = out.hidden_states[L][0].double()
            for H in cfg["heads"]:
                af = out.attentions[L][0, H].double()
                vd = head_values_dmodel(adapter, L, H, hid)
                al = require_normalized(af[qp], renormalize=True)
                for i, s in local_damage_scores(al, vd, protect={0, qp}).items():
                    calib.append(dict(layer=L, head=H, position=i, d_local=s))
    sensitivity = estimate_sensitivity(calib)

    rows_out = []
    for p_idx, prompt in enumerate(cfg["evaluation_prompts"]):
        ids, out = fwd(prompt); n = ids.shape[1]; qp = n - 1
        with torch.no_grad():
            base = model(ids).logits[0]

        A, VD, WO, RANK, CAND = {}, {}, {}, {}, {}
        for L in cfg["layers"]:
            hid = out.hidden_states[L][0].double()
            for H in cfg["heads"]:
                st = (L, H)
                af = out.attentions[L][0, H].double()
                A[st] = require_normalized(af, renormalize=True)
                _, _, vh = adapter.head_context_and_value(L, H, hid)
                VD[st] = vh.double(); WO[st] = adapter.w_o_head(L, H).double()
                vd = head_values_dmodel(adapter, L, H, hid)
                al = require_normalized(af[qp], renormalize=True)
                pr = {0, qp}
                mr = attention_mass_ranking(al, protect=pr)
                lr = local_damage_ranking(al, vd, protect=pr)
                CAND[st] = mr; RANK[st] = dict(mass=mr, local=lr)

        for q_global in GLOBAL_REMOVAL_TARGETS:
            ub = {k: q_global for k in A}
            ab = allocate_budget(sensitivity, q_global, min_frac=MIN_FRAC, max_frac=MAX_FRAC) if sensitivity else ub
            plans = {"uniform_mass": (ub, "mass", None), "uniform_local": (ub, "local", None),
                     "allocated_mass": (ab, "mass", None), "allocated_local": (ab, "local", None)}
            for s in range(cfg["n_random_seeds"]):
                plans[f"uniform_random_seed{s}"] = (ub, "random", s)

            for pol, (budget, rk, seed) in plans.items():
                sel = {}
                for st, cand in CAND.items():
                    k = round(budget.get(st, q_global) * len(cand))
                    idx = random_ranking(n, protect={0, qp}, seed=seed)[:k] if rk == "random" else RANK[st][rk][:k]
                    mk = torch.zeros(n, dtype=torch.bool)
                    for i in idx: mk[i] = True
                    sel[st] = (mk, k)

                for sem in ("A_any_row_skip", "B_measured_row_guard", "C_per_row_exclusion"):
                    by_layer, applied, planned, skipped, sing_rows = {}, 0, 0, [], 0
                    for st, (mk, k) in sel.items():
                        planned += k
                        if k == 0: continue
                        d, nr, ns = deltas_for_semantics(A[st], VD[st], WO[st], mk, qp, sem)
                        sing_rows += ns
                        if d is None:
                            skipped.append(list(st)); continue
                        applied += k
                        by_layer[st[0]] = by_layer.get(st[0], 0) + d
                    with ExitStack() as stack:
                        for L, d in by_layer.items():
                            stack.enter_context(additive_hook(adapter._layers()[L], d, position=None))
                        with torch.no_grad():
                            lg = model(ids).logits[0]
                    rows_out.append(dict(
                        prompt=p_idx, q_global=q_global, policy=pol, semantics=sem,
                        kl=float(kl_divergence(base[qp], lg[qp])),
                        planned_entries=planned, applied_entries=applied,
                        effective_q=q_global * applied / planned if planned else 0.0,
                        n_strata_skipped=len(skipped), n_singular_rows=sing_rows))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = dict(provenance=capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"]),
                   mass_epsilon=EPS, n_rows=len(rows_out))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "rows.json").write_text(json.dumps(rows_out, indent=2))
    print(f"wrote {len(rows_out)} rows")
    return summary


if __name__ == "__main__":
    run()
