r"""Diagnostic: is the token-importance score causal, and where does the
measured damage actually come from?

DIAGNOSTIC ONLY -- no production code path is modified.

Two questions the correlational experiments in this repository do not answer.

(1) COUNTERFACTUAL EVICTION (is the score causal?). `eviction_curve.py`
    always evicts the LOWEST-scored positions first. That shows a ranking is
    better than random, but not that the score tracks causal importance. Here
    each scorer is run in three directions at MATCHED budget:
        bottom  -- evict lowest-scored first  (the production direction)
        top     -- evict highest-scored first (the counterfactual)
        random  -- matched-size random subset
    If the score is causal, `top` must damage the output much more than
    `bottom`, and `random` must sit between them.

(2) WHICH QUERY ROWS CARRY THE MEASURED DAMAGE. The reported KL is read at
    the final position only. Eviction is nonetheless applied at every query
    row. Applying the identical delta at (a) every row versus (b) the final
    row alone separates "damage caused by evicting from the row being
    measured" from "damage arriving from other rows" -- the prefill/decode
    distinction, which this codebase does not otherwise separate.

Writes results/counterfactual_eviction/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.damage_metrics import kl_divergence, logit_l2
from kv_efficiency.eviction import require_normalized
from kv_efficiency.hooks import additive_hook
from kv_efficiency.policies import attention_mass_ranking, local_damage_scores
from kv_efficiency.provenance import capture_environment, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "counterfactual_eviction"
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_SEEDS = 5
EPS = 1e-9


def apply_delta(model, adapter, ids, layer, a, value_h, wo_h, mask, rows_mode, qp):
    """Exact per-row delta, applied either at every well-conditioned row or
    at the final row only. Returns (logits, n_rows_applied) or (None, 0)."""
    m = a[:, mask].sum(dim=-1)
    ok = m < 1.0 - EPS
    if rows_mode == "last_row_only":
        keep = torch.zeros_like(ok); keep[qp] = ok[qp]
    else:
        keep = ok
    if not bool(keep.any()):
        return None, 0
    o = a @ value_h
    o_b = a[:, mask] @ value_h[mask]
    denom = (1.0 - m).clone(); denom[~keep] = 1.0
    dc = (m.unsqueeze(-1) * o - o_b) / denom.unsqueeze(-1)
    dc[~keep] = 0.0
    delta = dc @ wo_h.T
    with additive_hook(adapter._layers()[layer], delta, position=None):
        with torch.no_grad():
            return model(ids).logits[0], int(keep.sum())


def run() -> dict:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(cfg["seed"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, cfg["dtype"]),
        attn_implementation=cfg["attn_implementation"]).eval()
    adapter = GPTNeoXAdapter(model)

    rows = []
    for p_idx, prompt in enumerate(cfg["evaluation_prompts"]):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]; qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
            base = model(ids).logits[0]

        for layer in cfg["layers"]:
            hid = out.hidden_states[layer][0].double()
            for head in cfg["heads"]:
                af = out.attentions[layer][0, head].double()
                a = require_normalized(af, renormalize=True)
                _, _, vh = adapter.head_context_and_value(layer, head, hid)
                vh = vh.double(); wo = adapter.w_o_head(layer, head).double()
                vd = head_values_dmodel(adapter, layer, head, hid)
                al = require_normalized(af[qp], renormalize=True)
                protect = {0, qp}
                cand = [i for i in range(n) if i not in protect]

                # two scorers, ascending order (least important first)
                mass_asc = attention_mass_ranking(al, protect=protect)
                dsc = local_damage_scores(al, vd, protect=protect)
                local_asc = sorted(dsc, key=lambda i: dsc[i]) + [i for i in cand if i not in dsc]

                for scorer, asc in (("attention_mass", mass_asc), ("local_damage", local_asc)):
                    for frac in FRACTIONS:
                        k = round(frac * len(cand))
                        if k == 0: continue
                        variants = {"bottom": asc[:k], "top": list(reversed(asc))[:k]}
                        for seed in range(N_SEEDS):
                            rng = np.random.default_rng(seed)
                            variants[f"random_seed{seed}"] = list(rng.permutation(cand)[:k])
                        for direction, idx in variants.items():
                            mask = torch.zeros(n, dtype=torch.bool)
                            for i in idx: mask[i] = True
                            for rows_mode in ("all_rows", "last_row_only"):
                                lg, nr = apply_delta(model, adapter, ids, layer, a, vh, wo, mask, rows_mode, qp)
                                rows.append(dict(
                                    prompt=p_idx, layer=layer, head=head, scorer=scorer,
                                    frac=frac, k=k, direction=direction, rows_mode=rows_mode,
                                    kl=None if lg is None else float(kl_divergence(base[qp], lg[qp])),
                                    logit_l2=None if lg is None else float(logit_l2(base[qp], lg[qp])),
                                    n_rows_applied=nr,
                                    status="skipped_all_rows_singular" if lg is None else "ok"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = dict(provenance=capture_environment(CONFIG_PATH, model=model, seed=cfg["seed"]),
                   n_rows=len(rows), n_ok=sum(1 for r in rows if r["status"] == "ok"),
                   n_skipped=sum(1 for r in rows if r["status"] != "ok"))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "rows.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
    return summary


if __name__ == "__main__":
    run()
