r"""Phase 7: reproduce the old F10 pilot as faithfully as possible.

Provenance: ports `~/Desktop/Token Research/files (29)/scratch/run_pilot.py
::phase4_f10_pilot` using the new package modules. Same model, same two
prompts, same layers/heads/block_size/n_blocks_per_head, same seed. This is
NOT a redesign -- the purpose is provenance, not improvement. See
docs/MIGRATION_NOTES.md.

Reports both the pooled and preregistered within-head statistics without
retrofitting a new hypothesis into the old preregistration (F10:
`falsification.yaml`, median within-head spearman(tier_a_kl, tier_b_kl),
fires_when < 0.7).
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
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import delta_from_block, require_normalized, tier_a_perturbation, tier_b_eviction

MODEL_ID = "EleutherAI/pythia-160m"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "pilot_reproduction"

PROMPTS = [
    "The history of the Roman Empire is often divided into three stages: "
    "Rome under kings, Rome as a republic governed by elected senators, and "
    "Rome under the rule of emperors. The empire reached its greatest extent "
    "under the reign of Trajan, stretching from Britain to the Persian Gulf.",
    "Photosynthesis is the process by which green plants and some other "
    "organisms use sunlight to synthesize nutrients from carbon dioxide and "
    "water. Photosynthesis in plants generally involves the green pigment "
    "chlorophyll and generates oxygen as a byproduct.",
]

LAYERS = [1, 4, 7, 10]
HEADS = [0, 3, 6, 9]
BLOCK_SIZE = 4
N_BLOCKS_PER_HEAD = 3


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tok, model


def run() -> dict:
    torch.manual_seed(0)
    tok, model = load_model()
    adapter = GPTNeoXAdapter(model)

    records = []
    for p_idx, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        logits_full = out.logits[0]

        for layer in LAYERS:
            hidden_in = out.hidden_states[layer][0].double()
            for head in HEADS:
                alpha_full = out.attentions[layer][0, head].double()
                values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
                starts = np.linspace(1, max(2, n - BLOCK_SIZE - 1), N_BLOCKS_PER_HEAD)
                for s in sorted(set(int(x) for x in starts)):
                    block = slice(s, min(s + BLOCK_SIZE, qp))
                    if block.stop - block.start < 2:
                        continue
                    try:
                        alpha_last = require_normalized(alpha_full[qp], renormalize=True)
                        m = float(alpha_last[block].sum())
                        if m >= 1.0 - 1e-6 or m <= 0.0:
                            continue
                        d_local = delta_from_block(alpha_last, values_dmodel, block, renormalize=False)
                        d_local_norm = float(torch.linalg.norm(d_local))

                        _, logits_a = tier_a_perturbation(
                            model, adapter, ids, layer=layer, delta=d_local.float(), position=qp
                        )
                        kl_a = float(kl_divergence(logits_full[qp], logits_a[0]))

                        logits_b_all, _ = tier_b_eviction(
                            model, adapter, ids, alpha_full, hidden_in, layer=layer, head=head, block=block
                        )
                        kl_b = float(kl_divergence(logits_full[qp], logits_b_all[qp]))

                        records.append(
                            dict(prompt=p_idx, layer=layer, head=head, block_start=s,
                                 m_B=m, d_local=d_local_norm, kl_a=kl_a, kl_b=kl_b)
                        )
                    except ValueError:
                        continue

    d_local = np.array([r["d_local"] for r in records])
    kl_a = np.array([r["kl_a"] for r in records])
    kl_b = np.array([r["kl_b"] for r in records])

    rho_local_b, p_local_b = spearmanr(d_local, kl_b)
    rho_a_b, p_a_b = spearmanr(kl_a, kl_b)

    per_head = {}
    for layer in LAYERS:
        for head in HEADS:
            sub = [r for r in records if r["layer"] == layer and r["head"] == head]
            if len(sub) < 3:
                continue
            a = np.array([r["kl_a"] for r in sub])
            b = np.array([r["kl_b"] for r in sub])
            rho_ab, _ = spearmanr(a, b)
            per_head[f"{layer}_{head}"] = dict(n=len(sub), rho_a_b=None if np.isnan(rho_ab) else rho_ab)

    rhos_ab = [v["rho_a_b"] for v in per_head.values() if v["rho_a_b"] is not None]
    median_within_head = float(np.median(rhos_ab)) if rhos_ab else float("nan")

    summary = dict(
        model_id=MODEL_ID,
        n_records=len(records),
        layers=LAYERS,
        heads=HEADS,
        block_size=BLOCK_SIZE,
        n_blocks_per_head=N_BLOCKS_PER_HEAD,
        n_prompts=len(PROMPTS),
        pooled_spearman_local_vs_tierB=float(rho_local_b),
        pooled_spearman_tierA_vs_tierB=float(rho_a_b),
        f10_median_within_head_spearman_tierA_tierB=median_within_head,
        f10_fires_when="< 0.7",
        f10_fires=bool(median_within_head < 0.7) if not np.isnan(median_within_head) else None,
        per_head=per_head,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "records.json", "w") as f:
        json.dump(records, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
