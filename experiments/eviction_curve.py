r"""Phase 10: the actual efficiency-vs-damage experiment.

NOT a correlation study. For a fixed (layer, head), removes a fraction `q`
of candidate key positions using each policy in `policies.py`, evicts them
jointly via `eviction.tier_b_eviction` (generalized to a boolean mask), and
measures downstream KL at the final position against the same baseline.
Same prompts used for every policy (paired comparison).

Scope, stated plainly: single (layer, head), not the whole model's KV
cache. Generalizing eviction to many heads/layers simultaneously is
non-trivial (deltas from different heads interact only additively at the
SAME layer, but layer-to-layer propagation would need re-deriving; not
attempted this round -- see docs/RESEARCH_STATUS.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import tier_b_eviction
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, random_ranking

MODEL_ID = "EleutherAI/pythia-160m"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "eviction_curve.json"

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

LAYER, HEAD = 3, 2  # same as validate_local_identity.py -- not cherry-picked from the pilot
REMOVAL_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_RANDOM_SEEDS = 5


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tok, model


def mask_from_indices(n: int, indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return mask


def run() -> dict:
    torch.manual_seed(0)
    tok, model = load_model()
    adapter = GPTNeoXAdapter(model)

    rows = []
    for p_idx, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        n = ids.shape[1]
        qp = n - 1
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
        logits_full = out.logits[0]
        hidden_in = out.hidden_states[LAYER][0].double()
        alpha_full = out.attentions[LAYER][0, HEAD].double()
        values_dmodel = head_values_dmodel(adapter, LAYER, HEAD, hidden_in)

        protect = {0, qp}
        candidates = [i for i in range(n) if i not in protect]
        n_cand = len(candidates)

        rankings = {
            "attention_mass": attention_mass_ranking(alpha_full[qp], protect=protect),
            "local_damage": local_damage_ranking(alpha_full[qp], values_dmodel, protect=protect),
        }

        for q in REMOVAL_FRACTIONS:
            k = round(q * n_cand)

            for policy_name, ranking in rankings.items():
                evict_idx = ranking[:k]
                mask = mask_from_indices(n, evict_idx)
                if k == 0:
                    kl = 0.0
                else:
                    try:
                        logits_b, _ = tier_b_eviction(
                            model, adapter, ids, alpha_full, hidden_in,
                            layer=LAYER, head=HEAD, block=mask,
                        )
                        kl = float(kl_divergence(logits_full[qp], logits_b[qp]))
                    except ValueError as e:
                        kl = None
                rows.append(dict(prompt=p_idx, policy=policy_name, q=q, k_evicted=k,
                                  n_candidates=n_cand, kl=kl))

            # random: average over seeds
            random_kls = []
            for seed in range(N_RANDOM_SEEDS):
                ranking = random_ranking(n, protect=protect, seed=seed)
                evict_idx = ranking[:k]
                mask = mask_from_indices(n, evict_idx)
                if k == 0:
                    random_kls.append(0.0)
                    continue
                try:
                    logits_b, _ = tier_b_eviction(
                        model, adapter, ids, alpha_full, hidden_in,
                        layer=LAYER, head=HEAD, block=mask,
                    )
                    random_kls.append(float(kl_divergence(logits_full[qp], logits_b[qp])))
                except ValueError:
                    continue
            if random_kls:
                rows.append(dict(prompt=p_idx, policy="random_mean", q=q, k_evicted=k,
                                  n_candidates=n_cand, kl=float(np.mean(random_kls)),
                                  kl_std=float(np.std(random_kls)), n_seeds=len(random_kls)))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(layer=LAYER, head=HEAD, rows=rows), f, indent=2)

    # Print a compact summary table: mean KL per (policy, q) across prompts
    print(f"layer={LAYER} head={HEAD}, {len(PROMPTS)} prompts, {N_RANDOM_SEEDS} random seeds")
    print(f"{'policy':>16} {'q':>6} {'mean_kl':>12}")
    policies = sorted(set(r["policy"] for r in rows))
    for policy in policies:
        for q in REMOVAL_FRACTIONS:
            kls = [r["kl"] for r in rows if r["policy"] == policy and r["q"] == q and r["kl"] is not None]
            if kls:
                print(f"{policy:>16} {q:>6.1f} {np.mean(kls):>12.3e}")
    return dict(layer=LAYER, head=HEAD, rows=rows)


if __name__ == "__main__":
    run()
