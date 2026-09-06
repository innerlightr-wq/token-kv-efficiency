r"""Phase 6: reproduce the engineering validation gate on a real model.

Re-measures (does not assume) the checks previously reported for the old
experiment: manual QKV/attention reconstruction vs. real model output,
delete-and-renormalize identity vs. direct recomputation, and determinism.
Numbers are measured fresh in THIS environment; the old ~7e-8 / ~1e-16
figures are not hard-coded as pass thresholds.

Classification: ENGINEERING VERIFIED (implementation correctness), never
upgraded to "mathematical proof" or to a claim about downstream damage
prediction.
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
from kv_efficiency.eviction import delta_from_block, tier_a_perturbation, tier_b_eviction, require_normalized
from kv_efficiency.hooks import residual_add_hook

MODEL_ID = "EleutherAI/pythia-160m"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "validate_local_identity.json"

PROMPT = (
    "The history of the Roman Empire is often divided into three stages: "
    "Rome under kings, Rome as a republic governed by elected senators, and "
    "Rome under the rule of emperors. The empire reached its greatest extent "
    "under the reign of Trajan, stretching from Britain to the Persian Gulf."
)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tok, model


def main() -> dict:
    torch.manual_seed(0)
    tok, model = load_model()
    adapter = GPTNeoXAdapter(model)
    results: dict = {"model_id": MODEL_ID, "torch_version": torch.__version__}

    ids = tok(PROMPT, return_tensors="pt").input_ids
    n = ids.shape[1]
    qp = n - 1
    results["prompt_tokens"] = n

    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    assert torch.isfinite(out.logits).all()
    results["A0_baseline_finite"] = True

    layer, head = 3, 2
    block = slice(4, 8)
    hidden_in = out.hidden_states[layer][0].double()
    alpha_full = out.attentions[layer][0, head].double()

    # A. manual QKV/attention reconstruction vs. real model's own computation
    capture: dict = {}
    with residual_add_hook(model.gpt_neox.layers[layer].attention.dense, capture):
        with torch.no_grad():
            model(ids, output_attentions=True, output_hidden_states=True)
    hs_size = adapter.head_size
    real_context_h = capture["x"][0, :, head * hs_size : (head + 1) * hs_size].double()
    _, _, value_h = adapter.head_context_and_value(layer, head, hidden_in)
    manual_context_h = alpha_full @ value_h
    rel_err_A = float((real_context_h - manual_context_h).abs().max() / real_context_h.abs().max())
    results["A_qkv_reconstruction_rel_err"] = rel_err_A

    # B. delete-and-renormalize identity vs. direct recomputation (real activations)
    values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)

    a_last = require_normalized(alpha_full[qp], renormalize=True)
    delta = delta_from_block(a_last, values_dmodel, block, renormalize=False)
    o_full = a_last @ values_dmodel.to(torch.float64)

    a_masked = a_last.clone()
    a_masked[block] = 0.0
    a_masked = a_masked / a_masked.sum(dim=-1, keepdim=True)
    o_direct = a_masked @ values_dmodel.to(torch.float64)

    predicted = o_full + delta
    err_B = float((predicted - o_direct).abs().max())
    results["B_delete_renormalize_identity_abs_err"] = err_B

    # C. determinism
    with torch.no_grad():
        logits_again = model(ids).logits[0]
    with torch.no_grad():
        logits_full_all = model(ids).logits[0]
    det_err = float((logits_full_all - logits_again).abs().max())
    results["C_determinism_abs_err"] = det_err

    # D. Tier A isolation + Tier B propagation, as additional engineering checks
    _, logits_a_last = tier_a_perturbation(model, adapter, ids, layer=layer, delta=delta.float(), position=qp)
    kl_a = float(kl_divergence(logits_full_all[qp], logits_a_last[0]))
    results["D_tier_a_kl_finite"] = bool(np.isfinite(kl_a))
    results["D_tier_a_kl"] = kl_a

    logits_b_all, _ = tier_b_eviction(model, adapter, ids, alpha_full, hidden_in, layer=layer, head=head, block=block)
    kl_b = float(kl_divergence(logits_full_all[qp], logits_b_all[qp]))
    results["D_tier_b_kl_finite"] = bool(np.isfinite(kl_b))
    results["D_tier_b_kl"] = kl_b

    results["verdict"] = "ENGINEERING VERIFIED" if (
        rel_err_A < 1e-3 and err_B < 1e-9 and det_err < 1e-6
    ) else "FAILED -- see individual checks"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
