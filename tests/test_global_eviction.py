"""Tests for `global_eviction`: real-model, but small and fast (~1-2s),
mirroring how `attention_reconstruction`/`eviction` are ultimately
validated in experiments/. Requires the cached EleutherAI/pythia-160m
weights (already required by every other real-model script in this repo).
"""

from __future__ import annotations

import torch

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter
from kv_efficiency.eviction import global_eviction, tier_b_eviction

MODEL_ID = "EleutherAI/pythia-160m"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tok, model


def test_global_eviction_single_stratum_matches_tier_b():
    """A `global_eviction` call selecting exactly one (layer, head) must
    reproduce `tier_b_eviction` exactly -- same-layer, single-head is the
    trivial case of the "same layer is exact" claim."""
    tok, model = _load()
    adapter = GPTNeoXAdapter(model)
    ids = tok("The quick brown fox jumps over the lazy dog today.", return_tensors="pt").input_ids
    n = ids.shape[1]
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)

    layer, head = 2, 1
    hidden_in = out.hidden_states[layer][0].double()
    alpha_full = out.attentions[layer][0, head].double()
    mask = torch.zeros(n, dtype=torch.bool)
    mask[2:5] = True

    logits_tb, _ = tier_b_eviction(model, adapter, ids, alpha_full, hidden_in, layer=layer, head=head, block=mask)
    logits_ge, _skipped = global_eviction(
        model, adapter, ids,
        per_stratum_data={(layer, head): (alpha_full, hidden_in)},
        selections={(layer, head): mask},
    )
    assert torch.allclose(logits_tb, logits_ge, atol=1e-6)


def test_global_eviction_empty_selection_matches_baseline():
    tok, model = _load()
    adapter = GPTNeoXAdapter(model)
    ids = tok("The quick brown fox jumps over the lazy dog today.", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
        baseline_logits = model(ids).logits[0]

    layer, head = 2, 1
    hidden_in = out.hidden_states[layer][0].double()
    alpha_full = out.attentions[layer][0, head].double()
    empty_mask = torch.zeros(ids.shape[1], dtype=torch.bool)

    logits_ge, _skipped = global_eviction(
        model, adapter, ids,
        per_stratum_data={(layer, head): (alpha_full, hidden_in)},
        selections={(layer, head): empty_mask},
    )
    assert torch.allclose(baseline_logits, logits_ge, atol=1e-6)


def test_global_eviction_same_layer_two_heads_is_additive():
    """Same-layer, multiple heads: evicting heads 1 and 3 together must
    equal the sum of their individually-computed deltas at that one layer
    (the "exact, additive" claim for same-layer combination)."""
    tok, model = _load()
    adapter = GPTNeoXAdapter(model)
    ids = tok("The quick brown fox jumps over the lazy dog today.", return_tensors="pt").input_ids
    n = ids.shape[1]
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)

    layer = 2
    hidden_in = out.hidden_states[layer][0].double()
    mask = torch.zeros(n, dtype=torch.bool)
    mask[2:5] = True

    per_stratum = {
        (layer, 1): (out.attentions[layer][0, 1].double(), hidden_in),
        (layer, 3): (out.attentions[layer][0, 3].double(), hidden_in),
    }
    selections = {(layer, 1): mask, (layer, 3): mask}

    logits_joint, _skipped = global_eviction(model, adapter, ids, per_stratum, selections)

    # sequential reference: apply head 1's delta, then re-derive head 3's
    # delta from the ORIGINAL hidden state and add both in one hook (this
    # IS what global_eviction does internally) -- cross-check via direct
    # two-hook composition using tier_b_eviction's own primitives.
    logits_h1, delta1 = tier_b_eviction(model, adapter, ids, per_stratum[(layer, 1)][0], hidden_in, layer=layer, head=1, block=mask)
    logits_h3, delta3 = tier_b_eviction(model, adapter, ids, per_stratum[(layer, 3)][0], hidden_in, layer=layer, head=3, block=mask)

    from kv_efficiency.hooks import additive_hook
    with additive_hook(adapter._layers()[layer], delta1 + delta3, position=None):
        with torch.no_grad():
            logits_manual = model(ids).logits[0]

    assert torch.allclose(logits_joint, logits_manual, atol=1e-6)
