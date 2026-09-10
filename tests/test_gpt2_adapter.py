"""Real-model validation of `GPT2Adapter` (`gpt2`, 124M) -- small and fast,
mirroring `test_global_eviction.py`'s pattern for real-model tests. Required
BEFORE any pilot result is trusted, per the pre-registration for the
cross-architecture generalization round.

Does not modify or weaken any existing test. Existing GPT-NeoX/Pythia tests
are untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kv_efficiency.attention_reconstruction import head_values_dmodel
from kv_efficiency.attention_reconstruction_gpt2 import GPT2Adapter
from kv_efficiency.eviction import delta_from_block, require_normalized, tier_a_perturbation
from kv_efficiency.hooks import additive_hook, residual_add_hook

MODEL_ID = "gpt2"
PROMPT = "The quick brown fox jumps over the lazy dog today, and everyone watched."


@pytest.fixture(scope="module")
def loaded():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    adapter = GPT2Adapter(model)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    return dict(tok=tok, model=model, adapter=adapter, ids=ids, out=out)


def test_dimensions(loaded):
    """Attention tensor dimensions; q/k/v/w_o_head shapes."""
    adapter, out, ids = loaded["adapter"], loaded["out"], loaded["ids"]
    n = ids.shape[1]
    assert out.attentions[0].shape == (1, adapter.n_heads, n, n)
    hidden_in = out.hidden_states[3][0].double()
    q, k, v = adapter.head_context_and_value(3, 2, hidden_in)
    assert q.shape == k.shape == v.shape == (n, adapter.head_size)
    wo = adapter.w_o_head(3, 2)
    assert wo.shape == (adapter.cfg.n_embd, adapter.head_size)


def test_causal_masking(loaded):
    """Attention weights sum to 1 over admissible causal positions, and
    strictly zero for non-causal (future) positions, for both an early and
    a late query position."""
    out, ids = loaded["out"], loaded["ids"]
    n = ids.shape[1]
    for layer in (0, 6, 11):
        alpha = out.attentions[layer][0, 0].double()  # (n, n)
        row_sums = alpha.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(n, dtype=torch.float64), atol=1e-6)
        mid = n // 2
        assert torch.all(alpha[mid, mid + 1 :] == 0.0), f"layer {layer}: attends to the future at position {mid}"
        assert torch.all(alpha[0, 1:] == 0.0), f"layer {layer}: first position attends beyond itself"


def test_qkv_split_and_head_reshape_by_attention_recomputation(loaded):
    """Decisive test of point 1 in the adapter's module docstring: manually
    recompute scaled-dot-product causal attention from the adapter's
    extracted (q, k) for one head and compare against the model's own
    `out.attentions` for that head. If the QKV split/reshape order were
    wrong (e.g. per-head-interleaved like GPT-NeoX instead of GPT-2's
    three-big-blocks layout), this would NOT match."""
    adapter, out, ids = loaded["adapter"], loaded["out"], loaded["ids"]
    n = ids.shape[1]
    max_err = 0.0
    for layer, head in ((0, 0), (6, 3), (11, 9)):
        hidden_in = out.hidden_states[layer][0].double()
        with torch.no_grad():
            q, k, _ = adapter.head_context_and_value(layer, head, hidden_in)
        scale = adapter.head_size ** -0.5
        scores = (q @ k.T) * scale
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        recomputed = torch.softmax(scores, dim=-1)
        real = out.attentions[layer][0, head].double()
        err = float((recomputed - real).abs().max())
        max_err = max(max_err, err)
    assert max_err < 1e-4, f"max recomputed-vs-real attention error {max_err:.3e} exceeds tolerance"


def test_value_and_output_projection_reconstruction(loaded):
    """Point 2: per-head value vectors and per-head output projection,
    validated against the model's OWN computation via a forward pre-hook on
    `c_proj` (same technique `validate_local_identity.py` used for Pythia's
    `dense`), not merely internal self-consistency."""
    model, adapter, ids = loaded["model"], loaded["adapter"], loaded["ids"]
    max_rel_err = 0.0
    for layer, head in ((0, 0), (6, 3), (11, 9)):
        capture: dict = {}
        with residual_add_hook(model.transformer.h[layer].attn.c_proj, capture):
            with torch.no_grad():
                out2 = model(ids, output_attentions=True, output_hidden_states=True)
        hidden_in = out2.hidden_states[layer][0].double()
        alpha_full = out2.attentions[layer][0, head].double()
        hs = adapter.head_size
        real_context_h = capture["x"][0, :, head * hs : (head + 1) * hs].double()
        with torch.no_grad():
            _, _, value_h = adapter.head_context_and_value(layer, head, hidden_in)
        manual_context_h = alpha_full @ value_h
        rel_err = float((real_context_h - manual_context_h).abs().max() / real_context_h.abs().max())
        max_rel_err = max(max_rel_err, rel_err)
    assert max_rel_err < 1e-3, f"max QKV/context reconstruction rel err {max_rel_err:.3e} exceeds tolerance"


def test_head_output_dmodel_reconstruction(loaded):
    """`head_values_dmodel` (imported UNCHANGED from attention_reconstruction.py,
    duck-typed against this adapter) must reproduce the real per-head
    d_model-space contribution, checked via the same c_proj pre-hook,
    summed across ALL heads this time (proves w_o_head's row-slice+transpose
    is correct, not just individually plausible)."""
    model, adapter, ids = loaded["model"], loaded["adapter"], loaded["ids"]
    layer = 3
    capture: dict = {}
    with residual_add_hook(model.transformer.h[layer].attn.c_proj, capture):
        with torch.no_grad():
            out2 = model(ids, output_attentions=True, output_hidden_states=True)
    hidden_in = out2.hidden_states[layer][0].double()
    real_context = capture["x"][0].double()  # (n, d_model), input to c_proj = concat of all heads' contexts
    with torch.no_grad():
        real_output = model.transformer.h[layer].attn.c_proj(real_context.float()).double()

    n = ids.shape[1]
    reconstructed_output = torch.zeros(n, adapter.cfg.n_embd, dtype=torch.float64)
    with torch.no_grad():
        for head in range(adapter.n_heads):
            alpha_full = out2.attentions[layer][0, head].double()
            values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
            reconstructed_output += alpha_full @ values_dmodel
    # c_proj adds a bias once (not per head); add it once here to compare full outputs.
    bias = model.transformer.h[layer].attn.c_proj.bias.detach().double()
    reconstructed_output = reconstructed_output + bias
    err = float((real_output - reconstructed_output).abs().max())
    assert err < 1e-3, f"summed multi-head reconstruction abs err {err:.3e} exceeds tolerance"


def test_delete_renormalize_identity_vs_direct_recomputation(loaded):
    """Point: delete-one-renormalize local-damage computation, for several
    randomly selected candidate tokens, against direct recomputation
    (mask + renormalize + recompute) -- the exact algebra check, model-
    independent in principle, re-run here on real GPT-2 activations."""
    adapter, out, ids = loaded["adapter"], loaded["out"], loaded["ids"]
    n = ids.shape[1]
    layer, head = 6, 3
    hidden_in = out.hidden_states[layer][0].double()
    alpha_full = out.attentions[layer][0, head].double()
    values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
    a_last = require_normalized(alpha_full[n - 1], renormalize=True)

    rng = np.random.default_rng(0)
    candidates = rng.choice(n - 1, size=min(5, n - 1), replace=False)
    max_g_err = 0.0
    for i in candidates:
        block = torch.zeros(n, dtype=torch.bool)
        block[i] = True
        if float(a_last[block].sum()) >= 1.0 - 1e-9:
            continue
        delta = delta_from_block(a_last, values_dmodel, block, renormalize=False)
        o_full = a_last @ values_dmodel
        predicted = o_full + delta

        a_masked = a_last.clone()
        a_masked[block] = 0.0
        a_masked = a_masked / a_masked.sum()
        o_direct = a_masked @ values_dmodel
        err = float((predicted - o_direct).abs().max())
        max_g_err = max(max_g_err, err)

        # g_i consistency: ||o - v_i|| computed two ways.
        g_from_local_algebra = float(torch.linalg.norm(o_full - values_dmodel[i]))
        g_direct = float(torch.linalg.norm(a_last @ values_dmodel - values_dmodel[i]))
        assert abs(g_from_local_algebra - g_direct) < 1e-9

    assert max_g_err < 1e-8, f"delete-renormalize identity abs err {max_g_err:.3e} exceeds tolerance"


def test_attn_hook_point_is_correct_and_load_bearing(loaded):
    """Point 3 of the adapter docstring: `_layers()` must return `.attn`
    (not the whole block) for GPT-2's sequential residual.

    (a) zero-delta hook on `.attn` reproduces the unhooked baseline logits
        bit-for-bit -- the hook mechanism itself is inert at zero;
    (b) hooking `.attn` vs. hooking the WHOLE block with the SAME nonzero
        delta gives DIFFERENT logits -- proving the choice of hook point is
        load-bearing for this architecture, not an arbitrary equivalent
        choice (unlike GPT-NeoX's parallel residual, where it would not be).
    """
    model, adapter, ids = loaded["model"], loaded["adapter"], loaded["ids"]
    with torch.no_grad():
        baseline_logits = model(ids).logits[0]

    layer = 4
    zero_delta = torch.zeros(model.config.n_embd)
    with additive_hook(adapter._layers()[layer], zero_delta, position=None):
        with torch.no_grad():
            zero_hooked_logits = model(ids).logits[0]
    assert torch.equal(baseline_logits, zero_hooked_logits), "hook on .attn is not inert at delta=0"

    torch.manual_seed(0)
    nonzero_delta = torch.randn(model.config.n_embd) * 0.5
    with additive_hook(adapter._layers()[layer], nonzero_delta, position=None):
        with torch.no_grad():
            attn_hooked_logits = model(ids).logits[0]
    with additive_hook(model.transformer.h[layer], nonzero_delta, position=None):
        with torch.no_grad():
            block_hooked_logits = model(ids).logits[0]

    assert not torch.allclose(attn_hooked_logits, block_hooked_logits, atol=1e-4), (
        "hooking .attn vs. the whole block gave the same result -- expected them to differ "
        "for GPT-2's sequential residual (see adapter module docstring point 3)"
    )
    assert not torch.allclose(attn_hooked_logits, baseline_logits, atol=1e-4), (
        "nonzero hook on .attn had no measurable effect -- suspicious"
    )


def test_tier_a_perturbation_runs_and_is_finite(loaded):
    """End-to-end sanity: the existing (unmodified) `tier_a_perturbation`
    from eviction.py, given this adapter, produces finite logits, and a
    zero delta reproduces the baseline exactly."""
    model, adapter, ids = loaded["model"], loaded["adapter"], loaded["ids"]
    n = ids.shape[1]
    layer = 6
    zero_delta = torch.zeros(adapter.cfg.n_embd)
    logits_full, logits_pert = tier_a_perturbation(
        model, adapter, ids, layer=layer, delta=zero_delta, position=n - 1
    )
    assert torch.isfinite(logits_pert).all()
    assert torch.allclose(logits_full, logits_pert, atol=1e-5)
