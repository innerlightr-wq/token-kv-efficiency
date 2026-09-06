r"""Model-specific attention reconstruction for GPT-NeoX (e.g. Pythia).

**Provenance.** Ported near-verbatim from
`~/Desktop/Token Research/files (29)/scratch/gate_test.py::GPTNeoXAdapter`
and `head_values_dmodel`, which were themselves new code written to fill a
gap: `kv_damage_atlas.adapters.base.resolve_adapter` referenced by the old
project's test suite does not exist anywhere in the old file set. See
`docs/MIGRATION_NOTES.md`.

**Epistemic status: ENGINEERING** (extraction plumbing). Whether the
extracted quantities are *correct* -- i.e. whether `head_context_and_value`
really reproduces what the live model computed -- is an empirical question
answered by `experiments/validate_local_identity.py`, not assumed here.
"""

from __future__ import annotations

import torch

__all__ = ["GPTNeoXAdapter", "head_values_dmodel"]


class GPTNeoXAdapter:
    """Just enough surface to hook Tier A/B interventions and extract
    per-head (context, value) pairs from a live GPT-NeoX forward pass.

    Not a general multi-architecture adapter (no
    `kv_damage_atlas.adapters.base` equivalent exists yet) -- scoped to
    GPT-NeoX (Pythia) only, matching what was actually built and validated.
    """

    def __init__(self, model):
        self.model = model
        self.cfg = model.config
        self.n_heads = self.cfg.num_attention_heads
        self.head_size = self.cfg.hidden_size // self.n_heads

    def _layers(self):
        return list(self.model.gpt_neox.layers)

    def head_context_and_value(self, layer: int, head: int, hidden: torch.Tensor):
        """Return `(q, k, value_h)`, each `(n, head_size)`, float64.

        `hidden` is the residual-stream input to `layer`
        (`out.hidden_states[layer]`). `value_h(k)` is the per-key value
        vector (pre-W_O, not RoPE'd -- V never receives rotary embeddings
        in GPT-NeoX). Attention weights themselves come from
        `output_attentions=True`, not recomputed here.
        """
        layer_module = self.model.gpt_neox.layers[layer]
        hs = self.head_size
        xn = layer_module.input_layernorm(hidden.to(layer_module.input_layernorm.weight.dtype))
        qkv = layer_module.attention.query_key_value(xn)  # (n, 3*hidden)
        n = qkv.shape[0]
        qkv = qkv.view(n, self.n_heads, 3 * hs)
        q, k, v = qkv[:, head, :hs], qkv[:, head, hs : 2 * hs], qkv[:, head, 2 * hs :]
        return q.double(), k.double(), v.double()

    def w_o_head(self, layer: int, head: int) -> torch.Tensor:
        """`(d_model, head_size)` slice of `dense.weight` for one head."""
        layer_module = self.model.gpt_neox.layers[layer]
        hs = self.head_size
        return layer_module.attention.dense.weight[:, head * hs : (head + 1) * hs].double()


def head_values_dmodel(adapter: GPTNeoXAdapter, layer: int, head: int, hidden_in: torch.Tensor) -> torch.Tensor:
    """Per-key value vectors already through `W_O_h` -- the `values`
    convention `local_algebra.py` expects (already in `d_model` space)."""
    _, _, v = adapter.head_context_and_value(layer, head, hidden_in)
    wo_h = adapter.w_o_head(layer, head)
    return v @ wo_h.T
