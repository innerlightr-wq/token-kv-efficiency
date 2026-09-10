r"""Model-specific attention reconstruction for GPT-2 (`GPT2LMHeadModel`).

**Does not modify, import from, or change the semantics of
`attention_reconstruction.py::GPTNeoXAdapter`** -- a fully separate, additive
adapter for a genuinely different architecture, written to conform to the
SAME external contract (`head_context_and_value`, `w_o_head`, `_layers()`)
so that `head_values_dmodel` (imported unchanged from `attention_reconstruction.py`)
and every function in `eviction.py` (`delta_from_block`, `tier_a_perturbation`,
`tier_b_eviction`, `tier_b_eviction_scoped`, `global_eviction`) work on this
adapter with **zero code changes**, because those functions are already
duck-typed against exactly this contract.

Two architectural differences from GPT-NeoX/Pythia that this adapter must
handle explicitly, both verified this session against the installed
`transformers` source (not assumed from memory):

1. **`c_attn` is a `Conv1D`, not `nn.Linear`, and its QKV layout differs.**
   `transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward` does:

       query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

   i.e. the `(n, 3*n_embd)` output is split into three big contiguous
   `(n, n_embd)` blocks `[Q_full | K_full | V_full]` -- NOT per-head-
   interleaved the way GPT-NeoX's fused `query_key_value` linear is (where
   `GPTNeoXAdapter` reshapes to `(n, n_heads, 3*head_size)` and slices each
   head's own `[q|k|v]` triple). Each big block is only THEN reshaped to
   `(n, n_heads, head_size)`. Reproduced exactly below, not approximated.

2. **`Conv1D.forward` computes `x @ weight + bias`, not `x @ weight.T`.**
   (`transformers.pytorch_utils.Conv1D.forward` uses `torch.addmm(bias, x,
   weight)`, confirmed by reading its source this session.) `c_attn` is
   called as a module (`layer_module.attn.c_attn(xn)`), which handles this
   correctly with no manual matrix reconstruction -- per the size-fitting
   constraint on this adapter, the QKV projection is never hand-rolled.
   `c_proj` (the per-head output projection) IS sliced manually, the same
   way `GPTNeoXAdapter.w_o_head` slices `dense.weight` -- but `c_proj`'s
   weight has shape `(in=d_model, out=d_model)` (Conv1D convention), so the
   per-head slice is along the INPUT (row) axis, `weight[head*hs:(head+1)*hs, :]`,
   the opposite axis from `GPTNeoXAdapter`'s column-slice of an `nn.Linear`
   weight. To satisfy the shared contract `w_o_head(...)` returns
   `(d_model, head_size)` such that `value_h @ w_o_head.T` reconstructs that
   head's d_model-space contribution (matching `GPTNeoXAdapter` exactly),
   this adapter transposes the row-slice once, deliberately, to conform to
   that contract -- not as an ad-hoc fix to make shapes fit, but so the
   SAME downstream code (`head_values_dmodel`, `eviction.py`) works
   unchanged. This is proved correct, not merely argued, by
   `tests/test_gpt2_adapter.py`'s real-model reconstruction check (compares
   against the model's own `c_proj` computation via a forward pre-hook,
   the same technique `validate_local_identity.py` used for Pythia).

3. **`_layers()` returns the ATTENTION submodule, not the whole transformer
   block -- a genuine architectural difference from `GPTNeoXAdapter`, not a
   cosmetic one.** `eviction.py`'s own module docstring states plainly that
   Tier B eviction "is derived from GPT-NeoX's parallel-residual structure...
   this is architecture-specific, not a generic routine": GPT-NeoX computes
   `h_out = h_in + attn(ln(h_in)) + mlp(ln(h_in))` (attn and mlp both from
   the SAME `h_in`, independently), so adding a correction anywhere between
   attn's own computation and the block's final output is equivalent --
   hooking the whole block's output is valid there only because of that
   parallel structure. GPT-2 is a SEQUENTIAL-residual architecture instead:
   `h_mid = h_in + attn(ln_1(h_in))`; `h_out = h_mid + mlp(ln_2(h_mid))` --
   the MLP consumes attn's own (corrected) output. Hooking the whole block
   (`transformer.h[i]`) would add the correction AFTER the MLP has already
   run on the WRONG (uncorrected) `h_mid`, silently breaking the
   intervention. `_layers()` therefore returns each block's `.attn`
   submodule directly, so `additive_hook` (already fully generic, unchanged
   from `hooks.py`) adds the correction to `attn`'s own output -- exactly
   where GPT-2's `h_mid = h_in + attn_output` residual-add would pick it up,
   correctly propagating into `ln_2`/`mlp` and everything downstream.
   Verified in `tests/test_gpt2_adapter.py`: a zero-delta hook on `.attn`
   reproduces the unhooked baseline logits bit-for-bit (hook is inert at
   zero), and hooking `.attn` vs. the whole block gives DIFFERENT results
   for a nonzero delta (proving the choice is load-bearing, not arbitrary).

**Epistemic status: ENGINEERING** (extraction plumbing), exactly matching
`GPTNeoXAdapter`'s own stated status -- correctness is an empirical question
answered by `tests/test_gpt2_adapter.py`, not assumed here.
"""

from __future__ import annotations

import torch

__all__ = ["GPT2Adapter"]


class GPT2Adapter:
    """Just enough surface, matching `GPTNeoXAdapter`'s contract exactly, to
    hook Tier A/B interventions and extract per-head (context, value) pairs
    from a live GPT-2 forward pass. Scoped to `GPT2LMHeadModel` only."""

    def __init__(self, model):
        self.model = model
        self.cfg = model.config
        self.n_heads = self.cfg.n_head
        self.head_size = self.cfg.n_embd // self.n_heads

    def _layers(self):
        """Returns the ATTENTION submodule of each block (see module
        docstring point 3) -- not `list(self.model.transformer.h)`."""
        return [block.attn for block in self.model.transformer.h]

    def head_context_and_value(self, layer: int, head: int, hidden: torch.Tensor):
        """Return `(q, k, value_h)`, each `(n, head_size)`, float64.

        `hidden` is the residual-stream input to `layer`
        (`out.hidden_states[layer]`), matching `GPTNeoXAdapter`'s contract
        exactly. Attention weights themselves come from
        `output_attentions=True`, not recomputed here (identical policy to
        `GPTNeoXAdapter`).
        """
        block = self.model.transformer.h[layer]
        hs = self.head_size
        embed_dim = self.n_heads * hs
        xn = block.ln_1(hidden.to(block.ln_1.weight.dtype))
        qkv = block.attn.c_attn(xn)  # (n, 3*embed_dim) -- module call, Conv1D orientation handled internally
        n = qkv.shape[0]
        q_full, k_full, v_full = qkv.split(embed_dim, dim=-1)  # GPT-2-specific: 3 big blocks, not per-head-interleaved
        q = q_full.view(n, self.n_heads, hs)[:, head, :]
        k = k_full.view(n, self.n_heads, hs)[:, head, :]
        v = v_full.view(n, self.n_heads, hs)[:, head, :]
        return q.double(), k.double(), v.double()

    def w_o_head(self, layer: int, head: int) -> torch.Tensor:
        """`(d_model, head_size)` -- same shape/contract as
        `GPTNeoXAdapter.w_o_head`, so `value_h @ w_o_head.T` reconstructs
        this head's d_model-space contribution either way (see module
        docstring point 2 for why the transpose is here, deliberately)."""
        block = self.model.transformer.h[layer]
        hs = self.head_size
        # Conv1D c_proj.weight: shape (in=d_model, out=d_model), forward computes x @ weight (no .T).
        # Row-slice the INPUT (per-head) axis -- opposite of GPTNeoXAdapter's nn.Linear column-slice --
        # then transpose once to conform to the shared (d_model, head_size) contract.
        row_slice = block.attn.c_proj.weight[head * hs : (head + 1) * hs, :].double()
        return row_slice.T
