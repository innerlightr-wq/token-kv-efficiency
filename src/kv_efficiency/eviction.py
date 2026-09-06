r"""Torch-native local algebra plus the Tier A / Tier B interventions.

`local_algebra.py` is deliberately numpy-only and model-independent. Real
model activations arrive as torch tensors, so this module re-implements the
same exact identity natively in torch (mirroring the old project's own
split between a numpy reference and a torch production path, cross-checked
in its `test_torch_and_reference.py`). The formula is identical; only the
tensor library differs. See `tests/test_eviction.py` for a cross-check
against `local_algebra.py` on the same inputs.

**Tier A / Tier B, provenance.** Definitions recovered verbatim from
`ablate.py`'s module docstring:

    Tier A: perturb the head output at a single query position (the last),
            then rerun. Not eviction -- isolates the propagation question.
    Tier B: true eviction -- remove the block from the head's KV for every
            query position in an evaluation window and rerun.

Tier A is implemented exactly as in `ablate.py::tier_a_perturbation`
(reruns the whole forward pass rather than only layers l..L -- "correct but
wasteful," per the original docstring; the suffix-only optimization was
never built and is not built here either). Tier B is implemented exactly as
in `scratch/gate_test.py::tier_b_eviction`, derived from GPT-NeoX's
parallel-residual structure (see that file's module docstring for the
derivation) -- this is architecture-specific, not a generic routine.

**`global_eviction`, Milestone 2 addition, exactness scope stated
precisely.** Multiple (layer, head) evictions applied in one forward pass:

- *Same layer, multiple heads*: EXACT. GPT-NeoX's parallel residual makes
  each head's contribution to that layer's output additive and
  independent (see `tier_b_eviction`'s docstring derivation), so summing
  per-head deltas at one layer and injecting the sum is the same identity,
  not an approximation.
- *Different layers*: an APPROXIMATION, stated plainly. Each layer's delta
  is computed from that layer's BASELINE (pre-eviction) attention/hidden
  state, then all layers' hooks fire together in one pass. A later layer's
  hook does not account for the fact that an earlier layer's eviction has
  already changed the hidden states feeding into it -- re-deriving that
  interaction exactly was out of scope for this milestone (see
  docs/RESEARCH_STATUS.md). This mirrors how a real deployed eviction
  policy would actually behave (decide per-layer independently, apply
  jointly), so it is a realistic simplification, not an arbitrary one --
  but it is not the exact identity `delta_from_block` is elsewhere.
"""

from __future__ import annotations

from contextlib import ExitStack

import torch

from .attention_reconstruction import GPTNeoXAdapter
from .hooks import additive_hook

__all__ = [
    "InputNormalizationError",
    "require_normalized",
    "delta_from_block",
    "tier_a_perturbation",
    "tier_b_eviction",
    "global_eviction",
]


class InputNormalizationError(ValueError):
    """Torch counterpart of `local_algebra.InputNormalizationError`."""


def require_normalized(alpha: torch.Tensor, *, atol: float = 1e-12, renormalize: bool = False) -> torch.Tensor:
    a = alpha.to(torch.float64)
    total = a.sum(dim=-1, keepdim=True)
    dev = float((total - 1.0).abs().max())
    if dev <= atol:
        return a
    if renormalize:
        return a / total
    raise InputNormalizationError(
        f"alpha deviates from summing to 1 by {dev:.3e} > atol={atol:.1e} (dtype {alpha.dtype})."
    )


def delta_from_block(
    alpha: torch.Tensor, values: torch.Tensor, block: slice, *, renormalize: bool = False
) -> torch.Tensor:
    r"""Torch-native `Delta_B = (m_B o - o_B)/(1 - m_B)`. See `local_algebra.py`
    for the exact same identity in numpy, and its docstring for provenance."""
    a = require_normalized(alpha, renormalize=renormalize)
    v = values.to(torch.float64)
    o = a @ v
    m = a[..., block].sum(dim=-1)
    o_b = a[..., block] @ v[block]
    if torch.any(m >= 1.0 - 1e-9):
        raise ValueError(
            f"Block carries attention mass {float(m.max()):.12f}; evicting it "
            "makes the renormalized output undefined."
        )
    return (m.unsqueeze(-1) * o - o_b) / (1.0 - m).unsqueeze(-1)


@torch.no_grad()
def tier_a_perturbation(
    model,
    adapter: GPTNeoXAdapter,
    input_ids: torch.Tensor,
    *,
    layer: int,
    delta: torch.Tensor,
    position: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add `delta` to the residual stream at one position and one layer.
    Returns `(logits_full, logits_perturbed)` for the target position."""
    logits_full = model(input_ids).logits[:, position, :]
    with additive_hook(adapter._layers()[layer], delta, position=position):
        logits_pert = model(input_ids).logits[:, position, :]
    return logits_full, logits_pert


@torch.no_grad()
def tier_b_eviction(
    model,
    adapter: GPTNeoXAdapter,
    input_ids: torch.Tensor,
    alpha_full: torch.Tensor,
    hidden_in: torch.Tensor,
    *,
    layer: int,
    head: int,
    block: slice,
) -> tuple[torch.Tensor, torch.Tensor]:
    """True multi-query eviction. Returns `(logits, delta_dmodel)` for the
    full sequence. See the module docstring for the exactness derivation."""
    _, _, value_h = adapter.head_context_and_value(layer, head, hidden_in)
    wo_h = adapter.w_o_head(layer, head)

    a = require_normalized(alpha_full, renormalize=True)
    delta_context = delta_from_block(a, value_h, block, renormalize=False)
    delta_dmodel = delta_context @ wo_h.T

    with additive_hook(adapter._layers()[layer], delta_dmodel, position=None):
        logits = model(input_ids).logits[0]
    return logits, delta_dmodel


@torch.no_grad()
def global_eviction(
    model,
    adapter: GPTNeoXAdapter,
    input_ids: torch.Tensor,
    per_stratum_data: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    selections: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Evict from multiple `(layer, head)` strata in one forward pass.

    `per_stratum_data[(layer, head)] = (alpha_full, hidden_in)` -- the
    BASELINE (pre-eviction) attention weights and layer input for that
    stratum. `selections[(layer, head)]` is a boolean mask of positions to
    evict there (all-`False` / absent strata are left untouched). See the
    module docstring for exactly which combinations are exact (same-layer,
    multi-head) versus an approximation (cross-layer).

    Returns `(logits, skipped)`: `skipped` lists any `(layer, head)` whose
    requested eviction was **not applied** because it would have consumed
    essentially all of that query's attention mass (`m_B -> 1`, the exact
    identity's singularity -- see `local_algebra.py`'s `m_B -> 1` edge-case
    test). This is surfaced, not silently absorbed: a caller that ignores
    `skipped` has silently gotten less eviction than requested.
    """
    deltas_by_layer: dict[int, torch.Tensor] = {}
    skipped: list[tuple[int, int]] = []
    for (layer, head), mask in selections.items():
        if not torch.any(mask):
            continue
        alpha_full, hidden_in = per_stratum_data[(layer, head)]
        _, _, value_h = adapter.head_context_and_value(layer, head, hidden_in)
        wo_h = adapter.w_o_head(layer, head)
        a = require_normalized(alpha_full, renormalize=True)
        try:
            delta_context = delta_from_block(a, value_h, mask, renormalize=False)
        except ValueError:
            skipped.append((layer, head))
            continue
        delta_dmodel = delta_context @ wo_h.T
        deltas_by_layer[layer] = deltas_by_layer.get(layer, 0) + delta_dmodel

    with ExitStack() as stack:
        for layer, delta in deltas_by_layer.items():
            stack.enter_context(additive_hook(adapter._layers()[layer], delta, position=None))
        logits = model(input_ids).logits[0]
    return logits, skipped
