r"""Small forward-hook utility shared by the Tier A/B interventions.

The old project (`ablate.py::tier_a_perturbation`,
`scratch/gate_test.py::tier_b_eviction`) each hand-rolled the same
register/try/finally/remove pattern. Extracted here once, semantics
unchanged, to avoid three copies of the same footgun (a hook left attached
after an exception).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

__all__ = ["residual_add_hook", "additive_hook"]


@contextmanager
def additive_hook(module: torch.nn.Module, delta: torch.Tensor, position: "int | None" = None):
    """Register a forward hook on `module` that adds `delta` to its output
    hidden-state tensor, either at one `position` (Tier A: single query) or
    at every position (Tier B: `position=None`, `delta` is `(n, d_model)`).

    Removes the hook on exit even if the wrapped forward pass raises.
    """
    handle = None

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs.clone()
        d = delta.to(hs.dtype).to(hs.device)
        if position is None:
            hs[0] = hs[0] + d
        else:
            hs[:, position, :] = hs[:, position, :] + d
        return (hs, *output[1:]) if isinstance(output, tuple) else hs

    try:
        handle = module.register_forward_hook(hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def residual_add_hook(module: torch.nn.Module, capture: dict, key: str = "x"):
    """Register a forward *pre*-hook that captures `module`'s input tensor
    into `capture[key]`. Used to validate manual extraction against the
    real model's own computation (see `validate_local_identity.py`)."""
    handle = None

    def hook(_module, inputs):
        capture[key] = inputs[0].detach().clone()

    try:
        handle = module.register_forward_pre_hook(hook)
        yield capture
    finally:
        if handle is not None:
            handle.remove()
