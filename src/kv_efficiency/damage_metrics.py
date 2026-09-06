r"""Downstream-damage metrics.

**Provenance.** `kl_divergence` is ported verbatim (semantics unchanged,
only reformatted) from
`~/Desktop/Token Research/files (29)/ablate.py::kl_divergence`.

**Epistemic status.** These are measurement functions only. Computing a KL
divergence, a logit L2 distance, or a hidden-state distance is EXACT
COMPUTATION given the two tensors being compared -- it says nothing by
itself about whether any particular *predictor* (e.g. `d_B^local`) explains
or ranks that damage well. That is an empirical question, addressed
separately in `experiments/` and `docs/RESEARCH_STATUS.md`.
"""

from __future__ import annotations

import torch

__all__ = ["kl_divergence", "logit_l2", "hidden_state_l2"]


def kl_divergence(
    logits_full: torch.Tensor,
    logits_compressed: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    r"""KL(p_full || p_compressed) in nats, computed in float64.

    Direction matters: the full (uncompressed) model is the reference
    distribution, so probability mass the full model places where the
    compressed model places none is penalized heavily -- the correct
    asymmetry for a compression study. Ported from `ablate.py`.
    """
    lp_full = torch.log_softmax(logits_full.to(torch.float64), dim=dim)
    lp_comp = torch.log_softmax(logits_compressed.to(torch.float64), dim=dim)
    p_full = lp_full.exp()
    return (p_full * (lp_full - lp_comp)).sum(dim=dim)


def logit_l2(logits_full: torch.Tensor, logits_compressed: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Euclidean distance between raw logit vectors, float64. A cheaper,
    KL-independent damage proxy requested for the efficiency-curve study."""
    a = logits_full.to(torch.float64)
    b = logits_compressed.to(torch.float64)
    return torch.linalg.norm(a - b, dim=dim)


def hidden_state_l2(h_full: torch.Tensor, h_compressed: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Euclidean distance between hidden-state vectors at some layer,
    float64. Useful when comparing damage before it reaches the unembedding."""
    a = h_full.to(torch.float64)
    b = h_compressed.to(torch.float64)
    return torch.linalg.norm(a - b, dim=dim)
