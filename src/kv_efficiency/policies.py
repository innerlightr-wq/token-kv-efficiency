r"""KV-cache eviction ranking policies (Phase 9).

Each policy ranks candidate key positions within one (layer, head) from
"evict first" to "evict last." `eviction_curve.py` removes a prefix of that
ranking to reach a target removal fraction `q` and measures downstream
damage. This is deliberately single-head scope for the first efficiency
experiment -- see docs/RESEARCH_STATUS.md for why joint multi-head/layer
eviction was not attempted this round.

Policy 4 from the milestone spec (head/layer budget allocation) is NOT
implemented as a running policy here: it requires allocating a removal
budget *across* heads/layers, a different experimental unit than ranking
positions *within* one head. The coarse-sensitivity data needed to support
it is computed in `experiments/coarse_vs_fine_analysis.py`; turning that
into an actual allocator is left OPEN (see docs/RESEARCH_STATUS.md).
"""

from __future__ import annotations

import numpy as np
import torch

from .eviction import delta_from_block, require_normalized

__all__ = [
    "random_ranking",
    "attention_mass_ranking",
    "local_damage_scores",
    "local_damage_ranking",
]


def random_ranking(n_keys: int, *, protect: set[int], seed: int) -> list[int]:
    """Random eviction order over candidate positions (excludes `protect`)."""
    rng = np.random.default_rng(seed)
    candidates = [i for i in range(n_keys) if i not in protect]
    order = rng.permutation(candidates).tolist()
    return order


def attention_mass_ranking(alpha: torch.Tensor, *, protect: set[int]) -> list[int]:
    """Evict lowest-attention-weight positions first."""
    a = alpha.detach().cpu().numpy()
    idx = [i for i in range(len(a)) if i not in protect]
    idx.sort(key=lambda i: a[i])
    return idx


def local_damage_scores(
    alpha: torch.Tensor, values_dmodel: torch.Tensor, *, protect: set[int]
) -> dict[int, float]:
    """`d_B^local` for every singleton candidate position (individually
    evicted). Exposed separately from `local_damage_ranking` so callers
    (e.g. the rank-divergence analysis) can compare raw scores to `alpha`,
    not just the induced ranking."""
    a = require_normalized(alpha, renormalize=True)
    n = a.shape[-1]
    scores: dict[int, float] = {}
    for i in range(n):
        if i in protect:
            continue
        mask = torch.zeros(n, dtype=torch.bool)
        mask[i] = True
        if float(a[mask].sum()) >= 1.0 - 1e-9:
            continue
        try:
            delta = delta_from_block(a, values_dmodel, mask)
            scores[i] = float(torch.linalg.norm(delta))
        except ValueError:
            continue
    return scores


def local_damage_ranking(
    alpha: torch.Tensor, values_dmodel: torch.Tensor, *, protect: set[int]
) -> list[int]:
    """Evict positions whose individual `d_B^local` (singleton block) is
    smallest first -- the exact local score used as a per-token proxy for
    "least damaging to remove."""
    n = alpha.shape[-1]
    scores = local_damage_scores(alpha, values_dmodel, protect=protect)
    idx = sorted(scores, key=lambda i: scores[i])
    # any candidates skipped (undefined score) go last, arbitrary order
    remaining = [i for i in range(n) if i not in protect and i not in scores]
    return idx + remaining
