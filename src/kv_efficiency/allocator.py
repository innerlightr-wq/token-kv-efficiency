r"""Head/layer budget allocator (Phase 7, EXPLORATORY).

Motivated by Milestone 1's exploratory coarse finding (between-head/layer
Spearman(mean d_B^local, mean Tier-B KL) ~= 0.77): more locally-sensitive
strata plausibly deserve a larger KV retention budget. This module does
NOT assume that helps in practice -- `global_budget_curve.py` tests it
against a uniform baseline.

Design, deliberately simple (no optimization solver): each stratum's
removal fraction is inversely proportional to its calibration-estimated
sensitivity, clipped to `[min_frac, max_frac]`, then water-filled so the
budget-weighted mean removal fraction exactly equals the requested global
target -- i.e. the total number of removed entries (assuming equal
candidate counts per stratum) matches the global budget exactly, not just
approximately.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_sensitivity", "allocate_budget"]


def estimate_sensitivity(calibration_records: list[dict]) -> dict[tuple[int, int], float]:
    """Mean `d_local` per (layer, head) stratum, from CALIBRATION records only."""
    by_stratum: dict[tuple[int, int], list[float]] = {}
    for r in calibration_records:
        key = (r["layer"], r["head"])
        by_stratum.setdefault(key, []).append(r["d_local"])
    return {k: float(np.mean(v)) for k, v in by_stratum.items()}


def allocate_budget(
    sensitivities: dict[tuple[int, int], float],
    total_budget: float,
    *,
    min_frac: float = 0.0,
    max_frac: float = 1.0,
    max_iters: int = 200,
) -> dict[tuple[int, int], float]:
    """Per-stratum removal fractions, water-filled to conserve the global
    budget exactly (mean removal fraction == `total_budget`), subject to
    `min_frac <= q_h <= max_frac`. More sensitive strata get a smaller
    removal fraction (monotone decreasing in sensitivity).
    """
    if not (0.0 <= min_frac <= max_frac <= 1.0):
        raise ValueError(f"require 0 <= min_frac <= max_frac <= 1, got {min_frac}, {max_frac}")
    if not (min_frac <= total_budget <= max_frac):
        raise ValueError(
            f"total_budget={total_budget} must lie within [min_frac, max_frac]="
            f"[{min_frac}, {max_frac}] for exact conservation to be feasible"
        )

    keys = list(sensitivities)
    n = len(keys)
    if n == 0:
        return {}
    s = np.array([sensitivities[k] for k in keys], dtype=np.float64)

    # Monotone decreasing weight in sensitivity, mean 1.
    inv = 1.0 / (s - s.min() + 1.0)
    inv = inv / inv.mean()
    q = np.clip(total_budget * inv, min_frac, max_frac)

    target_sum = total_budget * n
    for _ in range(max_iters):
        diff = target_sum - q.sum()
        if abs(diff) < 1e-9:
            break
        if diff > 0:
            adjustable = q < max_frac - 1e-12
        else:
            adjustable = q > min_frac + 1e-12
        if not adjustable.any():
            break
        q[adjustable] += diff / adjustable.sum()
        q = np.clip(q, min_frac, max_frac)

    return {k: float(v) for k, v in zip(keys, q)}
