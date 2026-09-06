"""Phase 16 required properties of the budget allocator: conservation,
no negative allocation, bounds respected."""

from __future__ import annotations

import numpy as np
import pytest

from kv_efficiency.allocator import allocate_budget, estimate_sensitivity


def _random_sensitivities(seed, n=20):
    rng = np.random.default_rng(seed)
    return {(i // 4, i % 4): float(rng.uniform(0.001, 2.0)) for i in range(n)}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("total_budget", [0.1, 0.3, 0.5])
def test_budget_conservation(seed, total_budget):
    sens = _random_sensitivities(seed)
    alloc = allocate_budget(sens, total_budget, min_frac=0.0, max_frac=1.0)
    mean_q = np.mean(list(alloc.values()))
    assert np.isclose(mean_q, total_budget, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_no_negative_allocation(seed):
    sens = _random_sensitivities(seed)
    alloc = allocate_budget(sens, 0.3, min_frac=0.0, max_frac=1.0)
    assert all(v >= 0.0 for v in alloc.values())


@pytest.mark.parametrize("seed", range(5))
def test_allocation_within_stratum_capacity(seed):
    sens = _random_sensitivities(seed)
    min_frac, max_frac = 0.05, 0.6
    alloc = allocate_budget(sens, 0.3, min_frac=min_frac, max_frac=max_frac)
    assert all(min_frac - 1e-9 <= v <= max_frac + 1e-9 for v in alloc.values())


def test_more_sensitive_strata_get_smaller_removal_fraction():
    sens = {(0, 0): 0.1, (0, 1): 1.0, (0, 2): 5.0}
    alloc = allocate_budget(sens, 0.3, min_frac=0.0, max_frac=1.0)
    # monotone decreasing in sensitivity
    assert alloc[(0, 0)] > alloc[(0, 1)] > alloc[(0, 2)]


def test_infeasible_budget_outside_bounds_raises():
    sens = {(0, 0): 1.0, (0, 1): 2.0}
    with pytest.raises(ValueError):
        allocate_budget(sens, 0.9, min_frac=0.0, max_frac=0.5)


def test_estimate_sensitivity_uses_only_given_records():
    records = [
        dict(layer=0, head=0, d_local=1.0),
        dict(layer=0, head=0, d_local=3.0),
        dict(layer=1, head=0, d_local=10.0),
    ]
    s = estimate_sensitivity(records)
    assert s[(0, 0)] == pytest.approx(2.0)
    assert s[(1, 0)] == pytest.approx(10.0)


def test_empty_sensitivities_returns_empty():
    assert allocate_budget({}, 0.3) == {}
