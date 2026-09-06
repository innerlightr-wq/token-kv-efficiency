"""Cross-check the torch-native local algebra (eviction.py) against the
numpy reference (local_algebra.py) on identical synthetic inputs -- no
model required. Real-model Tier A/B checks live in
experiments/validate_local_identity.py (they need a network/model cache
and are reported as ENGINEERING VERIFIED there, not asserted as unit tests).
"""

from __future__ import annotations

import numpy as np
import torch

from kv_efficiency import eviction, local_algebra
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking, random_ranking


def test_torch_delta_matches_numpy_reference():
    rng = np.random.default_rng(0)
    n, d = 96, 24
    logits = rng.standard_normal(n) * 2
    w = np.exp(logits - logits.max())
    alpha_np = w / w.sum()
    values_np = rng.standard_normal((n, d))
    block = slice(10, 40)

    delta_np = local_algebra.delta_from_block(alpha_np, values_np, block)

    alpha_t = torch.tensor(alpha_np, dtype=torch.float64)
    values_t = torch.tensor(values_np, dtype=torch.float64)
    delta_t = eviction.delta_from_block(alpha_t, values_t, block).numpy()

    assert np.allclose(delta_np, delta_t, atol=1e-12)


def test_torch_require_normalized_matches_numpy_behavior():
    bad_np = np.array([0.3, 0.3, 0.3, 0.3])
    bad_t = torch.tensor(bad_np)

    try:
        local_algebra.require_normalized(bad_np)
        raised_np = False
    except local_algebra.InputNormalizationError:
        raised_np = True

    try:
        eviction.require_normalized(bad_t)
        raised_t = False
    except eviction.InputNormalizationError:
        raised_t = True

    assert raised_np and raised_t


def test_delta_from_block_accepts_boolean_mask():
    """policies.py / eviction_curve.py rely on `delta_from_block` accepting
    an arbitrary boolean mask (not just a contiguous slice) for the `block`
    argument -- confirm this works and agrees with the slice form when the
    mask happens to be contiguous."""
    rng = np.random.default_rng(1)
    n, d = 40, 8
    logits = rng.standard_normal(n)
    w = np.exp(logits - logits.max())
    alpha = torch.tensor(w / w.sum(), dtype=torch.float64)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float64)

    block = slice(5, 12)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[5:12] = True

    delta_slice = eviction.delta_from_block(alpha, values, block)
    delta_mask = eviction.delta_from_block(alpha, values, mask)
    assert torch.allclose(delta_slice, delta_mask, atol=1e-12)


def test_policy_rankings_are_permutations_excluding_protected():
    rng = np.random.default_rng(2)
    n = 20
    logits = rng.standard_normal(n)
    w = np.exp(logits - logits.max())
    alpha = torch.tensor(w / w.sum(), dtype=torch.float64)
    values = torch.tensor(rng.standard_normal((n, 6)), dtype=torch.float64)
    protect = {0}

    r_random = random_ranking(n, protect=protect, seed=0)
    r_mass = attention_mass_ranking(alpha, protect=protect)
    r_local = local_damage_ranking(alpha, values, protect=protect)

    for ranking in (r_random, r_mass, r_local):
        assert 0 not in ranking
        assert sorted(ranking) == [i for i in range(n) if i != 0]

    # attention-mass ranking must be non-decreasing in alpha
    a = alpha.numpy()
    vals = [a[i] for i in r_mass]
    assert vals == sorted(vals)
