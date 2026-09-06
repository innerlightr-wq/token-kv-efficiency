"""Exact-identity tests for kv_efficiency.local_algebra.

Synthetic only -- no model, no Pythia, no network access. Verifies the
EXACT ALGEBRA claim: the closed-form `delta_from_block` must equal the
directly-recomputed delete-and-renormalize output change, to float64
machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from kv_efficiency.local_algebra import (
    InputNormalizationError,
    block_mass,
    delta_from_block,
    full_output,
    local_damage,
    renormalized_output_after_deletion,
    require_normalized,
    residual,
)


def _random_head(rng, n_keys=64, d=16, concentration=2.0):
    logits = rng.standard_normal(n_keys) * concentration
    w = np.exp(logits - logits.max())
    alpha = w / w.sum()
    values = rng.standard_normal((n_keys, d))
    return alpha, values


@pytest.mark.parametrize("seed", range(8))
def test_exact_identity_matches_direct_recomputation(seed):
    """delta_from_block predicts the *change* in head output from deleting
    the block; verify it against directly zeroing + renormalizing."""
    rng = np.random.default_rng(seed)
    alpha, values = _random_head(rng)
    block = slice(10, 20)

    o = full_output(alpha, values)
    o_direct = renormalized_output_after_deletion(alpha, values, block)
    delta = delta_from_block(alpha, values, block)

    predicted = o + delta
    assert np.allclose(predicted, o_direct, atol=1e-12, rtol=0)


@pytest.mark.parametrize("seed", range(8))
def test_local_damage_equals_norm_of_delta(seed):
    rng = np.random.default_rng(seed)
    alpha, values = _random_head(rng)
    block = slice(3, 9)
    delta = delta_from_block(alpha, values, block)
    d = local_damage(alpha, values, block)
    assert np.isclose(d, np.linalg.norm(delta), atol=1e-14)


def test_require_normalized_rejects_unnormalized_weights():
    bad = np.array([0.3, 0.3, 0.3, 0.3])  # sums to 1.2
    with pytest.raises(InputNormalizationError):
        require_normalized(bad)
    # opting in repairs it
    fixed = require_normalized(bad, renormalize=True)
    assert np.isclose(fixed.sum(), 1.0)


def test_require_normalized_accepts_exact_float64():
    a = np.full(4, 0.25)
    out = require_normalized(a)
    assert np.array_equal(out, a.astype(np.float64))


# ---------------------------------------------------------------------------
# Edge cases (Phase 5 requirement: do not conceal numerical instability)
# ---------------------------------------------------------------------------


def test_edge_case_tiny_removed_mass():
    """Block carries a tiny fraction of the attention mass -- identity must
    still hold exactly, and the resulting damage should be small."""
    rng = np.random.default_rng(0)
    n = 100
    logits = np.zeros(n)
    logits[0] = 20.0  # one dominant key
    w = np.exp(logits - logits.max())
    alpha = w / w.sum()
    values = rng.standard_normal((n, 8))
    block = slice(50, 51)  # a single, essentially-zero-weight key

    m_b = float(block_mass(alpha, block))
    assert m_b < 1e-6

    o = full_output(alpha, values)
    o_direct = renormalized_output_after_deletion(alpha, values, block)
    delta = delta_from_block(alpha, values, block)
    assert np.allclose(o + delta, o_direct, atol=1e-12)
    # tiny removed mass should produce tiny (not necessarily zero) damage
    assert np.linalg.norm(delta) < 1.0


def test_edge_case_multi_token_block():
    """A block spanning many contiguous keys, not just a pair."""
    rng = np.random.default_rng(1)
    alpha, values = _random_head(rng, n_keys=128, d=32)
    block = slice(0, 64)  # half the sequence
    m_b = float(block_mass(alpha, block))
    assert 0.0 < m_b < 1.0

    o = full_output(alpha, values)
    o_direct = renormalized_output_after_deletion(alpha, values, block)
    delta = delta_from_block(alpha, values, block)
    assert np.allclose(o + delta, o_direct, atol=1e-11)


def test_edge_case_near_zero_residual():
    """Construct a block whose r_B is (near) zero: its values equal the
    global mean, so deleting it should barely perturb the output despite
    carrying real attention mass."""
    rng = np.random.default_rng(2)
    n, d = 40, 8
    alpha = np.full(n, 1.0 / n)
    values = rng.standard_normal((d,))[None, :].repeat(n, axis=0)  # identical rows
    block = slice(5, 15)

    r_b = residual(alpha, values, block)
    assert np.allclose(r_b, 0.0, atol=1e-12)

    delta = delta_from_block(alpha, values, block)
    assert np.allclose(delta, 0.0, atol=1e-12)


def test_edge_case_denominator_safely_away_from_zero():
    """Moderate m_B (e.g. 0.5): 1-m_B is safely away from zero, identity
    should hold at full float64 precision with no amplification concerns."""
    rng = np.random.default_rng(3)
    n = 50
    alpha = np.full(n, 1.0 / n)
    values = rng.standard_normal((n, 10))
    block = slice(0, 25)  # exactly half the mass
    m_b = float(block_mass(alpha, block))
    assert np.isclose(m_b, 0.5)

    o = full_output(alpha, values)
    o_direct = renormalized_output_after_deletion(alpha, values, block)
    delta = delta_from_block(alpha, values, block)
    assert np.allclose(o + delta, o_direct, atol=1e-12)


def test_edge_case_m_B_approaches_one_raises_not_silent():
    """As m_B -> 1, the eviction identity's denominator (1-m_B) -> 0 and the
    renormalized output is genuinely undefined -- delta_from_block must
    raise, not silently return a huge or NaN value. Do not conceal this."""
    n = 10
    logits = np.zeros(n)
    logits[:9] = 30.0  # block carries ~all the mass
    w = np.exp(logits - logits.max())
    alpha = w / w.sum()
    values = np.random.default_rng(4).standard_normal((n, 4))
    block = slice(0, 9)

    m_b = float(block_mass(alpha, block))
    assert m_b > 1.0 - 1e-9

    with pytest.raises(ValueError, match="undefined"):
        delta_from_block(alpha, values, block)


def test_edge_case_m_B_moderately_large_is_finite_but_amplified():
    """Just below the raise threshold: identity still holds, but the
    amplification 1/(1-m_B) is large -- this must be visible, not hidden."""
    n = 1000
    logits = np.zeros(n)
    logits[:990] = 12.0
    w = np.exp(logits - logits.max())
    alpha = w / w.sum()
    rng = np.random.default_rng(5)
    values = rng.standard_normal((n, 4))
    block = slice(0, 990)

    m_b = float(block_mass(alpha, block))
    assert 0.9 < m_b < 1.0 - 1e-9

    o = full_output(alpha, values)
    o_direct = renormalized_output_after_deletion(alpha, values, block)
    delta = delta_from_block(alpha, values, block)
    amplification = 1.0 / (1.0 - m_b)
    assert amplification > 1.0
    assert np.isfinite(delta).all()
    # amplified regime: looser but still tight tolerance, scaled by amplification
    assert np.allclose(o + delta, o_direct, atol=1e-9 * amplification)
