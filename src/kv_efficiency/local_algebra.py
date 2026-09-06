r"""Exact local delete-and-renormalize algebra for one attention head.

Model-independent: everything here operates on plain numpy arrays (a
softmax-normalized attention-weight vector `alpha` over keys, and the
per-key vectors `values` already written into the residual stream, i.e.
`values[i] = W_O^h @ v_i` in the usual attention notation). No model, no
torch, no hooks.

**Provenance.** These definitions are recovered verbatim from the prior
experiment at `~/Desktop/Token Research/files (29)/README.md` (lines 9-11)
and `ablate.py::delta_from_block` (lines 127-145) -- not reconstructed from
memory. Exact notation preserved:

    o   = sum_i alpha_i v_i                    (full head output)
    m_B = sum_{i in B} alpha_i                 (attention mass on block B)
    o_B = sum_{i in B} alpha_i v_i             (block's raw contribution)
    r_B = o_B - m_B * o                        (residual)
    Delta_B = -r_B / (1 - m_B)                 (exact perturbation from evicting B)
    d_B^local = ||Delta_B|| = ||r_B|| / (1 - m_B)

**Epistemic status: EXACT ALGEBRA.** `Delta_B` is the exact, closed-form
change to the head's output when block `B` is deleted from the attention
and the remaining weights are renormalized -- an algebraic identity, proved
by direct substitution (see `tests/test_local_algebra.py`), not an
approximation and not (by itself) a claim about downstream network damage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "InputNormalizationError",
    "require_normalized",
    "full_output",
    "block_mass",
    "block_output",
    "residual",
    "delta_from_block",
    "local_damage",
    "renormalized_output_after_deletion",
    "LocalDamageResult",
]


class InputNormalizationError(ValueError):
    """`alpha` does not sum to 1 at float64 precision.

    Mirrors `ablate.py::InputNormalizationError`: a float32/bf16 softmax
    sums to 1 only approximately, and upcasting to float64 does not fix
    that -- it preserves the error exactly. Raised rather than silently
    repaired, since renormalizing changes the quantity being measured.
    """


def require_normalized(
    alpha: np.ndarray, *, atol: float = 1e-12, renormalize: bool = False
) -> np.ndarray:
    """Return float64 weights that sum to 1 along the last axis, or raise."""
    a = np.asarray(alpha, dtype=np.float64)
    total = a.sum(axis=-1, keepdims=True)
    dev = float(np.abs(total - 1.0).max())
    if dev <= atol:
        return a
    if renormalize:
        return a / total
    raise InputNormalizationError(
        f"alpha deviates from summing to 1 by {dev:.3e} > atol={atol:.1e}. "
        "Extract in float64, or pass renormalize=True and accept that the "
        "measured perturbation is then the one for the renormalized distribution."
    )


def full_output(alpha: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`o = sum_i alpha_i v_i`."""
    a = np.asarray(alpha, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    return a @ v


def block_mass(alpha: np.ndarray, block: slice) -> np.ndarray:
    """`m_B = sum_{i in B} alpha_i`."""
    a = np.asarray(alpha, dtype=np.float64)
    return a[..., block].sum(axis=-1)


def block_output(alpha: np.ndarray, values: np.ndarray, block: slice) -> np.ndarray:
    """`o_B = sum_{i in B} alpha_i v_i`."""
    a = np.asarray(alpha, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    return a[..., block] @ v[block]


def residual(alpha: np.ndarray, values: np.ndarray, block: slice) -> np.ndarray:
    """`r_B = o_B - m_B * o`."""
    o = full_output(alpha, values)
    m = block_mass(alpha, block)
    o_b = block_output(alpha, values, block)
    return o_b - np.expand_dims(m, -1) * o


def delta_from_block(
    alpha: np.ndarray,
    values: np.ndarray,
    block: slice,
    *,
    renormalize: bool = False,
    mass_epsilon: float = 1e-9,
) -> np.ndarray:
    r"""Closed-form `Delta_B = (m_B * o - o_B) / (1 - m_B)`.

    Equivalently `-r_B / (1 - m_B)`. Raises if the block carries essentially
    all of the attention mass (`m_B >= 1 - mass_epsilon`), matching
    `ablate.py::delta_from_block`'s `1e-9` threshold exactly -- eviction
    there makes the renormalized output undefined, not merely ill-conditioned.
    """
    a = require_normalized(alpha, renormalize=renormalize)
    v = np.asarray(values, dtype=np.float64)
    o = a @ v
    m = a[..., block].sum(axis=-1)
    o_b = a[..., block] @ v[block]
    if np.any(m >= 1.0 - mass_epsilon):
        raise ValueError(
            f"Block carries attention mass {float(np.max(m)):.12f}; evicting it "
            "makes the renormalized output undefined."
        )
    return (np.expand_dims(m, -1) * o - o_b) / np.expand_dims(1.0 - m, -1)


def local_damage(alpha: np.ndarray, values: np.ndarray, block: slice, **kwargs) -> np.ndarray:
    """`d_B^local = ||Delta_B||`, the exact local damage score."""
    delta = delta_from_block(alpha, values, block, **kwargs)
    return np.linalg.norm(delta, axis=-1)


def renormalized_output_after_deletion(
    alpha: np.ndarray, values: np.ndarray, block: slice, *, renormalize: bool = False
) -> np.ndarray:
    """Direct (non-algebraic) recomputation: zero the block's weights,
    renormalize, and recompute the head output. Used only to verify
    `delta_from_block` against direct recomputation -- see
    `ablate.py::apply_mask_mode` / `apply_residual_add_mode`, the
    "mask" vs. "residual_add" pair this mirrors.
    """
    a = require_normalized(alpha, renormalize=renormalize).copy()
    a[..., block] = 0.0
    denom = a.sum(axis=-1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("Masking removed all attention mass.")
    v = np.asarray(values, dtype=np.float64)
    return (a / denom) @ v


@dataclass(frozen=True)
class LocalDamageResult:
    """Bundle of the local-algebra quantities for one (alpha, values, block)."""

    m_B: float
    r_B: np.ndarray
    delta_B: np.ndarray
    d_B_local: float

    @staticmethod
    def compute(alpha: np.ndarray, values: np.ndarray, block: slice) -> "LocalDamageResult":
        m = float(block_mass(alpha, block))
        r = residual(alpha, values, block)
        delta = delta_from_block(alpha, values, block)
        return LocalDamageResult(
            m_B=m, r_B=r, delta_B=delta, d_B_local=float(np.linalg.norm(delta))
        )
