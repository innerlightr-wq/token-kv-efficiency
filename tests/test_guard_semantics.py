r"""Regression tests pinning down what the `m_B -> 1` guard actually detects.

These were written during the literature/reproduction audit (see
`docs/SINGULARITY_FINDINGS.md`). They document CURRENT behaviour so that any
future change to the guard is a deliberate decision rather than an accident.
They are not assertions that the current behaviour is the desired behaviour;
where the audit recommends a change, the test says so in its docstring.

The central control is `test_guard_silent_under_uniform_causal_attention`:
under perfectly uniform causal attention the guard must NOT fire, at any
eviction fraction. If it ever does, the guard has become confounded by
causal-window size and every aggregate statistic built on it is suspect.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kv_efficiency.eviction import delta_from_block, require_normalized

N = 64
EPS = 1e-9


def uniform_causal(n: int = N) -> np.ndarray:
    """`A[q, k] = 1/(q+1)` for `k <= q` -- maximal-entropy causal attention."""
    tri = np.tril(np.ones((n, n)))
    return tri / tri.sum(1, keepdims=True)


def one_hot_causal(n: int = N, off: float = 1e-17) -> np.ndarray:
    """Each row puts `1 - off` on its own diagonal key -- near-deterministic."""
    a = np.zeros((n, n))
    for q in range(n):
        if q:
            a[q, : q + 1] = off / q
        a[q, q] = 1.0 - (off if q else 0.0)
    return a / a.sum(1, keepdims=True)


def sink_causal(n: int = N, sink: float = 0.9) -> np.ndarray:
    """Every row `q >= 1` puts `sink` on key 0 -- an early-KEY attention sink."""
    a = np.zeros((n, n))
    a[0, 0] = 1.0
    for q in range(1, n):
        a[q, 0] = sink
        a[q, 1 : q + 1] = (1.0 - sink) / q
    return a / a.sum(1, keepdims=True)


def mask_lowest_at_last_row(a: np.ndarray, frac: float, protect=(0, N - 1)) -> torch.Tensor:
    """The repository's own selection rule: rank by attention at the LAST query
    row, evict the lowest-weighted `frac` of unprotected positions."""
    n = a.shape[0]
    cand = [i for i in range(n) if i not in protect]
    order = sorted(cand, key=lambda i: a[n - 1][i])
    k = round(frac * len(cand))
    m = torch.zeros(n, dtype=torch.bool)
    for i in order[:k]:
        m[i] = True
    return m


def row_masses(a: np.ndarray, mask: torch.Tensor) -> np.ndarray:
    t = require_normalized(torch.tensor(a), renormalize=True)
    return t[:, mask].sum(dim=-1).numpy()


def guard_fires(a: np.ndarray, mask: torch.Tensor) -> bool:
    values = torch.eye(a.shape[0], dtype=torch.float64)
    try:
        delta_from_block(torch.tensor(a), values, mask)
        return False
    except ValueError:
        return True


# --------------------------------------------------------------------------
# The control: uniform causal attention must never look pathological.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.1, 0.3, 0.5, 0.9, 0.95])
def test_guard_silent_under_uniform_causal_attention(frac):
    """Uniform causal attention is the maximum-entropy case: nothing about it
    is degenerate. The guard must stay silent even when 95% of candidate keys
    are evicted, because every row retains the protected key 0 with weight
    `1/(q+1) >= 1/n`, far above the `1e-9` threshold.

    This is the test that shows the guard is NOT a causal-window artifact.
    """
    a = uniform_causal()
    mask = mask_lowest_at_last_row(a, frac)
    m = row_masses(a, mask)
    assert not guard_fires(a, mask)
    assert m.max() < 1.0 - EPS
    # m_B is nonetheless strongly window-dependent -- documented, not a bug.
    assert m.max() > m[N - 1]


def test_m_B_is_window_size_dependent_even_when_guard_is_silent():
    """`m_B` differs by a factor of ~2 between an early and a late row for the
    SAME eviction set. Raw `m_B` is therefore not comparable across query
    rows, which is why `docs/METRIC_DEFINITIONS.md` tags it
    WINDOW-SIZE-DEPENDENT."""
    a = uniform_causal()
    m = row_masses(a, mask_lowest_at_last_row(a, 0.5))
    assert m[31] > 0.9
    assert m[N - 1] < 0.6


# --------------------------------------------------------------------------
# What the guard does detect: extreme concentration.
# --------------------------------------------------------------------------

def test_guard_fires_on_near_one_hot_attention():
    """Near-deterministic rows are the real trigger: the eviction set chosen
    at the last row captures another row's single attended key, and the
    remaining visible mass underflows."""
    a = one_hot_causal(off=1e-17)
    mask = mask_lowest_at_last_row(a, 0.5)
    assert guard_fires(a, mask)
    assert (row_masses(a, mask) >= 1.0 - EPS).sum() > 0


@pytest.mark.parametrize("off,expected", [(1e-2, False), (1e-6, False), (1e-10, True)])
def test_guard_threshold_is_set_by_underflow_not_by_concentration(off, expected):
    """The guard cannot distinguish 'very concentrated' (1e-6 off-diagonal)
    from uniform -- it passes both. It only fires once off-diagonal weight
    falls below roughly the `1e-9` mass epsilon. Documented so the threshold
    is understood as a float-precision boundary, not a modelling choice."""
    a = one_hot_causal(off=off)
    assert guard_fires(a, mask_lowest_at_last_row(a, 0.5)) is expected


# --------------------------------------------------------------------------
# Early-KEY vs early-QUERY: opposite effects, must not be conflated.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.5, 0.95])
def test_attention_sink_at_a_protected_key_suppresses_the_guard(frac):
    """An attention sink is an early-KEY phenomenon. Because key 0 is
    protected in every experiment, a strong sink bounds retained mass below
    and makes the guard unfireable. Sink presence SUPPRESSES the event -- the
    opposite of the 'sink-like attention' attribution in
    `docs/RESEARCH_STATUS.md`."""
    a = sink_causal(sink=0.9)
    mask = mask_lowest_at_last_row(a, frac)
    assert not guard_fires(a, mask)
    assert row_masses(a, mask).max() <= 0.1 + 1e-12


def test_unprotected_first_key_is_the_true_small_window_failure():
    """The genuine small-causal-window failure: row 0 sees only key 0, so
    evicting key 0 gives `m_B = 1` exactly. The repository's `protect={0, last}`
    prevents this everywhere, which is why it never appears in the real
    experiments."""
    a = uniform_causal()
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:8] = True                      # includes key 0 -- unprotected
    m = row_masses(a, mask)
    assert m[0] == pytest.approx(1.0)
    assert guard_fires(a, mask)


# --------------------------------------------------------------------------
# Shape and quantifier semantics.
# --------------------------------------------------------------------------

def test_block_mass_is_per_query_row_for_a_full_matrix():
    """Given a full `(seq, seq)` matrix, `m_B` is a vector of length `seq`,
    one entry per query row -- not a scalar. Every aggregate built on it must
    say which row, or which reduction, it means."""
    a = uniform_causal()
    m = row_masses(a, mask_lowest_at_last_row(a, 0.3))
    assert m.shape == (N,)
    assert m[0] == pytest.approx(0.0)    # key 0 protected, row 0 sees only key 0


def test_guard_is_an_or_over_all_query_rows():
    """CURRENT BEHAVIOUR, pinned deliberately: one pathological query row
    raises for the whole block, even when every other row -- including the row
    whose output is actually measured -- is well conditioned.

    The audit measured the consequence: 225 of 237 real skips are triggered by
    rows other than the measured one, costing 13.36% of planned eviction with
    no change in measured KL (`docs/SINGULARITY_FINDINGS.md`). If this test
    ever starts failing, the guard's scope was changed on purpose.
    """
    n = 8
    a = np.tril(np.ones((n, n)))
    a[1] = [1e-18, 1.0, 0, 0, 0, 0, 0, 0]      # row 1: all mass on key 1
    a = a / a.sum(1, keepdims=True)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[1] = True                              # evict only key 1
    m = row_masses(a, mask)
    assert m[1] >= 1.0 - EPS                    # row 1 is singular
    assert m[n - 1] < 0.5                       # the measured row is fine
    assert guard_fires(a, mask)                 # ...and the whole block raises
