r"""Tests for the Gate 1 passive instrumentation.

Covers the correctness properties the experiment's conclusions depend on:
grouped-query head mapping, the head-space computation of `g_i`, the `Gamma`
boundary theorem, warning/failure-step indexing, benign-control labelling,
and -- most importantly -- the absence of future-token leakage.

A tiny randomly-initialised Qwen2 is used so the suite stays fast and
hermetic; the real model is exercised by the experiment itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "gate1_early_warning"))

from gate1_lib import (  # noqa: E402
    DIAGNOSTIC_DIRECTION, Qwen2DiagnosticAdapter, g_norms,
    inversion_rate_and_gamma, label_format_constraint, label_refusal,
    step_diagnostics, verify_teacher_forcing,
)


@pytest.fixture(scope="module")
def tiny():
    """Deterministic tiny Qwen2 with GQA (4 query heads, 2 KV heads)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg).eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m, Qwen2DiagnosticAdapter(m)


# ------------------------------- shapes / GQA -------------------------------

def test_gqa_group_mapping(tiny):
    _, ad = tiny
    assert (ad.n_q, ad.n_kv, ad.group) == (4, 2, 2)
    assert [ad.kv_group_of(h) for h in range(4)] == [0, 0, 1, 1]


def test_value_capture_shapes(tiny):
    m, ad = tiny
    ids = torch.randint(0, 128, (1, 11))
    out, vals = ad.capture_values(ids)
    assert set(vals) == {0, 1}
    for L in vals:
        assert vals[L].shape == (ad.n_kv, 11, ad.head_dim)
    assert out.attentions[0].shape == (1, ad.n_q, 11, 11)
    assert ad.w_o_head(0, 0).shape == (m.config.hidden_size, ad.head_dim)
    assert ad.gram(0, 0).shape == (ad.head_dim, ad.head_dim)


# --------------------- g_i in head space == d_model space -------------------

def test_g_norms_match_direct_dmodel_computation(tiny):
    """`||W_O_h u||` via the Gram matrix must equal forming `o` and `v_i`
    explicitly in d_model space, which is what local_algebra.py does."""
    m, ad = tiny
    ids = torch.randint(0, 128, (1, 13))
    out, vals = ad.capture_values(ids)
    row, L, h = 9, 1, 3
    a = out.attentions[L][0, h, row, : row + 1].double()
    a = a / a.sum()
    v_head = vals[L][ad.kv_group_of(h)][: row + 1]
    fast = g_norms(ad, L, h, a, v_head).numpy()
    wo = ad.w_o_head(L, h)
    v_dmodel = v_head @ wo.T                 # (k, d_model)
    o = a @ v_dmodel
    direct = torch.linalg.norm(o.unsqueeze(0) - v_dmodel, dim=-1).numpy()
    assert np.allclose(fast, direct, rtol=1e-9, atol=1e-12)


# ------------------------------ Gamma theorem -------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_gamma_sign_matches_d_ordering(seed):
    """Theorem 5.1: for `alpha_i < alpha_j`, `Gamma_ij > 0` iff `d_i > d_j`."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0.01, 0.9, size=2))
    g = rng.uniform(0.1, 5.0, size=2)
    ai, aj, gi, gj = a[0], a[1], g[0], g[1]
    di, dj = ai / (1 - ai) * gi, aj / (1 - aj) * gj
    logG = np.log(gi) - np.log(gj)
    logM = (np.log(aj) + np.log1p(-ai)) - (np.log(ai) + np.log1p(-aj))
    gamma = logG - logM
    assert (gamma > 0) == (di > dj)


def test_inversion_rate_endpoints():
    """Perfect agreement -> 0; perfect reversal -> 1."""
    rng = np.random.default_rng(0)
    a = np.linspace(0.05, 0.5, 20)
    inv, anyinv, _, _ = inversion_rate_and_gamma(a, a * 3.0, rng, 100)
    assert inv == pytest.approx(0.0, abs=1e-12) and anyinv == 0.0
    inv2, anyinv2, _, _ = inversion_rate_and_gamma(a, -a, rng, 100)
    assert inv2 == pytest.approx(1.0, abs=1e-12) and anyinv2 == 1.0


def test_inversion_rate_degenerate_inputs():
    rng = np.random.default_rng(0)
    assert inversion_rate_and_gamma(np.array([0.5]), np.array([1.0]), rng, 10)[0] == 0.0
    assert inversion_rate_and_gamma(np.array([]), np.array([]), rng, 10)[0] == 0.0


# --------------------------- failure-step indexing --------------------------

class _FakeTok:
    """Character-level stand-in so onset indexing can be tested exactly."""
    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids) if isinstance(ids, list) else ids


def test_failure_step_matches_preregistered_worked_example():
    """Preregistration section 5: for 'YES\\n\\nThe sky...', t_F = 2."""
    tok = _FakeTok()
    gen = ["YES", "\n\n", "The", " sky", " is", " blue"]
    failure, t_F, _ = label_format_constraint(tok, gen, ["YES", "NO"])
    assert failure == 1
    assert t_F == 2


def test_compliant_generation_has_no_failure_step():
    tok = _FakeTok()
    for gen in (["YES"], ["NO"], ["YES", "."], [" ", "NO", " "]):
        failure, t_F, _ = label_format_constraint(tok, gen, ["YES", "NO"])
        assert failure == 0 and t_F is None


def test_partial_prefix_is_viable_then_fails():
    tok = _FakeTok()
    failure, t_F, _ = label_format_constraint(tok, ["Y", "E", "S", "!"], ["YES", "NO"])
    assert failure == 1 and t_F == 3


def test_immediate_violation_has_onset_zero():
    tok = _FakeTok()
    failure, t_F, _ = label_format_constraint(tok, ["Because", " the"], ["YES", "NO"])
    assert failure == 1 and t_F == 0


# ---------------------------- benign-control labels -------------------------

def test_benign_control_labelling_is_zero_for_compliant_answers():
    tok = _FakeTok()
    assert label_format_constraint(tok, ["TRUE"], ["TRUE", "FALSE"])[0] == 0


def test_refusal_label_requires_both_no_marker_and_substance():
    tok = _FakeTok()
    refused = ["I'm", " sorry", ",", " but", " I", " can't", " help", " with", " that"]
    assert label_refusal(tok, refused)[0] == 0
    short = ["No", "."]
    assert label_refusal(tok, short)[0] == 0          # not substantive
    complied = ["Sure", ",", " here", " is", " how", " you", " do", " it", " step", " one"]
    f, t_F, _ = label_refusal(tok, complied)
    assert f == 1 and t_F == 0


# ------------------------------ leakage checks ------------------------------

def test_no_future_token_leakage_in_attention_row(tiny):
    """The row used for step `t` must place zero weight on any key after it."""
    m, ad = tiny
    ids = torch.randint(0, 128, (1, 15))
    out, _ = ad.capture_values(ids)
    n = ids.shape[1]
    for L in range(ad.n_layers):
        A = out.attentions[L][0]
        for row in range(n - 1):            # last row has no future keys
            future = A[:, row, row + 1:]
            assert future.numel() > 0
            assert float(future.abs().max()) == 0.0, (L, row)


def test_teacher_forcing_reproduces_greedy_decoding(tiny):
    """One forward over prompt+generation must reproduce what incremental
    greedy decoding produced, which is what licenses reading every decode
    step out of a single pass."""
    m, _ = tiny
    torch.manual_seed(0)
    prompt = torch.randint(0, 128, (1, 7))
    with torch.no_grad():
        gen = m.generate(prompt, max_new_tokens=6, do_sample=False,
                         pad_token_id=0).tolist()[0][7:]
    assert verify_teacher_forcing(m, prompt, gen) == []


def test_step_diagnostics_use_only_keys_up_to_row(tiny):
    """Diagnostics at `row` must be unchanged by tokens appended after `row`."""
    m, ad = tiny
    torch.manual_seed(0)
    base = torch.randint(0, 128, (1, 12))
    extra = torch.cat([base, torch.randint(0, 128, (1, 4))], dim=1)
    row = 11
    o1, v1 = ad.capture_values(base)
    o2, v2 = ad.capture_values(extra)
    d1 = step_diagnostics(ad, o1, v1, row, np.random.default_rng(0))
    d2 = step_diagnostics(ad, o2, v2, row, np.random.default_rng(0))
    # float32 matmul blocking differs with sequence length, so agreement is to
    # float32 precision, not bit-exact. The hard no-leakage guarantee is the
    # causal mask, asserted exactly in the test above.
    for k in d1:
        assert np.allclose(d1[k]["head_max"], d2[k]["head_max"], rtol=1e-5, atol=1e-7), (
            k, d1[k]["head_max"], d2[k]["head_max"])


# ------------------------------ config sanity -------------------------------

def test_every_diagnostic_has_a_declared_direction():
    from gate1_lib import ATTENTION_BASELINES, OUTPUT_BASELINES, REPO_SPECIFIC
    for k in REPO_SPECIFIC + ATTENTION_BASELINES + OUTPUT_BASELINES:
        assert k in DIAGNOSTIC_DIRECTION, k
        assert DIAGNOSTIC_DIRECTION[k] in (+1, -1)


def test_singleton_block_mass_is_identically_max_attention(tiny):
    """Declared in the preregistration: at singleton granularity the
    repository's max block mass IS the max-attention baseline."""
    m, ad = tiny
    ids = torch.randint(0, 128, (1, 10))
    out, vals = ad.capture_values(ids)
    d = step_diagnostics(ad, out, vals, 9, np.random.default_rng(0))
    assert d["max_m_B_single"]["head_max"] == d["max_alpha"]["head_max"]
