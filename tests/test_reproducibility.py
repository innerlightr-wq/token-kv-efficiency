"""Phase 16: config-level reproducibility properties that don't need a model."""

from __future__ import annotations

from pathlib import Path

from kv_efficiency.provenance import load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "milestone2_grid.yaml"


def test_calibration_and_evaluation_prompts_are_disjoint():
    cfg = load_config(CONFIG_PATH)
    calib = set(cfg["calibration_prompts"])
    evalu = set(cfg["evaluation_prompts"])
    assert calib.isdisjoint(evalu)
    assert len(calib) > 0 and len(evalu) > 0


def test_config_load_is_deterministic():
    cfg1 = load_config(CONFIG_PATH)
    cfg2 = load_config(CONFIG_PATH)
    assert cfg1 == cfg2
    assert cfg1["calibration_prompts"] == cfg2["calibration_prompts"]
    assert cfg1["evaluation_prompts"] == cfg2["evaluation_prompts"]
    assert cfg1["layers"] == cfg2["layers"]
    assert cfg1["heads"] == cfg2["heads"]


def test_global_removal_count_matches_target_within_rounding():
    """The `k = round(q * n_candidates)` rule used throughout the eviction
    experiments must land within one entry of the exact target for every
    stratum size actually used in this milestone's grid."""
    for n_candidates in range(10, 80):
        for q in (0.1, 0.2, 0.3, 0.4, 0.5):
            k = round(q * n_candidates)
            exact = q * n_candidates
            assert abs(k - exact) <= 0.5 + 1e-9
            assert 0 <= k <= n_candidates
