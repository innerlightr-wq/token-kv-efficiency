r"""TOKEN-KV CLUSTER-AWARE STATISTICAL REANALYSIS (the single LOCK-B correction).

Re-estimates uncertainty for the already-stored DPH (downstream_propagation_sample)
and FLH (functional_propagation_audit) results by resampling whole clusters of
candidate pairs -- not individual pairs -- with replacement.

No model forward passes are run here. No new observations are generated. No
experimental definitions (flip/immediate/unresolved category logic, KL sign
convention, etc.) are altered. This script only reads the two existing
`cases.json` files and recomputes uncertainty.

Primary cluster unit (justified in Part II of the accompanying report):
    (architecture, prompt, layer, head)
i.e. the (prompt, layer, head) cell WITHIN an architecture. Candidate pairs
drawn from the same architecture/prompt/layer/head cell share the same
underlying softmax attention distribution and the same forward pass, so they
are not independent draws; different architectures never share a forward
pass at all, so architecture must be part of the cluster key, not folded
away by treating "prompt 0" as the same context across architectures.

Two alternate cluster definitions are computed only as a sensitivity check
(Part IX), not as a competing primary analysis:
    ALT-1: (layer, head)            -- collapses architecture AND prompt
    ALT-2: (architecture, prompt)   -- collapses layer AND head

All bootstrap resampling is stratified by architecture wherever a pooled
statistic is reported: within each replicate, the pythia clusters and the
gpt2 clusters are independently resampled (each drawing as many clusters,
with replacement, as were originally observed for that architecture), then
combined for the pooled statistic. This preserves the observed
architecture mix in every replicate and is what makes the same replicate
usable simultaneously for the pooled statistic, the two per-architecture
statistics, and the architecture-difference statistic.

For the paired DPH-vs-FLH comparisons (Part V/VI), the identical resampled
case-index multiset (built once per replicate from the resampled clusters)
is applied to BOTH the DPH category array and the FLH category array
(the two files are index-aligned 1:1, verified in Part I of this run) --
this is what makes the reported difference a genuinely paired estimate
rather than an independent-bootstrap difference of two separate CIs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260910
B = 10_000

ROOT = Path(__file__).resolve().parents[1]
DPH_PATH = ROOT / "results" / "downstream_propagation_sample" / "cases.json"
FLH_PATH = ROOT / "results" / "functional_propagation_audit" / "cases.json"
OUT_DIR = ROOT / "results" / "cluster_aware_reanalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLIP_CATS = {"single_flip", "multiple_flip"}
UNRESOLVED_CATS = {"never_consistent", "unresolved"}


def load_cases():
    dph = json.loads(DPH_PATH.read_text())
    flh = json.loads(FLH_PATH.read_text())
    assert len(dph) == len(flh) == 250
    n_mismatch = 0
    for d, f in zip(dph, flh):
        key_d = (d["arch"], d["prompt"], d["layer"], d["head"], d["i"], d["j"])
        key_f = (f["arch"], f["prompt"], f["layer"], f["head"], f["i"], f["j"])
        if key_d != key_f:
            n_mismatch += 1
    return dph, flh, n_mismatch


def cluster_key(case, level: str):
    if level == "primary":
        return (case["arch"], case["prompt"], case["layer"], case["head"])
    if level == "layer_head":
        return (case["layer"], case["head"])
    if level == "arch_prompt":
        return (case["arch"], case["prompt"])
    raise ValueError(level)


def build_cluster_index(cases, level: str, indices=None):
    """cluster_key -> list of case indices, restricted to `indices` if given."""
    idx_iter = indices if indices is not None else range(len(cases))
    out: dict = defaultdict(list)
    for i in idx_iter:
        out[cluster_key(cases[i], level)].append(i)
    return dict(out)


def cluster_size_stats(cluster_index: dict):
    sizes = [len(v) for v in cluster_index.values()]
    sizes_arr = np.array(sizes)
    return {
        "n_clusters": len(sizes),
        "min_size": int(sizes_arr.min()) if len(sizes) else None,
        "median_size": float(np.median(sizes_arr)) if len(sizes) else None,
        "max_size": int(sizes_arr.max()) if len(sizes) else None,
        "n_singleton_clusters": int((sizes_arr == 1).sum()) if len(sizes) else 0,
        "frac_singleton_clusters": float((sizes_arr == 1).mean()) if len(sizes) else None,
    }


def resample_indices(rng, cluster_index: dict) -> list:
    """One cluster-bootstrap draw: resample clusters (same count) with replacement,
    keep ALL case indices belonging to each drawn cluster (duplicated if a
    cluster is drawn more than once)."""
    keys = list(cluster_index.keys())
    n = len(keys)
    if n == 0:
        return []
    draw = rng.integers(0, n, size=n)
    out: list = []
    for d in draw:
        out.extend(cluster_index[keys[d]])
    return out


def rate(indicator: np.ndarray, idxs: list) -> float:
    if len(idxs) == 0:
        return float("nan")
    return float(indicator[idxs].mean())


def summarize_boot(observed: float, boot: np.ndarray) -> dict:
    boot = boot[~np.isnan(boot)]
    return {
        "observed": observed,
        "boot_mean": float(np.mean(boot)) if len(boot) else None,
        "boot_se": float(np.std(boot, ddof=1)) if len(boot) > 1 else None,
        "ci95_pct": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if len(boot)
        else None,
        "n_boot_used": int(len(boot)),
    }


def main():
    dph, flh, n_mismatch = load_cases()
    n = len(dph)

    arch = np.array([c["arch"] for c in dph])
    dph_cat = np.array([c["category"] for c in dph])
    flh_cat = np.array([c["functional_category"] for c in flh])
    final_winner = np.array([c["final_winner"] for c in dph])

    dph_flip = np.array([cat in FLIP_CATS for cat in dph_cat])
    dph_immediate = np.array([cat == "immediate_consistent" for cat in dph_cat])
    dph_unresolved = np.array([cat in UNRESOLVED_CATS for cat in dph_cat])
    dph_multi = np.array([cat == "multiple_flip" for cat in dph_cat])

    flh_flip = np.array([cat in FLIP_CATS for cat in flh_cat])
    flh_immediate = np.array([cat == "immediate_consistent" for cat in flh_cat])
    flh_unresolved = np.array([cat in UNRESOLVED_CATS for cat in flh_cat])

    # ---- validation against stored report numbers (Part I sanity check) ----
    validation = {
        "pooled_dph_flip_rate_recomputed": float(dph_flip.mean()),
        "pooled_dph_immediate_rate_recomputed": float(dph_immediate.mean()),
        "pooled_dph_unresolved_rate_recomputed": float(dph_unresolved.mean()),
        "pooled_dph_multiple_flip_rate_recomputed": float(dph_multi.mean()),
        "pooled_flh_flip_rate_recomputed": float(flh_flip.mean()),
        "pooled_flh_immediate_rate_recomputed": float(flh_immediate.mean()),
        "pooled_flh_unresolved_rate_recomputed": float(flh_unresolved.mean()),
        "expected_from_stored_reports": {
            "dph_flip_rate": 0.324,
            "dph_immediate_consistent": 0.432,
            "dph_unresolved_rate": 0.244,
            "dph_multiple_flip_rate": 0.096,
            "flh_flip_rate": 0.168,
            "flh_immediate_consistent": 0.196,
            "flh_unresolved_rate": 0.636,
        },
        "index_alignment_mismatches": n_mismatch,
    }

    # ---- cluster inventories ----
    primary_all = build_cluster_index(dph, "primary")
    primary_pythia = build_cluster_index(dph, "primary", [i for i in range(n) if arch[i] == "pythia"])
    primary_gpt2 = build_cluster_index(dph, "primary", [i for i in range(n) if arch[i] == "gpt2"])
    alt_layer_head = build_cluster_index(dph, "layer_head")
    alt_arch_prompt = build_cluster_index(dph, "arch_prompt")

    cluster_report = {
        "primary_pooled": cluster_size_stats(primary_all),
        "primary_pythia": cluster_size_stats(primary_pythia),
        "primary_gpt2": cluster_size_stats(primary_gpt2),
        "alt_layer_head": cluster_size_stats(alt_layer_head),
        "alt_arch_prompt": cluster_size_stats(alt_arch_prompt),
    }

    rng = np.random.default_rng(SEED)

    # ---- Part IV/V/VI/VIII: one joint bootstrap loop, architecture-stratified ----
    boot_pooled_dph_flip = np.empty(B)
    boot_pooled_dph_immediate = np.empty(B)
    boot_pooled_dph_unresolved = np.empty(B)
    boot_pooled_dph_multi = np.empty(B)

    boot_pythia_dph_flip = np.empty(B)
    boot_pythia_dph_immediate = np.empty(B)
    boot_pythia_dph_unresolved = np.empty(B)

    boot_gpt2_dph_flip = np.empty(B)
    boot_gpt2_dph_immediate = np.empty(B)
    boot_gpt2_dph_unresolved = np.empty(B)

    boot_arch_diff_flip = np.empty(B)

    boot_pooled_flh_flip = np.empty(B)
    boot_pooled_flh_immediate = np.empty(B)
    boot_pooled_flh_unresolved = np.empty(B)

    boot_diff_flip_flh_minus_dph = np.empty(B)
    boot_diff_unresolved_flh_minus_dph = np.empty(B)

    for b in range(B):
        idx_py = resample_indices(rng, primary_pythia)
        idx_gp = resample_indices(rng, primary_gpt2)
        idx_pooled = idx_py + idx_gp

        boot_pythia_dph_flip[b] = rate(dph_flip, idx_py)
        boot_pythia_dph_immediate[b] = rate(dph_immediate, idx_py)
        boot_pythia_dph_unresolved[b] = rate(dph_unresolved, idx_py)

        boot_gpt2_dph_flip[b] = rate(dph_flip, idx_gp)
        boot_gpt2_dph_immediate[b] = rate(dph_immediate, idx_gp)
        boot_gpt2_dph_unresolved[b] = rate(dph_unresolved, idx_gp)

        boot_arch_diff_flip[b] = boot_pythia_dph_flip[b] - boot_gpt2_dph_flip[b]

        boot_pooled_dph_flip[b] = rate(dph_flip, idx_pooled)
        boot_pooled_dph_immediate[b] = rate(dph_immediate, idx_pooled)
        boot_pooled_dph_unresolved[b] = rate(dph_unresolved, idx_pooled)
        boot_pooled_dph_multi[b] = rate(dph_multi, idx_pooled)

        boot_pooled_flh_flip[b] = rate(flh_flip, idx_pooled)
        boot_pooled_flh_immediate[b] = rate(flh_immediate, idx_pooled)
        boot_pooled_flh_unresolved[b] = rate(flh_unresolved, idx_pooled)

        boot_diff_flip_flh_minus_dph[b] = boot_pooled_flh_flip[b] - boot_pooled_dph_flip[b]
        boot_diff_unresolved_flh_minus_dph[b] = (
            boot_pooled_flh_unresolved[b] - boot_pooled_dph_unresolved[b]
        )

    dph_results = {
        "pooled": {
            "flip_rate": summarize_boot(float(dph_flip.mean()), boot_pooled_dph_flip),
            "immediate_consistent_rate": summarize_boot(
                float(dph_immediate.mean()), boot_pooled_dph_immediate
            ),
            "unresolved_rate": summarize_boot(float(dph_unresolved.mean()), boot_pooled_dph_unresolved),
            "multiple_flip_rate": summarize_boot(float(dph_multi.mean()), boot_pooled_dph_multi),
        },
        "pythia": {
            "flip_rate": summarize_boot(
                float(dph_flip[arch == "pythia"].mean()), boot_pythia_dph_flip
            ),
            "immediate_consistent_rate": summarize_boot(
                float(dph_immediate[arch == "pythia"].mean()), boot_pythia_dph_immediate
            ),
            "unresolved_rate": summarize_boot(
                float(dph_unresolved[arch == "pythia"].mean()), boot_pythia_dph_unresolved
            ),
        },
        "gpt2": {
            "flip_rate": summarize_boot(float(dph_flip[arch == "gpt2"].mean()), boot_gpt2_dph_flip),
            "immediate_consistent_rate": summarize_boot(
                float(dph_immediate[arch == "gpt2"].mean()), boot_gpt2_dph_immediate
            ),
            "unresolved_rate": summarize_boot(
                float(dph_unresolved[arch == "gpt2"].mean()), boot_gpt2_dph_unresolved
            ),
        },
    }

    architecture_difference = summarize_boot(
        float(dph_flip[arch == "pythia"].mean() - dph_flip[arch == "gpt2"].mean()),
        boot_arch_diff_flip,
    )

    flh_results = {
        "pooled": {
            "flip_rate": summarize_boot(float(flh_flip.mean()), boot_pooled_flh_flip),
            "immediate_consistent_rate": summarize_boot(
                float(flh_immediate.mean()), boot_pooled_flh_immediate
            ),
            "unresolved_rate": summarize_boot(float(flh_unresolved.mean()), boot_pooled_flh_unresolved),
        }
    }

    paired_diff = {
        "flip_rate_FLH_minus_DPH": summarize_boot(
            float(flh_flip.mean() - dph_flip.mean()), boot_diff_flip_flh_minus_dph
        ),
        "unresolved_rate_FLH_minus_DPH": summarize_boot(
            float(flh_unresolved.mean() - dph_unresolved.mean()), boot_diff_unresolved_flh_minus_dph
        ),
    }

    # ---- residual-never-consistent resolved-by-functional (paired subset) ----
    never_idx_all = [i for i in range(n) if dph_cat[i] == "never_consistent"]
    resolved_indicator = np.array(
        [flh_cat[i] not in UNRESOLVED_CATS for i in range(n)]
    )  # only meaningful restricted to never_idx_all
    never_cluster_pythia = build_cluster_index(
        dph, "primary", [i for i in never_idx_all if arch[i] == "pythia"]
    )
    never_cluster_gpt2 = build_cluster_index(
        dph, "primary", [i for i in never_idx_all if arch[i] == "gpt2"]
    )
    rng2 = np.random.default_rng(SEED + 1)
    boot_resolved = np.empty(B)
    for b in range(B):
        idx = resample_indices(rng2, never_cluster_pythia) + resample_indices(rng2, never_cluster_gpt2)
        boot_resolved[b] = rate(resolved_indicator, idx)
    observed_resolved = (
        float(resolved_indicator[never_idx_all].mean()) if never_idx_all else float("nan")
    )
    residual_never_resolved = summarize_boot(observed_resolved, boot_resolved)
    residual_never_resolved["n_cases_in_subset"] = len(never_idx_all)
    residual_never_resolved["n_clusters_in_subset"] = len(never_cluster_pythia) + len(never_cluster_gpt2)

    # ---- Part VII: local-vs-mass subgroup ----
    def subgroup_bootstrap(winner_label: str, seed_offset: int):
        sub_idx_py = [i for i in range(n) if arch[i] == "pythia" and final_winner[i] == winner_label]
        sub_idx_gp = [i for i in range(n) if arch[i] == "gpt2" and final_winner[i] == winner_label]
        sub_cluster_py = build_cluster_index(dph, "primary", sub_idx_py)
        sub_cluster_gp = build_cluster_index(dph, "primary", sub_idx_gp)
        rng_s = np.random.default_rng(SEED + seed_offset)
        boot_flip = np.empty(B)
        boot_imm = np.empty(B)
        for b in range(B):
            idx = resample_indices(rng_s, sub_cluster_py) + resample_indices(rng_s, sub_cluster_gp)
            boot_flip[b] = rate(dph_flip, idx)
            boot_imm[b] = rate(dph_immediate, idx)
        all_idx = sub_idx_py + sub_idx_gp
        return {
            "n_cases": len(all_idx),
            "n_clusters": len(sub_cluster_py) + len(sub_cluster_gp),
            "flip_rate": summarize_boot(float(dph_flip[all_idx].mean()), boot_flip),
            "immediate_consistent_rate": summarize_boot(float(dph_immediate[all_idx].mean()), boot_imm),
        }

    local_vs_mass = {
        "local_final_winner": subgroup_bootstrap("local", 10),
        "mass_final_winner": subgroup_bootstrap("mass", 20),
    }

    # ---- Part IX: sensitivity to cluster definition (DPH flip_rate & unresolved_rate only) ----
    def sensitivity_bootstrap(cluster_index_all: dict, seed_offset: int):
        rng_s = np.random.default_rng(SEED + seed_offset)
        boot_flip = np.empty(B)
        boot_unres = np.empty(B)
        for b in range(B):
            idx = resample_indices(rng_s, cluster_index_all)
            boot_flip[b] = rate(dph_flip, idx)
            boot_unres[b] = rate(dph_unresolved, idx)
        return {
            "flip_rate": summarize_boot(float(dph_flip.mean()), boot_flip),
            "unresolved_rate": summarize_boot(float(dph_unresolved.mean()), boot_unres),
        }

    sensitivity = {
        "primary_arch_prompt_layer_head": {
            "note": "same as pooled primary result above, recomputed here as an UNSTRATIFIED "
            "single-pool cluster bootstrap (no architecture stratification) purely for a "
            "like-for-like comparison against the two alternates below.",
            **sensitivity_bootstrap(primary_all, 100),
        },
        "alt_layer_head_collapses_arch_and_prompt": sensitivity_bootstrap(alt_layer_head, 200),
        "alt_arch_prompt_collapses_layer_and_head": sensitivity_bootstrap(alt_arch_prompt, 300),
    }

    summary = {
        "config": {"seed": SEED, "n_bootstrap": B, "n_cases": n},
        "validation": validation,
        "clusters": cluster_report,
        "dph_cluster_aware": dph_results,
        "architecture_difference_flip_rate_pythia_minus_gpt2": architecture_difference,
        "flh_cluster_aware": flh_results,
        "paired_dph_vs_flh_differences": paired_diff,
        "residual_never_consistent_resolved_by_functional": residual_never_resolved,
        "local_vs_mass_subgroup": local_vs_mass,
        "sensitivity_alternate_cluster_definitions": sensitivity,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("CLUSTER-AWARE STATISTICAL REANALYSIS (LOCK-B correction)")
    lines.append("=" * 100)
    lines.append(f"seed={SEED}  B={B}  n_cases={n}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART I -- VALIDATION (recomputed rates vs. stored report numbers)")
    lines.append("-" * 100)
    for k, v in validation.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART II -- CLUSTER INVENTORY (primary unit = architecture x prompt x layer x head)")
    lines.append("-" * 100)
    for k, v in cluster_report.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART IV -- DPH CLUSTER-AWARE ESTIMATES")
    lines.append("-" * 100)
    for scope, stats in dph_results.items():
        lines.append(f"  [{scope}]")
        for stat_name, s in stats.items():
            lines.append(f"    {stat_name}: observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  "
                          f"boot_se={s['boot_se']:.4f}  95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART VIII -- ARCHITECTURE DIFFERENCE (flip_rate, pythia - gpt2)")
    lines.append("-" * 100)
    s = architecture_difference
    lines.append(f"  observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  boot_se={s['boot_se']:.4f}  "
                  f"95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART V -- FLH CLUSTER-AWARE ESTIMATES (pooled)")
    lines.append("-" * 100)
    for stat_name, s in flh_results["pooled"].items():
        lines.append(f"    {stat_name}: observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  "
                      f"boot_se={s['boot_se']:.4f}  95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART VI -- PAIRED DPH-vs-FLH DIFFERENCES (same resampled clusters/cases)")
    lines.append("-" * 100)
    for stat_name, s in paired_diff.items():
        lines.append(f"    {stat_name}: observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  "
                      f"boot_se={s['boot_se']:.4f}  95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("residual never_consistent cases resolved in functional space (paired subset)")
    lines.append("-" * 100)
    s = residual_never_resolved
    lines.append(f"  n_cases_in_subset={s['n_cases_in_subset']}  n_clusters_in_subset={s['n_clusters_in_subset']}")
    lines.append(f"  observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  boot_se={s['boot_se']:.4f}  "
                  f"95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART VII -- LOCAL-vs-MASS SUBGROUP (DPH)")
    lines.append("-" * 100)
    for grp, d in local_vs_mass.items():
        lines.append(f"  [{grp}] n_cases={d['n_cases']} n_clusters={d['n_clusters']}")
        for stat_name in ("flip_rate", "immediate_consistent_rate"):
            s = d[stat_name]
            lines.append(f"    {stat_name}: observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  "
                          f"boot_se={s['boot_se']:.4f}  95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PART IX -- SENSITIVITY TO CLUSTER DEFINITION (DPH pooled, unstratified bootstrap)")
    lines.append("-" * 100)
    for defn, d in sensitivity.items():
        lines.append(f"  [{defn}]")
        for stat_name in ("flip_rate", "unresolved_rate"):
            s = d[stat_name]
            lines.append(f"    {stat_name}: observed={s['observed']:.4f}  boot_mean={s['boot_mean']:.4f}  "
                          f"boot_se={s['boot_se']:.4f}  95% CI={tuple(round(x,4) for x in s['ci95_pct'])}")
    lines.append("")
    (OUT_DIR / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nDone. Wrote {OUT_DIR / 'summary.json'} and {OUT_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
