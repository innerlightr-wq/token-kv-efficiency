r"""Phase 11: separate between-head/layer predictive power from within-head
block-ranking predictive power, using the pilot-reproduction records.

EXPLORATORY unless a new preregistration is explicitly frozen (none is).
Does not retrofit this analysis into the old F10 preregistration -- F10's
own within-head statistic is reported unchanged in reproduce_pilot.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RECORDS_PATH = Path(__file__).resolve().parent.parent / "results" / "pilot_reproduction" / "records.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "pilot_reproduction" / "coarse_vs_fine.json"


def run() -> dict:
    if not RECORDS_PATH.exists():
        print(f"missing {RECORDS_PATH} -- run experiments/reproduce_pilot.py first")
        sys.exit(1)
    records = json.loads(RECORDS_PATH.read_text())

    strata: dict[tuple[int, int], list[dict]] = {}
    for r in records:
        strata.setdefault((r["layer"], r["head"]), []).append(r)

    # A. coarse: one (mean d_local, mean kl_b) point per (layer, head) stratum
    coarse_rows = []
    for (layer, head), rows in sorted(strata.items()):
        coarse_rows.append(dict(
            layer=layer, head=head, n=len(rows),
            mean_d_local=float(np.mean([r["d_local"] for r in rows])),
            mean_kl_b=float(np.mean([r["kl_b"] for r in rows])),
        ))
    mean_d = np.array([r["mean_d_local"] for r in coarse_rows])
    mean_kl = np.array([r["mean_kl_b"] for r in coarse_rows])
    coarse_rho, coarse_p = spearmanr(mean_d, mean_kl)

    # B. fine: within-head block-ranking Spearman(d_local, kl_b), one per stratum
    fine_rhos = []
    for (layer, head), rows in sorted(strata.items()):
        if len(rows) < 3:
            continue
        d = np.array([r["d_local"] for r in rows])
        kl = np.array([r["kl_b"] for r in rows])
        rho, _ = spearmanr(d, kl)
        if not np.isnan(rho):
            fine_rhos.append(rho)

    result = dict(
        n_strata=len(coarse_rows),
        coarse_between_head_layer_spearman=dict(
            statistic="spearman(mean d_B^local per (layer,head), mean Tier-B KL per (layer,head))",
            n=len(coarse_rows), rho=float(coarse_rho), p=float(coarse_p),
        ),
        fine_within_head_spearman=dict(
            statistic="median over strata of spearman(d_B^local, Tier-B KL) within each (layer,head)",
            n_strata_with_variance=len(fine_rhos),
            median_rho=float(np.median(fine_rhos)) if fine_rhos else None,
            all_rhos=[float(x) for x in fine_rhos],
        ),
        interpretation=(
            "Coarse (between-head/layer) and fine (within-head block-ranking) "
            "predictive power are reported separately and must not be pooled "
            "into one statistic. EXPLORATORY -- not a frozen preregistration."
        ),
    )

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
