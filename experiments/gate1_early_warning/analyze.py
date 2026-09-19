r"""Gate 1 analysis, following docs/GATE1_EARLY_WARNING_PREREGISTRATION.md.

Thresholds are calibrated on the CALIBRATION split's benign runs only, frozen,
then applied once to the held-out split. Discrimination uses decode step 0,
which precedes every generated token and so cannot contain post-failure
information.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate1_lib import (ATTENTION_BASELINES, DIAGNOSTIC_DIRECTION,  # noqa: E402
                       OUTPUT_BASELINES, REPO_SPECIFIC)

ALL_DIAGS = REPO_SPECIFIC + ATTENTION_BASELINES + OUTPUT_BASELINES
PERSIST = 2
GROUPS = ["early", "middle", "late"]
ARMS = {"primary_preregistered_DE": ("D", "E"),
        "exploratory_amendment_FG": ("F", "G")}


def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1e-12); rec = tp / y.sum()
    ap, prev = 0.0, 0.0
    for p, rr in zip(prec, rec):
        ap += p * (rr - prev); prev = rr
    return float(ap)


def sens_at_fpr(y, s, target):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    neg = np.sort(s[y == 0])[::-1]
    k = max(int(np.floor(target * len(neg))), 0)
    thr = neg[k] if k < len(neg) else neg[-1] - 1e-12
    return float((s[y == 1] > thr).mean())


def calibrate(runs, key, fams):
    """Threshold = 95th (or 5th) percentile over all decode steps of
    CALIBRATION-split benign runs. Family A always contributes; the arm's own
    non-failing calibration runs also contribute, so thresholds are matched to
    the arm's context lengths. Held-out runs never contribute."""
    vals = [st[key] for r in runs
            if r["split"] == "calibration" and not r["failure"]
            and (r["family"] == "A" or r["family"] in fams)
            for st in r["steps"]]
    if not vals:
        return None
    d = DIAGNOSTIC_DIRECTION[key]
    return float(np.percentile(vals, 95 if d > 0 else 5))


def crossings(run, key, thr, persist=PERSIST):
    d = DIAGNOSTIC_DIRECTION[key]
    v = [st[key] for st in run["steps"]]
    ab = [(x > thr) if d > 0 else (x < thr) for x in v]
    out = []
    for t in range(len(ab)):
        if all(ab[t + k] for k in range(persist) if t + k < len(ab)) and ab[t]:
            out.append(t)
    return ab, out


def first_warning(run, key, thr, persist=PERSIST):
    _, c = crossings(run, key, thr, persist)
    return c[0] if c else None


def main():
    data = json.loads((HERE / "runs.json").read_text())
    runs, meta = data["runs"], data["meta"]
    R = {}

    # ---------------- behavioural outcomes ----------------
    R["outcomes"] = {}
    for fam in "ABCDEFG":
        sub = [r for r in runs if r["family"] == fam]
        nf = sum(r["failure"] for r in sub)
        R["outcomes"][fam] = dict(n=len(sub), failures=nf,
                                  rate=round(nf / max(len(sub), 1), 3))
    D_runs = [r for r in runs if r["family"] in ("D", "E")]
    R["outcomes"]["D+E_pooled"] = dict(n=len(D_runs),
                                       failures=sum(r["failure"] for r in D_runs))
    tf = [r["failure_step"] for r in runs if r["failure"] and r["failure_step"] is not None]
    R["failure_step_distribution"] = dict(sorted(Counter(tf).items()))
    R["failure_step_median"] = float(np.median(tf)) if tf else None
    R["teacher_forcing_mismatches"] = sum(r["teacher_forcing_mismatches"] for r in runs)

    # D-cell breakdown (does the effect depend on one template?)
    cells = defaultdict(lambda: [0, 0])
    for r in runs:
        if r["family"] == "D":
            c = cells[f"{r['constraint']}|{r['filler']}|{r['tempt']}"]
            c[0] += r["failure"]; c[1] += 1
    R["D_cells"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(cells.items())}

    # ---------------- per-arm analysis ----------------
    R["arms"] = {}
    for arm, fams in ARMS.items():
        pool = [r for r in runs if r["family"] in fams]
        thr = {k: calibrate(runs, k, fams) for k in ALL_DIAGS}
        A = dict(families=list(fams), n=len(pool),
                 failures=sum(r["failure"] for r in pool),
                 heldout_n=sum(1 for r in pool if r["split"] == "heldout"),
                 heldout_failures=sum(r["failure"] for r in pool if r["split"] == "heldout"),
                 thresholds={k: (round(v, 6) if v is not None else None) for k, v in thr.items()})

        def disc(split, feat, pl=pool):
            y = [r["failure"] for r in pl if r["split"] == split]
            if not y or len(set(y)) < 2:
                return None
            d = DIAGNOSTIC_DIRECTION[feat]
            s_ = [d * r["steps"][0][feat] for r in pl if r["split"] == split]
            return dict(n=len(y), pos=int(sum(y)), auroc=round(auroc(y, s_), 3),
                        auprc=round(auprc(y, s_), 3),
                        sens_at_5pct=round(sens_at_fpr(y, s_, 0.05), 3),
                        sens_at_10pct=round(sens_at_fpr(y, s_, 0.10), 3))

        A["discrimination_step0_heldout"] = {k: disc("heldout", k) for k in ALL_DIAGS}
        A["discrimination_step0_calibration"] = {k: disc("calibration", k) for k in ALL_DIAGS}

        # layer groups
        A["discrimination_by_layer_group"] = {}
        for k in REPO_SPECIFIC + ["max_alpha", "entropy"]:
            g_ = {}
            for gname in GROUPS:
                key = f"{k}__{gname}"
                if key in pool[0]["steps"][0]:
                    DIAGNOSTIC_DIRECTION[key] = DIAGNOSTIC_DIRECTION[k]
                    g_[gname] = disc("heldout", key)
            A["discrimination_by_layer_group"][k] = g_

        # lead times over this arm's failures that have an onset
        fails = [r for r in pool if r["failure"] and r["failure_step"] is not None]
        lead = {}
        for k in ALL_DIAGS:
            if thr[k] is None:
                continue
            rows = []
            for r in fails:
                tD = first_warning(r, k, thr[k])
                rows.append(dict(id=r["id"], split=r["split"], tF=r["failure_step"], tD=tD,
                                 L=None if tD is None else r["failure_step"] - tD))
            Ls = [x["L"] for x in rows if x["L"] is not None]
            pos = [x for x in Ls if x > 0]
            ho = [x["L"] for x in rows if x["split"] == "heldout" and x["L"] is not None]
            lead[k] = dict(n_failures=len(rows), n_warned=len(Ls), n_warned_before=len(pos),
                           frac_warned_before=round(len(pos) / max(len(rows), 1), 3),
                           median_L_all_warned=float(np.median(Ls)) if Ls else None,
                           median_L_positive=float(np.median(pos)) if pos else None,
                           ge1=sum(1 for x in Ls if x >= 1), ge2=sum(1 for x in Ls if x >= 2),
                           ge4=sum(1 for x in Ls if x >= 4),
                           L_distribution=dict(sorted(Counter(Ls).items())),
                           heldout_median_L=float(np.median(ho)) if ho else None)
        A["lead_times"] = lead

        # false alarms: family A, plus this arm's non-failing runs
        fa = {}
        for k in ALL_DIAGS:
            if thr[k] is None:
                continue
            for scope, pl in (("benign_A", [r for r in runs if r["family"] == "A"]),
                              ("arm_nonfailures", [r for r in pool if not r["failure"]])):
                nrun = sum(1 for r in pl if first_warning(r, k, thr[k]) is not None)
                nst = sum(len(r["steps"]) for r in pl)
                nw = sum(len(crossings(r, k, thr[k])[1]) for r in pl)
                fa.setdefault(k, {})[scope] = dict(
                    runs=len(pl), runs_with_warning=nrun,
                    run_fpr=round(nrun / max(len(pl), 1), 3),
                    warnings_per_token=round(nw / max(nst, 1), 4))
        A["false_alarms"] = fa

        # temporal stability
        stab = {}
        for k in REPO_SPECIFIC + ["max_alpha", "entropy"]:
            if thr[k] is None:
                continue
            def summarise(pl, kk=k, tt=thr[k]):
                mc, iso, integ = [], [], []
                for r in pl:
                    ab, _ = crossings(r, kk, tt, persist=1)
                    best = cur = 0
                    for a in ab:
                        cur = cur + 1 if a else 0
                        best = max(best, cur)
                    mc.append(best)
                    iso.append(sum(1 for i, a in enumerate(ab) if a
                                   and not (i and ab[i-1])
                                   and not (i+1 < len(ab) and ab[i+1])))
                    integ.append(sum(ab) / max(len(ab), 1))
                return dict(max_consecutive_median=float(np.median(mc)) if mc else None,
                            isolated_spikes_median=float(np.median(iso)) if iso else None,
                            abnormal_fraction_median=round(float(np.median(integ)), 3) if integ else None)
            stab[k] = dict(failures=summarise(fails),
                           benign=summarise([r for r in pool if not r["failure"]]))
        A["temporal_stability"] = stab

        # event-aligned
        ctrls = [r for r in pool if not r["failure"]]
        med = int(np.median([r["failure_step"] for r in fails])) if fails else 0
        ev = {}
        for k in REPO_SPECIFIC + ["max_alpha", "entropy", "out_entropy", "logit_margin"]:
            cf, cc = {}, {}
            for off in range(-10, 6):
                vf = [r["steps"][r["failure_step"] + off][k] for r in fails
                      if 0 <= r["failure_step"] + off < len(r["steps"])]
                vc = [r["steps"][med + off][k] for r in ctrls if 0 <= med + off < len(r["steps"])]
                if vf: cf[off] = round(float(np.median(vf)), 6)
                if vc: cc[off] = round(float(np.median(vc)), 6)
            ev[k] = dict(failures=cf, matched_controls=cc, n_failures=len(fails),
                         n_controls=len(ctrls), control_alignment_step=med)
        A["event_aligned"] = ev

        # pole events, pre-failure steps only vs benign
        def pole(pl, pre=False):
            tot = hit = 0
            for r in pl:
                lim = r["failure_step"] if (pre and r["failure_step"] is not None) else len(r["steps"])
                for st in r["steps"][:max(lim, 1)]:
                    tot += 1; hit += int(st["near_one_hot"] > 0)
            return dict(steps=tot, near_one_hot_steps=hit, rate=round(hit / max(tot, 1), 3))
        A["pole_events"] = dict(failures_prefailure_only=pole(fails, True),
                                failures_all_steps=pole(fails),
                                arm_nonfailures=pole(ctrls),
                                benign_A=pole([r for r in runs if r["family"] == "A"]))

        # incremental information
        def logistic_cv(X, y, folds=5):
            X = np.asarray(X, float); y = np.asarray(y, float)
            if len(set(y.tolist())) < 2:
                return None
            X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
            X = np.hstack([np.ones((len(X), 1)), X])
            idx = np.arange(len(y)); np.random.default_rng(0).shuffle(idx)
            ll, pr = [], np.zeros(len(y))
            for f in range(folds):
                te = idx[f::folds]; tr = np.setdiff1d(idx, te)
                if len(set(y[tr].tolist())) < 2:
                    return None
                w = np.zeros(X.shape[1])
                for _ in range(200):
                    pp = 1 / (1 + np.exp(-X[tr] @ w))
                    W = np.maximum(pp * (1 - pp), 1e-6)
                    H = X[tr].T @ (X[tr] * W[:, None]) + 1e-3 * np.eye(X.shape[1])
                    w += np.linalg.solve(H, X[tr].T @ (y[tr] - pp))
                q = np.clip(1 / (1 + np.exp(-X[te] @ w)), 1e-9, 1 - 1e-9)
                ll.append(float(-(y[te] * np.log(q) + (1 - y[te]) * np.log(1 - q)).mean()))
                pr[te] = q
            return dict(cv_logloss=round(float(np.mean(ll)), 4), cv_auroc=round(auroc(y, pr), 3))

        y = [r["failure"] for r in pool]
        inc = {}
        for base in ("entropy", "out_entropy"):
            inc[base] = dict(baseline_only=logistic_cv([[r["steps"][0][base]] for r in pool], y))
            for k in ("inversion_rate", "abs_gamma_mean", "max_d_local", "pole_proximity"):
                inc[base][f"plus_{k}"] = logistic_cv(
                    [[r["steps"][0][base], r["steps"][0][k]] for r in pool], y)
        A["incremental_information"] = inc

        # per-family (cross-prompt robustness)
        A["per_family"] = {}
        for fam in fams:
            fp = [r for r in runs if r["family"] == fam]
            A["per_family"][fam] = {k: disc("heldout", k, fp) for k in
                                    REPO_SPECIFIC + ["max_alpha", "entropy", "out_entropy"]}
        R["arms"][arm] = A

    R["meta"] = meta
    (HERE / "analysis.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("outcomes", "failure_step_distribution",
                                        "failure_step_median", "teacher_forcing_mismatches")}, indent=1))
    print("\nwrote analysis.json")


if __name__ == "__main__":
    main()
