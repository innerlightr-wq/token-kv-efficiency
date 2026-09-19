"""Decision-relevant plots only (preregistration section 28 / brief step 28)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
R = json.loads((HERE / "analysis.json").read_text())
runs = json.loads((HERE / "runs.json").read_text())["runs"]
arm = R["arms"]["exploratory_amendment_FG"]
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

# 1+2: event-aligned trajectories, repo diagnostic vs baselines
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7))
for ax, key, title in zip(axes,
        ["inversion_rate", "entropy", "out_entropy"],
        ["Gamma inversion rate (repo)", "attention entropy (baseline)", "output entropy (baseline)"]):
    ev = arm["event_aligned"][key]
    f = {int(k): v for k, v in ev["failures"].items()}
    c = {int(k): v for k, v in ev["matched_controls"].items()}
    ax.plot(sorted(f), [f[k] for k in sorted(f)], "o-", ms=3, lw=1.4, label=f"failures (n={ev['n_failures']})")
    ax.plot(sorted(c), [c[k] for k in sorted(c)], "s--", ms=3, lw=1.2, color="grey",
            label=f"matched controls (n={ev['n_controls']})")
    ax.axvline(0, color="crimson", lw=1, ls=":")
    ax.set_title(title); ax.set_xlabel("step relative to failure onset $t_F$")
    ax.legend(fontsize=6, frameon=False)
axes[0].set_ylabel("median value")
fig.suptitle("Event-aligned trajectories, F+G arm (t=0 is behavioural failure onset)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig1_event_aligned.png", bbox_inches="tight"); plt.close(fig)

# 3: AUROC comparison, held-out
fig, ax = plt.subplots(figsize=(5.2, 3.2))
repo = ["inversion_rate", "abs_gamma_mean", "abs_gamma_max", "max_d_local",
        "pole_proximity", "max_m_B_block8"]
base = ["max_alpha", "entropy", "eff_support", "out_entropy", "logit_margin",
        "refusal_prob", "hidden_norm"]
names, vals, cols = [], [], []
for k in repo:
    d = arm["discrimination_step0_heldout"].get(k)
    if d: names.append(k); vals.append(d["auroc"]); cols.append("#1f77b4")
for k in base:
    d = arm["discrimination_step0_heldout"].get(k)
    if d: names.append(k); vals.append(d["auroc"]); cols.append("#999999")
o = np.argsort(vals)
ax.barh([names[i] for i in o], [vals[i] for i in o], color=[cols[i] for i in o])
ax.axvline(0.5, color="crimson", lw=1, ls=":")
ax.set_xlabel("held-out AUROC at decode step 0"); ax.set_xlim(0, 1)
ax.set_title("Discrimination: repository-specific (blue) vs baselines (grey)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig2_auroc.png", bbox_inches="tight"); plt.close(fig)

# 4: lead-time distribution
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for k, m in (("inversion_rate", "o"), ("max_d_local", "s"), ("out_entropy", "^")):
    dist = arm["lead_times"][k]["L_distribution"]
    if not dist: continue
    xs = sorted(int(x) for x in dist)
    ax.plot(xs, [dist[str(x)] for x in xs], m + "-", ms=4, lw=1.2,
            label=f"{k} (warned {arm['lead_times'][k]['n_warned']}/27)")
ax.axvline(0, color="crimson", lw=1, ls=":")
ax.set_xlabel("lead time $L = t_F - t_D$  (>0 = warning before failure)")
ax.set_ylabel("failures"); ax.legend(fontsize=6, frameon=False)
ax.set_title("Lead-time distribution, F+G arm", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig3_lead_time.png", bbox_inches="tight"); plt.close(fig)

# 5: detection vs false-positive burden
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for k in repo + ["out_entropy", "entropy"]:
    lt, fa = arm["lead_times"].get(k), arm["false_alarms"].get(k)
    if not lt or not fa: continue
    x = fa["benign_A"]["run_fpr"]; y = lt["frac_warned_before"]
    ax.scatter(x, y, s=26, color="#1f77b4" if k in repo else "#999999")
    ax.annotate(k, (x, y), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
lim = max(0.4, ax.get_xlim()[1])
ax.plot([0, lim], [0, lim], color="crimson", lw=1, ls=":")
ax.set_xlabel("benign (family A) run false-positive rate")
ax.set_ylabel("fraction of failures warned before $t_F$")
ax.set_title("Detection vs false-alarm burden\n(below the dotted line = worse than its own FPR)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig4_detection_vs_fpr.png", bbox_inches="tight"); plt.close(fig)

# 6: layerwise
fig, ax = plt.subplots(figsize=(5.0, 2.8))
groups = ["early", "middle", "late"]
for k in ["inversion_rate", "max_d_local", "pole_proximity", "max_alpha", "entropy"]:
    g = arm["discrimination_by_layer_group"].get(k, {})
    ys = [g[gn]["auroc"] if g.get(gn) else np.nan for gn in groups]
    ax.plot(groups, ys, "o-", ms=4, lw=1.2, label=k)
ax.axhline(0.5, color="crimson", lw=1, ls=":")
ax.set_ylabel("held-out AUROC"); ax.set_ylim(0, 1)
ax.legend(fontsize=6, frameon=False, ncol=2)
ax.set_title("Layer-group dependence (layers 0-7 / 8-15 / 16-23)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig5_layerwise.png", bbox_inches="tight"); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
