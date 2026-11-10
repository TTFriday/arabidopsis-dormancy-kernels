"""Publication figure: external projection against the Dryad cohort."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

C_BLOCK = "#2166ac"
C_REP = "#4393c3"

pred = pd.read_csv(TABLES / "external_predictions.csv")
summary = json.loads((TABLES / "external_projection.json").read_text(encoding="utf-8"))
pheno = pd.read_csv(ROOT / "data" / "processed" / "dormancy_phenotype.csv")

ind = pred[(~pred["in_train83"]) & (pred["panel_col"] >= 0)].copy()
same = pred[pred["in_train83"]].merge(
    pheno[["nativename", "dsds50"]], on="nativename", how="left"
)
assert same["dsds50"].notna().all()

res_ind = summary["results"]["independent_transductive"]["block"]
rep = summary["measurement_repeatability_same38"]


def panel_label(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def loglog_scatter(ax, x, y, color):
    ax.scatter(x, y, s=14, color=color, alpha=0.65, linewidths=0, zorder=3)
    lo = min(x.min(), y.min()) * 0.8
    hi = max(x.max(), y.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], "--", color="#777777", linewidth=0.9, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#ececec", linewidth=0.5, zorder=0)


fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), dpi=170)

ax = axes[0]
loglog_scatter(ax, ind["dsds50_dryad"].to_numpy(), ind["pred_block"].clip(lower=0.5).to_numpy(), C_BLOCK)
ax.set_xlabel("Observed DSDS50, Dryad (d)")
ax.set_ylabel("Predicted DSDS50, block kernel (d)")
ax.text(0.04, 0.96,
        f"n = {res_ind['n']} independent\n"
        f"r = {res_ind['pearson_r']:.3f}, " + r"$\rho$ = " + f"{res_ind['spearman_rho']:.3f}\n"
        f"MAE = {res_ind['mae']:.1f} d",
        transform=ax.transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.5))
panel_label(ax, "a")

ax = axes[1]
loglog_scatter(ax, same["dsds50"].to_numpy(), same["dsds50_dryad"].to_numpy(), C_REP)
ax.set_xlabel("Local DSDS50, AtPolyDB batch (d)")
ax.set_ylabel("Dryad DSDS50, chilling protocol (d)")
ax.text(0.04, 0.96,
        f"n = {rep['n']} shared accessions\n"
        f"r = {rep['pearson_r']:.3f}, " + r"$\rho$ = " + f"{rep['spearman_rho']:.3f}\n"
        f"means: {rep['mean_local']:.1f} vs {rep['mean_dryad']:.1f} d",
        transform=ax.transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.5))
panel_label(ax, "b")

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "fig4.png", dpi=600, facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig4.pdf", facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved fig4.png / fig4.pdf")

# consistency check against the frozen summary
r_check = stats.pearsonr(ind["pred_block"], ind["dsds50_dryad"]).statistic
assert abs(r_check - res_ind["pearson_r"]) < 1e-10
r_rep = stats.pearsonr(same["dsds50"], same["dsds50_dryad"]).statistic
assert abs(r_rep - rep["pearson_r"]) < 1e-10
print("statistics verified against external_projection.json")