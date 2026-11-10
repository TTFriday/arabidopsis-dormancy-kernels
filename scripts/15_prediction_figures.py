"""Publication figures: kernel prediction comparison and pretraining diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "results" / "tables" / "prediction_figure_data.npz"
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
C_OTHER = "#878787"
C_BAD = "#b2182b"
C_BAR = "#4393c3"

d = np.load(NPZ)
y = d["y"]


def panel_label(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {name}.png / {name}.pdf")


# ---------------------------------------------------------------- Fig 1
fig, axes = plt.subplots(1, 2, figsize=(4.72, 2.35),
                         gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.38})

# (a) boxplot of per-repeat Pearson r
ax = axes[0]
keys = ["pc", "snp", "block", "ssl_graph", "combined_w01", "random_graph"]
labels = ["PC", "SNP", "Block", "SSL", "Comb.", "Random"]
data = [d[f"{k}_r"] for k in keys]
colors = [C_OTHER, C_OTHER, C_BLOCK, C_OTHER, C_OTHER, C_BAD]
bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.0),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.45)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.8)
rng = np.random.default_rng(7)
for i, (vals, c) in enumerate(zip(data, colors), start=1):
    x = i + rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(x, vals, s=6, color=c, edgecolor="none", zorder=3, alpha=0.9)
ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
ax.set_xticks(range(1, 7), labels, fontsize=7, rotation=30, ha="right",
              rotation_mode="anchor")
ax.set_ylabel("Pearson $r$")
ax.set_ylim(-0.35, 0.80)
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "a", dx=-0.22)

# (b) observed vs predicted (block kernel, mean of 10 repeats)
ax = axes[1]
preds = d["block_preds"]
mean_pred = preds.mean(axis=0)
sd_pred = preds.std(axis=0)
r_mean = float(d["block_r"].mean())
mae_mean = float(d["block_mae"].mean())
ax.errorbar(y, mean_pred, yerr=sd_pred, fmt="o", ms=3,
            color=C_BLOCK, ecolor="#9ecae1", elinewidth=0.6, capsize=1.2,
            markeredgecolor="white", markeredgewidth=0.3, linestyle="none")
lims = [min(y.min(), mean_pred.min()) - 8, max(y.max(), mean_pred.max()) + 8]
ax.plot(lims, lims, "--", color="black", linewidth=0.8)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Observed DSDS50 (d)")
ax.set_ylabel("Predicted DSDS50 (d)")
ax.text(0.05, 0.95, f"$r$ = {r_mean:.3f}\nMAE = {mae_mean:.2f} d",
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", linewidth=0.6))
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "b", dx=-0.20)
fig.tight_layout(pad=0.4)
save(fig, "fig1")

# ---------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(figsize=(3.94, 2.55))
w = d["abl_weights"]
rm = d["abl_r_mean"]
rs = d["abl_r_sd"]
block_r = float(d["block_r"].mean())
block_sd = float(d["block_r"].std())
ax.axhspan(block_r - block_sd, block_r + block_sd, color=C_BLOCK, alpha=0.10,
           linewidth=0)
ax.axhline(block_r, color=C_BLOCK, linestyle="--", linewidth=1.0)
ax.errorbar(w, rm, yerr=rs, fmt="o-", color="#333333", ms=3.5,
            capsize=2.5, elinewidth=0.8, markeredgecolor="white",
            markeredgewidth=0.4)
i_best = int(np.argmin(np.abs(w - 0.1)))
ax.scatter([w[i_best]], [rm[i_best]], s=45, facecolor="none",
           edgecolor=C_BAD, linewidth=1.2, zorder=4)
ax.annotate("selected $w$ = 0.1", xy=(w[i_best], rm[i_best]),
            xytext=(0.32, 0.7005), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.7))
ax.text(0.97, block_r + 0.0012, "pure block kernel", ha="right", va="bottom",
        fontsize=7.5, color=C_BLOCK)
ax.set_xlabel("SNP kernel weight $w$")
ax.set_ylabel("Pearson $r$")
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(0.685, 0.730)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4)
save(fig, "fig2")

# ---------------------------------------------------------------- Fig 3
fig, axes = plt.subplots(1, 2, figsize=(4.72, 2.35),
                         gridspec_kw={"wspace": 0.32})
ks = [6, 8, 10]
g_r = [-0.16038218936478454, -0.2357748224338594, -0.04857657333609706]
g_mae = [31.157334310411535, 32.561340823895904, 31.341652678234986]
within_r = float(d["block_r"].mean())
within_mae = float(d["block_mae"].mean())

ax = axes[0]
bars = ax.bar(range(3), g_r, width=0.55, color=C_BAD, alpha=0.75,
              edgecolor="black", linewidth=0.6)
ax.axhline(within_r, color=C_BLOCK, linestyle="--", linewidth=1.0)
ax.text(2.45, within_r + 0.025, f"within-population CV\n($r$ = {within_r:.3f})",
        ha="right", va="bottom", fontsize=7, color=C_BLOCK)
ax.axhline(0, color="black", linewidth=0.7)
for i, v in enumerate(g_r):
    ax.text(i, v - 0.035, f"{v:.3f}", ha="center", va="top", fontsize=7)
ax.set_xticks(range(3), [f"$k$ = {k}" for k in ks])
ax.set_ylabel("Pearson $r$")
ax.set_ylim(-0.35, 0.90)
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "a", dx=-0.18)

ax = axes[1]
bars = ax.bar(range(3), g_mae, width=0.55, color=C_BAR, alpha=0.85,
              edgecolor="black", linewidth=0.6)
ax.axhline(within_mae, color=C_BLOCK, linestyle="--", linewidth=1.0)
ax.text(0.98, 0.97, f"within-population CV\n(MAE = {within_mae:.2f} d)",
        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=C_BLOCK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="none", alpha=0.9))
for i, v in enumerate(g_mae):
    ax.text(i, v - 1.2, f"{v:.2f}", ha="center", va="top", fontsize=7,
            color="white", fontweight="bold")
ax.set_xticks(range(3), [f"$k$ = {k}" for k in ks])
ax.set_ylabel("MAE (d)")
ax.set_ylim(0, 40)
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "b", dx=-0.18)
fig.tight_layout(pad=0.4)
save(fig, "fig3")

print("All figures written to", OUT)