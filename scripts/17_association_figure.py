"""Publication figure: association QQ plots and Manhattan plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

res = pd.read_csv(TABLES / "block_scan_imhof.csv")
n_blocks = len(res)
bonf = 0.05 / n_blocks


def panel_label(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=170,
                         gridspec_kw={"width_ratios": [1, 1.5]})

# ---- panel a: QQ ----
ax = axes[0]
for col, color, label in [
    ("p_imhof", "#2166ac", "raw block (primary)"),
    ("p_graph_imhof", "#878787", "graph neighbourhood"),
    ("p_emb_imhof", "#b2182b", "SSL embedding"),
]:
    p = res[col].dropna().to_numpy()
    p = np.clip(p, 1e-300, 1.0)
    m = len(p)
    expected = -np.log10((np.arange(1, m + 1) - 0.5) / m)
    observed = -np.log10(np.sort(p))
    ax.plot(expected, observed, color=color, linewidth=1.1, label=label)
lim = max(3.0, float(np.max(-np.log10(np.clip(res["p_imhof"].dropna(), 1e-300, 1)))))
ax.plot([0, lim], [0, lim], "--", color="#999999", linewidth=0.8)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim * 1.05)
ax.set_xlabel(r"Expected $-\log_{10}(P)$")
ax.set_ylabel(r"Observed $-\log_{10}(P)$")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "a")

# ---- panel b: Manhattan (primary family) ----
ax = axes[1]
colors = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 4: "#c44e52", 5: "#8172b3"}
offsets = {}
off = 0.0
ticks, ticklabels = [], []
for c in range(1, 6):
    sub = res[res["chrom"] == c]
    span = float(sub["end"].max()) / 1e6
    offsets[c] = off
    ticks.append(off + span / 2)
    ticklabels.append(str(c))
    off += span + 2.0
ok = res["p_imhof"].notna()
x = res.loc[ok, "start"] / 1e6 + res.loc[ok, "chrom"].map(offsets)
y = -np.log10(res.loc[ok, "p_imhof"].clip(lower=1e-300))
for c in range(1, 6):
    m = ok & (res["chrom"] == c)
    ax.scatter(res.loc[m, "start"] / 1e6 + offsets[c],
               -np.log10(res.loc[m, "p_imhof"].clip(lower=1e-300)),
               s=3, color=colors[c], alpha=0.6, linewidths=0)
ax.axhline(-np.log10(bonf), color="#b2182b", linestyle="--", linewidth=0.9)
ax.set_ylim(0, max(-np.log10(bonf) + 0.7, float(y.max()) * 1.15))
ax.text(0.99, -np.log10(bonf) + 0.12, f"FWER 0.05 ({bonf:.1e})",
        transform=ax.get_yaxis_transform(), fontsize=6.5,
        ha="right", va="bottom", color="#b2182b")
# DOG1 region (chr5, 18.589-18.592 Mb)
dog = res[(res["chrom"] == 5) & (res["start"] <= 18_591_506) & (res["end"] >= 18_589_418)]
if len(dog):
    xd = dog["start"].mean() / 1e6 + offsets[5]
    yd = -np.log10(dog["p_imhof"].clip(lower=1e-300)).max()
    ax.annotate("DOG1", xy=(xd, yd), xytext=(xd - 18, yd + 0.9),
                fontsize=7, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7))
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
ax.set_xlabel("Chromosome")
ax.set_ylabel(r"$-\log_{10}(P)$, raw block kernel")
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "b", dx=-0.10)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "fig5.png", dpi=600, facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig5.pdf", facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved fig5.png / fig5.pdf")
print("lambda_gc (median chi2_1 / 0.4549):")
from scipy import stats as st
for col in ["p_imhof", "p_graph_imhof", "p_emb_imhof"]:
    p = res[col].dropna()
    chi = st.chi2.isf(p, 1)
    print(f"  {col}: {np.median(chi) / 0.4549364:.3f}")