"""
Regenerates the two-panel carbon-efficiency figure (risk-cost frontier +
energy-reduction heatmap) using the UPDATED cagt_carbon_estimate.csv
(5-value communication-energy sweep: 5/20/50/100/200 mJ, replacing the
earlier 3-value 10/50/100 mJ sweep). The frontier panel is unaffected by
this change (it doesn't depend on the CAGT assumptions at all) and is
included unchanged for a single combined figure.

The nominal scenario (3W, 20mJ -- the LoRaWAN Sensors 2021 representative
transmission figure) is outlined on the heatmap so it reads as the
headline number, with the full swept range shown around it rather than
hidden.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
EDA_DIR = ROOT_DIR / "Results"
FIG_DIR = EDA_DIR / "Figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

COLOR_R0 = "#4C72B0"
COLOR_R1 = "#C44E52"
COLOR_GATE = "#55A868"

NOMINAL_POWER_W = 3.0
NOMINAL_COMM_MJ = 20.0


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1 = axes[0]
    frontier_path = EDA_DIR / "aies_frontier.csv"
    if frontier_path.exists():
        d = pd.read_csv(frontier_path)
        ax1.plot(d["injection_rate"], d["macro_f1"], "-o", color=COLOR_GATE, ms=4, lw=1.5)
        pure_r1 = d.loc[d["injection_rate"].idxmax()]
        pure_r0 = d.loc[d["injection_rate"].idxmin()]
        ax1.scatter([pure_r1["injection_rate"]], [pure_r1["macro_f1"]], color=COLOR_R1, s=60, zorder=5)
        ax1.scatter([pure_r0["injection_rate"]], [pure_r0["macro_f1"]], color=COLOR_R0, s=60, zorder=5)
        chosen = d.iloc[(d["tau"] - 0.4).abs().idxmin()]
        ax1.scatter([chosen["injection_rate"]], [chosen["macro_f1"]], facecolors="none",
                    edgecolors="black", s=140, lw=1.5, zorder=6)
        ax1.annotate(f"$\\tau$=0.4\n{chosen['injection_rate']*100:.0f}% injection",
                     (chosen["injection_rate"], chosen["macro_f1"]),
                     xytext=(30, 20), textcoords="offset points", fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.7))
    else:
        ax1.text(0.5, 0.5, "aies_frontier.csv not found", ha="center", va="center")
    ax1.set_xlabel("Injection rate")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Risk\u2013cost frontier")

    ax2 = axes[1]
    carbon_path = EDA_DIR / "cagt_carbon_estimate.csv"
    d2 = pd.read_csv(carbon_path)
    pivot = d2.pivot(index="comm_energy_mj", columns="edge_power_w",
                      values="pct_reduction_aies_vs_always_r1").sort_index(ascending=False)

    im = ax2.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax2.set_xlabel("Assumed edge power (W)")
    ax2.set_ylabel("Assumed comm. energy/query (mJ)")
    ax2.set_title("Estimated energy reduction (%)\nAIES vs. always-on injection")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            is_nominal = (pivot.columns[j] == NOMINAL_POWER_W and pivot.index[i] == NOMINAL_COMM_MJ)
            ax2.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8.5,
                     color="black" if val < 60 else "white",
                     fontweight="bold" if is_nominal else "normal")
            if is_nominal:
                ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                             edgecolor="black", lw=2.2))

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIG_DIR / "fig3_carbon_efficiency_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    nominal_row = d2[(d2["edge_power_w"] == NOMINAL_POWER_W) & (d2["comm_energy_mj"] == NOMINAL_COMM_MJ)]
    print(f"[ok] {out}")
    if not nominal_row.empty:
        print(f"Nominal (outlined) cell: {nominal_row.iloc[0]['pct_reduction_aies_vs_always_r1']:.1f}% reduction")
    print(f"Full range: {d2['pct_reduction_aies_vs_always_r1'].min():.1f}% - "
          f"{d2['pct_reduction_aies_vs_always_r1'].max():.1f}%")

if __name__ == "__main__":
    main()