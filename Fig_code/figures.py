"""
Generates the figure set for the paper from the CSVs already produced by
earlier scripts. Run this locally, after all prior experiments -- it reads
files, does not retrain or recompute anything, so it's fast.

Reads (skips gracefully with a message if a file is missing):
  eda_output/aies_frontier.csv               -> fig1_risk_cost_frontier.png
  eda_output/gating_baseline_comparison.csv   -> fig2_gating_baseline_comparison.png
  eda_output/cagt_carbon_estimate.csv         -> fig3_carbon_reduction_heatmap.png
  eda_output/lightweight_model_comparison.csv -> fig4_model_weight_tradeoff.png
  eda_output/labeled_dataset.csv              -> fig5_label_distribution.png

Writes all figures to eda_output/figures/, at 300 DPI, sized for IEEE
single-column width (3.5in) where reasonable.
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

REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]
REGION_LABELS = ["Severe\ndown", "Moderate\ndown", "Stable", "Moderate\nup", "Severe\nup"]

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLOR_R0 = "#4C72B0"
COLOR_R1 = "#C44E52"
COLOR_GATE = "#55A868"
COLOR_RANDOM = "#8172B2"
COLOR_HEURISTIC = "#CCB974"


def fig1_frontier():
    path = EDA_DIR / "aies_frontier.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(df["injection_rate"], df["macro_f1"], "-o", color=COLOR_GATE,
            markersize=4, linewidth=1.5, label="AIES gate (sweep over $\\tau$)")

    pure_r1 = df.loc[df["injection_rate"].idxmax()]
    pure_r0 = df.loc[df["injection_rate"].idxmin()]
    ax.scatter([pure_r1["injection_rate"]], [pure_r1["macro_f1"]], color=COLOR_R1,
               s=50, zorder=5, label="Pure r1 (always inject)")
    ax.scatter([pure_r0["injection_rate"]], [pure_r0["macro_f1"]], color=COLOR_R0,
               s=50, zorder=5, label="Pure r0 (never inject)")

    chosen = df.iloc[(df["tau"] - 0.4).abs().idxmin()]
    ax.scatter([chosen["injection_rate"]], [chosen["macro_f1"]], facecolors="none",
               edgecolors="black", s=120, linewidth=1.5, zorder=6,
               label=f"Chosen operating point\n($\\tau$={chosen['tau']:.1f}, "
                     f"{chosen['injection_rate']*100:.0f}% injection)")

    ax.set_xlabel("Injection rate (fraction of inputs routed to r1)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Risk\u2013cost frontier: quality vs. injection rate")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig1_risk_cost_frontier.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok] {out}")


def fig2_gating_comparison():
    path = EDA_DIR / "gating_baseline_comparison.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path).set_index("policy")

    order = ["pure_r0", "pure_r1", "random_gating_mean", "heuristic_threshold", "learned_kus_gate"]
    labels = ["Pure r0", "Pure r1", "Random\ngating", "Heuristic\ngating", "Learned\n(KUS) gate"]
    colors = [COLOR_R0, COLOR_R1, COLOR_RANDOM, COLOR_HEURISTIC, COLOR_GATE]
    df = df.loc[order]

    metrics = [("macro_f1", "Macro-F1"), ("severe_recall", "Severe-class recall"),
               ("false_safe_rate", "False-safe rate\n(lower is better)")]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(labels, df[col], color=colors)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=6.5, rotation=0)
        ax.set_ylim(0, max(df[col]) * 1.25)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(df[col]) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle("Gating strategy comparison at matched injection rate (~36.6%)", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_DIR / "fig2_gating_baseline_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok] {out}")


def fig3_carbon_heatmap():
    path = EDA_DIR / "cagt_carbon_estimate.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    pivot = df.pivot(index="comm_energy_mj", columns="edge_power_w",
                      values="pct_reduction_aies_vs_always_r1")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("Assumed edge inference power (W)")
    ax.set_ylabel("Assumed comm. energy\nper query (mJ)")
    ax.set_title("Estimated energy reduction (%)\nAIES vs. always-on injection", fontsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color="black" if val < 60 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    out = FIG_DIR / "fig3_carbon_reduction_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok] {out}")


def fig4_model_tradeoff():
    path = EDA_DIR / "lightweight_model_comparison.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)
    short_labels = ["Heavy\n(300 trees)", "Medium\n(50 trees)", "Light\n(15 trees)"]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width / 2, df["latency_r0_ms"], width, color=COLOR_R0, label="Latency (ms)")
    ax1.set_ylabel("Single-row inference latency (ms)", color=COLOR_R0, fontsize=8)
    ax1.tick_params(axis="y", labelcolor=COLOR_R0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=7)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, df["severe_recall_r0"], width, color=COLOR_GATE,
            label="Severe recall (r0)")
    ax2.set_ylabel("Severe-class recall", color=COLOR_GATE, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_GATE)
    ax2.set_ylim(0, 1)

    ax1.set_title("Model weight vs. latency and\nsafety-critical recall", fontsize=8.5)
    fig.tight_layout()
    out = FIG_DIR / "fig4_model_weight_tradeoff.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok] {out}")


def fig5_label_distribution():
    path = EDA_DIR / "labeled_dataset.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path, usecols=["region"])
    counts = df["region"].value_counts().reindex(REGION_ORDER).fillna(0)
    pct = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colors = ["#C44E52", "#DD8452", "#4C72B0", "#8172B2", "#55A868"]
    bars = ax.bar(REGION_LABELS, counts, color=colors)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Number of labeled rows")
    ax.set_title("Ordinal ramp-region label distribution\n(all turbines, full year)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    out = FIG_DIR / "fig5_label_distribution.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok] {out}")


def main():
    print("Generating figures...\n")
    fig1_frontier()
    fig2_gating_comparison()
    fig3_carbon_heatmap()
    fig4_model_tradeoff()
    fig5_label_distribution()
    print(f"\nAll figures written to {FIG_DIR.resolve()}")

if __name__ == "__main__":
    main()