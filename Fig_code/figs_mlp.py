"""
Figure C: two panels.
  Left  - r0 vs r1 (neighbor-available subset, n=40,024) and r0 vs r2
          (full test set, n=92,934), shown as two separate labeled groups
          since they're different evaluation populations with different r0
          baselines -- NOT merged into one misleading comparison.
  Right - RF vs MLP, r0 vs r1, macro-F1 and severe recall, showing whether
          the knowledge-injection trade-off pattern replicates across
          model architectures.

Reads r2_physics_test_predictions.csv (produced by 20_physics_residual_r2.py).
The RF and MLP reference numbers are hardcoded from the already-confirmed
console outputs (both final, reproducible results).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
EDA_DIR = ROOT_DIR / "Results"
FIG_DIR = EDA_DIR / "Figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

COLOR_R0 = "#4C72B0"
COLOR_R1 = "#C44E52"
COLOR_R2 = "#55A868"
COLOR_RF = "#4C72B0"
COLOR_MLP = "#DD8452"


def metrics_for(y_true, pred):
    macro_f1 = f1_score(y_true, pred, average="macro")
    is_severe = y_true.isin(["severe_down", "severe_up"])
    pred_series = pd.Series(pred, index=y_true.index)
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred_series == y_true)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & (pred_series == "stable")).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


def main():
    r2_path = EDA_DIR / "r2_physics_test_predictions.csv"
    r2_df = pd.read_csv(r2_path)
    f1_r0_full, rec_r0_full, fs_r0_full = metrics_for(r2_df["region"], r2_df["r0_pred"])
    f1_r2, rec_r2, fs_r2 = metrics_for(r2_df["region"], r2_df["r2_pred"])

    f1_r0_sub, rec_r0_sub, fs_r0_sub = 0.2795, 0.7005, 0.0325
    f1_r1, rec_r1, fs_r1 = 0.2727, 0.7513, 0.0307

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    groups = [
        ("r0\n(n-subset)", f1_r0_sub, rec_r0_sub, COLOR_R0),
        ("r1\n(n-subset)", f1_r1, rec_r1, COLOR_R1),
        ("r0\n(full test)", f1_r0_full, rec_r0_full, COLOR_R0),
        ("r2\n(full test)", f1_r2, rec_r2, COLOR_R2),
    ]
    x = np.arange(len(groups))
    w = 0.35
    f1_vals = [g[1] for g in groups]
    rec_vals = [g[2] for g in groups]
    colors = [g[3] for g in groups]

    ax.bar(x - w / 2, f1_vals, w, color=colors, alpha=0.95, label="Macro-F1")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, rec_vals, w, color=colors, alpha=0.5, hatch="//", label="Severe recall")

    for xi, v in zip(x - w / 2, f1_vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=6)
    for xi, v in zip(x + w / 2, rec_vals):
        ax2.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
    ax.set_ylabel("Macro-F1 (solid)", fontsize=7.5)
    ax2.set_ylabel("Severe recall (hatched)", fontsize=7.5)
    ax.set_ylim(0, 0.4)
    ax2.set_ylim(0, 1.0)
    ax.axvline(1.5, color="gray", lw=0.6, linestyle=":")
    ax.set_title("Knowledge routes: r1 (graph path, comm. cost)\n"
                 "vs. r2 (physics path, no comm. cost)\n"
                 "-- note different test populations", fontsize=7.5)

    ax3 = axes[1]
    labels = ["RF\nr0", "RF\nr1", "MLP\nr0", "MLP\nr1"]
    f1s = [0.2795, 0.2727, 0.3104, 0.3041]
    recs = [0.7005, 0.7513, 0.5475, 0.6344]
    colors2 = [COLOR_RF, COLOR_RF, COLOR_MLP, COLOR_MLP]

    x3 = np.arange(4)
    ax3.bar(x3 - w / 2, f1s, w, color=colors2, alpha=0.95)
    ax4 = ax3.twinx()
    ax4.bar(x3 + w / 2, recs, w, color=colors2, alpha=0.5, hatch="//")

    for xi, v in zip(x3 - w / 2, f1s):
        ax3.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=6)
    for xi, v in zip(x3 + w / 2, recs):
        ax4.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=6)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels, fontsize=6.5)
    ax3.set_ylabel("Macro-F1 (solid)", fontsize=7.5)
    ax4.set_ylabel("Severe recall (hatched)", fontsize=7.5)
    ax3.set_ylim(0, 0.4)
    ax4.set_ylim(0, 1.0)
    ax3.axvline(1.5, color="gray", lw=0.6, linestyle=":")
    ax3.set_title("Does the r0\u2192r1 trade-off replicate\nacross architectures? (RF vs. small MLP)\n"
                  "MLP trained on class-matched subsample", fontsize=7.5)

    fig.suptitle("Figure C \u2014 Additional knowledge routes and cross-architecture check", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIG_DIR / "figureC_extra_experiments.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")

if __name__ == "__main__":
    main()