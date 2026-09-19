"""
Two comprehensive multi-panel figures, built from full_test_results.csv
plus the summary CSVs already produced. Run AFTER 18_generate_full_results.py.

Figure A "System behavior": spatial wake structure + time series of true vs
predicted region with injection decisions + confusion matrix + cross-site
portability + gating-strategy comparison, all as one composed figure.

Figure B "Carbon/efficiency story": risk-cost frontier + carbon heatmap +
model-weight tradeoff + a headline-numbers panel, as one composed figure.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
EDA_DIR = ROOT_DIR / "Results"
FIG_DIR = EDA_DIR / "Figures"
FIG_DIR.mkdir(exist_ok=True)

REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]
REGION_SHORT = ["Sev.\ndown", "Mod.\ndown", "Stable", "Mod.\nup", "Sev.\nup"]
REGION_COLORS = ["#C44E52", "#DD8452", "#4C72B0", "#8172B2", "#55A868"]
REGION_CMAP = ListedColormap(REGION_COLORS)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

COLOR_R0 = "#4C72B0"
COLOR_R1 = "#C44E52"
COLOR_GATE = "#55A868"
COLOR_RANDOM = "#8172B2"
COLOR_HEURISTIC = "#CCB974"


def load_static_metadata():
    frames = []
    for site_dir, fname in [("Kelmarsh", "Kelmarsh_WT_static.csv"),
                             ("Penmanshiel", "Penmanshiel_WT_static.csv")]:
        path = DATA_DIR / site_dir / fname
        if not path.exists():
            continue
        d = pd.read_csv(path, encoding="utf-8-sig")
        d = d.dropna(subset=["Title"])
        d["turbine_id"] = d["Title"].str.extract(r"(\d+)$").astype(int).astype(str)
        d["site"] = site_dir
        frames.append(d[["site", "turbine_id", "Latitude", "Longitude"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def local_xy(lat, lon, lat0, lon0):
    R = 6371000.0
    x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(lat - lat0)
    return x, y


def region_to_int(series):
    m = {name: i for i, name in enumerate(REGION_ORDER)}
    return series.map(m)


def build_figure_a(df, meta):
    fig = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.1, 1.0, 1.0],
                            hspace=0.55, wspace=0.55)

    gs_row0 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0, :], width_ratios=[1, 1, 2.1], wspace=0.45)

    present_turbines = df.groupby(["site", "turbine_id"])["neighbor_available"].mean().reset_index()
    site_axes = {"Kelmarsh": fig.add_subplot(gs_row0[0]), "Penmanshiel": fig.add_subplot(gs_row0[1])}
    for site, ax in site_axes.items():
        m_site = meta[meta["site"] == site]
        av_site = present_turbines[present_turbines["site"] == site].set_index("turbine_id")
        if m_site.empty:
            ax.set_visible(False)
            continue
        lat0, lon0 = m_site["Latitude"].mean(), m_site["Longitude"].mean()
        x, y = local_xy(m_site["Latitude"].values, m_site["Longitude"].values, lat0, lon0)
        avail = m_site["turbine_id"].map(av_site["neighbor_available"]).fillna(0).values * 100

        ax.scatter(x, y, color="#4C72B0", s=170, edgecolors="black", linewidths=0.6, zorder=3)
        for xi, yi, tid, av in zip(x, y, m_site["turbine_id"], avail):
            ax.annotate(f"{tid}\n{av:.0f}%", (xi, yi), ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold", zorder=4)

        wd = df.loc[df["site"] == site, "Wind direction (\u00b0)"]
        rad = np.deg2rad(wd.dropna())
        mean_dir = (np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360) % 360
        n = len(x)
        for a in range(n):
            best_j, best_d = None, np.inf
            for b in range(n):
                if a == b:
                    continue
                bearing = (np.degrees(np.arctan2(x[b] - x[a], y[b] - y[a]))) % 360
                diff = abs((bearing - mean_dir + 180) % 360 - 180)
                dist = np.hypot(x[b] - x[a], y[b] - y[a])
                if diff <= 30 and dist < best_d:
                    best_d, best_j = dist, b
            if best_j is not None:
                ax.annotate("", xy=(x[best_j], y[best_j]), xytext=(x[a], y[a]),
                            arrowprops=dict(arrowstyle="-|>", color="gray", alpha=0.55, lw=1),
                            zorder=2)

        ax.set_title(f"{site} (wind {mean_dir:.0f}\u00b0)\nlabel = turbine ID / neighbor avail. %",
                     fontsize=7)
        ax.set_xlabel("East (m)", fontsize=6.5)
        ax.set_ylabel("North (m)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")

    ax_ts_slot = gs_row0[2]
    gs_ts = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=ax_ts_slot, height_ratios=[3.0, 0.45, 0.45, 0.35], hspace=0.12)
    ax_main = fig.add_subplot(gs_ts[0])
    ax_true = fig.add_subplot(gs_ts[1], sharex=ax_main)
    ax_pred = fig.add_subplot(gs_ts[2], sharex=ax_main)
    ax_inj = fig.add_subplot(gs_ts[3], sharex=ax_main)

    df_sorted = df.sort_values("timestamp")
    is_severe = df_sorted["region"].isin(["severe_down", "severe_up"])
    if is_severe.any():
        anchor_row = df_sorted[is_severe].iloc[len(df_sorted[is_severe]) // 2]
    else:
        anchor_row = df_sorted.iloc[len(df_sorted) // 2]
    site_sel, tid_sel = anchor_row["site"], anchor_row["turbine_id"]
    window = df_sorted[(df_sorted["site"] == site_sel) & (df_sorted["turbine_id"] == tid_sel)]
    center = anchor_row["timestamp"]
    window = window[(window["timestamp"] >= center - pd.Timedelta(days=2)) &
                     (window["timestamp"] <= center + pd.Timedelta(days=2))]

    ax_main.plot(window["timestamp"], window["own_power_norm"], color="black", lw=0.9, zorder=3)
    ax_main.set_ylabel("Norm.\npower", fontsize=6.5)
    ax_main.set_title(f"{site_sel} T{tid_sel}: power, true vs. predicted region, injection",
                       fontsize=7.5)
    ax_main.tick_params(labelsize=6, labelbottom=False)

    true_int = region_to_int(window["region"]).values.reshape(1, -1)
    pred_int = region_to_int(pd.Series(window["final_pred"].values)).values.reshape(1, -1)
    t0, t1 = window["timestamp"].min(), window["timestamp"].max()
    extent = [matplotlib.dates.date2num(t0), matplotlib.dates.date2num(t1), 0, 1]

    ax_true.imshow(true_int, aspect="auto", cmap=REGION_CMAP, vmin=0, vmax=4, extent=extent)
    ax_true.set_yticks([]); ax_true.tick_params(labelbottom=False, bottom=False)
    ax_true.set_ylabel("True", fontsize=6, rotation=0, ha="right", va="center")

    ax_pred.imshow(pred_int, aspect="auto", cmap=REGION_CMAP, vmin=0, vmax=4, extent=extent)
    ax_pred.set_yticks([]); ax_pred.tick_params(labelbottom=False, bottom=False)
    ax_pred.set_ylabel("Pred.", fontsize=6, rotation=0, ha="right", va="center")

    inj = window["injected"].fillna(False).values.astype(int).reshape(1, -1)
    ax_inj.imshow(inj, aspect="auto", cmap=ListedColormap(["#EEEEEE", "black"]),
                  vmin=0, vmax=1, extent=extent)
    ax_inj.set_yticks([])
    ax_inj.set_ylabel("Inject", fontsize=6, rotation=0, ha="right", va="center")
    ax_inj.xaxis_date()
    ax_inj.tick_params(labelsize=5.5)

    ax_cm = fig.add_subplot(gs[1, 0])
    valid_pred = df["final_pred"].notna() & df["region"].notna()
    cm = confusion_matrix(df.loc[valid_pred, "region"], df.loc[valid_pred, "final_pred"],
                           labels=REGION_ORDER, normalize="true")
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(5)); ax_cm.set_xticklabels(REGION_SHORT, fontsize=5.5, rotation=45, ha="right")
    ax_cm.set_yticks(range(5)); ax_cm.set_yticklabels(REGION_SHORT, fontsize=5.5)
    ax_cm.set_xlabel("Predicted", fontsize=6.5)
    ax_cm.set_ylabel("True", fontsize=6.5)
    ax_cm.set_title("Gate confusion matrix\n(row-normalized)", fontsize=7.5)
    for i in range(5):
        for j in range(5):
            ax_cm.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=5,
                       color="white" if cm[i, j] > 0.5 else "black")

    ax_cs = fig.add_subplot(gs[1, 1])
    cross_path = EDA_DIR / "cross_site_check.csv"
    if cross_path.exists():
        cross = pd.read_csv(cross_path).set_index("policy")
        within_rec = None
        gate_path = EDA_DIR / "gating_baseline_comparison.csv"
        if gate_path.exists():
            gb = pd.read_csv(gate_path).set_index("policy")
            within_rec = gb.loc["learned_kus_gate", "severe_recall"]
        labels = ["Within-site\n(gate)", "Cross-site\nK\u2192P (gate)"]
        vals = [within_rec if within_rec is not None else np.nan,
                cross.loc["gate_tau04", "severe_recall"]]
        bars = ax_cs.bar(labels, vals, color=[COLOR_GATE, "#937860"])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax_cs.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                           ha="center", fontsize=6.5)
        ax_cs.set_ylim(0, 1)
        ax_cs.set_ylabel("Severe-class recall", fontsize=6.5)
        ax_cs.set_title("Portability: within- vs.\ncross-site (Kelmarsh\u2192Penm.)", fontsize=7.5)
        ax_cs.tick_params(labelsize=6)
    else:
        ax_cs.text(0.5, 0.5, "cross_site_check.csv\nnot found", ha="center", va="center", fontsize=7)
        ax_cs.set_axis_off()

    ax_g = fig.add_subplot(gs[1, 2:4])
    gate_path = EDA_DIR / "gating_baseline_comparison.csv"
    if gate_path.exists():
        gb = pd.read_csv(gate_path).set_index("policy")
        order = ["pure_r0", "pure_r1", "random_gating_mean", "heuristic_threshold", "learned_kus_gate"]
        labels = ["Pure\nr0", "Pure\nr1", "Random", "Heuristic", "Learned\n(KUS)"]
        colors = [COLOR_R0, COLOR_R1, COLOR_RANDOM, COLOR_HEURISTIC, COLOR_GATE]
        gb = gb.loc[order]
        x = np.arange(5)
        w = 0.35
        ax_g.bar(x - w / 2, gb["severe_recall"], w, color=colors, alpha=0.95)
        ax_g.set_ylabel("Severe-class recall", fontsize=6.5)
        ax_g.set_ylim(0, 1)
        ax_g2 = ax_g.twinx()
        ax_g2.bar(x + w / 2, gb["false_safe_rate"], w, color=colors, alpha=0.5, hatch="//")
        ax_g2.set_ylabel("False-safe rate", fontsize=6.5)
        ax_g.set_xticks(x); ax_g.set_xticklabels(labels, fontsize=6.5)
        ax_g.set_title("Gating strategies: severe recall (solid) vs. false-safe rate (hatched)",
                       fontsize=7.5)
        ax_g.tick_params(labelsize=6)
        ax_g2.tick_params(labelsize=6)
    else:
        ax_g.set_axis_off()

    fig.suptitle("Figure A \u2014 CAKI system behavior: spatial wake structure, "
                 "prediction accuracy, and gating strategy", fontsize=10, y=0.995)
    out = FIG_DIR / "figureA_system_behavior.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def build_figure_b():
    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    path = EDA_DIR / "aies_frontier.csv"
    if path.exists():
        d = pd.read_csv(path)
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
                     xytext=(30, 20), textcoords="offset points", fontsize=6.5,
                     arrowprops=dict(arrowstyle="->", lw=0.7))
    ax1.set_xlabel("Injection rate", fontsize=7.5)
    ax1.set_ylabel("Macro-F1", fontsize=7.5)
    ax1.set_title("Risk\u2013cost frontier", fontsize=8.5)
    ax1.tick_params(labelsize=6.5)

    ax2 = fig.add_subplot(gs[0, 1])
    path = EDA_DIR / "cagt_carbon_estimate.csv"
    if path.exists():
        d = pd.read_csv(path)
        pivot = d.pivot(index="comm_energy_mj", columns="edge_power_w",
                         values="pct_reduction_aies_vs_always_r1").sort_index(ascending=False)
        im = ax2.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax2.set_xticks(range(len(pivot.columns))); ax2.set_xticklabels([f"{v:g}" for v in pivot.columns])
        ax2.set_yticks(range(len(pivot.index))); ax2.set_yticklabels([f"{v:g}" for v in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax2.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.5,
                         color="black" if v < 60 else "white")
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
    ax2.set_xlabel("Edge power (W)", fontsize=7.5)
    ax2.set_ylabel("Comm. energy/query (mJ)", fontsize=7.5)
    ax2.set_title("Estimated energy reduction, AIES vs. always-on", fontsize=8.5)
    ax2.tick_params(labelsize=6.5)

    ax3 = fig.add_subplot(gs[1, 0])
    path = EDA_DIR / "lightweight_model_comparison.csv"
    if path.exists():
        d = pd.read_csv(path)
        x = np.arange(len(d))
        w = 0.35
        ax3.bar(x - w / 2, d["latency_r0_ms"], w, color=COLOR_R0)
        ax3.set_ylabel("Latency (ms)", color=COLOR_R0, fontsize=7.5)
        ax3.tick_params(axis="y", labelcolor=COLOR_R0, labelsize=6.5)
        ax3b = ax3.twinx()
        ax3b.bar(x + w / 2, d["severe_recall_r0"], w, color=COLOR_GATE)
        ax3b.set_ylabel("Severe recall", color=COLOR_GATE, fontsize=7.5)
        ax3b.tick_params(axis="y", labelcolor=COLOR_GATE, labelsize=6.5)
        ax3b.set_ylim(0, 1)
        ax3.set_xticks(x)
        ax3.set_xticklabels(["Heavy\n(300)", "Medium\n(50)", "Light\n(15)"], fontsize=6.5)
    ax3.set_title("Model weight vs. latency and recall", fontsize=8.5)

    ax4 = fig.add_subplot(gs[1, 1])
    path = EDA_DIR / "cagt_carbon_estimate.csv"
    if path.exists():
        d = pd.read_csv(path)
        row = d[(d["edge_power_w"] == 3.0) & (d["comm_energy_mj"] == 50.0)]
        if not row.empty:
            row = row.iloc[0]
            labels = ["Always\nr0", "Always\nr1", "AIES\n(\u03c4=0.4)"]
            vals = [row["total_always_r0_gco2"], row["total_always_r1_gco2"], row["total_aies_gco2"]]
            colors = [COLOR_R0, COLOR_R1, COLOR_GATE]
            bars = ax4.bar(labels, vals, color=colors)
            for b, v in zip(bars, vals):
                ax4.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}",
                          ha="center", fontsize=7)
            ax4.set_ylabel("Estimated gCO$_2$e\n(test period, all turbines)", fontsize=7)
            ax4.set_title("Absolute estimated emissions\n(edge power=3W, comm=50mJ)", fontsize=8.5)
            ax4.tick_params(labelsize=7)
    else:
        ax4.set_axis_off()

    fig.suptitle("Figure B \u2014 Carbon-aware adaptive injection: cost, quality, and efficiency",
                 fontsize=10, y=0.99)
    out = FIG_DIR / "figureB_carbon_efficiency.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def main():
    results_path = EDA_DIR / "full_test_results.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found -- run 18_generate_full_results.py first.")
        return
    df = pd.read_csv(results_path, parse_dates=["timestamp"])
    df["turbine_id"] = df["turbine_id"].astype(str)
    meta = load_static_metadata()

    print("Building Figure A...")
    build_figure_a(df, meta)
    print("Building Figure B...")
    build_figure_b()
    print(f"\nDone. Figures in {FIG_DIR.resolve()}")

if __name__ == "__main__":
    main()