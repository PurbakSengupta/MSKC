"""
Sanity-check the ordinal ramp-region labeling scheme visually against real
data, rather than only looking at its aggregate class distribution
(fig5_label_distribution.png).

Panel 1: normalized power for an ILLUSTRATIVE window, background-shaded by
         assigned region.
Panel 2 (shares x-axis with panel 1): the continuous delta_norm_power that
         actually drives the labeling, plotted against its own fixed
         thresholds (+/-0.10, +/-0.30) -- shows the mechanism directly, not
         just the outcome.
Panel 3: histogram of delta_norm_power across ALL turbines and the FULL
         year (not just the illustrative window), with the same threshold
         lines -- an unbiased check of the binning choice.

Window selection is automated, not chosen by eye: searches all turbines
for the 7-day window maximizing min(severe_up_count, severe_down_count)
within it, so the example can't be cherry-picked to show only one ramp
direction. The selected turbine/window and its event counts are printed so
the choice is auditable.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
EDA_DIR = ROOT_DIR / "Results"
FIG_DIR = EDA_DIR / "Figures"
FIG_DIR.mkdir(exist_ok=True)

REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]
REGION_COLORS = {
    "severe_down": "#C44E52", "moderate_down": "#DD8452", "stable": "#4C72B0",
    "moderate_up": "#8172B2", "severe_up": "#55A868",
}
ZOOM_DAYS = 1.5  # padding on each side of the closest up/down event pair, for panels 1 & 2

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def find_best_window(df, window_days=7):
    best = None
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        g = g.sort_values("timestamp").set_index("timestamp")
        is_up = (g["region"] == "severe_up").astype(int)
        is_down = (g["region"] == "severe_down").astype(int)

        roll_up = is_up.rolling(f"{window_days}D").sum()
        roll_down = is_down.rolling(f"{window_days}D").sum()
        combined = pd.DataFrame({"up": roll_up, "down": roll_down})
        combined["min_count"] = combined[["up", "down"]].min(axis=1)
        combined["total"] = combined["up"] + combined["down"]

        idx = combined.sort_values(["min_count", "total"], ascending=False).index
        if len(idx) == 0:
            continue
        top_ts = idx[0]
        row = combined.loc[top_ts]
        candidate = {
            "site": site, "turbine_id": tid, "window_end": top_ts,
            "min_count": row["min_count"], "total": row["total"],
            "up_count": row["up"], "down_count": row["down"],
        }
        if best is None or candidate["min_count"] > best["min_count"] or \
           (candidate["min_count"] == best["min_count"] and candidate["total"] > best["total"]):
            best = candidate
    return best


def find_closest_pair(win):
    """Within the selected window, find the closest-in-time severe_up /
    severe_down pair, so the zoomed plot can center tightly on a moment
    that genuinely shows both ramp directions near each other -- still
    determined by the data, not chosen by eye."""
    up_times = win.loc[win["region"] == "severe_up", "timestamp"].tolist()
    down_times = win.loc[win["region"] == "severe_down", "timestamp"].tolist()
    if not up_times or not down_times:
        return None
    best_pair, best_diff = None, None
    for ut in up_times:
        for dt in down_times:
            diff = abs((ut - dt).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff, best_pair = diff, (ut, dt)
    return best_pair


def main():
    print("Loading labeled dataset...")
    df = pd.read_csv(EDA_DIR / "labeled_dataset.csv", parse_dates=["timestamp"])
    df["turbine_id"] = df["turbine_id"].astype(str)
    print(f"Loaded {len(df)} rows ({df['region'].notna().sum()} labeled).")

    print(f"\nSearching all turbines for the best 7-day window "
          f"(maximizing min(severe_up, severe_down) count)...")
    best = find_best_window(df[df["region"].notna()], window_days=7)
    print(f"Selected: {best['site']} turbine {best['turbine_id']}, "
          f"window ending {best['window_end']}")
    print(f"  severe_up events in window: {int(best['up_count'])}, "
          f"severe_down events in window: {int(best['down_count'])}")

    window_end = best["window_end"]
    window_start = window_end - pd.Timedelta(days=7)
    win_wide = df[(df["site"] == best["site"]) & (df["turbine_id"] == best["turbine_id"]) &
                  (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)].sort_values("timestamp")

    pair = find_closest_pair(win_wide)
    if pair is not None:
        mid = pair[0] + (pair[1] - pair[0]) / 2
        print(f"  Closest severe_up/severe_down pair: {pair[0]} and {pair[1]} "
              f"({abs((pair[0]-pair[1]).total_seconds())/3600:.1f} hours apart) -- zooming around midpoint")
    else:
        mid = window_start + (window_end - window_start) / 2
        print("  No up/down pair found in this window -- zooming around window midpoint instead")

    zoom_start = mid - pd.Timedelta(days=ZOOM_DAYS)
    zoom_end = mid + pd.Timedelta(days=ZOOM_DAYS)
    win = df[(df["site"] == best["site"]) & (df["turbine_id"] == best["turbine_id"]) &
             (df["timestamp"] >= zoom_start) & (df["timestamp"] <= zoom_end)].sort_values("timestamp")

    fig = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 1.0, 1.0], hspace=0.55)

    gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2, 0],
                                                height_ratios=[1.4, 1.0], hspace=0.08)
    ax1 = fig.add_subplot(gs_top[0])
    ax2 = fig.add_subplot(gs_top[1], sharex=ax1)

    def shade_by_region(ax, sub):
        sub = sub.reset_index(drop=True)
        run_start = 0
        for i in range(1, len(sub) + 1):
            changed = i == len(sub) or sub.loc[i, "region"] != sub.loc[run_start, "region"]
            if changed:
                region = sub.loc[run_start, "region"]
                if pd.notna(region):
                    ax.axvspan(sub.loc[run_start, "timestamp"],
                               sub.loc[min(i, len(sub) - 1), "timestamp"],
                               color=REGION_COLORS[region], alpha=0.28, lw=0)
                run_start = i

    shade_by_region(ax1, win)
    ax1.plot(win["timestamp"], win["norm_power"], color="black", lw=1.1, zorder=3)
    ax1.set_ylabel("Normalized power", fontsize=8)
    ax1.set_title(f"{best['site']} turbine {best['turbine_id']}: "
                  f"{zoom_start.strftime('%Y-%m-%d %H:%M')} to {zoom_end.strftime('%Y-%m-%d %H:%M')}",
                  fontsize=9, pad=8)
    ax1.tick_params(labelbottom=False, labelsize=7)

    shade_by_region(ax2, win)
    ax2.plot(win["timestamp"], win["delta_norm_power"], color="black", lw=1.1, zorder=3)
    for thresh in (-0.30, -0.10, 0.10, 0.30):
        ax2.axhline(thresh, color="gray", lw=0.7, linestyle="--", zorder=2)
    ax2.set_ylabel("$\\Delta$ norm. power\n(30 min ahead)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_xlabel("Time", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[r], alpha=0.5) for r in REGION_ORDER]
    fig.legend(handles, REGION_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=5, fontsize=7, frameon=False)

    ax3 = fig.add_subplot(gs[2, 0])
    all_delta = df.loc[df["region"].notna(), "delta_norm_power"]
    bins = np.linspace(all_delta.min(), all_delta.max(), 150)
    ax3.hist(all_delta, bins=bins, color="#888888", alpha=0.6)
    for thresh in (-0.30, -0.10, 0.10, 0.30):
        ax3.axvline(thresh, color="black", lw=0.9, linestyle="--")
    edges = [all_delta.min(), -0.30, -0.10, 0.10, 0.30, all_delta.max()]
    for i, region in enumerate(REGION_ORDER):
        ax3.axvspan(edges[i], edges[i + 1], color=REGION_COLORS[region], alpha=0.12, zorder=0)
    ax3.set_xlabel("$\\Delta$ norm. power (30 min ahead) -- all turbines, full year", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)
    ax3.set_title("Full-dataset distribution of the labeling variable, with region thresholds",
                  fontsize=8.5)
    ax3.tick_params(labelsize=7)

    fig.suptitle("Label quality check: continuous ramp signal vs. assigned ordinal region",
                 fontsize=10, y=0.995)
    fig.subplots_adjust(top=0.90)
    out = FIG_DIR / "figureD_label_quality.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] {out}")

if __name__ == "__main__":
    main()