"""
Step 5: Define ordinal ramp-event region labels.

Target: change in normalized power (Power / rated_power) over a 30-minute
horizon (3 steps of 10-min data), bucketed into 5 ordinal, asymmetric
regions (severe down / moderate down / stable / moderate up / severe up).

A label is only assigned when:
  - the current row and every row through the horizon exist on a clean,
    contiguous 10-minute grid (no gaps), AND
  - none of those rows are downtime-affected or have reduced availability
    (per the check in 04c: downtime, not curtailment, explains the
    near-zero-power band, and a downtime-driven power collapse is not a
    physical wind-driven ramp event).

Rows that don't meet this are label = NaN / excluded, not silently
imputed, so we can see how much of the year is actually usable for this
task before building anything on top of it.

Thresholds (REGION_THRESHOLDS below) are a starting point, not final --
this script's main job right now is to show us the resulting label
distribution so we can decide if they need adjusting (e.g. if "stable"
swamps everything).
"""

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "eda_output" / "all_turbines_clean_final.csv"
OUT_DIR = ROOT_DIR / "Results"
OUT_DIR.mkdir(exist_ok=True)

RATED_POWER_KW = 2050.0   # confirmed identical across all turbines in metadata
HORIZON_STEPS = 3         # 3 x 10 min = 30 minutes ahead

# Ordinal region boundaries on normalized power CHANGE (delta_norm_power).
# Asymmetric by construction: up-ramps and down-ramps are separate labels,
# not folded into a symmetric "magnitude" scale.
REGION_THRESHOLDS = {
    "severe_down": (-np.inf, -0.30),
    "moderate_down": (-0.30, -0.10),
    "stable": (-0.10, 0.10),
    "moderate_up": (0.10, 0.30),
    "severe_up": (0.30, np.inf),
}
REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]


def assign_region(delta):
    if pd.isna(delta):
        return None
    for name, (lo, hi) in REGION_THRESHOLDS.items():
        if lo < delta <= hi or (lo == -np.inf and delta <= hi):
            return name
    return None


def process_turbine(g: pd.DataFrame) -> pd.DataFrame:
    """g: rows for ONE turbine, any subset of a year (gaps allowed), must
    contain timestamp, Power (kW), and the downtime/availability flags."""
    g = g.sort_values("timestamp").set_index("timestamp")

    # Reindex onto a strict 10-minute grid so shifts by HORIZON_STEPS are
    # only ever "true" 30-minutes-later values, never silently skipping a
    # gap and comparing across a missing chunk.
    full_index = pd.date_range(g.index.min(), g.index.max(), freq="10min")
    g = g.reindex(full_index)

    g["norm_power"] = g["Power (kW)"] / RATED_POWER_KW
    g["is_bad"] = g["Power (kW)"].isna() | (g["Time-based System Avail."] < 1.0) | \
                  (g["Lost Production to Downtime (kWh)"].fillna(0) > 0)

    # Future value and "was every step in between clean" check.
    g["future_norm_power"] = g["norm_power"].shift(-HORIZON_STEPS)
    # Rolling window covering [t, t+HORIZON_STEPS] must have zero bad rows.
    window_bad = g["is_bad"].rolling(HORIZON_STEPS + 1).sum().shift(-HORIZON_STEPS)
    g["window_clean"] = window_bad == 0

    g["delta_norm_power"] = g["future_norm_power"] - g["norm_power"]
    g.loc[~g["window_clean"], "delta_norm_power"] = np.nan

    g["region"] = g["delta_norm_power"].apply(assign_region)

    g = g.reset_index().rename(columns={"index": "timestamp"})
    return g


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows.")

    results = []
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        out = process_turbine(g.copy())
        out["site"] = site
        out["turbine_id"] = tid
        results.append(out)

    labeled = pd.concat(results, ignore_index=True)

    n_total = len(labeled)
    n_labeled = labeled["region"].notna().sum()
    print(f"\nTotal grid rows (all turbines, full-year 10-min grid): {n_total}")
    print(f"Rows with a valid region label: {n_labeled} ({100*n_labeled/n_total:.1f}%)")

    print("\n--- Overall label distribution (valid labels only) ---")
    dist = labeled["region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
    for name in REGION_ORDER:
        pct = 100 * dist[name] / n_labeled if n_labeled else 0
        print(f"  {name:15s}: {dist[name]:7d}  ({pct:5.2f}%)")

    print("\n--- Label distribution per site ---")
    for site, g in labeled.groupby("site"):
        g_labeled = g["region"].notna().sum()
        print(f"\n{site} (n_labeled={g_labeled}):")
        site_dist = g["region"].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
        for name in REGION_ORDER:
            pct = 100 * site_dist[name] / g_labeled if g_labeled else 0
            print(f"  {name:15s}: {site_dist[name]:7d}  ({pct:5.2f}%)")

    out_path = OUT_DIR / "labeled_dataset.csv"
    labeled.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(labeled)} rows, including unlabeled ones -- "
          f"filter on region.notna() downstream).")


if __name__ == "__main__":
    main()