"""
Step 4: EDA on the cleaned, combined turbine dataset.

By this point extraction + dedup are validated (batch_report.csv showed
zero conflicts across all 11 turbines). This script does the checks that
actually determine whether we can trust downstream modeling:
  - power-curve sanity (implausible power/wind-speed combinations)
  - wind-direction / nacelle-position bounds and circular statistics
  - cross-turbine timestamp overlap within each site (needed before any
    neighbor/wake feature, which requires simultaneous readings across
    turbines -- each turbine being individually ~99% complete does not
    guarantee the turbines are complete at the SAME timestamps)

Outputs (all under ./eda_output/):
  - turbine_metadata.csv        (parsed from the static files)
  - per_turbine_summary.csv     (counts, ranges, flagged-row counts)
  - power_curve_<site>.png      (one scatter plot per site, all turbines
                                  overlaid, colour-coded)
  - timestamp_overlap.csv       (per-site cross-turbine overlap stats)

Please upload the PNGs back to me directly (as image attachments) -- I can
read them -- along with pasting the console output and/or the CSV contents.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = ROOT_DIR / "extracted" / "clean"
OUT_DIR = ROOT_DIR / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

# Sanity-check thresholds. These are deliberately conservative (flag, don't
# auto-drop) so we can look at flagged rows before deciding a rule.
MIN_PLAUSIBLE_WIND_SPEED = -1.0     # m/s; small negative can be sensor noise near 0
MAX_PLAUSIBLE_WIND_SPEED = 45.0     # m/s; turbines typically cut out ~25 m/s, storms higher
MIN_PLAUSIBLE_POWER_KW = -150.0     # allow normal idle/hotel-load parasitic draw
RATED_POWER_TOLERANCE = 1.05        # allow 5% overshoot above nameplate rated power
# ----------------------------------------------------------------------


def load_static_metadata():
    frames = []
    for site_dir, fname in [("Kelmarsh", "Kelmarsh_WT_static.csv"),
                             ("Penmanshiel", "Penmanshiel_WT_static.csv")]:
        path = DATA_DIR / site_dir / fname
        if not path.exists():
            print(f"  WARNING: static file not found: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.dropna(subset=["Title"])  # drop trailing blank rows
        df["turbine_id"] = df["Title"].str.extract(r"(\d+)$").astype(int).astype(str)
        df["site"] = site_dir
        frames.append(df[["site", "turbine_id", "Latitude", "Longitude",
                           "Rated power (kW)", "Hub Height (m)", "Rotor Diameter (m)"]])
    meta = pd.concat(frames, ignore_index=True)
    meta.to_csv(OUT_DIR / "turbine_metadata.csv", index=False)
    return meta


def load_all_clean():
    files = sorted(CLEAN_DIR.glob("*_clean.csv"))
    if not files:
        raise FileNotFoundError(f"No clean files found in {CLEAN_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["turbine_id"] = df["turbine_id"].astype(str)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def power_curve_checks(df, meta):
    df = df.merge(meta[["site", "turbine_id", "Rated power (kW)"]],
                   on=["site", "turbine_id"], how="left")

    df["flag_windspeed_oor"] = (df["Wind speed (m/s)"] < MIN_PLAUSIBLE_WIND_SPEED) | \
                                (df["Wind speed (m/s)"] > MAX_PLAUSIBLE_WIND_SPEED)
    df["flag_power_too_low"] = df["Power (kW)"] < MIN_PLAUSIBLE_POWER_KW
    df["flag_power_too_high"] = df["Power (kW)"] > (df["Rated power (kW)"] * RATED_POWER_TOLERANCE)
    df["flag_winddir_oor"] = (df["Wind direction (\u00b0)"] < 0) | (df["Wind direction (\u00b0)"] > 360)
    df["flag_nacelle_oor"] = (df["Nacelle position (\u00b0)"] < 0) | (df["Nacelle position (\u00b0)"] > 360)

    return df


def summarize_per_turbine(df):
    rows = []
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        rows.append({
            "site": site, "turbine_id": tid, "n_rows": len(g),
            "date_min": g["timestamp"].min(), "date_max": g["timestamp"].max(),
            "power_min": g["Power (kW)"].min(), "power_max": g["Power (kW)"].max(),
            "power_mean": round(g["Power (kW)"].mean(), 1),
            "windspeed_min": g["Wind speed (m/s)"].min(),
            "windspeed_max": g["Wind speed (m/s)"].max(),
            "windspeed_mean": round(g["Wind speed (m/s)"].mean(), 2),
            "n_flag_windspeed_oor": int(g["flag_windspeed_oor"].sum()),
            "n_flag_power_too_low": int(g["flag_power_too_low"].sum()),
            "n_flag_power_too_high": int(g["flag_power_too_high"].sum()),
            "n_flag_winddir_oor": int(g["flag_winddir_oor"].sum()),
            "n_flag_nacelle_oor": int(g["flag_nacelle_oor"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "per_turbine_summary.csv", index=False)
    return summary


def circular_wind_direction_stats(df):
    """Circular mean/std of wind direction per turbine, handling 0/360 wrap
    properly via sin/cos rather than naive arithmetic mean."""
    rows = []
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        rad = np.deg2rad(g["Wind direction (\u00b0)"].dropna())
        sin_mean = np.sin(rad).mean()
        cos_mean = np.cos(rad).mean()
        circ_mean_deg = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360) % 360
        resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)  # 1 = no spread, 0 = uniform/random
        rows.append({"site": site, "turbine_id": tid,
                      "circular_mean_deg": round(circ_mean_deg, 1),
                      "concentration": round(resultant_length, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "wind_direction_circular_stats.csv", index=False)
    return out


def timestamp_overlap_check(df):
    rows = []
    for site, g in df.groupby("site"):
        turbines = g["turbine_id"].unique()
        ts_sets = {tid: set(g.loc[g["turbine_id"] == tid, "timestamp"]) for tid in turbines}
        common = set.intersection(*ts_sets.values())
        union = set.union(*ts_sets.values())
        rows.append({
            "site": site,
            "n_turbines": len(turbines),
            "union_timestamps": len(union),
            "common_to_all_turbines": len(common),
            "pct_common": round(100 * len(common) / len(union), 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "timestamp_overlap.csv", index=False)
    return out


def plot_power_curves(df):
    for site, g in df.groupby("site"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for tid, tg in g.groupby("turbine_id"):
            ax.scatter(tg["Wind speed (m/s)"], tg["Power (kW)"], s=1, alpha=0.15, label=f"T{tid}")
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"{site}: Power curve, all turbines")
        ax.legend(markerscale=10, fontsize=8)
        fig.tight_layout()
        out_path = OUT_DIR / f"power_curve_{site}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    print("Loading static metadata...")
    meta = load_static_metadata()
    print(meta.to_string())

    print("\nLoading all clean turbine files...")
    df = load_all_clean()
    print(f"Combined rows: {len(df)}")

    print("\nRunning power-curve / bounds checks...")
    df = power_curve_checks(df, meta)

    print("\nPer-turbine summary:")
    summary = summarize_per_turbine(df)
    print(summary.to_string())

    print("\nWind direction circular stats:")
    circ = circular_wind_direction_stats(df)
    print(circ.to_string())

    print("\nCross-turbine timestamp overlap per site:")
    overlap = timestamp_overlap_check(df)
    print(overlap.to_string())

    print("\nSaving power curve plots...")
    plot_power_curves(df)

    print(f"\nAll outputs written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()