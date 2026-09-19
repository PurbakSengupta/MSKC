"""
Step 4b: Decide whether the flagged extreme wind-speed readings are local
sensor faults or genuine severe-weather events, using sibling turbines at
the same site as the check.

Logic: at each flagged timestamp, compare the flagged turbine's wind speed
to the OTHER turbines at the same site at that exact timestamp.
  - If siblings are also elevated (e.g. within the same rough range) ->
    consistent with real weather -> keep the reading.
  - If siblings are normal (e.g. under half the flagged value) while this
    one turbine spikes -> consistent with a local sensor fault -> candidate
    for exclusion.

This produces a per-flagged-row verdict and a summary, rather than a single
blanket threshold, because a single site-wide storm and a single faulty
anemometer look identical in a single turbine's power curve but look very
different once you check what its neighbours measured at the same instant.

Run this AFTER re-running 03_batch_extract_clean.py (which now flags
instead of drops).
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
CLEAN_DIR = ROOT_DIR / "extracted" / "clean"
OUT_DIR = ROOT_DIR / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

# A flagged reading is treated as "corroborated by siblings" if the median
# sibling wind speed at that timestamp is at least this fraction of the
# flagged turbine's reading.
SIBLING_COROBORATION_RATIO = 0.5


def load_all_clean():
    files = sorted(CLEAN_DIR.glob("*_clean.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["turbine_id"] = df["turbine_id"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all_clean()
    if "flag_windspeed_extreme" not in df.columns:
        raise ValueError(
            "No 'flag_windspeed_extreme' column found -- make sure you re-ran "
            "the updated 03_batch_extract_clean.py before this script."
        )

    flagged = df[df["flag_windspeed_extreme"]].copy()
    print(f"Total flagged rows across all turbines: {len(flagged)}")
    if flagged.empty:
        print("Nothing to check.")
        return

    print(flagged.groupby(["site", "turbine_id"]).size().to_string())

    # Build a fast lookup: (site, timestamp) -> all turbines' wind speeds
    lookup = df.pivot_table(
        index=["site", "timestamp"], columns="turbine_id",
        values="Wind speed (m/s)", aggfunc="first",
    )

    results = []
    for _, row in flagged.iterrows():
        site, tid, ts, ws = row["site"], row["turbine_id"], row["timestamp"], row["Wind speed (m/s)"]
        try:
            sibling_values = lookup.loc[(site, ts)].drop(labels=[tid], errors="ignore")
        except KeyError:
            sibling_values = pd.Series(dtype=float)
        sibling_values = sibling_values.dropna()

        sibling_median = sibling_values.median() if not sibling_values.empty else float("nan")
        corroborated = (
            not pd.isna(sibling_median)
            and sibling_median >= SIBLING_COROBORATION_RATIO * ws
        )

        results.append({
            "site": site, "turbine_id": tid, "timestamp": ts,
            "wind_speed": ws, "n_siblings_available": len(sibling_values),
            "sibling_median_windspeed": round(sibling_median, 2) if not pd.isna(sibling_median) else None,
            "corroborated_by_siblings": corroborated,
        })

    verdicts = pd.DataFrame(results)
    verdicts.to_csv(OUT_DIR / "windspeed_extreme_verdicts.csv", index=False)

    print("\n--- Verdict summary ---")
    print(verdicts.groupby(["site", "turbine_id"])["corroborated_by_siblings"]
          .agg(["sum", "count"]).rename(columns={"sum": "corroborated", "count": "total_flagged"}).to_string())

    n_corroborated = int(verdicts["corroborated_by_siblings"].sum())
    n_total = len(verdicts)
    print(f"\nOverall: {n_corroborated} / {n_total} flagged readings corroborated by sibling turbines.")
    print(f"Full per-row verdicts written to {OUT_DIR / 'windspeed_extreme_verdicts.csv'}")

if __name__ == "__main__":
    main()