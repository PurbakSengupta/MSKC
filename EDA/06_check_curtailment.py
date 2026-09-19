"""
Step 4c: Does curtailment/downtime explain the near-zero-power band at high
wind speed seen in both power-curve plots? Also applies the one confirmed
sensor-fault exclusion from the windspeed cross-check (Penmanshiel 12,
2024-01-20 23:30 to 2024-01-21 00:00).

Logic: define "anomalous" rows as wind speed comfortably above rated
(> 8 m/s, well past cut-in, approaching where power should be near max)
but power far below what the power curve would predict (< 200 kW). Then
check what fraction of those rows have non-zero curtailment loss, non-zero
downtime loss, or reduced system availability. If most of them do, the
near-zero-power band is explained by legitimate operational states, not a
data quality problem -- which matters a lot, because if we defined ramp
events straight from raw power without this check, curtailment episodes
would get mislabeled as physical ramp-down events.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
CLEAN_DIR = ROOT_DIR / "extracted" / "clean"
OUT_DIR = ROOT_DIR / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

# The one confirmed sensor fault from the windspeed cross-check.
FAULT_SITE = "Penmanshiel"
FAULT_TURBINE = "12"
FAULT_START = pd.Timestamp("2024-01-20 23:30:00")
FAULT_END = pd.Timestamp("2024-01-21 00:00:00")

ANOMALY_WINDSPEED_MIN = 8.0   # m/s
ANOMALY_POWER_MAX = 200.0     # kW


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
    print(f"Loaded {len(df)} rows total.")

    # Apply the one confirmed sensor-fault exclusion.
    fault_mask = (
        (df["site"] == FAULT_SITE) & (df["turbine_id"] == FAULT_TURBINE) &
        (df["timestamp"] >= FAULT_START) & (df["timestamp"] <= FAULT_END)
    )
    n_fault = int(fault_mask.sum())
    df = df[~fault_mask].copy()
    print(f"Excluded {n_fault} row(s) from the confirmed sensor fault "
          f"({FAULT_SITE} {FAULT_TURBINE}, {FAULT_START} to {FAULT_END}).")

    required_cols = ["Lost Production to Curtailment (Total) (kWh)",
                      "Lost Production to Downtime (kWh)",
                      "Time-based System Avail."]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns (re-run 03 first?): {missing}")

    anomaly = df[(df["Wind speed (m/s)"] > ANOMALY_WINDSPEED_MIN) &
                 (df["Power (kW)"] < ANOMALY_POWER_MAX)].copy()
    print(f"\nAnomalous rows (wind speed > {ANOMALY_WINDSPEED_MIN} m/s, "
          f"power < {ANOMALY_POWER_MAX} kW): {len(anomaly)} "
          f"({100 * len(anomaly) / len(df):.3f}% of all rows)")

    if anomaly.empty:
        print("No anomalous rows found -- nothing further to check.")
        return

    anomaly["has_curtailment"] = anomaly["Lost Production to Curtailment (Total) (kWh)"] > 0
    anomaly["has_downtime"] = anomaly["Lost Production to Downtime (kWh)"] > 0
    anomaly["reduced_availability"] = anomaly["Time-based System Avail."] < 1.0
    anomaly["explained"] = (
        anomaly["has_curtailment"] | anomaly["has_downtime"] | anomaly["reduced_availability"]
    )

    print("\n--- Explanation breakdown (rows can overlap categories) ---")
    print(f"  Has curtailment loss > 0:      {anomaly['has_curtailment'].sum()} "
          f"({100 * anomaly['has_curtailment'].mean():.1f}%)")
    print(f"  Has downtime loss > 0:         {anomaly['has_downtime'].sum()} "
          f"({100 * anomaly['has_downtime'].mean():.1f}%)")
    print(f"  Time-based System Avail. < 1:  {anomaly['reduced_availability'].sum()} "
          f"({100 * anomaly['reduced_availability'].mean():.1f}%)")
    print(f"  Explained by at least one:     {anomaly['explained'].sum()} "
          f"({100 * anomaly['explained'].mean():.1f}%)")

    unexplained = anomaly[~anomaly["explained"]]
    print(f"\nUnexplained anomalous rows: {len(unexplained)}")
    if not unexplained.empty:
        print("Breakdown by site/turbine:")
        print(unexplained.groupby(["site", "turbine_id"]).size().to_string())
        out_path = OUT_DIR / "unexplained_anomalies.csv"
        unexplained.to_csv(out_path, index=False)
        print(f"Full unexplained rows written to {out_path}")

    # Save the cleaned, fault-excluded, full dataset for the next stage
    # (labeling) so we don't have to redo the fault exclusion every time.
    final_path = OUT_DIR / "all_turbines_clean_final.csv"
    df.to_csv(final_path, index=False)
    print(f"\nWrote final cleaned combined dataset: {final_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()