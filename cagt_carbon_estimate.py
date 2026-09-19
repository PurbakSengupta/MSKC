"""
Step 11 (revised): CAGT (Carbon-Adjusted Gating Threshold) -- real cost
measurement, with corrected assumption sourcing after external fact-check.

Two things are kept explicitly separate and labeled, because they have very
different evidentiary status:

  MEASURED (real, on this machine):
    - r0 vs r1 inference latency, timed directly with time.perf_counter()
      over many repetitions.

  ASSUMED (sensitivity-swept, NOT measured):
    - Edge/embedded inference power draw. 1-5 W is an assumed envelope
      representative of a small edge gateway (e.g. Raspberry-Pi-class or
      Jetson-class device), NOT substantiated by the Njor et al. tinyML
      primer -- that paper's own devices target milliwatt-range power,
      which is a different hardware class. Do not cite Njor et al. for
      this specific wattage figure.
    - Per-query communication energy for retrieving a neighbor turbine's
      live reading. Revised sweep below replaces the earlier vague "tens
      of mJ" claim with a literature-informed range: Mabon et al. measured
      0.99-266.65 mJ for a 25-byte LoRa payload depending on spreading
      factor/power config; a LoRaWAN energy model (Sensors, 2021) reports
      19.56 mJ for a representative 50-byte, 1 km, DR5 transmission, used
      here as the nominal case. This is reported as a literature-INFORMED
      assumption, not a claim that "the literature says X" universally --
      the range is swept explicitly so no single figure is load-bearing.

Grid carbon intensity is NOT used to compute the headline percentage
reduction below, since it is a fixed multiplicative factor applied equally
to both the always-r1 and AIES totals and therefore cancels in the ratio.
It is only needed if an ABSOLUTE emissions figure is wanted; if so, use
141 gCO2/kWh (Purely Energy UK Grid Report, 2026 year-to-date, derived
from NESO half-hourly data), stated explicitly as a running YTD figure at
the time of analysis, not a final annual average. Do not cite a live
carbon-intensity dashboard (e.g. British Energy Compliance) for this
specific YTD-average number; it displays a live instantaneous value, not
the aggregated YTD average Purely Energy reports.

Why r0 needs no communication term: r0 only uses the turbine's own local
sensors. r1 additionally requires a message to/from the neighbor turbine,
which is the actual resource cost this experiment is really about (maps to
the D_k(s) data-movement term in the PhD plan's resource-feasibility
formalism), not the classifier itself, which is nearly the same size for
both routes.

Reports BOTH absolute totals (small, given this is one year of 11 turbines,
a proof-of-concept scale, and only meaningful if the grid-intensity
citation question below is resolved) and relative % reduction (the number
that actually scales to a real fleet-wide, multi-year deployment, and does
not depend on the grid-intensity figure at all).
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "Results" / "labeled_with_neighbor.csv"
OUT_DIR = ROOT_DIR / "Results"

RATED_POWER_KW = 2050.0
SPLIT_DATE = pd.Timestamp("2024-11-01")
RANDOM_STATE = 42

OWN_FEATURES = [
    "Wind speed (m/s)", "wind_dir_sin", "wind_dir_cos", "own_power_norm",
    "own_power_norm_lag1", "own_power_norm_lag2", "own_power_norm_lag3",
    "own_windspeed_lag1", "own_windspeed_lag2", "own_windspeed_lag3",
    "month", "hour",
]
NEIGHBOR_FEATURES = ["neighbor_power_norm", "neighbor_windspeed", "neighbor_distance_m"]
KUS_FEATURES = OWN_FEATURES + NEIGHBOR_FEATURES

# From step 9/10 (LIGHT config, re-run after step 12's comparison):
CHOSEN_TAU = 0.4
INJECTION_RATE_AT_TAU = 0.366  # from the light-model frontier printout at tau=0.4

N_TIMING_REPS = 500  # single-row timing repetitions per model

MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}  # light config, matches 09/12

# ---- Assumptions, sensitivity-swept, not pinned to a single value ----
EDGE_POWER_W_RANGE = [1.0, 3.0, 5.0]                 # Watts, assumed edge-gateway envelope
COMM_ENERGY_MJ_RANGE = [5.0, 20.0, 50.0, 100.0, 200.0]  # mJ per neighbor query; 20 mJ = nominal
                                                       # (19.56 mJ, LoRaWAN 50-byte/1km/DR5 model,
                                                       # Sensors 2021), swept 5-200 mJ per Mabon et
                                                       # al.'s measured 0.99-266.65 mJ range across
                                                       # LoRa spreading-factor/power configurations.
GRID_CARBON_INTENSITY_GCO2_PER_KWH = 141.0           # GB grid, 2026 YTD "at time of analysis",
                                                       # Purely Energy UK Grid Report, derived from
                                                       # NESO half-hourly data. Only affects the
                                                       # ABSOLUTE emissions figures below, not the
                                                       # headline % reduction, which cancels this
                                                       # factor out of the ratio entirely.


def add_own_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site", "turbine_id", "timestamp"]).copy()
    df["own_power_norm"] = df["Power (kW)"] / RATED_POWER_KW
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["Wind direction (\u00b0)"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["Wind direction (\u00b0)"]))
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    grp = df.groupby(["site", "turbine_id"])
    for lag in (1, 2, 3):
        df[f"own_power_norm_lag{lag}"] = grp["own_power_norm"].shift(lag)
        df[f"own_windspeed_lag{lag}"] = grp["Wind speed (m/s)"].shift(lag)
    return df


def time_single_row_inference(model, X_row, n_reps):
    """Times single-row .predict() calls, the realistic edge deployment
    pattern (one inference per incoming SCADA sample), not batched."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict(X_row)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)


def main():
    print("Loading and featurizing data...")
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df["turbine_id"] = df["turbine_id"].astype(str)
    df = add_own_features(df)
    df["neighbor_power_norm"] = df["neighbor_power_kw"] / RATED_POWER_KW

    valid = df["region"].notna()
    for col in OWN_FEATURES:
        valid &= df[col].notna()
    df = df[valid].copy()

    train_mask = df["timestamp"] < SPLIT_DATE
    r0_train = df[train_mask]
    r1_train = df[train_mask & df["neighbor_available"]]
    test_common = df[~train_mask & df["neighbor_available"]]
    n_test = len(test_common)
    print(f"Test set (neighbor available): {n_test} rows")

    print("\nTraining r0...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print("Training r1...")
    r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

    # ---- MEASURED: single-row inference latency ----
    print(f"\nTiming single-row inference ({N_TIMING_REPS} reps each)...")
    row0 = test_common[OWN_FEATURES].iloc[[0]].to_numpy()
    row1 = test_common[KUS_FEATURES].iloc[[0]].to_numpy()

    r0.n_jobs = 1
    r1.n_jobs = 1
    r0.predict(row0)  # warm-up
    r1.predict(row1)  # warm-up
    t_r0 = time_single_row_inference(r0, row0, N_TIMING_REPS)
    t_r1 = time_single_row_inference(r1, row1, N_TIMING_REPS)

    print(f"  r0: mean={t_r0.mean()*1000:.4f} ms, median={np.median(t_r0)*1000:.4f} ms, "
          f"std={t_r0.std()*1000:.4f} ms")
    print(f"  r1: mean={t_r1.mean()*1000:.4f} ms, median={np.median(t_r1)*1000:.4f} ms, "
          f"std={t_r1.std()*1000:.4f} ms")
    print(f"  r1/r0 latency ratio: {t_r1.mean()/t_r0.mean():.3f}")

    latency_r0_s = np.median(t_r0)
    latency_r1_s = np.median(t_r1)

    # ---- ASSUMED: convert to energy, sweep the assumption ranges ----
    print("\n=== Energy/carbon estimate, swept over assumed parameters ===")
    print(f"(latency measured directly; edge power draw and communication energy are\n"
          f" assumed sensitivity parameters, NOT measured and NOT claimed as a single\n"
          f" literature-verified figure -- see script docstring for sourcing status)")

    n_inject = int(round(n_test * INJECTION_RATE_AT_TAU))
    n_no_inject = n_test - n_inject

    results = []
    for power_w in EDGE_POWER_W_RANGE:
        e_compute_r0_j = latency_r0_s * power_w
        e_compute_r1_j = latency_r1_s * power_w
        for comm_mj in COMM_ENERGY_MJ_RANGE:
            e_comm_j = comm_mj / 1000.0
            cagt_r0_j = e_compute_r0_j
            cagt_r1_j = e_compute_r1_j + e_comm_j

            total_always_r0_j = n_test * cagt_r0_j
            total_always_r1_j = n_test * cagt_r1_j
            total_aies_j = n_inject * cagt_r1_j + n_no_inject * cagt_r0_j

            pct_reduction_vs_always_r1 = 100 * (1 - total_aies_j / total_always_r1_j)

            def to_gco2(joules):
                kwh = joules / 3_600_000.0
                return kwh * GRID_CARBON_INTENSITY_GCO2_PER_KWH

            results.append({
                "edge_power_w": power_w, "comm_energy_mj": comm_mj,
                "cagt_r0_mj": round(cagt_r0_j * 1000, 4),
                "cagt_r1_mj": round(cagt_r1_j * 1000, 4),
                "total_always_r0_gco2": round(to_gco2(total_always_r0_j), 4),
                "total_always_r1_gco2": round(to_gco2(total_always_r1_j), 4),
                "total_aies_gco2": round(to_gco2(total_aies_j), 4),
                "pct_reduction_aies_vs_always_r1": round(pct_reduction_vs_always_r1, 1),
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(OUT_DIR / "cagt_carbon_estimate.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'cagt_carbon_estimate.csv'}")

    nominal = results_df[(results_df["edge_power_w"] == 3.0) & (results_df["comm_energy_mj"] == 20.0)]
    if not nominal.empty:
        row = nominal.iloc[0]
        print(f"\nNominal scenario (3W edge power, 20mJ/query -- the LoRaWAN Sensors 2021 "
              f"representative-transmission figure): estimated reduction "
              f"{row['pct_reduction_aies_vs_always_r1']:.1f}%")

    print(f"\nHeadline (robust across the whole assumption grid above): AIES at tau={CHOSEN_TAU} "
          f"(injection rate {INJECTION_RATE_AT_TAU:.1%}) reduces estimated operational energy "
          f"by ~{results_df['pct_reduction_aies_vs_always_r1'].min():.0f}-"
          f"{results_df['pct_reduction_aies_vs_always_r1'].max():.0f}% versus always-on knowledge "
          f"injection, at statistically indistinguishable classification quality (step 10).")

if __name__ == "__main__":
    main()