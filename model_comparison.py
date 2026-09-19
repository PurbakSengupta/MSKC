"""
Step 12: Does a genuinely lightweight model (fewer trees, shallower depth --
closer to what an actual tinyML-class edge deployment would run) preserve
enough accuracy while cutting compute latency enough to make the CAGT
carbon-savings estimate robust across the assumption grid, rather than
swinging 3.5%-40.6% depending on assumed edge power draw?

This retrains r0 and r1 at THREE configs (the original 300-tree version for
reference, plus two lighter ones), reports accuracy (macro-F1, severe
recall) and measured single-row latency for each, so we can see the actual
accuracy-vs-latency trade-off rather than assuming a lighter model "is
probably fine."

NOTE on the gate: the KUS gate itself was trained on the 300-tree models'
OOB predictions. For this latency/accuracy comparison we reuse the SAME
tau=0.4 injection rate (48.7%) as a fixed input to keep this a clean,
isolated compute-cost comparison; a fully rigorous version would retrain
the gate on the light model's own OOB predictions, which is a natural
next step if the light model looks good here.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

INJECTION_RATE_AT_TAU = 0.487
N_TIMING_REPS = 500

CONFIGS = {
    "heavy (n=300, depth=12) -- current":  {"n_estimators": 300, "max_depth": 12},
    "medium (n=50, depth=8)":              {"n_estimators": 50,  "max_depth": 8},
    "light (n=15, depth=5)":               {"n_estimators": 15,  "max_depth": 5},
}

EDGE_POWER_W_RANGE = [1.0, 3.0, 5.0]
COMM_ENERGY_MJ_RANGE = [10.0, 50.0, 100.0]
GRID_CARBON_INTENSITY_GCO2_PER_KWH = 141.0


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


def metrics_for(y_true, pred):
    macro_f1 = f1_score(y_true, pred, average="macro")
    is_severe = y_true.isin(["severe_down", "severe_up"])
    pred_series = pd.Series(pred, index=y_true.index)
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred_series == y_true)).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall


def time_single_row(model, X_row_values, n_reps):
    """Times single-row .predict() calls using n_jobs=1 (the realistic
    edge-deployment pattern: one sample scored at a time, not a batch).
    n_jobs=-1 was used for TRAINING (legitimate, batch operation) but must
    be disabled here -- for a single row, joblib's parallel dispatch
    overhead dominates the actual (tiny) computation, which is why latency
    barely dropped going from 300 trees to 15 in the first run. Takes a
    plain numpy array, not a DataFrame slice, to avoid pandas construction
    overhead being measured as if it were model cost.
    """
    original_n_jobs = model.n_jobs
    model.n_jobs = 1
    model.predict(X_row_values)  # warm-up call, excluded from timing
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict(X_row_values)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    model.n_jobs = original_n_jobs
    return np.median(times)


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

    all_results = []
    for config_name, params in CONFIGS.items():
        print(f"\n{'=' * 70}\nConfig: {config_name}")

        r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample", **params)
        r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

        r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample", **params)
        r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

        r0_pred = r0.predict(test_common[OWN_FEATURES])
        r1_pred = r1.predict(test_common[KUS_FEATURES])
        f1_r0, rec_r0 = metrics_for(test_common["region"], r0_pred)
        f1_r1, rec_r1 = metrics_for(test_common["region"], r1_pred)
        print(f"  r0: macro-F1={f1_r0:.4f}, severe_recall={rec_r0:.4f}")
        print(f"  r1: macro-F1={f1_r1:.4f}, severe_recall={rec_r1:.4f}")

        row0 = test_common[OWN_FEATURES].iloc[[0]].to_numpy()
        row1 = test_common[KUS_FEATURES].iloc[[0]].to_numpy()
        lat_r0 = time_single_row(r0, row0, N_TIMING_REPS)
        lat_r1 = time_single_row(r1, row1, N_TIMING_REPS)
        print(f"  Latency: r0={lat_r0*1000:.4f} ms, r1={lat_r1*1000:.4f} ms")

        n_inject = int(round(n_test * INJECTION_RATE_AT_TAU))
        n_no_inject = n_test - n_inject

        pct_range = []
        for power_w in EDGE_POWER_W_RANGE:
            e_compute_r0 = lat_r0 * power_w
            e_compute_r1 = lat_r1 * power_w
            for comm_mj in COMM_ENERGY_MJ_RANGE:
                e_comm = comm_mj / 1000.0
                cagt_r0 = e_compute_r0
                cagt_r1 = e_compute_r1 + e_comm
                total_always_r1 = n_test * cagt_r1
                total_aies = n_inject * cagt_r1 + n_no_inject * cagt_r0
                pct_reduction = 100 * (1 - total_aies / total_always_r1)
                pct_range.append(pct_reduction)

        pct_range = np.array(pct_range)
        print(f"  AIES vs always-r1 energy reduction across assumption grid: "
              f"{pct_range.min():.1f}% - {pct_range.max():.1f}% "
              f"(mean {pct_range.mean():.1f}%)")

        all_results.append({
            "config": config_name, "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "f1_r0": round(f1_r0, 4), "f1_r1": round(f1_r1, 4),
            "severe_recall_r0": round(rec_r0, 4), "severe_recall_r1": round(rec_r1, 4),
            "latency_r0_ms": round(lat_r0 * 1000, 4), "latency_r1_ms": round(lat_r1 * 1000, 4),
            "carbon_reduction_min_pct": round(pct_range.min(), 1),
            "carbon_reduction_max_pct": round(pct_range.max(), 1),
            "carbon_reduction_mean_pct": round(pct_range.mean(), 1),
        })

    summary = pd.DataFrame(all_results)
    summary.to_csv(OUT_DIR / "lightweight_model_comparison.csv", index=False)
    print(f"\n{'=' * 70}\nFull comparison written to {OUT_DIR / 'lightweight_model_comparison.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()