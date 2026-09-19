"""
Is the ~0.27-0.33 macro-F1 ceiling a property of the task, or specific to
RandomForest? Trains a small MLPClassifier on the same features as r0/r1
and compares directly.

Class imbalance handling differs from the RF experiments: MLPClassifier
has no class_weight parameter, so imbalance is handled via CAPPED
UNDERSAMPLING of majority classes in training (not the same mechanism as
RandomForest's class_weight='balanced_subsample' -- flagged explicitly so
the two aren't read as directly equivalent techniques).

Also times single-row inference latency for the MLP, since even if it's
somewhat more accurate, a large latency/energy cost would still favor the
lightweight tree-based approach within this paper's own resource-aware
framing -- this is itself a relevant, reportable result either way.

Reference numbers printed alongside are the light-RF results already
established (r0: macro_f1=0.2795, severe_recall=0.7005, false_safe=0.0325;
r1: macro_f1=0.2727, severe_recall=0.7513, false_safe=0.0307), both on the
neighbor-available test subset -- this script evaluates on the same subset
for a fair comparison.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

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

UNDERSAMPLE_CAP = 8000
N_TIMING_REPS = 300

REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]
REGION_TO_INT = {name: i for i, name in enumerate(REGION_ORDER)}
INT_TO_REGION = {i: name for name, i in REGION_TO_INT.items()}

RF_REFERENCE = {
    "r0": {"macro_f1": 0.2795, "severe_recall": 0.7005, "false_safe": 0.0325, "latency_ms": 0.85},
    "r1": {"macro_f1": 0.2727, "severe_recall": 0.7513, "false_safe": 0.0307, "latency_ms": 0.85},
}


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


def undersample(df, label_col, cap, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for cls, g in df.groupby(label_col):
        if len(g) > cap:
            idx = rng.choice(g.index, size=cap, replace=False)
            parts.append(g.loc[idx])
        else:
            parts.append(g)
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out


def metrics_for(y_true, pred):
    macro_f1 = f1_score(y_true, pred, average="macro")
    is_severe = y_true.isin(["severe_down", "severe_up"])
    pred_series = pd.Series(pred, index=y_true.index)
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred_series == y_true)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & (pred_series == "stable")).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


def time_single_row(model, X_row, n_reps):
    model.predict(X_row)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict(X_row)
        t1 = time.perf_counter()
        times.append(t1 - t0)
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

    # Belt-and-suspenders check: 'neighbor_available' is supposed to
    # guarantee all three NEIGHBOR_FEATURES are populated, but
    # RandomForestClassifier (sklearn >=1.4) silently tolerates NaN via
    # native missing-value routing, so earlier RF-based results could have
    # been trained on a few incomplete rows without ever raising an error.
    # MLPClassifier has no such tolerance, which is how this surfaced.
    # Explicitly require all three columns non-null here.
    valid_r1 = df["neighbor_available"].copy()
    for col in NEIGHBOR_FEATURES:
        n_before = valid_r1.sum()
        valid_r1 &= df[col].notna()
        n_after = valid_r1.sum()
        if n_after < n_before:
            print(f"  NOTE: {n_before - n_after} additional row(s) dropped due to NaN in '{col}' "
                  f"despite neighbor_available=True")

    r0_train_full = df[train_mask]
    r1_train_full = df[train_mask & valid_r1]
    test_common = df[~train_mask & valid_r1]
    print(f"r0 train (full): {len(r0_train_full)} | r1 train (neighbor avail.): {len(r1_train_full)} | "
          f"test (common): {len(test_common)}")

    print(f"\nComputing a MATCHED undersampling cap from r1's own class counts, "
          f"applied identically to r0 and r1 so both train on the same class ratios...")
    r1_class_counts = r1_train_full["region"].value_counts()
    matched_cap = int(r1_class_counts.min())
    print(f"  r1 class counts (pre-cap): \n{r1_class_counts.to_string()}")
    print(f"  Matched cap (= r1's smallest class count): {matched_cap}")

    r0_train = undersample(r0_train_full, "region", matched_cap, RANDOM_STATE)
    r1_train = undersample(r1_train_full, "region", matched_cap, RANDOM_STATE)
    print(f"  r0 train after matched undersampling: {len(r0_train)} rows")
    print(f"  r1 train after matched undersampling: {len(r1_train)} rows")
    print(f"  r0 class counts:\n{r0_train['region'].value_counts().to_string()}")
    print(f"  r1 class counts:\n{r1_train['region'].value_counts().to_string()}")

    print("\nTraining MLP r0 (own features only)...")
    scaler_r0 = StandardScaler().fit(r0_train[OWN_FEATURES])
    X_r0_train = scaler_r0.transform(r0_train[OWN_FEATURES])
    X_r0_test = scaler_r0.transform(test_common[OWN_FEATURES])

    mlp_r0 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                            random_state=RANDOM_STATE)
    t0 = time.time()
    # Encode labels as integers -- MLPClassifier's internal early-stopping
    # validation score has a known incompatibility with non-numeric class
    # labels (raises TypeError from np.isnan on string predictions) in some
    # sklearn versions.
    y_r0_train = r0_train["region"].map(REGION_TO_INT).values
    mlp_r0.fit(X_r0_train, y_r0_train)
    print(f"  Trained in {time.time()-t0:.1f}s, {mlp_r0.n_iter_} iterations")

    pred_r0_int = mlp_r0.predict(X_r0_test)
    pred_r0 = pd.Series(pred_r0_int).map(INT_TO_REGION).values
    f1_r0, rec_r0, fs_r0 = metrics_for(test_common["region"], pred_r0)
    lat_r0 = time_single_row(mlp_r0, X_r0_test[:1], N_TIMING_REPS)

    print("\nTraining MLP r1 (own + neighbor features)...")
    scaler_r1 = StandardScaler().fit(r1_train[KUS_FEATURES])
    X_r1_train = scaler_r1.transform(r1_train[KUS_FEATURES])
    X_r1_test = scaler_r1.transform(test_common[KUS_FEATURES])

    mlp_r1 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                            random_state=RANDOM_STATE)
    t0 = time.time()
    y_r1_train = r1_train["region"].map(REGION_TO_INT).values
    mlp_r1.fit(X_r1_train, y_r1_train)
    print(f"  Trained in {time.time()-t0:.1f}s, {mlp_r1.n_iter_} iterations")

    pred_r1_int = mlp_r1.predict(X_r1_test)
    pred_r1 = pd.Series(pred_r1_int).map(INT_TO_REGION).values
    f1_r1, rec_r1, fs_r1 = metrics_for(test_common["region"], pred_r1)
    lat_r1 = time_single_row(mlp_r1, X_r1_test[:1], N_TIMING_REPS)

    print(f"\n{'='*90}\nMLP vs. RF (light config) reference, on the same neighbor-available test subset\n{'='*90}")
    print(f"{'model':<20}{'macro_f1':>10}{'severe_recall':>15}{'false_safe':>12}{'latency_ms':>12}")
    print(f"{'RF r0 (ref)':<20}{RF_REFERENCE['r0']['macro_f1']:>10.4f}"
          f"{RF_REFERENCE['r0']['severe_recall']:>15.4f}{RF_REFERENCE['r0']['false_safe']:>12.4f}"
          f"{RF_REFERENCE['r0']['latency_ms']:>12.4f}")
    print(f"{'RF r1 (ref)':<20}{RF_REFERENCE['r1']['macro_f1']:>10.4f}"
          f"{RF_REFERENCE['r1']['severe_recall']:>15.4f}{RF_REFERENCE['r1']['false_safe']:>12.4f}"
          f"{RF_REFERENCE['r1']['latency_ms']:>12.4f}")
    print(f"{'MLP r0':<20}{f1_r0:>10.4f}{rec_r0:>15.4f}{fs_r0:>12.4f}{lat_r0*1000:>12.4f}")
    print(f"{'MLP r1':<20}{f1_r1:>10.4f}{rec_r1:>15.4f}{fs_r1:>12.4f}{lat_r1*1000:>12.4f}")

    out = pd.DataFrame([
        {"model": "RF_r0_ref", **RF_REFERENCE["r0"]},
        {"model": "RF_r1_ref", **RF_REFERENCE["r1"]},
        {"model": "MLP_r0", "macro_f1": f1_r0, "severe_recall": rec_r0, "false_safe": fs_r0, "latency_ms": lat_r0*1000},
        {"model": "MLP_r1", "macro_f1": f1_r1, "severe_recall": rec_r1, "false_safe": fs_r1, "latency_ms": lat_r1*1000},
    ])
    out.to_csv(OUT_DIR / "mlp_vs_rf_comparison.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'mlp_vs_rf_comparison.csv'}")

if __name__ == "__main__":
    main()