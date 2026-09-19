"""
Produces a comprehensive per-row test-set results file, preserving
timestamp/site/turbine_id (which earlier scripts dropped once they only
needed aggregate metrics). This is what the dashboard figures are built
from: time series of true vs predicted region, injection decisions, and
per-turbine/per-site breakdowns.

Deployed-system behavior modeled here: EVERY test row gets an r0
prediction (r0 only needs local data). Rows with an available neighbor
ADDITIONALLY get an r1 prediction and a gate decision; the gate falls back
to r0 automatically when no neighbor is available (exactly what a real
edge deployment would do), rather than being restricted to the
neighbor-available subset as earlier analysis scripts were.

"""

from pathlib import Path
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
MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}
CHOSEN_TAU = 0.4

REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]
REGION_TO_INT = {name: i - 2 for i, name in enumerate(REGION_ORDER)}

OWN_FEATURES = [
    "Wind speed (m/s)", "wind_dir_sin", "wind_dir_cos", "own_power_norm",
    "own_power_norm_lag1", "own_power_norm_lag2", "own_power_norm_lag3",
    "own_windspeed_lag1", "own_windspeed_lag2", "own_windspeed_lag3",
    "month", "hour",
]
NEIGHBOR_FEATURES = ["neighbor_power_norm", "neighbor_windspeed", "neighbor_distance_m"]
KUS_FEATURES = OWN_FEATURES + NEIGHBOR_FEATURES


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
    test = df[~train_mask].copy()
    print(f"Train: {len(r0_train)} rows | Test (all): {len(test)} rows "
          f"(neighbor available: {test['neighbor_available'].sum()})")

    print("\nTraining r0...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print("Training r1...")
    r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

    print("Building KUS gate (from OOB predictions)...")
    r0_oob_pred = pd.Series(r0.classes_[np.nanargmax(r0.oob_decision_function_, axis=1)],
                             index=r0_train.index)
    r1_oob_pred = pd.Series(r1.classes_[np.nanargmax(r1.oob_decision_function_, axis=1)],
                             index=r1_train.index)
    kus_data = r1_train.copy()
    kus_data["r0_oob_pred"] = r0_oob_pred.loc[kus_data.index]
    kus_data["r1_oob_pred"] = r1_oob_pred.loc[kus_data.index]
    true_int = kus_data["region"].map(REGION_TO_INT)
    r0_int = kus_data["r0_oob_pred"].map(REGION_TO_INT)
    r1_int = kus_data["r1_oob_pred"].map(REGION_TO_INT)
    kus_data["r1_helps"] = ((true_int - r1_int).abs() < (true_int - r0_int).abs()).astype(int)

    kus_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE,
                                        n_jobs=-1, class_weight="balanced_subsample")
    kus_model.fit(kus_data[KUS_FEATURES], kus_data["r1_helps"])
    kus_class_idx = list(kus_model.classes_).index(1)

    # ---- Predict on the FULL test set (not just neighbor-available) ----
    print("\nPredicting on full test set...")
    test["r0_pred"] = r0.predict(test[OWN_FEATURES])

    test["r1_pred"] = pd.Series([np.nan] * len(test), index=test.index, dtype="object")
    test["kus_score"] = np.nan
    has_nb = test["neighbor_available"]
    test.loc[has_nb, "r1_pred"] = r1.predict(test.loc[has_nb, KUS_FEATURES])
    test.loc[has_nb, "kus_score"] = kus_model.predict_proba(test.loc[has_nb, KUS_FEATURES])[:, kus_class_idx]

    test["injected"] = has_nb & (test["kus_score"] > CHOSEN_TAU)
    test["final_pred"] = np.where(test["injected"], test["r1_pred"], test["r0_pred"])

    keep_cols = [
        "timestamp", "site", "turbine_id", "Wind speed (m/s)", "Wind direction (\u00b0)",
        "own_power_norm", "region", "neighbor_available", "neighbor_turbine_id",
        "neighbor_distance_m", "r0_pred", "r1_pred", "kus_score", "injected", "final_pred",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    out = test[keep_cols].copy()

    out_path = OUT_DIR / "full_test_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")
    print("\nQuick sanity check -- overall injection rate: "
          f"{out['injected'].mean():.3f} (should be close to 0.366 * neighbor-available fraction)")


if __name__ == "__main__":
    main()