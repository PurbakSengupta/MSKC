"""
Quick check 1: OOB-reliability sensitivity.

The light config (n_estimators=15) triggered sklearn's "too few trees for
reliable OOB estimates" warning when training with oob_score=True. This
retrains the full gate pipeline (r0, r1, KUS estimator, frontier) at
n_estimators in {15, 25, 50} (max_depth fixed at 5, matching the light
config's shallow-tree choice) and reports the tau=0.4 operating point for
each, so we can see whether the chosen operating point's metrics are
sensitive to tree count or are stable once OOB estimates become reliable.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "Results" / "labeled_with_neighbor.csv"
OUT_DIR = ROOT_DIR / "Results"

RATED_POWER_KW = 2050.0
SPLIT_DATE = pd.Timestamp("2024-11-01")
RANDOM_STATE = 42

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

TREE_COUNTS_TO_CHECK = [15, 25, 50]
FIXED_TAU = 0.4


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
    false_safe = (is_severe & (pred_series == "stable")).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


def run_for_tree_count(n_estimators, r0_train, r1_train, test_common):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r0 = RandomForestClassifier(n_estimators=n_estimators, max_depth=5,
                                     random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     oob_score=True, bootstrap=True)
        r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

        r1 = RandomForestClassifier(n_estimators=n_estimators, max_depth=5,
                                     random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     oob_score=True, bootstrap=True)
        r1.fit(r1_train[KUS_FEATURES], r1_train["region"])
        oob_warning = any("OOB" in str(w.message) for w in caught)

    r0_oob_probs = r0.oob_decision_function_
    r0_oob_pred = pd.Series(r0.classes_[np.nanargmax(r0_oob_probs, axis=1)], index=r0_train.index)
    r1_oob_probs = r1.oob_decision_function_
    r1_oob_pred = pd.Series(r1.classes_[np.nanargmax(r1_oob_probs, axis=1)], index=r1_train.index)

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

    r0_test_pred = r0.predict(test_common[OWN_FEATURES])
    r1_test_pred = r1.predict(test_common[KUS_FEATURES])
    kus_score = kus_model.predict_proba(test_common[KUS_FEATURES])[:, kus_class_idx]

    inject = kus_score > FIXED_TAU
    combined_pred = np.where(inject, r1_test_pred, r0_test_pred)
    f1, rec, fs = metrics_for(test_common["region"], combined_pred)

    f1_pure_r1, rec_pure_r1, fs_pure_r1 = metrics_for(test_common["region"], r1_test_pred)

    return {
        "n_estimators": n_estimators, "oob_warning_triggered": oob_warning,
        "kus_positive_rate": round(kus_data["r1_helps"].mean(), 4),
        "injection_rate_at_tau04": round(inject.mean(), 4),
        "macro_f1_at_tau04": round(f1, 4), "severe_recall_at_tau04": round(rec, 4),
        "false_safe_at_tau04": round(fs, 4),
        "macro_f1_pure_r1": round(f1_pure_r1, 4), "severe_recall_pure_r1": round(rec_pure_r1, 4),
        "false_safe_pure_r1": round(fs_pure_r1, 4),
    }


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

    results = []
    for n_trees in TREE_COUNTS_TO_CHECK:
        print(f"\nRunning n_estimators={n_trees}...")
        res = run_for_tree_count(n_trees, r0_train, r1_train, test_common)
        print(f"  OOB warning triggered: {res['oob_warning_triggered']}")
        print(f"  tau=0.4: injection_rate={res['injection_rate_at_tau04']}, "
              f"macro_f1={res['macro_f1_at_tau04']}, severe_recall={res['severe_recall_at_tau04']}, "
              f"false_safe={res['false_safe_at_tau04']}")
        print(f"  pure r1: macro_f1={res['macro_f1_pure_r1']}, "
              f"severe_recall={res['severe_recall_pure_r1']}, false_safe={res['false_safe_pure_r1']}")
        results.append(res)

    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "oob_reliability_check.csv", index=False)
    print(f"\n{'=' * 70}\nFull comparison:")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'oob_reliability_check.csv'}")

if __name__ == "__main__":
    main()