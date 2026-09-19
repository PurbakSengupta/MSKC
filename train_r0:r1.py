"""
Step 7: r0 vs r1 head-to-head comparison.

r0 (base route): predicts the 5-class ordinal ramp region from the
turbine's OWN recent history only (current + lagged wind speed/power,
wind direction, time-of-year). No neighbor information.

r1 (knowledge route): r0's features PLUS the upwind neighbor's simultaneous
power/wind speed (only computable where neighbor_available == True).

Both are trained on a chronological split (train: before 2024-11-01, test:
on/after 2024-11-01) to avoid leaking autocorrelated future information
into training. r0 is trained on ALL valid training rows (it doesn't need a
neighbor); r1 is trained only on training rows where a neighbor was
available. Both are EVALUATED on the same test subset -- test rows where a
neighbor was available -- since that's the only subset r1 can be scored on
at all, and comparing r0 there too makes it apples-to-apples.

This does NOT yet build the KUS/CAGT/AIES gate. It answers the prior
question: is there anything for a gate to gate on? If r1 doesn't beat r0
here, the gate has nothing useful to decide and we need to revisit the
neighbor feature before going further.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "Results" / "labeled_with_neighbor.csv"
OUT_DIR = ROOT_DIR / "Results"

RATED_POWER_KW = 2050.0
SPLIT_DATE = pd.Timestamp("2024-11-01")

REGION_ORDER = ["severe_down", "moderate_down", "stable", "moderate_up", "severe_up"]
REGION_TO_INT = {name: i - 2 for i, name in enumerate(REGION_ORDER)}  # -2..+2

RANDOM_STATE = 42


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


OWN_FEATURES = [
    "Wind speed (m/s)", "wind_dir_sin", "wind_dir_cos", "own_power_norm",
    "own_power_norm_lag1", "own_power_norm_lag2", "own_power_norm_lag3",
    "own_windspeed_lag1", "own_windspeed_lag2", "own_windspeed_lag3",
    "month", "hour",
]
NEIGHBOR_FEATURES = ["neighbor_power_norm", "neighbor_windspeed", "neighbor_distance_m"]


def evaluate(model, X_test, y_test, label):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    y_int = y_test.map(REGION_TO_INT)
    pred_int = pd.Series(pred, index=y_test.index).map(REGION_TO_INT)
    ordinal_mae = (y_int - pred_int).abs().mean()

    is_severe = y_test.isin(["severe_down", "severe_up"])
    pred_stable = pred == "stable"
    n_severe = is_severe.sum()
    false_safe_rate = (is_severe & pred_stable).sum() / n_severe if n_severe else float("nan")
    severe_recall = (is_severe & (pd.Series(pred, index=y_test.index) == y_test)).sum() / n_severe if n_severe else float("nan")

    print(f"\n--- {label} ---")
    print(f"  n_test = {len(y_test)}")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Macro F1:            {macro_f1:.4f}")
    print(f"  Ordinal MAE:         {ordinal_mae:.4f}  (0=perfect, integer region distance)")
    print(f"  Severe-class recall: {severe_recall:.4f}  (n_severe={n_severe})")
    print(f"  False-safe rate:     {false_safe_rate:.4f}  (severe predicted as 'stable' -- the dangerous error)")

    cm = confusion_matrix(y_test, pred, labels=REGION_ORDER)
    print(f"  Confusion matrix (rows=true, cols=pred), order {REGION_ORDER}:")
    print(pd.DataFrame(cm, index=REGION_ORDER, columns=REGION_ORDER).to_string())

    return {"label": label, "n_test": len(y_test), "accuracy": acc, "macro_f1": macro_f1,
            "ordinal_mae": ordinal_mae, "severe_recall": severe_recall, "false_safe_rate": false_safe_rate}


def main():
    print("Loading data...")
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df["turbine_id"] = df["turbine_id"].astype(str)
    print(f"Loaded {len(df)} rows.")

    df = add_own_features(df)
    df["neighbor_power_norm"] = df["neighbor_power_kw"] / RATED_POWER_KW

    # Valid rows: has a region label, and all own-lag features present
    # (first 3 rows of each turbine's year will be NaN here -- negligible).
    valid = df["region"].notna()
    for col in OWN_FEATURES:
        valid &= df[col].notna()
    df = df[valid].copy()
    print(f"Rows with valid label + own features: {len(df)}")

    train_mask = df["timestamp"] < SPLIT_DATE
    test_mask = ~train_mask
    print(f"Train rows: {train_mask.sum()}  |  Test rows: {test_mask.sum()}")

    # ---- r0: trained on ALL valid training rows, own features only ----
    r0_train = df[train_mask]
    r0 = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
                                 n_jobs=-1, class_weight="balanced_subsample")
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])
    print(f"\nr0 trained on {len(r0_train)} rows (own features only, class-balanced).")

    # ---- r1: trained only on training rows with an available neighbor ----
    r1_train = df[train_mask & df["neighbor_available"]]
    r1_features = OWN_FEATURES + NEIGHBOR_FEATURES
    r1 = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
                                 n_jobs=-1, class_weight="balanced_subsample")
    r1.fit(r1_train[r1_features], r1_train["region"])
    print(f"r1 trained on {len(r1_train)} rows (own + neighbor features, class-balanced).")

    # ---- Evaluate BOTH on the same test subset: neighbor-available test rows ----
    test_common = df[test_mask & df["neighbor_available"]]
    print(f"\nCommon test subset (neighbor available at test time): {len(test_common)} rows")
    print(test_common["region"].value_counts().reindex(REGION_ORDER).to_string())

    results = []

    # Naive majority-class baseline, explicitly, so we can never again
    # mistake "beats majority-class guessing" for "the model learned
    # anything." Predicts the single most frequent training-set class for
    # every test row.
    majority_class = r0_train["region"].value_counts().idxmax()
    majority_pred = pd.Series([majority_class] * len(test_common), index=test_common.index)
    print(f"\n--- Naive majority-class baseline (always predicts '{majority_class}') ---")
    maj_acc = accuracy_score(test_common["region"], majority_pred)
    maj_f1 = f1_score(test_common["region"], majority_pred, average="macro")
    print(f"  Accuracy: {maj_acc:.4f}   Macro F1: {maj_f1:.4f}   (severe recall and false-safe rate are "
          f"trivially 0.0 and 1.0 by construction)")
    results.append({"label": f"majority-class baseline ('{majority_class}')", "n_test": len(test_common),
                     "accuracy": maj_acc, "macro_f1": maj_f1, "ordinal_mae": None,
                     "severe_recall": 0.0, "false_safe_rate": 1.0})

    results.append(evaluate(r0, test_common[OWN_FEATURES], test_common["region"], "r0 (base, no knowledge, class-balanced)"))
    results.append(evaluate(r1, test_common[r1_features], test_common["region"], "r1 (knowledge-injected, class-balanced)"))

    # Save per-row predictions from both models on the common test subset,
    # plus state variables needed for subgroup analysis, so we can run
    # significance testing / subgroup analysis without retraining.
    predictions_out = test_common[["site", "turbine_id", "timestamp", "region",
                                    "Wind speed (m/s)", "neighbor_windspeed",
                                    "neighbor_distance_m"]].copy()
    predictions_out["r0_pred"] = r0.predict(test_common[OWN_FEATURES])
    predictions_out["r1_pred"] = r1.predict(test_common[r1_features])
    predictions_out.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print(f"\nSaved per-row test predictions to {OUT_DIR / 'test_predictions.csv'}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "r0_vs_r1_results.csv", index=False)
    print(f"\nSummary written to {OUT_DIR / 'r0_vs_r1_results.csv'}")
    print(results_df.to_string())


if __name__ == "__main__":
    main()