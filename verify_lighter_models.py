"""
Re-verification: does the light (15-tree) model's r0 vs r1 comparison on
the neighbor-available population actually support the paper's claim of
"a small but statistically significant improvement in macro-F1"?

This exists because Table I, once correctly split into paired-base blocks,
shows r0=0.280 vs r1=0.273 on the neighbor-available population -- a
DECREASE in the point estimate, not an increase. The paper's prose was
written against an earlier bootstrap run that may predate the switch from
the 300-tree to the 15-tree model. This script re-runs that specific
comparison against the CURRENT model configuration so the claim in the
text can be corrected or confirmed against real numbers, not memory.

Trains r0 on all light-model training rows, r1 on training rows with an
available neighbor (same population restriction as everywhere else in the
paper), evaluates BOTH on the same neighbor-available test subset (a
genuine paired comparison, matching what Table I now reports), and
bootstraps the r1-r0 difference on macro-F1, severe-class recall, and
false-safe rate.
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
MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}  # the paper's current primary model
N_BOOTSTRAP = 1000

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


def metrics_for(y_true, pred):
    macro_f1 = f1_score(y_true, pred, average="macro")
    is_severe = y_true.isin(["severe_down", "severe_up"])
    pred_series = pd.Series(pred, index=y_true.index)
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred_series == y_true)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & (pred_series == "stable")).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


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
    print(f"r0 train (full): {len(r0_train)} | r1 train (neighbor avail.): {len(r1_train)} | "
          f"test (paired, neighbor avail.): {len(test_common)}")

    print(f"\nTraining r0 (light config: {MODEL_PARAMS})...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print(f"Training r1 (light config: {MODEL_PARAMS})...")
    r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

    r0_pred = r0.predict(test_common[OWN_FEATURES])
    r1_pred = r1.predict(test_common[KUS_FEATURES])
    y = test_common["region"]

    f1_r0, rec_r0, fs_r0 = metrics_for(y, r0_pred)
    f1_r1, rec_r1, fs_r1 = metrics_for(y, r1_pred)

    print(f"\n--- Point estimates (neighbor-available test subset, n={len(test_common)}) ---")
    print(f"  r0 (base):       macro_f1={f1_r0:.4f}, severe_recall={rec_r0:.4f}, false_safe={fs_r0:.4f}")
    print(f"  r1 (graph path): macro_f1={f1_r1:.4f}, severe_recall={rec_r1:.4f}, false_safe={fs_r1:.4f}")
    print(f"  Diff (r1 - r0): macro_f1={f1_r1-f1_r0:+.4f}, severe_recall={rec_r1-rec_r0:+.4f}, "
          f"false_safe={fs_r1-fs_r0:+.4f}")
    print(f"  (These should match Table I's neighbor-available block: r0=0.280/0.701, r1=0.273/0.751,"
          f" if the table's numbers were also drawn from this exact model/split/population.)")

    print(f"\nRunning bootstrap (N={N_BOOTSTRAP})...")
    per_row = pd.DataFrame({"region": y.values, "r0_pred": r0_pred, "r1_pred": r1_pred})
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(per_row)
    diffs_f1, diffs_rec, diffs_fs = [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        sample = per_row.iloc[idx]
        f1_r0_b, rec_r0_b, fs_r0_b = metrics_for(sample["region"], sample["r0_pred"])
        f1_r1_b, rec_r1_b, fs_r1_b = metrics_for(sample["region"], sample["r1_pred"])
        diffs_f1.append(f1_r1_b - f1_r0_b)
        diffs_rec.append(rec_r1_b - rec_r0_b)
        diffs_fs.append(fs_r1_b - fs_r0_b)

    def report(name, values):
        values = np.array(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        verdict = "SIGNIFICANT" if excludes_zero else "NOT significant (CI includes 0)"
        print(f"  {name}: mean diff (r1 - r0) = {values.mean():+.4f}, 95% CI = [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    print("\n--- Bootstrap results (this is the number that determines whether the paper's ---")
    print("--- 'small but statistically significant improvement' claim is still true) ---")
    report("Macro F1", diffs_f1)
    report("Severe-class recall", diffs_rec)
    report("False-safe rate", diffs_fs)

    per_row.to_csv(OUT_DIR / "r0_r1_light_model_verification.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'r0_r1_light_model_verification.csv'}")

if __name__ == "__main__":
    main()