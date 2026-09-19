"""
Second knowledge route: physics-informed power-curve residual (r2).

Maps to CAKI's "domain-constrained path" (physical relationships as soft
constraints) -- distinct from r1, which is the "graph path" (cross-turbine
relational knowledge). Unlike r1, this route requires NO communication: the
expected power curve is a small per-turbine lookup fit once on training
data, so at inference time it costs a local computation only.

Method: per turbine, fit isotonic regression (monotonic, wind speed ->
normalized power) on TRAINING rows only. residual = observed - expected.
Isotonic regression assumes non-decreasing power with wind speed, which is
a simplification near cut-out (~25 m/s) where real power drops for safety
shutdown -- noted as a caveat, not hidden, since cut-out events are rare in
this dataset (the storm events characterized earlier).

r2 = OWN_FEATURES + [power_curve_residual], trained on ALL training rows
(no neighbor-availability restriction, since this route doesn't need one)
and evaluated on the FULL test set -- not the neighbor-available subset --
since that's r2's genuine advantage: universal availability.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
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
MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}  # same light config as r0/r1

OWN_FEATURES = [
    "Wind speed (m/s)", "wind_dir_sin", "wind_dir_cos", "own_power_norm",
    "own_power_norm_lag1", "own_power_norm_lag2", "own_power_norm_lag3",
    "own_windspeed_lag1", "own_windspeed_lag2", "own_windspeed_lag3",
    "month", "hour",
]
R2_FEATURES = OWN_FEATURES + ["power_curve_residual"]

N_BOOTSTRAP = 1000


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

    valid = df["region"].notna()
    for col in OWN_FEATURES:
        valid &= df[col].notna()
    df = df[valid].copy()

    train_mask = df["timestamp"] < SPLIT_DATE

    print("\nFitting per-turbine isotonic power curves (train data only)...")
    df["power_curve_residual"] = np.nan
    for (site, tid), g_train in df[train_mask].groupby(["site", "turbine_id"]):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(g_train["Wind speed (m/s)"], g_train["own_power_norm"])

        mask = (df["site"] == site) & (df["turbine_id"] == tid)
        expected = iso.predict(df.loc[mask, "Wind speed (m/s)"])
        df.loc[mask, "power_curve_residual"] = df.loc[mask, "own_power_norm"].values - expected

    print(f"Residual summary (train rows): "
          f"mean={df.loc[train_mask, 'power_curve_residual'].mean():.4f}, "
          f"std={df.loc[train_mask, 'power_curve_residual'].std():.4f}")

    r0_train = df[train_mask]
    r2_train = df[train_mask]
    test = df[~train_mask]

    print(f"\nTrain rows: {len(r0_train)} | Test rows (full): {len(test)}")

    print("\nTraining r0 (own features only)...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print("Training r2 (own features + physics residual)...")
    r2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r2.fit(r2_train[R2_FEATURES], r2_train["region"])

    r0_pred = r0.predict(test[OWN_FEATURES])
    r2_pred = r2.predict(test[R2_FEATURES])
    y = test["region"]

    f1_r0, rec_r0, fs_r0 = metrics_for(y, r0_pred)
    f1_r2, rec_r2, fs_r2 = metrics_for(y, r2_pred)

    print(f"\n--- Point estimates (full test set, n={len(test)}) ---")
    print(f"  r0 (no knowledge):        macro_f1={f1_r0:.4f}, severe_recall={rec_r0:.4f}, false_safe={fs_r0:.4f}")
    print(f"  r2 (+ physics residual):  macro_f1={f1_r2:.4f}, severe_recall={rec_r2:.4f}, false_safe={fs_r2:.4f}")
    print(f"  Diff (r2 - r0): macro_f1={f1_r2-f1_r0:+.4f}, severe_recall={rec_r2-rec_r0:+.4f}, "
          f"false_safe={fs_r2-fs_r0:+.4f}")

    importances = pd.Series(r2.feature_importances_, index=R2_FEATURES).sort_values(ascending=False)
    print(f"\nr2 feature importances:\n{importances.to_string()}")

    print(f"\nRunning bootstrap (N={N_BOOTSTRAP})...")
    per_row = pd.DataFrame({"region": y.values, "r0_pred": r0_pred, "r2_pred": r2_pred})
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(per_row)
    diffs_f1, diffs_rec, diffs_fs = [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        sample = per_row.iloc[idx]
        f1_r0_b, rec_r0_b, fs_r0_b = metrics_for(sample["region"], sample["r0_pred"])
        f1_r2_b, rec_r2_b, fs_r2_b = metrics_for(sample["region"], sample["r2_pred"])
        diffs_f1.append(f1_r2_b - f1_r0_b)
        diffs_rec.append(rec_r2_b - rec_r0_b)
        diffs_fs.append(fs_r2_b - fs_r0_b)

    def report(name, values):
        values = np.array(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        verdict = "SIGNIFICANT" if excludes_zero else "not significant (CI includes 0)"
        print(f"  {name}: mean diff (r2 - r0) = {values.mean():+.4f}, 95% CI = [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    print("\n--- Bootstrap results ---")
    report("Macro F1", diffs_f1)
    report("Severe-class recall", diffs_rec)
    report("False-safe rate", diffs_fs)

    per_row.to_csv(OUT_DIR / "r2_physics_test_predictions.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'r2_physics_test_predictions.csv'}")


if __name__ == "__main__":
    main()