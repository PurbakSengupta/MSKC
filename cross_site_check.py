"""
Quick check 2: cross-site portability.

Everything so far was trained and tested on a chronological split that
mixes BOTH sites in both train and test. This instead trains r0, r1, and
the KUS gate using ONLY Kelmarsh data (all of Kelmarsh's year, not just the
pre-Nov-1 period -- we want maximum training signal from the single
training site here), then evaluates at the FIXED tau=0.4 operating point
(not re-tuned) on ALL of Penmanshiel's data -- a site with a different
turbine layout, spacing, and geometry the model has never seen.

This is the actual portability question the PhD plan cares about (ST6/H5):
does the selector principle transfer across sites, not just across time
within one farm layout. tau is deliberately NOT re-tuned for Penmanshiel --
the point is to test whether a threshold chosen on one site still works
reasonably on an unseen one, which is the realistic deployment scenario.
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

TRAIN_SITE = "Kelmarsh"
TEST_SITE = "Penmanshiel"
FIXED_TAU = 0.4
MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}  # light config, matches the chosen paper model


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

    r0_train = df[df["site"] == TRAIN_SITE]
    r1_train = df[(df["site"] == TRAIN_SITE) & df["neighbor_available"]]
    test_common = df[(df["site"] == TEST_SITE) & df["neighbor_available"]]
    print(f"Train site: {TRAIN_SITE} ({len(r0_train)} rows, {len(r1_train)} with neighbor)")
    print(f"Test site:  {TEST_SITE} ({len(test_common)} rows with neighbor)")

    print(f"\nTraining r0 on {TRAIN_SITE} only...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print(f"Training r1 on {TRAIN_SITE} only...")
    r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

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
    print(f"\nKUS training data ({TRAIN_SITE}): {len(kus_data)} rows, "
          f"{kus_data['r1_helps'].mean()*100:.1f}% where r1 improved on r0 (OOB)")

    print("Training KUS estimator...")
    kus_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE,
                                        n_jobs=-1, class_weight="balanced_subsample")
    kus_model.fit(kus_data[KUS_FEATURES], kus_data["r1_helps"])
    kus_class_idx = list(kus_model.classes_).index(1)

    # ---- Evaluate on the UNSEEN test site ----
    r0_test_pred = r0.predict(test_common[OWN_FEATURES])
    r1_test_pred = r1.predict(test_common[KUS_FEATURES])
    kus_score = kus_model.predict_proba(test_common[KUS_FEATURES])[:, kus_class_idx]

    f1_r0, rec_r0, fs_r0 = metrics_for(test_common["region"], r0_test_pred)
    f1_r1, rec_r1, fs_r1 = metrics_for(test_common["region"], r1_test_pred)

    inject = kus_score > FIXED_TAU
    combined_pred = np.where(inject, r1_test_pred, r0_test_pred)
    f1_gate, rec_gate, fs_gate = metrics_for(test_common["region"], combined_pred)
    injection_rate = inject.mean()

    print(f"\n{'=' * 70}")
    print(f"Cross-site result: trained on {TRAIN_SITE}, tested on {TEST_SITE}, gate fixed at tau={FIXED_TAU}")
    print(f"{'=' * 70}")
    print(f"{'policy':<20}{'inject_rate':>12}{'macro_f1':>10}{'severe_recall':>15}{'false_safe':>12}")
    print(f"{'pure r0':<20}{0.0:>12.3f}{f1_r0:>10.4f}{rec_r0:>15.4f}{fs_r0:>12.4f}")
    print(f"{'pure r1':<20}{1.0:>12.3f}{f1_r1:>10.4f}{rec_r1:>15.4f}{fs_r1:>12.4f}")
    print(f"{'gate (tau=0.4)':<20}{injection_rate:>12.3f}{f1_gate:>10.4f}{rec_gate:>15.4f}{fs_gate:>12.4f}")

    out = pd.DataFrame([
        {"policy": "pure_r0", "injection_rate": 0.0, "macro_f1": f1_r0, "severe_recall": rec_r0, "false_safe_rate": fs_r0},
        {"policy": "pure_r1", "injection_rate": 1.0, "macro_f1": f1_r1, "severe_recall": rec_r1, "false_safe_rate": fs_r1},
        {"policy": "gate_tau04", "injection_rate": injection_rate, "macro_f1": f1_gate, "severe_recall": rec_gate, "false_safe_rate": fs_gate},
    ])
    out.to_csv(OUT_DIR / "cross_site_check.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'cross_site_check.csv'}")


if __name__ == "__main__":
    main()