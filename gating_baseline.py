"""
Baseline comparison: does the LEARNED KUS gate actually beat naive
alternatives at the same injection rate, or would a trivial gating rule
have done just as well?

Three competing gating strategies, all constrained to inject at
approximately the same rate as the learned gate (~36.6%) so the comparison
is fair (matched cost):

  1. RANDOM gating: route ~36.6% of rows to r1 uniformly at random, no
     information used. Repeated over many random seeds to get a
     distribution, not a single lucky/unlucky draw. If the learned gate
     doesn't clearly beat the bulk of this distribution, "learned" is not
     adding anything over chance.

  2. HEURISTIC gating: a simple, hand-picked, non-learned rule --
     inject when |own wind speed - neighbor wind speed| exceeds a
     threshold, calibrated to hit ~36.6% injection rate. This is the
     "intuitive human-designed rule" a reviewer might propose as a
     simpler alternative to training a classifier.

  3. LEARNED (KUS) gating: the actual proposed method from steps 9/10.

All three are evaluated against the same pure r0 / pure r1 endpoints, on
the same test set, so this is a genuine head-to-head under matched cost,
not just "our method vs itself."
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

MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}  # light config, matches the chosen paper model
TARGET_INJECTION_RATE = 0.366  # matched to the learned gate's tau=0.4 rate
N_RANDOM_TRIALS = 200


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
    test_common = df[~train_mask & df["neighbor_available"]].copy()
    n_test = len(test_common)
    print(f"Test set: {n_test} rows, target injection rate: {TARGET_INJECTION_RATE:.1%}")

    print("\nTraining r0...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print("Training r1...")
    r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample", **MODEL_PARAMS)
    r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

    r0_pred = r0.predict(test_common[OWN_FEATURES])
    r1_pred = r1.predict(test_common[KUS_FEATURES])
    y = test_common["region"]

    f1_r0, rec_r0, fs_r0 = metrics_for(y, r0_pred)
    f1_r1, rec_r1, fs_r1 = metrics_for(y, r1_pred)
    print(f"\npure r0: macro_f1={f1_r0:.4f}, severe_recall={rec_r0:.4f}, false_safe={fs_r0:.4f}")
    print(f"pure r1: macro_f1={f1_r1:.4f}, severe_recall={rec_r1:.4f}, false_safe={fs_r1:.4f}")

    # ---- Baseline 1: RANDOM gating, many trials ----
    print(f"\nRunning RANDOM gating ({N_RANDOM_TRIALS} trials at {TARGET_INJECTION_RATE:.1%})...")
    rng = np.random.default_rng(RANDOM_STATE)
    random_f1s, random_recs, random_fss = [], [], []
    for _ in range(N_RANDOM_TRIALS):
        inject = rng.random(n_test) < TARGET_INJECTION_RATE
        pred = np.where(inject, r1_pred, r0_pred)
        f1, rec, fs = metrics_for(y, pred)
        random_f1s.append(f1)
        random_recs.append(rec)
        random_fss.append(fs)
    random_f1s, random_recs, random_fss = map(np.array, (random_f1s, random_recs, random_fss))
    print(f"  RANDOM gating: macro_f1 mean={random_f1s.mean():.4f} "
          f"[{np.percentile(random_f1s, 2.5):.4f}, {np.percentile(random_f1s, 97.5):.4f}]")
    print(f"  RANDOM gating: severe_recall mean={random_recs.mean():.4f} "
          f"[{np.percentile(random_recs, 2.5):.4f}, {np.percentile(random_recs, 97.5):.4f}]")
    print(f"  RANDOM gating: false_safe mean={random_fss.mean():.4f} "
          f"[{np.percentile(random_fss, 2.5):.4f}, {np.percentile(random_fss, 97.5):.4f}]")

    # ---- Baseline 2: HEURISTIC threshold gating (wind-speed divergence) ----
    print(f"\nCalibrating HEURISTIC threshold to hit ~{TARGET_INJECTION_RATE:.1%} injection rate...")
    divergence = (test_common["Wind speed (m/s)"] - test_common["neighbor_windspeed"]).abs()
    threshold = divergence.quantile(1 - TARGET_INJECTION_RATE)
    heuristic_inject = divergence > threshold
    heuristic_pred = np.where(heuristic_inject, r1_pred, r0_pred)
    f1_heur, rec_heur, fs_heur = metrics_for(y, heuristic_pred)
    print(f"  Threshold: {threshold:.3f} m/s, actual injection rate: {heuristic_inject.mean():.3f}")
    print(f"  HEURISTIC gating: macro_f1={f1_heur:.4f}, severe_recall={rec_heur:.4f}, false_safe={fs_heur:.4f}")

    # ---- Baseline 3: LEARNED (KUS) gating, retrained here for a clean comparison ----
    print("\nTraining LEARNED KUS gate...")
    r0_oob = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r0_oob.fit(r0_train[OWN_FEATURES], r0_train["region"])
    r1_oob = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r1_oob.fit(r1_train[KUS_FEATURES], r1_train["region"])

    r0_oob_pred = pd.Series(r0_oob.classes_[np.nanargmax(r0_oob.oob_decision_function_, axis=1)],
                             index=r0_train.index)
    r1_oob_pred = pd.Series(r1_oob.classes_[np.nanargmax(r1_oob.oob_decision_function_, axis=1)],
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
    kus_score = kus_model.predict_proba(test_common[KUS_FEATURES])[:, kus_class_idx]
    learned_inject = kus_score > 0.4
    learned_pred = np.where(learned_inject, r1_pred, r0_pred)
    f1_learn, rec_learn, fs_learn = metrics_for(y, learned_pred)
    print(f"  LEARNED gating: injection_rate={learned_inject.mean():.3f}, "
          f"macro_f1={f1_learn:.4f}, severe_recall={rec_learn:.4f}, false_safe={fs_learn:.4f}")

    # ---- Summary ----
    print(f"\n{'=' * 80}\nSummary (all at ~{TARGET_INJECTION_RATE:.1%} injection rate, matched cost)")
    print(f"{'=' * 80}")
    print(f"{'policy':<30}{'inject_rate':>12}{'macro_f1':>10}{'severe_recall':>15}{'false_safe':>12}")
    print(f"{'pure r0':<30}{0.0:>12.3f}{f1_r0:>10.4f}{rec_r0:>15.4f}{fs_r0:>12.4f}")
    print(f"{'pure r1':<30}{1.0:>12.3f}{f1_r1:>10.4f}{rec_r1:>15.4f}{fs_r1:>12.4f}")
    print(f"{'random gating (mean)':<30}{TARGET_INJECTION_RATE:>12.3f}{random_f1s.mean():>10.4f}"
          f"{random_recs.mean():>15.4f}{random_fss.mean():>12.4f}")
    print(f"{'heuristic threshold gating':<30}{heuristic_inject.mean():>12.3f}{f1_heur:>10.4f}"
          f"{rec_heur:>15.4f}{fs_heur:>12.4f}")
    print(f"{'LEARNED (KUS) gating':<30}{learned_inject.mean():>12.3f}{f1_learn:>10.4f}"
          f"{rec_learn:>15.4f}{fs_learn:>12.4f}")

    pct_random_beaten_f1 = 100 * (random_f1s < f1_learn).mean()
    pct_random_beaten_rec = 100 * (random_recs < rec_learn).mean()
    print(f"\nLearned gate beats {pct_random_beaten_f1:.0f}% of random-gating trials on macro-F1, "
          f"{pct_random_beaten_rec:.0f}% on severe recall.")

    out = pd.DataFrame([
        {"policy": "pure_r0", "injection_rate": 0.0, "macro_f1": f1_r0, "severe_recall": rec_r0, "false_safe_rate": fs_r0},
        {"policy": "pure_r1", "injection_rate": 1.0, "macro_f1": f1_r1, "severe_recall": rec_r1, "false_safe_rate": fs_r1},
        {"policy": "random_gating_mean", "injection_rate": TARGET_INJECTION_RATE, "macro_f1": random_f1s.mean(), "severe_recall": random_recs.mean(), "false_safe_rate": random_fss.mean()},
        {"policy": "heuristic_threshold", "injection_rate": heuristic_inject.mean(), "macro_f1": f1_heur, "severe_recall": rec_heur, "false_safe_rate": fs_heur},
        {"policy": "learned_kus_gate", "injection_rate": learned_inject.mean(), "macro_f1": f1_learn, "severe_recall": rec_learn, "false_safe_rate": fs_learn},
    ])
    out.to_csv(OUT_DIR / "gating_baseline_comparison.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'gating_baseline_comparison.csv'}")

    # Save per-row data for a paired bootstrap comparison of learned vs
    # heuristic specifically (the two closest competitors), without
    # retraining.
    per_row = pd.DataFrame({
        "region": y.values,
        "r0_pred": r0_pred,
        "r1_pred": r1_pred,
        "heuristic_inject": heuristic_inject.values,
        "learned_inject": learned_inject,
    })
    per_row.to_csv(OUT_DIR / "gating_baseline_per_row.csv", index=False)
    print(f"Wrote {OUT_DIR / 'gating_baseline_per_row.csv'} (for paired bootstrap)")

if __name__ == "__main__":
    main()