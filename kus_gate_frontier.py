"""
Step 9 (corrected v2): fixes the leakage WITHOUT changing r0/r1's own
training data, so "always-inject-r1" on the test set stays byte-identical
to Table III's r1 row and every other table in the paper.

The previous fix (v1) carved a validation slice out of the overall
training period, which accidentally shrank r0/r1's own training window by
six weeks, making them different models than the ones behind every other
table (visible as the "Always-r1" mismatch: 0.2809/0.7100 here vs.
0.273/0.751 in Table III).

This version instead:
  - Trains r0 and r1 on the EXACT SAME full pre-SPLIT_DATE data as before
    (identical to whatever produced Tables III-VI). Their OOB predictions
    are still computed over that full training set, unchanged.
  - Splits ONLY the KUS gate's own training rows (r1_train) into a
    KUS-fit slice and a KUS-validation slice (chronological, last ~6
    weeks of the training period reserved for validation).
  - Trains the KUS classifier on the KUS-fit slice only.
  - Sweeps TAU_GRID on the KUS-validation slice, using r0/r1's OOB
    predictions on those rows (a genuine out-of-sample read of r0/r1's
    behavior there, since OOB is precisely an unbiased per-row estimate)
    and the KUS model's score on those same rows (rows it was never
    trained on).
  - Selects tau_star from that validation sweep only.
  - Evaluates tau_star exactly once on the untouched TEST set, using r0
    and r1 as originally trained, so this number is directly comparable
    to every other table in the paper.
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
SPLIT_DATE = pd.Timestamp("2024-11-01")           # unchanged: train+ / test boundary
KUS_VAL_WINDOW_DAYS = 42                          # ~6 weeks, carved from WITHIN r1_train only
KUS_VAL_START = SPLIT_DATE - pd.Timedelta(days=KUS_VAL_WINDOW_DAYS)
RANDOM_STATE = 42
N_BOOTSTRAP = 1000

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

TAU_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MODEL_PARAMS = {"n_estimators": 15, "max_depth": 5}


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
    y_true = pd.Series(y_true).reset_index(drop=True)
    pred = pd.Series(pred).reset_index(drop=True)
    macro_f1 = f1_score(y_true, pred, average="macro")
    is_severe = y_true.isin(["severe_down", "severe_up"])
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred == y_true)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & (pred == "stable")).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


def sweep_frontier(kus_score, r0_pred, r1_pred, y_true):
    rows = []
    for tau in TAU_GRID:
        inject = kus_score > tau
        combined = np.where(inject, r1_pred, r0_pred)
        f1, rec, fs = metrics_for(y_true, combined)
        rows.append({"tau": tau, "injection_rate": inject.mean(),
                     "macro_f1": f1, "severe_recall": rec, "false_safe_rate": fs})
    return pd.DataFrame(rows)


def select_tau_star(val_frontier, n_severe_val):
    """Largest tau (lowest injection rate) whose false-safe rate on
    VALIDATION is no worse than always-injecting r1's false-safe rate on
    that same validation set, allowing a small, explicit tolerance rather
    than an exact zero-tolerance match. The tolerance is sized to roughly
    one additional missed severe event on the validation slice, not an
    arbitrary constant, so it can be reported and defended as such."""
    always_inject_row = val_frontier.loc[val_frontier["tau"] == 0.0].iloc[0]
    baseline_fs = always_inject_row["false_safe_rate"]
    tolerance = 1.5 / n_severe_val  # ~1 additional missed severe event, explicit and reportable
    ok = val_frontier[val_frontier["false_safe_rate"] <= baseline_fs + tolerance]
    return ok.sort_values("tau", ascending=False).iloc[0]["tau"], baseline_fs, tolerance


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

    train_mask = df["timestamp"] < SPLIT_DATE      # SAME boundary as Tables III-VI
    test_mask = ~train_mask

    r0_train = df[train_mask]
    r1_train = df[train_mask & df["neighbor_available"]]
    test_common = df[test_mask & df["neighbor_available"]]
    print(f"r0 train: {len(r0_train)} | r1 train: {len(r1_train)} | "
          f"test (neighbor available): {len(test_common)} ")

    print("\nTraining r0 (with OOB, light config, FULL pre-split data, unchanged)...")
    r0 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r0.fit(r0_train[OWN_FEATURES], r0_train["region"])

    print("Training r1 (with OOB, light config, FULL pre-split data, unchanged)...")
    r1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, **MODEL_PARAMS)
    r1.fit(r1_train[KUS_FEATURES], r1_train["region"])

    # ---- OOB predictions over the FULL training set, unchanged mechanism ----
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

    # ---- NEW: split kus_data itself into KUS-fit and KUS-validation ----
    kus_fit_mask = kus_data["timestamp"] < KUS_VAL_START
    kus_val_mask = ~kus_fit_mask
    kus_fit = kus_data[kus_fit_mask]
    kus_val = kus_data[kus_val_mask]
    print(f"\nKUS fit rows: {len(kus_fit)} | KUS validation rows (held out from KUS training only): {len(kus_val)}")
    if len(kus_val) < 2000:
        print(f"WARNING: KUS validation slice is small ({len(kus_val)} rows); "
              f"consider increasing KUS_VAL_WINDOW_DAYS.")

    print(f"KUS-fit data: {kus_fit['r1_helps'].mean()*100:.1f}% where r1 improved on r0 (OOB)")
    print("Training KUS estimator on KUS-fit rows only...")
    kus_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE,
                                        n_jobs=-1, class_weight="balanced_subsample")
    kus_model.fit(kus_fit[KUS_FEATURES], kus_fit["r1_helps"])
    kus_class_idx = list(kus_model.classes_).index(1)

    # ---- Sweep tau on the KUS-validation slice, using OOB r0/r1 preds there ----
    print("\nComputing frontier on KUS-VALIDATION slice (used only to select tau; "
          "r0/r1 predictions here are their OOB predictions, not fit on these exact rows' labels)...")
    kus_val_score = kus_model.predict_proba(kus_val[KUS_FEATURES])[:, kus_class_idx]
    val_frontier = sweep_frontier(kus_val_score, kus_val["r0_oob_pred"].values,
                                   kus_val["r1_oob_pred"].values, kus_val["region"].values)
    print(val_frontier.to_string(index=False))
    val_frontier.to_csv(OUT_DIR / "aies_frontier_validation.csv", index=False)

    tau_star, val_baseline_fs, tolerance_used = select_tau_star(val_frontier, int((kus_val["region"].isin(["severe_down","severe_up"])).sum()))
    print(f"\nSelected tau_star = {tau_star} on KUS-VALIDATION "
          f"(largest tau with false_safe_rate <= always-inject's {val_baseline_fs:.4f} "
          f"plus a tolerance of {tolerance_used:.4f}, sized to ~1 extra missed severe event)")

    # ---- Evaluate tau_star ONCE on TEST, using r0/r1 exactly as trained for Tables III-VI ----
    print("\nEvaluating tau_star ONCE on TEST set (r0/r1 identical to Tables III-VI)...")
    r0_test_pred = r0.predict(test_common[OWN_FEATURES])
    r1_test_pred = r1.predict(test_common[KUS_FEATURES])
    kus_test_score = kus_model.predict_proba(test_common[KUS_FEATURES])[:, kus_class_idx]
    y_test = test_common["region"].values

    inject_test = kus_test_score > tau_star
    gate_pred = np.where(inject_test, r1_test_pred, r0_test_pred)
    f1_gate, rec_gate, fs_gate = metrics_for(y_test, gate_pred)
    f1_always, rec_always, fs_always = metrics_for(y_test, r1_test_pred)
    injection_rate_test = inject_test.mean()

    print(f"  Injection rate on TEST at tau_star={tau_star}: {injection_rate_test:.4f}")
    print(f"  Gate:       macro_f1={f1_gate:.4f}, severe_recall={rec_gate:.4f}, false_safe={fs_gate:.4f}")
    print(f"  Always-r1:  macro_f1={f1_always:.4f}, severe_recall={rec_always:.4f}, false_safe={fs_always:.4f}")
    print(f"  (Always-r1 here should match Table III's r1 row: 0.273 / 0.751)")
    print(f"  Diff (gate - always_r1): macro_f1={f1_gate-f1_always:+.4f}, "
          f"severe_recall={rec_gate-rec_always:+.4f}, false_safe={fs_gate-fs_always:+.4f}")

    print(f"\nBootstrapping tau_star vs always-inject on TEST (N={N_BOOTSTRAP})...")
    per_row = pd.DataFrame({"region": y_test, "gate_pred": gate_pred, "r1_pred": r1_test_pred})
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(per_row)
    diffs_f1, diffs_rec, diffs_fs = [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        s = per_row.iloc[idx]
        gf1, grec, gfs = metrics_for(s["region"], s["gate_pred"])
        af1, arec, afs = metrics_for(s["region"], s["r1_pred"])
        diffs_f1.append(gf1 - af1)
        diffs_rec.append(grec - arec)
        diffs_fs.append(gfs - afs)

    def report(name, values):
        values = np.array(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "NOT significant (CI includes 0)"
        print(f"  {name}: mean={values.mean():+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}] -> {sig}")

    report("Macro F1      ", diffs_f1)
    report("Severe recall ", diffs_rec)
    report("False-safe    ", diffs_fs)

    print("\nComputing full frontier on TEST for the illustrative figure only (not used for selection)...")
    test_frontier = sweep_frontier(kus_test_score, r0_test_pred, r1_test_pred, y_test)
    test_frontier.to_csv(OUT_DIR / "aies_frontier_test.csv", index=False)
    print(test_frontier.to_string(index=False))


if __name__ == "__main__":
    main()