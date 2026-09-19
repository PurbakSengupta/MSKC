"""
Step 8: (a) Is the r0-vs-r1 aggregate difference statistically real, or
noise? (b) Is there an identifiable subgroup where r1 clearly wins, even if
the aggregate is a wash?

(a) Bootstrap: resample the test set with replacement N times, recompute
    macro-F1(r1) - macro-F1(r0), severe-recall(r1) - severe-recall(r0), and
    false-safe-rate(r1) - false-safe-rate(r0) each time, report the 95% CI
    of each difference. If a CI excludes 0, that difference is real at this
    sample size. If it includes 0, we cannot claim r1 differs from r0 on
    that metric here.

(b) Subgroup: bin test rows by |own wind speed - neighbor wind speed| (a
    proxy for "the neighbor is currently showing something different from
    what's visible locally right now") and compare r0 vs r1 macro-F1 within
    each bin. This is the direct empirical basis for KUS: if there's a
    divergence-magnitude regime where r1 clearly helps, that regime is
    where injection should be gated ON.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "Results" / "test_predictions.csv"
OUT_DIR = ROOT_DIR / "Results"

N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def metrics_for(y_true, pred):
    macro_f1 = f1_score(y_true, pred, average="macro")
    is_severe = y_true.isin(["severe_down", "severe_up"])
    pred_stable = pred == "stable"
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred == y_true)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & pred_stable).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


def bootstrap_diffs(df, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    diffs = {"macro_f1": [], "severe_recall": [], "false_safe_rate": []}

    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        sample = df.iloc[idx]
        f1_r0, rec_r0, fs_r0 = metrics_for(sample["region"], sample["r0_pred"])
        f1_r1, rec_r1, fs_r1 = metrics_for(sample["region"], sample["r1_pred"])
        diffs["macro_f1"].append(f1_r1 - f1_r0)
        diffs["severe_recall"].append(rec_r1 - rec_r0)
        diffs["false_safe_rate"].append(fs_r1 - fs_r0)

    return {k: np.array(v) for k, v in diffs.items()}


def report_ci(name, values, better_direction):
    lo, hi = np.percentile(values, [2.5, 97.5])
    mean_diff = values.mean()
    excludes_zero = (lo > 0) or (hi < 0)
    verdict = "SIGNIFICANT" if excludes_zero else "not significant (CI includes 0)"
    print(f"  {name}: mean diff (r1 - r0) = {mean_diff:+.4f}, 95% CI = [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    print(f"    (better direction for r1 is '{better_direction}')")


def subgroup_analysis(df):
    df = df.copy()
    df["ws_divergence"] = (df["Wind speed (m/s)"] - df["neighbor_windspeed"]).abs()

    bins = [0, 0.5, 1.0, 2.0, 4.0, np.inf]
    labels = ["0-0.5", "0.5-1", "1-2", "2-4", "4+"]
    df["divergence_bin"] = pd.cut(df["ws_divergence"], bins=bins, labels=labels)

    print("\n--- Subgroup analysis: macro-F1 by |own windspeed - neighbor windspeed| ---")
    rows = []
    for b in labels:
        sub = df[df["divergence_bin"] == b]
        if sub.empty:
            continue
        f1_r0, rec_r0, fs_r0 = metrics_for(sub["region"], sub["r0_pred"])
        f1_r1, rec_r1, fs_r1 = metrics_for(sub["region"], sub["r1_pred"])
        rows.append({
            "divergence_bin": b, "n": len(sub),
            "macro_f1_r0": round(f1_r0, 4), "macro_f1_r1": round(f1_r1, 4),
            "macro_f1_diff": round(f1_r1 - f1_r0, 4),
            "severe_recall_r0": round(rec_r0, 4), "severe_recall_r1": round(rec_r1, 4),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    result.to_csv(OUT_DIR / "subgroup_analysis.csv", index=False)
    return result


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} test predictions.")

    print("\n=== Bootstrap significance test (r1 - r0), N =", N_BOOTSTRAP, "===")
    diffs = bootstrap_diffs(df)
    report_ci("Macro F1", diffs["macro_f1"], "higher")
    report_ci("Severe-class recall", diffs["severe_recall"], "higher")
    report_ci("False-safe rate", diffs["false_safe_rate"], "lower (negative diff is better)")

    subgroup_analysis(df)


if __name__ == "__main__":
    main()