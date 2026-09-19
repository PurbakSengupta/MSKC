"""
Is 'heuristic beats learned' in the point estimates real, or noise?

Paired bootstrap (same resampled rows scored under both policies each
draw) comparing learned-gate vs heuristic-gate on macro-F1, severe recall,
and false-safe rate. Needs gating_baseline_per_row.csv from the updated
15_gating_baseline_comparison.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "Results" / "gating_baseline_per_row.csv"
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def metrics_for_policy(df, inject_col):
    inject = df[inject_col]
    pred = np.where(inject, df["r1_pred"], df["r0_pred"])
    y = df["region"]
    macro_f1 = f1_score(y, pred, average="macro")
    is_severe = y.isin(["severe_down", "severe_up"])
    pred_series = pd.Series(pred, index=df.index)
    n_severe = is_severe.sum()
    severe_recall = (is_severe & (pred_series == y)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & (pred_series == "stable")).sum() / n_severe if n_severe else np.nan
    return macro_f1, severe_recall, false_safe


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} rows.")

    f1_l, rec_l, fs_l = metrics_for_policy(df, "learned_inject")
    f1_h, rec_h, fs_h = metrics_for_policy(df, "heuristic_inject")
    print(f"\nPoint estimates:")
    print(f"  Learned:   macro_f1={f1_l:.4f}, severe_recall={rec_l:.4f}, false_safe={fs_l:.4f}")
    print(f"  Heuristic: macro_f1={f1_h:.4f}, severe_recall={rec_h:.4f}, false_safe={fs_h:.4f}")
    print(f"  Diff (learned - heuristic): macro_f1={f1_l-f1_h:+.4f}, "
          f"severe_recall={rec_l-rec_h:+.4f}, false_safe={fs_l-fs_h:+.4f}")

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df)
    diffs_f1, diffs_rec, diffs_fs = [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        f1_l_b, rec_l_b, fs_l_b = metrics_for_policy(sample, "learned_inject")
        f1_h_b, rec_h_b, fs_h_b = metrics_for_policy(sample, "heuristic_inject")
        diffs_f1.append(f1_l_b - f1_h_b)
        diffs_rec.append(rec_l_b - rec_h_b)
        diffs_fs.append(fs_l_b - fs_h_b)

    def report(name, values):
        values = np.array(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        verdict = "SIGNIFICANT" if excludes_zero else "not significant (CI includes 0)"
        print(f"  {name}: mean diff (learned - heuristic) = {values.mean():+.4f}, "
              f"95% CI = [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    print(f"\nBootstrap (N={N_BOOTSTRAP}), learned vs heuristic:")
    report("Macro F1", diffs_f1)
    report("Severe-class recall", diffs_rec)
    report("False-safe rate", diffs_fs)


if __name__ == "__main__":
    main()