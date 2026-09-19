"""
Step 10: Is the AIES gate's near-parity-with-r1-at-lower-cost claim
statistically real, or just a point-estimate coincidence?

Compares macro-F1 at a chosen "knee" tau (default 0.6, ~20% injection rate
per the frontier printout) against pure r1 (tau=0, 100% injection rate),
via paired bootstrap on the same test rows (same resample used for both, so
this is a paired comparison of two policies on the same data -- appropriate
since they're evaluated on identical rows, not independent samples).

"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
IN_PATH = ROOT_DIR / "Results" / "aies_frontier_test.csv"

N_BOOTSTRAP = 1000
RANDOM_STATE = 42
KNEE_TAU = 0.4   # revised after reviewing severe_recall/false_safe_rate at 0.6


def macro_f1_for_policy(df, tau):
    inject = df["kus_score"] > tau
    pred = np.where(inject, df["r1_pred"], df["r0_pred"])
    return f1_score(df["region"], pred, average="macro"), inject.mean()


def severe_recall_and_false_safe(df, tau):
    inject = df["kus_score"] > tau
    pred = pd.Series(np.where(inject, df["r1_pred"], df["r0_pred"]), index=df.index)
    y = df["region"]
    is_severe = y.isin(["severe_down", "severe_up"])
    n_severe = is_severe.sum()
    recall = (is_severe & (pred == y)).sum() / n_severe if n_severe else np.nan
    false_safe = (is_severe & (pred == "stable")).sum() / n_severe if n_severe else np.nan
    return recall, false_safe


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} rows.")

    knee_f1, knee_rate = macro_f1_for_policy(df, KNEE_TAU)
    full_f1, full_rate = macro_f1_for_policy(df, 0.0)
    print(f"\nPoint estimates:")
    print(f"  Knee (tau={KNEE_TAU}): macro-F1={knee_f1:.4f}, injection_rate={knee_rate:.3f}")
    print(f"  Pure r1 (tau=0.0):    macro-F1={full_f1:.4f}, injection_rate={full_rate:.3f}")
    print(f"  Difference (knee - pure_r1): {knee_f1 - full_f1:+.4f}, at "
          f"{100*(1 - knee_rate/full_rate):.0f}% fewer injections")

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df)
    diffs_f1, diffs_recall, diffs_fs = [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        f1_knee, _ = macro_f1_for_policy(sample, KNEE_TAU)
        f1_full, _ = macro_f1_for_policy(sample, 0.0)
        diffs_f1.append(f1_knee - f1_full)

        rec_knee, fs_knee = severe_recall_and_false_safe(sample, KNEE_TAU)
        rec_full, fs_full = severe_recall_and_false_safe(sample, 0.0)
        diffs_recall.append(rec_knee - rec_full)
        diffs_fs.append(fs_knee - fs_full)

    def report(name, values):
        values = np.array(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        verdict = "SIGNIFICANT" if excludes_zero else "not significant (CI includes 0)"
        print(f"  {name}: mean diff (knee - pure_r1) = {values.mean():+.4f}, "
              f"95% CI = [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    print(f"\nBootstrap (N={N_BOOTSTRAP}) comparing knee (tau={KNEE_TAU}) vs pure r1 (tau=0.0):")
    report("Macro F1", diffs_f1)
    report("Severe-class recall", diffs_recall)
    report("False-safe rate", diffs_fs)



if __name__ == "__main__":
    main()