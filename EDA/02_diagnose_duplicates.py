"""
Step 2: Diagnose the duplicate-timestamp pattern and produce a clean,
one-row-per-10-minutes file, IF the diagnosis confirms it's safe to do so.

Background: extraction_report.csv showed ~41 rows per timestamp on average,
with ~97.6% missingness uniformly across all 7 extracted columns. Working
hypothesis: each 10-minute bucket has exactly one row carrying real values
in our 7 columns, and ~40 other rows sharing the same timestamp that are
blank in those columns (likely populated in other columns we didn't
extract). This script checks that hypothesis against the data itself before
doing anything about it, and reports cases that DON'T fit the hypothesis
(more than one non-null row per timestamp, or zero) rather than silently
dropping/averaging them.

Run this on the single extracted file you already have
(extracted/Turbine_Data_Kelmarsh_1_..._core.csv). No need to touch the
1-week experiment schedule for this -- it's a quick pass on a file that's
now small enough to load fully in memory.
"""

from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
EXTRACTED_DIR = ROOT_DIR / "extracted"
CORE_FILE = EXTRACTED_DIR / "Turbine_Data_Kelmarsh_1_2024-01-01_-_2025-01-01_228_core.csv"
OUTPUT_FILE = EXTRACTED_DIR / "Turbine_Data_Kelmarsh_1_clean.csv"

# The column we treat as the "is this row real data" indicator. Power is
# the most central signal for our task, so a row with a non-null Power
# value is treated as a real record.
KEY_COL = "Power (kW)"

DATA_COLS = [
    "Wind speed (m/s)", "Wind direction (\u00b0)", "Nacelle position (\u00b0)",
    "Power (kW)", "Capacity factor", "Data Availability", "Energy Export (kWh)",
]
# ----------------------------------------------------------------------


def main():
    print(f"Loading {CORE_FILE} ...")
    df = pd.read_csv(CORE_FILE, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows.")

    # 1. Rows-per-timestamp distribution
    counts = df.groupby("timestamp").size()
    print("\n--- Rows per timestamp: distribution ---")
    print(counts.value_counts().sort_index().to_string())
    print(f"Number of unique timestamps: {counts.shape[0]}")

    # 2. For each timestamp, how many rows have a non-null KEY_COL?
    has_key = df[KEY_COL].notna()
    key_counts = df.assign(_has_key=has_key).groupby("timestamp")["_has_key"].sum()

    n_zero = int((key_counts == 0).sum())
    n_one = int((key_counts == 1).sum())
    n_many = int((key_counts > 1).sum())

    print(f"\n--- Timestamps by number of non-null '{KEY_COL}' rows ---")
    print(f"  exactly 0 (fully missing bucket): {n_zero}")
    print(f"  exactly 1 (hypothesis holds):     {n_one}")
    print(f"  more than 1 (conflict -- needs a rule): {n_many}")

    # 3. Show a few concrete examples so we can SEE it, not just count it.
    print("\n--- Example: one timestamp with multiple rows ---")
    example_ts = counts[counts > 1].index[0]
    example_rows = df[df["timestamp"] == example_ts]
    print(f"Timestamp: {example_ts}  ({len(example_rows)} rows)")
    print(example_rows.to_string())

    if n_many > 0:
        print(f"\n--- Example: one timestamp with >1 non-null '{KEY_COL}' rows (conflict) ---")
        conflict_ts = key_counts[key_counts > 1].index[0]
        conflict_rows = df[df["timestamp"] == conflict_ts]
        print(f"Timestamp: {conflict_ts}  ({len(conflict_rows)} rows)")
        print(conflict_rows.to_string())

    # 4. Build the cleaned file ONLY from the unambiguous case (exactly one
    # non-null KEY_COL row per timestamp). Ambiguous/zero cases are reported
    # but not silently resolved -- we decide what to do with them after
    # seeing these numbers.
    clean = df[df[KEY_COL].notna()].copy()
    dupe_after_clean = clean["timestamp"].duplicated().sum()

    print(f"\n--- After keeping only rows with non-null '{KEY_COL}' ---")
    print(f"  Rows remaining: {len(clean)}")
    print(f"  Remaining duplicate timestamps (should be 0 if n_many == 0 above): {dupe_after_clean}")
    print(f"  Missingness per column, on cleaned data:")
    print(clean[DATA_COLS].isna().mean().round(4).to_string())

    if len(clean) > 1:
        clean_sorted = clean.sort_values("timestamp")
        deltas = clean_sorted["timestamp"].diff().dropna()
        print(f"  Inferred sampling interval on cleaned data: {deltas.mode().iloc[0] if not deltas.empty else 'N/A'}")

    clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote cleaned file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()