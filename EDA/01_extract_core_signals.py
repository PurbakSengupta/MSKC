"""
Step 1: Targeted extraction of core signals from the raw Turbine_Data CSVs.

Why this script exists (read before running):
- Each Kelmarsh Turbine_Data file is ~2.7GB, each Penmanshiel one ~3.6GB.
  11 turbines total = 30GB+ of raw CSV across ~300 columns each. Loading
  these directly with pd.read_csv() will exhaust RAM on a MacBook. We don't
  need most of those 300 columns for the CAKI proof-of-concept, so this
  script streams each file in chunks and keeps only a handful of signals,
  writing a much smaller per-turbine file we can freely load afterwards.
- The header row is unusual: it's the 10th line of the file (after 9 lines
  of '#'-prefixed metadata), and the header line itself also starts with
  '# ' before the first column name ("# Date and time"). This script finds
  that header line automatically rather than assuming a fixed line number,
  and normalizes the first column to 'timestamp'.
- This script does NOT trust the filename's stated date range
  ("2024-01-01_-_2025-01-01"). The inventory output showed a sampled
  timestamp of 2023-01-01 in a file whose name claims 2024-2025, which
  needs to be resolved from the actual data, not assumed. This script
  records the real min/max timestamp and sampling interval found.

Output: one compact CSV per turbine under ./extracted/, plus
extraction_report.csv summarizing what was found, so we can sanity-check
before doing anything else.

IMPORTANT: LIMIT_FILES is set to 1 below. Run it once on a single file
first, check the printed report and extraction_report.csv, paste the
output back to me, and only then set LIMIT_FILES = None to run the full
batch (which will take a while given the file sizes -- better to confirm
correctness on one file first than wait through 11 files and find a
column-mapping bug at the end).
"""

import csv
import time
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"          # same folder as your inventory script
OUTPUT_DIR = ROOT_DIR / "extracted"
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNKSIZE = 200_000                # rows per chunk; lower if memory is tight

# Exact column names we want to keep, as they appear in the header (the
# '#' prefix on the timestamp column is handled separately below).
TARGET_COLUMNS = [
    "Wind speed (m/s)",
    "Wind direction (\u00b0)",
    "Nacelle position (\u00b0)",
    "Power (kW)",
    "Capacity factor",
    "Data Availability",
    "Energy Export (kWh)",
]

# Set to an int (e.g. 1) to test on just the first N Turbine_Data files
# before running the full batch. Set to None to process all.
LIMIT_FILES = 1
# ----------------------------------------------------------------------


def find_header_line(path: Path, probe_lines: int = 30):
    """Scan the first `probe_lines` lines to find the real header row (the
    one containing 'Date and time'), return (line_index, column_fields)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= probe_lines:
                break
            if "Date and time" in line:
                fields = next(csv.reader([line]))  # handles quoted commas
                return i, fields
    raise ValueError(f"Could not find header line in first {probe_lines} lines of {path}")


def resolve_target_columns(header_fields):
    """Map TARGET_COLUMNS (+ timestamp) to the exact strings pandas will see
    as column names, tolerating the '# ' prefix on the first field."""
    cleaned = [c.strip().lstrip("#").strip() for c in header_fields]
    name_map = dict(zip(cleaned, header_fields))  # cleaned -> original

    timestamp_col = name_map.get("Date and time")
    if timestamp_col is None:
        raise ValueError("No 'Date and time' column found after cleaning header.")

    resolved = {"timestamp": timestamp_col}
    missing = []
    for target in TARGET_COLUMNS:
        if target in name_map:
            resolved[target] = name_map[target]
        else:
            missing.append(target)
    return resolved, missing


def extract_one_file(path: Path):
    print(f"\n{'=' * 80}\nProcessing: {path.name}")
    t0 = time.time()

    header_idx, header_fields = find_header_line(path)
    print(f"  Header found at line index {header_idx} ({len(header_fields)} columns total)")

    resolved, missing = resolve_target_columns(header_fields)
    if missing:
        print(f"  WARNING: target columns not found, skipped: {missing}")

    usecols = list(resolved.values())
    rename_map = {v: k for k, v in resolved.items()}

    chunks = []
    n_rows = 0
    reader = pd.read_csv(
        path, skiprows=header_idx, header=0,
        usecols=usecols, chunksize=CHUNKSIZE, low_memory=False,
    )
    for i, chunk in enumerate(reader):
        chunk = chunk.rename(columns=rename_map)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunks.append(chunk)
        n_rows += len(chunk)
        if i % 5 == 0:
            print(f"    ...chunk {i}, cumulative rows: {n_rows}")

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    valid_ts = df["timestamp"].dropna().sort_values()
    inferred_interval = None
    if len(valid_ts) > 1:
        deltas = valid_ts.diff().dropna()
        if not deltas.empty:
            inferred_interval = deltas.mode().iloc[0]

    n_bad_ts = int(df["timestamp"].isna().sum())
    n_dupe_ts = int(df["timestamp"].duplicated().sum())
    missingness = df.drop(columns=["timestamp"]).isna().mean().round(4).to_dict()

    report = {
        "file": path.name,
        "rows": n_rows,
        "columns_extracted": ", ".join(rename_map.values()),
        "columns_missing": ", ".join(missing) if missing else "",
        "min_timestamp": valid_ts.min() if len(valid_ts) else None,
        "max_timestamp": valid_ts.max() if len(valid_ts) else None,
        "inferred_interval": str(inferred_interval),
        "n_bad_timestamps": n_bad_ts,
        "n_duplicate_timestamps": n_dupe_ts,
        "seconds_taken": round(time.time() - t0, 1),
    }
    for col, pct in missingness.items():
        report[f"pct_missing__{col}"] = pct

    out_path = OUTPUT_DIR / f"{path.stem}_core.csv"
    df.to_csv(out_path, index=False)

    print(f"  Wrote {out_path} ({n_rows} rows) in {report['seconds_taken']}s")
    print(f"  Actual date range in data: {report['min_timestamp']} to {report['max_timestamp']}")
    print(f"  Inferred sampling interval: {report['inferred_interval']}")
    print(f"  Bad timestamps: {n_bad_ts}, duplicate timestamps: {n_dupe_ts}")
    print(f"  Missingness per column: {missingness}")

    return report


def main():
    turbine_files = sorted(DATA_DIR.rglob("Turbine_Data_*.csv"))
    if not turbine_files:
        print(f"No Turbine_Data_*.csv files found under {DATA_DIR.resolve()}")
        return

    if LIMIT_FILES is not None:
        turbine_files = turbine_files[:LIMIT_FILES]

    print(f"Found {len(turbine_files)} Turbine_Data file(s) to process (LIMIT_FILES={LIMIT_FILES}).")

    reports = []
    for path in turbine_files:
        try:
            reports.append(extract_one_file(path))
        except Exception as e:
            print(f"  !! FAILED on {path.name}: {e}")

    report_df = pd.DataFrame(reports)
    report_path = OUTPUT_DIR / "extraction_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n{'=' * 80}\nSummary written to {report_path}")
    print(report_df.to_string())


if __name__ == "__main__":
    main()