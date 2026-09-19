"""
Step 3: Batch extract + clean all Turbine_Data files (Kelmarsh + Penmanshiel).

This combines the two validated pieces from steps 1 and 2:
  - streaming extraction of 7 core columns via chunked reads, with automatic
    header-line detection (handles the '# Date and time' quirk)
  - the duplicate-timestamp fix: keep only the row per timestamp that has a
    non-null 'Power (kW)' value (confirmed on Kelmarsh 1: n_many == 0, i.e.
    no conflicting real-value rows, just padding rows to drop)

It does NOT assume the padding pattern is universal -- it recomputes and
reports n_zero / n_one / n_many for every turbine, so we can see immediately
if Penmanshiel (a different site/export) behaves differently before trusting
the same cleaning rule there.

Output: one clean CSV per turbine under ./extracted/clean/, named
<site>_<turbine_id>_clean.csv, each with a 'turbine_id' and 'site' column
added (needed later for cross-turbine / wake-neighbor features). Plus
batch_report.csv summarizing every file.

Set LIMIT_FILES = None to run all 11. It's set to None by default now since
steps 1 and 2 already validated the approach on one file -- if you'd rather
sanity-check on 2-3 files first, set it to a small int.
"""

import csv
import re
import time
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "extracted" / "clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200_000
KEY_COL = "Power (kW)"   # row is "real" if this is non-null

TARGET_COLUMNS = [
    "Wind speed (m/s)",
    "Wind direction (\u00b0)",
    "Nacelle position (\u00b0)",
    "Power (kW)",
    "Capacity factor",
    "Data Availability",
    "Energy Export (kWh)",
    # Added after EDA round 1: direct, time-aligned curtailment/availability
    # signals, so we don't have to fuzzily interval-join the Status text
    # logs to explain the near-zero-power band seen at high wind speed.
    "Lost Production to Curtailment (Total) (kWh)",
    "Lost Production to Downtime (kWh)",
    "Time-based System Avail.",
]

# Rows above this are suspicious, not automatically discarded. EDA round 1
# flagged only 2 rows total at >45 m/s; at >30 m/s round 2 flagged 834 rows
# concentrated in a few turbines (up to 1.3% of one turbine's year) -- too
# large a volume to discard blindly, since real severe-storm readings would
# also land here and are scientifically relevant (extreme-wind events are a
# candidate decision class in the PhD plan). We flag and cross-check against
# sibling turbines instead of dropping.
WINDSPEED_FLAG_THRESHOLD = 25.0

LIMIT_FILES = None  # set to an int to test on a few files first
# ----------------------------------------------------------------------


def find_header_line(path: Path, probe_lines: int = 30):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= probe_lines:
                break
            if "Date and time" in line:
                fields = next(csv.reader([line]))
                return i, fields
    raise ValueError(f"Could not find header line in first {probe_lines} lines of {path}")


def resolve_target_columns(header_fields):
    cleaned = [c.strip().lstrip("#").strip() for c in header_fields]
    name_map = dict(zip(cleaned, header_fields))

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


def parse_site_and_turbine(filename: str):
    """'Turbine_Data_Kelmarsh_1_2024-...csv' -> ('Kelmarsh', '1')
       'Turbine_Data_Penmanshiel_11_2024-...csv' -> ('Penmanshiel', '11')"""
    m = re.match(r"Turbine_Data_([A-Za-z]+)_(\d+)_", filename)
    if not m:
        return "unknown", "unknown"
    return m.group(1), m.group(2)


def process_one_file(path: Path):
    print(f"\n{'=' * 80}\nProcessing: {path.name}")
    t0 = time.time()
    site, turbine_id = parse_site_and_turbine(path.name)

    header_idx, header_fields = find_header_line(path)
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
    for chunk in reader:
        chunk = chunk.rename(columns=rename_map)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunks.append(chunk)
        n_rows += len(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Read {n_rows} raw rows.")

    # Diagnose the padding pattern for THIS file, don't assume it.
    has_key = df[KEY_COL].notna()
    key_counts = df.assign(_h=has_key).groupby("timestamp")["_h"].sum()
    n_zero = int((key_counts == 0).sum())
    n_one = int((key_counts == 1).sum())
    n_many = int((key_counts > 1).sum())
    print(f"  Timestamps: {key_counts.shape[0]} unique | non-null-'{KEY_COL}' counts -> "
          f"0: {n_zero}, 1: {n_one}, >1: {n_many}")

    if n_many > 0:
        print(f"  !! {n_many} timestamps have MORE THAN ONE row with real data -- "
              f"the simple filter is not safe here. Flagging for manual review; "
              f"writing the file anyway but marking it.")

    # Keep rows with real data. If n_many > 0 for some timestamps, this keeps
    # ALL of their rows (does not silently pick one) so the conflict is
    # visible in the output rather than hidden.
    clean = df[df[KEY_COL].notna()].copy()

    # Flag (do not drop) suspicious wind speed readings -- see comment above
    # WINDSPEED_FLAG_THRESHOLD. Cross-turbine cross-checking happens in the
    # next script, not here.
    n_flagged_windspeed = 0
    if "Wind speed (m/s)" in clean.columns:
        clean["flag_windspeed_extreme"] = clean["Wind speed (m/s)"] > WINDSPEED_FLAG_THRESHOLD
        n_flagged_windspeed = int(clean["flag_windspeed_extreme"].sum())
        if n_flagged_windspeed > 0:
            print(f"  Flagged (not dropped) {n_flagged_windspeed} row(s) with wind speed > {WINDSPEED_FLAG_THRESHOLD} m/s")

    clean["site"] = site
    clean["turbine_id"] = turbine_id

    n_bad_ts = int(df["timestamp"].isna().sum())
    out_path = OUTPUT_DIR / f"{site}_{turbine_id}_clean.csv"
    clean.to_csv(out_path, index=False)

    elapsed = round(time.time() - t0, 1)
    print(f"  Wrote {out_path} ({len(clean)} clean rows) in {elapsed}s")

    return {
        "file": path.name,
        "site": site,
        "turbine_id": turbine_id,
        "raw_rows": n_rows,
        "clean_rows": len(clean),
        "unique_timestamps": key_counts.shape[0],
        "ts_with_zero_data_rows": n_zero,
        "ts_with_one_data_row": n_one,
        "ts_with_conflict_gt1": n_many,
        "n_flagged_windspeed_extreme": n_flagged_windspeed,
        "bad_timestamps": n_bad_ts,
        "columns_missing": ", ".join(missing) if missing else "",
        "seconds_taken": elapsed,
    }


def main():
    turbine_files = sorted(DATA_DIR.rglob("Turbine_Data_*.csv"))
    if not turbine_files:
        print(f"No Turbine_Data_*.csv files found under {DATA_DIR.resolve()}")
        return
    if LIMIT_FILES is not None:
        turbine_files = turbine_files[:LIMIT_FILES]

    print(f"Found {len(turbine_files)} file(s) to process.")

    reports = []
    for path in turbine_files:
        try:
            reports.append(process_one_file(path))
        except Exception as e:
            print(f"  !! FAILED on {path.name}: {e}")

    report_df = pd.DataFrame(reports)
    report_path = Path("./extracted") / "batch_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n{'=' * 80}\nBatch summary written to {report_path}")
    print(report_df.to_string())

    total_conflicts = report_df["ts_with_conflict_gt1"].sum() if len(report_df) else 0
    print(f"\nTotal timestamps with conflicting (>1) real-data rows across all files: {total_conflicts}")


if __name__ == "__main__":
    main()