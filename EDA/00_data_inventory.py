"""
Step 0: Data inventory.

Purpose: before writing any real EDA/preprocessing code, we need to see the
ACTUAL structure of the files you downloaded (Kelmarsh / Penmanshiel from
Zenodo). Cubico's SCADA CSVs are known to have a few metadata rows above the
real header, and filenames/columns have varied slightly across dataset
versions, so this script does NOT assume any schema. It just reports what's
there. Paste the full printed output back to me.

Usage:
    1. Edit DATA_DIR below to point at the folder where you unzipped the
       downloaded dataset(s).
    2. pip install pandas openpyxl   (if not already installed)
    3. python 00_data_inventory.py > inventory_output.txt

This script only reads files. It does not modify or write anything except
the redirected stdout you choose above.
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: project/
ROOT_DIR = SCRIPT_DIR.parent

DATA_DIR = ROOT_DIR / "data"

MAX_RAW_LINES = 20          # how many raw lines to print per CSV
MAX_FILES_PER_EXTENSION = None  # set an int to cap output if there are hundreds of files


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def inspect_csv(path: Path):
    print(f"    size: {human_size(path.stat().st_size)}")
    print(f"    first {MAX_RAW_LINES} raw lines:")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= MAX_RAW_LINES:
                    print("    ...")
                    break
                print(f"    {i:>3} | {line.rstrip()}")
    except Exception as e:
        print(f"    !! could not read as text: {e}")


def inspect_excel(path: Path):
    print(f"    size: {human_size(path.stat().st_size)}")
    try:
        import pandas as pd
        xls = pd.ExcelFile(path)
        print(f"    sheet names: {xls.sheet_names}")
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, nrows=10)
            print(f"    -- sheet '{sheet}' -- shape (first 10 rows read): {df.shape}")
            print(f"       columns: {list(df.columns)}")
            print(df.head(5).to_string())
    except Exception as e:
        print(f"    !! could not read as excel: {e}")


def inspect_kmz(path: Path):
    print(f"    size: {human_size(path.stat().st_size)}")
    print("    (KMZ file — turbine layout/coordinates. Will parse separately once "
          "we confirm we need it; skipping content dump here.)")


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: DATA_DIR does not exist: {DATA_DIR.resolve()}")
        print("Edit DATA_DIR at the top of this script to point at your data folder.")
        sys.exit(1)

    print(f"Scanning: {DATA_DIR.resolve()}\n")

    all_files = sorted(DATA_DIR.rglob("*"))
    files_by_ext = {}
    for p in all_files:
        if p.is_file():
            files_by_ext.setdefault(p.suffix.lower(), []).append(p)

    print("=" * 80)
    print("FILE TYPE SUMMARY")
    print("=" * 80)
    for ext, files in sorted(files_by_ext.items()):
        print(f"  {ext or '(no extension)'}: {len(files)} file(s)")
    print()

    print("=" * 80)
    print("DIRECTORY TREE (relative paths)")
    print("=" * 80)
    for p in all_files:
        if p.is_file():
            print(f"  {p.relative_to(DATA_DIR)}  [{human_size(p.stat().st_size)}]")
    print()

    print("=" * 80)
    print("FILE CONTENTS (structure only)")
    print("=" * 80)

    for ext, files in sorted(files_by_ext.items()):
        if MAX_FILES_PER_EXTENSION:
            files = files[:MAX_FILES_PER_EXTENSION]
        for p in files:
            print("-" * 80)
            print(f"FILE: {p.relative_to(DATA_DIR)}")
            if ext == ".csv":
                inspect_csv(p)
            elif ext in (".xlsx", ".xls"):
                inspect_excel(p)
            elif ext == ".kmz":
                inspect_kmz(p)
            else:
                print(f"    size: {human_size(p.stat().st_size)} (not inspected — extension '{ext}')")
            print()

    print("=" * 80)


if __name__ == "__main__":
    main()