"""
scripts/rebuild_processed.py
=============================
Rebuild data/processed/ from data/raw/ using the shared src/ modules.

This script is the canonical entry point for regenerating the Phase 2
cleaned dataset. It produces identical output to the exploratory
`phase2_unified_cleaning.py` script but routes through the reusable
`src.preprocessing` functions so that the logic is not duplicated.

Usage:
    From the project root:
        python scripts/rebuild_processed.py

    The script also works when invoked from anywhere inside the project
    tree because `src.data_loader.find_project_root()` walks ancestors.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make `src` importable when this script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd   # noqa: E402  (import after sys.path manipulation)

from src.data_loader import (         # noqa: E402
    find_project_root,
    MAIN_COUNTRIES,
    SUPPLEMENTARY_COUNTRIES,
)
from src.preprocessing import build_all_processed, write_schema_md   # noqa: E402


def main() -> int:
    root = find_project_root()
    processed_dir = root / 'data' / 'processed'
    doc_dir = root / 'data' / 'documentation'
    processed_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {root}")
    print("Building processed datasets from data/raw/ ...\n")

    datasets = build_all_processed()

    # Write CSVs
    rows = []
    for country, df in datasets.items():
        prefix = 'main_' if country in MAIN_COUNTRIES else 'supplementary_'
        path = processed_dir / f'{prefix}{country.lower()}.csv'
        df.to_csv(path, float_format='%.6f')
        nan_total = int(df.isna().sum().sum())
        print(
            f"  wrote {path.relative_to(root).as_posix():<36} "
            f"{df.shape[0]:>3} rows x {df.shape[1]} cols   "
            f"{df.index.min():%Y-%m} -> {df.index.max():%Y-%m}   "
            f"NaN={nan_total}"
        )
        rows.append({
            'file': path.name,
            'country': country,
            'category': 'main' if country in MAIN_COUNTRIES else 'supplementary',
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
            'start': df.index.min().strftime('%Y-%m'),
            'end': df.index.max().strftime('%Y-%m'),
            'nan_total': nan_total,
        })

    # Audit log (append, not overwrite, to preserve run history)
    log_df = pd.DataFrame(rows)
    log_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = doc_dir / 'phase2_cleaning_log.csv'
    if log_path.exists():
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\n  audit log   : {log_path.relative_to(root)}")

    # Schema specification (auto-generated documentation of processed/)
    schema_path = processed_dir / 'schema.md'
    write_schema_md(datasets, schema_path)
    print(f"  schema      : {schema_path.relative_to(root)}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
