"""
scripts/inspect_csv_columns.py
==============================
Quick column-name discovery for Phase 5 + Phase 6 Step 2 CSVs,
used to debug phase6_step2_s7_notebook_figures.py column-name
dispatch failures.

Usage
-----
    python scripts/inspect_csv_columns.py

Output
------
Per-CSV: filename and complete column list. Share console output
verbatim so the figure script can be patched with correct names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOC_DIR = PROJECT_ROOT / 'data' / 'documentation'

# CSVs referenced by the failed figures, in the order the figure script needs them.
TARGETS = [
    # F1 — residual whiteness
    'phase6_step2_s2_var_diagnostics.csv',
    'phase6_step2_s2b_var_diagnostics.csv',
    # F2 — Granger
    'phase6_step2_s3_granger_full_matrix.csv',
    'phase6_step2_s3_granger_cpi_receivers.csv',
    # F3 / F7 — IRF
    'phase6_step2_s4_irf_cpi_target.csv',
    'phase6_step2_s4_irf_peak_summary.csv',
    # F4 — Phillips Trilogy (Phase 5 sources)
    'phase5_step3_phillips_fit.csv',
    'phase5_step2_base_correlation.csv',
    # F5 — Quantity Theory (Phase 5 cross-lag)
    'phase5_step2_lag_correlation.csv',
    # F7 — Japan Sextuple (Phase 5 ACF + S1 + Step 1 ARIMA)
    'phase5_step4_acf_pacf_values.csv',
    'phase6_step1_arima_residuals.csv',
    'phase6_step2_var_lag_selection_country.csv',
    'phase6_step2_var_lag_selection_summary.csv',
]

bar = '=' * 78
print(bar)
print('CSV column name inspection for Phase 6 Step 2 figure script debugging')
print(bar)

missing: list[str] = []
for name in TARGETS:
    path = DOC_DIR / name
    if not path.exists():
        # Maybe it's a glob pattern like "*_{country}_*.csv"? Try glob with stem.
        candidates = sorted(DOC_DIR.glob(name.replace('.csv', '*.csv')))
        if candidates:
            path = candidates[0]
            print(f'\n◇ {name}  (actual: {path.name})')
        else:
            missing.append(name)
            print(f'\n✗ {name}  — NOT FOUND')
            continue
    else:
        print(f'\n◇ {name}')

    df = pd.read_csv(path, nrows=3)
    print(f'   shape:   ({len(pd.read_csv(path))} rows, {len(df.columns)} cols)')
    print(f'   cols:    {list(df.columns)}')
    if len(df) > 0:
        sample = df.iloc[0].to_dict()
        # Short rep: first 5 key:value pairs
        short_sample = {k: v for i, (k, v) in enumerate(sample.items()) if i < 5}
        print(f'   row 0:   {short_sample}')

# Bonus — also list ALL phase6_step2 CSVs to catch anything unexpected
print('\n' + bar)
print('All phase6_step2_*.csv files on disk:')
print(bar)
for p in sorted(DOC_DIR.glob('phase6_step2_*.csv')):
    try:
        df = pd.read_csv(p, nrows=0)
        print(f'  {p.name}  →  {list(df.columns)}')
    except Exception as exc:
        print(f'  {p.name}  →  ERROR: {exc}')

print('\n' + bar)
if missing:
    print(f'{len(missing)} CSV(s) not found on disk:')
    for m in missing:
        print(f'  - {m}')
else:
    print('All expected CSVs present.')
print(bar)
