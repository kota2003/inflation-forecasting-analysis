#!/usr/bin/env python3
"""
scripts/phase4_step5_assemble.py
================================
Phase 4 Step 5 — Final feature matrix assembly + consistency validation.

Responsibilities
----------------
  1. Import src.feature_engineering (v0.4.0).
  2. Call build_all_features() to produce per-country feature DataFrames.
  3. Write data/processed/features_{country}.csv × 4.
  4. Generate data/processed/features_schema.md.
  5. Consistency check: rebuild the S2/S3/S4 sub-matrices via the module
     and compare against the scratch CSVs at data/documentation/
     phase4_step[2-4]_*.csv.  Any column value differing by more than
     1e-10 (NaN-vs-NaN treated as equal) fails the check.

Outputs
-------
data/processed/features_{usa,japan,uk,germany}.csv
data/processed/features_schema.md
data/documentation/phase4_step5_category_counts.csv
data/documentation/phase4_step5_joint_valid_summary.csv
data/documentation/phase4_step5_consistency_check.csv

No plots.  Phase 4 ends here; Phase 5 (EDA) consumes features_*.csv
via ``src.feature_engineering.build_all_features()`` or directly from disk.

Decision references
-------------------
D-037  src/feature_engineering.py module API
D-038  D-031 override embedded in module as REGISTRY_OVERRIDES
D-039  Per-country wide CSV; leading NaN preserved for Phase 6 flexibility
D-040  No preliminary sparsity; feature selection deferred to Phase 6
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    MAIN_COUNTRIES,
    find_project_root,
    load_processed_all_main,
    __version__ as SRC_VERSION,
)
from src.feature_engineering import (  # noqa: E402
    LAG_PERIODS,
    ROLLING_WINDOWS,
    ROLLING_STATS,
    load_effective_registry,
    transform_country,
    build_lag_matrix,
    build_rolling_matrix,
    build_regime_matrix,
    build_all_features,
    write_features_schema_md,
)


TOL = 1e-10


def _categorise_column(col: str) -> str:
    if '_x_D_' in col:
        return 'interaction'
    if '_P_' in col:
        return 'period'
    if '_D_' in col:
        return 'split'
    if '_roll' in col and ('_mean' in col or '_std' in col):
        return 'rolling'
    if '_lag' in col and col.rsplit('_lag', 1)[-1].isdigit():
        return 'lag'
    return 'base'


def _compare_wide(module_df: pd.DataFrame,
                  scratch_df: pd.DataFrame,
                  tol: float = TOL) -> dict:
    """NaN-aware 1e-10 elementwise comparison.  NaN-vs-NaN is equal."""
    cols_match = list(module_df.columns) == list(scratch_df.columns)
    if not cols_match:
        return {
            'cols_match': False, 'rows_match': False,
            'n_value_mismatch': -1, 'max_abs_diff': np.nan,
            'note': 'column mismatch',
        }
    rows_match = list(module_df.index) == list(scratch_df.index)
    if not rows_match:
        return {
            'cols_match': True, 'rows_match': False,
            'n_value_mismatch': -1, 'max_abs_diff': np.nan,
            'note': 'row index mismatch',
        }

    m_nan = module_df.isna()
    s_nan = scratch_df.isna()
    # NaN-mask XOR: cells where exactly one side is NaN.
    nan_mismatch = int((m_nan ^ s_nan).to_numpy().sum())

    # Value diff only where both non-NaN.
    both_valid = (~m_nan) & (~s_nan)
    if both_valid.to_numpy().any():
        diff_abs = (module_df.where(both_valid)
                    - scratch_df.where(both_valid)).abs()
        max_abs = float(diff_abs.max().max())
        value_mismatch = int((diff_abs > tol).to_numpy().sum())
    else:
        max_abs = 0.0
        value_mismatch = 0

    total_mismatch = nan_mismatch + value_mismatch
    return {
        'cols_match': True,
        'rows_match': True,
        'n_value_mismatch': total_mismatch,
        'max_abs_diff': max_abs,
        'note': 'ok' if total_mismatch == 0 else 'mismatch',
    }


def main() -> None:
    pd.options.display.max_columns = 80
    pd.options.display.width        = 220

    project_root = find_project_root()
    doc_dir      = project_root / 'data' / 'documentation'
    proc_dir     = project_root / 'data' / 'processed'
    doc_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('Phase 4 Step 5 — Feature matrix assembly + consistency validation')
    print('=' * 78)
    print(f'Project root : {project_root}')
    print(f'src version  : {SRC_VERSION}')
    print()

    # 1. Build all features via module
    print('[1/5] Building features via src.feature_engineering.build_all_features()')
    processed = load_processed_all_main()
    features  = build_all_features(processed=processed, project_root=project_root)
    print(f'      Built feature frames for {len(features)} countries.')
    print()

    # 2. Per-country summary (shape + joint-valid + category counts)
    print('[2/5] Per-country feature summary:')
    print()
    count_rows = []
    jv_rows    = []
    for c in MAIN_COUNTRIES:
        feat = features[c]
        jv   = feat.dropna(how='any')
        jv_start = jv.index.min() if len(jv) else None
        jv_last  = jv.index.max() if len(jv) else None
        cats   = pd.Series({col: _categorise_column(col) for col in feat.columns})
        counts = cats.value_counts()
        print(f'  {c:7s} shape={feat.shape}  '
              f'joint_valid=[{jv_start.date() if jv_start is not None else "n/a"} .. '
              f'{jv_last.date() if jv_last  is not None else "n/a"}]  '
              f'n_joint_valid={len(jv)}')
        print(f'          base={int(counts.get("base",0))} '
              f'lag={int(counts.get("lag",0))} '
              f'rolling={int(counts.get("rolling",0))} '
              f'split={int(counts.get("split",0))} '
              f'period={int(counts.get("period",0))} '
              f'interaction={int(counts.get("interaction",0))} '
              f'total={len(feat.columns)}')
        count_rows.append({
            'country':      c,
            'base':         int(counts.get('base', 0)),
            'lag':          int(counts.get('lag', 0)),
            'rolling':      int(counts.get('rolling', 0)),
            'split':        int(counts.get('split', 0)),
            'period':       int(counts.get('period', 0)),
            'interaction':  int(counts.get('interaction', 0)),
            'total':        len(feat.columns),
        })
        jv_rows.append({
            'country':            c,
            'n_rows':             len(feat),
            'n_columns':          feat.shape[1],
            'joint_valid_start':  jv_start,
            'joint_valid_end':    jv_last,
            'n_joint_valid':      len(jv),
            'n_nan_total':        int(feat.isna().sum().sum()),
        })
    print()

    # 3. Write features CSVs + schema
    print('[3/5] Writing data/processed/features_*.csv and features_schema.md')
    for c in MAIN_COUNTRIES:
        out = proc_dir / f'features_{c.lower()}.csv'
        features[c].to_csv(out)
    schema_path = write_features_schema_md(features, project_root=project_root)
    print(f'      Wrote features_*.csv × {len(MAIN_COUNTRIES)}')
    print(f'      Wrote {schema_path.relative_to(project_root)}')
    print()

    # 4. Consistency check vs S2/S3/S4 scratch CSVs
    print('[4/5] Consistency check: module output vs S2/S3/S4 scratch CSVs '
          f'(tol={TOL:.0e})')
    print()
    eff_reg = load_effective_registry(project_root)
    consistency_rows = []
    for c in MAIN_COUNTRIES:
        base_df = transform_country(processed[c], c, eff_reg)

        # S2: lag matrix
        mod_lags = build_lag_matrix(base_df, LAG_PERIODS)
        scr_lags = pd.read_csv(
            doc_dir / f'phase4_step2_lag_{c.lower()}.csv',
            parse_dates=['date'],
        ).set_index('date')
        r = _compare_wide(mod_lags, scr_lags)
        r.update({'country': c, 'step': 'S2_lag'})
        consistency_rows.append(r)
        print(f'  {c:7s} S2_lag        '
              f'cols={r["cols_match"]} rows={r["rows_match"]} '
              f'mismatch={r["n_value_mismatch"]:>3d} '
              f'max_abs_diff={r["max_abs_diff"]:.2e}')

        # S3: rolling matrix
        mod_roll = build_rolling_matrix(base_df, ROLLING_WINDOWS, ROLLING_STATS)
        scr_roll = pd.read_csv(
            doc_dir / f'phase4_step3_rolling_{c.lower()}.csv',
            parse_dates=['date'],
        ).set_index('date')
        r = _compare_wide(mod_roll, scr_roll)
        r.update({'country': c, 'step': 'S3_rolling'})
        consistency_rows.append(r)
        print(f'  {c:7s} S3_rolling    '
              f'cols={r["cols_match"]} rows={r["rows_match"]} '
              f'mismatch={r["n_value_mismatch"]:>3d} '
              f'max_abs_diff={r["max_abs_diff"]:.2e}')

        # S4: regime matrix
        mod_reg = build_regime_matrix(c, base_df)
        scr_reg = pd.read_csv(
            doc_dir / f'phase4_step4_regime_{c.lower()}.csv',
            parse_dates=['date'],
        ).set_index('date')
        r = _compare_wide(mod_reg, scr_reg)
        r.update({'country': c, 'step': 'S4_regime'})
        consistency_rows.append(r)
        print(f'  {c:7s} S4_regime     '
              f'cols={r["cols_match"]} rows={r["rows_match"]} '
              f'mismatch={r["n_value_mismatch"]:>3d} '
              f'max_abs_diff={r["max_abs_diff"]:.2e}')
    print()

    consistency = pd.DataFrame(consistency_rows)
    passed = (consistency['cols_match']
              & consistency['rows_match']
              & (consistency['n_value_mismatch'] == 0))
    n_ok    = int(passed.sum())
    n_total = len(consistency)
    print(f'      Consistency summary: {n_ok}/{n_total} checks passed')
    print()

    if n_ok < n_total:
        bad = consistency[~passed]
        print('!! Consistency failures:')
        print(bad.to_string(index=False))
        print()

    # 5. Emit documentation CSVs
    print('[5/5] Audit CSVs:')
    counts_df = pd.DataFrame(count_rows)
    jv_df     = pd.DataFrame(jv_rows)

    out_counts = doc_dir / 'phase4_step5_category_counts.csv'
    out_jv     = doc_dir / 'phase4_step5_joint_valid_summary.csv'
    out_cons   = doc_dir / 'phase4_step5_consistency_check.csv'

    counts_df.to_csv(out_counts, index=False)
    jv_df.to_csv(out_jv, index=False)
    consistency.to_csv(out_cons, index=False)
    print(f'      {out_counts.relative_to(project_root)}')
    print(f'      {out_jv.relative_to(project_root)}')
    print(f'      {out_cons.relative_to(project_root)}')
    for c in MAIN_COUNTRIES:
        print(f'      data/processed/features_{c.lower()}.csv')
    print(f'      {schema_path.relative_to(project_root)}')
    print()
    print('Phase 4 Step 5 complete.')


if __name__ == '__main__':
    main()