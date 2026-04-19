#!/usr/bin/env python3
"""
scripts/phase4_step2_lag_matrix.py
===================================
Phase 4 Step 2 — Lag matrix construction per ProjectScope §9.

For each (country, indicator), produce lagged features at t-k for
k in {1, 3, 6, 12} — ProjectScope §9's uniform sparse grid.  This
grid deliberately samples short-run (1m), quarterly (3m), semi-annual
(6m), and annual (12m) dynamics without flooding Phase 6 with a dense
lag specification.

Pipeline
--------
  1. Rebuild the per-country base feature frame (S1 output) from the
     processed data + effective registry (D-031 applied).
  2. For each of the 5 indicators per country, generate 4 lag columns
     via pd.Series.shift(k).  Column naming: {COUNTRY}_{INDICATOR}_lag{k}.
  3. Write per-country lag matrices and a long-form summary CSV.
  4. Validate: each lag column's first_valid_date must equal
     (source_first_valid_date + k months).  Exact match across all
     80 lag columns is the correctness proof.

Outputs
-------
data/documentation/phase4_step2_lag_{country}.csv   × 4
    Wide per-country lag matrix; 20 columns (5 indicators × 4 lag periods).
data/documentation/phase4_step2_lag_summary.csv
    Per-column diagnostics incl. first_valid_match correctness flag.

Decision references
-------------------
D-034  Lag grid = ProjectScope §9 literal, {1, 3, 6, 12} uniform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── src import ──────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    MAIN_COUNTRIES,
    INDICATORS,
    load_processed_all_main,
    apply_transform,
    find_project_root,
)

# ── D-031 runtime overrides (single source of truth, copied from S1) ─
REGISTRY_OVERRIDES: dict[tuple[str, str], str] = {
    ('JAPAN',   'CPI'): 'first_diff',
    ('GERMANY', 'CPI'): 'first_diff',
    ('UK',      'CPI'): 'log_diff_pct',
}

# ── D-034: ProjectScope §9 uniform sparse grid ──────────────────────
LAG_PERIODS: tuple[int, ...] = (1, 3, 6, 12)


# ── helpers ─────────────────────────────────────────────────────────
def build_effective_registry(project_root: Path) -> pd.DataFrame:
    """Load phase3 registry + apply D-031 overrides."""
    path = project_root / 'data' / 'documentation' \
        / 'phase3_transformation_registry_final.csv'
    reg = pd.read_csv(path)[['country', 'indicator', 'phase6_var_input']].copy()
    reg['effective_phase6_var_input'] = [
        REGISTRY_OVERRIDES.get((r['country'], r['indicator']),
                               r['phase6_var_input'])
        for _, r in reg.iterrows()
    ]
    return reg


def transform_country(df: pd.DataFrame, country: str,
                      eff_reg: pd.DataFrame) -> pd.DataFrame:
    """Apply per-indicator effective transform; return wide base frame."""
    transformed = {}
    for ind in INDICATORS:
        src_col = f'{country}_{ind}'
        row = eff_reg[(eff_reg['country'] == country)
                      & (eff_reg['indicator'] == ind)]
        form = row.iloc[0]['effective_phase6_var_input']
        transformed[src_col], _ = apply_transform(df[src_col], form, ind)
    out = pd.DataFrame(transformed)
    out.index.name = 'date'
    return out


def build_lag_matrix(base_df: pd.DataFrame,
                     lag_periods: tuple[int, ...]) -> pd.DataFrame:
    """Generate {col}_lag{k} for each (col, k).  Uses pd.Series.shift(k),
    which prepends k leading NaN to preserve index alignment."""
    lag_cols = {}
    for col in base_df.columns:
        for k in lag_periods:
            lag_cols[f'{col}_lag{k}'] = base_df[col].shift(k)
    return pd.DataFrame(lag_cols, index=base_df.index)


def _nan_is_leading_only(s: pd.Series) -> bool:
    if s.isna().sum() == 0:
        return True
    first_valid = s.first_valid_index()
    return bool(s.loc[first_valid:].notna().all())


def summarise_lag_matrix(country: str,
                         lag_df: pd.DataFrame,
                         base_df: pd.DataFrame) -> pd.DataFrame:
    """Per-lag-column diagnostics with exact-match correctness check.

    For each lag column, the first_valid_date must equal
    (source_first_valid + k months) if shift(k) is behaving correctly.
    The first_valid_match boolean is the correctness proof.
    """
    rows = []
    prefix = f'{country}_'
    for col in lag_df.columns:
        s = lag_df[col]
        # Parse "{COUNTRY}_{INDICATOR}_lag{k}" -> (indicator, k)
        # Indicator may contain an underscore (POLICY_RATE), so we strip
        # the prefix and rsplit on "_lag".
        tail = col[len(prefix):]
        indicator, lag_str = tail.rsplit('_lag', 1)
        lag_k = int(lag_str)

        source_col          = f'{country}_{indicator}'
        source_first_valid  = base_df[source_col].first_valid_index()
        expected_first_valid = (
            source_first_valid + pd.DateOffset(months=lag_k)
            if source_first_valid is not None else None
        )
        got_first_valid = s.first_valid_index()

        rows.append({
            'country':              country,
            'indicator':            indicator,
            'lag_k':                lag_k,
            'column':               col,
            'source_column':        source_col,
            'n_total':              int(len(s)),
            'n_nan':                int(s.isna().sum()),
            'n_valid':              int(s.notna().sum()),
            'source_first_valid':   source_first_valid,
            'expected_first_valid': expected_first_valid,
            'got_first_valid':      got_first_valid,
            'first_valid_match':    got_first_valid == expected_first_valid,
            'nan_is_leading_only':  _nan_is_leading_only(s),
            'last_valid_date':      s.last_valid_index(),
            'dtype':                str(s.dtype),
        })
    return pd.DataFrame(rows)


# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    pd.options.display.max_columns = 60
    pd.options.display.width        = 220

    project_root = find_project_root()
    doc_dir      = project_root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('Phase 4 Step 2 — Lag matrix construction')
    print('=' * 78)
    print(f'Project root : {project_root}')
    print(f'Lag periods  : {LAG_PERIODS}  (ProjectScope §9 uniform sparse, D-034)')
    print(f'Countries    : {MAIN_COUNTRIES}')
    print(f'Indicators   : {INDICATORS}')
    print()

    # 1. Rebuild effective registry + base features
    eff_reg   = build_effective_registry(project_root)
    processed = load_processed_all_main()

    # 2. Build per-country lag matrices
    lag_matrices: dict[str, pd.DataFrame]  = {}
    base_frames:  dict[str, pd.DataFrame]  = {}
    summary_rows: list[pd.DataFrame]        = []

    for c in MAIN_COUNTRIES:
        base_frames[c]  = transform_country(processed[c], c, eff_reg)
        lag_matrices[c] = build_lag_matrix(base_frames[c], LAG_PERIODS)
        summary_rows.append(
            summarise_lag_matrix(c, lag_matrices[c], base_frames[c])
        )

    summary = pd.concat(summary_rows, ignore_index=True)

    # 3. Per-country joint-valid diagnostics
    print('[1/3] Per-country lag matrix shape and joint-valid window:')
    print()
    for c in MAIN_COUNTRIES:
        ldf         = lag_matrices[c]
        joint_valid = ldf.dropna(how='any')
        first       = joint_valid.index.min() if len(joint_valid) else None
        last        = joint_valid.index.max() if len(joint_valid) else None
        print(f'  {c:7s} shape={ldf.shape}  '
              f'joint_valid=[{first.date() if first is not None else "n/a"} .. '
              f'{last.date()  if last  is not None else "n/a"}]  '
              f'n_joint_valid={len(joint_valid)}')
    print()

    # 4. First-valid pivot (compact view)
    print('[2/3] First-valid-date per (country, indicator, lag_k):')
    pivot = summary.pivot_table(
        index=['country', 'indicator'],
        columns='lag_k',
        values='got_first_valid',
        aggfunc='first',
    )
    pivot.columns = [f'lag{k}' for k in pivot.columns]
    print(pivot.to_string())
    print()

    # 5. Correctness assertions
    n_rows              = len(summary)
    n_match             = int(summary['first_valid_match'].sum())
    n_leading_only      = int(summary['nan_is_leading_only'].sum())
    n_float64           = int((summary['dtype'] == 'float64').sum())
    n_lagk_gte_nan_base = int((summary['n_nan'] >= summary['lag_k']).sum())

    print('[3/3] Correctness assertions (expected denominator = {}):'.format(n_rows))
    print(f'  first_valid_match        : {n_match}/{n_rows}')
    print(f'  nan_is_leading_only      : {n_leading_only}/{n_rows}')
    print(f'  dtype == float64         : {n_float64}/{n_rows}')
    print(f'  n_nan >= lag_k           : {n_lagk_gte_nan_base}/{n_rows}')
    print()

    if n_match < n_rows:
        mismatches = summary[~summary['first_valid_match']][
            ['country', 'indicator', 'lag_k', 'source_first_valid',
             'expected_first_valid', 'got_first_valid']
        ]
        print('!! first_valid_date mismatches:')
        print(mismatches.to_string(index=False))
        print()

    # 6. Emit audit CSVs
    written = []
    for c in MAIN_COUNTRIES:
        out = doc_dir / f'phase4_step2_lag_{c.lower()}.csv'
        lag_matrices[c].to_csv(out)
        written.append(out)
    out_summary = doc_dir / 'phase4_step2_lag_summary.csv'
    summary.to_csv(out_summary, index=False)
    written.append(out_summary)

    print('Audit CSVs written:')
    for p in written:
        print(f'  {p.relative_to(project_root)}')
    print()
    print('Phase 4 Step 2 complete.')


if __name__ == '__main__':
    main()