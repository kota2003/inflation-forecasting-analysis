#!/usr/bin/env python3
"""
scripts/phase4_step3_rolling.py
================================
Phase 4 Step 3 — Rolling statistics construction per ProjectScope §9 + D-035.

For each (country, indicator), compute rolling mean and rolling std
at windows w in {3, 12}.  The spec exceeds §9's "mean only" by adding
a sample std (ddof=1) to capture volatility regime — directly motivated
by Phase 3's structural-break findings (COVID/ENERGY shocks manifest as
both level and variance shifts).

Pipeline
--------
  1. Rebuild the per-country base feature frame (S1 state) from the
     processed data + effective registry (D-031 applied).
  2. For each of the 5 indicators per country, generate 4 rolling
     columns: 2 windows × 2 statistics.  Column naming:
         {COUNTRY}_{INDICATOR}_roll{w}_{stat}
     with stat in {'mean', 'std'}.
  3. Write per-country rolling matrices + a long-form summary CSV.
  4. Validate two ways:
       a. first_valid_date == source_first_valid + (w-1) months
          (pandas rolling's strict min_periods = window behaviour)
       b. Spot-check: rolling[first_valid_date] equals the manually
          computed reduction over the source's last w observations up
          to that date, to within 1e-10.

Alignment / min_periods
-----------------------
Right-aligned inclusive (pandas default): rolling(w) at t covers
[t-w+1, t].  Strict min_periods = w, so leading (w-1) obs are NaN.
Phase 6 downstream may shift by 1 for strict-trailing forecasting use;
this module keeps the general-purpose form.

Outputs
-------
data/documentation/phase4_step3_rolling_{country}.csv   × 4
    Wide per-country rolling matrix; 20 columns
    (5 indicators × 2 windows × 2 statistics).
data/documentation/phase4_step3_rolling_summary.csv
    Per-column diagnostics incl. first_valid_match and spot_check_match.

Decision references
-------------------
D-035  Rolling spec: {3, 12} windows × {mean, std}; right-aligned
       inclusive; strict min_periods = window; ddof = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

# ── D-031 runtime overrides (same as S1/S2) ─────────────────────────
REGISTRY_OVERRIDES: dict[tuple[str, str], str] = {
    ('JAPAN',   'CPI'): 'first_diff',
    ('GERMANY', 'CPI'): 'first_diff',
    ('UK',      'CPI'): 'log_diff_pct',
}

# ── D-035: ProjectScope §9 + volatility upgrade ─────────────────────
ROLLING_WINDOWS: tuple[int, ...] = (3, 12)
ROLLING_STATS:   tuple[str, ...] = ('mean', 'std')
SPOT_CHECK_TOL:  float           = 1e-10


# ── helpers ─────────────────────────────────────────────────────────
def build_effective_registry(project_root: Path) -> pd.DataFrame:
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


def build_rolling_matrix(base_df: pd.DataFrame,
                         windows: tuple[int, ...],
                         stats: tuple[str, ...]) -> pd.DataFrame:
    """Generate {col}_roll{w}_{stat} for each (col, w, stat).

    Uses strict pandas defaults: right-aligned inclusive window,
    min_periods = w, std ddof = 1.  Leading (w-1) observations are NaN.
    """
    out_cols: dict[str, pd.Series] = {}
    for col in base_df.columns:
        s = base_df[col]
        for w in windows:
            roll = s.rolling(window=w, min_periods=w)
            for stat in stats:
                if stat == 'mean':
                    out_cols[f'{col}_roll{w}_mean'] = roll.mean()
                elif stat == 'std':
                    out_cols[f'{col}_roll{w}_std'] = roll.std(ddof=1)
                else:
                    raise ValueError(f'Unsupported rolling stat: {stat!r}')
    return pd.DataFrame(out_cols, index=base_df.index)


def _nan_is_leading_only(s: pd.Series) -> bool:
    if s.isna().sum() == 0:
        return True
    first_valid = s.first_valid_index()
    return bool(s.loc[first_valid:].notna().all())


def spot_check_rolling(rolling_series: pd.Series,
                       source_series: pd.Series,
                       window: int,
                       stat: str,
                       tol: float = SPOT_CHECK_TOL) -> tuple[bool, float, float]:
    """Verify rolling[first_valid] equals manual reduction over source
    [first_valid_of_source .. first_valid_of_rolling] (last `window` obs).

    Returns (match, got, expected).  If either input is empty/NaN,
    returns (False, nan, nan).
    """
    fv = rolling_series.first_valid_index()
    if fv is None:
        return False, np.nan, np.nan

    # Take the last `window` source observations with date <= fv.
    src_slice = source_series.loc[:fv].dropna().tail(window)
    if len(src_slice) != window:
        return False, float(rolling_series.loc[fv]), np.nan

    if stat == 'mean':
        expected = float(src_slice.mean())
    elif stat == 'std':
        expected = float(src_slice.std(ddof=1))
    else:
        return False, float(rolling_series.loc[fv]), np.nan

    got = float(rolling_series.loc[fv])
    if np.isnan(got) or np.isnan(expected):
        return False, got, expected
    return (abs(got - expected) < tol), got, expected


def summarise_rolling_matrix(country: str,
                             rolling_df: pd.DataFrame,
                             base_df: pd.DataFrame) -> pd.DataFrame:
    """Per-rolling-column diagnostics with first_valid_match and spot_check."""
    rows = []
    prefix = f'{country}_'
    for col in rolling_df.columns:
        s = rolling_df[col]
        # Parse "{COUNTRY}_{INDICATOR}_roll{w}_{stat}"
        tail = col[len(prefix):]
        head_part, stat = tail.rsplit('_', 1)
        indicator, win_str = head_part.rsplit('_roll', 1)
        window = int(win_str)

        source_col           = f'{country}_{indicator}'
        source_series        = base_df[source_col]
        source_first_valid   = source_series.first_valid_index()
        expected_first_valid = (
            source_first_valid + pd.DateOffset(months=window - 1)
            if source_first_valid is not None else None
        )
        got_first_valid = s.first_valid_index()
        first_valid_match = got_first_valid == expected_first_valid

        spot_match, spot_got, spot_expected = spot_check_rolling(
            s, source_series, window, stat
        )

        rows.append({
            'country':              country,
            'indicator':            indicator,
            'window':               window,
            'stat':                 stat,
            'column':               col,
            'source_column':        source_col,
            'n_total':              int(len(s)),
            'n_nan':                int(s.isna().sum()),
            'n_valid':              int(s.notna().sum()),
            'source_first_valid':   source_first_valid,
            'expected_first_valid': expected_first_valid,
            'got_first_valid':      got_first_valid,
            'first_valid_match':    bool(first_valid_match),
            'spot_check_match':     bool(spot_match),
            'spot_check_got':       spot_got,
            'spot_check_expected':  spot_expected,
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
    print('Phase 4 Step 3 — Rolling statistics construction')
    print('=' * 78)
    print(f'Project root : {project_root}')
    print(f'Windows      : {ROLLING_WINDOWS}')
    print(f'Statistics   : {ROLLING_STATS}')
    print(f'Alignment    : right-inclusive, strict min_periods = window, ddof=1')
    print(f'D-035 spec   : §9 mean + volatility upgrade (std)')
    print()

    # 1. Rebuild effective registry + base features
    eff_reg   = build_effective_registry(project_root)
    processed = load_processed_all_main()

    # 2. Build per-country rolling matrices
    rolling_matrices: dict[str, pd.DataFrame] = {}
    base_frames:      dict[str, pd.DataFrame] = {}
    summary_rows: list[pd.DataFrame]          = []

    for c in MAIN_COUNTRIES:
        base_frames[c]      = transform_country(processed[c], c, eff_reg)
        rolling_matrices[c] = build_rolling_matrix(
            base_frames[c], ROLLING_WINDOWS, ROLLING_STATS
        )
        summary_rows.append(
            summarise_rolling_matrix(c, rolling_matrices[c], base_frames[c])
        )

    summary = pd.concat(summary_rows, ignore_index=True)

    # 3. Per-country shape and joint-valid diagnostics
    print('[1/3] Per-country rolling matrix shape and joint-valid window:')
    print()
    for c in MAIN_COUNTRIES:
        rdf         = rolling_matrices[c]
        joint_valid = rdf.dropna(how='any')
        first       = joint_valid.index.min() if len(joint_valid) else None
        last        = joint_valid.index.max() if len(joint_valid) else None
        print(f'  {c:7s} shape={rdf.shape}  '
              f'joint_valid=[{first.date() if first is not None else "n/a"} .. '
              f'{last.date()  if last  is not None else "n/a"}]  '
              f'n_joint_valid={len(joint_valid)}')
    print()

    # 4. First-valid pivot (compact view per country × indicator × window × stat)
    print('[2/3] First-valid-date per (country, indicator, window, stat):')
    pivot = summary.pivot_table(
        index=['country', 'indicator'],
        columns=['window', 'stat'],
        values='got_first_valid',
        aggfunc='first',
    )
    pivot.columns = [f'w{w}_{st}' for w, st in pivot.columns]
    print(pivot.to_string())
    print()

    # 5. Correctness assertions
    n_rows          = len(summary)
    n_fv_match      = int(summary['first_valid_match'].sum())
    n_spot_match    = int(summary['spot_check_match'].sum())
    n_leading_only  = int(summary['nan_is_leading_only'].sum())
    n_float64       = int((summary['dtype'] == 'float64').sum())

    print(f'[3/3] Correctness assertions (expected denominator = {n_rows}):')
    print(f'  first_valid_match        : {n_fv_match}/{n_rows}')
    print(f'  spot_check_match (1e-10) : {n_spot_match}/{n_rows}')
    print(f'  nan_is_leading_only      : {n_leading_only}/{n_rows}')
    print(f'  dtype == float64         : {n_float64}/{n_rows}')
    print()

    if n_fv_match < n_rows or n_spot_match < n_rows:
        bad = summary[
            (~summary['first_valid_match']) | (~summary['spot_check_match'])
        ][['country', 'indicator', 'window', 'stat',
           'expected_first_valid', 'got_first_valid',
           'spot_check_got', 'spot_check_expected']]
        print('!! Mismatches:')
        print(bad.to_string(index=False))
        print()

    # 6. Emit audit CSVs
    written = []
    for c in MAIN_COUNTRIES:
        out = doc_dir / f'phase4_step3_rolling_{c.lower()}.csv'
        rolling_matrices[c].to_csv(out)
        written.append(out)
    out_summary = doc_dir / 'phase4_step3_rolling_summary.csv'
    summary.to_csv(out_summary, index=False)
    written.append(out_summary)

    print('Audit CSVs written:')
    for p in written:
        print(f'  {p.relative_to(project_root)}')
    print()
    print('Phase 4 Step 3 complete.')


if __name__ == '__main__':
    main()