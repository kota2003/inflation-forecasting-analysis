#!/usr/bin/env python3
"""
scripts/phase4_step1_base_features.py
======================================
Phase 4 Step 1 — Registry-driven base feature frame construction.

Produces the per-country t=0 transformed feature frame that downstream
Phase 4 steps (lags, rolling stats, regime dummies) will build upon.

Pipeline
--------
  1. Load 4 main-country processed datasets via src.data_loader.
  2. Load phase3_transformation_registry_final.csv (D-027 source of truth).
  3. Apply D-031 runtime overrides (Japan CPI, Germany CPI, UK CPI).
  4. For each (country, indicator), apply src.apply_transform() with the
     effective phase6_var_input specification.
  5. Assemble per-country wide DataFrame of t=0 transformed values.
  6. Emit audit CSVs + stdout validation panel.

Outputs
-------
data/documentation/phase4_step1_effective_registry.csv
    Post-override transformation specification per (country, indicator).
data/documentation/phase4_step1_base_features_summary.csv
    Per-column shape / NaN / first-valid-index / dtype / transform used.
data/documentation/phase4_step1_base_features_preview.csv
    First 6 + last 6 rows per country (long form) for visual audit.

No plots; no model-output files.  This is the S1 scratch stage of the
standard P3 workflow (script -> review -> Portfolio notebook).

Decision references
-------------------
D-027  Transformation Registry (phase6_var_input column drives Phase 4).
D-031  Japan / Germany / UK CPI runtime overrides.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── src import ──────────────────────────────────────────────────────
# Assume invocation from the project root: `python scripts/phase4_step1_*.py`.
# Prepend the project root to sys.path so `src` resolves regardless of cwd.
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    MAIN_COUNTRIES,
    INDICATORS,
    load_processed_all_main,
    apply_transform,
    strip_suffix,
    find_project_root,
)

# ── D-031 runtime overrides ─────────────────────────────────────────
# The phase3 registry CSV preserves the Phase 3 Step 3 *initial* registry
# state for JPN / UK / GER CPI.  D-031 revised these post-hoc; the revision
# lives in `scripts/phase3_step4_chow_structural_breaks.py::REGISTRY_OVERRIDES`.
# Phase 4 must apply the revision before feeding the registry to the
# transformation dispatch, so the effective specification matches the
# Phase 3 final table in notebooks/03_*.ipynb section 4.3.
REGISTRY_OVERRIDES: dict[tuple[str, str], str] = {
    ('JAPAN',   'CPI'): 'first_diff',    # was 'yoy_pct_with_regime_dummy'
    ('GERMANY', 'CPI'): 'first_diff',    # was 'yoy_pct_with_caveat'
    ('UK',      'CPI'): 'log_diff_pct',  # was 'log_diff_pct_with_caveat'
}


# ── helpers ─────────────────────────────────────────────────────────
def build_effective_registry(project_root: Path) -> pd.DataFrame:
    """Load phase3 registry, apply D-031 overrides, return a clean table.

    Returns a DataFrame with columns:
      country, indicator,
      registry_phase6_var_input   (raw value from the CSV),
      effective_phase6_var_input  (post-override),
      override_applied            (bool),
      effective_transform_base    (registry value with _with_* suffix stripped).
    """
    path = project_root / 'data' / 'documentation' \
        / 'phase3_transformation_registry_final.csv'
    reg = pd.read_csv(path)
    reg = reg[['country', 'indicator', 'phase6_var_input']].copy()
    reg = reg.rename(columns={'phase6_var_input': 'registry_phase6_var_input'})

    effective, overridden = [], []
    for _, r in reg.iterrows():
        key = (r['country'], r['indicator'])
        if key in REGISTRY_OVERRIDES:
            effective.append(REGISTRY_OVERRIDES[key])
            overridden.append(True)
        else:
            effective.append(r['registry_phase6_var_input'])
            overridden.append(False)
    reg['effective_phase6_var_input'] = effective
    reg['override_applied']           = overridden
    reg['effective_transform_base']   = reg['effective_phase6_var_input'].map(strip_suffix)
    return reg


def transform_country(
    df: pd.DataFrame,
    country: str,
    eff_reg: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-indicator effective transform; return wide DataFrame.

    Output columns: {COUNTRY}_{INDICATOR} (5 columns); index is the union
    of each transformed series' own index (leading NaN emerges where a
    transformation truncates, e.g. yoy_pct drops 12 initial observations).
    """
    transformed: dict[str, pd.Series] = {}
    for ind in INDICATORS:
        src_col = f'{country}_{ind}'
        if src_col not in df.columns:
            raise KeyError(f"{src_col} not found in {country} processed data")
        row = eff_reg[(eff_reg['country'] == country)
                      & (eff_reg['indicator'] == ind)]
        if row.empty:
            raise KeyError(f"No registry row for ({country}, {ind})")
        form = row.iloc[0]['effective_phase6_var_input']
        transformed_series, _ = apply_transform(df[src_col], form, ind)
        transformed[src_col] = transformed_series

    out = pd.DataFrame(transformed)
    out.index.name = 'date'
    return out


def summarise_base_features(
    country: str,
    base_df: pd.DataFrame,
    eff_reg: pd.DataFrame,
) -> pd.DataFrame:
    """Per-column diagnostic row: shape, NaN, first-valid, transform used."""
    rows = []
    for col in base_df.columns:
        s = base_df[col]
        ind = col.split('_', 1)[1]
        form = eff_reg[
            (eff_reg['country'] == country) & (eff_reg['indicator'] == ind)
        ].iloc[0]['effective_phase6_var_input']
        rows.append({
            'country':             country,
            'indicator':           ind,
            'column':              col,
            'effective_transform': form,
            'n_total':             int(len(s)),
            'n_nan':               int(s.isna().sum()),
            'n_valid':             int(s.notna().sum()),
            'first_valid_date':    s.first_valid_index(),
            'last_valid_date':     s.last_valid_index(),
            'nan_is_leading_only': _nan_is_leading_only(s),
            'dtype':               str(s.dtype),
        })
    return pd.DataFrame(rows)


def _nan_is_leading_only(s: pd.Series) -> bool:
    """True iff NaN mass is contiguous at the head (no internal NaN)."""
    if s.isna().sum() == 0:
        return True
    first_valid = s.first_valid_index()
    return bool(s.loc[first_valid:].notna().all())


def country_preview(country: str, base_df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """First-n + last-n rows in long form (country, date, column, value)."""
    head = base_df.head(n).reset_index()
    tail = base_df.tail(n).reset_index()
    head['position'] = 'head'
    tail['position'] = 'tail'
    wide = pd.concat([head, tail], axis=0, ignore_index=True)
    long = wide.melt(
        id_vars=['date', 'position'], var_name='column', value_name='value'
    )
    long.insert(0, 'country', country)
    return long


# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    pd.options.display.max_columns = 50
    pd.options.display.width        = 200

    project_root = find_project_root()
    doc_dir      = project_root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('Phase 4 Step 1 — Registry-driven base feature frame')
    print('=' * 78)
    print(f'Project root : {project_root}')
    print(f'Countries    : {MAIN_COUNTRIES}')
    print(f'Indicators   : {INDICATORS}')
    print()

    # 1. Build effective registry (apply D-031 overrides)
    eff_reg = build_effective_registry(project_root)
    n_override = int(eff_reg['override_applied'].sum())
    print(f'[1/4] Effective registry built — {len(eff_reg)} rows, '
          f'{n_override} D-031 overrides applied.')
    print()
    print('D-031 overrides (runtime):')
    print(eff_reg[eff_reg['override_applied']][
        ['country', 'indicator',
         'registry_phase6_var_input', 'effective_phase6_var_input']
    ].to_string(index=False))
    print()
    print('Effective transform distribution:')
    print(eff_reg['effective_phase6_var_input']
          .value_counts()
          .rename_axis('transform').reset_index(name='count')
          .to_string(index=False))
    print()

    # 2. Load processed main datasets
    processed = load_processed_all_main()
    print('[2/4] Loaded processed main datasets:')
    for c in MAIN_COUNTRIES:
        df = processed[c]
        print(f'  {c:7s} shape={df.shape}  '
              f'range=[{df.index.min().date()} .. {df.index.max().date()}]  '
              f'nan={int(df.isna().sum().sum())}')
    print()

    # 3. Transform per country
    base_features: dict[str, pd.DataFrame] = {}
    summary_rows: list[pd.DataFrame]       = []
    preview_rows: list[pd.DataFrame]       = []

    for c in MAIN_COUNTRIES:
        base_features[c] = transform_country(processed[c], c, eff_reg)
        summary_rows.append(summarise_base_features(c, base_features[c], eff_reg))
        preview_rows.append(country_preview(c, base_features[c]))

    summary = pd.concat(summary_rows, ignore_index=True)
    preview = pd.concat(preview_rows, ignore_index=True)

    print('[3/4] Per-country base-feature diagnostics:')
    print()
    for c in MAIN_COUNTRIES:
        bdf          = base_features[c]
        joint_valid  = bdf.dropna(how='any')
        joint_first  = joint_valid.index.min() if len(joint_valid) else None
        joint_last   = joint_valid.index.max() if len(joint_valid) else None
        print(f'  {c:7s} shape={bdf.shape}  '
              f'joint_valid=[{joint_first.date() if joint_first is not None else "n/a"} .. '
              f'{joint_last.date()  if joint_last  is not None else "n/a"}]  '
              f'n_joint_valid={len(joint_valid)}')
    print()

    print('Per-column summary (20 rows):')
    print(summary.to_string(index=False))
    print()

    leading_only_ok = bool(summary['nan_is_leading_only'].all())
    dtypes_ok       = bool((summary['dtype'] == 'float64').all())
    print(f'NaN pattern (leading-only, no internal NaN) : {leading_only_ok}')
    print(f'All columns dtype == float64                : {dtypes_ok}')
    print()

    # 4. Emit audit CSVs
    out1 = doc_dir / 'phase4_step1_effective_registry.csv'
    out2 = doc_dir / 'phase4_step1_base_features_summary.csv'
    out3 = doc_dir / 'phase4_step1_base_features_preview.csv'

    eff_reg.to_csv(out1, index=False)
    summary.to_csv(out2, index=False)
    preview.to_csv(out3, index=False)

    print(f'[4/4] Audit CSVs written:')
    print(f'      {out1.relative_to(project_root)}')
    print(f'      {out2.relative_to(project_root)}')
    print(f'      {out3.relative_to(project_root)}')
    print()
    print('Phase 4 Step 1 complete.')


if __name__ == '__main__':
    main()