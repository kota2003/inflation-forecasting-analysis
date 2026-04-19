#!/usr/bin/env python3
"""
scripts/phase4_step4_regime_dummies.py
=======================================
Phase 4 Step 4 — Regime dummy construction per D-030 + D-036.

Produces three categories of regime features per country:

  1. Split dummies (persistent) for all 3 KNOWN_BREAKS × 4 countries.
     Column: {COUNTRY}_D_{BREAK_NAME}.  12 total (3 per country).
     Phase 6 filters per D-030 Bonferroni gating.

  2. Period dummies (temporary window) per ProjectScope §9:
       GFC   : 2008-09-01 .. 2009-06-01  (Lehman to NBER end, 10 months)
       COVID : 2020-03-01 .. 2020-09-01  (D-029 / src.structural_breaks, 7 months)
     Column: {COUNTRY}_P_{PERIOD_NAME}.  8 total (2 per country).

  3. Interaction terms (D-030 gated, regressor driver only):
       D_{BREAK} × X_{DRIVER, transformed}
     for the 6 (country × break) pairs where the dominant driver is a
     regressor.  Column: {COUNTRY}_{INDICATOR}_x_D_{BREAK_NAME}.
     Constant-driver and non-significant cases generate NO interaction.

Pipeline
--------
  1. Rebuild the base feature frame (S1 state): registry + D-031.
  2. Split dummies via src.make_split_dummy at all 3 KNOWN_BREAKS.
  3. Period dummies via _make_period_dummy for GFC and COVID windows.
  4. Interactions per the D-030 matrix (PHASE6_REGIME_SPEC).
  5. Validate via three targeted invariants (see `*_check` helpers).

Outputs
-------
data/documentation/phase4_step4_regime_{country}.csv   × 4
data/documentation/phase4_step4_regime_summary.csv
data/documentation/phase4_step4_regime_specification.csv

Decision references
-------------------
D-030  Regime-dummy interactions on dominant driver per (country, break).
D-036  Split + period + interaction structure; period windows:
       GFC = 2008-09..2009-06, COVID = 2020-03..2020-09.
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
    make_split_dummy,
    KNOWN_BREAKS,
)

# ── D-031 runtime overrides (same as S1/S2/S3) ──────────────────────
REGISTRY_OVERRIDES: dict[tuple[str, str], str] = {
    ('JAPAN',   'CPI'): 'first_diff',
    ('GERMANY', 'CPI'): 'first_diff',
    ('UK',      'CPI'): 'log_diff_pct',
}

# ── D-030 dominant driver matrix (Bonferroni-significant pairs only) ─
# Value is the dominant regressor indicator name, 'const' for intercept-
# shift cases (no interaction emitted), or None for non-significant pairs.
PHASE6_REGIME_SPEC: dict[tuple[str, str], str | None] = {
    ('USA',     'GFC_2008'):    'M2',
    ('USA',     'COVID_2020'):  'POLICY_RATE',
    ('USA',     'ENERGY_2022'): 'POLICY_RATE',
    ('JAPAN',   'GFC_2008'):    None,           # not significant
    ('JAPAN',   'COVID_2020'):  'const',        # intercept shift only
    ('JAPAN',   'ENERGY_2022'): 'const',
    ('UK',      'GFC_2008'):    None,
    ('UK',      'COVID_2020'):  'const',
    ('UK',      'ENERGY_2022'): 'GDP',
    ('GERMANY', 'GFC_2008'):    None,
    ('GERMANY', 'COVID_2020'):  'GDP',
    ('GERMANY', 'ENERGY_2022'): 'GDP',
}

# ── D-036 period windows (editable single source of truth) ──────────
PERIOD_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    'GFC':   (pd.Timestamp('2008-09-01'), pd.Timestamp('2009-06-01')),
    'COVID': (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-09-01')),
}

SPOT_CHECK_TOL: float = 1e-10


# ── helpers: base reconstruction ────────────────────────────────────
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


# ── helpers: dummy constructors ─────────────────────────────────────
def _make_period_dummy(index: pd.DatetimeIndex,
                       start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.Series:
    """P_t = 1{start <= t <= end}, as float64 Series aligned on index."""
    mask = (index >= start) & (index <= end)
    return pd.Series(mask.astype('float64'), index=index)


def _ensure_float(s: pd.Series) -> pd.Series:
    return s.astype('float64')


# ── builders ────────────────────────────────────────────────────────
def build_split_dummies(country: str,
                        index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """3 split dummies per country, one per KNOWN_BREAKS entry."""
    out: dict[str, pd.Series] = {}
    for name, dt in KNOWN_BREAKS.items():
        col = f'{country}_D_{name}'
        out[col] = _ensure_float(make_split_dummy(index, dt))
    return out


def build_period_dummies(country: str,
                         index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """2 period dummies per country (GFC, COVID) per D-036 windows."""
    out: dict[str, pd.Series] = {}
    for name, (start, end) in PERIOD_WINDOWS.items():
        col = f'{country}_P_{name}'
        out[col] = _make_period_dummy(index, start, end)
    return out


def build_interactions(country: str,
                       base_df: pd.DataFrame,
                       splits: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Interactions per D-030 matrix (regressor drivers only, gated)."""
    out: dict[str, pd.Series] = {}
    for (c, break_name), driver in PHASE6_REGIME_SPEC.items():
        if c != country:
            continue
        if driver is None or driver == 'const':
            continue   # no interaction for non-sig or intercept-shift cases
        split_col  = f'{country}_D_{break_name}'
        driver_col = f'{country}_{driver}'
        if split_col not in splits:
            raise KeyError(f'Expected split dummy {split_col} not built')
        if driver_col not in base_df.columns:
            raise KeyError(f'Expected driver column {driver_col} not built')
        inter_col = f'{country}_{driver}_x_D_{break_name}'
        # Elementwise multiply aligns on DatetimeIndex; NaN in driver
        # (leading first_diff / log_diff_pct NaN) propagates.
        out[inter_col] = splits[split_col] * base_df[driver_col]
    return out


# ── validators ──────────────────────────────────────────────────────
def check_split(country: str, name: str, s: pd.Series,
                break_date: pd.Timestamp) -> dict:
    idx = s.index
    # D at break_date
    at_break = float(s.loc[break_date]) if break_date in idx else np.nan
    # D immediately before break (use index position)
    pos_break = idx.get_loc(break_date) if break_date in idx else None
    at_prev = float(s.iloc[pos_break - 1]) if pos_break not in (None, 0) else np.nan
    # D at last
    at_last = float(s.iloc[-1])
    # Expected n_ones
    expected_ones = int((idx >= break_date).sum())
    got_ones = int((s == 1.0).sum())
    match = (
        at_break == 1.0
        and at_prev == 0.0
        and at_last == 1.0
        and expected_ones == got_ones
    )
    return {
        'country': country, 'category': 'split', 'name': name,
        'column': s.name if s.name else f'{country}_D_{name}',
        'at_break': at_break, 'at_prev': at_prev, 'at_last': at_last,
        'expected_ones': expected_ones, 'got_ones': got_ones,
        'check_match': bool(match),
    }


def check_period(country: str, name: str, s: pd.Series,
                 start: pd.Timestamp, end: pd.Timestamp) -> dict:
    idx = s.index
    at_start = float(s.loc[start]) if start in idx else np.nan
    at_end   = float(s.loc[end])   if end   in idx else np.nan
    pos_start = idx.get_loc(start) if start in idx else None
    pos_end   = idx.get_loc(end)   if end   in idx else None
    at_pre  = float(s.iloc[pos_start - 1]) if pos_start not in (None, 0) else np.nan
    at_post = (float(s.iloc[pos_end + 1])
               if pos_end is not None and pos_end + 1 < len(s) else np.nan)
    expected_ones = int(((idx >= start) & (idx <= end)).sum())
    got_ones = int((s == 1.0).sum())
    match = (
        at_start == 1.0
        and at_end == 1.0
        and (np.isnan(at_pre)  or at_pre  == 0.0)
        and (np.isnan(at_post) or at_post == 0.0)
        and expected_ones == got_ones
    )
    return {
        'country': country, 'category': 'period', 'name': name,
        'column': f'{country}_P_{name}',
        'at_start': at_start, 'at_end': at_end,
        'at_pre': at_pre, 'at_post': at_post,
        'expected_ones': expected_ones, 'got_ones': got_ones,
        'check_match': bool(match),
    }


def check_interaction(country: str, break_name: str, driver: str,
                      inter: pd.Series,
                      split: pd.Series,
                      driver_series: pd.Series,
                      break_date: pd.Timestamp) -> dict:
    """3-point spot check: first_valid, break_date, last.  tol=1e-10."""
    # Construct reference = split * driver.  NaN preserved.
    ref = split * driver_series
    # Compare at three reference dates, treating NaN == NaN as match.
    def _eq(a: float, b: float) -> bool:
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(a - b) < SPOT_CHECK_TOL

    fv = inter.first_valid_index()
    dates = [d for d in (fv, break_date, inter.index[-1]) if d is not None]
    matches = []
    details = {}
    for d in dates:
        got = float(inter.loc[d]) if d in inter.index else np.nan
        exp = float(ref.loc[d])   if d in ref.index   else np.nan
        matches.append(_eq(got, exp))
        details[f'at_{d.date()}_got']      = got
        details[f'at_{d.date()}_expected'] = exp

    # Full-index elementwise match within tol (treat paired NaN as equal).
    full_match = True
    for t in inter.index:
        a = inter.loc[t]
        b = ref.loc[t]
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) or np.isnan(b):
            full_match = False; break
        if abs(a - b) >= SPOT_CHECK_TOL:
            full_match = False; break

    return {
        'country': country, 'category': 'interaction',
        'name': f'{driver}_x_D_{break_name}',
        'column': inter.name if inter.name else '',
        'driver': driver, 'break': break_name,
        'spot_check_3pt_match': bool(all(matches)),
        'full_index_match':     bool(full_match),
        'check_match':          bool(all(matches) and full_match),
        **details,
    }


# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    pd.options.display.max_columns = 60
    pd.options.display.width        = 220

    project_root = find_project_root()
    doc_dir      = project_root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('Phase 4 Step 4 — Regime dummy construction')
    print('=' * 78)
    print(f'Project root : {project_root}')
    print(f'Known breaks : {dict((k, str(v.date())) for k, v in KNOWN_BREAKS.items())}')
    print(f'Periods      : '
          + ', '.join(f'{n}=[{s.date()}..{e.date()}]'
                      for n, (s, e) in PERIOD_WINDOWS.items()))
    print(f'Interactions : D-030-gated, regressor drivers only')
    print()

    # 1. Rebuild base frames
    eff_reg   = build_effective_registry(project_root)
    processed = load_processed_all_main()
    base_frames: dict[str, pd.DataFrame] = {}
    for c in MAIN_COUNTRIES:
        base_frames[c] = transform_country(processed[c], c, eff_reg)

    # 2. Build regime matrices + validators
    regime_matrices: dict[str, pd.DataFrame] = {}
    check_rows: list[dict] = []

    for c in MAIN_COUNTRIES:
        idx = base_frames[c].index

        splits = build_split_dummies(c, idx)
        for name, s in splits.items():
            s.name = name
            check_rows.append(
                check_split(c, name.split('_D_')[1], s, KNOWN_BREAKS[name.split('_D_')[1]])
            )
            check_rows[-1]['column'] = name

        periods = build_period_dummies(c, idx)
        for name, s in periods.items():
            s.name = name
            period_name = name.split('_P_')[1]
            start, end = PERIOD_WINDOWS[period_name]
            check_rows.append(check_period(c, period_name, s, start, end))

        interactions = build_interactions(c, base_frames[c], splits)
        for name, s in interactions.items():
            s.name = name
            # Recover break and driver from the cached spec
            break_name = name.rsplit('_x_D_', 1)[1]
            driver = name[len(f'{c}_'):].split('_x_D_', 1)[0]
            split_col  = f'{c}_D_{break_name}'
            driver_col = f'{c}_{driver}'
            check_rows.append(check_interaction(
                c, break_name, driver, s,
                splits[split_col], base_frames[c][driver_col],
                KNOWN_BREAKS[break_name],
            ))
            check_rows[-1]['column'] = name

        # Assemble per-country regime DataFrame
        all_cols = {**splits, **periods, **interactions}
        regime_matrices[c] = pd.DataFrame(all_cols, index=idx)

    summary = pd.DataFrame(check_rows)

    # 3. Per-country shape + NaN + joint-valid
    print('[1/3] Per-country regime matrix shape and NaN summary:')
    print()
    for c in MAIN_COUNTRIES:
        rdf = regime_matrices[c]
        n_nan_total = int(rdf.isna().sum().sum())
        print(f'  {c:7s} shape={rdf.shape}  '
              f'cols={list(rdf.columns)}'[:140] + ('...' if len(rdf.columns) > 5 else ''))
        print(f'          n_nan_total={n_nan_total}')
    print()

    # 4. Category correctness
    print('[2/3] Correctness assertions by category:')
    for cat in ('split', 'period', 'interaction'):
        sub = summary[summary['category'] == cat]
        n_rows = len(sub)
        n_match = int(sub['check_match'].sum())
        print(f'  {cat:12s}  check_match: {n_match}/{n_rows}')
    print()

    # 5. Any failures?
    bad = summary[~summary['check_match']]
    if len(bad) > 0:
        print('!! Failures:')
        print(bad.to_string(index=False))
        print()

    # 6. Specification echo (audit-trail of the hard-coded D-030 matrix)
    spec_rows = []
    for (c, bn), driver in PHASE6_REGIME_SPEC.items():
        spec_rows.append({
            'country': c, 'break': bn,
            'dominant_driver': driver if driver else 'not_significant',
            'generates_interaction': bool(driver not in (None, 'const')),
        })
    spec_df = pd.DataFrame(spec_rows)
    print('[3/3] D-030 specification (single source of truth):')
    print(spec_df.to_string(index=False))
    print()

    # 7. Column-count table
    print('Regime column counts per country:')
    count_rows = []
    for c in MAIN_COUNTRIES:
        sdf = summary[summary['country'] == c]
        count_rows.append({
            'country':      c,
            'n_split':      int((sdf['category'] == 'split').sum()),
            'n_period':     int((sdf['category'] == 'period').sum()),
            'n_interaction':int((sdf['category'] == 'interaction').sum()),
            'n_total':      int(len(sdf)),
        })
    print(pd.DataFrame(count_rows).to_string(index=False))
    print()

    # 8. Emit audit CSVs
    written = []
    for c in MAIN_COUNTRIES:
        out = doc_dir / f'phase4_step4_regime_{c.lower()}.csv'
        regime_matrices[c].to_csv(out)
        written.append(out)
    out_summary = doc_dir / 'phase4_step4_regime_summary.csv'
    summary.to_csv(out_summary, index=False)
    written.append(out_summary)
    out_spec = doc_dir / 'phase4_step4_regime_specification.csv'
    spec_df.to_csv(out_spec, index=False)
    written.append(out_spec)

    print('Audit CSVs written:')
    for p in written:
        print(f'  {p.relative_to(project_root)}')
    print()
    print('Phase 4 Step 4 complete.')


if __name__ == '__main__':
    main()