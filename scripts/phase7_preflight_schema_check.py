"""
scripts/phase7_preflight_schema_check.py

Phase 7 pre-flight diagnostic — reports CSV schemas for the three Phase 6
OOS forecast artefacts and cross-checks VAR ↔ Ridge walk-forward origins
for the D-068 matched-origins claim.

Stdout only; no CSV output. Throw-away diagnostic — not a Phase 7 sub-step.

Decision linkage
----------------
D-068    Ridge origins matched to VAR S6 for paired DM.
D-075    src/evaluation.py v0.4.3 API surface; CSV adapter design
         depends on the actual column schema verified here.
"""

from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd

# ── Project root resolution ────────────────────────────────────────
try:
    from src.data_loader import find_project_root
    PROJECT_ROOT = find_project_root()
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOC_DIR = PROJECT_ROOT / 'data' / 'documentation'

TARGETS = {
    'ARIMA (Phase 6 Step 1)'   : 'phase6_step1_arima_forecast.csv',
    'VAR   (Phase 6 Step 2 S6)': 'phase6_step2_s6_var_oos_forecasts.csv',
    'Ridge (Phase 6 Step 3 S4)': 'phase6_step3_s4_ridge_oos_forecasts.csv',
}

SEP = '=' * 76
SUB = '-' * 76

# ── Part 1: per-CSV schema report ──────────────────────────────────
print(SEP)
print('Phase 7 Pre-flight — CSV Schema Report')
print(SEP)
print(f'Project root : {PROJECT_ROOT}')
print(f'Doc dir      : {DOC_DIR}')
print(f'Python       : {sys.version.split()[0]}')
print(f'pandas       : {pd.__version__}')
print()

loaded: dict[str, pd.DataFrame] = {}
for label, fname in TARGETS.items():
    path = DOC_DIR / fname
    print(SUB)
    print(f'[{label}]')
    print(f'file  : {fname}')
    if not path.exists():
        print(f'  ✗ NOT FOUND at {path}')
        print()
        continue
    df = pd.read_csv(path)
    loaded[label] = df
    print(f'shape : {df.shape}')
    print()
    print('dtypes:')
    print(df.dtypes.to_string())
    print()
    print('head(10):')
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.max_colwidth', 32):
        print(df.head(10).to_string())
    print()

# ── Part 2: walk-forward origin integrity (VAR ↔ Ridge) ────────────
print(SEP)
print('Walk-Forward Origin Integrity Check')
print('D-068 claim: Ridge origins matched to VAR S6 per (country, horizon)')
print(SEP)

var_df   = loaded.get('VAR   (Phase 6 Step 2 S6)')
ridge_df = loaded.get('Ridge (Phase 6 Step 3 S4)')

if var_df is None or ridge_df is None:
    print('⚠  One or both CSVs not loaded; skipping origin check.')
else:
    def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    ORIGIN_CANDS  = ['origin_date', 'origin', 'origin_dt', 'fcst_origin', 'start_date']
    COUNTRY_CANDS = ['country', 'Country', 'country_code', 'cntry']
    HORIZON_CANDS = ['horizon', 'h', 'h_months', 'fcst_horizon']

    v_origin  = find_col(var_df,   ORIGIN_CANDS)
    r_origin  = find_col(ridge_df, ORIGIN_CANDS)
    v_country = find_col(var_df,   COUNTRY_CANDS)
    r_country = find_col(ridge_df, COUNTRY_CANDS)
    v_horizon = find_col(var_df,   HORIZON_CANDS)
    r_horizon = find_col(ridge_df, HORIZON_CANDS)

    print(f'VAR   columns detected: origin={v_origin!r}, country={v_country!r}, horizon={v_horizon!r}')
    print(f'Ridge columns detected: origin={r_origin!r}, country={r_country!r}, horizon={r_horizon!r}')
    print()

    if None in (v_origin, r_origin, v_country, r_country):
        print('⚠  Could not auto-discover origin/country columns.')
        print('   Please share head(10) output above; adapter design will be tailored to actual schema.')
    else:
        # Per-country origin-set comparison (aggregated across horizons + variables)
        all_countries = sorted(set(var_df[v_country].astype(str).unique())
                               | set(ridge_df[r_country].astype(str).unique()))
        print(f'{"country":<10} {"VAR_n":>7} {"Ridge_n":>8} {"inter":>7} {"V-R":>5} {"R-V":>5} {"match":>6}')
        print('-' * 50)
        for c in all_countries:
            v_set = set(var_df.loc[var_df[v_country].astype(str) == c,
                                   v_origin].astype(str).unique())
            r_set = set(ridge_df.loc[ridge_df[r_country].astype(str) == c,
                                     r_origin].astype(str).unique())
            inter = v_set & r_set
            v_only = v_set - r_set
            r_only = r_set - v_set
            match = 'yes' if v_set == r_set else 'no'
            print(f'{c:<10} {len(v_set):>7} {len(r_set):>8} '
                  f'{len(inter):>7} {len(v_only):>5} {len(r_only):>5} {match:>6}')

        # Optional: if horizon columns exist, sanity-check that all 4 horizons present per country
        if v_horizon and r_horizon:
            print()
            print('Horizon coverage per country:')
            print(f'{"country":<10} {"VAR_horizons":<30} {"Ridge_horizons":<30}')
            print('-' * 72)
            for c in all_countries:
                v_h = sorted(var_df.loc[var_df[v_country].astype(str) == c,
                                        v_horizon].unique().tolist())
                r_h = sorted(ridge_df.loc[ridge_df[r_country].astype(str) == c,
                                          r_horizon].unique().tolist())
                print(f'{c:<10} {str(v_h):<30} {str(r_h):<30}')

print()
print(SEP)
print('End of report. Please share the full stdout back in chat.')
print(SEP)