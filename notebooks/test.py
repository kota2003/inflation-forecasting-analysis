"""
Phase 2 — Unified Cleaning & Alignment
=======================================
Single-pass implementation of Phase 2 decisions:

  D-012 (amended) — M2 YoY harmonisation
    USA:                level  -> YoY = (level[t]/level[t-12] - 1) * 100
    JAPAN/UK/GERMANY:   MoM %  -> YoY = (prod_{i=0..11}(1 + MoM/100) - 1) * 100
    CHINA:              level  -> YoY (supplementary only)

  D-018 — GDP quarterly -> monthly
    1. Reindex to monthly, linear interpolate level
    2. Compute YoY from interpolated monthly level

  D-019 — Alignment strategy: Option (b) country-wise full
    Each country trimmed to its own effective window; all start at 2001-01
    (due to YoY 12-month window loss)

  D-022 — Single-month NaN interpolation
    Internal NaN gaps of 1 month: linear interpolation
    Trailing NaN: trim via effective-window logic

  D-023 — Output format: wide CSV per country
    data/processed/main_{country}.csv, supplementary_china.csv

Outputs:
  data/processed/main_usa.csv, main_japan.csv, main_uk.csv, main_germany.csv
  data/processed/supplementary_china.csv
  data/processed/schema.md
  data/documentation/phase2_cleaning_log.csv
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════
# 1. Paths and configuration
# ═════════════════════════════════════════════════════════════════════
def find_project_root() -> Path:
    cur = Path.cwd().resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / 'data').is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate project root from cwd={Path.cwd()}")


PROJECT_ROOT = find_project_root()
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
DOC_DIR = PROJECT_ROOT / 'data' / 'documentation'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root : {PROJECT_ROOT}")

MAIN_COUNTRIES = ['USA', 'JAPAN', 'UK', 'GERMANY']
SUPPLEMENTARY = ['CHINA']
ALL_COUNTRIES = MAIN_COUNTRIES + SUPPLEMENTARY
INDICATORS = ['CPI', 'POLICY_RATE', 'UNEMPLOYMENT', 'GDP', 'M2']

# M2 source units (per D-012 amendment empirical audit)
M2_UNITS = {
    'USA':     'level',     # M2SL in $bn
    'JAPAN':   'mom_pct',   # MABMM301JPM657S
    'UK':      'mom_pct',   # MABMM301GBM657S
    'GERMANY': 'mom_pct',   # MABMM301EZM657S (Euro area, per D-021)
    'CHINA':   'level',     # MANMM101CNM189S (supplementary)
}

ANALYSIS_START = pd.Timestamp('2001-01-01')   # 12-month YoY window loss
YOY_LOOKBACK = 12
GAP_INTERP_MAX = 1   # D-022: interpolate single-month gaps only


# ═════════════════════════════════════════════════════════════════════
# 2. Loaders
# ═════════════════════════════════════════════════════════════════════
def load_raw_series(country: str, indicator: str) -> pd.Series:
    """Load a single raw CSV and return as pd.Series with monthly DatetimeIndex."""
    path = RAW_DIR / f'{country}_{indicator}.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    col = f'{country}_{indicator}'
    if col not in df.columns:
        raise ValueError(f"Column {col} missing in {path.name}; cols={list(df.columns)}")
    s = df[col].copy()
    # Normalise to first-of-month MS for monthly / first-of-quarter for quarterly
    s.index = pd.to_datetime(s.index)
    return s


def load_all_raw() -> dict:
    """Load all 25 raw series into nested dict: raw[country][indicator] = Series."""
    print("\n" + "=" * 70)
    print("STEP 1: Load raw data (25 series)")
    print("=" * 70)
    raw = {}
    for country in ALL_COUNTRIES:
        raw[country] = {}
        for ind in INDICATORS:
            s = load_raw_series(country, ind)
            raw[country][ind] = s
            start = s.dropna().index.min()
            end = s.dropna().index.max()
            freq = 'M' if len(s) > 100 else 'Q/A'
            print(f"  {country:<8} {ind:<13} "
                  f"n={len(s.dropna()):<4} "
                  f"{start:%Y-%m} to {end:%Y-%m}  ({freq})")
    return raw


# ═════════════════════════════════════════════════════════════════════
# 3. Transformations
# ═════════════════════════════════════════════════════════════════════
def m2_to_yoy(series: pd.Series, unit: str) -> pd.Series:
    """
    Convert M2 to YoY % growth.
      level   -> (level[t]/level[t-12] - 1) * 100
      mom_pct -> (prod_{i=0..11}(1 + MoM[t-i]/100) - 1) * 100  via log-sum
    """
    if unit == 'level':
        yoy = (series / series.shift(YOY_LOOKBACK) - 1.0) * 100.0
    elif unit == 'mom_pct':
        factors = 1.0 + series / 100.0
        log_factors = np.log(factors)
        rolled = log_factors.rolling(window=YOY_LOOKBACK, min_periods=YOY_LOOKBACK).sum()
        yoy = (np.exp(rolled) - 1.0) * 100.0
    else:
        raise ValueError(f"Unknown M2 unit: {unit}")
    return yoy


def gdp_quarterly_to_monthly_yoy(q_series: pd.Series) -> pd.Series:
    """
    D-018: Quarterly GDP level -> monthly YoY %.

    Method:
      1. Drop NaN (use only actual observations)
      2. Reindex to monthly range spanning the observations
      3. Linear interpolation between quarterly points
      4. Compute YoY growth from the monthly-interpolated level
    """
    s = q_series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)

    # Normalise quarterly dates to first-of-month
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()

    # Build monthly index spanning the quarterly observations
    monthly_idx = pd.date_range(s.index.min(), s.index.max(), freq='MS')
    monthly_level = s.reindex(monthly_idx).interpolate('linear')

    # YoY from interpolated monthly level
    yoy = (monthly_level / monthly_level.shift(YOY_LOOKBACK) - 1.0) * 100.0
    return yoy


def normalise_monthly_index(s: pd.Series) -> pd.Series:
    """Force the index to first-of-month MS frequency."""
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    # Reindex to a continuous monthly range between first and last observation
    idx = pd.date_range(s.index.min(), s.index.max(), freq='MS')
    return s.reindex(idx)


def interpolate_single_gaps(s: pd.Series, max_gap: int = GAP_INTERP_MAX) -> pd.Series:
    """
    D-022: Linearly interpolate internal NaN runs of length <= max_gap.
    Trailing / leading NaN are NOT filled (handled by effective-window trim).
    """
    s = s.copy()
    isna = s.isna()
    if not isna.any():
        return s

    # Identify runs of NaN
    groups = (isna != isna.shift()).cumsum()
    for _, block in s.groupby(groups):
        if block.isna().all() and len(block) <= max_gap:
            # Make sure this run is internal (not at the start or end)
            first_idx = s.index.get_loc(block.index[0])
            last_idx = s.index.get_loc(block.index[-1])
            if first_idx == 0 or last_idx == len(s) - 1:
                continue   # trailing / leading — skip
            # Linear interp via the full series
            s = s.interpolate(method='linear', limit=max_gap, limit_area='inside')
            break
    return s


# ═════════════════════════════════════════════════════════════════════
# 4. Pipeline orchestration
# ═════════════════════════════════════════════════════════════════════
def process_country(country: str, raw: dict) -> dict:
    """Apply all transformations for one country, return dict of named Series."""
    r = raw[country]

    # CPI, Policy Rate, Unemployment — already monthly level, just normalise and gap-fill
    cpi = interpolate_single_gaps(normalise_monthly_index(r['CPI']))
    pol = interpolate_single_gaps(normalise_monthly_index(r['POLICY_RATE']))
    une = normalise_monthly_index(r['UNEMPLOYMENT'])
    # China unemployment is annual — forward-fill each year's value to subsequent months
    if country == 'CHINA':
        une = une.ffill()
    else:
        une = interpolate_single_gaps(une)

    # GDP quarterly -> monthly YoY
    gdp_yoy = gdp_quarterly_to_monthly_yoy(r['GDP'])
    gdp_yoy = normalise_monthly_index(gdp_yoy)

    # M2 -> YoY
    m2_yoy = m2_to_yoy(r['M2'], M2_UNITS[country])
    m2_yoy = normalise_monthly_index(m2_yoy)

    return {
        f'{country}_CPI': cpi,
        f'{country}_POLICY_RATE': pol,
        f'{country}_UNEMPLOYMENT': une,
        f'{country}_GDP': gdp_yoy,
        f'{country}_M2': m2_yoy,
    }


def assemble_wide(country_cols: dict) -> pd.DataFrame:
    """Concat named Series on monthly index -> wide DataFrame."""
    df = pd.concat(country_cols.values(), axis=1)
    df.columns = list(country_cols.keys())
    df.index.name = 'date'
    return df


def trim_effective_window(df: pd.DataFrame, country: str, is_supplementary: bool) -> pd.DataFrame:
    """
    D-019 (option b): trim to country-wise effective window.
    Start: ANALYSIS_START (2001-01).
    End for main: last row where all 5 variables are non-NaN.
    For supplementary China: keep full range, do not trim end (sparse by design).
    """
    df = df[df.index >= ANALYSIS_START].copy()

    if is_supplementary:
        return df

    # Trim end to last row where all columns non-NaN
    mask_all_present = df.notna().all(axis=1)
    if not mask_all_present.any():
        raise RuntimeError(f"{country}: no row has all 5 indicators present after 2001-01")
    last_full = mask_all_present[mask_all_present].index.max()
    df = df[df.index <= last_full].copy()
    return df


# ═════════════════════════════════════════════════════════════════════
# 5. Validation
# ═════════════════════════════════════════════════════════════════════
def validate_main(df: pd.DataFrame, country: str) -> dict:
    """Sanity checks for main-country processed DataFrame."""
    issues = []
    # Schema
    expected = [f'{country}_{ind}' for ind in INDICATORS]
    if list(df.columns) != expected:
        issues.append(f"columns mismatch: got {list(df.columns)}, expected {expected}")
    # NaN
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        issues.append(f"NaN present:\n{nan_counts.to_string()}")
    # Monotone index
    if not df.index.is_monotonic_increasing:
        issues.append("index not monotonic")
    # Monthly freq
    freq = pd.infer_freq(df.index)
    if freq not in ('MS', 'M'):
        issues.append(f"index freq not monthly: inferred={freq}")
    return {
        'country': country,
        'n_rows': len(df),
        'start': df.index.min().strftime('%Y-%m'),
        'end': df.index.max().strftime('%Y-%m'),
        'issues': '; '.join(issues) if issues else 'none',
    }


def summarise_stats(df: pd.DataFrame, country: str) -> pd.DataFrame:
    desc = df.describe().T[['count', 'mean', 'std', 'min', 'max']].round(2)
    desc.index.name = 'variable'
    return desc


# ═════════════════════════════════════════════════════════════════════
# 6. Main execution
# ═════════════════════════════════════════════════════════════════════
raw = load_all_raw()

print("\n" + "=" * 70)
print("STEP 2: Transform each country (M2 YoY, GDP YoY, NaN interp)")
print("=" * 70)

processed_all = {}
for country in ALL_COUNTRIES:
    print(f"\n[{country}]")
    cols = process_country(country, raw)
    df = assemble_wide(cols)
    df = trim_effective_window(df, country, is_supplementary=(country in SUPPLEMENTARY))
    processed_all[country] = df
    print(f"  Final shape : {df.shape}")
    print(f"  Period      : {df.index.min():%Y-%m} to {df.index.max():%Y-%m}")
    print(f"  NaN counts  : {df.isna().sum().to_dict()}")

# ═════════════════════════════════════════════════════════════════════
# 7. Validation
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Validate main-country outputs")
print("=" * 70)

validation_rows = []
for country in MAIN_COUNTRIES:
    v = validate_main(processed_all[country], country)
    validation_rows.append(v)
    status = 'OK' if v['issues'] == 'none' else 'ISSUES'
    print(f"  {country:<8} {status}  {v['start']} to {v['end']} ({v['n_rows']} rows)")
    if v['issues'] != 'none':
        print(f"           {v['issues']}")

print("\n" + "=" * 70)
print("STEP 4: Descriptive statistics per country")
print("=" * 70)
for country in MAIN_COUNTRIES:
    print(f"\n[{country}]")
    print(summarise_stats(processed_all[country], country).to_string())

# China supplementary (sparse — report separately)
print("\n[CHINA] (supplementary, sparse by design)")
china_df = processed_all['CHINA']
print(f"  Shape       : {china_df.shape}")
print(f"  Period      : {china_df.index.min():%Y-%m} to {china_df.index.max():%Y-%m}")
print(f"  NaN counts  :")
print(china_df.isna().sum().to_string())

# ═════════════════════════════════════════════════════════════════════
# 8. Write outputs (D-023)
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Write processed CSVs")
print("=" * 70)

output_rows = []
for country in MAIN_COUNTRIES:
    df = processed_all[country]
    path = PROCESSED_DIR / f'main_{country.lower()}.csv'
    df.to_csv(path, float_format='%.6f')
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    output_rows.append({
        'file': f'main_{country.lower()}.csv',
        'country': country, 'category': 'main',
        'n_rows': df.shape[0], 'n_cols': df.shape[1],
        'start': df.index.min().strftime('%Y-%m'),
        'end': df.index.max().strftime('%Y-%m'),
    })

# China supplementary
path = PROCESSED_DIR / 'supplementary_china.csv'
china_df.to_csv(path, float_format='%.6f')
print(f"  wrote {path.relative_to(PROJECT_ROOT)}  ({china_df.shape[0]} rows x {china_df.shape[1]} cols)")
output_rows.append({
    'file': 'supplementary_china.csv',
    'country': 'CHINA', 'category': 'supplementary',
    'n_rows': china_df.shape[0], 'n_cols': china_df.shape[1],
    'start': china_df.index.min().strftime('%Y-%m'),
    'end': china_df.index.max().strftime('%Y-%m'),
})

# ═════════════════════════════════════════════════════════════════════
# 9. Write schema.md
# ═════════════════════════════════════════════════════════════════════
schema_md = f"""# data/processed/ — Schema

Auto-generated by `phase2_unified_cleaning.py` on {datetime.now():%Y-%m-%d %H:%M:%S}.

## Files

| File | Country | Category | Period | Rows |
|---|---|---|---|---|
""" + '\n'.join([
    f"| `{r['file']}` | {r['country']} | {r['category']} | {r['start']} to {r['end']} | {r['n_rows']} |"
    for r in output_rows
]) + f"""

## Column Schema (main_*.csv)

All main-country files share the same schema: 5 indicator columns plus a
date index, each column name prefixed with `{{COUNTRY}}_`.

| Column | Unit | Source | Phase 2 Transformation |
|---|---|---|---|
| `date` | YYYY-MM-01 (MS frequency) | — | index |
| `{{CC}}_CPI` | Index (base year varies per country) | FRED / stat.go.jp (JPN) | single-month gap interp |
| `{{CC}}_POLICY_RATE` | % | FRED | single-month gap interp |
| `{{CC}}_UNEMPLOYMENT` | % | FRED | single-month gap interp |
| `{{CC}}_GDP` | **YoY % growth** | FRED (quarterly level) | linear interp to monthly, then YoY |
| `{{CC}}_M2` | **YoY % growth** | FRED (level for USA, MoM % for JPN/UK/GERMANY/CHINA-source) | YoY harmonisation per D-012 amendment |

## Supplementary China

`supplementary_china.csv` follows the same column schema but is **excluded from
the main VAR** per D-001. Unemployment is annual (forward-filled to monthly).
Several variables have CRITICAL staleness per the Phase 1 v2 diagnostic.

## Decision Lineage

- **D-012 (amended)**: M2 YoY harmonisation — empirical audit identified
  `MABMM301...657S` as MoM (not YoY as originally assumed), necessitating
  cumulative-product conversion.
- **D-018**: GDP quarterly -> monthly via linear interpolation on level,
  then compute YoY. Chow-Lin considered but rejected as disproportionate
  for macro VAR.
- **D-019**: Country-wise effective window (Option b). Each country trimmed
  to `[2001-01, last fully-observed month]`.
- **D-021**: Germany M2 -> Euro area M2 (`MABMM301EZM657S`) given that
  Germany-specific M2 terminated 1998-12 with Eurozone entry.
- **D-022**: Single-month NaN gaps interpolated linearly; longer gaps or
  trailing NaN handled via effective-window trim.
- **D-023**: Wide format per country, main/supplementary structurally split.
"""

schema_path = PROCESSED_DIR / 'schema.md'
schema_path.write_text(schema_md, encoding='utf-8')
print(f"  wrote {schema_path.relative_to(PROJECT_ROOT)}")

# ═════════════════════════════════════════════════════════════════════
# 10. Audit log
# ═════════════════════════════════════════════════════════════════════
audit_path = DOC_DIR / 'phase2_cleaning_log.csv'
audit_df = pd.DataFrame(output_rows)
audit_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
audit_df.to_csv(audit_path, index=False)
print(f"  wrote {audit_path.relative_to(PROJECT_ROOT)}")

# ═════════════════════════════════════════════════════════════════════
# 11. Final summary
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2 UNIFIED CLEANING — COMPLETE")
print("=" * 70)
for r in output_rows:
    print(f"  {r['file']:<30} {r['start']} to {r['end']:<8} ({r['n_rows']} rows)")
print("\nNext: review validation output, then create notebooks/02_cleaning_alignment.ipynb")
print("Done.")
