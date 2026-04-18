"""
src/preprocessing.py
====================
Phase 2 data-cleaning and alignment transformations.

Decisions implemented here (see ProjectDriven.md):
    D-012 (amended) — M2 YoY harmonisation
                      USA: level -> YoY via (L[t]/L[t-12] - 1)*100
                      JPN/UK/GER/CHN: MoM% -> YoY via cumulative product
    D-018           — Quarterly GDP -> monthly via linear interpolation,
                      then compute YoY from the interpolated monthly level
    D-019           — Country-wise effective window
                      (start at 2001-01 due to YoY window loss;
                       end at last row with all 5 indicators non-NaN)
    D-021           — Germany M2 source: Euro-area M2 (MABMM301EZM657S)
                      (unit encoded in M2_UNITS below)
    D-022           — Internal single-month NaN gaps filled by linear
                      interpolation; trailing/leading NaN handled by
                      effective-window trim

Public API:
    m2_to_yoy(series, unit)
    gdp_quarterly_to_monthly_yoy(q_series)
    normalise_monthly_index(s)
    interpolate_single_gaps(s, max_gap=1)
    process_country(country, raw)
    assemble_wide(country_cols)
    trim_effective_window(df, is_supplementary=False)
    build_processed(country, raw=None)
    build_all_processed(raw=None)

Module constants:
    YOY_LOOKBACK    = 12           # months
    GAP_INTERP_MAX  = 1            # months — D-022 threshold
    ANALYSIS_START  = 2001-01-01   # YoY 12-month loss window
    M2_UNITS        = {country: 'level' | 'mom_pct'}
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data_loader import (
    MAIN_COUNTRIES,
    SUPPLEMENTARY_COUNTRIES,
    ALL_COUNTRIES,
    INDICATORS,
    load_all_raw,
)


# ──────────────────────────────────────────────────────────────────
# Module constants
# ──────────────────────────────────────────────────────────────────
YOY_LOOKBACK: int = 12
GAP_INTERP_MAX: int = 1
ANALYSIS_START: pd.Timestamp = pd.Timestamp('2001-01-01')

# M2 source units per country (D-012 amendment — empirically audited).
# Phase 1 v2 naming convention MABMM301...657S was initially assumed YoY;
# the audit in Phase 2 established it is MoM growth rate.
M2_UNITS: Dict[str, str] = {
    'USA':     'level',      # M2SL in USD billions
    'JAPAN':   'mom_pct',    # MABMM301JPM657S
    'UK':      'mom_pct',    # MABMM301GBM657S
    'GERMANY': 'mom_pct',    # MABMM301EZM657S (Euro area, D-021)
    'CHINA':   'level',      # MANMM101CNM189S (supplementary)
}


# ──────────────────────────────────────────────────────────────────
# Transformation primitives
# ──────────────────────────────────────────────────────────────────
def m2_to_yoy(series: pd.Series, unit: str) -> pd.Series:
    """
    Convert an M2 series to year-over-year % growth.

    Parameters
    ----------
    series : pd.Series
        Monthly M2 values.
    unit : {'level', 'mom_pct'}
        - 'level'   : Stock level (e.g., USD billions). YoY =
                      (L[t]/L[t-12] - 1) * 100.
        - 'mom_pct' : Month-over-month percentage growth (e.g., 0.5 for
                      0.5%). YoY is computed as the 12-month cumulative
                      product:
                          YoY[t] = (prod_{i=0..11}(1 + MoM[t-i]/100) - 1) * 100
                      Implemented via log-sum for numerical stability.
    """
    if unit == 'level':
        return (series / series.shift(YOY_LOOKBACK) - 1.0) * 100.0
    elif unit == 'mom_pct':
        factors = 1.0 + series / 100.0
        log_factors = np.log(factors)
        rolled = log_factors.rolling(
            window=YOY_LOOKBACK, min_periods=YOY_LOOKBACK
        ).sum()
        return (np.exp(rolled) - 1.0) * 100.0
    else:
        raise ValueError(f"Unknown M2 unit: {unit!r} (expected 'level' or 'mom_pct')")


def gdp_quarterly_to_monthly_yoy(q_series: pd.Series) -> pd.Series:
    """
    D-018 implementation: convert quarterly GDP level to monthly YoY.

    Steps:
      1. Drop NaN entries, normalise index to first-of-month MS.
      2. Reindex onto a dense monthly range covering the observations.
      3. Linearly interpolate between quarterly observations.
      4. Compute YoY = (level[t] / level[t-12] - 1) * 100 on the result.

    Returns
    -------
    pd.Series
        Monthly YoY % growth. The first 12 months from the earliest
        observation will be NaN by construction.
    """
    s = q_series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)

    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    monthly_idx = pd.date_range(s.index.min(), s.index.max(), freq='MS')
    monthly_level = s.reindex(monthly_idx).interpolate('linear')
    return (monthly_level / monthly_level.shift(YOY_LOOKBACK) - 1.0) * 100.0


def normalise_monthly_index(s: pd.Series) -> pd.Series:
    """
    Force index to first-of-month MS frequency, reindexed onto a dense
    monthly range between the first and last observation (introducing
    NaN at any missing months).
    """
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    idx = pd.date_range(s.index.min(), s.index.max(), freq='MS')
    return s.reindex(idx)


def interpolate_single_gaps(
    s: pd.Series,
    max_gap: int = GAP_INTERP_MAX,
) -> pd.Series:
    """
    D-022: Linearly interpolate *internal* NaN gaps of length <= max_gap.
    Trailing and leading NaN are *not* filled (handled downstream by
    effective-window trimming).
    """
    s = s.copy()
    isna = s.isna()
    if not isna.any():
        return s

    # Identify consecutive NaN runs
    groups = (isna != isna.shift()).cumsum()
    should_interp = False
    for _, block in s.groupby(groups):
        if not block.isna().all():
            continue
        if len(block) > max_gap:
            continue
        # Skip leading/trailing runs
        first_idx = s.index.get_loc(block.index[0])
        last_idx = s.index.get_loc(block.index[-1])
        if first_idx == 0 or last_idx == len(s) - 1:
            continue
        should_interp = True
        break

    if should_interp:
        s = s.interpolate(method='linear', limit=max_gap, limit_area='inside')
    return s


# ──────────────────────────────────────────────────────────────────
# Per-country pipeline
# ──────────────────────────────────────────────────────────────────
def process_country(
    country: str,
    raw: Dict[str, Dict[str, pd.Series]],
) -> Dict[str, pd.Series]:
    """
    Apply Phase 2 transformations to one country's raw series.

    Returns
    -------
    dict
        Keys: `{COUNTRY}_{INDICATOR}`. Values: transformed Series.
    """
    r = raw[country]

    cpi = interpolate_single_gaps(normalise_monthly_index(r['CPI']))
    pol = interpolate_single_gaps(normalise_monthly_index(r['POLICY_RATE']))

    une = normalise_monthly_index(r['UNEMPLOYMENT'])
    if country == 'CHINA':
        # China unemployment is annual per D-010; forward-fill yearly
        # values across intervening months (supplementary only).
        une = une.ffill()
    else:
        une = interpolate_single_gaps(une)

    gdp_yoy = normalise_monthly_index(gdp_quarterly_to_monthly_yoy(r['GDP']))
    m2_yoy = normalise_monthly_index(m2_to_yoy(r['M2'], M2_UNITS[country]))

    return {
        f'{country}_CPI': cpi,
        f'{country}_POLICY_RATE': pol,
        f'{country}_UNEMPLOYMENT': une,
        f'{country}_GDP': gdp_yoy,
        f'{country}_M2': m2_yoy,
    }


def assemble_wide(country_cols: Dict[str, pd.Series]) -> pd.DataFrame:
    """Stack named Series horizontally into a wide DataFrame."""
    df = pd.concat(country_cols.values(), axis=1)
    df.columns = list(country_cols.keys())
    df.index.name = 'date'
    return df


def trim_effective_window(
    df: pd.DataFrame,
    is_supplementary: bool = False,
) -> pd.DataFrame:
    """
    D-019 Option (b): trim to country-wise effective window.

    - Start: ANALYSIS_START (2001-01) — the YoY 12-month lookback means
      no earlier row has all indicators populated.
    - End (main): last row where all 5 variables are non-NaN.
    - End (supplementary): keep full range (sparse by design).
    """
    df = df[df.index >= ANALYSIS_START].copy()
    if is_supplementary:
        return df

    mask = df.notna().all(axis=1)
    if not mask.any():
        raise RuntimeError(
            "No row has all indicators populated after ANALYSIS_START"
        )
    last_full = mask[mask].index.max()
    return df[df.index <= last_full].copy()


# ──────────────────────────────────────────────────────────────────
# End-to-end builders
# ──────────────────────────────────────────────────────────────────
def build_processed(
    country: str,
    raw: Optional[Dict[str, Dict[str, pd.Series]]] = None,
) -> pd.DataFrame:
    """
    End-to-end: country name -> cleaned DataFrame ready for VAR.
    If `raw` is not supplied it is loaded via `load_all_raw()`.
    """
    if raw is None:
        raw = load_all_raw()
    is_supp = country in SUPPLEMENTARY_COUNTRIES
    cols = process_country(country, raw)
    df = assemble_wide(cols)
    return trim_effective_window(df, is_supplementary=is_supp)


def build_all_processed(
    raw: Optional[Dict[str, Dict[str, pd.Series]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Return {country: DataFrame} for all 5 countries."""
    if raw is None:
        raw = load_all_raw()
    return {c: build_processed(c, raw) for c in ALL_COUNTRIES}


# ──────────────────────────────────────────────────────────────────
# Schema documentation generator
# ──────────────────────────────────────────────────────────────────
def write_schema_md(
    datasets: Dict[str, pd.DataFrame],
    out_path,
) -> None:
    """
    Write an auto-generated schema.md that documents the processed/ CSVs.

    Generated content includes:
      - Per-file summary (rows, columns, effective window, NaN count)
      - Column-level schema (dtype, description) shared across main files
      - Decision-ID references (D-012 amended, D-018, D-019, D-021, D-022, D-023)
      - Reproduction pointer (scripts/rebuild_processed.py)

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Keys are country names; values are the processed DataFrames
        as produced by build_all_processed().
    out_path : str | Path
        Destination path, typically data/processed/schema.md.
    """
    from datetime import datetime
    from pathlib import Path as _Path

    lines = []
    lines.append("# `data/processed/` - Schema Specification")
    lines.append("")
    lines.append(
        "*Auto-generated by `src.preprocessing.write_schema_md()`. "
        "Do not edit by hand; regenerate via `python scripts/rebuild_processed.py` "
        "or by executing `notebooks/02_cleaning_alignment.ipynb`.*"
    )
    lines.append("")
    lines.append(f"**Last generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- File summary ----
    lines.append("## File Summary")
    lines.append("")
    lines.append("| File | Category | Rows | Cols | Effective Window | NaN Count |")
    lines.append("|---|---|---:|---:|---|---:|")
    for country, df in datasets.items():
        category = 'main' if country in MAIN_COUNTRIES else 'supplementary'
        prefix = 'main_' if category == 'main' else 'supplementary_'
        filename = f'{prefix}{country.lower()}.csv'
        start = df.index.min().strftime('%Y-%m')
        end = df.index.max().strftime('%Y-%m')
        nan_total = int(df.isna().sum().sum())
        lines.append(
            f"| `{filename}` | {category} | {df.shape[0]} | {df.shape[1]} | "
            f"{start} to {end} | {nan_total} |"
        )
    lines.append("")

    # ---- Column schema ----
    lines.append("## Column Schema")
    lines.append("")
    lines.append(
        "Each file follows the same wide-format schema (per D-023). "
        "Column names follow the `{COUNTRY}_{INDICATOR}` convention (per D-011)."
    )
    lines.append("")
    lines.append("| Column | Type | Unit | Source Decision | Description |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| `date` | DatetimeIndex (MS) | - | D-011 | "
        "First-of-month monthly index |"
    )
    lines.append(
        "| `{COUNTRY}_CPI` | float64 | index | Phase 1 v2 | "
        "Consumer Price Index (national base; normalise in EDA for cross-country overlay) |"
    )
    lines.append(
        "| `{COUNTRY}_POLICY_RATE` | float64 | % | Phase 1 v2 (D-014 for Japan) | "
        "Central-bank policy rate |"
    )
    lines.append(
        "| `{COUNTRY}_UNEMPLOYMENT` | float64 | % | Phase 1 v2 | "
        "Harmonised unemployment rate |"
    )
    lines.append(
        "| `{COUNTRY}_GDP` | float64 | YoY % | D-018 | "
        "GDP YoY growth, from linearly-interpolated monthly level |"
    )
    lines.append(
        "| `{COUNTRY}_M2` | float64 | YoY % | D-012 amended, D-021 | "
        "Broad money YoY growth (Germany is Euro-area aggregate) |"
    )
    lines.append("")

    # ---- Per-country dtype dump ----
    lines.append("## Per-Country Column Detail")
    lines.append("")
    for country, df in datasets.items():
        category = 'main' if country in MAIN_COUNTRIES else 'supplementary'
        prefix = 'main_' if category == 'main' else 'supplementary_'
        filename = f'{prefix}{country.lower()}.csv'
        lines.append(f"### `{filename}`")
        lines.append("")
        lines.append("| Column | dtype | Non-null | Min | Max | Mean |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for col in df.columns:
            s = df[col]
            non_null = int(s.notna().sum())
            try:
                cmin = f"{float(s.min()):.2f}"
                cmax = f"{float(s.max()):.2f}"
                cmean = f"{float(s.mean()):.2f}"
            except (ValueError, TypeError):
                cmin = cmax = cmean = "-"
            lines.append(
                f"| `{col}` | {s.dtype} | {non_null} | {cmin} | {cmax} | {cmean} |"
            )
        lines.append("")

    # ---- Provenance and caveats ----
    lines.append("## Provenance and Caveats")
    lines.append("")
    lines.append(
        "- **Main vs supplementary**: The main/supplementary split is structural, "
        "not a column flag. It reflects D-001 (China is excluded from the main VAR "
        "due to documented reliability concerns) and D-023 (output format)."
    )
    lines.append(
        "- **Germany M2**: `GERMANY_M2` represents Euro-area broad money "
        "(`MABMM301EZM657S`), not a German national aggregate, per D-021. "
        "This is institutionally correct but differs in scope from Germany's other "
        "four indicators."
    )
    lines.append(
        "- **GDP monthly variation**: Within-quarter GDP variation is a linear "
        "interpolation artefact (D-018). It is not empirical monthly GDP. Downstream "
        "models should treat GDP coefficients accordingly."
    )
    lines.append(
        "- **M2 YoY computation**: For Japan, UK, and Germany, the YoY is computed "
        "from monthly MoM growth via cumulative product (D-012 amended). The source "
        "FRED series `MABMM301...657S` is MoM %, not YoY as their documentation "
        "suggests - see `notebooks/02_cleaning_alignment.ipynb` section 5 for the "
        "empirical audit."
    )
    lines.append(
        "- **China sparsity**: `supplementary_china.csv` contains NaN where the "
        "supplementary series (notably M2 ending 2018-12, unemployment only annual) "
        "are stale or missing. VAR consumers must filter; EDA consumers may use "
        "as-is for cross-country context."
    )
    lines.append("")

    # ---- Reproduction ----
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# Canonical CLI path (fast, non-interactive):")
    lines.append("python scripts/rebuild_processed.py")
    lines.append("")
    lines.append("# Narrated notebook path (same output, plus commentary and figures):")
    lines.append("jupyter lab notebooks/02_cleaning_alignment.ipynb")
    lines.append("```")
    lines.append("")
    lines.append(
        "Both paths import from `src.preprocessing` - the single source of truth "
        "for all transformation logic. See `ProjectDriven.md` D-018 through D-023 "
        "for decision rationale."
    )
    lines.append("")

    _Path(out_path).write_text("\n".join(lines), encoding='utf-8')
