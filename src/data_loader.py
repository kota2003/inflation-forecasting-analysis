"""
src/data_loader.py
==================
I/O helpers for raw and processed artefacts produced by the project
pipeline. These functions are intended to be imported by notebooks
and downstream phase scripts so that loading logic is not duplicated.

Public API:
    find_project_root()
    load_raw_series(country, indicator)
    load_all_raw()
    load_processed_main(country)
    load_processed_all_main()
    load_processed_china()

Design notes:
    - All paths are resolved relative to the project root, which is
      discovered by walking ancestors from the caller's cwd looking
      for a `data/` directory.
    - DateTimeIndex is normalised to MS (month-start) for monthly series.
    - The function signatures accept an optional `project_root` override
      so tests and alternative drivers can point at a different tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────
# Country / indicator canonical lists (Phase 1 output contract)
# ──────────────────────────────────────────────────────────────────
MAIN_COUNTRIES = ['USA', 'JAPAN', 'UK', 'GERMANY']
SUPPLEMENTARY_COUNTRIES = ['CHINA']
ALL_COUNTRIES = MAIN_COUNTRIES + SUPPLEMENTARY_COUNTRIES
INDICATORS = ['CPI', 'POLICY_RATE', 'UNEMPLOYMENT', 'GDP', 'M2']


# ──────────────────────────────────────────────────────────────────
# Project root resolution
# ──────────────────────────────────────────────────────────────────
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk ancestors from `start` (or cwd) until a `data/` directory is
    found. Matches the pattern used in Phase 1 notebooks.

    Raises
    ------
    FileNotFoundError
        If no ancestor contains `data/`.
    """
    cur = Path(start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / 'data').is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root (data/ directory) from {cur}"
    )


# ──────────────────────────────────────────────────────────────────
# Raw data (Phase 1 output)
# ──────────────────────────────────────────────────────────────────
def load_raw_series(
    country: str,
    indicator: str,
    project_root: Optional[Path] = None,
) -> pd.Series:
    """
    Load a single raw CSV produced by Phase 1 and return its value
    column as a pandas Series with DatetimeIndex.

    Parameters
    ----------
    country : str
        One of MAIN_COUNTRIES or SUPPLEMENTARY_COUNTRIES.
    indicator : str
        One of INDICATORS.
    project_root : Path, optional
        Override the auto-detected project root.

    Returns
    -------
    pd.Series
        Named `{COUNTRY}_{INDICATOR}`; DatetimeIndex (unaligned freq —
        monthly raw data will have freq MS-equivalent spacing but is
        not explicitly typed).
    """
    root = project_root or find_project_root()
    path = root / 'data' / 'raw' / f'{country}_{indicator}.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    col = f'{country}_{indicator}'
    if col not in df.columns:
        raise ValueError(
            f"Expected column '{col}' in {path.name}; found {list(df.columns)}"
        )
    s = df[col].copy()
    s.index = pd.to_datetime(s.index)
    return s


def load_all_raw(project_root: Optional[Path] = None) -> Dict[str, Dict[str, pd.Series]]:
    """
    Load all 25 raw series into a nested dict:
        raw[country][indicator] = pd.Series
    """
    raw: Dict[str, Dict[str, pd.Series]] = {}
    for country in ALL_COUNTRIES:
        raw[country] = {}
        for ind in INDICATORS:
            raw[country][ind] = load_raw_series(country, ind, project_root)
    return raw


# ──────────────────────────────────────────────────────────────────
# Processed data (Phase 2 output — consumed by Phase 3+ notebooks)
# ──────────────────────────────────────────────────────────────────
def load_processed_main(
    country: str,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a main-country processed DataFrame from data/processed/.

    Returns
    -------
    pd.DataFrame
        Columns: {COUNTRY}_CPI, _POLICY_RATE, _UNEMPLOYMENT, _GDP, _M2.
        GDP and M2 are harmonised to YoY % growth.
    """
    root = project_root or find_project_root()
    path = root / 'data' / 'processed' / f'main_{country.lower()}.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    return df


def load_processed_all_main(
    project_root: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Return {country: DataFrame} for all 4 main countries."""
    return {c: load_processed_main(c, project_root) for c in MAIN_COUNTRIES}


def load_processed_china(
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the supplementary China DataFrame (sparse; excluded from VAR)."""
    root = project_root or find_project_root()
    path = root / 'data' / 'processed' / 'supplementary_china.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    return df
