"""
src/feature_engineering.py
==========================
Phase 4 feature matrix construction for the inflation forecasting project.

Consolidates the four Phase 4 pipeline stages (base transform → lags →
rolling stats → regime dummies) into a single API used by:
  - notebooks/04_feature_engineering.ipynb
  - scripts/phase4_step5_*.py

Public API
----------
Constants (single source of truth)
    REGISTRY_OVERRIDES     D-031 runtime overrides
    LAG_PERIODS            D-034 lag grid {1, 3, 6, 12}
    ROLLING_WINDOWS        D-035 rolling windows {3, 12}
    ROLLING_STATS          D-035 ('mean', 'std')
    PERIOD_WINDOWS         D-036 anomaly flag windows
    PHASE6_REGIME_SPEC     D-030 dominant-driver matrix

Registry
    load_effective_registry(project_root=None)

Components
    transform_country(df, country, eff_reg)
    build_lag_matrix(base_df, lag_periods=LAG_PERIODS)
    build_rolling_matrix(base_df, windows=..., stats=...)
    build_split_dummies(country, index)
    build_period_dummies(country, index)
    build_interactions(country, base_df, splits)
    build_regime_matrix(country, base_df)

Assembly
    build_country_features(country, df=None, eff_reg=None, project_root=None)
    build_all_features(processed=None, project_root=None)

Schema
    write_features_schema_md(features, dest=None, project_root=None)

Decision references: D-030, D-031, D-034, D-035, D-036, D-037, D-038,
D-039, D-040.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import (
    MAIN_COUNTRIES,
    INDICATORS,
    find_project_root,
    load_processed_main,
    load_processed_all_main,
)
from .stationarity import apply_transform, strip_suffix
from .structural_breaks import KNOWN_BREAKS, make_split_dummy


# ── Constants (decision-log single source of truth) ────────────────

#: D-031 runtime overrides to the Phase 3 transformation registry.
#: The CSV preserves the Phase 3 Step 3 initial state for JPN / UK / GER
#: CPI; these overrides encode the D-031 revised specification that
#: drives Phase 4 and Phase 6 input forms.
REGISTRY_OVERRIDES: dict[tuple[str, str], str] = {
    ('JAPAN',   'CPI'): 'first_diff',    # was 'yoy_pct_with_regime_dummy'
    ('GERMANY', 'CPI'): 'first_diff',    # was 'yoy_pct_with_caveat'
    ('UK',      'CPI'): 'log_diff_pct',  # was 'log_diff_pct_with_caveat'
}

#: D-034 — ProjectScope §9 uniform sparse lag grid.  Samples short-run
#: (1m), quarterly (3m), semi-annual (6m), and annual (12m) dynamics
#: without flooding Phase 6 with a dense specification.
LAG_PERIODS: tuple[int, ...] = (1, 3, 6, 12)

#: D-035 — rolling window / statistic grid.  Spec exceeds §9 "mean only"
#: by adding sample std (ddof=1) to capture volatility regime directly
#: motivated by Phase 3 structural-break findings (COVID/ENERGY shocks
#: manifest as both level and variance shifts).
ROLLING_WINDOWS: tuple[int, ...] = (3, 12)
ROLLING_STATS:   tuple[str, ...] = ('mean', 'std')

#: D-036 — anomaly flag period windows per ProjectScope §9.  COVID
#: matches src.structural_breaks.COVID_DUMMY_START/END (D-029) exactly.
#: GFC is break-date-anchored to NBER recession end (2009-06).
PERIOD_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    'GFC':   (pd.Timestamp('2008-09-01'), pd.Timestamp('2009-06-01')),
    'COVID': (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-09-01')),
}

#: D-030 dominant-driver matrix (Bonferroni-significant pairs only).
#: Value is the dominant regressor name, 'const' for intercept-shift
#: cases (no interaction emitted), or None for non-significant pairs.
PHASE6_REGIME_SPEC: dict[tuple[str, str], Optional[str]] = {
    ('USA',     'GFC_2008'):    'M2',
    ('USA',     'COVID_2020'):  'POLICY_RATE',
    ('USA',     'ENERGY_2022'): 'POLICY_RATE',
    ('JAPAN',   'GFC_2008'):    None,         # not significant
    ('JAPAN',   'COVID_2020'):  'const',      # intercept shift only
    ('JAPAN',   'ENERGY_2022'): 'const',
    ('UK',      'GFC_2008'):    None,
    ('UK',      'COVID_2020'):  'const',
    ('UK',      'ENERGY_2022'): 'GDP',
    ('GERMANY', 'GFC_2008'):    None,
    ('GERMANY', 'COVID_2020'):  'GDP',
    ('GERMANY', 'ENERGY_2022'): 'GDP',
}


# ── Registry loading ───────────────────────────────────────────────

def load_effective_registry(
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load phase3_transformation_registry_final.csv and apply D-031.

    Returns
    -------
    pd.DataFrame with columns:
        country, indicator,
        registry_phase6_var_input    raw value from the CSV,
        effective_phase6_var_input   post-override,
        override_applied             bool,
        effective_transform_base     suffix-stripped base transform name.
    """
    root = project_root or find_project_root()
    path = root / 'data' / 'documentation' \
        / 'phase3_transformation_registry_final.csv'
    reg = pd.read_csv(path)[['country', 'indicator', 'phase6_var_input']].copy()
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
    reg['effective_transform_base']   = (
        reg['effective_phase6_var_input'].map(strip_suffix)
    )
    return reg


# ── Base transformation ────────────────────────────────────────────

def transform_country(
    df: pd.DataFrame,
    country: str,
    eff_reg: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-indicator effective transform; return wide base frame.

    Output columns: ``{COUNTRY}_{INDICATOR}`` × 5.  Leading NaN is
    preserved where a transformation truncates (e.g. ``yoy_pct`` drops
    12 observations).
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
        transformed[src_col], _ = apply_transform(df[src_col], form, ind)
    out = pd.DataFrame(transformed)
    out.index.name = 'date'
    return out


# ── Lag matrix ─────────────────────────────────────────────────────

def build_lag_matrix(
    base_df: pd.DataFrame,
    lag_periods: tuple[int, ...] = LAG_PERIODS,
) -> pd.DataFrame:
    """Generate ``{col}_lag{k}`` for each (col, k in lag_periods).

    Uses ``pd.Series.shift(k)`` which prepends k leading NaN on the same
    DatetimeIndex.  Column order: outer indicator × inner lag.
    """
    out_cols: dict[str, pd.Series] = {}
    for col in base_df.columns:
        for k in lag_periods:
            out_cols[f'{col}_lag{k}'] = base_df[col].shift(k)
    return pd.DataFrame(out_cols, index=base_df.index)


# ── Rolling matrix ─────────────────────────────────────────────────

def build_rolling_matrix(
    base_df: pd.DataFrame,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    stats:   tuple[str, ...] = ROLLING_STATS,
) -> pd.DataFrame:
    """Generate ``{col}_roll{w}_{stat}`` for each (col, w, stat).

    Right-aligned inclusive; strict ``min_periods = window``; std
    ``ddof = 1``.  Leading (w-1) observations are NaN.
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


# ── Regime dummies ─────────────────────────────────────────────────

def _make_period_dummy(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end:   pd.Timestamp,
) -> pd.Series:
    """``P_t = 1{start <= t <= end}`` as float64 Series aligned on index."""
    mask = (index >= start) & (index <= end)
    return pd.Series(mask.astype('float64'), index=index)


def build_split_dummies(
    country: str,
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """3 persistent split dummies: ``{COUNTRY}_D_{BREAK_NAME}`` per KNOWN_BREAKS."""
    out: dict[str, pd.Series] = {}
    for name, dt in KNOWN_BREAKS.items():
        col = f'{country}_D_{name}'
        d = make_split_dummy(index, dt).astype('float64')
        d.name = col
        out[col] = d
    return out


def build_period_dummies(
    country: str,
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """2 temporary period dummies per D-036 windows."""
    out: dict[str, pd.Series] = {}
    for name, (start, end) in PERIOD_WINDOWS.items():
        col = f'{country}_P_{name}'
        p = _make_period_dummy(index, start, end)
        p.name = col
        out[col] = p
    return out


def build_interactions(
    country: str,
    base_df: pd.DataFrame,
    splits: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """D-030 interaction terms (regressor drivers only; gated by spec).

    Returns 0–3 columns per country depending on how many (country ×
    break) pairs have a regressor dominant driver.  Constant drivers
    and non-significant pairs emit NO interaction.
    """
    out: dict[str, pd.Series] = {}
    for (c, break_name), driver in PHASE6_REGIME_SPEC.items():
        if c != country:
            continue
        if driver is None or driver == 'const':
            continue
        split_col  = f'{country}_D_{break_name}'
        driver_col = f'{country}_{driver}'
        if split_col not in splits:
            raise KeyError(f'Split dummy {split_col} missing')
        if driver_col not in base_df.columns:
            raise KeyError(f'Driver column {driver_col} missing in base_df')
        inter_col = f'{country}_{driver}_x_D_{break_name}'
        inter = splits[split_col] * base_df[driver_col]
        inter.name = inter_col
        out[inter_col] = inter
    return out


def build_regime_matrix(
    country: str,
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble splits + periods + interactions.

    Column order: 3 splits → 2 periods → 0–3 interactions.
    """
    index = base_df.index
    splits       = build_split_dummies(country, index)
    periods      = build_period_dummies(country, index)
    interactions = build_interactions(country, base_df, splits)
    return pd.DataFrame({**splits, **periods, **interactions}, index=index)


# ── Assembly ───────────────────────────────────────────────────────

def build_country_features(
    country: str,
    df: Optional[pd.DataFrame] = None,
    eff_reg: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the full Phase 4 feature matrix for a single country.

    Column ordering
    ---------------
        1. Base t=0 columns (5)
        2. Lag columns (5 × len(LAG_PERIODS))
        3. Rolling columns (5 × len(ROLLING_WINDOWS) × len(ROLLING_STATS))
        4. Regime columns (splits → periods → interactions)
    """
    if eff_reg is None:
        eff_reg = load_effective_registry(project_root)
    if df is None:
        df = load_processed_main(country, project_root)

    base    = transform_country(df, country, eff_reg)
    lags    = build_lag_matrix(base, LAG_PERIODS)
    rolling = build_rolling_matrix(base, ROLLING_WINDOWS, ROLLING_STATS)
    regime  = build_regime_matrix(country, base)

    out = pd.concat([base, lags, rolling, regime], axis=1)
    out.index.name = 'date'
    return out


def build_all_features(
    processed: Optional[dict[str, pd.DataFrame]] = None,
    project_root: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Return ``{country: feature DataFrame}`` for all MAIN_COUNTRIES."""
    if processed is None:
        processed = load_processed_all_main(project_root)
    eff_reg = load_effective_registry(project_root)
    return {
        c: build_country_features(c, processed[c], eff_reg, project_root)
        for c in MAIN_COUNTRIES
    }


# ── Schema writer ──────────────────────────────────────────────────

def _categorise_column(col: str) -> str:
    """One of: 'base', 'lag', 'rolling', 'split', 'period', 'interaction'."""
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


def write_features_schema_md(
    features: dict[str, pd.DataFrame],
    dest: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """Write ``data/processed/features_schema.md`` documenting the output.

    Mirrors the pattern of ``src.preprocessing.write_schema_md()``.
    """
    root = project_root or find_project_root()
    dest = dest or (root / 'data' / 'processed' / 'features_schema.md')

    lines: list[str] = []
    lines += [
        '# `data/processed/features_*.csv` — Phase 4 Feature Schema',
        '',
        '*Auto-generated by `src.feature_engineering.write_features_schema_md()`. '
        'Do not edit by hand; regenerate via '
        '`python scripts/phase4_step5_assemble.py` or by executing '
        '`notebooks/04_feature_engineering.ipynb`.*',
        '',
        f'**Last generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '---',
        '',
        '## File Summary',
        '',
        '| File | Rows | Cols | Joint-Valid Start | Joint-Valid End | NaN (total) |',
        '|---|---:|---:|---|---|---:|',
    ]
    for c in MAIN_COUNTRIES:
        feat = features[c]
        jv = feat.dropna(how='any')
        jv_start = jv.index.min().date() if len(jv) else 'n/a'
        jv_last  = jv.index.max().date() if len(jv) else 'n/a'
        lines.append(
            f'| `features_{c.lower()}.csv` | {len(feat)} | {feat.shape[1]} | '
            f'{jv_start} | {jv_last} | {int(feat.isna().sum().sum())} |'
        )

    lines += [
        '',
        '## Column Categories',
        '',
        '| Category | Pattern | Count / country | Decision |',
        '|---|---|---:|---|',
        '| base | `{COUNTRY}_{INDICATOR}` | 5 | D-027 / D-031 |',
        f'| lag | `{{COUNTRY}}_{{INDICATOR}}_lag{{k}}` | '
        f'{5 * len(LAG_PERIODS)} | D-034 (k ∈ {list(LAG_PERIODS)}) |',
        f'| rolling | `{{COUNTRY}}_{{INDICATOR}}_roll{{w}}_{{stat}}` | '
        f'{5 * len(ROLLING_WINDOWS) * len(ROLLING_STATS)} | '
        f'D-035 (w ∈ {list(ROLLING_WINDOWS)}, stat ∈ {list(ROLLING_STATS)}) |',
        '| split | `{COUNTRY}_D_{BREAK}` | 3 | D-030 / D-036 |',
        '| period | `{COUNTRY}_P_{PERIOD}` | 2 | D-036 |',
        '| interaction | `{COUNTRY}_{DRIVER}_x_D_{BREAK}` | 0–3 | D-030 gated |',
        '',
        '## Per-Country Category Counts',
        '',
        '| Country | base | lag | rolling | split | period | interaction | total |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for c in MAIN_COUNTRIES:
        feat = features[c]
        cats = pd.Series({col: _categorise_column(col) for col in feat.columns})
        counts = cats.value_counts()
        lines.append(
            f'| {c} | {int(counts.get("base", 0))} | '
            f'{int(counts.get("lag", 0))} | {int(counts.get("rolling", 0))} | '
            f'{int(counts.get("split", 0))} | {int(counts.get("period", 0))} | '
            f'{int(counts.get("interaction", 0))} | {len(feat.columns)} |'
        )

    lines += [
        '',
        '## Decision References',
        '',
        '- **D-027, D-031** — base column transforms (Phase 3 Transformation Registry + D-031 overrides)',
        '- **D-030** — interaction term specification (dominant-driver matrix, Bonferroni-gated)',
        '- **D-034** — lag grid {1, 3, 6, 12} (ProjectScope §9 literal)',
        '- **D-035** — rolling windows {3, 12} × {mean, std} (§9 mean + volatility upgrade)',
        '- **D-036** — regime dummy structure: 3 splits + 2 periods + 0–3 interactions',
        '- **D-037** — `src/feature_engineering.py` module API',
        '- **D-038** — D-031 override embedded in module as `REGISTRY_OVERRIDES`',
        '- **D-039** — per-country wide CSV; leading NaN preserved for Phase 6 flexibility',
        '- **D-040** — no preliminary sparsity; feature selection deferred to Phase 6',
        '',
        'See `ProjectDriven.md` for full decision rationale.',
        '',
    ]
    dest.write_text('\n'.join(lines), encoding='utf-8')
    return dest


__all__ = [
    # Constants
    'REGISTRY_OVERRIDES',
    'LAG_PERIODS',
    'ROLLING_WINDOWS',
    'ROLLING_STATS',
    'PERIOD_WINDOWS',
    'PHASE6_REGIME_SPEC',
    # Registry
    'load_effective_registry',
    # Components
    'transform_country',
    'build_lag_matrix',
    'build_rolling_matrix',
    'build_split_dummies',
    'build_period_dummies',
    'build_interactions',
    'build_regime_matrix',
    # Assembly
    'build_country_features',
    'build_all_features',
    # Schema
    'write_features_schema_md',
]