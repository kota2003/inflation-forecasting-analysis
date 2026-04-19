"""
src/stationarity.py
===================
Univariate stationarity testing and transformation utilities.

This module consolidates the single-series diagnostics used in Phase 3
Task 1 (stationarity) of the inflation forecasting project.  It is
imported by:
  - notebooks/03_stationarity_structural_breaks.ipynb  (Portfolio narrative)
  - scripts/phase3_step[1-3]_*.py                      (scratch runners)
  - src/structural_breaks.py                           (shared transforms)

Public API
----------
Constants
    DEFAULT_ALPHA              Significance level for 4-quadrant classification
    ADF_REGRESSION_LEVEL       Per-indicator default regression spec for ADF
    TRANSFORM_FN               Mapping from transform name -> callable
    FOUR_QUADRANT_CLASSES      Tuple of the 4 classification strings
    FLAGGED_CLASSES            Classifications that trigger differencing
    CONFLICT_CLASS             'Trend-stationary (conflict)' label
    SMALL_SAMPLE_WARN, SMALL_SAMPLE_HARD
                               Sub-sample observation-count thresholds

Transforms
    first_difference, second_difference, yoy_pct, log_first_diff_pct
    apply_transform(series, transform, indicator)
    strip_suffix(form)

Testing
    schwert_maxlag(n_obs)
    run_adf(series, regression='c', autolag='AIC', maxlag=None)
    run_kpss(series, regression='c', nlags='auto')
    classify_4quadrant(adf_p, kpss_p, alpha=DEFAULT_ALPHA)
    test_series(series, regression, alpha=DEFAULT_ALPHA)

Convenience
    test_all_series(datasets, indicators, regressions)

Decision references: D-024 (joint protocol), D-025 (ADF spec per var),
D-026 (lag selection), D-027 (transformation registry).
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


# ──────────────────────────────────────────────────────────────────
# Constants (public)
# ──────────────────────────────────────────────────────────────────
DEFAULT_ALPHA: float = 0.05

#: Per-indicator default ADF deterministic specification.
#: CPI is a level index with a clear upward trend → constant + trend.
#: Rates, unemployment, and already-YoY growth series → constant only.
#: See D-025 for full rationale.
ADF_REGRESSION_LEVEL: dict[str, str] = {
    'CPI':          'ct',
    'POLICY_RATE':  'c',
    'UNEMPLOYMENT': 'c',
    'GDP':          'c',
    'M2':           'c',
}

#: The four possible classifications under the ADF+KPSS joint protocol.
FOUR_QUADRANT_CLASSES: tuple[str, ...] = (
    'Stationary',
    'Non-stationary',
    'Trend-stationary (conflict)',
    'Inconclusive',
)

#: Classifications that trigger first-differencing in Step 2.
FLAGGED_CLASSES: frozenset[str] = frozenset({'Non-stationary', 'Inconclusive'})
CONFLICT_CLASS: str = 'Trend-stationary (conflict)'

#: Small-sample warning thresholds (Step 3 / Step 4 sub-samples).
SMALL_SAMPLE_WARN: int = 50
SMALL_SAMPLE_HARD: int = 30


# ──────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────
def first_difference(series: pd.Series) -> pd.Series:
    """Return Δx_t = x_t − x_{t−1}, with leading NaN dropped."""
    return series.diff().dropna()


def second_difference(series: pd.Series) -> pd.Series:
    """Return Δ²x_t = Δx_t − Δx_{t−1}, with leading NaNs dropped."""
    return series.diff().diff().dropna()


def yoy_pct(series: pd.Series, periods: int = 12) -> pd.Series:
    """Year-on-year percent change: 100·(x_t/x_{t-periods} − 1).

    Parameters
    ----------
    series : monthly (or any fixed-frequency) time series.
    periods : 12 for monthly YoY; override for quarterly etc.
    """
    return (100.0 * (series / series.shift(periods) - 1.0)).dropna()


def log_first_diff_pct(series: pd.Series) -> pd.Series:
    """Monthly log-change in percent: 100·ln(x_t / x_{t-1}).

    Numerically prefers positive series.  For series that may include
    zeros or negatives, use ``first_difference`` instead.
    """
    s = series.astype(float)
    return (100.0 * np.log(s / s.shift(1))).dropna()


#: Dispatch table for transform strings → functions.
TRANSFORM_FN: dict[str, Callable[[pd.Series], pd.Series]] = {
    'level':         lambda s: s.dropna(),
    'first_diff':    first_difference,
    'second_diff':   second_difference,
    'yoy_pct':       yoy_pct,
    'log_diff_pct':  log_first_diff_pct,
}


# Default per-transform ADF regression spec.  None means "use the
# per-indicator default in ADF_REGRESSION_LEVEL".
_TRANSFORM_DEFAULT_REG: dict[str, str | None] = {
    'level':         None,
    'first_diff':    'c',
    'second_diff':   'c',
    'yoy_pct':       'c',
    'log_diff_pct':  'c',
}


def strip_suffix(form: str) -> str:
    """Normalise a registry value like 'yoy_pct_with_regime_dummy' → 'yoy_pct'.

    Registry entries may carry annotative suffixes such as
    ``_with_regime_dummy`` or ``_with_caveat`` that refine the Phase-6
    treatment instruction but do not change the base transformation.
    This helper recovers the base transform name usable with
    :data:`TRANSFORM_FN`.
    """
    for suffix in ('_with_regime_dummy', '_with_caveat'):
        if form.endswith(suffix):
            return form[: -len(suffix)]
    if form == 'level_with_linear_trend':
        return 'level'
    return form


def apply_transform(series: pd.Series,
                    transform: str,
                    indicator: str) -> tuple[pd.Series, str]:
    """Apply ``transform`` to ``series`` and return (transformed, regression).

    The returned regression is the ADF deterministic spec appropriate for
    the transformed series:
      - For level: defer to ``ADF_REGRESSION_LEVEL[indicator]``
      - For any differencing/YoY: use 'c' (trend absorbed by the transform)

    Parameters
    ----------
    series    : raw input series.
    transform : key of :data:`TRANSFORM_FN` (optionally carrying a
                ``_with_*`` suffix, which is stripped).
    indicator : CPI / POLICY_RATE / UNEMPLOYMENT / GDP / M2 — used only
                when transform resolves to 'level'.

    Returns
    -------
    (transformed_series, regression_spec)
    """
    base = strip_suffix(transform)
    if base not in TRANSFORM_FN:
        raise ValueError(f"Unknown transform: {transform!r} (base {base!r})")
    transformed = TRANSFORM_FN[base](series)
    reg = _TRANSFORM_DEFAULT_REG[base] or ADF_REGRESSION_LEVEL[indicator]
    return transformed, reg


# ──────────────────────────────────────────────────────────────────
# Schwert max-lag rule (D-026)
# ──────────────────────────────────────────────────────────────────
def schwert_maxlag(n_obs: int) -> int:
    """Schwert (1989) upper bound on ADF lag search.

        maxlag = floor(12 · (T/100)^(1/4))

    Floored to a minimum of 1 so that ``autolag='AIC'`` always has at
    least one candidate lag.  For monthly macro series of T ≈ 300 this
    gives maxlag ≈ 15–16, which is the Phase 3 setting.
    """
    return max(1, int(np.floor(12 * (n_obs / 100.0) ** 0.25)))


# ──────────────────────────────────────────────────────────────────
# Core single-series ADF / KPSS wrappers
# ──────────────────────────────────────────────────────────────────
def run_adf(series: pd.Series,
            regression: str = 'c',
            autolag: str = 'AIC',
            maxlag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test with AIC-selected lag.

    H0 : series has a unit root (non-stationary).

    Parameters
    ----------
    series     : non-empty numeric pd.Series (NaNs are dropped).
    regression : 'c', 'ct', 'ctt', or 'n' — deterministic component.
    autolag    : 'AIC', 'BIC', 't-stat', or None.
    maxlag     : If None, use ``schwert_maxlag(len(series.dropna()))``.

    Returns
    -------
    dict with keys:
      adf_stat, adf_pvalue, adf_usedlag, adf_nobs,
      adf_crit_1pct, adf_crit_5pct, adf_crit_10pct,
      adf_maxlag, adf_error (None if success else repr(exception)).
    """
    s = series.dropna()
    if maxlag is None:
        maxlag = schwert_maxlag(len(s))
    try:
        stat, pvalue, usedlag, nobs, crit, _ic = adfuller(
            s.values, maxlag=maxlag, regression=regression, autolag=autolag,
        )
        return {
            'adf_stat':       float(stat),
            'adf_pvalue':     float(pvalue),
            'adf_usedlag':    int(usedlag),
            'adf_nobs':       int(nobs),
            'adf_crit_1pct':  float(crit['1%']),
            'adf_crit_5pct':  float(crit['5%']),
            'adf_crit_10pct': float(crit['10%']),
            'adf_maxlag':     int(maxlag),
            'adf_error':      None,
        }
    except Exception as e:
        return {
            'adf_stat':       np.nan,
            'adf_pvalue':     np.nan,
            'adf_usedlag':    -1,
            'adf_nobs':       -1,
            'adf_crit_1pct':  np.nan,
            'adf_crit_5pct':  np.nan,
            'adf_crit_10pct': np.nan,
            'adf_maxlag':     int(maxlag),
            'adf_error':      repr(e),
        }


def run_kpss(series: pd.Series,
             regression: str = 'c',
             nlags: str | int = 'auto') -> dict:
    """KPSS test for (level- or trend-) stationarity.

    H0 : series is (trend-)stationary around the specified deterministic.

    Parameters
    ----------
    series     : non-empty numeric pd.Series (NaNs are dropped).
    regression : 'c' (level) or 'ct' (trend).
    nlags      : 'auto' (Hobijn et al. 1998) or an integer lag count.

    Notes
    -----
    ``statsmodels.tsa.stattools.kpss`` issues an ``InterpolationWarning``
    when the observed statistic falls outside the tabulated p-value range
    [0.01, 0.10] and clamps the returned p-value to the boundary.  This
    wrapper captures that flag under ``kpss_interp_flag`` so downstream
    classification can be audited.

    Returns
    -------
    dict with keys:
      kpss_stat, kpss_pvalue, kpss_lags,
      kpss_crit_1pct, kpss_crit_5pct, kpss_crit_10pct,
      kpss_interp_flag, kpss_error (None if success else repr(exception)).
    """
    s = series.dropna()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', InterpolationWarning)
        try:
            stat, pvalue, used_nlags, crit = kpss(
                s.values, regression=regression, nlags=nlags,
            )
            interp = any(issubclass(w.category, InterpolationWarning)
                         for w in caught)
            return {
                'kpss_stat':       float(stat),
                'kpss_pvalue':     float(pvalue),
                'kpss_lags':       int(used_nlags),
                'kpss_crit_1pct':  float(crit['1%']),
                'kpss_crit_5pct':  float(crit['5%']),
                'kpss_crit_10pct': float(crit['10%']),
                'kpss_interp_flag': bool(interp),
                'kpss_error':      None,
            }
        except Exception as e:
            return {
                'kpss_stat':       np.nan,
                'kpss_pvalue':     np.nan,
                'kpss_lags':       -1,
                'kpss_crit_1pct':  np.nan,
                'kpss_crit_5pct':  np.nan,
                'kpss_crit_10pct': np.nan,
                'kpss_interp_flag': False,
                'kpss_error':      repr(e),
            }


def classify_4quadrant(adf_p: float,
                       kpss_p: float,
                       alpha: float = DEFAULT_ALPHA) -> str:
    """Four-quadrant classification from joint ADF+KPSS p-values (D-024).

        ADF reject + KPSS non-reject      → 'Stationary'
        ADF non-reject + KPSS reject      → 'Non-stationary'
        ADF reject + KPSS reject          → 'Trend-stationary (conflict)'
        ADF non-reject + KPSS non-reject  → 'Inconclusive'

    Returns 'Error' when either p-value is NaN.
    """
    if np.isnan(adf_p) or np.isnan(kpss_p):
        return 'Error'
    adf_rej  = adf_p  < alpha
    kpss_rej = kpss_p < alpha
    if adf_rej and not kpss_rej:
        return 'Stationary'
    if (not adf_rej) and kpss_rej:
        return 'Non-stationary'
    if adf_rej and kpss_rej:
        return 'Trend-stationary (conflict)'
    return 'Inconclusive'


def test_series(series: pd.Series,
                regression: str,
                alpha: float = DEFAULT_ALPHA) -> dict:
    """Combined ADF + KPSS + 4-quadrant classification on a single series.

    Both tests use the same ``regression`` spec so their null hypotheses
    are comparable on the same deterministic specification.

    Returns
    -------
    dict containing every ``run_adf`` key, every ``run_kpss`` key, plus
    ``classification``, ``n_obs``, ``small_sample_flag``, ``hard_warn_flag``.
    Guards against n < 10 by returning an 'Error' classification.
    """
    n = int(series.dropna().shape[0])
    if n < 10:
        return {
            'adf_stat': np.nan, 'adf_pvalue': np.nan,
            'adf_usedlag': -1,  'adf_nobs': -1,
            'adf_crit_1pct': np.nan, 'adf_crit_5pct': np.nan,
            'adf_crit_10pct': np.nan, 'adf_maxlag': 0,
            'adf_error': 'n < 10',
            'kpss_stat': np.nan, 'kpss_pvalue': np.nan, 'kpss_lags': -1,
            'kpss_crit_1pct': np.nan, 'kpss_crit_5pct': np.nan,
            'kpss_crit_10pct': np.nan,
            'kpss_interp_flag': False, 'kpss_error': 'n < 10',
            'classification': 'Error',
            'n_obs': n,
            'small_sample_flag': True,
            'hard_warn_flag': True,
        }
    adf = run_adf(series, regression=regression)
    kps = run_kpss(series, regression=regression)
    cls = classify_4quadrant(adf['adf_pvalue'], kps['kpss_pvalue'], alpha)
    return {
        **adf, **kps,
        'classification': cls,
        'n_obs': n,
        'small_sample_flag': n < SMALL_SAMPLE_WARN,
        'hard_warn_flag':    n < SMALL_SAMPLE_HARD,
    }


# ──────────────────────────────────────────────────────────────────
# Convenience: batch test across (country × indicator) grid
# ──────────────────────────────────────────────────────────────────
def test_all_series(datasets: dict[str, pd.DataFrame],
                    indicators: list[str] | None = None,
                    regressions: dict[str, str] | None = None,
                    alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Run ``test_series`` on every (country × indicator) level series.

    Parameters
    ----------
    datasets   : ``{country: DataFrame}`` where each DataFrame has columns
                 ``{COUNTRY}_{INDICATOR}``.  Typically the output of
                 ``src.data_loader.load_processed_all_main()``.
    indicators : list of indicators to iterate.  Defaults to the five
                 project indicators in :data:`ADF_REGRESSION_LEVEL`.
    regressions: override ADF regression spec per indicator.  Defaults
                 to :data:`ADF_REGRESSION_LEVEL`.
    alpha      : classification threshold.

    Returns
    -------
    DataFrame with one row per (country, indicator).
    """
    if indicators is None:
        indicators = list(ADF_REGRESSION_LEVEL.keys())
    if regressions is None:
        regressions = dict(ADF_REGRESSION_LEVEL)

    rows: list[dict] = []
    for country, df in datasets.items():
        for indicator in indicators:
            col = f"{country}_{indicator}"
            if col not in df.columns:
                continue
            reg = regressions.get(indicator, 'c')
            result = test_series(df[col], regression=reg, alpha=alpha)
            rows.append({
                'country':   country,
                'indicator': indicator,
                'column':    col,
                'regression': reg,
                **result,
            })
    return pd.DataFrame(rows)
