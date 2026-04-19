"""
scripts/phase3_step2_differencing.py
====================================
Phase 3 · Step 2 — Differencing, re-testing, and CPI transformation comparison.

Purpose
-------
Operate on the Step 1 findings to finalise the stationarity transformation
strategy per series.  Four logical parts:

    Part 0: Re-verify Step 1 level results (self-contained; independent
            of whether Step 1 was run).
    Part 1: Apply first differencing to all series flagged
            'Non-stationary' or 'Inconclusive' at level, re-run ADF+KPSS,
            and re-classify.  If still non-stationary, fall back to
            second-differencing.
    Part 2: Re-test the two 'Trend-stationary (conflict)' series with the
            'ct' specification (constant + trend) to determine whether
            they are genuine trend-stationary or whether differencing is
            preferable.
    Part 3: For CPI (level; non-stationary in all 4 countries), compare
            three alternative transformations on stationarity grounds:
                (a) first_diff  -- ΔCPI  (in index points per month)
                (b) yoy_pct     -- 100·(CPI_t/CPI_{t-12} - 1)    (annual inflation %)
                (c) log_diff    -- 100·Δlog(CPI)                 (monthly inflation %, log-approx)
            This directly informs論点 4 (Chow-test dependent variable)
            and the Phase 6 VAR CPI input specification.
    Part 4: Assemble a Transformation Registry draft (20 rows, one per
            country × indicator) with the recommended Phase 6 input form.

Design decisions embedded (draft — subject to D-024 / D-025 / D-026 / D-027)
---------------------------------------------------------------------------
* First-differenced data is tested with regression='c' (constant only);
  trend terms are removed by differencing itself.
* Second-difference is attempted only as a fallback when first-difference
  is still classified Non-stationary or Inconclusive.  Economic series
  rarely require I(2); persistent I(2) classification flags a
  specification concern rather than a routine result.
* CPI YoY is computed as (CPI_t / CPI_{t-12} - 1) * 100, consistent with
  D-018 / D-012 amended conventions used elsewhere in the project.
* CPI log first-difference is computed as 100 * ln(CPI_t / CPI_{t-1})
  and reported in percent for scale comparability with YoY.
* α = 0.05 for classification.
* The four ADF/KPSS helper functions are duplicated from Step 1 for
  script-level independence; they will be extracted to
  src/stationarity.py in Step 3 per ProjectScope §12.

Inputs
------
data/processed/main_{usa,japan,uk,germany}.csv  via load_processed_all_main().

Outputs
-------
stdout
    Human-readable sectioned report (Parts 0 through 4).
data/documentation/phase3_differencing_log.csv
    One row per series × differencing order tested.
data/documentation/phase3_conflict_ct_retest.csv
    One row per conflict series with 'ct' spec.
data/documentation/phase3_cpi_transform_comparison.csv
    4 countries × 4 transforms (level + 3 alternatives).
data/documentation/phase3_transformation_registry_draft.csv
    20 rows, one per country × indicator, with recommended form.

Usage
-----
Run from the project root:

    python scripts/phase3_step2_differencing.py

Step 1 does not need to have been run first (Part 0 reproduces it).
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from statsmodels.tsa.stattools import adfuller, kpss                # noqa: E402
from statsmodels.tools.sm_exceptions import InterpolationWarning    # noqa: E402

from src.data_loader import (                                       # noqa: E402
    load_processed_all_main,
    INDICATORS,
    MAIN_COUNTRIES,
)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
ALPHA = 0.05

# ADF regression specification at LEVEL (same as Step 1; D-025 draft)
ADF_REGRESSION_LEVEL = {
    'CPI':          'ct',
    'POLICY_RATE':  'c',
    'UNEMPLOYMENT': 'c',
    'GDP':          'c',
    'M2':           'c',
}

# Differenced series always use regression='c' (trend is absorbed by Δ)
REGRESSION_DIFF = 'c'

# Flagged classifications that trigger first-differencing
FLAGGED_CLASSES = {'Non-stationary', 'Inconclusive'}
CONFLICT_CLASS = 'Trend-stationary (conflict)'


def schwert_maxlag(n_obs: int) -> int:
    """Schwert (1989) upper bound on ADF lag search."""
    return int(np.floor(12 * (n_obs / 100.0) ** 0.25))


# ──────────────────────────────────────────────────────────────────
# Test wrappers (duplicated from Step 1; will move to src/stationarity.py in S3)
# ──────────────────────────────────────────────────────────────────
def run_adf(series: pd.Series, regression: str) -> dict:
    s = series.dropna()
    maxlag = schwert_maxlag(len(s))
    try:
        stat, pvalue, usedlag, nobs, crit, _ic = adfuller(
            s.values, maxlag=maxlag, regression=regression, autolag='AIC',
        )
        return {
            'adf_stat': float(stat),         'adf_pvalue': float(pvalue),
            'adf_usedlag': int(usedlag),     'adf_nobs': int(nobs),
            'adf_crit_1pct': float(crit['1%']),
            'adf_crit_5pct': float(crit['5%']),
            'adf_crit_10pct': float(crit['10%']),
            'adf_maxlag': int(maxlag),       'adf_error': None,
        }
    except Exception as e:
        return {
            'adf_stat': np.nan, 'adf_pvalue': np.nan,
            'adf_usedlag': -1,  'adf_nobs': -1,
            'adf_crit_1pct': np.nan, 'adf_crit_5pct': np.nan,
            'adf_crit_10pct': np.nan, 'adf_maxlag': int(maxlag),
            'adf_error': repr(e),
        }


def run_kpss(series: pd.Series, regression: str) -> dict:
    s = series.dropna()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', InterpolationWarning)
        try:
            stat, pvalue, nlags, crit = kpss(
                s.values, regression=regression, nlags='auto',
            )
            interp = any(issubclass(w.category, InterpolationWarning)
                         for w in caught)
            return {
                'kpss_stat': float(stat),    'kpss_pvalue': float(pvalue),
                'kpss_lags': int(nlags),
                'kpss_crit_1pct':  float(crit['1%']),
                'kpss_crit_5pct':  float(crit['5%']),
                'kpss_crit_10pct': float(crit['10%']),
                'kpss_interp_flag': bool(interp), 'kpss_error': None,
            }
        except Exception as e:
            return {
                'kpss_stat': np.nan, 'kpss_pvalue': np.nan, 'kpss_lags': -1,
                'kpss_crit_1pct': np.nan, 'kpss_crit_5pct': np.nan,
                'kpss_crit_10pct': np.nan,
                'kpss_interp_flag': False, 'kpss_error': repr(e),
            }


def classify_4quadrant(adf_p: float, kpss_p: float,
                       alpha: float = ALPHA) -> str:
    if np.isnan(adf_p) or np.isnan(kpss_p):
        return 'Error'
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        return 'Stationary'
    if (not adf_rej) and kpss_rej:
        return 'Non-stationary'
    if adf_rej and kpss_rej:
        return 'Trend-stationary (conflict)'
    return 'Inconclusive'


def test_series(series: pd.Series, regression: str) -> dict:
    """Combined ADF + KPSS + 4-quadrant classification on one series."""
    adf = run_adf(series, regression)
    kps = run_kpss(series, regression)
    cls = classify_4quadrant(adf['adf_pvalue'], kps['kpss_pvalue'])
    return {**adf, **kps, 'classification': cls,
            'n_obs': int(series.dropna().shape[0])}


# ──────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────
def first_difference(series: pd.Series) -> pd.Series:
    """Δx_t = x_t - x_{t-1}."""
    return series.diff().dropna()


def second_difference(series: pd.Series) -> pd.Series:
    """Δ²x_t = Δx_t - Δx_{t-1}."""
    return series.diff().diff().dropna()


def yoy_pct(series: pd.Series, periods: int = 12) -> pd.Series:
    """Year-on-year percent change: 100·(x_t/x_{t-12} - 1)."""
    return (100.0 * (series / series.shift(periods) - 1.0)).dropna()


def log_first_diff_pct(series: pd.Series) -> pd.Series:
    """Monthly log change in percent: 100·ln(x_t / x_{t-1})."""
    s = series.astype(float)
    return (100.0 * np.log(s / s.shift(1))).dropna()


# ──────────────────────────────────────────────────────────────────
# Helpers for pretty printing
# ──────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def subsection(title: str) -> None:
    print("\n" + "-" * 79)
    print(title)
    print("-" * 79)


def fmt_numeric_cols(df: pd.DataFrame, cols_map: dict) -> pd.DataFrame:
    """Format numeric columns in-place to fixed-precision strings."""
    out = df.copy()
    for c, fmt in cols_map.items():
        if c in out.columns:
            out[c] = out[c].map(lambda x: fmt.format(x)
                                if pd.notnull(x) and not isinstance(x, str)
                                else x)
    return out


# ──────────────────────────────────────────────────────────────────
# Part 0: Level results (reproduces Step 1, self-contained)
# ──────────────────────────────────────────────────────────────────
def run_part0_levels(datasets: dict) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = datasets[country]
        for indicator in INDICATORS:
            col = f"{country}_{indicator}"
            if col not in df.columns:
                continue
            regression = ADF_REGRESSION_LEVEL[indicator]
            result = test_series(df[col], regression=regression)
            rows.append({
                'country': country, 'indicator': indicator, 'column': col,
                'transform': 'level', 'regression': regression,
                **result,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 1: First- (and if needed second-) differencing of flagged series
# ──────────────────────────────────────────────────────────────────
def run_part1_differencing(datasets: dict,
                           level_results: pd.DataFrame) -> pd.DataFrame:
    flagged = level_results[
        level_results['classification'].isin(FLAGGED_CLASSES)
    ][['country', 'indicator', 'column', 'classification']].rename(
        columns={'classification': 'level_classification'}
    )

    rows = []
    for _, r in flagged.iterrows():
        country, indicator, col = r['country'], r['indicator'], r['column']
        series = datasets[country][col]

        # First-difference test
        d1 = first_difference(series)
        res_d1 = test_series(d1, regression=REGRESSION_DIFF)
        rows.append({
            'country': country, 'indicator': indicator, 'column': col,
            'level_classification': r['level_classification'],
            'transform': 'first_diff', 'regression': REGRESSION_DIFF,
            **res_d1,
        })

        # Second-difference fallback only if first-difference is not resolved
        if res_d1['classification'] in (FLAGGED_CLASSES | {'Error'}):
            d2 = second_difference(series)
            res_d2 = test_series(d2, regression=REGRESSION_DIFF)
            rows.append({
                'country': country, 'indicator': indicator, 'column': col,
                'level_classification': r['level_classification'],
                'transform': 'second_diff', 'regression': REGRESSION_DIFF,
                **res_d2,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 2: 'ct' retest on conflict series
# ──────────────────────────────────────────────────────────────────
def run_part2_conflict_ct(datasets: dict,
                          level_results: pd.DataFrame) -> pd.DataFrame:
    conflicts = level_results[
        level_results['classification'] == CONFLICT_CLASS
    ][['country', 'indicator', 'column', 'regression', 'classification']].copy()
    conflicts = conflicts.rename(columns={
        'regression': 'original_regression',
        'classification': 'original_classification',
    })

    rows = []
    for _, r in conflicts.iterrows():
        country, indicator, col = r['country'], r['indicator'], r['column']
        series = datasets[country][col]
        # Retest with 'ct'
        result = test_series(series, regression='ct')
        rows.append({
            'country': country, 'indicator': indicator, 'column': col,
            'original_regression': r['original_regression'],
            'original_classification': r['original_classification'],
            'retest_regression': 'ct',
            **result,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 3: CPI transformation comparison
# ──────────────────────────────────────────────────────────────────
def run_part3_cpi_transforms(datasets: dict) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        col = f"{country}_CPI"
        series = datasets[country][col]

        transforms = {
            'level':        (series,                         'ct'),
            'first_diff':   (first_difference(series),       'c'),
            'yoy_pct':      (yoy_pct(series, periods=12),    'c'),
            'log_diff_pct': (log_first_diff_pct(series),     'c'),
        }
        for tname, (s, reg) in transforms.items():
            result = test_series(s, regression=reg)
            rows.append({
                'country': country, 'indicator': 'CPI', 'column': col,
                'transform': tname, 'regression': reg,
                **result,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 4: Transformation registry draft
# ──────────────────────────────────────────────────────────────────
def build_transformation_registry(level_results: pd.DataFrame,
                                  diff_results: pd.DataFrame,
                                  ct_results: pd.DataFrame,
                                  cpi_results: pd.DataFrame) -> pd.DataFrame:
    """Per country × indicator, recommend Phase 6 input form + rationale.

    Policy encoded here (to be ratified in D-027):
      - CPI: adopt YoY if yoy_pct is Stationary; otherwise fall back to
             log_diff_pct or first_diff in that priority order.
      - Stationary at level (GDP/M2 etc.): keep level.
      - Non-stationary / Inconclusive: use first_diff if it resolves;
             else second_diff with a flagged caveat.
      - Trend-stationary conflict: prefer first_diff over detrending
             ('ct') for cross-country consistency and because trend-
             stationarity on one variable complicates VAR lag specs.
             Record both outcomes for reviewer transparency.
    """
    rows = []
    lvl = level_results.set_index(['country', 'indicator'])
    d1  = diff_results[diff_results['transform'] == 'first_diff'] \
              .set_index(['country', 'indicator'])
    d2  = diff_results[diff_results['transform'] == 'second_diff'] \
              .set_index(['country', 'indicator'])
    ct  = ct_results.set_index(['country', 'indicator']) \
              if len(ct_results) else None
    cpi = cpi_results.set_index(['country', 'transform'])

    for country in MAIN_COUNTRIES:
        for indicator in INDICATORS:
            lvl_row = lvl.loc[(country, indicator)]
            lvl_cls = lvl_row['classification']
            lvl_pv_adf = lvl_row['adf_pvalue']
            lvl_pv_kps = lvl_row['kpss_pvalue']

            recommended = None
            justification = None
            post_transform_cls = None

            if indicator == 'CPI':
                # CPI policy: prefer YoY if stationary, else log_diff, else first_diff
                yoy_cls = cpi.loc[(country, 'yoy_pct')]['classification']
                log_cls = cpi.loc[(country, 'log_diff_pct')]['classification']
                fd_cls  = cpi.loc[(country, 'first_diff')]['classification']
                if yoy_cls == 'Stationary':
                    recommended = 'yoy_pct'
                    post_transform_cls = yoy_cls
                    justification = ("CPI YoY stationary; aligns with D-018/D-012 "
                                     "YoY convention across GDP/M2; chosen for "
                                     "cross-indicator consistency and N1 Phillips "
                                     "Curve narrative (inflation as CPI growth).")
                elif log_cls == 'Stationary':
                    recommended = 'log_diff_pct'
                    post_transform_cls = log_cls
                    justification = ("YoY not fully stationary; monthly log change "
                                     "is statistically preferred as Phase 6 input; "
                                     "YoY reported as narrative variable only.")
                elif fd_cls == 'Stationary':
                    recommended = 'first_diff'
                    post_transform_cls = fd_cls
                    justification = ("YoY and log-diff both non-stationary at α; "
                                     "ΔCPI (index points) used as VAR input with "
                                     "caveat that scale differs from pct forms.")
                else:
                    recommended = 'yoy_pct_with_caveat'
                    post_transform_cls = yoy_cls
                    justification = ("All three CPI transforms remain flagged; "
                                     "adopt YoY for narrative alignment and "
                                     "re-examine in Phase 6 robustness.")

            elif lvl_cls == 'Stationary':
                recommended = 'level'
                post_transform_cls = 'Stationary'
                justification = ("Level-stationary under joint ADF+KPSS; used "
                                 "directly in Phase 6 VAR.")

            elif lvl_cls == CONFLICT_CLASS:
                # Compare 'c' (level, conflict) vs 'ct' retest vs first_diff
                ct_cls = ct.loc[(country, indicator)]['classification'] \
                            if ct is not None else 'Error'
                d1_cls = d1.loc[(country, indicator)]['classification'] \
                            if (country, indicator) in d1.index else 'Error'
                if d1_cls == 'Stationary':
                    recommended = 'first_diff'
                    post_transform_cls = d1_cls
                    justification = (f"Level 'c' gave conflict; 'ct' retest = "
                                     f"{ct_cls}; first_diff = Stationary and "
                                     "preferred for VAR cross-country consistency.")
                elif ct_cls == 'Stationary':
                    recommended = 'level_ct'
                    post_transform_cls = ct_cls
                    justification = (f"Trend-stationary under 'ct' (p-values "
                                     f"resolve); first_diff = {d1_cls}. Use level "
                                     "with deterministic trend in VAR.")
                else:
                    recommended = 'first_diff_with_caveat'
                    post_transform_cls = d1_cls
                    justification = ("Neither 'ct' nor first_diff fully resolves; "
                                     "first_diff adopted with caveat.")

            elif lvl_cls in FLAGGED_CLASSES:
                # Flagged at level; check first-diff then second-diff
                d1_cls = d1.loc[(country, indicator)]['classification'] \
                            if (country, indicator) in d1.index else 'Error'
                if d1_cls == 'Stationary':
                    recommended = 'first_diff'
                    post_transform_cls = d1_cls
                    justification = (f"Level {lvl_cls}; first_diff yields "
                                     "Stationary. Standard I(1) treatment.")
                elif (country, indicator) in d2.index:
                    d2_cls = d2.loc[(country, indicator)]['classification']
                    if d2_cls == 'Stationary':
                        recommended = 'second_diff'
                        post_transform_cls = d2_cls
                        justification = (f"Level {lvl_cls}; first_diff {d1_cls}; "
                                         "second_diff resolves. I(2) - review in "
                                         "Phase 6 for economic interpretability.")
                    else:
                        recommended = 'first_diff_with_caveat'
                        post_transform_cls = d1_cls
                        justification = (f"Neither first nor second difference "
                                         f"fully resolves (d1={d1_cls}, d2={d2_cls}). "
                                         "Adopt first_diff with robustness caveat.")
                else:
                    recommended = 'first_diff_with_caveat'
                    post_transform_cls = d1_cls
                    justification = (f"Level {lvl_cls}; first_diff = {d1_cls}. "
                                     "Second-diff not attempted. Adopt first_diff.")
            else:
                recommended = 'unknown'
                post_transform_cls = lvl_cls
                justification = "Unclassified at level; investigate."

            rows.append({
                'country': country,
                'indicator': indicator,
                'level_classification': lvl_cls,
                'level_adf_pvalue': lvl_pv_adf,
                'level_kpss_pvalue': lvl_pv_kps,
                'recommended_transform': recommended,
                'post_transform_classification': post_transform_cls,
                'justification': justification,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 79)
    print("Phase 3 · Step 2 — Differencing, conflict retest, CPI comparison")
    print(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Project   : {PROJECT_ROOT}")
    print(f"Alpha     : {ALPHA}")
    print("=" * 79)

    datasets = load_processed_all_main()
    print(f"\nLoaded {len(datasets)} main-country datasets:")
    for c, df in datasets.items():
        print(f"  {c:<8s} : {df.shape[0]} rows x {df.shape[1]} cols  "
              f"({df.index.min():%Y-%m} -> {df.index.max():%Y-%m})")

    # ── Part 0 ────────────────────────────────────────────────
    section("PART 0 — Level results (Step 1 reproduction, self-contained)")
    level_results = run_part0_levels(datasets)
    p0 = level_results[['country', 'indicator', 'regression', 'n_obs',
                        'adf_pvalue', 'kpss_pvalue', 'classification']].copy()
    p0 = fmt_numeric_cols(p0, {
        'adf_pvalue': '{:.4f}', 'kpss_pvalue': '{:.4f}',
    })
    print(p0.to_string(index=False))

    cnt = level_results['classification'].value_counts()
    subsection("Level summary")
    for k, v in cnt.items():
        print(f"  {k:<32s} {v:>3d}")

    # ── Part 1 ────────────────────────────────────────────────
    section("PART 1 — First-difference re-test (flagged series)")
    diff_results = run_part1_differencing(datasets, level_results)

    fd = diff_results[diff_results['transform'] == 'first_diff'].copy()
    sd = diff_results[diff_results['transform'] == 'second_diff'].copy()

    subsection(f"First-difference results  ({len(fd)} series)")
    p1 = fd[['country', 'indicator', 'level_classification', 'n_obs',
             'adf_stat', 'adf_pvalue', 'kpss_stat', 'kpss_pvalue',
             'classification']].copy()
    p1 = fmt_numeric_cols(p1, {
        'adf_stat': '{:>8.4f}',   'adf_pvalue':  '{:.4f}',
        'kpss_stat': '{:>7.4f}',  'kpss_pvalue': '{:.4f}',
    })
    print(p1.to_string(index=False))

    fd_cnt = fd['classification'].value_counts()
    subsection("First-difference classification summary")
    for k, v in fd_cnt.items():
        print(f"  {k:<32s} {v:>3d}")

    unresolved = fd[fd['classification'].isin(FLAGGED_CLASSES | {'Error'})]
    if len(unresolved):
        subsection(f"Unresolved after first-diff "
                   f"({len(unresolved)} series) -> trying second-diff")
        if len(sd):
            p1b = sd[['country', 'indicator', 'n_obs', 'adf_pvalue',
                      'kpss_pvalue', 'classification']].copy()
            p1b = fmt_numeric_cols(p1b, {
                'adf_pvalue': '{:.4f}', 'kpss_pvalue': '{:.4f}',
            })
            print(p1b.to_string(index=False))
    else:
        subsection("All first-differenced series resolved "
                   "(no second-diff required).")

    # ── Part 2 ────────────────────────────────────────────────
    section("PART 2 — 'ct' retest on conflict series")
    ct_results = run_part2_conflict_ct(datasets, level_results)
    if len(ct_results):
        p2 = ct_results[['country', 'indicator',
                         'original_regression', 'original_classification',
                         'retest_regression', 'n_obs',
                         'adf_pvalue', 'kpss_pvalue',
                         'classification']].copy()
        p2 = fmt_numeric_cols(p2, {
            'adf_pvalue': '{:.4f}', 'kpss_pvalue': '{:.4f}',
        })
        print(p2.to_string(index=False))
        subsection("Interpretation hint")
        print("  'Stationary' under 'ct' indicates trend-stationary: the series is "
              "stationary once a linear trend is removed. For VAR use, either:")
        print("    (a) include a deterministic trend term, or")
        print("    (b) first-difference. Cross-country consistency typically "
              "favours (b).")
    else:
        print("  No conflict series detected at level; Part 2 skipped.")

    # ── Part 3 ────────────────────────────────────────────────
    section("PART 3 — CPI transformation comparison (levels vs. 3 alternatives)")
    cpi_results = run_part3_cpi_transforms(datasets)
    p3 = cpi_results[['country', 'transform', 'regression', 'n_obs',
                      'adf_stat', 'adf_pvalue', 'kpss_stat', 'kpss_pvalue',
                      'classification']].copy()
    p3 = fmt_numeric_cols(p3, {
        'adf_stat': '{:>8.4f}',   'adf_pvalue':  '{:.4f}',
        'kpss_stat': '{:>7.4f}',  'kpss_pvalue': '{:.4f}',
    })
    print(p3.to_string(index=False))

    subsection("CPI transform classification matrix "
               "(rows=country, cols=transform)")
    pivot = (cpi_results
             .pivot(index='country', columns='transform',
                    values='classification')
             .reindex(index=MAIN_COUNTRIES,
                      columns=['level', 'first_diff', 'yoy_pct', 'log_diff_pct']))
    print(pivot.to_string())

    subsection("Preferred CPI transform per country (policy: YoY > log_diff > first_diff)")
    for country in MAIN_COUNTRIES:
        pref = None
        for t in ['yoy_pct', 'log_diff_pct', 'first_diff']:
            cls = pivot.loc[country, t]
            if cls == 'Stationary':
                pref = t
                break
        if pref is None:
            # fallback: report best p-values
            pref = 'yoy_pct (caveat)'
        print(f"  {country:<8s} -> {pref}")

    # ── Part 4 ────────────────────────────────────────────────
    section("PART 4 — Transformation Registry (draft for D-027)")
    registry = build_transformation_registry(
        level_results, diff_results, ct_results, cpi_results,
    )
    p4 = registry[['country', 'indicator', 'level_classification',
                   'recommended_transform', 'post_transform_classification']].copy()
    print(p4.to_string(index=False))

    subsection("Justifications (truncated to 100 chars per row)")
    for _, r in registry.iterrows():
        j = (r['justification'] or '')[:100]
        print(f"  {r['country']:<8s} {r['indicator']:<12s} "
              f"[{r['recommended_transform']:<22s}]  {j}")

    subsection("Registry summary — recommended transform counts")
    reg_cnt = registry['recommended_transform'].value_counts()
    for k, v in reg_cnt.items():
        print(f"  {k:<30s} {v:>3d}")

    # ── Write audit CSVs ──────────────────────────────────────
    section("CSV outputs")
    doc_dir = PROJECT_ROOT / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [
        ('phase3_differencing_log.csv',              diff_results),
        ('phase3_conflict_ct_retest.csv',            ct_results),
        ('phase3_cpi_transform_comparison.csv',      cpi_results),
        ('phase3_transformation_registry_draft.csv', registry),
    ]:
        path = doc_dir / name
        df.to_csv(path, index=False)
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        print(f"  wrote {rel:<55s}  ({len(df)} rows)")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
