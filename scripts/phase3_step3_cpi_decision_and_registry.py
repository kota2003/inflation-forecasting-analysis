"""
scripts/phase3_step3_cpi_decision_and_registry.py
==================================================
Phase 3 · Step 3 — Sub-period diagnostics, CPI decision, final
Transformation Registry with VAR/Chow-test input split, and Chow-test
break-window pre-check.

Purpose
-------
Five sequential parts that finalise Phase 3 Task 1 and prepare Task 2:

    Part 0: Conflict first-diff fix. Step 2's Part 1 flagged only
            {Non-stationary, Inconclusive}; the two 'Trend-stationary
            (conflict)' series (USA UNEMPLOYMENT, Germany POLICY_RATE)
            were missed.  This part applies first-diff ADF+KPSS to both
            and resolves the Error entries in the Step 2 registry.

    Part 1: Sub-period ADF+KPSS on all 20 series at their recommended
            Phase-6 input form.  Split at 2020-01-01:
                pre  = 2001-01 .. 2019-12
                post = 2020-01 .. dataset end
            Purpose: diagnose whether full-sample non-stationarity is
            driven by a regime shift (each sub-period stationary) or
            by intrinsic non-stationarity (at least one sub-period
            non-stationary).

    Part 2: CPI-specific deep dive.  For each country × each CPI form
            (first_diff, yoy_pct, log_diff_pct), classify under full,
            pre-2020, and post-2020 samples.  This separates the
            Phase 6 VAR decision from the Chow-test decision per
            論点 8 (D-028 rationale).

    Part 3: Final Transformation Registry.  Two input columns
            (phase6_var_input, chow_test_input) per country × indicator
            with justification and caveats.  Ratifies D-027 draft from
            Step 2 and encodes D-031 (Japan CPI: YoY + 2022 regime
            dummy as the Option B-style treatment).

    Part 4: Chow-test break-window pre-check.  For each of the three
            break dates (2008-09, 2020-03, 2022-02) and each country ×
            indicator, test stationarity of the chow_test_input form
            within the pre-break and post-break sub-samples
            separately.  Flag windows with n_obs below a small-sample
            threshold to temper Step 4's F-statistic interpretation.

Design decisions embedded (draft — ratify D-024..D-031 on finalisation)
-----------------------------------------------------------------------
* Sub-period split date 2020-01-01 chosen because the COVID shock
  (2020-03) is the most recent clean regime boundary and yields ~228
  pre-period observations and 63–70 post-period observations per
  country.  Larger than Energy 2022-02 split for power.
* 'Conflict' classification is treated as provisionally usable with a
  caveat if ADF rejects unit root — the KPSS rejection then indicates
  a level shift that will be absorbed by the Chow test's regime dummy.
* For Phase-6 VAR input, when yoy_pct and log_diff_pct both resolve,
  yoy_pct is preferred for narrative alignment (inflation = annual CPI
  growth) and cross-indicator consistency with GDP/M2 YoY.  log_diff_pct
  is recorded as robustness input.
* Japan CPI specific (D-031): when all three CPI transforms remain
  non-stationary on the full sample but resolve under sub-period
  analysis, the recommendation is yoy_pct with an explicit post-2022
  regime dummy in Phase 6 estimation.  Second-differencing is rejected
  because I(2) monthly CPI lacks clean economic interpretation.
* Small-sample threshold: n_obs < 50 triggers a power caveat on
  sub-sample tests; n_obs < 30 triggers a hard warning.

Inputs
------
data/processed/main_{usa,japan,uk,germany}.csv via load_processed_all_main().

Outputs
-------
stdout
    Human-readable sectioned report (Parts 0 through 4).
data/documentation/phase3_conflict_firstdiff_fix.csv
data/documentation/phase3_subperiod_stationarity.csv
data/documentation/phase3_cpi_deep_dive.csv
data/documentation/phase3_transformation_registry_final.csv
data/documentation/phase3_break_window_stationarity.csv

Usage
-----
Run from the project root:

    python scripts/phase3_step3_cpi_decision_and_registry.py

Step 1 and Step 2 do not need to have been run first (Part 0 reproduces
conflict identification; Part 1 reproduces level diagnostics internally).
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

ADF_REGRESSION_LEVEL = {
    'CPI':          'ct',
    'POLICY_RATE':  'c',
    'UNEMPLOYMENT': 'c',
    'GDP':          'c',
    'M2':           'c',
}
REGRESSION_DIFF = 'c'

FLAGGED_CLASSES = {'Non-stationary', 'Inconclusive'}
CONFLICT_CLASS  = 'Trend-stationary (conflict)'
GOOD_CLASSES    = {'Stationary'}

# Sub-period split (論点 / Part 1)
SUBPERIOD_SPLIT = pd.Timestamp('2020-01-01')

# Chow-test break dates (Phase 3 Task 2, ProjectScope §9)
BREAK_DATES = {
    'GFC_2008':     pd.Timestamp('2008-09-01'),
    'COVID_2020':   pd.Timestamp('2020-03-01'),
    'ENERGY_2022':  pd.Timestamp('2022-02-01'),
}

# Small-sample warning thresholds for sub-sample ADF/KPSS power
SMALL_SAMPLE_WARN = 50
SMALL_SAMPLE_HARD = 30


# ──────────────────────────────────────────────────────────────────
# Test helpers (identical to Step 1/2; will move to src/stationarity.py
# in the forthcoming S3.5 extraction)
# ──────────────────────────────────────────────────────────────────
def schwert_maxlag(n_obs: int) -> int:
    return max(1, int(np.floor(12 * (n_obs / 100.0) ** 0.25)))


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
            'adf_crit_5pct': float(crit['5%']),
            'adf_maxlag': int(maxlag),       'adf_error': None,
        }
    except Exception as e:
        return {
            'adf_stat': np.nan, 'adf_pvalue': np.nan,
            'adf_usedlag': -1, 'adf_nobs': -1,
            'adf_crit_5pct': np.nan, 'adf_maxlag': int(maxlag),
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
                'kpss_crit_5pct': float(crit['5%']),
                'kpss_interp_flag': bool(interp), 'kpss_error': None,
            }
        except Exception as e:
            return {
                'kpss_stat': np.nan, 'kpss_pvalue': np.nan, 'kpss_lags': -1,
                'kpss_crit_5pct': np.nan,
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
    n = int(series.dropna().shape[0])
    if n < 10:
        return {'adf_pvalue': np.nan, 'kpss_pvalue': np.nan,
                'classification': 'Error', 'n_obs': n,
                'small_sample_flag': True, 'hard_warn_flag': True,
                'adf_stat': np.nan, 'kpss_stat': np.nan,
                'adf_error': 'n < 10', 'kpss_error': 'n < 10'}
    adf = run_adf(series, regression)
    kps = run_kpss(series, regression)
    cls = classify_4quadrant(adf['adf_pvalue'], kps['kpss_pvalue'])
    return {
        **adf, **kps, 'classification': cls, 'n_obs': n,
        'small_sample_flag': n < SMALL_SAMPLE_WARN,
        'hard_warn_flag': n < SMALL_SAMPLE_HARD,
    }


# ──────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────
def first_difference(series: pd.Series) -> pd.Series:
    return series.diff().dropna()


def second_difference(series: pd.Series) -> pd.Series:
    return series.diff().diff().dropna()


def yoy_pct(series: pd.Series, periods: int = 12) -> pd.Series:
    return (100.0 * (series / series.shift(periods) - 1.0)).dropna()


def log_first_diff_pct(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return (100.0 * np.log(s / s.shift(1))).dropna()


TRANSFORM_FN = {
    'level':         lambda s: s.dropna(),
    'first_diff':    first_difference,
    'second_diff':   second_difference,
    'yoy_pct':       yoy_pct,
    'log_diff_pct':  log_first_diff_pct,
}

TRANSFORM_REG = {
    'level':         None,            # use ADF_REGRESSION_LEVEL[indicator]
    'first_diff':    'c',
    'second_diff':   'c',
    'yoy_pct':       'c',
    'log_diff_pct':  'c',
}


def apply_transform(series: pd.Series, transform: str,
                    indicator: str) -> tuple[pd.Series, str]:
    """Return (transformed_series, regression_spec)."""
    s = TRANSFORM_FN[transform](series)
    reg = TRANSFORM_REG[transform] or ADF_REGRESSION_LEVEL[indicator]
    return s, reg


# ──────────────────────────────────────────────────────────────────
# Pretty-print helpers
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
    out = df.copy()
    for c, fmt in cols_map.items():
        if c in out.columns:
            out[c] = out[c].map(lambda x: fmt.format(x)
                                if pd.notnull(x) and not isinstance(x, str)
                                else x)
    return out


# ──────────────────────────────────────────────────────────────────
# Part 0: Conflict first-diff fix
# ──────────────────────────────────────────────────────────────────
CONFLICT_SERIES = [('USA',     'UNEMPLOYMENT'),
                   ('GERMANY', 'POLICY_RATE')]


def run_part0(datasets: dict) -> pd.DataFrame:
    rows = []
    for country, indicator in CONFLICT_SERIES:
        col = f"{country}_{indicator}"
        series = datasets[country][col]

        # Level (confirms conflict classification)
        lvl = test_series(series, regression=ADF_REGRESSION_LEVEL[indicator])
        rows.append({
            'country': country, 'indicator': indicator, 'column': col,
            'transform': 'level',
            'regression': ADF_REGRESSION_LEVEL[indicator],
            **lvl,
        })

        # 'ct' retest (from Step 2 Part 2)
        ct = test_series(series, regression='ct')
        rows.append({
            'country': country, 'indicator': indicator, 'column': col,
            'transform': 'level_ct', 'regression': 'ct',
            **ct,
        })

        # First-difference (the missing test)
        d1_series = first_difference(series)
        d1 = test_series(d1_series, regression=REGRESSION_DIFF)
        rows.append({
            'country': country, 'indicator': indicator, 'column': col,
            'transform': 'first_diff', 'regression': REGRESSION_DIFF,
            **d1,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 1: Sub-period ADF/KPSS
# ──────────────────────────────────────────────────────────────────
def recommended_phase6_from_step2_pattern(country: str, indicator: str,
                                          datasets: dict) -> str:
    """Mirror the Step 2 registry policy to pick a Phase 6 form per series.

    This is the provisional form used by Part 1 to run sub-period tests;
    Part 3 may override it based on the sub-period findings.
    """
    series = datasets[country][f"{country}_{indicator}"]
    lvl = test_series(series, regression=ADF_REGRESSION_LEVEL[indicator])
    if lvl['classification'] == 'Stationary':
        return 'level'
    if indicator == 'CPI':
        # Prefer yoy_pct provisionally; Part 3 will refine per-country
        return 'yoy_pct'
    # Otherwise first_diff
    return 'first_diff'


def run_part1_subperiod(datasets: dict,
                        split_date: pd.Timestamp = SUBPERIOD_SPLIT) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = datasets[country]
        for indicator in INDICATORS:
            col = f"{country}_{indicator}"
            if col not in df.columns:
                continue
            series = df[col]
            transform = recommended_phase6_from_step2_pattern(
                country, indicator, datasets)
            transformed, reg = apply_transform(series, transform, indicator)

            # Full sample baseline
            full = test_series(transformed, regression=reg)
            rows.append({
                'country': country, 'indicator': indicator,
                'transform': transform, 'regression': reg,
                'subperiod': 'full',
                'start': transformed.index.min().strftime('%Y-%m')
                         if len(transformed) else '',
                'end':   transformed.index.max().strftime('%Y-%m')
                         if len(transformed) else '',
                **full,
            })

            # Pre-2020
            pre = transformed[transformed.index < split_date]
            pre_res = test_series(pre, regression=reg)
            rows.append({
                'country': country, 'indicator': indicator,
                'transform': transform, 'regression': reg,
                'subperiod': 'pre_2020',
                'start': pre.index.min().strftime('%Y-%m') if len(pre) else '',
                'end':   pre.index.max().strftime('%Y-%m') if len(pre) else '',
                **pre_res,
            })

            # Post-2020
            post = transformed[transformed.index >= split_date]
            post_res = test_series(post, regression=reg)
            rows.append({
                'country': country, 'indicator': indicator,
                'transform': transform, 'regression': reg,
                'subperiod': 'post_2020',
                'start': post.index.min().strftime('%Y-%m') if len(post) else '',
                'end':   post.index.max().strftime('%Y-%m') if len(post) else '',
                **post_res,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 2: CPI deep dive
# ──────────────────────────────────────────────────────────────────
CPI_FORMS = ['first_diff', 'yoy_pct', 'log_diff_pct']


def run_part2_cpi_deepdive(datasets: dict,
                           split_date: pd.Timestamp = SUBPERIOD_SPLIT) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        series = datasets[country][f"{country}_CPI"]
        for form in CPI_FORMS:
            transformed, reg = apply_transform(series, form, 'CPI')
            full = test_series(transformed, regression=reg)
            pre  = transformed[transformed.index < split_date]
            post = transformed[transformed.index >= split_date]
            pre_res  = test_series(pre,  regression=reg)
            post_res = test_series(post, regression=reg)

            for sub_name, sub_res, sub in [('full', full, transformed),
                                           ('pre_2020', pre_res, pre),
                                           ('post_2020', post_res, post)]:
                rows.append({
                    'country': country, 'indicator': 'CPI',
                    'transform': form, 'regression': reg,
                    'subperiod': sub_name,
                    'start': sub.index.min().strftime('%Y-%m') if len(sub) else '',
                    'end':   sub.index.max().strftime('%Y-%m') if len(sub) else '',
                    **sub_res,
                })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 3: Final Transformation Registry
# ──────────────────────────────────────────────────────────────────
def decide_cpi_inputs(country: str,
                      cpi_deep: pd.DataFrame) -> tuple[str, str, str, str]:
    """
    Return (phase6_var_input, chow_test_input, decision_code, justification).

    Policy:
      - phase6_var_input needs a stationary form (or flagged with dummy
        per D-031).  Preference order: yoy_pct > log_diff_pct > first_diff.
        Conflict is acceptable if ADF rejects unit root.
      - chow_test_input prefers yoy_pct regardless of stationarity,
        because the break test is specifically looking for the level
        shift that causes the non-stationarity.
    """
    full = cpi_deep[(cpi_deep['country'] == country)
                    & (cpi_deep['subperiod'] == 'full')]
    cls = {r['transform']: r['classification']
           for _, r in full.iterrows()}

    def is_usable(x: str) -> bool:
        return x in GOOD_CLASSES or x == CONFLICT_CLASS

    # Phase 6 VAR input decision
    preferred_order = ['yoy_pct', 'log_diff_pct', 'first_diff']
    phase6 = None
    decision_code = None

    # Sub-period based: check if pre/post each stationary under yoy_pct
    pre  = cpi_deep[(cpi_deep['country'] == country)
                    & (cpi_deep['subperiod'] == 'pre_2020')
                    & (cpi_deep['transform'] == 'yoy_pct')]
    post = cpi_deep[(cpi_deep['country'] == country)
                    & (cpi_deep['subperiod'] == 'post_2020')
                    & (cpi_deep['transform'] == 'yoy_pct')]
    pre_cls  = pre.iloc[0]['classification']  if len(pre)  else 'Error'
    post_cls = post.iloc[0]['classification'] if len(post) else 'Error'

    if cls.get('yoy_pct') == 'Stationary':
        phase6 = 'yoy_pct'
        decision_code = 'CPI_YOY_STATIONARY'
    elif cls.get('log_diff_pct') in GOOD_CLASSES:
        phase6 = 'log_diff_pct'
        decision_code = 'CPI_LOGDIFF_STATIONARY'
    elif (cls.get('yoy_pct') == 'Non-stationary'
          and pre_cls in GOOD_CLASSES
          and post_cls in GOOD_CLASSES):
        phase6 = 'yoy_pct_with_regime_dummy'
        decision_code = 'CPI_YOY_REGIME_SHIFT'  # D-031 trigger
    elif cls.get('first_diff') in GOOD_CLASSES:
        phase6 = 'first_diff'
        decision_code = 'CPI_FIRSTDIFF_FALLBACK'
    elif cls.get('log_diff_pct') == CONFLICT_CLASS:
        phase6 = 'log_diff_pct_with_caveat'
        decision_code = 'CPI_LOGDIFF_CONFLICT'
    elif cls.get('yoy_pct') == CONFLICT_CLASS:
        phase6 = 'yoy_pct_with_caveat'
        decision_code = 'CPI_YOY_CONFLICT'
    else:
        phase6 = 'yoy_pct_with_regime_dummy'
        decision_code = 'CPI_REQUIRES_REGIME_DUMMY'  # D-031 fallback

    # Chow test input: almost always yoy_pct (論点 8 + D-028 rationale)
    chow = 'yoy_pct'

    # Justification text
    just = (
        f"Full-sample CPI classifications: "
        f"first_diff={cls.get('first_diff')}, "
        f"yoy_pct={cls.get('yoy_pct')}, "
        f"log_diff_pct={cls.get('log_diff_pct')}. "
        f"Sub-period yoy_pct: pre-2020={pre_cls}, post-2020={post_cls}. "
        f"Decision code: {decision_code}."
    )
    return phase6, chow, decision_code, just


def build_final_registry(datasets: dict,
                         part0: pd.DataFrame,
                         part1: pd.DataFrame,
                         cpi_deep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        for indicator in INDICATORS:
            col = f"{country}_{indicator}"

            # Level baseline from Part 1 full sample
            full = part1[(part1['country'] == country)
                         & (part1['indicator'] == indicator)
                         & (part1['subperiod'] == 'full')]
            lvl_cls = full.iloc[0]['classification'] if len(full) else 'Error'
            lvl_form_used = full.iloc[0]['transform'] if len(full) else 'level'

            # Also pull the raw level classification (transform='level')
            # so we can distinguish "level-stationary" from "transformed-stationary"
            raw_series = datasets[country][col]
            raw_lvl = test_series(raw_series,
                                  regression=ADF_REGRESSION_LEVEL[indicator])
            raw_lvl_cls = raw_lvl['classification']

            if indicator == 'CPI':
                phase6, chow, code, just = decide_cpi_inputs(country, cpi_deep)
            else:
                chow = None  # will be set equal to phase6 below unless overridden

                if raw_lvl_cls == 'Stationary':
                    phase6 = 'level'
                    code = 'LEVEL_STATIONARY'
                    just = ("Level series is Stationary under joint ADF+KPSS "
                            "at the specified regression spec.")
                elif (country, indicator) in CONFLICT_SERIES:
                    # Use Part 0 first-diff result
                    p0_fd = part0[(part0['country'] == country)
                                  & (part0['indicator'] == indicator)
                                  & (part0['transform'] == 'first_diff')]
                    fd_cls = p0_fd.iloc[0]['classification'] \
                                 if len(p0_fd) else 'Error'
                    p0_ct = part0[(part0['country'] == country)
                                  & (part0['indicator'] == indicator)
                                  & (part0['transform'] == 'level_ct')]
                    ct_cls = p0_ct.iloc[0]['classification'] \
                                 if len(p0_ct) else 'Error'
                    if fd_cls == 'Stationary':
                        phase6 = 'first_diff'
                        code = 'CONFLICT_RESOLVED_BY_FIRSTDIFF'
                        just = (f"Level='c' = Conflict; 'ct' retest = {ct_cls}; "
                                f"first_diff = Stationary. Adopted first_diff "
                                "for cross-country VAR consistency.")
                    elif ct_cls == 'Stationary':
                        phase6 = 'level_with_linear_trend'
                        code = 'CONFLICT_TREND_STATIONARY'
                        just = (f"Level='c' = Conflict; 'ct' = Stationary; "
                                f"first_diff = {fd_cls}. Usable as trend-"
                                "stationary with deterministic trend in VAR.")
                    else:
                        phase6 = 'first_diff_with_caveat'
                        code = 'CONFLICT_UNRESOLVED'
                        just = (f"Level='c' = Conflict; 'ct' = {ct_cls}; "
                                f"first_diff = {fd_cls}. First_diff adopted "
                                "with robustness caveat; inspect Step 4 results.")
                else:
                    # Flagged at level; first-diff should resolve per Step 2
                    # Look it up from Part 1 (which ran on 'first_diff' for
                    # these indicators by the provisional policy).
                    fd_cls = lvl_cls if lvl_form_used == 'first_diff' \
                                     else 'Unknown'
                    if fd_cls == 'Stationary':
                        phase6 = 'first_diff'
                        code = 'I1_STANDARD'
                        just = (f"Level = {raw_lvl_cls}; first_diff = "
                                "Stationary. Standard I(1) treatment.")
                    elif fd_cls == 'Trend-stationary (conflict)':
                        phase6 = 'first_diff_with_caveat'
                        code = 'I1_WITH_CONFLICT'
                        just = (f"Level = {raw_lvl_cls}; first_diff yields "
                                "Conflict (ADF rejects unit root but KPSS "
                                "rejects stationarity — level shift). "
                                "Adopted with caveat; dummy-augment in Phase 6.")
                    else:
                        phase6 = 'first_diff_with_caveat'
                        code = 'I1_UNRESOLVED'
                        just = (f"Level = {raw_lvl_cls}; first_diff = "
                                f"{fd_cls}. Adopted first_diff with caveat; "
                                "inspect Step 4 and Phase 6 diagnostics.")

                # Non-CPI chow_test_input mirrors phase6 unless flagged
                chow = phase6

            # Sub-period diagnostic flags
            pre = part1[(part1['country'] == country)
                        & (part1['indicator'] == indicator)
                        & (part1['subperiod'] == 'pre_2020')]
            post = part1[(part1['country'] == country)
                         & (part1['indicator'] == indicator)
                         & (part1['subperiod'] == 'post_2020')]
            pre_cls  = pre.iloc[0]['classification']  if len(pre)  else 'Error'
            post_cls = post.iloc[0]['classification'] if len(post) else 'Error'
            regime_flag = ((pre_cls == 'Stationary'
                            and post_cls == 'Stationary'
                            and lvl_cls != 'Stationary')
                           or (code in ('CPI_YOY_REGIME_SHIFT',
                                        'CPI_REQUIRES_REGIME_DUMMY')))

            rows.append({
                'country': country,
                'indicator': indicator,
                'raw_level_classification': raw_lvl_cls,
                'phase6_classification': lvl_cls,
                'phase6_var_input': phase6,
                'chow_test_input': chow,
                'decision_code': code,
                'pre_2020_classification':  pre_cls,
                'post_2020_classification': post_cls,
                'regime_shift_flag': bool(regime_flag),
                'justification': just,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Part 4: Break-window pre-check
# ──────────────────────────────────────────────────────────────────
def run_part4_break_windows(datasets: dict,
                            registry: pd.DataFrame,
                            break_dates: dict = BREAK_DATES) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = datasets[country]
        for indicator in INDICATORS:
            col = f"{country}_{indicator}"
            series = df[col]

            reg_row = registry[(registry['country'] == country)
                               & (registry['indicator'] == indicator)].iloc[0]
            chow_form = reg_row['chow_test_input']
            # Normalise 'with_caveat' / 'with_regime_dummy' to the base form
            # for the actual transform lookup
            base_form = chow_form
            for suffix in ('_with_caveat', '_with_regime_dummy'):
                if base_form.endswith(suffix):
                    base_form = base_form[: -len(suffix)]
                    break
            if base_form == 'level_with_linear_trend':
                base_form = 'level'
            transformed, reg = apply_transform(series, base_form, indicator)

            for break_name, break_date in break_dates.items():
                pre  = transformed[transformed.index <  break_date]
                post = transformed[transformed.index >= break_date]
                for win_name, win in [('pre_break',  pre),
                                      ('post_break', post)]:
                    res = test_series(win, regression=reg)
                    rows.append({
                        'country': country, 'indicator': indicator,
                        'chow_test_input': chow_form,
                        'base_form_tested': base_form,
                        'break_name': break_name,
                        'break_date': break_date.strftime('%Y-%m'),
                        'window': win_name,
                        'start': win.index.min().strftime('%Y-%m') if len(win) else '',
                        'end':   win.index.max().strftime('%Y-%m') if len(win) else '',
                        **res,
                    })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 79)
    print("Phase 3 · Step 3 — CPI decision, sub-period diagnostics,")
    print("                   final Registry, Chow-test window pre-check")
    print(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Project   : {PROJECT_ROOT}")
    print(f"Alpha     : {ALPHA}")
    print(f"Subperiod split : {SUBPERIOD_SPLIT:%Y-%m}")
    print(f"Break dates     : "
          + ", ".join(f"{k}={v:%Y-%m}" for k, v in BREAK_DATES.items()))
    print("=" * 79)

    datasets = load_processed_all_main()
    print(f"\nLoaded {len(datasets)} main-country datasets:")
    for c, df in datasets.items():
        print(f"  {c:<8s} : {df.shape[0]} rows x {df.shape[1]} cols  "
              f"({df.index.min():%Y-%m} -> {df.index.max():%Y-%m})")

    # ── Part 0 ────────────────────────────────────────────────
    section("PART 0 — Conflict first-diff fix (S2 bug補填)")
    part0 = run_part0(datasets)
    p0 = part0[['country', 'indicator', 'transform', 'regression', 'n_obs',
                'adf_pvalue', 'kpss_pvalue', 'classification']].copy()
    p0 = fmt_numeric_cols(p0, {
        'adf_pvalue': '{:.4f}', 'kpss_pvalue': '{:.4f}',
    })
    print(p0.to_string(index=False))

    subsection("Conflict resolution verdict")
    for (c, ind) in CONFLICT_SERIES:
        row = part0[(part0['country'] == c) & (part0['indicator'] == ind)
                    & (part0['transform'] == 'first_diff')].iloc[0]
        fd_cls = row['classification']
        verdict = ("RESOLVED" if fd_cls == 'Stationary'
                   else "PARTIAL"    if fd_cls == CONFLICT_CLASS
                   else "UNRESOLVED")
        print(f"  {c:<8s} {ind:<12s}  first_diff -> {fd_cls:<32s} [{verdict}]")

    # ── Part 1 ────────────────────────────────────────────────
    section(f"PART 1 — Sub-period ADF+KPSS on recommended phase6 form "
            f"(split at {SUBPERIOD_SPLIT:%Y-%m})")
    part1 = run_part1_subperiod(datasets)

    # Pivot for readability: rows=country+indicator, cols=subperiod, val=cls
    pivot = (part1
             .pivot_table(index=['country', 'indicator', 'transform'],
                          columns='subperiod', values='classification',
                          aggfunc='first'))
    pivot = pivot[['full', 'pre_2020', 'post_2020']]
    subsection("Classification matrix (rows: country × indicator × transform)")
    print(pivot.to_string())

    # Flag series with regime-shift signature
    regime_candidates = []
    for (c, ind, tr), row in pivot.iterrows():
        if (row['full'] != 'Stationary'
            and row['pre_2020']  == 'Stationary'
            and row['post_2020'] == 'Stationary'):
            regime_candidates.append((c, ind, tr))
    subsection(f"Regime-shift candidates "
               f"(full Non-stat but both sub-periods Stat): "
               f"{len(regime_candidates)}")
    for c, ind, tr in regime_candidates:
        print(f"  {c:<8s} {ind:<12s} ({tr})")

    # ── Part 2 ────────────────────────────────────────────────
    section("PART 2 — CPI deep dive (3 forms × full / pre-2020 / post-2020)")
    cpi_deep = run_part2_cpi_deepdive(datasets)
    cpi_pivot = (cpi_deep
                 .pivot_table(index=['country', 'transform'],
                              columns='subperiod', values='classification',
                              aggfunc='first'))
    cpi_pivot = cpi_pivot[['full', 'pre_2020', 'post_2020']]
    print(cpi_pivot.to_string())

    subsection("Per-country CPI input decision (phase6 / chow)")
    for country in MAIN_COUNTRIES:
        phase6, chow, code, _ = decide_cpi_inputs(country, cpi_deep)
        print(f"  {country:<8s} phase6={phase6:<34s} chow={chow:<12s} "
              f"[{code}]")

    # ── Part 3 ────────────────────────────────────────────────
    section("PART 3 — Final Transformation Registry (ratifies D-027; D-031 where flagged)")
    registry = build_final_registry(datasets, part0, part1, cpi_deep)

    subsection("Registry (compact view)")
    r_compact = registry[['country', 'indicator',
                          'raw_level_classification',
                          'phase6_var_input',
                          'chow_test_input',
                          'regime_shift_flag',
                          'decision_code']].copy()
    print(r_compact.to_string(index=False))

    subsection("Registry — recommended-transform counts (phase6_var_input)")
    for k, v in registry['phase6_var_input'].value_counts().items():
        print(f"  {k:<35s} {v:>3d}")

    subsection("Registry — decision codes")
    for k, v in registry['decision_code'].value_counts().items():
        print(f"  {k:<35s} {v:>3d}")

    subsection("Justifications")
    for _, r in registry.iterrows():
        j = (r['justification'] or '')[:120]
        print(f"  {r['country']:<8s} {r['indicator']:<12s} "
              f"[{r['phase6_var_input']:<28s}]  {j}")

    # ── Part 4 ────────────────────────────────────────────────
    section("PART 4 — Chow-test break-window stationarity pre-check")
    part4 = run_part4_break_windows(datasets, registry)

    # Pivot: rows=country+indicator, cols=break+window, val=cls
    part4['col_key'] = (part4['break_name'] + '|' + part4['window'])
    piv4 = (part4
            .pivot_table(index=['country', 'indicator'],
                         columns='col_key', values='classification',
                         aggfunc='first'))
    col_order = []
    for bn in BREAK_DATES:
        for wn in ['pre_break', 'post_break']:
            col_order.append(f"{bn}|{wn}")
    piv4 = piv4.reindex(columns=[c for c in col_order if c in piv4.columns])
    subsection("Classification matrix (cols: break|window)")
    print(piv4.to_string())

    # n_obs pivot (for small-sample flags)
    piv4n = (part4
             .pivot_table(index=['country', 'indicator'],
                          columns='col_key', values='n_obs',
                          aggfunc='first'))
    piv4n = piv4n.reindex(columns=[c for c in col_order if c in piv4n.columns])
    subsection("n_obs matrix (small-sample diagnostic)")
    print(piv4n.to_string())

    small = part4[part4['small_sample_flag']]
    hard  = part4[part4['hard_warn_flag']]
    subsection(f"Small-sample warnings: "
               f"{len(small)} windows with n < {SMALL_SAMPLE_WARN}, "
               f"of which {len(hard)} with n < {SMALL_SAMPLE_HARD}")
    if len(small):
        for _, r in small.iterrows():
            tag = ' HARD' if r['hard_warn_flag'] else ''
            print(f"  {r['country']:<8s} {r['indicator']:<12s} "
                  f"{r['break_name']:<13s} {r['window']:<10s} "
                  f"n={r['n_obs']:>3d}{tag}")

    # Classify per break: how many of 5 indicators are stationary in
    # BOTH pre and post windows (a "Chow-ready" metric)?
    subsection("Chow-readiness per (country × break)")
    chow_ready = []
    for country in MAIN_COUNTRIES:
        for break_name in BREAK_DATES:
            block = part4[(part4['country'] == country)
                          & (part4['break_name'] == break_name)]
            pairs = block.groupby('indicator')['classification'].apply(list)
            ready_count = 0
            for ind, cls_list in pairs.items():
                # Require both windows to be in {Stationary, Conflict}
                ok = all(c in GOOD_CLASSES | {CONFLICT_CLASS} for c in cls_list)
                if ok:
                    ready_count += 1
            chow_ready.append((country, break_name, ready_count))
            print(f"  {country:<8s} {break_name:<13s} "
                  f"{ready_count}/5 indicators Chow-ready "
                  f"(stationary or conflict in both windows)")

    # ── Write audit CSVs ──────────────────────────────────────
    section("CSV outputs")
    doc_dir = PROJECT_ROOT / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ('phase3_conflict_firstdiff_fix.csv',         part0),
        ('phase3_subperiod_stationarity.csv',         part1),
        ('phase3_cpi_deep_dive.csv',                  cpi_deep),
        ('phase3_transformation_registry_final.csv',  registry),
        ('phase3_break_window_stationarity.csv',      part4),
    ]
    for name, df in outputs:
        path = doc_dir / name
        df.to_csv(path, index=False)
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        print(f"  wrote {rel:<56s}  ({len(df)} rows)")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
