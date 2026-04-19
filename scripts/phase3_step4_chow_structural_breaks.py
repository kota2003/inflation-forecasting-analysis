"""
scripts/phase3_step4_chow_structural_breaks.py
===============================================
Phase 3 · Step 4 — Chow Structural Break Tests.

(BUG-FIX revision: D_split and D_covid are now constructed as
pd.Series aligned on X_full.index so that DataFrame operations
.rename(), .to_frame(), and .multiply() work correctly.  The prior
version built D_covid as a bare numpy ndarray which broke Part 3.)

Purpose
-------
Execute the Chow-test battery for Phase 3 Task 2 on the finalised
Transformation Registry from Step 3 (with three revisions applied in
this script per 论点 8 / 论点 9 / D-031 revision).

Six sequential parts:

    Part 0: Load the Step 3 registry, apply D-031-revised overrides
            (Japan CPI → first_diff, Germany CPI → first_diff, UK CPI
            → log_diff_pct), build the per-country Chow dataset, and
            print the analytical plan.

    Part 1: Classical Chow F-test (homoskedasticity + no autocorr
            assumed) at each of the three break dates, per country.
            12 tests total (4 countries × 3 break dates).

    Part 2: HAC-robust Wald-form Chow via dummy-interaction OLS.
            Newey-West standard errors with lag = 4 absorb residual
            autocorrelation and conditional heteroskedasticity.
            Same 12 tests.

    Part 3: COVID-dummy-augmented Chow (HAC).  An additive dummy for
            2020-03 to 2020-09 absorbs the COVID outlier shock
            without interfering with the break-detection degrees of
            freedom.  Applied to GFC_2008 and ENERGY_2022 only
            (COVID_2020 overlaps with the dummy and is skipped).
            8 tests (4 countries × 2 break dates).

    Part 4: Per-coefficient break decomposition.  For each (country,
            break), fit separate HAC-OLS regressions on the pre- and
            post-windows and report each regressor's coefficient,
            standard error, Δ, and z-statistic for the difference.
            Pinpoints which of the five economic channels drive any
            detected break.  Informs Phase 6 regime-dummy specification.

    Part 5: Bonferroni-adjusted significance summary.  With m = 12
            family-wise tests, α_Bonf = 0.05 / 12 ≈ 0.00417.  Reports
            each test's rejection status at nominal α=0.05 and at
            α_Bonf.  Cross-tab by (country × break) and (variant × break).

    Part 6: Small-sample caveat report.  ENERGY_2022 post-window is
            n = 38–45 per country; all tests at that date are flagged
            with an explicit power limitation note.

Revised Transformation Registry (applied in this script)
-------------------------------------------------------
    (JAPAN,   CPI): chow_test_input = 'first_diff'   [D-031 revised]
    (GERMANY, CPI): chow_test_input = 'first_diff'   [D-031 revised]
    (UK,      CPI): chow_test_input = 'log_diff_pct' [论点 8 rigor]

Rationale: Chow test F/Wald inference relies on the test statistic's
asymptotic distribution, which requires the dependent variable to be
stationary within each sub-sample.  The S3 deep-dive showed CPI YoY
is non-stationary for JAPAN, UK, and GERMANY full-sample, so for
rigour we run Chow tests on the phase6-input (MoM-type) form.  The
YoY form is retained for Phase 5 EDA narrative plots.

Usage
-----
Run from the project root:

    python scripts/phase3_step4_chow_structural_breaks.py
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
from scipy import stats                                             # noqa: E402
import statsmodels.api as sm                                        # noqa: E402

from src.data_loader import (                                       # noqa: E402
    load_processed_all_main,
    INDICATORS,
    MAIN_COUNTRIES,
)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
ALPHA_NOMINAL = 0.05

BREAK_DATES = {
    'GFC_2008':    pd.Timestamp('2008-09-01'),
    'COVID_2020':  pd.Timestamp('2020-03-01'),
    'ENERGY_2022': pd.Timestamp('2022-02-01'),
}

COVID_DUMMY_START = pd.Timestamp('2020-03-01')
COVID_DUMMY_END   = pd.Timestamp('2020-09-30')

COVID_DUMMY_BREAKS = ['GFC_2008', 'ENERGY_2022']

HAC_LAG = 4

SMALL_SAMPLE_WARN = 50
SMALL_SAMPLE_HARD = 30

REGISTRY_OVERRIDES = {
    ('JAPAN',   'CPI'): {
        'chow_test_input':  'first_diff',
        'decision_code':    'CPI_I1_ACCEPTED_D031_REVISED',
        'justification':    ("S3 Part 2 showed Japan CPI non-stationary across "
                             "all three transforms and all three sub-periods. "
                             "Revised D-031: accept I(1); Chow test uses "
                             "first_diff (MoM inflation). YoY retained for "
                             "Phase 5 EDA narrative only."),
    },
    ('GERMANY', 'CPI'): {
        'chow_test_input':  'first_diff',
        'decision_code':    'CPI_I1_FROM_REGIME_SHIFT',
        'justification':    ("S3 Part 2 showed Germany CPI first_diff has a "
                             "clean regime-shift pattern (pre-2020 Stationary, "
                             "post-2020 Stationary, full Non-stationary). "
                             "Using first_diff for Chow test is rigorous and "
                             "lets the Chow F capture the level shift directly."),
    },
    ('UK',      'CPI'): {
        'chow_test_input':  'log_diff_pct',
        'decision_code':    'CPI_LOGDIFF_PHASE6_MATCH',
        'justification':    ("S3 Part 2 showed UK CPI no fully-stationary form "
                             "on full sample. log_diff_pct selected as phase6 "
                             "input; Chow test now matches for inferential "
                             "consistency. YoY retained for narrative plots."),
    },
}

Y_INDICATOR = 'CPI'
X_INDICATORS = ['POLICY_RATE', 'UNEMPLOYMENT', 'GDP', 'M2']


# ──────────────────────────────────────────────────────────────────
# Transform helpers
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


def strip_suffix(form: str) -> str:
    for suffix in ('_with_regime_dummy', '_with_caveat'):
        if form.endswith(suffix):
            return form[: -len(suffix)]
    if form == 'level_with_linear_trend':
        return 'level'
    return form


# ──────────────────────────────────────────────────────────────────
# Registry loading with overrides
# ──────────────────────────────────────────────────────────────────
def load_revised_registry() -> pd.DataFrame:
    path = (PROJECT_ROOT / 'data' / 'documentation'
            / 'phase3_transformation_registry_final.csv')
    if not path.exists():
        raise FileNotFoundError(
            f"Expected registry at {path}. Run phase3_step3_* first."
        )
    reg = pd.read_csv(path)
    for (country, indicator), overrides in REGISTRY_OVERRIDES.items():
        mask = (reg['country'] == country) & (reg['indicator'] == indicator)
        if not mask.any():
            raise KeyError(f"Registry missing ({country},{indicator})")
        for k, v in overrides.items():
            reg.loc[mask, k] = v
    return reg


# ──────────────────────────────────────────────────────────────────
# Chow dataset construction
# ──────────────────────────────────────────────────────────────────
def build_chow_dataset(country: str,
                       datasets: dict,
                       registry: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_raw = datasets[country]
    cols = {}
    forms_used = {}
    for indicator in INDICATORS:
        reg_row = registry[(registry['country'] == country)
                           & (registry['indicator'] == indicator)].iloc[0]
        form = strip_suffix(reg_row['chow_test_input'])
        forms_used[indicator] = form
        series = df_raw[f"{country}_{indicator}"]
        transformed = TRANSFORM_FN[form](series)
        cols[indicator] = transformed
    df = pd.concat(cols, axis=1).dropna()
    df.index.name = 'date'
    return df, forms_used


# ──────────────────────────────────────────────────────────────────
# Dummy construction helpers  (FIX: return pd.Series, not ndarray)
# ──────────────────────────────────────────────────────────────────
def make_split_dummy(index: pd.DatetimeIndex,
                     break_date: pd.Timestamp) -> pd.Series:
    """D_t = 1 if t >= break_date, else 0, as a pd.Series on `index`."""
    return pd.Series(
        (index >= break_date).astype(float),
        index=index, name='D_split',
    )


def make_covid_dummy(index: pd.DatetimeIndex,
                     covid_start: pd.Timestamp = COVID_DUMMY_START,
                     covid_end: pd.Timestamp = COVID_DUMMY_END) -> pd.Series:
    """COVID_t = 1 if covid_start <= t <= covid_end, else 0, on `index`."""
    return pd.Series(
        ((index >= covid_start) & (index <= covid_end)).astype(float),
        index=index, name='COVID',
    )


# ──────────────────────────────────────────────────────────────────
# Chow test implementations
# ──────────────────────────────────────────────────────────────────
def chow_test_classical(y: pd.Series, X: pd.DataFrame,
                        break_date: pd.Timestamp) -> dict:
    """Classical Chow F-test on structural break (iid errors assumed)."""
    X_full = sm.add_constant(X, has_constant='add')
    pre_mask  = X_full.index <  break_date
    post_mask = X_full.index >= break_date

    y_pre,  y_post  = y[pre_mask],  y[post_mask]
    X_pre,  X_post  = X_full[pre_mask], X_full[post_mask]
    n, k = len(y), X_full.shape[1]

    if len(y_pre) <= k or len(y_post) <= k:
        return {'F': np.nan, 'p_value': np.nan,
                'df_num': k, 'df_denom': max(0, n - 2 * k),
                'RSS_restricted': np.nan, 'RSS_unrestricted': np.nan,
                'n_total': n, 'n_pre': len(y_pre), 'n_post': len(y_post),
                'error': f"Sub-sample too small (pre={len(y_pre)}, "
                         f"post={len(y_post)}; requires > {k})"}

    m_r    = sm.OLS(y, X_full).fit()
    m_pre  = sm.OLS(y_pre,  X_pre).fit()
    m_post = sm.OLS(y_post, X_post).fit()

    RSS_r  = float(m_r.ssr)
    RSS_ur = float(m_pre.ssr + m_post.ssr)
    df_num = k
    df_denom = n - 2 * k
    F = ((RSS_r - RSS_ur) / df_num) / (RSS_ur / df_denom)
    p = float(1.0 - stats.f.cdf(F, df_num, df_denom))
    return {'F': float(F), 'p_value': p,
            'df_num': df_num, 'df_denom': df_denom,
            'RSS_restricted': RSS_r, 'RSS_unrestricted': RSS_ur,
            'n_total': n, 'n_pre': len(y_pre), 'n_post': len(y_post),
            'error': None}


def chow_test_hac(y: pd.Series, X: pd.DataFrame,
                  break_date: pd.Timestamp,
                  hac_lag: int = HAC_LAG) -> dict:
    """HAC-robust Chow test via dummy-interaction Wald."""
    X_full = sm.add_constant(X, has_constant='add')
    D = make_split_dummy(X_full.index, break_date)
    pre_n  = int((1 - D).sum())
    post_n = int(D.sum())

    k = X_full.shape[1]
    if pre_n <= k or post_n <= k:
        return {'F': np.nan, 'p_value': np.nan,
                'df_num': k, 'df_denom': max(0, len(y) - 2 * k),
                'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
                'error': f"Sub-sample too small (pre={pre_n}, post={post_n})"}

    DX = X_full.multiply(D, axis=0)
    DX.columns = [f"D_{c}" for c in X_full.columns]
    X_big = pd.concat([X_full, DX], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X_big).fit(cov_type='HAC',
                                     cov_kwds={'maxlags': hac_lag})

    p_full = X_big.shape[1]
    R = np.zeros((k, p_full))
    R[:, k:2 * k] = np.eye(k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wald = model.wald_test(R, use_f=True)
    F = float(np.asarray(wald.statistic).ravel()[0])
    p = float(np.asarray(wald.pvalue).ravel()[0])
    df_num = int(np.asarray(wald.df_num).ravel()[0]) \
             if hasattr(wald, 'df_num') else k
    df_denom = int(len(y) - p_full)

    return {'F': F, 'p_value': p,
            'df_num': df_num, 'df_denom': df_denom,
            'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
            'error': None}


def chow_test_covid_dummy(y: pd.Series, X: pd.DataFrame,
                          break_date: pd.Timestamp,
                          covid_start: pd.Timestamp = COVID_DUMMY_START,
                          covid_end:   pd.Timestamp = COVID_DUMMY_END,
                          hac_lag: int = HAC_LAG) -> dict:
    """HAC-robust Chow test with additive COVID dummy absorbing 2020 outlier."""
    X_full = sm.add_constant(X, has_constant='add')
    D = make_split_dummy(X_full.index, break_date)
    D_covid = make_covid_dummy(X_full.index, covid_start, covid_end)
    pre_n  = int((1 - D).sum())
    post_n = int(D.sum())
    k = X_full.shape[1]

    if pre_n <= k or post_n <= k:
        return {'F': np.nan, 'p_value': np.nan,
                'df_num': k, 'df_denom': max(0, len(y) - 2 * k - 1),
                'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
                'covid_n': int(D_covid.sum()),
                'error': f"Sub-sample too small (pre={pre_n}, post={post_n})"}

    DX = X_full.multiply(D, axis=0)
    DX.columns = [f"D_{c}" for c in X_full.columns]
    X_big = pd.concat([X_full, DX, D_covid.to_frame()], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X_big).fit(cov_type='HAC',
                                     cov_kwds={'maxlags': hac_lag})

    p_full = X_big.shape[1]
    R = np.zeros((k, p_full))
    R[:, k:2 * k] = np.eye(k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wald = model.wald_test(R, use_f=True)
    F = float(np.asarray(wald.statistic).ravel()[0])
    p = float(np.asarray(wald.pvalue).ravel()[0])
    df_num = int(np.asarray(wald.df_num).ravel()[0]) \
             if hasattr(wald, 'df_num') else k
    df_denom = int(len(y) - p_full)
    covid_coef = float(model.params.get('COVID', np.nan))
    covid_se   = float(model.bse.get('COVID',   np.nan))

    return {'F': F, 'p_value': p,
            'df_num': df_num, 'df_denom': df_denom,
            'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
            'covid_n': int(D_covid.sum()),
            'covid_coef': covid_coef, 'covid_se': covid_se,
            'error': None}


def coefficient_decomposition(y: pd.Series, X: pd.DataFrame,
                              break_date: pd.Timestamp,
                              hac_lag: int = HAC_LAG) -> list[dict]:
    """Pre vs post OLS coefficients per regressor, with HAC standard errors."""
    X_full = sm.add_constant(X, has_constant='add')
    pre_mask  = X_full.index <  break_date
    post_mask = X_full.index >= break_date
    y_pre,  y_post  = y[pre_mask],  y[post_mask]
    X_pre,  X_post  = X_full[pre_mask], X_full[post_mask]
    k = X_full.shape[1]

    if len(y_pre) <= k or len(y_post) <= k:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_pre  = sm.OLS(y_pre,  X_pre).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lag})
        m_post = sm.OLS(y_post, X_post).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lag})

    rows = []
    for var in X_full.columns:
        b_pre   = float(m_pre.params[var])
        se_pre  = float(m_pre.bse[var])
        b_post  = float(m_post.params[var])
        se_post = float(m_post.bse[var])
        delta    = b_post - b_pre
        se_delta = float(np.sqrt(se_pre ** 2 + se_post ** 2))
        z        = float(delta / se_delta) if se_delta > 0 else np.nan
        p_z      = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) \
                      if np.isfinite(z) else np.nan
        rows.append({
            'variable': var,
            'coef_pre':   b_pre,   'se_pre':   se_pre,
            'coef_post':  b_post,  'se_post':  se_post,
            'delta':      delta,   'se_delta': se_delta,
            'z_stat':     z,       'p_value':  p_z,
            'n_pre':      len(y_pre),
            'n_post':     len(y_post),
        })
    return rows


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


def fmt(df: pd.DataFrame, cols_map: dict) -> pd.DataFrame:
    out = df.copy()
    for c, f in cols_map.items():
        if c in out.columns:
            out[c] = out[c].map(
                lambda x: f.format(x)
                if pd.notnull(x) and not isinstance(x, str)
                else x
            )
    return out


def sig_marker(p: float, alpha: float) -> str:
    if p is None or pd.isna(p):
        return '    '
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '** '
    if p < alpha:
        return '*  '
    return '   '


# ──────────────────────────────────────────────────────────────────
# Part runners
# ──────────────────────────────────────────────────────────────────
def run_part0_summary(datasets: dict, registry: pd.DataFrame) -> dict:
    chow_datasets = {}
    for country in MAIN_COUNTRIES:
        df, forms = build_chow_dataset(country, datasets, registry)
        chow_datasets[country] = {'df': df, 'forms': forms}
    return chow_datasets


def run_part1_classical(chow_datasets: dict) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        forms = chow_datasets[country]['forms']
        y = df[Y_INDICATOR]
        X = df[X_INDICATORS]
        for break_name, break_date in BREAK_DATES.items():
            res = chow_test_classical(y, X, break_date)
            rows.append({
                'country': country,
                'break_name': break_name,
                'break_date': break_date.strftime('%Y-%m'),
                'y_form':       forms[Y_INDICATOR],
                'X_forms':      ','.join(f"{k}={v}" for k, v in forms.items()
                                         if k in X_INDICATORS),
                **res,
            })
    return pd.DataFrame(rows)


def run_part2_hac(chow_datasets: dict) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        forms = chow_datasets[country]['forms']
        y = df[Y_INDICATOR]
        X = df[X_INDICATORS]
        for break_name, break_date in BREAK_DATES.items():
            res = chow_test_hac(y, X, break_date)
            rows.append({
                'country': country,
                'break_name': break_name,
                'break_date': break_date.strftime('%Y-%m'),
                'hac_lag':     HAC_LAG,
                'y_form':       forms[Y_INDICATOR],
                **res,
            })
    return pd.DataFrame(rows)


def run_part3_covid_dummy(chow_datasets: dict) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        forms = chow_datasets[country]['forms']
        y = df[Y_INDICATOR]
        X = df[X_INDICATORS]
        for break_name in COVID_DUMMY_BREAKS:
            break_date = BREAK_DATES[break_name]
            res = chow_test_covid_dummy(y, X, break_date)
            rows.append({
                'country': country,
                'break_name': break_name,
                'break_date': break_date.strftime('%Y-%m'),
                'covid_dummy_start': COVID_DUMMY_START.strftime('%Y-%m'),
                'covid_dummy_end':   COVID_DUMMY_END.strftime('%Y-%m'),
                'hac_lag':           HAC_LAG,
                'y_form':            forms[Y_INDICATOR],
                **res,
            })
    return pd.DataFrame(rows)


def run_part4_decomposition(chow_datasets: dict) -> pd.DataFrame:
    rows = []
    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        forms = chow_datasets[country]['forms']
        y = df[Y_INDICATOR]
        X = df[X_INDICATORS]
        for break_name, break_date in BREAK_DATES.items():
            coeffs = coefficient_decomposition(y, X, break_date)
            for c in coeffs:
                rows.append({
                    'country': country,
                    'break_name': break_name,
                    'break_date': break_date.strftime('%Y-%m'),
                    'y_form':      forms[Y_INDICATOR],
                    **c,
                })
    return pd.DataFrame(rows)


def run_part5_bonferroni(classical: pd.DataFrame,
                         hac: pd.DataFrame,
                         covid: pd.DataFrame,
                         alpha: float = ALPHA_NOMINAL) -> pd.DataFrame:
    rows = []
    all_tables = [
        ('classical',   classical),
        ('hac',         hac),
        ('covid_dummy', covid),
    ]
    m_family = 12
    alpha_bonf = alpha / m_family

    for variant, tbl in all_tables:
        for _, r in tbl.iterrows():
            p = r['p_value']
            rows.append({
                'variant':      variant,
                'country':      r['country'],
                'break_name':   r['break_name'],
                'break_date':   r['break_date'],
                'F':            r.get('F'),
                'p_value':      p,
                'reject_nom':   bool(p < alpha) if pd.notnull(p) else False,
                'alpha_bonf':   alpha_bonf,
                'reject_bonf':  bool(p < alpha_bonf) if pd.notnull(p) else False,
                'n_total':      r.get('n_total'),
                'n_pre':        r.get('n_pre'),
                'n_post':       r.get('n_post'),
                'error':        r.get('error'),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 79)
    print("Phase 3 · Step 4 — Chow Structural Break Tests  (bug-fix rev.)")
    print(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Project   : {PROJECT_ROOT}")
    print(f"Alpha nom.: {ALPHA_NOMINAL}")
    print(f"HAC lag   : {HAC_LAG}")
    print(f"Break dates: "
          + ", ".join(f"{k}={v:%Y-%m}" for k, v in BREAK_DATES.items()))
    print(f"COVID dummy window: "
          f"{COVID_DUMMY_START:%Y-%m} to {COVID_DUMMY_END:%Y-%m}")
    print("=" * 79)

    datasets = load_processed_all_main()
    registry = load_revised_registry()

    # ── Part 0 ────────────────────────────────────────────────
    section("PART 0 — Setup: datasets, registry overrides, Chow datasets")
    for c, df in datasets.items():
        print(f"  {c:<8s} : {df.shape[0]} rows x {df.shape[1]} cols  "
              f"({df.index.min():%Y-%m} -> {df.index.max():%Y-%m})")

    subsection("Registry overrides applied (S4 revision)")
    for (country, indicator), ov in REGISTRY_OVERRIDES.items():
        print(f"  ({country:<8s}, {indicator:<4s})  "
              f"chow_test_input -> {ov['chow_test_input']}  "
              f"[{ov['decision_code']}]")

    chow_datasets = run_part0_summary(datasets, registry)
    subsection("Per-country Chow dataset (y=CPI, X=PR/UE/GDP/M2)")
    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        forms = chow_datasets[country]['forms']
        forms_s = ' | '.join(f"{k}={forms[k]}" for k in INDICATORS)
        print(f"  {country:<8s} n={df.shape[0]:>3}  "
              f"window={df.index.min():%Y-%m}..{df.index.max():%Y-%m}  "
              f"[{forms_s}]")

    # ── Part 1 ────────────────────────────────────────────────
    section("PART 1 — Classical Chow F-test (homoskedasticity assumed)")
    classical = run_part1_classical(chow_datasets)
    p1 = classical[['country', 'break_name', 'break_date',
                    'n_total', 'n_pre', 'n_post',
                    'df_num', 'df_denom',
                    'F', 'p_value']].copy()
    p1 = fmt(p1, {'F': '{:>8.3f}', 'p_value': '{:.4f}'})
    p1['sig'] = classical['p_value'].apply(
        lambda p: sig_marker(p, ALPHA_NOMINAL))
    print(p1.to_string(index=False))

    subsection("Part 1 summary: rejections at α=0.05")
    p1_sig = classical[classical['p_value'] < ALPHA_NOMINAL]
    print(f"  {len(p1_sig)} of {len(classical)} tests reject H0 "
          f"at nominal α={ALPHA_NOMINAL}")
    for _, r in p1_sig.iterrows():
        print(f"   + {r['country']:<8s} {r['break_name']:<13s}  "
              f"F={r['F']:.3f}  p={r['p_value']:.4g}")

    # ── Part 2 ────────────────────────────────────────────────
    section(f"PART 2 — HAC-robust Chow Wald test (Newey-West, lag={HAC_LAG})")
    hac = run_part2_hac(chow_datasets)
    p2 = hac[['country', 'break_name', 'break_date',
              'n_total', 'n_pre', 'n_post',
              'df_num', 'df_denom',
              'F', 'p_value']].copy()
    p2 = fmt(p2, {'F': '{:>8.3f}', 'p_value': '{:.4f}'})
    p2['sig'] = hac['p_value'].apply(lambda p: sig_marker(p, ALPHA_NOMINAL))
    print(p2.to_string(index=False))

    subsection("Part 2 summary: rejections at α=0.05 (HAC)")
    p2_sig = hac[hac['p_value'] < ALPHA_NOMINAL]
    print(f"  {len(p2_sig)} of {len(hac)} HAC tests reject H0 "
          f"at nominal α={ALPHA_NOMINAL}")
    for _, r in p2_sig.iterrows():
        print(f"   + {r['country']:<8s} {r['break_name']:<13s}  "
              f"F={r['F']:.3f}  p={r['p_value']:.4g}")

    subsection("Part 1 vs Part 2 concordance")
    merged = (classical
              .merge(hac[['country', 'break_name', 'F', 'p_value']]
                     .rename(columns={'F': 'F_hac', 'p_value': 'p_hac'}),
                     on=['country', 'break_name']))
    merged['rej_classical'] = merged['p_value'] < ALPHA_NOMINAL
    merged['rej_hac']       = merged['p_hac']   < ALPHA_NOMINAL
    merged['agree']         = merged['rej_classical'] == merged['rej_hac']
    n_agree = int(merged['agree'].sum())
    print(f"  {n_agree}/{len(merged)} tests agree between classical and HAC "
          f"on reject/non-reject at α={ALPHA_NOMINAL}")
    disagreements = merged[~merged['agree']]
    for _, r in disagreements.iterrows():
        print(f"   ! {r['country']:<8s} {r['break_name']:<13s}  "
              f"classical p={r['p_value']:.4f}  HAC p={r['p_hac']:.4f}")

    # ── Part 3 ────────────────────────────────────────────────
    section("PART 3 — COVID-dummy-augmented Chow "
            "(GFC and ENERGY only; COVID dummy absorbs 2020-03..09)")
    covid = run_part3_covid_dummy(chow_datasets)
    p3 = covid[['country', 'break_name', 'break_date',
                'n_total', 'n_pre', 'n_post',
                'df_num', 'df_denom',
                'F', 'p_value', 'covid_coef', 'covid_se']].copy()
    p3 = fmt(p3, {'F': '{:>8.3f}', 'p_value': '{:.4f}',
                  'covid_coef': '{:>8.3f}', 'covid_se': '{:>7.3f}'})
    p3['sig'] = covid['p_value'].apply(lambda p: sig_marker(p, ALPHA_NOMINAL))
    print(p3.to_string(index=False))

    subsection("Part 2 vs Part 3 — does COVID-dummy change the verdict "
               "for GFC / ENERGY breaks?")
    hac_ge = hac[hac['break_name'].isin(COVID_DUMMY_BREAKS)]
    compare = (hac_ge.merge(
        covid[['country', 'break_name', 'F', 'p_value']]
              .rename(columns={'F': 'F_covid', 'p_value': 'p_covid'}),
        on=['country', 'break_name']))
    for _, r in compare.iterrows():
        flip = ""
        rej_hac   = r['p_value'] < ALPHA_NOMINAL
        rej_covid = r['p_covid'] < ALPHA_NOMINAL
        if rej_hac != rej_covid:
            flip = "  <-- verdict flip"
        print(f"  {r['country']:<8s} {r['break_name']:<13s}  "
              f"HAC p={r['p_value']:.4f}  COVID-dummy p={r['p_covid']:.4f}{flip}")

    # ── Part 4 ────────────────────────────────────────────────
    section("PART 4 — Per-coefficient pre/post decomposition (HAC SE)")
    decomp = run_part4_decomposition(chow_datasets)

    hac_sig_set = set(
        (r['country'], r['break_name'])
        for _, r in hac.iterrows()
        if pd.notnull(r['p_value']) and r['p_value'] < ALPHA_NOMINAL
    )

    subsection("Decomposition rows with |z|>=1.96 (or in HAC-rejected tests)")
    show_mask = (decomp['z_stat'].abs() >= 1.96) | \
                decomp.apply(lambda r: (r['country'], r['break_name'])
                             in hac_sig_set, axis=1)
    p4 = decomp[show_mask][['country', 'break_name', 'variable',
                            'coef_pre', 'se_pre',
                            'coef_post', 'se_post',
                            'delta', 'se_delta',
                            'z_stat', 'p_value']].copy()
    p4 = fmt(p4, {
        'coef_pre':  '{:>8.4f}', 'se_pre':   '{:>7.4f}',
        'coef_post': '{:>8.4f}', 'se_post':  '{:>7.4f}',
        'delta':     '{:>8.4f}', 'se_delta': '{:>7.4f}',
        'z_stat':    '{:>6.2f}', 'p_value':  '{:.4f}',
    })
    if len(p4):
        print(p4.to_string(index=False))
    else:
        print("  (no rows pass the |z|>=1.96 threshold; see CSV for full table)")

    subsection("Dominant drivers per (country × break) — |z|-ranked regressor")
    for country in MAIN_COUNTRIES:
        for break_name in BREAK_DATES:
            block = decomp[(decomp['country'] == country)
                           & (decomp['break_name'] == break_name)].copy()
            if block.empty:
                continue
            block = block.reindex(
                block['z_stat'].abs().sort_values(ascending=False).index)
            top = block.iloc[0]
            tag = '*' if pd.notnull(top['p_value']) \
                and top['p_value'] < ALPHA_NOMINAL else ' '
            print(f"  {country:<8s} {break_name:<13s}  "
                  f"top: {top['variable']:<13s}  "
                  f"Δ={top['delta']:+.4f}  z={top['z_stat']:+6.2f}{tag}")

    # ── Part 5 ────────────────────────────────────────────────
    section("PART 5 — Bonferroni-adjusted significance "
            f"(family size m=12, α_bonf={ALPHA_NOMINAL/12:.4f})")
    bonf = run_part5_bonferroni(classical, hac, covid)

    subsection("Rejection counts (nominal α=0.05 vs Bonferroni α=0.00417)")
    for variant in ['classical', 'hac', 'covid_dummy']:
        block = bonf[bonf['variant'] == variant]
        n_nom  = int(block['reject_nom'].sum())
        n_bonf = int(block['reject_bonf'].sum())
        print(f"  {variant:<12s}  n={len(block):>2d}  "
              f"nominal rejects={n_nom}  Bonferroni rejects={n_bonf}")

    subsection("Tests surviving Bonferroni correction (reject_bonf == True)")
    survivors = bonf[bonf['reject_bonf']].copy()
    if len(survivors):
        s = survivors[['variant', 'country', 'break_name',
                       'F', 'p_value']].copy()
        s = fmt(s, {'F': '{:>8.3f}', 'p_value': '{:.2e}'})
        print(s.to_string(index=False))
    else:
        print("  (none — no break is significant at the family-wise level "
              "after Bonferroni correction)")

    # ── Part 6 ────────────────────────────────────────────────
    section("PART 6 — Small-sample caveats")
    subsection("ENERGY_2022 post-window n_post per country")
    energy_rows = hac[hac['break_name'] == 'ENERGY_2022'].copy()
    for _, r in energy_rows.iterrows():
        n_post = r['n_post']
        tag = ('  HARD-WARN' if n_post < SMALL_SAMPLE_HARD
               else '  SMALL-WARN' if n_post < SMALL_SAMPLE_WARN
               else '')
        print(f"  {r['country']:<8s} n_post={n_post:>3d}  "
              f"HAC p={r['p_value']:.4f}{tag}")

    subsection("Interpretation note")
    print(("  ENERGY_2022 Chow tests are run over a pre-window of 240+ and a\n"
           "  post-window of ~38–45 observations. With k=5 regressors, the\n"
           "  F statistic remains usable (degrees of freedom sufficient), but\n"
           "  the power against moderate slope changes is reduced. A\n"
           "  non-rejection at ENERGY_2022 should not be read as strong\n"
           "  evidence of structural stability; a rejection, conversely, is\n"
           "  robust because the post-shock signal is large relative to\n"
           "  sample noise."))

    # ── CSV outputs ───────────────────────────────────────────
    section("CSV outputs")
    doc_dir = PROJECT_ROOT / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ('phase3_chow_tests_classical.csv',           classical),
        ('phase3_chow_tests_hac.csv',                 hac),
        ('phase3_chow_tests_covid_dummy.csv',         covid),
        ('phase3_chow_coefficient_decomposition.csv', decomp),
        ('phase3_chow_bonferroni_summary.csv',        bonf),
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
