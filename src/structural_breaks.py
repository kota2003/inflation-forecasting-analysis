"""
src/structural_breaks.py
========================
Multivariate regression structural-break test utilities.

This module consolidates the Chow-test battery and the Quandt-Andrews
sup-Wald scanner used in Phase 3 Task 2 of the inflation forecasting
project.  It is imported by:
  - notebooks/03_stationarity_structural_breaks.ipynb  (Portfolio narrative)
  - scripts/phase3_step[4,5,5b]_*.py                   (scratch runners)

The dependent variable is typically a transformed CPI series; the
regressors are typically the four other macro indicators (policy rate,
unemployment, GDP, M2) likewise transformed per the final
Transformation Registry (D-027 / D-031).

Public API
----------
Constants
    DEFAULT_ALPHA, DEFAULT_HAC_LAG
    KNOWN_BREAKS               Pre-specified Phase 3 Task 2 break dates
    COVID_DUMMY_START, COVID_DUMMY_END
    ANDREWS_1993_TABLE_I       π₀ × k critical-value grid for sup-F

Dummy constructors
    make_split_dummy(index, break_date)
    make_covid_dummy(index, covid_start, covid_end)

Chow tests
    chow_test_classical(y, X, break_date)
    chow_test_hac(y, X, break_date, hac_lag)
    chow_test_covid_dummy(y, X, break_date, covid_start, covid_end, hac_lag)

Per-coefficient decomposition
    coefficient_decomposition(y, X, break_date, hac_lag)

Quandt-Andrews sup-Wald
    wald_at_break(y, X, break_date, hac_lag)
    quandt_andrews_scan(y, X, pi0, hac_lag)
    summarise_scan(country, curve, k, pi0)
    align_argmax_to_known(argmax_date, known_breaks, tol_months)

Decision references: D-028 (Chow y-form), D-029 (COVID dummy),
D-030 (Phase 6 regime strategy), D-032 (module separation).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ──────────────────────────────────────────────────────────────────
# Constants (public)
# ──────────────────────────────────────────────────────────────────
DEFAULT_ALPHA:   float = 0.05
DEFAULT_HAC_LAG: int   = 4   #: Newey-West lag for monthly macro data

#: The three Phase 3 Task 2 break dates from ProjectScope §9 and D-002.
KNOWN_BREAKS: dict[str, pd.Timestamp] = {
    'GFC_2008':    pd.Timestamp('2008-09-01'),
    'COVID_2020':  pd.Timestamp('2020-03-01'),
    'ENERGY_2022': pd.Timestamp('2022-02-01'),
}

COVID_DUMMY_START: pd.Timestamp = pd.Timestamp('2020-03-01')
COVID_DUMMY_END:   pd.Timestamp = pd.Timestamp('2020-09-30')

#: Breaks for which COVID-dummy-augmented Chow is meaningful (COVID_2020
#: overlaps with the dummy period itself and is skipped by convention).
COVID_DUMMY_BREAKS: tuple[str, ...] = ('GFC_2008', 'ENERGY_2022')

#: Andrews (1993) Table I asymptotic critical values for SupF, indexed
#: by (π₀ trim fraction, k restrictions).  See D-024 / Step 5b.
#: Reference: Andrews, D.W.K. (1993). "Tests for Parameter Instability
#: and Structural Change with Unknown Change Point", Econometrica 61(4),
#: Table I.
ANDREWS_1993_TABLE_I: dict[float, dict[int, dict[str, float]]] = {
    0.05: {
        1: {'10%':  7.63, '5%':  9.31, '1%': 13.00},
        2: {'10%': 10.44, '5%': 12.41, '1%': 16.45},
        3: {'10%': 12.86, '5%': 14.94, '1%': 19.17},
        4: {'10%': 15.00, '5%': 17.13, '1%': 21.42},
        5: {'10%': 17.00, '5%': 19.39, '1%': 23.74},
        6: {'10%': 18.91, '5%': 21.36, '1%': 25.87},
        7: {'10%': 20.78, '5%': 23.26, '1%': 27.93},
    },
    0.10: {
        1: {'10%':  7.37, '5%':  9.03, '1%': 12.45},
        2: {'10%': 10.10, '5%': 12.02, '1%': 15.78},
        3: {'10%': 12.42, '5%': 14.50, '1%': 18.44},
        4: {'10%': 14.43, '5%': 16.57, '1%': 20.62},
        5: {'10%': 16.44, '5%': 18.82, '1%': 23.04},
        6: {'10%': 18.31, '5%': 20.72, '1%': 25.12},
        7: {'10%': 20.17, '5%': 22.60, '1%': 27.14},
    },
    0.15: {
        1: {'10%':  7.17, '5%':  8.85, '1%': 12.16},
        2: {'10%':  9.84, '5%': 11.79, '1%': 15.32},
        3: {'10%': 12.17, '5%': 14.17, '1%': 17.90},
        4: {'10%': 14.21, '5%': 16.31, '1%': 20.26},
        5: {'10%': 16.19, '5%': 18.48, '1%': 22.53},
        6: {'10%': 18.12, '5%': 20.52, '1%': 24.67},
        7: {'10%': 20.02, '5%': 22.52, '1%': 26.75},
    },
    0.20: {
        1: {'10%':  6.94, '5%':  8.56, '1%': 11.69},
        2: {'10%':  9.65, '5%': 11.54, '1%': 14.96},
        3: {'10%': 11.90, '5%': 13.84, '1%': 17.41},
        4: {'10%': 13.99, '5%': 16.06, '1%': 19.93},
        5: {'10%': 15.90, '5%': 18.24, '1%': 22.23},
        6: {'10%': 17.80, '5%': 20.27, '1%': 24.33},
        7: {'10%': 19.71, '5%': 22.26, '1%': 26.33},
    },
    0.25: {
        1: {'10%':  6.74, '5%':  8.33, '1%': 11.38},
        2: {'10%':  9.38, '5%': 11.22, '1%': 14.70},
        3: {'10%': 11.69, '5%': 13.61, '1%': 16.98},
        4: {'10%': 13.73, '5%': 15.75, '1%': 19.52},
        5: {'10%': 15.67, '5%': 17.97, '1%': 21.85},
        6: {'10%': 17.52, '5%': 19.91, '1%': 23.88},
        7: {'10%': 19.41, '5%': 21.93, '1%': 25.93},
    },
}


# ──────────────────────────────────────────────────────────────────
# Dummy constructors — always return index-aligned pd.Series
# ──────────────────────────────────────────────────────────────────
def make_split_dummy(index: pd.DatetimeIndex,
                     break_date: pd.Timestamp) -> pd.Series:
    """D_t = 1{t ≥ break_date}, as a pd.Series aligned on ``index``.

    Returning a pd.Series (rather than a bare ndarray) ensures DataFrame
    operations like ``.multiply(axis=0)``, ``.rename()``, and
    ``.to_frame()`` work seamlessly in subsequent pipeline stages.
    """
    return pd.Series(
        (index >= break_date).astype(float),
        index=index, name='D_split',
    )


def make_covid_dummy(index: pd.DatetimeIndex,
                     covid_start: pd.Timestamp = COVID_DUMMY_START,
                     covid_end:   pd.Timestamp = COVID_DUMMY_END) -> pd.Series:
    """D_covid,t = 1{covid_start ≤ t ≤ covid_end}, on ``index``."""
    return pd.Series(
        ((index >= covid_start) & (index <= covid_end)).astype(float),
        index=index, name='COVID',
    )


# ──────────────────────────────────────────────────────────────────
# Chow test implementations
# ──────────────────────────────────────────────────────────────────
def chow_test_classical(y: pd.Series,
                        X: pd.DataFrame,
                        break_date: pd.Timestamp) -> dict:
    """Classical Chow F-test (iid errors assumed).

        H0 : β_pre = β_post  (full-sample model adequate).
        F  = ((RSS_r − RSS_ur) / k) / (RSS_ur / (n − 2k))
           ~ F(k, n − 2k)     under H0 + iid.

    where k is the number of regressors (constant included).  A sub-sample
    size ≤ k returns NaN and records an error string.
    """
    X_full = sm.add_constant(X, has_constant='add')
    pre_mask  = X_full.index <  break_date
    post_mask = X_full.index >= break_date

    y_pre,  y_post  = y[pre_mask],  y[post_mask]
    X_pre,  X_post  = X_full[pre_mask], X_full[post_mask]
    n, k = len(y), X_full.shape[1]

    if len(y_pre) <= k or len(y_post) <= k:
        return {
            'F': np.nan, 'p_value': np.nan,
            'df_num': k, 'df_denom': max(0, n - 2 * k),
            'RSS_restricted': np.nan, 'RSS_unrestricted': np.nan,
            'n_total': n, 'n_pre': len(y_pre), 'n_post': len(y_post),
            'error': f"Sub-sample too small (pre={len(y_pre)}, "
                     f"post={len(y_post)}; requires > {k})",
        }

    m_r    = sm.OLS(y, X_full).fit()
    m_pre  = sm.OLS(y_pre,  X_pre).fit()
    m_post = sm.OLS(y_post, X_post).fit()

    RSS_r    = float(m_r.ssr)
    RSS_ur   = float(m_pre.ssr + m_post.ssr)
    df_num   = k
    df_denom = n - 2 * k
    F        = ((RSS_r - RSS_ur) / df_num) / (RSS_ur / df_denom)
    p        = float(1.0 - stats.f.cdf(F, df_num, df_denom))
    return {
        'F': float(F), 'p_value': p,
        'df_num': df_num, 'df_denom': df_denom,
        'RSS_restricted': RSS_r, 'RSS_unrestricted': RSS_ur,
        'n_total': n, 'n_pre': len(y_pre), 'n_post': len(y_post),
        'error': None,
    }


def chow_test_hac(y: pd.Series,
                  X: pd.DataFrame,
                  break_date: pd.Timestamp,
                  hac_lag: int = DEFAULT_HAC_LAG) -> dict:
    """HAC-robust Chow test via dummy-interaction Wald.

    Regression
    ----------
        y_t = α + X_t β + D_t (α' + X_t β') + ε_t,   D_t = 1{t ≥ break}

    Hypothesis
    ----------
        H0 : α' = 0 AND β' = 0   (k joint restrictions)

    The Wald statistic is computed with Newey-West HAC standard errors,
    so the test is robust to autocorrelation up to ``hac_lag`` and to
    conditional heteroskedasticity.  Returns dict analogous to
    :func:`chow_test_classical`.
    """
    X_full = sm.add_constant(X, has_constant='add')
    D      = make_split_dummy(X_full.index, break_date)
    pre_n, post_n = int((1 - D).sum()), int(D.sum())
    k = X_full.shape[1]

    if pre_n <= k or post_n <= k:
        return {
            'F': np.nan, 'p_value': np.nan,
            'df_num': k, 'df_denom': max(0, len(y) - 2 * k),
            'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
            'error': f"Sub-sample too small (pre={pre_n}, post={post_n})",
        }

    DX = X_full.multiply(D, axis=0)
    DX.columns = [f"D_{c}" for c in X_full.columns]
    X_big = pd.concat([X_full, DX], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X_big).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lag})

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

    return {
        'F': F, 'p_value': p,
        'df_num': df_num, 'df_denom': df_denom,
        'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
        'error': None,
    }


def chow_test_covid_dummy(y: pd.Series,
                          X: pd.DataFrame,
                          break_date: pd.Timestamp,
                          covid_start: pd.Timestamp = COVID_DUMMY_START,
                          covid_end:   pd.Timestamp = COVID_DUMMY_END,
                          hac_lag: int = DEFAULT_HAC_LAG) -> dict:
    """HAC-robust Chow test with an additive COVID-period level dummy.

    Regression
    ----------
        y_t = α + X_t β + D_t (α' + X_t β') + γ·1{COVID_t} + ε_t

    Where ``D_t`` is the split dummy at ``break_date`` and the COVID
    indicator equals 1 for ``covid_start ≤ t ≤ covid_end``.  The break
    test restriction remains

        H0 : α' = 0 AND β' = 0

    with γ estimated as a nuisance parameter that absorbs the 2020
    outlier shock so it does not contaminate the pre/post slope
    comparison (D-029).

    Additional returned fields beyond :func:`chow_test_hac`:
      covid_n     : number of observations flagged COVID.
      covid_coef  : fitted γ̂.
      covid_se    : HAC standard error of γ̂.
    """
    X_full = sm.add_constant(X, has_constant='add')
    D       = make_split_dummy(X_full.index, break_date)
    D_covid = make_covid_dummy(X_full.index, covid_start, covid_end)
    pre_n, post_n = int((1 - D).sum()), int(D.sum())
    k = X_full.shape[1]

    if pre_n <= k or post_n <= k:
        return {
            'F': np.nan, 'p_value': np.nan,
            'df_num': k, 'df_denom': max(0, len(y) - 2 * k - 1),
            'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
            'covid_n': int(D_covid.sum()),
            'error': f"Sub-sample too small (pre={pre_n}, post={post_n})",
        }

    DX = X_full.multiply(D, axis=0)
    DX.columns = [f"D_{c}" for c in X_full.columns]
    X_big = pd.concat([X_full, DX, D_covid.to_frame()], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X_big).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lag})

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

    return {
        'F': F, 'p_value': p,
        'df_num': df_num, 'df_denom': df_denom,
        'n_total': len(y), 'n_pre': pre_n, 'n_post': post_n,
        'covid_n': int(D_covid.sum()),
        'covid_coef': covid_coef, 'covid_se': covid_se,
        'error': None,
    }


# ──────────────────────────────────────────────────────────────────
# Per-coefficient decomposition
# ──────────────────────────────────────────────────────────────────
def coefficient_decomposition(y: pd.Series,
                              X: pd.DataFrame,
                              break_date: pd.Timestamp,
                              hac_lag: int = DEFAULT_HAC_LAG) -> list[dict]:
    """Fit OLS separately on pre/post sub-samples and report each β.

    For each regressor (constant included), returns
        Δβ_j    = β_post_j − β_pre_j
        SE(Δβ_j) ≈ sqrt(SE_pre_j² + SE_post_j²)   (independent sub-samples)
        z        = Δβ_j / SE(Δβ_j)
        p_value  = two-sided normal tail

    Returns a list of dicts (empty if either sub-sample is too small).
    Used for Phase 6 regime-dummy specification (D-030): the coefficient
    showing the largest |z| identifies the economic channel that drives
    the break and so should enter the regime-dummy interaction.
    """
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

    rows: list[dict] = []
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
            'coef_pre':  b_pre,  'se_pre':  se_pre,
            'coef_post': b_post, 'se_post': se_post,
            'delta':     delta,  'se_delta': se_delta,
            'z_stat':    z,      'p_value':  p_z,
            'n_pre':     len(y_pre),
            'n_post':    len(y_post),
        })
    return rows


# ──────────────────────────────────────────────────────────────────
# Quandt-Andrews sup-Wald
# ──────────────────────────────────────────────────────────────────
def wald_at_break(y: pd.Series,
                  X: pd.DataFrame,
                  break_date: pd.Timestamp,
                  hac_lag: int = DEFAULT_HAC_LAG) -> float:
    """HAC-robust Wald-F statistic at a single candidate break date.

    Identical construction to :func:`chow_test_hac` but returns only the
    scalar F statistic (no ancillary info), for efficient use inside a
    Quandt-Andrews scan loop.  Returns ``np.nan`` on small sub-samples
    or any numerical failure.
    """
    X_full = sm.add_constant(X, has_constant='add')
    D = make_split_dummy(X_full.index, break_date)
    pre_n, post_n = int((1 - D).sum()), int(D.sum())
    k = X_full.shape[1]
    if pre_n <= k or post_n <= k:
        return np.nan

    DX = X_full.multiply(D, axis=0)
    DX.columns = [f"D_{c}" for c in X_full.columns]
    X_big = pd.concat([X_full, DX], axis=1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X_big).fit(
                cov_type='HAC', cov_kwds={'maxlags': hac_lag})
        p_full = X_big.shape[1]
        R = np.zeros((k, p_full))
        R[:, k:2 * k] = np.eye(k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wald = model.wald_test(R, use_f=True)
        return float(np.asarray(wald.statistic).ravel()[0])
    except Exception:
        return np.nan


def quandt_andrews_scan(y: pd.Series,
                        X: pd.DataFrame,
                        pi0: float = 0.15,
                        hac_lag: int = DEFAULT_HAC_LAG) -> pd.DataFrame:
    """Compute HAC-Wald F at every candidate τ in the Andrews-trimmed window.

    Scan window is ``[⌈π₀·T⌉, ⌊(1−π₀)·T⌋]``.  π₀ = 0.15 is Andrews'
    standard trim; π₀ = 0.10 widens the search to capture late/early
    break candidates at the cost of slightly higher critical values.

    Returns a DataFrame indexed by ``candidate_date`` with one column
    ``wald_f``.  Dates at which the Wald regression cannot be fit
    (e.g., trivially small sub-samples) are recorded as NaN.
    """
    idx = X.index
    T = len(idx)
    lo = int(np.ceil(pi0 * T))
    hi = int(np.floor((1 - pi0) * T))
    candidates = idx[lo:hi + 1]
    rows = [{'candidate_date': tau,
             'wald_f': wald_at_break(y, X, tau, hac_lag=hac_lag)}
            for tau in candidates]
    return pd.DataFrame(rows).set_index('candidate_date')


def summarise_scan(country: str,
                   curve: pd.DataFrame,
                   k: int,
                   pi0: float = 0.15) -> dict:
    """Derive sup/avg/exp statistics + Andrews (1993) verdict from a scan.

    Parameters
    ----------
    country : Label to embed in the returned dict.
    curve   : Output of :func:`quandt_andrews_scan`.
    k       : Number of restrictions (= number of columns including
              constant in the original regression).
    pi0     : Trim fraction, used to look up critical values from
              :data:`ANDREWS_1993_TABLE_I`.

    Returns
    -------
    dict with: country, pi0, k_restrictions, sup_w, argmax_date,
    avg_w, exp_w, andrews_{1,5,10}pct, verdict_5pct, n_candidates,
    trim_window_start, trim_window_end.
    """
    wf = curve['wald_f'].dropna()
    if wf.empty:
        return {
            'country': country, 'pi0': pi0, 'k_restrictions': k,
            'sup_w': np.nan, 'argmax_date': None,
            'avg_w': np.nan, 'exp_w': np.nan,
            'andrews_1pct':  np.nan, 'andrews_5pct':  np.nan,
            'andrews_10pct': np.nan,
            'verdict_5pct':  'error',
            'n_candidates':  0,
            'trim_window_start': None, 'trim_window_end': None,
        }

    sup_w       = float(wf.max())
    argmax_date = wf.idxmax()
    avg_w       = float(wf.mean())

    # exp-W (Andrews-Ploberger 1994) computed via log-sum-exp for stability.
    z = 0.5 * wf.values
    z_max = z.max()
    exp_w = float(z_max - np.log(len(z)) + np.log(np.sum(np.exp(z - z_max))))

    crit = ANDREWS_1993_TABLE_I.get(pi0, {}).get(k, {})
    c1  = crit.get('1%',  np.nan)
    c5  = crit.get('5%',  np.nan)
    c10 = crit.get('10%', np.nan)

    if not np.isfinite(c5):
        verdict = 'no_crit_available'
    elif sup_w > c1:
        verdict = 'reject @ 1%'
    elif sup_w > c5:
        verdict = 'reject @ 5%'
    elif sup_w > c10:
        verdict = 'reject @ 10%'
    else:
        verdict = 'fail to reject'

    return {
        'country': country,
        'pi0':            pi0,
        'k_restrictions': k,
        'sup_w':          sup_w,
        'argmax_date':    argmax_date,
        'avg_w':          avg_w,
        'exp_w':          exp_w,
        'andrews_1pct':   c1,
        'andrews_5pct':   c5,
        'andrews_10pct':  c10,
        'verdict_5pct':   verdict,
        'n_candidates':   int(len(wf)),
        'trim_window_start': curve.index.min(),
        'trim_window_end':   curve.index.max(),
    }


def align_argmax_to_known(argmax_date: pd.Timestamp,
                          known_breaks: dict[str, pd.Timestamp] = KNOWN_BREAKS,
                          tol_months: int = 6) -> dict:
    """How close is the Quandt-Andrews argmax to each pre-specified break?

    For each named break, returns:
      months_to_<name>  : |argmax_date − break_date| in months
      aligned_<name>    : whether that distance is ≤ tol_months

    Plus:
      closest_known          : name of the closest known break
      closest_known_date     : its pd.Timestamp
      months_to_closest      : distance in months
      aligned_to_any_known   : whether aligned to at least one known break
    """
    out: dict = {}
    for name, date in known_breaks.items():
        months = abs((argmax_date.year - date.year) * 12
                     + (argmax_date.month - date.month))
        out[f'months_to_{name}'] = int(months)
        out[f'aligned_{name}']   = bool(months <= tol_months)
    dists = {n: abs((argmax_date - d).days) for n, d in known_breaks.items()}
    closest_name = min(dists, key=dists.get)
    out['closest_known']      = closest_name
    out['closest_known_date'] = known_breaks[closest_name]
    out['months_to_closest']  = int(
        abs((argmax_date.year - known_breaks[closest_name].year) * 12
            + (argmax_date.month - known_breaks[closest_name].month))
    )
    out['aligned_to_any_known'] = bool(
        any(out[f'aligned_{n}'] for n in known_breaks))
    return out
