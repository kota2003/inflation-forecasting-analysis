"""
scripts/phase3_step1_adf_kpss_levels.py
=======================================
Phase 3 · Step 1 — ADF + KPSS joint stationarity test on level series.

Purpose
-------
Apply the Augmented Dickey-Fuller and KPSS tests to all 20 series
(4 main countries × 5 indicators) at their current (post-Phase-2)
transformation state, and classify each series into the four-quadrant
ADF-KPSS decision space:

    ADF reject + KPSS non-reject        →  Stationary
    ADF non-reject + KPSS reject        →  Non-stationary
    ADF reject + KPSS reject            →  Trend-stationary (conflicting)
    ADF non-reject + KPSS non-reject    →  Inconclusive

This is the first scratch script of Phase 3; its results feed the
eventual D-024 (joint-test protocol) and D-027 (transformation registry)
entries in ProjectDriven.md.

Design decisions embedded (draft — subject to D-024 / D-025 / D-026)
--------------------------------------------------------------------
* ADF regression spec per indicator (論点 2):
      CPI           → 'ct' (constant + trend; CPI is a level index)
      POLICY_RATE   → 'c'
      UNEMPLOYMENT  → 'c'
      GDP  (YoY %)  → 'c'
      M2   (YoY %)  → 'c'
  KPSS regression is matched to ADF so the null hypotheses are
  comparable on the same deterministic specification.
* Lag selection for ADF: AIC (`autolag='AIC'`) up to a Schwert (1989)
  upper bound ⌊12·(T/100)^(1/4)⌋ ≈ 15 for T ≈ 298.
* KPSS lag selection: 'auto' (Hobijn et al. 1998).
* Significance level α = 0.05 for the four-quadrant classification.

Inputs
------
data/processed/main_{usa,japan,uk,germany}.csv
    via `src.data_loader.load_processed_all_main()`.

Outputs
-------
stdout
    Human-readable per-series table + classification summary.
data/documentation/phase3_adf_kpss_levels.csv
    Structured results (one row per country×indicator).

Usage
-----
Run from the project root:

    python scripts/phase3_step1_adf_kpss_levels.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

# Make `src` importable when running this file directly
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

# ADF regression specification per indicator (see D-025 draft)
ADF_REGRESSION = {
    'CPI':          'ct',   # level index with clear upward trend
    'POLICY_RATE':  'c',
    'UNEMPLOYMENT': 'c',
    'GDP':          'c',    # already YoY %
    'M2':           'c',    # already YoY %
}


def schwert_maxlag(n_obs: int) -> int:
    """Schwert (1989) upper bound on the ADF lag search:

        maxlag = floor(12 * (T / 100) ** (1/4))
    """
    return int(np.floor(12 * (n_obs / 100.0) ** 0.25))


# ──────────────────────────────────────────────────────────────────
# Test wrappers
# ──────────────────────────────────────────────────────────────────
def run_adf(series: pd.Series, regression: str) -> dict:
    """Run ADF with AIC-selected lag up to the Schwert upper bound."""
    s = series.dropna()
    maxlag = schwert_maxlag(len(s))
    try:
        stat, pvalue, usedlag, nobs, crit, _icbest = adfuller(
            s.values,
            maxlag=maxlag,
            regression=regression,
            autolag='AIC',
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


def run_kpss(series: pd.Series, regression: str) -> dict:
    """Run KPSS with 'auto' lag (Hobijn et al. 1998).

    statsmodels issues an InterpolationWarning when the test statistic
    lands outside the tabulated p-value range [0.01, 0.10] and clamps
    the returned p-value.  We capture that flag so the classification
    can be audited later.
    """
    s = series.dropna()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', InterpolationWarning)
        try:
            stat, pvalue, nlags, crit = kpss(
                s.values,
                regression=regression,
                nlags='auto',
            )
            interpolated = any(
                issubclass(w.category, InterpolationWarning) for w in caught
            )
            return {
                'kpss_stat':        float(stat),
                'kpss_pvalue':      float(pvalue),
                'kpss_lags':        int(nlags),
                'kpss_crit_1pct':   float(crit['1%']),
                'kpss_crit_5pct':   float(crit['5%']),
                'kpss_crit_10pct':  float(crit['10%']),
                'kpss_interp_flag': bool(interpolated),
                'kpss_error':       None,
            }
        except Exception as e:
            return {
                'kpss_stat':        np.nan,
                'kpss_pvalue':      np.nan,
                'kpss_lags':        -1,
                'kpss_crit_1pct':   np.nan,
                'kpss_crit_5pct':   np.nan,
                'kpss_crit_10pct':  np.nan,
                'kpss_interp_flag': False,
                'kpss_error':       repr(e),
            }


def classify_4quadrant(adf_p: float, kpss_p: float,
                       alpha: float = ALPHA) -> str:
    """Four-quadrant ADF-KPSS joint classification (D-024 draft).

    ADF  H0: unit root (non-stationary)       → reject  ⇒ "stationary"
    KPSS H0: (trend-)stationary               → reject  ⇒ "non-stationary"
    """
    if np.isnan(adf_p) or np.isnan(kpss_p):
        return 'Error'
    adf_rej = adf_p < alpha
    kpss_rej = kpss_p < alpha
    if adf_rej and not kpss_rej:
        return 'Stationary'
    if (not adf_rej) and kpss_rej:
        return 'Non-stationary'
    if adf_rej and kpss_rej:
        return 'Trend-stationary (conflict)'
    return 'Inconclusive'


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 79)
    print("Phase 3 · Step 1 — ADF + KPSS on Level Series")
    print(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Project   : {PROJECT_ROOT}")
    print(f"Alpha     : {ALPHA}")
    print("=" * 79)

    datasets = load_processed_all_main()
    print(f"\nLoaded {len(datasets)} main-country datasets:")
    for c, df in datasets.items():
        print(f"  {c:<8s} : {df.shape[0]} rows x {df.shape[1]} cols  "
              f"({df.index.min():%Y-%m} -> {df.index.max():%Y-%m})")

    rows = []
    for country in MAIN_COUNTRIES:
        df = datasets[country]
        for indicator in INDICATORS:
            col = f"{country}_{indicator}"
            if col not in df.columns:
                print(f"  WARN: column {col} missing; skipped")
                continue

            series = df[col]
            regression = ADF_REGRESSION[indicator]

            adf = run_adf(series, regression=regression)
            kps = run_kpss(series, regression=regression)
            cls = classify_4quadrant(adf['adf_pvalue'], kps['kpss_pvalue'])

            rows.append({
                'country':        country,
                'indicator':      indicator,
                'column':         col,
                'n_obs':          int(series.dropna().shape[0]),
                'regression':     regression,
                **adf,
                **kps,
                'classification': cls,
            })

    results = pd.DataFrame(rows)

    # ── Per-series human-readable table ───────────────────────
    print("\n" + "-" * 79)
    print("Per-series results (levels):")
    print("-" * 79)
    pretty = results[[
        'country', 'indicator', 'regression', 'n_obs',
        'adf_stat', 'adf_pvalue', 'adf_usedlag',
        'kpss_stat', 'kpss_pvalue', 'kpss_lags',
        'classification',
    ]].copy()
    pretty['adf_stat']    = pretty['adf_stat'].map(lambda x: f"{x:>8.4f}")
    pretty['adf_pvalue']  = pretty['adf_pvalue'].map(lambda x: f"{x:.4f}")
    pretty['kpss_stat']   = pretty['kpss_stat'].map(lambda x: f"{x:>7.4f}")
    pretty['kpss_pvalue'] = pretty['kpss_pvalue'].map(lambda x: f"{x:.4f}")
    print(pretty.to_string(index=False))

    # ── Classification summary ────────────────────────────────
    print("\n" + "-" * 79)
    print("Classification summary:")
    print("-" * 79)
    summary = results['classification'].value_counts()
    for k, v in summary.items():
        print(f"  {k:<32s} {v:>3d}")
    print(f"  {'TOTAL':<32s} {len(results):>3d}")

    # ── KPSS p-value interpolation notice ─────────────────────
    n_interp = int(results['kpss_interp_flag'].sum())
    if n_interp:
        print(f"\nNote: {n_interp} KPSS p-values were clamped at the "
              "tabulated boundary (0.01 or 0.10) by statsmodels.")
        print("      These rows are flagged in the CSV via kpss_interp_flag.")

    # ── Per-indicator breakdown ───────────────────────────────
    print("\n" + "-" * 79)
    print("Breakdown by indicator (rows=indicator, cols=classification):")
    print("-" * 79)
    pivot = (results
             .groupby(['indicator', 'classification'])
             .size()
             .unstack(fill_value=0))
    print(pivot.to_string())

    # ── Write audit CSV ───────────────────────────────────────
    doc_dir = PROJECT_ROOT / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    out_path = doc_dir / 'phase3_adf_kpss_levels.csv'
    results.to_csv(out_path, index=False)
    rel = out_path.relative_to(PROJECT_ROOT).as_posix()
    print(f"\nWrote audit CSV : {rel}  ({len(results)} rows)")

    # ── Hint for Step 2 ───────────────────────────────────────
    needs_diff = results[results['classification'].isin(
        ['Non-stationary', 'Inconclusive']
    )]
    print("\n" + "-" * 79)
    print(f"Series flagged for first-differencing in Step 2 "
          f"(Non-stationary or Inconclusive): {len(needs_diff)}")
    print("-" * 79)
    if len(needs_diff):
        for _, r in needs_diff.iterrows():
            print(f"  {r['country']:<8s} {r['indicator']:<12s}  "
                  f"({r['classification']})")

    conflicts = results[results['classification'] == 'Trend-stationary (conflict)']
    if len(conflicts):
        print("\n" + "-" * 79)
        print(f"Series flagged for detrending / re-specification "
              f"(Trend-stationary conflict): {len(conflicts)}")
        print("-" * 79)
        for _, r in conflicts.iterrows():
            print(f"  {r['country']:<8s} {r['indicator']:<12s}  "
                  f"(ADF='{r['regression']}'; consider detrend + re-test)")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
