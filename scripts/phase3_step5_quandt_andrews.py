"""
scripts/phase3_step5b_quandt_andrews_trim10.py
==============================================
Phase 3 · Step 5b — Quandt-Andrews sup-Wald test at π₀ = 0.10 (wider trim).

Motivation
----------
Step 5 used the Andrews (1993) standard trim π₀ = 0.15, which scans the
middle 70% of the sample.  For the UK and Germany panels (both ending
2025-03, n = 290 monthly observations), this placed the right-hand
scan boundary at 2021-08 — *before* the ENERGY_2022 known break
(2022-02).  Consequently, the Step 5 argmax for those two countries
was driven to the boundary (UK) or to an earlier region (Germany)
rather than to the economically-motivated candidate break.

Step 5b rescans with π₀ = 0.10 (middle 80%), placing the right-hand
boundary at 2023-03 for UK/Germany.  This fully encloses all three
known break dates for all four countries.

Why this is defensible
----------------------
Andrews (1993) explicitly tabulates critical values for π₀ ∈ {0.05,
0.10, 0.15, 0.20, 0.25}; π₀ = 0.10 is within the standard range and
merely widens the search by 5 percentage points at each end.  The
asymptotic critical values are slightly higher (~0.3-0.5 at the 5%
level for k = 5) to reflect the wider search.  We use the π₀ = 0.10
row of Andrews Table I directly; the resulting inference is valid.

Removed from Step 5
-------------------
* The Hansen (1997) approximate p-value: the polynomial coefficients
  used in Step 5 did not match Hansen's published formula; the
  resulting p-values were internally inconsistent with the Andrews
  critical-value verdict.  This script uses Andrews (1993) Table I
  critical values only, which are well-established and
  numerically exact (not approximations).  The narrative "did
  sup-W exceed the Andrews critical value at 1/5/10%?" is sufficient
  and defensible for the Portfolio use.

Comparison to Step 5
--------------------
Outputs to new file names (`*_trim10.csv`) so the π₀ = 0.15 Step 5
results remain on disk as `*_trim15`-equivalent (the original file
names; no rename to preserve audit integrity).  The Portfolio notebook
will show both scans side-by-side.

Inputs / Outputs
----------------
Inputs: same as Step 5.
Outputs:
    stdout
    data/documentation/phase3_quandt_andrews_supwald_trim10.csv
    data/documentation/phase3_quandt_andrews_curve_trim10.csv

Usage
-----
Run from the project root:

    python scripts/phase3_step5b_quandt_andrews_trim10.py
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
import statsmodels.api as sm                                        # noqa: E402

from src.data_loader import (                                       # noqa: E402
    load_processed_all_main,
    INDICATORS,
    MAIN_COUNTRIES,
)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
TRIM_PI0 = 0.10   # S5b widens from 0.15 to 0.10; middle 80% scanned
HAC_LAG  = 4      # Match Step 4 Newey-West

# Andrews (1993) Table I — asymptotic critical values for SupF test,
# parameterised by (trim_pi0, k).  k = number of restrictions.
#
# Source: Andrews (1993) "Tests for Parameter Instability and Structural
# Change with Unknown Change Point", Econometrica 61(4), Table I.
# Values reproduced as published.  For our case k = 5 (constant shift +
# 4 regressor shifts), the relevant row is ANDREWS_1993_TABLE_I[pi0][5].
ANDREWS_1993_TABLE_I = {
    # pi0 = 0.05
    0.05: {
        1: {'10%':  7.63, '5%':  9.31, '1%': 13.00},
        2: {'10%': 10.44, '5%': 12.41, '1%': 16.45},
        3: {'10%': 12.86, '5%': 14.94, '1%': 19.17},
        4: {'10%': 15.00, '5%': 17.13, '1%': 21.42},
        5: {'10%': 17.00, '5%': 19.39, '1%': 23.74},
        6: {'10%': 18.91, '5%': 21.36, '1%': 25.87},
        7: {'10%': 20.78, '5%': 23.26, '1%': 27.93},
    },
    # pi0 = 0.10
    0.10: {
        1: {'10%':  7.37, '5%':  9.03, '1%': 12.45},
        2: {'10%': 10.10, '5%': 12.02, '1%': 15.78},
        3: {'10%': 12.42, '5%': 14.50, '1%': 18.44},
        4: {'10%': 14.43, '5%': 16.57, '1%': 20.62},
        5: {'10%': 16.44, '5%': 18.82, '1%': 23.04},   # <-- our case
        6: {'10%': 18.31, '5%': 20.72, '1%': 25.12},
        7: {'10%': 20.17, '5%': 22.60, '1%': 27.14},
    },
    # pi0 = 0.15 (same values hardcoded in Step 5 for continuity)
    0.15: {
        1: {'10%':  7.17, '5%':  8.85, '1%': 12.16},
        2: {'10%':  9.84, '5%': 11.79, '1%': 15.32},
        3: {'10%': 12.17, '5%': 14.17, '1%': 17.90},
        4: {'10%': 14.21, '5%': 16.31, '1%': 20.26},
        5: {'10%': 16.19, '5%': 18.48, '1%': 22.53},
        6: {'10%': 18.12, '5%': 20.52, '1%': 24.67},
        7: {'10%': 20.02, '5%': 22.52, '1%': 26.75},
    },
    # pi0 = 0.20
    0.20: {
        1: {'10%':  6.94, '5%':  8.56, '1%': 11.69},
        2: {'10%':  9.65, '5%': 11.54, '1%': 14.96},
        3: {'10%': 11.90, '5%': 13.84, '1%': 17.41},
        4: {'10%': 13.99, '5%': 16.06, '1%': 19.93},
        5: {'10%': 15.90, '5%': 18.24, '1%': 22.23},
        6: {'10%': 17.80, '5%': 20.27, '1%': 24.33},
        7: {'10%': 19.71, '5%': 22.26, '1%': 26.33},
    },
    # pi0 = 0.25
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

KNOWN_BREAKS = {
    'GFC_2008':    pd.Timestamp('2008-09-01'),
    'COVID_2020':  pd.Timestamp('2020-03-01'),
    'ENERGY_2022': pd.Timestamp('2022-02-01'),
}

ALIGN_TOL_MONTHS = 6

REGISTRY_OVERRIDES = {
    ('JAPAN',   'CPI'): {'chow_test_input': 'first_diff'},
    ('GERMANY', 'CPI'): {'chow_test_input': 'first_diff'},
    ('UK',      'CPI'): {'chow_test_input': 'log_diff_pct'},
}

Y_INDICATOR  = 'CPI'
X_INDICATORS = ['POLICY_RATE', 'UNEMPLOYMENT', 'GDP', 'M2']


# ──────────────────────────────────────────────────────────────────
# Transform helpers (copied from S4/S5)
# ──────────────────────────────────────────────────────────────────
def first_difference(s):       return s.diff().dropna()
def second_difference(s):      return s.diff().diff().dropna()
def yoy_pct(s, periods=12):    return (100.0 * (s / s.shift(periods) - 1.0)).dropna()
def log_first_diff_pct(s):
    s = s.astype(float); return (100.0 * np.log(s / s.shift(1))).dropna()


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


def load_revised_registry() -> pd.DataFrame:
    path = (PROJECT_ROOT / 'data' / 'documentation'
            / 'phase3_transformation_registry_final.csv')
    if not path.exists():
        raise FileNotFoundError(f"Expected registry at {path}.")
    reg = pd.read_csv(path)
    for (country, indicator), overrides in REGISTRY_OVERRIDES.items():
        mask = (reg['country'] == country) & (reg['indicator'] == indicator)
        for k, v in overrides.items():
            reg.loc[mask, k] = v
    return reg


def build_chow_dataset(country: str,
                       datasets: dict,
                       registry: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_raw = datasets[country]
    cols, forms_used = {}, {}
    for indicator in INDICATORS:
        reg_row = registry[(registry['country'] == country)
                           & (registry['indicator'] == indicator)].iloc[0]
        form = strip_suffix(reg_row['chow_test_input'])
        forms_used[indicator] = form
        series = df_raw[f"{country}_{indicator}"]
        cols[indicator] = TRANSFORM_FN[form](series)
    df = pd.concat(cols, axis=1).dropna()
    df.index.name = 'date'
    return df, forms_used


# ──────────────────────────────────────────────────────────────────
# Wald-F at a single candidate break date (same as S4/S5)
# ──────────────────────────────────────────────────────────────────
def wald_at_break(y: pd.Series, X: pd.DataFrame,
                  break_date: pd.Timestamp,
                  hac_lag: int = HAC_LAG) -> float:
    X_full = sm.add_constant(X, has_constant='add')
    D = pd.Series(
        (X_full.index >= break_date).astype(float),
        index=X_full.index, name='D_split',
    )
    pre_n  = int((1 - D).sum())
    post_n = int(D.sum())
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


# ──────────────────────────────────────────────────────────────────
# Quandt-Andrews full scan + summary (same as S5 minus Hansen)
# ──────────────────────────────────────────────────────────────────
def quandt_andrews_scan(y: pd.Series, X: pd.DataFrame,
                        pi0: float = TRIM_PI0,
                        hac_lag: int = HAC_LAG) -> pd.DataFrame:
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
                   pi0: float) -> dict:
    wf = curve['wald_f'].dropna()
    if wf.empty:
        return {'country': country, 'sup_w': np.nan,
                'argmax_date': None, 'avg_w': np.nan, 'exp_w': np.nan,
                'andrews_1pct': np.nan, 'andrews_5pct': np.nan,
                'andrews_10pct': np.nan, 'verdict_5pct': 'error',
                'n_candidates': 0,
                'trim_window_start': None, 'trim_window_end': None}

    sup_w = float(wf.max())
    argmax_date = wf.idxmax()
    avg_w = float(wf.mean())
    z = 0.5 * wf.values
    z_max = z.max()
    exp_w = float(z_max - np.log(len(z)) + np.log(np.sum(np.exp(z - z_max))))

    crit = ANDREWS_1993_TABLE_I.get(pi0, {}).get(k, {})
    c1   = crit.get('1%',  np.nan)
    c5   = crit.get('5%',  np.nan)
    c10  = crit.get('10%', np.nan)

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
                          known_breaks: dict = KNOWN_BREAKS,
                          tol_months: int = ALIGN_TOL_MONTHS) -> dict:
    out = {}
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


# ──────────────────────────────────────────────────────────────────
# Optional: load Step 5 (π₀=0.15) summary for side-by-side comparison
# ──────────────────────────────────────────────────────────────────
def load_step5_summary_if_exists() -> pd.DataFrame | None:
    """Step 5 produced phase3_quandt_andrews_supwald.csv at π0 = 0.15.
    Read it for stdout comparison; return None if missing."""
    path = (PROJECT_ROOT / 'data' / 'documentation'
            / 'phase3_quandt_andrews_supwald.csv')
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 79); print(title); print("=" * 79)


def subsection(title: str) -> None:
    print("\n" + "-" * 79); print(title); print("-" * 79)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 79)
    print("Phase 3 · Step 5b — Quandt-Andrews sup-Wald at π₀ = 0.10")
    print(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Project   : {PROJECT_ROOT}")
    print(f"Trim π₀   : {TRIM_PI0}  (scan window = middle "
          f"{100*(1-2*TRIM_PI0):.0f}%)")
    print(f"HAC lag   : {HAC_LAG}")
    print(f"Known breaks: "
          + ", ".join(f"{n}={d:%Y-%m}" for n, d in KNOWN_BREAKS.items()))
    print("=" * 79)

    datasets = load_processed_all_main()
    registry = load_revised_registry()

    # ── Part 0 ────────────────────────────────────────────────
    section("PART 0 — Setup: per-country datasets and scan window")
    chow_datasets = {}
    for country in MAIN_COUNTRIES:
        df, forms = build_chow_dataset(country, datasets, registry)
        chow_datasets[country] = {'df': df, 'forms': forms}

    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        T  = len(df)
        lo = int(np.ceil(TRIM_PI0 * T))
        hi = int(np.floor((1 - TRIM_PI0) * T))
        window = df.index[lo:hi + 1]
        # Check coverage of each known break
        coverage = []
        for bname, bdate in KNOWN_BREAKS.items():
            inside = window[0] <= bdate <= window[-1]
            coverage.append(f"{bname}={'✓' if inside else '✗'}")
        print(f"  {country:<8s} n={T:>3}  "
              f"trim-window={window[0]:%Y-%m}..{window[-1]:%Y-%m} "
              f"({len(window)} dates)   coverage: [{', '.join(coverage)}]")

    # ── Part 1: Scan ──────────────────────────────────────────
    section("PART 1 — Sup-Wald scan at π₀ = 0.10")
    curves = {}
    summaries = []
    k_restrictions = len(X_INDICATORS) + 1  # = 5

    for country in MAIN_COUNTRIES:
        df = chow_datasets[country]['df']
        y  = df[Y_INDICATOR]
        X  = df[X_INDICATORS]

        print(f"\n  Scanning {country}...", end=' ', flush=True)
        curve = quandt_andrews_scan(y, X, pi0=TRIM_PI0, hac_lag=HAC_LAG)
        curves[country] = curve
        summ = summarise_scan(country, curve, k=k_restrictions,
                              pi0=TRIM_PI0)
        summ.update(align_argmax_to_known(summ['argmax_date']))
        summaries.append(summ)
        print(f"done (sup-W={summ['sup_w']:.3f} at "
              f"{summ['argmax_date']:%Y-%m})")

    summary_df = pd.DataFrame(summaries)

    # ── Part 2: Summary ───────────────────────────────────────
    section("PART 2 — Per-country sup-Wald summary (π₀ = 0.10)")
    show = summary_df[['country', 'sup_w', 'argmax_date',
                       'avg_w', 'exp_w',
                       'andrews_10pct', 'andrews_5pct', 'andrews_1pct',
                       'verdict_5pct',
                       'n_candidates']].copy()
    show['sup_w']         = show['sup_w'].map('{:>8.3f}'.format)
    show['avg_w']         = show['avg_w'].map('{:>7.3f}'.format)
    show['exp_w']         = show['exp_w'].map('{:>7.3f}'.format)
    show['andrews_10pct'] = show['andrews_10pct'].map('{:>6.2f}'.format)
    show['andrews_5pct']  = show['andrews_5pct'].map('{:>6.2f}'.format)
    show['andrews_1pct']  = show['andrews_1pct'].map('{:>6.2f}'.format)
    show['argmax_date']   = show['argmax_date'].map(
        lambda d: d.strftime('%Y-%m') if pd.notnull(d) else '')
    print(show.to_string(index=False))

    subsection(f"Andrews (1993) Table I critical values: "
               f"π₀ = {TRIM_PI0}, k = {k_restrictions}")
    crit = ANDREWS_1993_TABLE_I[TRIM_PI0][k_restrictions]
    print(f"  10% : {crit['10%']:.2f}")
    print(f"   5% : {crit['5%']:.2f}")
    print(f"   1% : {crit['1%']:.2f}")

    subsection("Verdicts at 5%")
    for _, r in summary_df.iterrows():
        tag = ('*** reject' if 'reject' in r['verdict_5pct']
               else '    fail to reject')
        print(f"  {r['country']:<8s} sup-W={r['sup_w']:>7.3f}  "
              f"[{r['verdict_5pct']}]")

    # ── Part 3: Alignment ─────────────────────────────────────
    section("PART 3 — Alignment: does argmax land near a known break?")
    align_cols = ['country', 'argmax_date',
                  'months_to_GFC_2008',   'aligned_GFC_2008',
                  'months_to_COVID_2020', 'aligned_COVID_2020',
                  'months_to_ENERGY_2022','aligned_ENERGY_2022',
                  'closest_known', 'months_to_closest',
                  'aligned_to_any_known']
    align_show = summary_df[align_cols].copy()
    align_show['argmax_date'] = align_show['argmax_date'].map(
        lambda d: d.strftime('%Y-%m') if pd.notnull(d) else '')
    print(align_show.to_string(index=False))

    subsection(f"Interpretation (alignment tolerance ±{ALIGN_TOL_MONTHS} months)")
    for _, r in summary_df.iterrows():
        aligned = r['aligned_to_any_known']
        tag = "CONFIRMS" if aligned else "NOVEL   "
        print(f"  {tag} {r['country']:<8s} argmax={r['argmax_date']:%Y-%m}  "
              f"closest known = {r['closest_known']} "
              f"(Δ={r['months_to_closest']:>2d} months)")

    n_aligned = int(summary_df['aligned_to_any_known'].sum())
    print(f"\n  {n_aligned}/{len(summary_df)} countries' argmax aligns with a "
          f"known break (within ±{ALIGN_TOL_MONTHS}mo).")

    # ── Part 4: Top-3 local maxima ────────────────────────────
    section("PART 4 — Top-3 candidate dates per country")
    for country in MAIN_COUNTRIES:
        curve = curves[country]
        top = curve['wald_f'].dropna().sort_values(ascending=False).head(3)
        print(f"\n  {country}:")
        for rank, (date, w) in enumerate(top.items(), 1):
            dists = {n: abs((date.year - d.year) * 12
                            + (date.month - d.month))
                     for n, d in KNOWN_BREAKS.items()}
            closest = min(dists, key=dists.get)
            print(f"    #{rank}  W={w:>7.3f}  date={date:%Y-%m}  "
                  f"(closest known: {closest}, Δ={dists[closest]}mo)")

    # ── Part 5: Step 5 vs Step 5b comparison ─────────────────
    section("PART 5 — Step 5 (π₀=0.15) vs Step 5b (π₀=0.10) comparison")
    prev = load_step5_summary_if_exists()
    if prev is None:
        print("  (no previous Step 5 summary file found — skipping comparison)")
    else:
        subsection("Side-by-side sup-W and argmax per country")
        print(f"  {'country':<8s} "
              f"{'pi0=0.15':^30s} | {'pi0=0.10':^30s}")
        print(f"  {'':<8s} {'sup_w':>8s} {'argmax':>10s} {'verdict':>10s} | "
              f"{'sup_w':>8s} {'argmax':>10s} {'verdict':>10s}")
        print("  " + "-" * 77)
        prev_by_c = {r['country']: r for _, r in prev.iterrows()}
        for _, r in summary_df.iterrows():
            c = r['country']
            pv = prev_by_c.get(c)
            if pv is not None:
                pv_sup    = f"{pv['sup_w']:.3f}" if pd.notnull(pv.get('sup_w')) else '—'
                pv_argmax = str(pv.get('argmax_date', ''))[:10] \
                    if pd.notnull(pv.get('argmax_date')) else '—'
                pv_verd   = str(pv.get('verdict_5pct', ''))[:10]
            else:
                pv_sup, pv_argmax, pv_verd = '—', '—', '—'
            cur_sup    = f"{r['sup_w']:.3f}"
            cur_argmax = r['argmax_date'].strftime('%Y-%m') \
                if pd.notnull(r['argmax_date']) else '—'
            cur_verd   = str(r['verdict_5pct'])[:10]
            print(f"  {c:<8s} "
                  f"{pv_sup:>8s} {pv_argmax:>10s} {pv_verd:>10s} | "
                  f"{cur_sup:>8s} {cur_argmax:>10s} {cur_verd:>10s}")

    # ── Part 6: Step 4 vs Step 5b coherence ──────────────────
    section("PART 6 — Step 4 vs Step 5b coherence at known break dates")
    subsection("Wald value at each known break (should match Step 4 Part 2)")
    header = f"  {'country':<8s} {'break':<13s} {'date':<8s} {'W(τ)':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for country in MAIN_COUNTRIES:
        curve = curves[country]
        for bname, bdate in KNOWN_BREAKS.items():
            if bdate in curve.index:
                w = curve.loc[bdate, 'wald_f']
                marker = ''
            else:
                nearest = curve.index[
                    np.argmin(np.abs(curve.index - bdate))]
                w = curve.loc[nearest, 'wald_f']
                marker = f"  (nearest scan date: {nearest:%Y-%m})"
            print(f"  {country:<8s} {bname:<13s} "
                  f"{bdate:%Y-%m}  {w:>8.3f}{marker}")

    # ── CSV outputs ───────────────────────────────────────────
    section("CSV outputs")
    doc_dir = PROJECT_ROOT / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    summary_out = summary_df.copy()
    summary_out['argmax_date'] = summary_out['argmax_date'].map(
        lambda d: d.strftime('%Y-%m-%d') if pd.notnull(d) else '')
    summary_out['trim_window_start'] = summary_out['trim_window_start'].map(
        lambda d: d.strftime('%Y-%m-%d') if pd.notnull(d) else '')
    summary_out['trim_window_end'] = summary_out['trim_window_end'].map(
        lambda d: d.strftime('%Y-%m-%d') if pd.notnull(d) else '')
    summary_out['closest_known_date'] = summary_out['closest_known_date'].map(
        lambda d: d.strftime('%Y-%m-%d') if pd.notnull(d) else '')
    sp = doc_dir / 'phase3_quandt_andrews_supwald_trim10.csv'
    summary_out.to_csv(sp, index=False)
    print(f"  wrote {sp.relative_to(PROJECT_ROOT).as_posix():<56s}  "
          f"({len(summary_out)} rows)")

    curve_rows = []
    for country, curve in curves.items():
        for date, row in curve.iterrows():
            curve_rows.append({
                'country': country,
                'pi0': TRIM_PI0,
                'candidate_date': date.strftime('%Y-%m-%d'),
                'wald_f': float(row['wald_f'])
                if pd.notnull(row['wald_f']) else np.nan,
            })
    curve_df = pd.DataFrame(curve_rows)
    cp = doc_dir / 'phase3_quandt_andrews_curve_trim10.csv'
    curve_df.to_csv(cp, index=False)
    print(f"  wrote {cp.relative_to(PROJECT_ROOT).as_posix():<56s}  "
          f"({len(curve_df)} rows)")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
