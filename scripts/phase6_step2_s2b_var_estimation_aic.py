"""
scripts/phase6_step2_s2b_var_estimation_aic.py
===============================================
Phase 6 · Step 2 · S2b — VAR refit at AIC-selected lag per country.

Purpose
-------
Refit per-country VAR at the AIC-selected lag order (from S1) and
compare residual whiteness against the S2 BIC-primary baseline. This
script is invoked because S2 residual Ljung-Box diagnostics revealed
universal rejection of white-noise residuals at p*=2 (BIC-primary),
indicating the BIC-parsimony choice is insufficient for the inferential
VAR use-case (Granger / IRF / FEVD). S1 AIC selections were reserved
in the D-050 draft as a sensitivity clause for exactly this contingency.

Lag orders (AIC-selected, from phase6_step2_var_lag_selection_summary.csv)
-------------------------------------------------------------------------
    USA     : p = 12  (boundary-hit but S1b confirmed accept_lag12)
    JAPAN   : p =  5  (interior minimum — D-049 echo)
    UK      : p = 12  (boundary-hit but S1b confirmed accept_lag12)
    GERMANY : p = 12  (boundary-hit but S1b confirmed accept_lag12;
                       AIC actually INCREASES in lag 13..18 extension)

DOF check (safety margin at AIC p)
----------------------------------
Per-equation regressor count = p * 5 + 1 (const) + n_exog; residual DOF
must stay well above the 10-per-regressor rule of thumb:

    USA p=12 : 12*5 + 1 + 8 = 69 regs → 286-69 = 217 residual DOF ✓
    JPN p= 5 :  5*5 + 1 + 5 = 31 regs → 297-31 = 266 residual DOF ✓
    UK  p=12 : 12*5 + 1 + 6 = 67 regs → 290-67 = 223 residual DOF ✓
    GER p=12 : 12*5 + 1 + 7 = 68 regs → 290-68 = 222 residual DOF ✓

Scope boundaries (S2b only)
---------------------------
- Single refit at AIC p per country on the full sample.
- Identical exog structure to S2 (3 split + 2 period + 0..3 interactions).
- Residual Ljung-Box diagnostics at lag {12, 24}, same as S2.
- Side-by-side comparison CSV vs S2 diagnostics (whiteness only).
- No Granger (S3), IRF (S4), FEVD (S5), OOS forecast (S6).

Output artefacts
----------------
data/documentation/
    phase6_step2_s2b_var_coefficients_{country}.csv       (x4)
    phase6_step2_s2b_var_diagnostics.csv                  (20 rows)
    phase6_step2_s2b_var_stability.csv                    (roots/country)
    phase6_step2_s2b_fit_summary.csv                      (4 rows)
    phase6_step2_s2b_whiteness_comparison.csv             (20 rows)
        Per-country × per-equation side-by-side LB p-values:
        s2_p_star, s2_lb12_pvalue, s2_reject, s2b_p_star, s2b_lb12_pvalue,
        s2b_reject, lb12_improvement (difference in p-value).

Decisions referenced
--------------------
D-050  (locked draft, revised here)
       - Pre-revision: BIC p*=2 primary / AIC sensitivity
       - Post-revision: AIC-selected p primary (inferential) /
                        BIC p=2 parsimony reference (Phase 7 DM)
D-048  ARIMA precedent — diagnostic-driven stage escalation.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s2b_var_estimation_aic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path wiring ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (                                              # noqa: E402
    MAIN_COUNTRIES,
    INDICATORS,
    build_all_features,
    find_project_root,
    PHASE6_REGIME_SPEC,
    KNOWN_BREAKS,
)
from statsmodels.tsa.vector_ar.var_model import VAR            # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox        # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────

#: Per-country AIC-selected lag orders from S1
#: (phase6_step2_var_lag_selection_summary.csv).
P_PER_COUNTRY: dict[str, int] = {
    'USA':     12,
    'JAPAN':    5,
    'UK':      12,
    'GERMANY': 12,
}

VAR_TREND: str = 'c'
LB_LAGS: list[int] = [12, 24]
BASE_INDICATORS: list[str] = list(INDICATORS)
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())
PERIOD_KEYS: list[str] = ['GFC', 'COVID']


# ── Exog discovery (D-036 / D-030) — identical to S2 ─────────────────

def build_exog_column_list(country: str,
                           features_cols: list[str]) -> dict:
    split_cols = [f'{country}_D_{b}' for b in SPLIT_BREAK_NAMES]
    period_cols = [f'{country}_P_{p}' for p in PERIOD_KEYS]

    interaction_cols: list[str] = []
    for (c, break_name), driver in PHASE6_REGIME_SPEC.items():
        if c != country:
            continue
        if driver is None or driver == 'const':
            continue
        col = f'{country}_{driver}_x_D_{break_name}'
        interaction_cols.append(col)

    all_cols = split_cols + period_cols + interaction_cols
    missing = [c for c in all_cols if c not in features_cols]
    if missing:
        raise KeyError(f"{country}: missing exog columns {missing}")

    return {
        'split':       split_cols,
        'period':      period_cols,
        'interaction': interaction_cols,
        'all':         all_cols,
    }


def extract_endog_exog(features_df: pd.DataFrame,
                       country: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    endog_cols = [f'{country}_{ind}' for ind in BASE_INDICATORS]
    exog_info = build_exog_column_list(country, list(features_df.columns))
    exog_cols = exog_info['all']

    joint = features_df[endog_cols + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    endog = joint[endog_cols].copy()
    exog = joint[exog_cols].copy() if exog_cols else None
    return endog, exog, exog_info


# ── Fit + diagnostics — identical to S2 ──────────────────────────────

def fit_var(endog, exog, p, trend):
    return VAR(endog, exog=exog).fit(maxlags=p, trend=trend)


def coefficients_long_form(results, country, endog_cols, exog_cols):
    params, stderr, tvalues, pvalues = (
        results.params, results.stderr, results.tvalues, results.pvalues
    )
    rows = []
    for eq in endog_cols:
        for reg in params.index:
            c, se = float(params.loc[reg, eq]), float(stderr.loc[reg, eq])
            t, p = float(tvalues.loc[reg, eq]), float(pvalues.loc[reg, eq])
            rows.append({
                'country': country, 'equation': eq, 'regressor': reg,
                'coef': c, 'std_err': se, 't_stat': t, 'p_value': p,
                'ci_lo': c - 1.96 * se, 'ci_hi': c + 1.96 * se,
                'signif_5pct': bool(p < 0.05),
            })
    return pd.DataFrame(rows)


def residual_diagnostics(results, country, endog_cols, lb_lags):
    resid = results.resid
    rows = []
    for eq in endog_cols:
        r = resid[eq].dropna()
        lb = acorr_ljungbox(r, lags=lb_lags, return_df=True)
        row = {
            'country': country, 'equation': eq,
            'n_residuals': int(len(r)),
            'resid_mean': float(r.mean()),
            'resid_std':  float(r.std(ddof=1)),
            'resid_min':  float(r.min()),
            'resid_max':  float(r.max()),
        }
        for lag in lb_lags:
            row[f'lb_stat_lag{lag}']   = float(lb.loc[lag, 'lb_stat'])
            row[f'lb_pvalue_lag{lag}'] = float(lb.loc[lag, 'lb_pvalue'])
            row[f'lb_reject_wn_lag{lag}_5pct'] = bool(
                lb.loc[lag, 'lb_pvalue'] < 0.05
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stability_report(results, country):
    roots = np.asarray(results.roots)
    is_stable = bool(results.is_stable(verbose=False))
    rows = []
    for i, r in enumerate(roots):
        rows.append({
            'country': country, 'root_index': i,
            'root_real': float(np.real(r)),
            'root_imag': float(np.imag(r)),
            'abs_root':  float(np.abs(r)),
            'is_stable_overall': is_stable,
        })
    return pd.DataFrame(rows)


# ── S2 vs S2b whiteness comparison ───────────────────────────────────

def build_whiteness_comparison(
    s2b_diag: pd.DataFrame,
    doc_dir: Path,
) -> pd.DataFrame:
    """Side-by-side LB(12/24) comparison vs S2 (BIC p*=2) baseline.

    Reads data/documentation/phase6_step2_s2_var_diagnostics.csv and
    joins on (country, equation).
    """
    s2_path = doc_dir / 'phase6_step2_s2_var_diagnostics.csv'
    if not s2_path.exists():
        raise FileNotFoundError(
            f"S2 diagnostics not found at {s2_path}. "
            "Run phase6_step2_s2_var_estimation.py first."
        )
    s2 = pd.read_csv(s2_path)

    keep_s2 = ['country', 'equation',
               'lb_pvalue_lag12', 'lb_reject_wn_lag12_5pct',
               'lb_pvalue_lag24', 'lb_reject_wn_lag24_5pct']
    s2_small = s2[keep_s2].rename(columns={
        'lb_pvalue_lag12':          's2_lb12_pvalue',
        'lb_reject_wn_lag12_5pct':  's2_lb12_reject',
        'lb_pvalue_lag24':          's2_lb24_pvalue',
        'lb_reject_wn_lag24_5pct':  's2_lb24_reject',
    })

    keep_s2b = keep_s2
    s2b_small = s2b_diag[keep_s2b].rename(columns={
        'lb_pvalue_lag12':          's2b_lb12_pvalue',
        'lb_reject_wn_lag12_5pct':  's2b_lb12_reject',
        'lb_pvalue_lag24':          's2b_lb24_pvalue',
        'lb_reject_wn_lag24_5pct':  's2b_lb24_reject',
    })

    merged = pd.merge(s2_small, s2b_small, on=['country', 'equation'])
    merged['p_star_s2'] = 2
    merged['p_star_s2b'] = merged['country'].map(P_PER_COUNTRY)
    merged['lb12_pvalue_delta'] = (
        merged['s2b_lb12_pvalue'] - merged['s2_lb12_pvalue']
    )
    merged['lb24_pvalue_delta'] = (
        merged['s2b_lb24_pvalue'] - merged['s2_lb24_pvalue']
    )
    merged['improved_at_lb12'] = (
        merged['s2_lb12_reject'] & ~merged['s2b_lb12_reject']
    )
    merged['improved_at_lb24'] = (
        merged['s2_lb24_reject'] & ~merged['s2b_lb24_reject']
    )

    # Reorder
    ordered = ['country', 'equation',
               'p_star_s2', 's2_lb12_pvalue', 's2_lb12_reject',
               's2_lb24_pvalue', 's2_lb24_reject',
               'p_star_s2b', 's2b_lb12_pvalue', 's2b_lb12_reject',
               's2b_lb24_pvalue', 's2b_lb24_reject',
               'lb12_pvalue_delta', 'lb24_pvalue_delta',
               'improved_at_lb12', 'improved_at_lb24']
    return merged[ordered]


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 72
    print(bar)
    print('Phase 6 · Step 2 · S2b — VAR Refit at AIC-selected p per country')
    print(bar)
    print('Per-country lag orders (AIC-primary, D-050 revised):')
    for c in MAIN_COUNTRIES:
        print(f'    {c:<10s} p* = {P_PER_COUNTRY[c]}')
    print(f'trend:          {VAR_TREND!r}')
    print(f'LB lags:        {LB_LAGS}')
    print(f'exog structure: identical to S2 (3 split + 2 period + interactions)')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    all_diagnostics: list[pd.DataFrame] = []
    all_stability:   list[pd.DataFrame] = []
    fit_meta_rows:   list[dict] = []

    for country in MAIN_COUNTRIES:
        p_star = P_PER_COUNTRY[country]
        print(f'>>> {country}  p* = {p_star}')
        endog, exog, info = extract_endog_exog(features[country], country)
        n_exog = exog.shape[1] if exog is not None else 0
        n_reg_per_eq = p_star * len(BASE_INDICATORS) + 1 + n_exog
        residual_dof = len(endog) - n_reg_per_eq

        print(f'    n_obs={len(endog)}   n_exog={n_exog}   '
              f'regs/eq={n_reg_per_eq}   residual_DOF={residual_dof}')

        try:
            results = fit_var(endog, exog, p_star, VAR_TREND)
        except Exception as exc:
            print(f'    !! fit failed: {type(exc).__name__}: {exc}')
            raise

        is_stable = bool(results.is_stable(verbose=False))
        max_abs_root = float(np.max(np.abs(results.roots)))
        llf = float(results.llf)

        print(f'    log-lik={llf:+.4f}   AIC={results.aic:+.4f}   '
              f'BIC={results.bic:+.4f}')
        print(f'    max |root|={max_abs_root:.4f}   is_stable={is_stable}')

        coefs = coefficients_long_form(
            results, country,
            list(endog.columns),
            list(exog.columns) if exog is not None else [],
        )
        coefs_path = (doc_dir /
                      f'phase6_step2_s2b_var_coefficients_{country.lower()}.csv')
        coefs.to_csv(coefs_path, index=False)

        diag = residual_diagnostics(results, country, list(endog.columns), LB_LAGS)
        all_diagnostics.append(diag)

        stab = stability_report(results, country)
        all_stability.append(stab)

        for eq in endog.columns:
            row = diag[diag['equation'] == eq].iloc[0]
            f12 = '★' if row['lb_reject_wn_lag12_5pct'] else ' '
            f24 = '★' if row['lb_reject_wn_lag24_5pct'] else ' '
            print(f'      {eq:<22s}  LB(12) p={row["lb_pvalue_lag12"]:.4f}{f12}'
                  f'   LB(24) p={row["lb_pvalue_lag24"]:.4f}{f24}'
                  f'   σ={row["resid_std"]:.4f}')

        fit_meta_rows.append({
            'country': country,
            'p_star':  p_star,
            'n_obs':   int(len(endog)),
            'n_exog':  n_exog,
            'n_reg_per_eq': n_reg_per_eq,
            'residual_dof': residual_dof,
            'log_likelihood': llf,
            'aic': float(results.aic),
            'bic': float(results.bic),
            'hqic': float(results.hqic),
            'max_abs_root': max_abs_root,
            'is_stable': is_stable,
            'lb12_any_reject': bool(diag['lb_reject_wn_lag12_5pct'].any()),
            'lb24_any_reject': bool(diag['lb_reject_wn_lag24_5pct'].any()),
            'lb12_n_reject':   int(diag['lb_reject_wn_lag12_5pct'].sum()),
            'lb24_n_reject':   int(diag['lb_reject_wn_lag24_5pct'].sum()),
        })
        print()

    # Consolidated outputs
    diag_all = pd.concat(all_diagnostics, ignore_index=True)
    diag_path = doc_dir / 'phase6_step2_s2b_var_diagnostics.csv'
    diag_all.to_csv(diag_path, index=False)

    stab_all = pd.concat(all_stability, ignore_index=True)
    stab_path = doc_dir / 'phase6_step2_s2b_var_stability.csv'
    stab_all.to_csv(stab_path, index=False)

    meta_df = pd.DataFrame(fit_meta_rows)
    meta_path = doc_dir / 'phase6_step2_s2b_fit_summary.csv'
    meta_df.to_csv(meta_path, index=False)

    # S2 vs S2b comparison
    try:
        comp = build_whiteness_comparison(diag_all, doc_dir)
        comp_path = doc_dir / 'phase6_step2_s2b_whiteness_comparison.csv'
        comp.to_csv(comp_path, index=False)
        comparison_available = True
    except FileNotFoundError as exc:
        print(f'WARNING: {exc}')
        comparison_available = False

    # ------------------------------------------------------------------
    # Summary panels
    # ------------------------------------------------------------------
    print(bar)
    print('S2b Fit summary')
    print(bar)
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(meta_df[['country', 'p_star', 'n_obs', 'residual_dof',
                       'log_likelihood', 'aic', 'bic',
                       'max_abs_root', 'is_stable',
                       'lb12_n_reject', 'lb24_n_reject']]
              .to_string(index=False))
    print()

    if comparison_available:
        print(bar)
        print('S2 (BIC p*=2) → S2b (AIC p*) whiteness comparison')
        print(bar)

        # Per-country pass-rate summary
        rows = []
        for country in MAIN_COUNTRIES:
            sub = comp[comp['country'] == country]
            rows.append({
                'country':       country,
                'p_s2':          2,
                'p_s2b':         P_PER_COUNTRY[country],
                's2_lb12_pass':  int((~sub['s2_lb12_reject']).sum()),
                's2b_lb12_pass': int((~sub['s2b_lb12_reject']).sum()),
                's2_lb24_pass':  int((~sub['s2_lb24_reject']).sum()),
                's2b_lb24_pass': int((~sub['s2b_lb24_reject']).sum()),
                'lb12_improved': int(sub['improved_at_lb12'].sum()),
                'lb24_improved': int(sub['improved_at_lb24'].sum()),
            })
        pass_rate = pd.DataFrame(rows)
        print(pass_rate.to_string(index=False))
        print()
        total_eqs = len(comp)
        s2_lb12_pass = int((~comp['s2_lb12_reject']).sum())
        s2b_lb12_pass = int((~comp['s2b_lb12_reject']).sum())
        s2_lb24_pass = int((~comp['s2_lb24_reject']).sum())
        s2b_lb24_pass = int((~comp['s2b_lb24_reject']).sum())
        print(f'OVERALL (out of {total_eqs} equations):')
        print(f'  LB(12) pass rate:  S2 = {s2_lb12_pass}/{total_eqs}   '
              f'S2b = {s2b_lb12_pass}/{total_eqs}   '
              f'Δ = {s2b_lb12_pass - s2_lb12_pass:+d}')
        print(f'  LB(24) pass rate:  S2 = {s2_lb24_pass}/{total_eqs}   '
              f'S2b = {s2b_lb24_pass}/{total_eqs}   '
              f'Δ = {s2b_lb24_pass - s2_lb24_pass:+d}')
        print()

    # Written outputs
    print(bar)
    print('Output artefacts written:')
    for p in [diag_path, stab_path, meta_path]:
        print(f'  data/documentation/{p.name}')
    for country in MAIN_COUNTRIES:
        print(f'  data/documentation/'
              f'phase6_step2_s2b_var_coefficients_{country.lower()}.csv')
    if comparison_available:
        print(f'  data/documentation/{comp_path.name}')
    print()

    # Final verdict pointer
    print(bar)
    if comparison_available:
        improvement_12 = s2b_lb12_pass - s2_lb12_pass
        improvement_24 = s2b_lb24_pass - s2_lb24_pass

        if s2b_lb12_pass >= total_eqs * 0.8:
            print('Verdict: AIC-p refit SUCCESSFULLY whitens residuals.')
            print(f'         ≥80% LB(12) pass rate ({s2b_lb12_pass}/{total_eqs}).')
            print('         → Adopt AIC-p primary for S3 Granger causality.')
            print('         → D-050 revision CONFIRMED.')
        elif improvement_12 > 0 or improvement_24 > 0:
            print('Verdict: AIC-p refit PARTIALLY whitens residuals.')
            print(f'         LB(12) improvement: +{improvement_12} equations')
            print(f'         LB(24) improvement: +{improvement_24} equations')
            print('         → Adopt AIC-p primary; note remaining autocorr')
            print('           in Phase 7 caveat discussion.')
        else:
            print('Verdict: AIC-p refit did NOT improve residual whiteness.')
            print('         → Deeper investigation required.')
            print('         → Candidate issues: omitted variable, regime')
            print('           exog insufficient, heteroskedasticity, outliers.')
    print()
    print('Next sub-step: S3 = Granger causality battery on the selected fit.')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
