"""
scripts/phase6_step2_s2_var_estimation.py
==========================================
Phase 6 · Step 2 · S2 — VAR estimation with D-030 regime interactions.

Purpose
-------
Fit per-country VAR(p*=2) on the 5-variable base endogenous block
{CPI, POLICY_RATE, UNEMPLOYMENT, GDP, M2} in D-031 stationary form,
with D-030 dominant-driver regime interactions as exogenous controls,
and run residual whiteness + system stability diagnostics.

Lag order p*=2 is the S1/S1b-confirmed BIC-primary pick (D-050 locked).
Sensitivity candidates (AIC=12 for USA/UK/GER, AIC=5 for JPN, HQIC=3
for GER) are deferred to Phase 7 Diebold-Mariano forecast comparison
and do not execute in this script.

Scope boundaries (S2 only)
--------------------------
- Single estimation at p*=2 per country on the full sample.
- Diagnostics: residual Ljung-Box whiteness + VAR stability (roots).
- No Granger causality (S3), no IRF (S4), no FEVD (S5),
  no OOS forecasting (S6). Those are later sub-steps.
- D-005 train/test split is reserved for S6; S2 uses the full sample
  for the coefficient point estimates and residual diagnostics.

Exogenous structure (per D-030 × D-036)
---------------------------------------
For every country:
    3 split dummies  (D_GFC, D_COVID, D_ENERGY)
    2 period dummies (P_GFC, P_COVID)
  + 0-3 interaction terms gated by PHASE6_REGIME_SPEC:
      USA     : M2×D_GFC, POLICY_RATE×D_COVID, POLICY_RATE×D_ENERGY
      JAPAN   : (none — intercept-only shifts per D-030)
      UK      : GDP×D_ENERGY
      GERMANY : GDP×D_COVID, GDP×D_ENERGY

Output artefacts
----------------
data/documentation/
    phase6_step2_s2_exog_schema.csv
        4 rows × (country, n_split, n_period, n_interaction, n_exog_total,
                  exog_cols_joined).
    phase6_step2_s2_var_coefficients_{country}.csv
        Long-form coefficient table per country:
        (equation, regressor, coef, std_err, t_stat, p_value, ci_lo, ci_hi).
    phase6_step2_s2_var_diagnostics.csv
        Per-country × per-equation Ljung-Box (lag 12 / 24) + residual
        std, mean, min/max; plus per-country portmanteau whiteness.
    phase6_step2_s2_var_stability.csv
        Per-country roots of characteristic polynomial with |root|
        and max|root|; plus is_stable boolean (|root| < 1 for all).

Decisions referenced
--------------------
D-005  Train / test split 2000–2019 vs 2020+ (applied at S6, not here).
D-030  Dominant-driver matrix — source of interaction gating.
D-031  Base transformation registry (per-series form).
D-036  Regime dummy structure: 3 splits + 2 periods + 0–3 interactions.
D-048  ARIMA Stage (b) precedent and OOS-saturation stopping rule.
D-050  (locked draft) — BIC-primary p*=2 VAR lag selection protocol.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s2_var_estimation.py
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

#: VAR lag order confirmed by S1 / S1b (BIC-primary, locked as D-050
#: draft). All four countries share p*=2.
VAR_P_STAR: int = 2

#: Trend specification. 'c' = constant only. Stationary inputs do not
#: require a linear time trend; including one would over-fit.
VAR_TREND: str = 'c'

#: Ljung-Box lags for residual whiteness testing. {12, 24} echoes the
#: D-044 three-horizon convention used in Phase 5 ACF/PACF.
LB_LAGS: list[int] = [12, 24]

#: Five base endogenous variables forming the VAR system.
BASE_INDICATORS: list[str] = list(INDICATORS)

#: Break names that generate split dummies. Sourced from
#: src.structural_breaks.KNOWN_BREAKS so the dummy-column names stay in
#: sync with the module of record. Actual column name convention per
#: src.feature_engineering.build_split_dummies is
#: ``{COUNTRY}_D_{BREAK_NAME}`` where BREAK_NAME is the full dict key
#: (e.g. 'GFC_2008'), NOT the truncated root.
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())

#: Period-dummy names (PERIOD_WINDOWS keys).
PERIOD_KEYS: list[str] = ['GFC', 'COVID']


# ── Exog column discovery (D-036 / D-030) ────────────────────────────

def build_exog_column_list(country: str,
                           features_cols: list[str]) -> dict:
    """Discover the regime exogenous columns for a country.

    Returns a dict with keys:
        split        : list of split-dummy column names (length 3)
        period       : list of period-dummy column names (length 2)
        interaction  : list of interaction column names (length 0..3)
        all          : concatenation in fixed order
    """
    split_cols = [f'{country}_D_{b}' for b in SPLIT_BREAK_NAMES]
    period_cols = [f'{country}_P_{p}' for p in PERIOD_KEYS]

    # Interactions are discovered from PHASE6_REGIME_SPEC; only entries
    # whose value is a real indicator name (not None, not 'const')
    # produce a column.
    interaction_cols: list[str] = []
    for (c, break_name), driver in PHASE6_REGIME_SPEC.items():
        if c != country:
            continue
        if driver is None or driver == 'const':
            continue
        # Column convention per src.feature_engineering.build_interactions:
        #   {COUNTRY}_{DRIVER}_x_D_{BREAK_NAME}
        # where BREAK_NAME is the full KNOWN_BREAKS key (e.g. 'GFC_2008'),
        # matching how build_split_dummies names its output columns.
        col = f'{country}_{driver}_x_D_{break_name}'
        interaction_cols.append(col)

    # Sanity check — every candidate column must exist in features df.
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
    """Build joint endogenous + exogenous matrices, dropna-aligned."""
    endog_cols = [f'{country}_{ind}' for ind in BASE_INDICATORS]
    exog_info = build_exog_column_list(country, list(features_df.columns))
    exog_cols = exog_info['all']

    joint = features_df[endog_cols + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    endog = joint[endog_cols].copy()
    exog = joint[exog_cols].copy() if exog_cols else None
    return endog, exog, exog_info


# ── Fitting + diagnostics ────────────────────────────────────────────

def fit_var(endog: pd.DataFrame, exog: pd.DataFrame | None,
            p: int, trend: str):
    """Fit VAR(p) with exogenous regressors. Returns VARResults."""
    model = VAR(endog, exog=exog)
    return model.fit(maxlags=p, trend=trend)


def coefficients_long_form(results, country: str,
                           endog_cols: list[str],
                           exog_cols: list[str]) -> pd.DataFrame:
    """Extract coefficient matrix as long-form DataFrame.

    Columns: equation (endog target), regressor, coef, std_err, t_stat,
             p_value, ci_lo, ci_hi.
    """
    # statsmodels VARResults exposes per-equation regressions through
    # .params (DataFrame: rows = regressors, cols = equations),
    # .stderr, .tvalues, .pvalues. We'll also build 95% CI from coef
    # and stderr via Normal approximation (standard VAR practice).
    params = results.params
    stderr = results.stderr
    tvalues = results.tvalues
    pvalues = results.pvalues

    rows: list[dict] = []
    for eq in endog_cols:
        for reg in params.index:
            c = float(params.loc[reg, eq])
            se = float(stderr.loc[reg, eq])
            t = float(tvalues.loc[reg, eq])
            p = float(pvalues.loc[reg, eq])
            rows.append({
                'country':  country,
                'equation': eq,
                'regressor': reg,
                'coef':     c,
                'std_err':  se,
                't_stat':   t,
                'p_value':  p,
                'ci_lo':    c - 1.96 * se,
                'ci_hi':    c + 1.96 * se,
                'signif_5pct': bool(p < 0.05),
            })
    return pd.DataFrame(rows)


def residual_diagnostics(results, country: str,
                         endog_cols: list[str],
                         lb_lags: list[int]) -> pd.DataFrame:
    """Per-equation residual Ljung-Box + summary stats."""
    resid = results.resid  # DataFrame: index = dates, cols = endog
    rows: list[dict] = []
    for eq in endog_cols:
        r = resid[eq].dropna()
        lb = acorr_ljungbox(r, lags=lb_lags, return_df=True)
        # lb index = lag; cols = lb_stat, lb_pvalue.
        row = {
            'country':       country,
            'equation':      eq,
            'n_residuals':   int(len(r)),
            'resid_mean':    float(r.mean()),
            'resid_std':     float(r.std(ddof=1)),
            'resid_min':     float(r.min()),
            'resid_max':     float(r.max()),
        }
        for lag in lb_lags:
            row[f'lb_stat_lag{lag}']   = float(lb.loc[lag, 'lb_stat'])
            row[f'lb_pvalue_lag{lag}'] = float(lb.loc[lag, 'lb_pvalue'])
            row[f'lb_reject_wn_lag{lag}_5pct'] = (
                bool(lb.loc[lag, 'lb_pvalue'] < 0.05)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stability_report(results, country: str) -> pd.DataFrame:
    """Characteristic-polynomial roots + |root| + is_stable.

    statsmodels VARResults.roots returns the reciprocals of the roots
    of the determinantal polynomial of (I - A(L)); a stable VAR has
    all abs(roots) > 1 in that convention. We expose both the
    reported value and its absolute magnitude.
    """
    roots = np.asarray(results.roots)
    is_stable = bool(results.is_stable(verbose=False))
    rows = []
    for i, r in enumerate(roots):
        rows.append({
            'country':    country,
            'root_index': i,
            'root_real':  float(np.real(r)),
            'root_imag':  float(np.imag(r)),
            'abs_root':   float(np.abs(r)),
            'is_stable_overall': is_stable,
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 72
    print(bar)
    print('Phase 6 · Step 2 · S2 — VAR Estimation (p*=2) with D-030 exog')
    print(bar)
    print(f'lag order p*:   {VAR_P_STAR} (D-050 BIC-primary, locked)')
    print(f'trend:          {VAR_TREND!r}')
    print(f'endogenous:     {BASE_INDICATORS}')
    print(f'exog structure: 3 split + 2 period + 0..3 D-030 interactions')
    print(f'sample scope:   full (D-005 train/test reserved for S6)')
    print(f'LB lags:        {LB_LAGS}')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Part 1 — Load features.
    # ------------------------------------------------------------------
    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    # ------------------------------------------------------------------
    # Part 2 — Per-country estimation + diagnostics.
    # ------------------------------------------------------------------
    all_coefs:       list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_stability:   list[pd.DataFrame] = []
    exog_schema_rows: list[dict] = []
    fit_meta_rows:    list[dict] = []

    for country in MAIN_COUNTRIES:
        print(f'>>> {country}')
        endog, exog, info = extract_endog_exog(features[country], country)

        n_exog = exog.shape[1] if exog is not None else 0
        print(f'    n_obs           = {len(endog)}')
        print(f'    endog cols      = {len(endog.columns)}')
        print(f'    exog: split={len(info["split"])}  '
              f'period={len(info["period"])}  '
              f'interaction={len(info["interaction"])}  '
              f'total={n_exog}')
        if info['interaction']:
            print(f'    interactions    = {info["interaction"]}')

        # Fit
        try:
            results = fit_var(endog, exog, VAR_P_STAR, VAR_TREND)
        except Exception as exc:
            print(f'    !! fit failed: {type(exc).__name__}: {exc}')
            raise

        is_stable = bool(results.is_stable(verbose=False))
        llf       = float(results.llf)
        aic_fit   = float(results.aic)
        bic_fit   = float(results.bic)
        hqic_fit  = float(results.hqic)
        max_abs_root = float(np.max(np.abs(results.roots)))

        print(f'    log-likelihood  = {llf:+.4f}')
        print(f'    AIC / BIC / HQIC = '
              f'{aic_fit:+.4f} / {bic_fit:+.4f} / {hqic_fit:+.4f}')
        print(f'    max |root|      = {max_abs_root:.4f}  '
              f'(>1 required)  → is_stable = {is_stable}')

        # Coefficient long-form
        coefs = coefficients_long_form(
            results, country,
            endog_cols=list(endog.columns),
            exog_cols=list(exog.columns) if exog is not None else [],
        )
        coefs_path = (doc_dir /
                      f'phase6_step2_s2_var_coefficients_{country.lower()}.csv')
        coefs.to_csv(coefs_path, index=False)
        all_coefs.append(coefs)

        # Residual diagnostics
        diag = residual_diagnostics(results, country, list(endog.columns), LB_LAGS)
        all_diagnostics.append(diag)

        # Stability
        stab = stability_report(results, country)
        all_stability.append(stab)

        # Per-equation LB summary (one-line print)
        for eq in endog.columns:
            row = diag[diag['equation'] == eq].iloc[0]
            flag12 = '*' if row['lb_reject_wn_lag12_5pct'] else ' '
            flag24 = '*' if row['lb_reject_wn_lag24_5pct'] else ' '
            print(f'      {eq:<22s}  LB(12) p={row["lb_pvalue_lag12"]:.4f}{flag12}'
                  f'   LB(24) p={row["lb_pvalue_lag24"]:.4f}{flag24}'
                  f'   σ={row["resid_std"]:.4f}')
        print(f'    written: {coefs_path.name}')
        print()

        exog_schema_rows.append({
            'country': country,
            'n_split':       len(info['split']),
            'n_period':      len(info['period']),
            'n_interaction': len(info['interaction']),
            'n_exog_total':  n_exog,
            'exog_cols':     ';'.join(info['all']),
        })
        fit_meta_rows.append({
            'country':          country,
            'n_obs':            int(len(endog)),
            'p_star':           VAR_P_STAR,
            'n_endog':          len(endog.columns),
            'n_exog':           n_exog,
            'log_likelihood':   llf,
            'aic':              aic_fit,
            'bic':              bic_fit,
            'hqic':             hqic_fit,
            'max_abs_root':     max_abs_root,
            'is_stable':        is_stable,
            'lb12_any_reject':  bool(diag['lb_reject_wn_lag12_5pct'].any()),
            'lb24_any_reject':  bool(diag['lb_reject_wn_lag24_5pct'].any()),
        })

    # ------------------------------------------------------------------
    # Part 3 — Write consolidated CSVs.
    # ------------------------------------------------------------------
    schema_df = pd.DataFrame(exog_schema_rows)
    schema_path = doc_dir / 'phase6_step2_s2_exog_schema.csv'
    schema_df.to_csv(schema_path, index=False)

    diag_all = pd.concat(all_diagnostics, ignore_index=True)
    diag_path = doc_dir / 'phase6_step2_s2_var_diagnostics.csv'
    diag_all.to_csv(diag_path, index=False)

    stab_all = pd.concat(all_stability, ignore_index=True)
    stab_path = doc_dir / 'phase6_step2_s2_var_stability.csv'
    stab_all.to_csv(stab_path, index=False)

    meta_df = pd.DataFrame(fit_meta_rows)
    meta_path = doc_dir / 'phase6_step2_s2_fit_summary.csv'
    meta_df.to_csv(meta_path, index=False)

    # ------------------------------------------------------------------
    # Part 4 — Console summary.
    # ------------------------------------------------------------------
    print(bar)
    print('Fit summary (cross-country)')
    print(bar)
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(meta_df[['country', 'n_obs', 'n_exog',
                       'log_likelihood', 'aic', 'bic',
                       'max_abs_root', 'is_stable',
                       'lb12_any_reject', 'lb24_any_reject']]
              .to_string(index=False))
    print()

    print(bar)
    print('Residual Ljung-Box p-values  (★ = reject white noise at 5%)')
    print(bar)
    pvt = diag_all.pivot(index='equation', columns='country',
                         values='lb_pvalue_lag12')
    with pd.option_context('display.float_format', lambda v: f'{v:.4f}'):
        for eq in pvt.index:
            cells = []
            for c in MAIN_COUNTRIES:
                v = pvt.loc[eq, c] if c in pvt.columns else np.nan
                mark = '★' if v < 0.05 else ' '
                cells.append(f'{c}: {v:.4f}{mark}')
            print(f'  LB(12)  {eq:<22s}  ' + '   '.join(cells))
    print()

    print(bar)
    print('Output artefacts written:')
    for p in [schema_path, diag_path, stab_path, meta_path]:
        print(f'  data/documentation/{p.name}')
    for country in MAIN_COUNTRIES:
        print(f'  data/documentation/'
              f'phase6_step2_s2_var_coefficients_{country.lower()}.csv')
    print()

    # ------------------------------------------------------------------
    # Part 5 — Next sub-step pointer.
    # ------------------------------------------------------------------
    print(bar)
    all_stable = all(r['is_stable'] for r in fit_meta_rows)
    any_lb_reject = any(r['lb12_any_reject'] or r['lb24_any_reject']
                        for r in fit_meta_rows)
    print('System stability: '
          + ('ALL stable ✓' if all_stable
             else 'SOME UNSTABLE — investigate before S3'))
    print('Residual whiteness: '
          + ('all equations pass LB at 5% ✓' if not any_lb_reject
             else 'at least one equation shows residual autocorrelation'))
    print()
    print('Next sub-step: S3 = Granger causality battery on the fitted VARs.')
    print('If any LB rejection at lag 12/24 is concentrated in one equation,')
    print('consider a sensitivity fit at AIC-selected p (per D-050 draft).')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
