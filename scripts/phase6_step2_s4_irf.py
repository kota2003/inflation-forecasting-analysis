"""
scripts/phase6_step2_s4_irf.py
===============================
Phase 6 · Step 2 · S4 — Impulse Response Functions with asymptotic CI.

Purpose
-------
Compute orthogonalized impulse response functions (IRF) with 95%
asymptotic delta-method confidence intervals for each country's
VAR(p*) fit where p* is the D-050 AIC-primary lag order (USA/UK/GER
= 12, JPN = 5). Focus deliverable: the M2→CPI and POLICY_RATE→CPI
dynamic profiles that anchor the N2 Monetary Policy Lag narrative.

S3 Granger established the "who affects whom" binary structure.
S4 quantifies the "how long, how strong, with what persistence"
dimensions that Granger cannot reveal.

Confidence-interval methodology (revised from initial bootstrap spec)
--------------------------------------------------------------------
Initial script spec specified Monte Carlo bootstrap CI via statsmodels
``IRAnalysis.errband_mc``. Diagnostic inspection of first-run output
revealed that method returns ``(lower, upper)`` tuples with
``lower == upper`` across all 2 500 (horizon × shock × response)
cells — a statsmodels version-incompatibility issue where both
tuple elements bind to the same underlying array.

Revised methodology uses ``IRAnalysis.stderr(orth=True)`` which
returns asymptotic delta-method standard errors that are:
  - well-defined (CI contains the point estimate by construction);
  - fast (no bootstrap refitting);
  - standard in applied VAR practice (Lütkepohl 2005 Ch. 3.7).

CI bands: point ± Z * SE where Z = 1.96 for 95% coverage.

The Gaussian-residual assumption underlying delta-method SE is in
tension with the D-051 partial residual whitening caveat; a proper
residual-bootstrap CI is reserved as Phase 7 sensitivity.

Cholesky ordering (D-054 candidate)
-----------------------------------
    [GDP, UNEMPLOYMENT, CPI, POLICY_RATE, M2]

Rationale (Bernanke-Blinder 1992; Stock-Watson 2001 convention):
    1. GDP              — real economy predetermined within a month
    2. UNEMPLOYMENT     — natural-rate dynamics, slow adjustment
    3. CPI              — prices respond to output / slack
    4. POLICY_RATE      — Taylor-rule feedback on π + output
    5. M2               — endogenous money supply responds to all above

The VAR base feature matrix columns are stored in the order
[CPI, POLICY_RATE, UNEMPLOYMENT, GDP, M2] (per src.INDICATORS). We
reorder the endogenous block to the D-054 specification before
fitting so the Cholesky decomposition is applied in the intended
economic ordering.

Scope boundaries (S4 only)
--------------------------
- Orthogonalized IRF (Cholesky) with asymptotic delta-method CI.
- Horizon = 24 months (covers typical monetary transmission lag
  of 12–18 months plus a generous observation margin).
- Output focuses on CPI responses; full matrix is written for audit.
- No FEVD (S5), no OOS forecast (S6).

Output artefacts
----------------
data/documentation/
    phase6_step2_s4_irf_full_matrix.csv
        Long form: horizon × shock × response × country.
        4 countries × 25 horizons × 5 shocks × 5 responses = 2500 rows.
        cols: country, horizon, shock, response, orth_irf,
              orth_se, ci_lo, ci_up, cum_irf, cum_se,
              cum_ci_lo, cum_ci_up.
    phase6_step2_s4_irf_cpi_responses.csv
        Filter to response=CPI (4 shocks × 25 h × 4 countries = 400 rows).
        This is the N2-narrative-ready CSV.
    phase6_step2_s4_irf_peak_summary.csv
        Per-country × per-shock-to-CPI peak summary (16 rows):
        country, shock, peak_horizon, peak_irf, peak_se, peak_ci_lo,
        peak_ci_up, trough_horizon, trough_irf,
        pct_horizons_ci_excludes_zero, narrative_label.

Decisions referenced
--------------------
D-030  Dominant-driver matrix — regime-exog carries forward to IRF.
D-050  AIC-primary VAR lag selection (confirmed in S2b).
D-051  Partial residual whitening caveat — Gaussian-SE approximate.
D-052/53 (candidates) Granger triangulation + correlation-vs-Granger
       methodology echoes. IRF provides direct cross-reference for
       the M2→CPI Granger-null finding.
D-054  (candidate) Cholesky ordering [GDP, UE, CPI, PR, M2].

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s4_irf.py

Runtime expectation: seconds (asymptotic SE, no bootstrap refitting).
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


# ── Constants ─────────────────────────────────────────────────────────

#: D-050 AIC-primary lag orders per country.
P_PER_COUNTRY: dict[str, int] = {
    'USA':     12,
    'JAPAN':    5,
    'UK':      12,
    'GERMANY': 12,
}

#: D-054 candidate — Cholesky ordering (slow-to-fast response).
CHOLESKY_ORDER: list[str] = [
    'GDP', 'UNEMPLOYMENT', 'CPI', 'POLICY_RATE', 'M2',
]

#: IRF horizon in months. 24 covers typical 12–18 month monetary
#: transmission lag plus observation margin.
IRF_HORIZON: int = 24

#: Confidence level for delta-method CI. Z = 1.96 gives 95% coverage
#: under Gaussian asymptotic theory. Z = 2.576 would give 99%.
CI_CONFIDENCE: float = 0.95
Z_CRITICAL: float = 1.96

VAR_TREND: str = 'c'
BASE_INDICATORS: list[str] = list(INDICATORS)
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())
PERIOD_KEYS: list[str] = ['GFC', 'COVID']

#: Narrative labels for CPI-response panel.
CPI_SHOCK_NARRATIVE: dict[str, str] = {
    'POLICY_RATE':   'N2 · Monetary Policy Lag (direct channel)',
    'M2':            'N2 · Quantity Theory of Money',
    'UNEMPLOYMENT':  'N1 · Phillips Curve',
    'GDP':           'Demand-side inflation',
}


# ── Exog / endog discovery ───────────────────────────────────────────

def build_exog_column_list(country: str,
                           features_cols: list[str]) -> list[str]:
    split_cols  = [f'{country}_D_{b}' for b in SPLIT_BREAK_NAMES]
    period_cols = [f'{country}_P_{p}' for p in PERIOD_KEYS]
    interaction_cols: list[str] = []
    for (c, break_name), driver in PHASE6_REGIME_SPEC.items():
        if c != country:
            continue
        if driver is None or driver == 'const':
            continue
        interaction_cols.append(f'{country}_{driver}_x_D_{break_name}')
    all_cols = split_cols + period_cols + interaction_cols
    missing = [c for c in all_cols if c not in features_cols]
    if missing:
        raise KeyError(f"{country}: missing exog columns {missing}")
    return all_cols


def extract_endog_exog_cholesky(features_df: pd.DataFrame,
                                country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return endog ordered by CHOLESKY_ORDER (D-054) and exog aligned."""
    endog_cols_chol = [f'{country}_{ind}' for ind in CHOLESKY_ORDER]
    exog_cols = build_exog_column_list(country, list(features_df.columns))
    joint = features_df[endog_cols_chol + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    return joint[endog_cols_chol].copy(), joint[exog_cols].copy()


def fit_var(endog, exog, p, trend):
    return VAR(endog, exog=exog).fit(maxlags=p, trend=trend)


# ── IRF extraction ───────────────────────────────────────────────────

def compute_irf_with_ci(results, horizon: int,
                        z_critical: float) -> dict:
    """Compute orthogonalized IRF and asymptotic delta-method CI band.

    Returns a dict:
        orth     : np.ndarray (horizon+1, n, n) — point estimate
        orth_se  : np.ndarray (horizon+1, n, n) — asymptotic SE
        ci_lo    : np.ndarray (horizon+1, n, n) — lower bound
        ci_up    : np.ndarray (horizon+1, n, n) — upper bound
        cum      : np.ndarray (horizon+1, n, n) — cumulative orth IRF
        cum_se   : np.ndarray (horizon+1, n, n) — cumulative IRF SE
        cum_lo   : np.ndarray (horizon+1, n, n) — cum CI lower
        cum_up   : np.ndarray (horizon+1, n, n) — cum CI upper

    Notes
    -----
    Methodology switched from Monte Carlo bootstrap (errband_mc) to
    asymptotic delta-method (stderr / cum_effect_stderr) after
    observing that errband_mc returns identical lower and upper
    arrays in the installed statsmodels version. See module docstring.
    """
    irf_obj = results.irf(horizon)

    # Point estimates
    orth = irf_obj.orth_irfs                  # (T+1, n, n)
    cum  = irf_obj.orth_cum_effects           # (T+1, n, n)

    # Asymptotic delta-method standard errors
    orth_se = irf_obj.stderr(orth=True)       # (T+1, n, n)
    cum_se  = irf_obj.cum_effect_stderr(orth=True)

    ci_lo   = orth - z_critical * orth_se
    ci_up   = orth + z_critical * orth_se
    cum_lo  = cum  - z_critical * cum_se
    cum_up  = cum  + z_critical * cum_se

    return {
        'orth':    orth,
        'orth_se': orth_se,
        'ci_lo':   ci_lo,
        'ci_up':   ci_up,
        'cum':     cum,
        'cum_se':  cum_se,
        'cum_lo':  cum_lo,
        'cum_up':  cum_up,
    }


def build_irf_long(irf_data: dict, country: str,
                   endog_order: list[str]) -> pd.DataFrame:
    """Flatten (horizon, shock, response) tensors to long-form DataFrame."""
    rows: list[dict] = []
    horizon_axis = irf_data['orth'].shape[0]  # horizon + 1
    for h in range(horizon_axis):
        for j, shock in enumerate(endog_order):   # column index = shock
            for i, response in enumerate(endog_order):  # row index = response
                rows.append({
                    'country':    country,
                    'horizon':    h,
                    'shock':      shock,
                    'response':   response,
                    'orth_irf':   float(irf_data['orth'][h, i, j]),
                    'orth_se':    float(irf_data['orth_se'][h, i, j]),
                    'ci_lo':      float(irf_data['ci_lo'][h, i, j]),
                    'ci_up':      float(irf_data['ci_up'][h, i, j]),
                    'cum_irf':    float(irf_data['cum'][h, i, j]),
                    'cum_se':     float(irf_data['cum_se'][h, i, j]),
                    'cum_ci_lo':  float(irf_data['cum_lo'][h, i, j]),
                    'cum_ci_up':  float(irf_data['cum_up'][h, i, j]),
                })
    return pd.DataFrame(rows)


def build_peak_summary(irf_long: pd.DataFrame) -> pd.DataFrame:
    """Per-(country × shock → CPI) peak/trough summary."""
    rows: list[dict] = []
    for country in MAIN_COUNTRIES:
        for shock in CPI_SHOCK_NARRATIVE.keys():
            sub = irf_long[(irf_long['country']  == country) &
                           (irf_long['shock']    == shock) &
                           (irf_long['response'] == 'CPI') &
                           (irf_long['horizon']  >= 1)]
            if sub.empty:
                continue
            peak_row = sub.loc[sub['orth_irf'].abs().idxmax()]
            trough_row = sub.loc[sub['orth_irf'].idxmin()]

            # CI exclusion rate: how many horizons (1..H) have CI not
            # straddling zero? Informal proxy for "significant" IRF.
            ci_excl_zero = sub[
                (sub['ci_lo'] > 0) | (sub['ci_up'] < 0)
            ]
            pct_excl = 100.0 * len(ci_excl_zero) / len(sub)

            rows.append({
                'country':             country,
                'shock':               shock,
                'narrative_label':     CPI_SHOCK_NARRATIVE[shock],
                'peak_horizon':        int(peak_row['horizon']),
                'peak_irf':            float(peak_row['orth_irf']),
                'peak_se':             float(peak_row['orth_se']),
                'peak_ci_lo':          float(peak_row['ci_lo']),
                'peak_ci_up':          float(peak_row['ci_up']),
                'trough_horizon':      int(trough_row['horizon']),
                'trough_irf':          float(trough_row['orth_irf']),
                'pct_horizons_ci_excludes_zero': round(pct_excl, 1),
            })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 80
    print(bar)
    print('Phase 6 · Step 2 · S4 — Impulse Response Functions (asymptotic CI)')
    print(bar)
    print(f'lag orders:         {P_PER_COUNTRY}')
    print(f'Cholesky order:     {CHOLESKY_ORDER}  (D-054 candidate)')
    print(f'IRF horizon:        {IRF_HORIZON} months')
    print(f'CI method:          asymptotic delta-method (stderr)')
    print(f'CI coverage:        {CI_CONFIDENCE * 100:.0f}%   '
          f'(Z = {Z_CRITICAL})')
    print(f'caveat:             Gaussian-residual assumption vs D-051; '
          f'bootstrap sensitivity deferred to Phase 7')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    all_irf_long: list[pd.DataFrame] = []

    for country in MAIN_COUNTRIES:
        p_star = P_PER_COUNTRY[country]
        print(f'>>> {country}  p* = {p_star}  — fit VAR → IRF → asymptotic SE ...')
        endog, exog = extract_endog_exog_cholesky(features[country], country)
        results = fit_var(endog, exog, p_star, VAR_TREND)

        irf_data = compute_irf_with_ci(results, IRF_HORIZON, Z_CRITICAL)
        irf_long = build_irf_long(irf_data, country, CHOLESKY_ORDER)
        all_irf_long.append(irf_long)

        # Quick per-country preview: POLICY_RATE and M2 shocks on CPI
        for shock in ['POLICY_RATE', 'M2', 'UNEMPLOYMENT', 'GDP']:
            sub = irf_long[(irf_long['shock'] == shock) &
                           (irf_long['response'] == 'CPI') &
                           (irf_long['horizon'] >= 1) &
                           (irf_long['horizon'] <= 18)]
            if sub.empty:
                continue
            peak_row = sub.loc[sub['orth_irf'].abs().idxmax()]
            ci_excl = sub[(sub['ci_lo'] > 0) | (sub['ci_up'] < 0)]
            pct_excl = 100.0 * len(ci_excl) / len(sub)
            print(f'    {shock:<14s} → CPI   '
                  f'peak at h={int(peak_row["horizon"]):2d}  '
                  f'IRF={peak_row["orth_irf"]:+.4f}   '
                  f'CI=[{peak_row["ci_lo"]:+.4f}, {peak_row["ci_up"]:+.4f}]   '
                  f'%h:CI≠0 = {pct_excl:4.1f}%')
        print()

    # ------------------------------------------------------------------
    # Consolidated CSVs
    # ------------------------------------------------------------------
    full_irf = pd.concat(all_irf_long, ignore_index=True)
    full_path = doc_dir / 'phase6_step2_s4_irf_full_matrix.csv'
    full_irf.to_csv(full_path, index=False)

    cpi_resp = full_irf[full_irf['response'] == 'CPI'].copy()
    cpi_path = doc_dir / 'phase6_step2_s4_irf_cpi_responses.csv'
    cpi_resp.to_csv(cpi_path, index=False)

    peak_summary = build_peak_summary(full_irf)
    peak_path = doc_dir / 'phase6_step2_s4_irf_peak_summary.csv'
    peak_summary.to_csv(peak_path, index=False)

    # ------------------------------------------------------------------
    # Narrative panel print
    # ------------------------------------------------------------------
    print(bar)
    print('Peak IRF summary — 4 shocks × 4 countries → CPI')
    print(bar)
    print(f'{"country":<10s}  {"shock":<14s}  {"narrative":<42s}  '
          f'{"h*":>4s}  {"peak":>9s}  {"CI low":>9s}  {"CI up":>9s}  {"%≠0":>5s}')
    for country in MAIN_COUNTRIES:
        for shock in ['POLICY_RATE', 'M2', 'UNEMPLOYMENT', 'GDP']:
            rec = peak_summary[(peak_summary['country'] == country) &
                               (peak_summary['shock'] == shock)]
            if rec.empty:
                continue
            r = rec.iloc[0]
            print(f'{country:<10s}  {shock:<14s}  '
                  f'{r["narrative_label"]:<42s}  '
                  f'{int(r["peak_horizon"]):>4d}  '
                  f'{r["peak_irf"]:>+9.4f}  '
                  f'{r["peak_ci_lo"]:>+9.4f}  '
                  f'{r["peak_ci_up"]:>+9.4f}  '
                  f'{r["pct_horizons_ci_excludes_zero"]:>5.1f}')
    print()

    # ------------------------------------------------------------------
    # Written artefacts + next pointer
    # ------------------------------------------------------------------
    print(bar)
    print('Output artefacts written:')
    for p in [full_path, cpi_path, peak_path]:
        print(f'  data/documentation/{p.name}')
    print()

    print(bar)
    print('Interpretation columns:')
    print('  peak_horizon   = month at which |IRF| is largest (1..18 window)')
    print('  peak_irf       = the IRF value at that horizon (signed)')
    print('  CI low / CI up = 95% bootstrap band around peak_irf')
    print('  %≠0            = fraction of horizons 1..18 whose CI excludes')
    print('                   zero — an informal "significance" proxy; '
          'cross-references')
    print('                   S3 Granger for consistency checking.')
    print()
    print('Next sub-step: S5 = Forecast Error Variance Decomposition.')
    print('Final sub-step: S6 = OOS forecast (D-005 train/test split).')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
