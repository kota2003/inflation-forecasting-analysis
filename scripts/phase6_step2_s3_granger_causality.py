"""
scripts/phase6_step2_s3_granger_causality.py
=============================================
Phase 6 · Step 2 · S3 — Granger causality battery on AIC-p VARs.

Purpose
-------
Run the complete Granger causality battery on each country's VAR(p*)
fit where p* is the D-050 revised AIC-primary selection (USA/UK/GER
= 12, JPN = 5). Reports the 5×5 causation matrix per country and
extracts CPI-as-target panels for the N1 Phillips Curve and N2
Monetary Policy Lag narratives.

Null hypothesis per test
------------------------
    H0: lags of {causer} do NOT enter the {caused} equation
        (i.e., causer does not Granger-cause caused at the selected lag)

Rejection at α ∈ {0.05, 0.01} taken from statsmodels VARResults
.test_causality (Wald test with standard VAR covariance). HAC-robust
sensitivity is deferred to Phase 7 per D-051 partial-whitening caveat.

Scope boundaries (S3 only)
--------------------------
- Granger causality only. No IRF (S4), FEVD (S5), OOS forecast (S6).
- Uses VAR at AIC-selected p per country (D-050 primary).
- Full 5×5 matrix per country (25 tests) — diagonal is trivial
  self-cause and is reported for completeness, flagged as such.
- Total non-trivial tests: 20 off-diagonal × 4 countries = 80 tests.
- Focus panel: 4 causers → CPI × 4 countries = 16 primary narrative tests.

Output artefacts
----------------
data/documentation/
    phase6_step2_s3_granger_full_matrix.csv
        100 rows × 9 cols
        (country, causer, caused, test_stat, p_value, df1, df2,
         signif_5pct, signif_1pct, is_diagonal)
    phase6_step2_s3_granger_cpi_receivers.csv
        16 rows × 8 cols — focused CPI-as-target panel
        (country, causer, p_value, signif_5pct, signif_1pct,
         narrative_label, test_stat, df1)
    phase6_step2_s3_granger_country_summary.csv
        4 rows × 8 cols — per-country summary
        (country, p_star, n_tests_total, n_tests_nontrivial,
         n_sig_5pct, n_sig_1pct, pct_sig_5pct, pct_sig_1pct)

Narrative labels (for CPI receivers panel)
------------------------------------------
    POLICY_RATE → CPI    : N2 Monetary Policy Lag (direct channel)
    M2 → CPI             : N2 Quantity Theory of Money
    UNEMPLOYMENT → CPI   : N1 Phillips Curve
    GDP → CPI            : Demand-side inflation

Decisions referenced
--------------------
D-004  Three-layer modelling architecture.
D-030  Dominant-driver matrix (exog preserves regime structure).
D-050  AIC-primary VAR lag selection (revised, confirmed in S2b).
D-051  (candidate) Partial residual whitening caveat — approximate SEs.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s3_granger_causality.py
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

#: AIC-primary lag orders (D-050 revised, from S2b).
P_PER_COUNTRY: dict[str, int] = {
    'USA':     12,
    'JAPAN':    5,
    'UK':      12,
    'GERMANY': 12,
}

VAR_TREND: str = 'c'
BASE_INDICATORS: list[str] = list(INDICATORS)
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())
PERIOD_KEYS: list[str] = ['GFC', 'COVID']

#: Significance thresholds.
ALPHA_MAIN:   float = 0.05
ALPHA_STRICT: float = 0.01

#: Narrative labels for the CPI-receivers panel.
CPI_CAUSER_NARRATIVE: dict[str, str] = {
    'POLICY_RATE':   'N2 · Monetary Policy Lag (direct channel)',
    'M2':            'N2 · Quantity Theory of Money',
    'UNEMPLOYMENT':  'N1 · Phillips Curve',
    'GDP':           'Demand-side inflation',
}


# ── Exog discovery — identical to S2/S2b ─────────────────────────────

def build_exog_column_list(country: str,
                           features_cols: list[str]) -> list[str]:
    """Return all exogenous column names (split + period + interactions)."""
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


def extract_endog_exog(features_df: pd.DataFrame,
                       country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    endog_cols = [f'{country}_{ind}' for ind in BASE_INDICATORS]
    exog_cols = build_exog_column_list(country, list(features_df.columns))
    joint = features_df[endog_cols + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    return joint[endog_cols].copy(), joint[exog_cols].copy()


def fit_var(endog, exog, p, trend):
    return VAR(endog, exog=exog).fit(maxlags=p, trend=trend)


# ── Granger battery ──────────────────────────────────────────────────

def granger_test_pair(results, causer_col: str, caused_col: str) -> dict:
    """Run VARResults.test_causality and unwrap diagnostics.

    VARResults.test_causality returns a CausalityTestResults object
    carrying:
        .test_statistic  (float, Wald F or chi2 depending on signif)
        .pvalue          (float)
        .df              (int or tuple)
        .crit_value      (float, at 5%)

    We additionally record whether rejection holds at α=0.05 and 0.01.
    """
    # Default kind='f' returns F-test; more robust for finite samples.
    ctr = results.test_causality(
        caused=caused_col, causing=causer_col, kind='f', signif=ALPHA_MAIN
    )
    test_stat = float(ctr.test_statistic)
    p_value = float(ctr.pvalue)
    df = ctr.df  # may be scalar or tuple

    if isinstance(df, tuple):
        df1, df2 = int(df[0]), int(df[1])
    else:
        df1, df2 = int(df), -1

    return {
        'test_stat': test_stat,
        'p_value':   p_value,
        'df1':       df1,
        'df2':       df2,
        'signif_5pct': bool(p_value < ALPHA_MAIN),
        'signif_1pct': bool(p_value < ALPHA_STRICT),
    }


def build_granger_matrix(results, country: str,
                         endog_cols: list[str]) -> pd.DataFrame:
    """Run full 5×5 Granger battery. Diagonal is flagged trivially."""
    rows: list[dict] = []
    for causer in endog_cols:
        causer_ind = causer.replace(f'{country}_', '')
        for caused in endog_cols:
            caused_ind = caused.replace(f'{country}_', '')
            if causer == caused:
                # Self-cause — trivial; emit diagonal flag without test
                rows.append({
                    'country':     country,
                    'causer':      causer_ind,
                    'caused':      caused_ind,
                    'test_stat':   np.nan,
                    'p_value':     np.nan,
                    'df1':         -1,
                    'df2':         -1,
                    'signif_5pct': False,
                    'signif_1pct': False,
                    'is_diagonal': True,
                })
                continue
            r = granger_test_pair(results, causer, caused)
            rows.append({
                'country':     country,
                'causer':      causer_ind,
                'caused':      caused_ind,
                **r,
                'is_diagonal': False,
            })
    return pd.DataFrame(rows)


def build_cpi_receivers_panel(matrix: pd.DataFrame) -> pd.DataFrame:
    """Extract 4-causer → CPI panel for N1/N2 narratives."""
    panel = matrix[
        (matrix['caused'] == 'CPI') & (~matrix['is_diagonal'])
    ].copy()
    panel['narrative_label'] = panel['causer'].map(CPI_CAUSER_NARRATIVE)
    return panel[['country', 'causer', 'narrative_label',
                  'test_stat', 'df1', 'df2',
                  'p_value', 'signif_5pct', 'signif_1pct']]


def build_country_summary(matrix: pd.DataFrame,
                          p_star_map: dict[str, int]) -> pd.DataFrame:
    """Per-country summary: total tests, significant count, percentages."""
    rows = []
    for country in MAIN_COUNTRIES:
        sub = matrix[matrix['country'] == country]
        non_trivial = sub[~sub['is_diagonal']]
        n_total = len(sub)
        n_nontriv = len(non_trivial)
        n_5 = int(non_trivial['signif_5pct'].sum())
        n_1 = int(non_trivial['signif_1pct'].sum())
        rows.append({
            'country':             country,
            'p_star':              p_star_map[country],
            'n_tests_total':       n_total,
            'n_tests_nontrivial':  n_nontriv,
            'n_sig_5pct':          n_5,
            'n_sig_1pct':          n_1,
            'pct_sig_5pct':        round(100.0 * n_5 / n_nontriv, 1),
            'pct_sig_1pct':        round(100.0 * n_1 / n_nontriv, 1),
        })
    return pd.DataFrame(rows)


# ── Console-friendly 5×5 heatmap ─────────────────────────────────────

def format_country_heatmap(matrix: pd.DataFrame, country: str) -> str:
    """ASCII-art 5×5 Granger p-value matrix. ★ = reject at 5%, ★★ = 1%."""
    sub = matrix[matrix['country'] == country]
    indicators = BASE_INDICATORS
    header = f'  cause↓ \\ effect→   ' + '  '.join(
        f'{ind[:11]:>11s}' for ind in indicators
    )
    lines = [header]
    for causer in indicators:
        row_cells: list[str] = []
        for caused in indicators:
            rec = sub[(sub['causer'] == causer) & (sub['caused'] == caused)]
            if rec.empty:
                row_cells.append('      —    ')
                continue
            row = rec.iloc[0]
            if row['is_diagonal']:
                row_cells.append('      —    ')
            else:
                pv = float(row['p_value'])
                mark = '★★ ' if row['signif_1pct'] else ('★  ' if row['signif_5pct'] else '   ')
                row_cells.append(f' {pv:8.4f}{mark}')
        lines.append(f'  {causer[:15]:<15s}    ' + '  '.join(row_cells))
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 80
    print(bar)
    print('Phase 6 · Step 2 · S3 — Granger Causality Battery (AIC-p VARs)')
    print(bar)
    print('Per-country lag orders (D-050 primary, confirmed in S2b):')
    for c in MAIN_COUNTRIES:
        print(f'    {c:<10s} p* = {P_PER_COUNTRY[c]}')
    print(f'test kind:      F (finite-sample standard Wald)')
    print(f'significance:   main α = {ALPHA_MAIN}; strict α = {ALPHA_STRICT}')
    print(f'caveat:         D-051 partial whitening → SEs approximate;')
    print(f'                HAC-robust sensitivity deferred to Phase 7.')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    all_matrices: list[pd.DataFrame] = []

    for country in MAIN_COUNTRIES:
        p_star = P_PER_COUNTRY[country]
        print(f'>>> {country}  p* = {p_star}  — fitting VAR and running 5×5 battery ...')
        endog, exog = extract_endog_exog(features[country], country)
        results = fit_var(endog, exog, p_star, VAR_TREND)
        matrix = build_granger_matrix(
            results, country, list(endog.columns)
        )
        all_matrices.append(matrix)

        # Per-country heatmap
        print()
        print(format_country_heatmap(matrix, country))
        sub = matrix[matrix['country'] == country]
        sub_nd = sub[~sub['is_diagonal']]
        print(f'  → {country} non-trivial tests = {len(sub_nd)}  '
              f'sig@5% = {int(sub_nd["signif_5pct"].sum())}  '
              f'sig@1% = {int(sub_nd["signif_1pct"].sum())}')
        print()

    # ------------------------------------------------------------------
    # Consolidated outputs
    # ------------------------------------------------------------------
    full_matrix = pd.concat(all_matrices, ignore_index=True)
    full_path = doc_dir / 'phase6_step2_s3_granger_full_matrix.csv'
    full_matrix.to_csv(full_path, index=False)

    cpi_panel = build_cpi_receivers_panel(full_matrix)
    cpi_path = doc_dir / 'phase6_step2_s3_granger_cpi_receivers.csv'
    cpi_panel.to_csv(cpi_path, index=False)

    country_summary = build_country_summary(full_matrix, P_PER_COUNTRY)
    summary_path = doc_dir / 'phase6_step2_s3_granger_country_summary.csv'
    country_summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------------
    # CPI receivers panel — THE narrative centerpiece
    # ------------------------------------------------------------------
    print(bar)
    print('CPI-as-target panel (N1 Phillips + N2 Monetary Policy Lag)')
    print(bar)
    print(f'{"country":<10s}  {"causer":<14s}  {"narrative":<42s}  '
          f'{"p-value":>9s}  {"sig":<4s}')
    for country in MAIN_COUNTRIES:
        for causer in ['POLICY_RATE', 'M2', 'UNEMPLOYMENT', 'GDP']:
            rec = cpi_panel[(cpi_panel['country'] == country) &
                            (cpi_panel['causer'] == causer)]
            if rec.empty:
                continue
            row = rec.iloc[0]
            pv = row['p_value']
            sig = '★★' if row['signif_1pct'] else ('★' if row['signif_5pct'] else '')
            label = CPI_CAUSER_NARRATIVE[causer]
            print(f'{country:<10s}  {causer:<14s}  {label:<42s}  '
                  f'{pv:9.4f}  {sig:<4s}')
    print()

    # ------------------------------------------------------------------
    # Country summary panel
    # ------------------------------------------------------------------
    print(bar)
    print('Per-country significance summary (off-diagonal tests only)')
    print(bar)
    with pd.option_context('display.max_columns', None,
                           'display.width', 180,
                           'display.float_format', lambda v: f'{v:.1f}'):
        print(country_summary.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Artefact list + forward pointer
    # ------------------------------------------------------------------
    print(bar)
    print('Output artefacts written:')
    for p in [full_path, cpi_path, summary_path]:
        print(f'  data/documentation/{p.name}')
    print()

    print(bar)
    total_sig5 = int(country_summary['n_sig_5pct'].sum())
    total_non = int(country_summary['n_tests_nontrivial'].sum())
    total_sig1 = int(country_summary['n_sig_1pct'].sum())
    print(f'OVERALL: {total_sig5}/{total_non} non-trivial Granger tests sig@5%  '
          f'({100.0 * total_sig5 / total_non:.1f}%)')
    print(f'         {total_sig1}/{total_non} non-trivial Granger tests sig@1%  '
          f'({100.0 * total_sig1 / total_non:.1f}%)')
    print()
    print('Next sub-step: S4 = Impulse Response Functions with 95% bootstrap CI')
    print('(focus: M2 → CPI, POLICY_RATE → CPI for N2 Monetary Policy Lag).')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
