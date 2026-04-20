"""
scripts/phase6_step2_s5_fevd.py
================================
Phase 6 · Step 2 · S5 — Forecast Error Variance Decomposition (FEVD).

Purpose
-------
Decompose the forecast-error variance of each endogenous variable
into proportions explained by orthogonalized shocks to each variable
in the system, at a grid of horizons {1, 3, 6, 12, 24}. Focus
deliverable: the CPI-as-target FEVD panel that quantifies
"how much of inflation forecast variability is explained by each
driver" — a complementary metric to S4 IRF which traces the
dynamic response shape rather than the variance share.

S3 Granger answered "who Granger-causes whom?" (binary).
S4 IRF quantified "how does a shock propagate?" (dynamic profile).
S5 FEVD quantifies "how important is each shock for explaining
a target's forecast variance?" (variance-share).

Together the three provide a complete inference triangle for
the N1 Phillips / N2 Monetary Policy Lag / N3 Japan Isolation
narratives.

Cholesky ordering (D-054 candidate — same as S4)
------------------------------------------------
    [GDP, UNEMPLOYMENT, CPI, POLICY_RATE, M2]

FEVD semantics
--------------
Let `decomp[h, i, j]` = fraction of variable i's forecast error
variance at horizon h that is attributable to shocks to variable j.
By construction:
    sum over j of decomp[h, i, j]  =  1  (100%)  for every (h, i).

Focus observations
------------------
For the CPI row (i = CPI):
  - decomp[h, CPI, CPI]        = self-explained share (auto-correlation)
  - decomp[h, CPI, POLICY_RATE] = N2 direct-channel share
  - decomp[h, CPI, M2]         = N2 money-supply channel share
  - decomp[h, CPI, UNEMP]      = N1 Phillips share
  - decomp[h, CPI, GDP]        = demand-side share

N3 Japan Isolation prediction: Japan CPI self-share >> other countries.

Scope boundaries (S5 only)
--------------------------
- Orthogonalized FEVD (Cholesky); consistent with S4 D-054 ordering.
- Horizon grid: {1, 3, 6, 12, 24} months for the reporting panel.
- Full FEVD (h = 0..24) computed and saved for audit.
- No OOS forecast (S6).

Output artefacts
----------------
data/documentation/
    phase6_step2_s5_fevd_full_matrix.csv
        4 countries × 25 horizons × 5 responses × 5 shocks = 2500 rows.
        cols: country, horizon, response, shock, share.
    phase6_step2_s5_fevd_cpi_target.csv
        Focus on CPI-as-response across reporting horizons.
        4 countries × 5 horizons × 5 shocks = 100 rows.
        cols: country, horizon, shock, share, narrative_label.
    phase6_step2_s5_fevd_top_contributors.csv
        Per-country × per-reporting-horizon top-3 shock contributors
        to each variable's forecast error variance. 4 × 5 × 5 × 3 =
        300 rows.
    phase6_step2_s5_fevd_cpi_summary.csv
        Per-country × per-horizon CPI variance share summary:
        self-share + each driver's share + "other" (= 1 - self - drivers).
        4 countries × 5 horizons = 20 rows.

Decisions referenced
--------------------
D-050  AIC-primary VAR lag selection.
D-051  Partial residual whitening caveat — FEVD shares are point
       estimates; variance is deterministic given the VAR.
D-054  (candidate) Cholesky ordering [GDP, UE, CPI, PR, M2].
D-056/57 (candidates) IRF-layer narratives — FEVD provides the
       cross-variance anchor for the same narratives.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s5_fevd.py

Runtime: seconds (deterministic VAR→FEVD computation).
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

P_PER_COUNTRY: dict[str, int] = {
    'USA':     12,
    'JAPAN':    5,
    'UK':      12,
    'GERMANY': 12,
}

#: Cholesky ordering (identical to S4).
CHOLESKY_ORDER: list[str] = [
    'GDP', 'UNEMPLOYMENT', 'CPI', 'POLICY_RATE', 'M2',
]

#: Full FEVD horizon (h = 0..24). 25 horizons stored per cell for audit.
FEVD_MAX_HORIZON: int = 24

#: Reporting horizons for the narrative-ready panels.
REPORTING_HORIZONS: list[int] = [1, 3, 6, 12, 24]

VAR_TREND: str = 'c'
BASE_INDICATORS: list[str] = list(INDICATORS)
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())
PERIOD_KEYS: list[str] = ['GFC', 'COVID']

#: Narrative labels for CPI-receivers panel (same as S3/S4).
CPI_SHOCK_NARRATIVE: dict[str, str] = {
    'CPI':           'Self-share (auto-explanatory)',
    'POLICY_RATE':   'N2 · Monetary Policy Lag (direct channel)',
    'M2':            'N2 · Quantity Theory of Money',
    'UNEMPLOYMENT':  'N1 · Phillips Curve',
    'GDP':           'Demand-side inflation',
}


# ── Exog / endog helpers — identical to S4 ───────────────────────────

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
    endog_cols_chol = [f'{country}_{ind}' for ind in CHOLESKY_ORDER]
    exog_cols = build_exog_column_list(country, list(features_df.columns))
    joint = features_df[endog_cols_chol + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    return joint[endog_cols_chol].copy(), joint[exog_cols].copy()


def fit_var(endog, exog, p, trend):
    return VAR(endog, exog=exog).fit(maxlags=p, trend=trend)


# ── FEVD extraction ──────────────────────────────────────────────────

def compute_fevd(results, horizon: int) -> np.ndarray:
    """Return FEVD decomposition array of shape (horizon+1, n, n).

    decomp[h, i, j] = share of variable i's forecast error variance
    at horizon h explained by shocks to variable j.
    Rows (axis 1) sum to 1 along axis 2 for every (h, i).
    """
    fevd_obj = results.fevd(periods=horizon + 1)
    # fevd_obj.decomp is the orthogonalized variance decomposition.
    # Shape: (periods, neqs, neqs)
    decomp = np.asarray(fevd_obj.decomp)
    return decomp


def build_fevd_long(decomp: np.ndarray, country: str,
                    endog_order: list[str]) -> pd.DataFrame:
    """Flatten 3D decomposition to long form."""
    rows: list[dict] = []
    n_horizons = decomp.shape[0]
    for h in range(n_horizons):
        for i, response in enumerate(endog_order):
            for j, shock in enumerate(endog_order):
                rows.append({
                    'country':  country,
                    'horizon':  h,
                    'response': response,
                    'shock':    shock,
                    'share':    float(decomp[h, i, j]),
                })
    return pd.DataFrame(rows)


def build_cpi_target_panel(
    full_long: pd.DataFrame,
    reporting_horizons: list[int],
) -> pd.DataFrame:
    """CPI-as-target focused panel at the reporting horizons."""
    cpi = full_long[
        (full_long['response'] == 'CPI') &
        (full_long['horizon'].isin(reporting_horizons))
    ].copy()
    cpi['narrative_label'] = cpi['shock'].map(CPI_SHOCK_NARRATIVE)
    cpi = cpi[['country', 'horizon', 'shock', 'share', 'narrative_label']]
    return cpi.sort_values(['country', 'horizon', 'shock']).reset_index(drop=True)


def build_top_contributors(
    full_long: pd.DataFrame,
    reporting_horizons: list[int],
    top_k: int = 3,
) -> pd.DataFrame:
    """Top-K shock contributors per (country × horizon × response)."""
    rows = []
    sub = full_long[full_long['horizon'].isin(reporting_horizons)]
    for (country, h, response), g in sub.groupby(
        ['country', 'horizon', 'response']
    ):
        g_sorted = g.sort_values('share', ascending=False).head(top_k)
        for rank, (_, row) in enumerate(g_sorted.iterrows(), 1):
            rows.append({
                'country':  country,
                'horizon':  int(h),
                'response': response,
                'rank':     rank,
                'shock':    row['shock'],
                'share':    round(float(row['share']), 4),
            })
    return pd.DataFrame(rows)


def build_cpi_summary(
    full_long: pd.DataFrame,
    reporting_horizons: list[int],
) -> pd.DataFrame:
    """Per-country × per-horizon compact CPI variance breakdown."""
    rows: list[dict] = []
    cpi = full_long[
        (full_long['response'] == 'CPI') &
        (full_long['horizon'].isin(reporting_horizons))
    ]
    for (country, h), g in cpi.groupby(['country', 'horizon']):
        g = g.set_index('shock')['share']
        row = {
            'country':       country,
            'horizon':       int(h),
            'share_CPI_self':         round(float(g.get('CPI', 0.0)), 4),
            'share_POLICY_RATE':      round(float(g.get('POLICY_RATE', 0.0)), 4),
            'share_M2':               round(float(g.get('M2', 0.0)), 4),
            'share_UNEMPLOYMENT':     round(float(g.get('UNEMPLOYMENT', 0.0)), 4),
            'share_GDP':              round(float(g.get('GDP', 0.0)), 4),
        }
        row['share_total_check'] = round(sum([
            row['share_CPI_self'], row['share_POLICY_RATE'],
            row['share_M2'], row['share_UNEMPLOYMENT'], row['share_GDP']
        ]), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['country', 'horizon']).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 80
    print(bar)
    print('Phase 6 · Step 2 · S5 — Forecast Error Variance Decomposition')
    print(bar)
    print(f'lag orders:          {P_PER_COUNTRY}')
    print(f'Cholesky order:      {CHOLESKY_ORDER}')
    print(f'FEVD full horizon:   0..{FEVD_MAX_HORIZON}')
    print(f'reporting horizons:  {REPORTING_HORIZONS}')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    all_fevd_long: list[pd.DataFrame] = []

    for country in MAIN_COUNTRIES:
        p_star = P_PER_COUNTRY[country]
        print(f'>>> {country}  p* = {p_star}  — fit VAR → FEVD ...')
        endog, exog = extract_endog_exog_cholesky(features[country], country)
        results = fit_var(endog, exog, p_star, VAR_TREND)
        decomp = compute_fevd(results, FEVD_MAX_HORIZON)
        fevd_long = build_fevd_long(decomp, country, CHOLESKY_ORDER)
        all_fevd_long.append(fevd_long)

        # Quick per-country preview: CPI response at h=12
        cpi_at_12 = fevd_long[
            (fevd_long['response'] == 'CPI') & (fevd_long['horizon'] == 12)
        ].sort_values('share', ascending=False)
        print(f'    CPI variance decomposition at h=12 (sorted):')
        for _, row in cpi_at_12.iterrows():
            print(f'      {row["shock"]:<14s}  {row["share"]*100:5.1f}%')
        print()

    # ------------------------------------------------------------------
    # Consolidated outputs
    # ------------------------------------------------------------------
    full_long = pd.concat(all_fevd_long, ignore_index=True)
    full_path = doc_dir / 'phase6_step2_s5_fevd_full_matrix.csv'
    full_long.to_csv(full_path, index=False)

    cpi_panel = build_cpi_target_panel(full_long, REPORTING_HORIZONS)
    cpi_path = doc_dir / 'phase6_step2_s5_fevd_cpi_target.csv'
    cpi_panel.to_csv(cpi_path, index=False)

    top_contrib = build_top_contributors(full_long, REPORTING_HORIZONS, top_k=3)
    top_path = doc_dir / 'phase6_step2_s5_fevd_top_contributors.csv'
    top_contrib.to_csv(top_path, index=False)

    cpi_summary = build_cpi_summary(full_long, REPORTING_HORIZONS)
    summary_path = doc_dir / 'phase6_step2_s5_fevd_cpi_summary.csv'
    cpi_summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------------
    # Console narrative panels
    # ------------------------------------------------------------------
    print(bar)
    print('CPI variance decomposition panel (shares in %)')
    print(bar)
    # Pivot for a country × horizon matrix view, CPI-self share
    self_pivot = cpi_summary.pivot(
        index='country', columns='horizon', values='share_CPI_self',
    ) * 100
    print('\n  CPI self-share (auto-explanatory; high = isolated dynamics):')
    print(self_pivot.round(1).to_string())

    for shock_name in ['POLICY_RATE', 'M2', 'UNEMPLOYMENT', 'GDP']:
        pivot = cpi_summary.pivot(
            index='country', columns='horizon',
            values=f'share_{shock_name}',
        ) * 100
        print(f'\n  CPI share explained by {shock_name} shocks '
              f'({CPI_SHOCK_NARRATIVE[shock_name]}):')
        print(pivot.round(1).to_string())
    print()

    # Top contributors at h=12 for CPI
    print(bar)
    print('Top 3 contributors to CPI forecast error variance at h=12')
    print(bar)
    cpi_top12 = top_contrib[
        (top_contrib['response'] == 'CPI') &
        (top_contrib['horizon'] == 12)
    ]
    for country in MAIN_COUNTRIES:
        sub = cpi_top12[cpi_top12['country'] == country]
        top_desc = '  '.join(
            f'#{int(r["rank"])} {r["shock"]} ({r["share"]*100:.1f}%)'
            for _, r in sub.iterrows()
        )
        print(f'  {country:<10s}  {top_desc}')
    print()

    # ------------------------------------------------------------------
    # Artefact list + forward pointer
    # ------------------------------------------------------------------
    print(bar)
    print('Output artefacts written:')
    for p in [full_path, cpi_path, top_path, summary_path]:
        print(f'  data/documentation/{p.name}')
    print()

    print(bar)
    print('Interpretation notes:')
    print('  - CPI self-share = fraction of CPI forecast-error variance')
    print('    driven by its own innovations. High self-share (>80%)')
    print('    indicates isolated / predetermined dynamics (N3 echo).')
    print('  - Driver shares sum with self-share to ~100%; verify via')
    print('    share_total_check column in cpi_summary CSV.')
    print('  - FEVD is a point estimate per VAR fit; SE is not reported')
    print('    here (consistent with S4 asymptotic convention).')
    print()
    print('Next sub-step: S6 = OOS walk-forward forecast (D-005 train/test)')
    print('for Phase 7 Diebold-Mariano comparison vs ARIMA and Ridge.')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
