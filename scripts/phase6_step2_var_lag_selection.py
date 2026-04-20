"""
scripts/phase6_step2_var_lag_selection.py
==========================================
Phase 6 · Step 2 · Sub-step 1 — VAR lag order selection.

Purpose
-------
Determine per-country VAR lag order via information criteria
(AIC / BIC / HQIC / FPE) on the 5-variable base endogenous block
{CPI, POLICY_RATE, UNEMPLOYMENT, GDP, M2} in D-031 stationary form.

This is the first of five Phase 6 Step 2 sub-steps:
    S1 — lag order selection          ← this script
    S2 — VAR estimation (with D-030 regime interactions)
    S3 — Granger causality battery
    S4 — Impulse Response Functions (+ 95 % bootstrap CI)
    S5 — Forecast Error Variance Decomposition

Scope boundaries (S1 only)
--------------------------
- Selection of lag order p via information criteria. No estimation.
- Unconditional 5-variable VAR. D-030 regime-dummy interactions are
  DEFERRED to S2 estimation (standard practice: select p on the base
  system, then add exogenous terms for estimation).
- Full-sample selection. The D-005 train/test boundary (2019-12 / 2020-01)
  is applied at the estimation step; information-criteria lag selection
  is not a forecasting operation and benefits from the full sample.
- No Granger, no IRF, no FEVD. Those are S3 / S4 / S5.

Design choices (for portfolio defensibility)
--------------------------------------------
- maxlag = 12. Matches the D-034 Phase 4 lag grid upper bound {1, 3, 6, 12}
  and covers a full annual seasonal cycle; sufficient residual degrees of
  freedom at n_obs ≈ 280 (≈ 61 coeffs per equation at lag 12 vs ≈ 220
  remaining obs).
- All four information criteria are reported. Final single-criterion pick
  per country is a separate decision (D-050 candidate) to be made in the
  upcoming portfolio review step, not inside this coding-only deliverable.

Output artefacts
----------------
data/documentation/
    phase6_step2_var_lag_selection_{country}.csv
        13 rows (lag 0..12) × 10 cols
        (country, lag, aic, bic, hqic, fpe,
         is_selected_aic, is_selected_bic, is_selected_hqic, is_selected_fpe)
    phase6_step2_var_lag_selection_summary.csv
        4 rows × 9 cols
        (country, n_obs_joint_valid, sample_start, sample_end, maxlag_tested,
         aic_selected_lag, bic_selected_lag, hqic_selected_lag, fpe_selected_lag)

Decisions referenced
--------------------
D-004  Three-layer modelling architecture (ARIMA → VAR → Ridge).
D-005  Train / test split 2000–2019 vs 2020+ (applied at estimation, not here).
D-030  Dominant-driver matrix (interactions deferred to S2).
D-031  Base transformation registry (USA yoy_pct, JPN/GER first_diff,
       UK log_diff_pct, others per Phase 3).
D-034  Phase 4 lag grid {1, 3, 6, 12} — maxlag = 12 here matches upper bound.
D-048  ARIMA grid-search precedent — reporting multiple ICs before selection.

Reproducibility
---------------
Deterministic. No random seed. Requires:
    - Phase 2 processed CSVs at data/processed/main_{country}.csv
    - Phase 3 registry at data/documentation/phase3_transformation_registry_final.csv
    - src/ module v0.4.0 (provides build_all_features, MAIN_COUNTRIES, etc.)

Usage
-----
    (p3_inflation) $ cd C:\\Users\\kotae\\Documents\\Portfolio\\project\\Project 3\\inflation-forecasting-analysis
    (p3_inflation) $ python scripts/phase6_step2_var_lag_selection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Path wiring so `from src import ...` works from scripts/ ──────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (                                              # noqa: E402
    MAIN_COUNTRIES,
    INDICATORS,
    build_all_features,
    find_project_root,
)
from statsmodels.tsa.vector_ar.var_model import VAR            # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────

#: Maximum lag tested in VAR.select_order.  Matches D-034 Phase 4 grid
#: upper bound {1, 3, 6, 12}; covers the annual seasonal cycle without
#: starving degrees of freedom at n_obs ≈ 280 (≈ 61 coeffs per equation).
VAR_MAXLAG: int = 12

#: Information criteria reported per country. All four statsmodels-native
#: criteria are exposed; the single-criterion final pick per country is
#: recorded as a separate decision (candidate D-050) in the next sub-step.
VAR_CRITERIA: tuple[str, ...] = ('aic', 'bic', 'hqic', 'fpe')

#: Five base endogenous variables forming the VAR system (ProjectScope §2).
#: Applied in D-031 stationary form as determined by Phase 4's effective
#: registry. Column naming convention: f"{COUNTRY}_{INDICATOR}".
BASE_INDICATORS: list[str] = list(INDICATORS)


# ── Helper functions ──────────────────────────────────────────────────

def extract_base_block(features_df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Subset a country feature matrix to its 5-variable endogenous block.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of `src.build_country_features(country)` — wide feature
        matrix with 50–53 columns including lag / rolling / regime.
    country : str
        One of ('USA', 'JAPAN', 'UK', 'GERMANY').

    Returns
    -------
    pd.DataFrame
        5-column DataFrame in D-031 stationary form, rows dropna-trimmed
        to the joint-valid window across the five series.
    """
    cols = [f'{country}_{ind}' for ind in BASE_INDICATORS]
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise KeyError(
            f"{country}: missing base columns {missing} "
            f"in features matrix (cols present: {list(features_df.columns)[:8]}...)"
        )
    base = features_df[cols].copy()
    base = base.dropna(how='any')
    if base.empty:
        raise ValueError(f"{country}: 5-var base block is empty after dropna")
    return base


def select_var_order(base_df: pd.DataFrame, maxlag: int) -> dict:
    """Run `statsmodels` VAR.select_order and extract the full IC table.

    Parameters
    ----------
    base_df : pd.DataFrame
        5-column VAR endogenous block (post-dropna).
    maxlag : int
        Upper bound on lag orders tested.

    Returns
    -------
    dict with keys:
        ic_table  : pd.DataFrame  (rows lag 0..maxlag, cols AIC/BIC/HQIC/FPE)
        selected  : dict[str, int]  criterion → selected lag
        n_obs     : int             joint-valid sample size
        start     : pd.Timestamp    sample window start
        end       : pd.Timestamp    sample window end
    """
    model = VAR(base_df)
    lag_res = model.select_order(maxlags=maxlag)

    # lag_res.ics is dict: {criterion: sequence of length (maxlag + 1)}.
    # Cast to DataFrame to align on the shared lag index (0..maxlag).
    ic_table = pd.DataFrame(
        {crit: list(lag_res.ics[crit]) for crit in VAR_CRITERIA}
    )
    ic_table.index.name = 'lag'
    ic_table.index = ic_table.index.astype(int)

    selected = {
        crit: int(lag_res.selected_orders[crit]) for crit in VAR_CRITERIA
    }

    return {
        'ic_table': ic_table,
        'selected': selected,
        'n_obs':    len(base_df),
        'start':    base_df.index.min(),
        'end':      base_df.index.max(),
    }


def build_country_table(result: dict, country: str) -> pd.DataFrame:
    """Build the per-country output CSV (lag × IC values + selection flags).

    One row per tested lag 0..maxlag; selection-flag columns mark which
    lag is chosen by each criterion for easy downstream filtering.
    """
    tbl = result['ic_table'].copy()
    for crit in VAR_CRITERIA:
        tbl[f'is_selected_{crit}'] = (tbl.index == result['selected'][crit])
    tbl = tbl.reset_index()  # lag becomes a column
    tbl.insert(0, 'country', country)
    return tbl


def build_summary_table(
    results: dict[str, dict], maxlag: int
) -> pd.DataFrame:
    """Cross-country one-row-per-country summary."""
    rows: list[dict] = []
    for country, res in results.items():
        row = {
            'country':           country,
            'n_obs_joint_valid': res['n_obs'],
            'sample_start':      res['start'].strftime('%Y-%m-%d'),
            'sample_end':        res['end'].strftime('%Y-%m-%d'),
            'maxlag_tested':     maxlag,
        }
        for crit in VAR_CRITERIA:
            row[f'{crit}_selected_lag'] = res['selected'][crit]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    """Entry point. Returns 0 on success; non-zero reserved for future use."""

    # ------------------------------------------------------------------
    # Part 0 — Header & configuration
    # ------------------------------------------------------------------
    bar = '=' * 72
    print(bar)
    print('Phase 6 · Step 2 · S1 — VAR Lag Order Selection')
    print(bar)
    print(f'maxlag tested:     {VAR_MAXLAG}')
    print(f'criteria reported: {", ".join(c.upper() for c in VAR_CRITERIA)}')
    print(f'endogenous block:  {BASE_INDICATORS}')
    print(f'transformation:    D-031 (per-country effective registry)')
    print(f'exogenous:         none — D-030 interactions deferred to S2')
    print(f'sample scope:      full (D-005 train/test applies at S2)')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Part 1 — Load Phase 4 feature matrices for all four main countries.
    # ------------------------------------------------------------------
    print('>>> Loading Phase 4 feature matrices via src.build_all_features()')
    features = build_all_features()
    for c in MAIN_COUNTRIES:
        rows, cols = features[c].shape
        print(f'    {c:<8}  features_{c.lower()}: {rows} rows × {cols} cols')
    print()

    # ------------------------------------------------------------------
    # Part 2 — Per-country lag selection + per-country CSV write.
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    for country in MAIN_COUNTRIES:
        print(f'>>> {country} — VAR.select_order(maxlags={VAR_MAXLAG}) ...')
        base = extract_base_block(features[country], country)
        res = select_var_order(base, VAR_MAXLAG)
        results[country] = res

        country_tbl = build_country_table(res, country)
        out_path = doc_dir / (
            f'phase6_step2_var_lag_selection_{country.lower()}.csv'
        )
        country_tbl.to_csv(out_path, index=False)

        sel = res['selected']
        print(f'    sample:       {res["start"].date()} .. {res["end"].date()}'
              f'  (n_obs = {res["n_obs"]})')
        print(f'    selected lag: '
              + ', '.join(f'{c.upper()}={sel[c]}' for c in VAR_CRITERIA))
        print(f'    written:      data/documentation/{out_path.name}')
        print()

    # ------------------------------------------------------------------
    # Part 3 — Cross-country summary CSV.
    # ------------------------------------------------------------------
    summary = build_summary_table(results, VAR_MAXLAG)
    summary_path = doc_dir / 'phase6_step2_var_lag_selection_summary.csv'
    summary.to_csv(summary_path, index=False)

    print(bar)
    print('Cross-country summary')
    print(bar)
    # Pretty-print summary with column alignment.
    with pd.option_context('display.max_columns', None,
                           'display.width', 160):
        print(summary.to_string(index=False))
    print()
    print(f'written: data/documentation/{summary_path.name}')
    print()

    # ------------------------------------------------------------------
    # Part 4 — Console-friendly IC values table (compact form).
    # ------------------------------------------------------------------
    print(bar)
    print('Per-country IC values (rounded; ★ marks criterion-selected lag)')
    print(bar)
    for country, res in results.items():
        print(f'\n  {country}:')
        ic = res['ic_table'].round(4)
        sel = res['selected']
        # Build display DataFrame with star-marked selected cells.
        disp = pd.DataFrame(index=ic.index)
        for crit in VAR_CRITERIA:
            col = []
            for lag, val in ic[crit].items():
                star = '★' if lag == sel[crit] else ' '
                col.append(f'{val:+9.4f} {star}')
            disp[crit.upper()] = col
        print(disp.to_string())
    print()

    # ------------------------------------------------------------------
    # Part 5 — Next-step pointer (for interactive workflow continuity).
    # ------------------------------------------------------------------
    print(bar)
    print('Done. Next sub-step (S2) — VAR estimation at the selected p*,')
    print('with D-030 regime-interaction exogenous terms. Single-criterion')
    print('final pick per country = upcoming decision (D-050 candidate).')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
