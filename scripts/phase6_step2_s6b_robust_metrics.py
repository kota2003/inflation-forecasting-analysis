"""
scripts/phase6_step2_s6b_robust_metrics.py
===========================================
Phase 6 · Step 2 · S6b — Robust forecast-accuracy diagnostic.

Purpose
-------
S6 aggregate RMSE for UK at h=12 = 138.75 is dominated by individual
forecast explosions. RMSE/MAE ratio ≈ 5.5 (vs the 1.2–1.5 range of a
well-behaved distribution) is the tell. This script:

    1. Identifies the worst forecast origins per (country × horizon)
       via absolute error ranking.
    2. Computes robust metrics:
         - median absolute error (MedAE)
         - trimmed RMSE/MAE at 5% (both tails)
         - MASE using MedAE (MedASE)
    3. Compares full vs robust metrics side-by-side.

This supplies the Phase 7 Diebold-Mariano robustness caveat: DM tests
with squared-error loss are sensitive to outliers; reporting both
full and trimmed versions is standard practice.

Input
-----
Reads phase6_step2_s6_var_oos_forecasts.csv (written by S6).
Requires the D-005 test-window forecast long-form table.

Output artefacts
----------------
data/documentation/
    phase6_step2_s6b_worst_origins.csv
        Top 5 worst forecast origins per (country × variable × horizon).
        4 countries × 5 variables × 4 horizons × 5 = 400 rows.
        cols: country, variable, horizon, rank, origin_date,
              target_date, forecast, actual, abs_error.
    phase6_step2_s6b_robust_metrics.csv
        Per (country × variable × horizon), both full and robust metrics.
        80 rows × 10 cols.
    phase6_step2_s6b_cpi_robust_summary.csv
        Filtered to variable=CPI for Phase 7 input.
        16 rows.

Decisions referenced
--------------------
D-005, D-050, D-060 (candidate) — VAR inference-vs-forecast trade-off.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s6b_robust_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (                                              # noqa: E402
    MAIN_COUNTRIES,
    build_all_features,
    find_project_root,
)


CHOLESKY_ORDER: list[str] = [
    'GDP', 'UNEMPLOYMENT', 'CPI', 'POLICY_RATE', 'M2',
]
FORECAST_HORIZONS: list[int] = [1, 3, 6, 12]
TRIM_PCT: float = 0.05
TEST_START: pd.Timestamp = pd.Timestamp('2020-01-01')


def robust_metrics(errors: np.ndarray,
                   trim_pct: float,
                   naive_mae: float) -> dict:
    """Compute full and robust metrics on an error array."""
    err = errors[~np.isnan(errors)]
    if len(err) == 0:
        return {
            'n_obs': 0, 'rmse': np.nan, 'mae': np.nan, 'medae': np.nan,
            'rmse_trimmed': np.nan, 'mae_trimmed': np.nan,
            'mase': np.nan, 'medase': np.nan,
            'rmse_to_mae_ratio': np.nan,
        }
    abs_err = np.abs(err)
    sq_err = err ** 2
    full_rmse = float(np.sqrt(sq_err.mean()))
    full_mae = float(abs_err.mean())
    medae = float(np.median(abs_err))
    mase = float(full_mae / naive_mae) if naive_mae > 0 else np.nan
    medase = float(medae / naive_mae) if naive_mae > 0 else np.nan

    # Two-sided trimming on absolute error
    lo_q = np.quantile(abs_err, trim_pct)
    hi_q = np.quantile(abs_err, 1 - trim_pct)
    trim_mask = (abs_err >= lo_q) & (abs_err <= hi_q)
    trim_err = err[trim_mask]
    trim_abs = abs_err[trim_mask]
    if len(trim_err) > 0:
        rmse_t = float(np.sqrt((trim_err ** 2).mean()))
        mae_t = float(trim_abs.mean())
    else:
        rmse_t = mae_t = np.nan

    ratio = full_rmse / full_mae if full_mae > 0 else np.nan
    return {
        'n_obs':             int(len(err)),
        'rmse':              full_rmse,
        'mae':               full_mae,
        'medae':             medae,
        'rmse_trimmed':      rmse_t,
        'mae_trimmed':       mae_t,
        'mase':              mase,
        'medase':            medase,
        'rmse_to_mae_ratio': ratio,
    }


def main() -> int:
    bar = '=' * 80
    print(bar)
    print('Phase 6 · Step 2 · S6b — Robust Forecast-Accuracy Diagnostic')
    print(bar)
    print(f'trim percentile:   {TRIM_PCT * 100:.1f}% each tail')
    print(f'diagnostic metric: RMSE/MAE ratio '
          '(normal ≈ 1.2–1.5; outlier-dominated ≫ 2)')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    s6_path = doc_dir / 'phase6_step2_s6_var_oos_forecasts.csv'
    if not s6_path.exists():
        raise FileNotFoundError(f'S6 forecasts not found: {s6_path}')

    print(f'>>> Loading {s6_path.name} ...')
    f = pd.read_csv(s6_path, parse_dates=['origin_date', 'target_date'])
    print(f'    {len(f)} forecast rows')
    print()

    # Build naive random-walk MAE per (country × variable) on TRAIN window
    # for MASE denominator. This mirrors S6.
    print('>>> Loading Phase 4 features for MASE denominator ...')
    features = build_all_features()
    print()

    # ------------------------------------------------------------------
    # Worst origins per (country × variable × horizon)
    # ------------------------------------------------------------------
    worst_rows: list[dict] = []
    for (country, var, h), g in f.dropna(subset=['error']).groupby(
        ['country', 'variable', 'horizon']
    ):
        g = g.copy()
        g['abs_error'] = g['error'].abs()
        g_sorted = g.sort_values('abs_error', ascending=False).head(5)
        for rank, (_, row) in enumerate(g_sorted.iterrows(), 1):
            worst_rows.append({
                'country':     country,
                'variable':    var,
                'horizon':     int(h),
                'rank':        rank,
                'origin_date': row['origin_date'].strftime('%Y-%m-%d'),
                'target_date': row['target_date'].strftime('%Y-%m-%d'),
                'forecast':    float(row['forecast']),
                'actual':      float(row['actual']),
                'abs_error':   float(row['abs_error']),
            })
    worst_df = pd.DataFrame(worst_rows)
    worst_path = doc_dir / 'phase6_step2_s6b_worst_origins.csv'
    worst_df.to_csv(worst_path, index=False)

    # ------------------------------------------------------------------
    # Robust metrics
    # ------------------------------------------------------------------
    metric_rows: list[dict] = []
    for country in MAIN_COUNTRIES:
        endog_cols = [f'{country}_{v}' for v in CHOLESKY_ORDER]
        country_feat = features[country][endog_cols].dropna(how='any')
        for var in CHOLESKY_ORDER:
            endog_col = f'{country}_{var}'
            train = country_feat.loc[
                :TEST_START - pd.DateOffset(days=1),
                endog_col,
            ]
            naive_mae = float(train.diff().dropna().abs().mean())
            for h in FORECAST_HORIZONS:
                sub = f[
                    (f['country']  == country)
                    & (f['variable'] == var)
                    & (f['horizon']  == h)
                ]
                err = sub['error'].to_numpy()
                m = robust_metrics(err, TRIM_PCT, naive_mae)
                metric_rows.append({
                    'country':  country,
                    'variable': var,
                    'horizon':  h,
                    **m,
                })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = doc_dir / 'phase6_step2_s6b_robust_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)

    # CPI subset
    cpi_robust = metrics_df[metrics_df['variable'] == 'CPI'].copy()
    cpi_path = doc_dir / 'phase6_step2_s6b_cpi_robust_summary.csv'
    cpi_robust.to_csv(cpi_path, index=False)

    # ------------------------------------------------------------------
    # Console panels
    # ------------------------------------------------------------------
    print(bar)
    print('RMSE/MAE ratio diagnostic (CPI; ≫2 flags outlier domination)')
    print(bar)
    ratio_pivot = cpi_robust.pivot(
        index='country', columns='horizon', values='rmse_to_mae_ratio',
    )
    print(ratio_pivot.round(2).to_string())
    print()

    print(bar)
    print('CPI forecast accuracy — full vs robust metrics')
    print(bar)
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.float_format', lambda v: f'{v:.4f}'):
        display_cols = ['country', 'horizon', 'n_obs',
                        'rmse', 'rmse_trimmed',
                        'mae', 'medae',
                        'mase', 'medase',
                        'rmse_to_mae_ratio']
        print(cpi_robust[display_cols].to_string(index=False))
    print()

    # Robust MASE pivot
    print(bar)
    print('Robust MedASE pivot (CPI; <1 beats naive on median absolute error)')
    print(bar)
    medase_pivot = cpi_robust.pivot(
        index='country', columns='horizon', values='medase',
    )
    print(medase_pivot.round(4).to_string())
    print()

    # Worst 5 origins for UK h=12 (the catastrophic case)
    print(bar)
    print('UK h=12 worst origins (investigating RMSE = 138.75)')
    print(bar)
    uk12 = worst_df[
        (worst_df['country']  == 'UK')
        & (worst_df['variable'] == 'CPI')
        & (worst_df['horizon']  == 12)
    ]
    print(uk12.to_string(index=False))
    print()

    # USA h=1 worst origins (since USA MASE = 3.73 at h=1 is surprising)
    print(bar)
    print('USA h=1 worst origins (investigating MASE = 3.73 at shortest horizon)')
    print(bar)
    usa1 = worst_df[
        (worst_df['country']  == 'USA')
        & (worst_df['variable'] == 'CPI')
        & (worst_df['horizon']  == 1)
    ]
    print(usa1.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    print(bar)
    print('Output artefacts written:')
    for p in [worst_path, metrics_path, cpi_path]:
        print(f'  data/documentation/{p.name}')
    print()

    print(bar)
    print('Interpretation:')
    print('  - RMSE/MAE ratio > 2 → aggregate dominated by outliers.')
    print('    Report trimmed RMSE and/or MedAE as robust alternative.')
    print('  - MedASE < 1 → VAR beats naive on the MEDIAN observation')
    print('    even if aggregate MASE is > 1 due to outlier dominance.')
    print('  - Outlier origin list informs Phase 7 narrative: regime')
    print('    shifts (2020 COVID, 2022 Energy) are likely culprits.')
    print()
    print('Next: notebooks/07_var_model.ipynb narrative assembly.')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
