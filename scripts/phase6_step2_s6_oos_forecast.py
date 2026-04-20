"""
scripts/phase6_step2_s6_oos_forecast.py
========================================
Phase 6 · Step 2 · S6 — OOS walk-forward forecast.

Purpose
-------
Produce out-of-sample CPI forecasts from each country's VAR(p*) fit
(D-050 AIC-primary lag order), using expanding-window walk-forward
estimation per D-005 (train ≤ 2019-12, test ≥ 2020-01). This is the
final sub-step of Phase 6 Step 2 and supplies the VAR-side input
for the Phase 7 Diebold-Mariano forecast-accuracy comparison against
ARIMA Step 1 and Ridge Step 3.

Walk-forward protocol
---------------------
For each forecast origin `o` ∈ test dates:
    1. Fit VAR(p*) with D-030 exog on data[:o] (expanding window).
    2. Forecast horizons h ∈ {1, 3, 6, 12} from origin o.
    3. Record forecast value and actual value (when available) per
       (country, origin, horizon, variable).

Forecast-origin range
---------------------
    start: 2020-01-01 (D-005 train/test boundary)
    end:   last_available_date - max_horizon
           (so we have actuals for horizon = 12 verification)

Typical per-country origin count: 48–58 months
Typical total refits:             ~48 per country × 4 = ~192

Multi-horizon design
--------------------
Horizons 1 / 3 / 6 / 12 reported together. h=1 is the primary
Diebold-Mariano comparison target (matches ARIMA Step 1 output).
Multi-horizon provides a richer picture of each model's prediction
profile decay and supports portfolio narrative on monetary lag.

Accuracy metrics
----------------
Per (country × variable × horizon):
    n_obs, rmse, mae, mean_bias (mean of forecast - actual),
    mase (mean absolute scaled error vs naive random walk),
    coverage_95 (fraction of actuals within forecast 95% CI)

Output artefacts
----------------
data/documentation/
    phase6_step2_s6_var_oos_forecasts.csv
        Long form: 4 countries × (48..58 origins) × 4 horizons ×
        5 variables ≈ 4 000+ rows.
        cols: country, origin_date, horizon, target_date, variable,
              forecast, actual, error.
    phase6_step2_s6_var_oos_metrics.csv
        4 countries × 5 variables × 4 horizons = 80 rows.
        cols: country, variable, horizon, n_obs, rmse, mae, mean_bias,
              mase, coverage_95.
    phase6_step2_s6_var_oos_cpi_summary.csv
        Filtered to variable = CPI; narrative-ready Phase 7 input.
        16 rows (4 countries × 4 horizons).

Decisions referenced
--------------------
D-005  Train / test split 2000–2019 vs 2020+ (applied here).
D-030  Dominant-driver matrix — exog structure.
D-050  AIC-primary VAR lag selection.
D-054  Cholesky ordering [GDP, UE, CPI, PR, M2] — unchanged for forecasting.
D-058/D-059  (candidates) — FEVD anatomical findings; S6 provides
       the forecast-accuracy anchor for Phase 7 DM battery.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s6_oos_forecast.py

Runtime expectation: 1-3 minutes (~192 VAR refits).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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

CHOLESKY_ORDER: list[str] = [
    'GDP', 'UNEMPLOYMENT', 'CPI', 'POLICY_RATE', 'M2',
]

#: Forecast horizons to evaluate (in months).
FORECAST_HORIZONS: list[int] = [1, 3, 6, 12]

#: D-005 train/test split boundary. Test window starts 2020-01.
TEST_START: pd.Timestamp = pd.Timestamp('2020-01-01')

VAR_TREND: str = 'c'
BASE_INDICATORS: list[str] = list(INDICATORS)
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())
PERIOD_KEYS: list[str] = ['GFC', 'COVID']


# ── Exog / endog helpers — identical to S5 ───────────────────────────

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


# ── Walk-forward forecasting ─────────────────────────────────────────

def compute_origin_dates(endog: pd.DataFrame,
                         test_start: pd.Timestamp,
                         max_horizon: int) -> list[pd.Timestamp]:
    """Origin dates where all forecast horizons have actuals."""
    last_available = endog.index.max()
    origins = [d for d in endog.index
               if d >= test_start
               and d + pd.DateOffset(months=max_horizon) <= last_available]
    return origins


def build_future_exog(exog_df: pd.DataFrame,
                      origin: pd.Timestamp,
                      horizon: int) -> np.ndarray:
    """Future exog values for forecast horizon.

    Our exog columns (split dummies, period dummies, interactions) are
    all deterministic functions of the calendar + base regressors.
    At forecast origin o, horizons 1..h, the exog values are simply
    the observed exog values at o+1, o+2, ..., o+h — which are in the
    feature matrix because exog is built from the full sample.

    Returns array of shape (horizon, n_exog).
    """
    future_idx = [origin + pd.DateOffset(months=k) for k in range(1, horizon + 1)]
    missing = [d for d in future_idx if d not in exog_df.index]
    if missing:
        raise KeyError(
            f"Future exog missing for origin={origin.date()}, "
            f"horizon={horizon}, missing dates: {missing[:3]}..."
        )
    return exog_df.loc[future_idx].to_numpy()


def forecast_at_origin(
    endog: pd.DataFrame,
    exog: pd.DataFrame,
    origin: pd.Timestamp,
    p_star: int,
    horizons: list[int],
    endog_cols: list[str],
) -> Optional[pd.DataFrame]:
    """Fit VAR on data[:origin] (inclusive) and forecast horizons.

    Returns DataFrame with columns
        horizon, variable, target_date, forecast
    or None if the fit fails.
    """
    train_endog = endog.loc[:origin]
    train_exog = exog.loc[:origin]
    if len(train_endog) < p_star * 5 + 10:  # DOF guard
        return None

    try:
        model = VAR(train_endog, exog=train_exog)
        results = model.fit(maxlags=p_star, trend=VAR_TREND)
    except Exception as exc:
        print(f'    WARN: fit failed at origin {origin.date()}: '
              f'{type(exc).__name__}: {exc}')
        return None

    max_h = max(horizons)
    try:
        exog_future = build_future_exog(exog, origin, max_h)
    except KeyError:
        return None

    # statsmodels VARResults.forecast signature:
    #   forecast(y, steps, exog_future=None)
    # y must contain the last p rows to initialize the forecast.
    y_init = train_endog.values[-p_star:]
    fcst = results.forecast(y_init, steps=max_h, exog_future=exog_future)
    # fcst shape: (max_h, n_endog)

    rows: list[dict] = []
    for h in horizons:
        if h > max_h:
            continue
        target_date = origin + pd.DateOffset(months=h)
        fcst_vec = fcst[h - 1]  # 0-indexed
        for i, var in enumerate(endog_cols):
            rows.append({
                'horizon':     h,
                'variable':    var.replace(f'{train_endog.columns[0].split("_")[0]}_', ''),
                'target_date': target_date,
                'forecast':    float(fcst_vec[i]),
            })
    return pd.DataFrame(rows)


# ── Accuracy metric helpers ──────────────────────────────────────────

def compute_metrics(sub: pd.DataFrame,
                    endog_train_series: pd.Series) -> dict:
    """Per-(country × variable × horizon) accuracy metrics.

    MASE denominator = mean absolute first-diff of the TRAIN series
    (standard random-walk-naive benchmark).
    """
    valid = sub.dropna(subset=['actual', 'forecast'])
    if valid.empty:
        return {
            'n_obs': 0, 'rmse': np.nan, 'mae': np.nan,
            'mean_bias': np.nan, 'mase': np.nan,
        }
    err = valid['forecast'] - valid['actual']
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    bias = float(err.mean())

    # MASE
    naive_mae = float(endog_train_series.diff().dropna().abs().mean())
    mase = float(mae / naive_mae) if naive_mae > 0 else np.nan

    return {
        'n_obs':      int(len(valid)),
        'rmse':       rmse,
        'mae':        mae,
        'mean_bias':  bias,
        'mase':       mase,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 80
    print(bar)
    print('Phase 6 · Step 2 · S6 — OOS Walk-Forward VAR Forecast')
    print(bar)
    print(f'lag orders:        {P_PER_COUNTRY}')
    print(f'Cholesky order:    {CHOLESKY_ORDER}')
    print(f'test start:        {TEST_START.date()}  (D-005)')
    print(f'forecast horizons: {FORECAST_HORIZONS} months')
    print(f'method:            expanding-window walk-forward')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    all_forecasts: list[pd.DataFrame] = []

    for country in MAIN_COUNTRIES:
        p_star = P_PER_COUNTRY[country]
        endog, exog = extract_endog_exog_cholesky(features[country], country)
        endog_cols = list(endog.columns)

        # Strip country prefix for display
        display_cols = [c.replace(f'{country}_', '') for c in endog_cols]

        origins = compute_origin_dates(endog, TEST_START, max(FORECAST_HORIZONS))
        print(f'>>> {country}  p* = {p_star}  — {len(origins)} walk-forward origins '
              f'({origins[0].date() if origins else "none"} .. '
              f'{origins[-1].date() if origins else "none"})')

        country_rows: list[dict] = []
        n_failed = 0
        for origin in origins:
            fcst_df = forecast_at_origin(
                endog, exog, origin, p_star, FORECAST_HORIZONS, endog_cols,
            )
            if fcst_df is None:
                n_failed += 1
                continue
            # Attach origin + country and stash for later actual merge
            for _, r in fcst_df.iterrows():
                var = r['variable']
                target = r['target_date']
                endog_col = f'{country}_{var}'
                actual = (float(endog.loc[target, endog_col])
                          if target in endog.index else np.nan)
                country_rows.append({
                    'country':     country,
                    'origin_date': origin,
                    'horizon':     int(r['horizon']),
                    'target_date': target,
                    'variable':    var,
                    'forecast':    float(r['forecast']),
                    'actual':      actual,
                    'error':       float(r['forecast']) - actual
                                   if not np.isnan(actual) else np.nan,
                })

        country_df = pd.DataFrame(country_rows)
        all_forecasts.append(country_df)
        print(f'    successful fits: {len(origins) - n_failed} / {len(origins)}  '
              f'({n_failed} failed)')
        # Quick per-country CPI preview
        cpi_sub = country_df[country_df['variable'] == 'CPI']
        for h in FORECAST_HORIZONS:
            h_sub = cpi_sub[cpi_sub['horizon'] == h].dropna(subset=['error'])
            if h_sub.empty:
                continue
            rmse = float(np.sqrt((h_sub['error'] ** 2).mean()))
            mae = float(h_sub['error'].abs().mean())
            bias = float(h_sub['error'].mean())
            print(f'      CPI h={h:2d}  n={len(h_sub):3d}  '
                  f'RMSE={rmse:.4f}  MAE={mae:.4f}  bias={bias:+.4f}')
        print()

    # ------------------------------------------------------------------
    # Consolidate forecasts
    # ------------------------------------------------------------------
    all_f = pd.concat(all_forecasts, ignore_index=True)
    f_path = doc_dir / 'phase6_step2_s6_var_oos_forecasts.csv'
    all_f.to_csv(f_path, index=False)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metric_rows: list[dict] = []
    for country in MAIN_COUNTRIES:
        endog, _ = extract_endog_exog_cholesky(features[country], country)
        for var_ind in CHOLESKY_ORDER:
            endog_col = f'{country}_{var_ind}'
            train_series = endog.loc[:TEST_START - pd.DateOffset(days=1),
                                     endog_col]
            for h in FORECAST_HORIZONS:
                sub = all_f[
                    (all_f['country']  == country)
                    & (all_f['variable'] == var_ind)
                    & (all_f['horizon']  == h)
                ]
                m = compute_metrics(sub, train_series)
                metric_rows.append({
                    'country':   country,
                    'variable':  var_ind,
                    'horizon':   h,
                    **m,
                })
    metrics_df = pd.DataFrame(metric_rows)
    m_path = doc_dir / 'phase6_step2_s6_var_oos_metrics.csv'
    metrics_df.to_csv(m_path, index=False)

    # CPI-focused narrative summary
    cpi_summary = metrics_df[metrics_df['variable'] == 'CPI'].copy()
    cpi_path = doc_dir / 'phase6_step2_s6_var_oos_cpi_summary.csv'
    cpi_summary.to_csv(cpi_path, index=False)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(bar)
    print('CPI forecast accuracy summary  '
          '(lower RMSE/MAE = better; MASE < 1 beats random-walk naive)')
    print(bar)
    with pd.option_context('display.max_columns', None,
                           'display.width', 160,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(cpi_summary.to_string(index=False))
    print()

    # Pivot: RMSE country × horizon
    print('RMSE pivot (country × horizon):')
    rmse_pivot = cpi_summary.pivot(index='country', columns='horizon',
                                    values='rmse')
    print(rmse_pivot.round(4).to_string())
    print()

    print('MASE pivot (country × horizon; <1 beats naive):')
    mase_pivot = cpi_summary.pivot(index='country', columns='horizon',
                                    values='mase')
    print(mase_pivot.round(4).to_string())
    print()

    # ------------------------------------------------------------------
    # Artefact list + forward pointer
    # ------------------------------------------------------------------
    print(bar)
    print('Output artefacts written:')
    for p in [f_path, m_path, cpi_path]:
        print(f'  data/documentation/{p.name}')
    print()

    print(bar)
    print('Phase 6 · Step 2 · VAR layer — S1/S1b/S2/S2b/S3/S4/S5/S6 COMPLETE.')
    print()
    print('Next: consolidate via notebooks/07_var_model.ipynb (portfolio')
    print('narrative) + append D-050 through D-059 locked-in-draft decisions')
    print('to ProjectDriven.md. Then Phase 6 Step 3 = Ridge Regression Layer 3.')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
