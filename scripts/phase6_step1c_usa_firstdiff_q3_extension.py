"""
phase6_step1c_usa_firstdiff_q3_extension.py
============================================

Phase 6 Step 1c — USA_first_diff Q=3 grid extension per D-048 amendment.

Motivation
----------
Step 1b boundary sensitivity check (D-048 sensitivity protocol) returned
three verdicts:

    USA_yoy_pct      accept_Q2   (ΔAIC = -0.21, below threshold)
    USA_first_diff   extend_to_Q3 (ΔAIC = -9.14, meaningfully better)  ← this step
    UK_log_diff_pct  accept_Q2   (ΔAIC = +12.33, Q=3 worse)

Only USA_first_diff met the ΔAIC ≤ -2.0 threshold.  This script
executes the targeted extension for that variant alone:

    grid:  p ∈ [0, 4], d = 0, q ∈ [0, 4],
           P ∈ [0, 2], D ∈ {0, 1}, Q = 3, s = 12
    size:  5 × 1 × 5 × 3 × 2 × 1 = 150 orders (Q=3 sweep only)

These 150 Q=3 rows are unioned with the existing Step 1 USA_first_diff
grid (450 rows, Q ∈ [0, 2]) to yield the effective 600-row search
space.  The new AIC-best is selected from the union.

Artifact convention
-------------------
* Step 1 grid CSVs remain pristine (Q ∈ [0, 2] evidence).
* This script writes a new CSV for the 150 Q=3 rows:
    ``phase6_step1c_arima_grid_usa_first_diff_q3.csv``
* Consolidated summary CSVs (selection, residuals, forecast,
  window_errors) are updated IN PLACE for the USA_first_diff row(s).
  Other variants' rows are unchanged.

Portfolio state:
    Grid evidence    = Step 1 CSVs + Step 1b boundary CSVs + Step 1c Q=3 CSV
    Final selection  = Step 1 consolidated CSVs (post-amendment)

Decision references:
    D-048 (parent grid), D-048 amendment (this step), D-033 precedent.

Runtime estimate: 150 fits × ~1.5 s + 70 refits × ~0.3 s ≈ 4-6 min.
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import find_project_root, first_difference, load_processed_main


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
VARIANT_ID:      str = 'USA_first_diff'
SEASONAL_PERIOD: int = 12

# Q=3 sweep grid (other dims identical to Step 1 D-048 spec)
GRID_P:  Tuple[int, ...] = (0, 1, 2, 3, 4)
GRID_D:  Tuple[int, ...] = (0,)
GRID_Q:  Tuple[int, ...] = (0, 1, 2, 3, 4)
GRID_SP: Tuple[int, ...] = (0, 1, 2)
GRID_SD: Tuple[int, ...] = (0, 1)
GRID_SQ_EXTENSION: Tuple[int, ...] = (3,)        # Q=3 only (Q∈[0,2] in Step 1)

TRAIN_END:   pd.Timestamp = pd.Timestamp('2019-12-01')
TEST_START:  pd.Timestamp = pd.Timestamp('2020-01-01')

COVID_START:  pd.Timestamp = pd.Timestamp('2020-01-01')
COVID_END:    pd.Timestamp = pd.Timestamp('2021-12-01')
ENERGY_START: pd.Timestamp = pd.Timestamp('2022-01-01')

LJUNG_BOX_LAGS: Tuple[int, ...] = (12, 24)
ARCH_LAGS: int = 12

SARIMAX_METHOD:  str = 'lbfgs'
SARIMAX_MAXITER: int = 200


# ─────────────────────────────────────────────────────────────
# Helpers — verbatim from Step 1 for script independence
# ─────────────────────────────────────────────────────────────
def _force_monthly_freq(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).to_period('M').to_timestamp(how='start')
    return out.asfreq('MS')


def build_usa_first_diff(project_root: Path) -> pd.Series:
    """Construct USA_first_diff from level CPI via first_difference."""
    usa_level = load_processed_main('USA', project_root=project_root)['USA_CPI']
    return _force_monthly_freq(first_difference(usa_level))


def fit_sarimax(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    trend: str = 'c',
) -> Dict:
    t0 = time.perf_counter()
    p, d, q = order
    P, D, Q, s = seasonal_order
    row: Dict = {
        'p': p, 'd': d, 'q': q,
        'P': P, 'D': D, 'Q': Q, 's': s,
        'n_params': p + q + P + Q + (1 if trend == 'c' else 0),
        'aic': np.nan, 'bic': np.nan, 'hqic': np.nan, 'llf': np.nan,
        'converged': False,
        'runtime_sec': np.nan,
        'error': '',
        '_fit': None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mod = SARIMAX(
                y, order=order, seasonal_order=seasonal_order, trend=trend,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit = mod.fit(method=SARIMAX_METHOD,
                          maxiter=SARIMAX_MAXITER, disp=False)
        row['aic']  = float(fit.aic)
        row['bic']  = float(fit.bic)
        row['hqic'] = float(fit.hqic)
        row['llf']  = float(fit.llf)
        retvals = getattr(fit, 'mle_retvals', {}) or {}
        row['converged'] = bool(retvals.get('converged', False))
        row['_fit'] = fit
    except Exception as exc:                          # noqa: BLE001
        row['error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
    row['runtime_sec'] = time.perf_counter() - t0
    return row


def residual_diagnostics(variant_id: str, fit_result, aic_best_order_str: str) -> Dict:
    resid = pd.Series(fit_result.resid).dropna()
    lb = acorr_ljungbox(resid, lags=list(LJUNG_BOX_LAGS), return_df=True)
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    try:
        arch_lm, arch_p, *_ = het_arch(resid, nlags=ARCH_LAGS)
    except Exception:                                  # noqa: BLE001
        arch_lm, arch_p = np.nan, np.nan

    out = {
        'variant_id':    variant_id,
        'n_resid':       int(len(resid)),
        'resid_mean':    float(resid.mean()),
        'resid_std':     float(resid.std(ddof=1)),
        'skew':          float(skew),
        'kurt_excess':   float(kurt - 3.0),
        'jb_stat':       float(jb_stat),
        'jb_p':          float(jb_p),
        'arch_lm_stat':  float(arch_lm) if not pd.isna(arch_lm) else np.nan,
        'arch_lm_p':     float(arch_p) if not pd.isna(arch_p) else np.nan,
    }
    for lag in LJUNG_BOX_LAGS:
        out[f'ljungbox_q{lag}_stat'] = float(lb.loc[lag, 'lb_stat'])
        out[f'ljungbox_q{lag}_p']    = float(lb.loc[lag, 'lb_pvalue'])
    out['aic_best_order'] = aic_best_order_str
    return out


def expanding_forecast(
    variant_id: str,
    y_full: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> pd.DataFrame:
    test_idx = y_full.index[y_full.index >= TEST_START]
    print(f'\n--- [{variant_id}] expanding-refit 1-step forecast: '
          f'{len(test_idx)} steps ---', flush=True)
    records: List[Dict] = []
    t_start = time.perf_counter()
    for i, t in enumerate(test_idx, start=1):
        y_hist = y_full.loc[:t].iloc[:-1]
        actual = float(y_full.loc[t])
        pred   = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mod = SARIMAX(
                    y_hist, order=order, seasonal_order=seasonal_order,
                    trend='c',
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = mod.fit(method=SARIMAX_METHOD,
                              maxiter=SARIMAX_MAXITER, disp=False)
            f = fit.get_forecast(steps=1)
            pred = float(f.predicted_mean.iloc[0])
        except Exception:                              # noqa: BLE001
            pred = np.nan
        records.append({
            'variant_id': variant_id, 'date': t,
            'actual': actual, 'predicted': pred,
            'residual': actual - pred if not np.isnan(pred) else np.nan,
        })
        if i % 12 == 0 or i == len(test_idx):
            elapsed = time.perf_counter() - t_start
            n_ok = sum(1 for r in records if not np.isnan(r['predicted']))
            print(f'  [{variant_id}] {i:>3}/{len(test_idx)}  '
                  f'elapsed={elapsed:6.1f}s  refit_ok={n_ok}/{i}',
                  flush=True)
    return pd.DataFrame(records)


def window_errors(forecast_df: pd.DataFrame) -> pd.DataFrame:
    def _rmse_mae(sub: pd.DataFrame, label: str) -> Dict:
        resid = sub['residual'].dropna()
        if len(resid) == 0:
            return {'window': label, 'n': 0,
                    'rmse': np.nan, 'mae': np.nan, 'bias': np.nan}
        return {'window': label, 'n': int(len(resid)),
                'rmse': float(np.sqrt((resid ** 2).mean())),
                'mae':  float(resid.abs().mean()),
                'bias': float(resid.mean())}
    vid = forecast_df['variant_id'].iloc[0]
    rows: List[Dict] = []
    dates = forecast_df['date']
    for sub, label in [
        (forecast_df, 'full_test'),
        (forecast_df[(dates >= COVID_START) & (dates <= COVID_END)],
         'covid_2020_2021'),
        (forecast_df[dates >= ENERGY_START], 'energy_2022_plus'),
    ]:
        rec = _rmse_mae(sub, label)
        rec['variant_id'] = vid
        rows.append(rec)
    return pd.DataFrame(rows)[['variant_id', 'window', 'n',
                               'rmse', 'mae', 'bias']]


# ─────────────────────────────────────────────────────────────
# Q=3 sweep + union selection
# ─────────────────────────────────────────────────────────────
def run_q3_sweep(y_train: pd.Series) -> pd.DataFrame:
    orders = list(itertools.product(
        GRID_P, GRID_D, GRID_Q, GRID_SP, GRID_SD, GRID_SQ_EXTENSION
    ))
    total = len(orders)
    print(f'\n=== [{VARIANT_ID}] Q=3 extension sweep: {total} orders '
          f'over n_train={len(y_train)} '
          f'({y_train.index.min().date()}..{y_train.index.max().date()}) ===',
          flush=True)
    rows: List[Dict] = []
    t_start = time.perf_counter()
    for i, (p, d, q, P, D, Q) in enumerate(orders, start=1):
        fit_row = fit_sarimax(y_train, (p, d, q), (P, D, Q, SEASONAL_PERIOD))
        fit_row.pop('_fit', None)
        rows.append(fit_row)
        if i % 25 == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            best_aic = np.nanmin([r['aic'] for r in rows])
            conv = sum(1 for r in rows if r['converged'])
            print(f'  [{VARIANT_ID}] {i:>3}/{total}  '
                  f'elapsed={elapsed:6.1f}s  conv={conv}/{i}  '
                  f'best_AIC={best_aic:8.3f}', flush=True)
    return pd.DataFrame(rows)


def _select_by(df: pd.DataFrame, metric: str) -> pd.Series:
    valid = df[df['converged'] & df[metric].notna()].copy()
    if valid.empty:
        valid = df[df[metric].notna()].copy()
    valid = valid.sort_values([metric, 'n_params'], ascending=[True, True])
    return valid.iloc[0]


def _ord_str(row: pd.Series) -> str:
    return (f'({int(row.p)},{int(row.d)},{int(row.q)})'
            f'({int(row.P)},{int(row.D)},{int(row.Q)},{int(row.s)})')


def select_best_union(grid_union: pd.DataFrame) -> Dict:
    best_aic  = _select_by(grid_union, 'aic')
    best_bic  = _select_by(grid_union, 'bic')
    best_hqic = _select_by(grid_union, 'hqic')
    aic_bic_agree  = _ord_str(best_aic) == _ord_str(best_bic)
    aic_hqic_agree = _ord_str(best_aic) == _ord_str(best_hqic)
    return {
        'variant_id':      VARIANT_ID,
        'aic_best_order':  _ord_str(best_aic),
        'aic_value':       float(best_aic['aic']),
        'bic_best_order':  _ord_str(best_bic),
        'bic_value':       float(best_bic['bic']),
        'hqic_best_order': _ord_str(best_hqic),
        'hqic_value':      float(best_hqic['hqic']),
        'aic_bic_agree':   aic_bic_agree,
        'aic_hqic_agree':  aic_hqic_agree,
        'n_converged':     int(grid_union['converged'].sum()),
        'n_total':         int(len(grid_union)),
        '_best_order':     (int(best_aic.p), int(best_aic.d), int(best_aic.q)),
        '_best_seasonal_order': (int(best_aic.P), int(best_aic.D),
                                 int(best_aic.Q), int(best_aic.s)),
    }


# ─────────────────────────────────────────────────────────────
# In-place updates to consolidated CSVs
# ─────────────────────────────────────────────────────────────
def update_csv_row(
    csv_path: Path,
    key_col: str,
    key_value: str,
    new_rows: pd.DataFrame,
) -> None:
    """Replace all rows in csv_path where `key_col == key_value`
    with `new_rows`.  Preserves column order and other rows."""
    existing = pd.read_csv(csv_path)
    kept = existing[existing[key_col] != key_value].copy()
    # Ensure new_rows columns match existing schema
    new_rows_aligned = new_rows.reindex(columns=existing.columns)
    updated = pd.concat([kept, new_rows_aligned], ignore_index=True)
    # Preserve original ordering: sort by key_col to match Step 1 order
    variant_order = existing[key_col].drop_duplicates().tolist()
    updated['_sort'] = updated[key_col].map(
        {v: i for i, v in enumerate(variant_order)}
    )
    updated = updated.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    updated.to_csv(csv_path, index=False)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    t_script = time.perf_counter()
    root    = find_project_root()
    doc_dir = root / 'data' / 'documentation'

    print(f'Project root: {root}')
    print(f'Doc dir:      {doc_dir}')
    print(f'Variant:      {VARIANT_ID}')
    print(f'Q=3 sweep:    {len(GRID_P)*len(GRID_D)*len(GRID_Q)*len(GRID_SP)*len(GRID_SD)*len(GRID_SQ_EXTENSION)} orders')

    # 1. Build variant
    y_full  = build_usa_first_diff(root).dropna().asfreq('MS')
    y_train = y_full.loc[:TRAIN_END]
    print(f'  USA_first_diff n={len(y_full)} '
          f'[{y_full.index.min().date()}..{y_full.index.max().date()}]  '
          f'train n={len(y_train)}')

    # 2. Q=3 sweep
    q3_df = run_q3_sweep(y_train)
    q3_path = doc_dir / 'phase6_step1c_arima_grid_usa_first_diff_q3.csv'
    q3_df.to_csv(q3_path, index=False)
    print(f'  wrote {q3_path.name}')

    # 3. Load Step 1 grid (Q∈[0,2]) and union
    step1_path = doc_dir / f'phase6_step1_arima_grid_{VARIANT_ID}.csv'
    step1_df = pd.read_csv(step1_path)
    # Align columns (Step 1 has no '_fit' key in serialized form)
    common_cols = [c for c in step1_df.columns if c in q3_df.columns]
    union_df = pd.concat(
        [step1_df[common_cols], q3_df[common_cols]],
        ignore_index=True,
    )
    print(f'  union grid: {len(step1_df)} (Step 1) + {len(q3_df)} (Q=3) '
          f'= {len(union_df)} orders')

    # 4. Select new best over union
    sel = select_best_union(union_df)
    # Identify Step 1 best for delta reporting
    step1_best = _select_by(step1_df, 'aic')
    step1_best_order = _ord_str(step1_best)
    step1_best_aic   = float(step1_best['aic'])
    delta_aic = sel['aic_value'] - step1_best_aic
    print(f'\n  Step 1 best: {step1_best_order}  AIC={step1_best_aic:.3f}')
    print(f'  Union best:  {sel["aic_best_order"]}  AIC={sel["aic_value"]:.3f}')
    print(f'  ΔAIC (union − Step 1): {delta_aic:+.3f}')
    print(f'  AIC-BIC agree: {sel["aic_bic_agree"]}  '
          f'AIC-HQIC agree: {sel["aic_hqic_agree"]}')

    # 5. Refit new best for diagnostics
    best_order  = sel['_best_order']
    best_sorder = sel['_best_seasonal_order']
    refit_row = fit_sarimax(y_train, best_order, best_sorder)
    if refit_row['_fit'] is None:
        print('  ⚠ refit failed; aborting diagnostics')
        return 1
    diag = residual_diagnostics(VARIANT_ID, refit_row['_fit'],
                                sel['aic_best_order'])
    print(f'  resid: LB_Q12_p={diag["ljungbox_q12_p"]:.4f}  '
          f'LB_Q24_p={diag["ljungbox_q24_p"]:.4f}  '
          f'JB_p={diag["jb_p"]:.4f}  ARCH_LM_p={diag["arch_lm_p"]:.4f}')

    # 6. Expanding-refit 1-step-ahead forecast
    fc_df = expanding_forecast(VARIANT_ID, y_full, best_order, best_sorder)
    win_df = window_errors(fc_df)
    print('  window errors:')
    for _, row in win_df.iterrows():
        print(f'    {row["window"]:<18}  n={row["n"]:>3}  '
              f'RMSE={row["rmse"]:.4f}  MAE={row["mae"]:.4f}  '
              f'bias={row["bias"]:+.4f}')

    # 7. In-place update consolidated CSVs
    sel_csv = doc_dir / 'phase6_step1_arima_selection.csv'
    res_csv = doc_dir / 'phase6_step1_arima_residuals.csv'
    fc_csv  = doc_dir / 'phase6_step1_arima_forecast.csv'
    win_csv = doc_dir / 'phase6_step1_arima_window_errors.csv'

    # Selection: one row per variant
    sel_row = pd.DataFrame([{k: v for k, v in sel.items()
                             if not k.startswith('_')}])
    update_csv_row(sel_csv, 'variant_id', VARIANT_ID, sel_row)
    print(f'  updated {sel_csv.name}')

    # Residuals: one row per variant
    res_row = pd.DataFrame([diag])
    update_csv_row(res_csv, 'variant_id', VARIANT_ID, res_row)
    print(f'  updated {res_csv.name}')

    # Forecast: multiple rows per variant
    update_csv_row(fc_csv, 'variant_id', VARIANT_ID, fc_df)
    print(f'  updated {fc_csv.name}')

    # Window errors: 3 rows per variant
    update_csv_row(win_csv, 'variant_id', VARIANT_ID, win_df)
    print(f'  updated {win_csv.name}')

    # 8. Summary delta CSV for portfolio trace
    delta_row = pd.DataFrame([{
        'variant_id':          VARIANT_ID,
        'step1_best_order':    step1_best_order,
        'step1_best_aic':      step1_best_aic,
        'step1c_best_order':   sel['aic_best_order'],
        'step1c_best_aic':     sel['aic_value'],
        'delta_aic':           delta_aic,
        'step1b_verdict':      'extend_to_Q3',
    }])
    delta_path = doc_dir / 'phase6_step1c_selection_delta.csv'
    delta_row.to_csv(delta_path, index=False)
    print(f'  wrote {delta_path.name}')

    print('\n' + '=' * 78)
    print('STEP 1c AMENDMENT SUMMARY')
    print('=' * 78)
    print(delta_row.to_string(index=False))

    total_sec = time.perf_counter() - t_script
    print(f'\nTotal runtime: {total_sec / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
