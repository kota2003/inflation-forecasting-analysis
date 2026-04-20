"""
phase6_step1_arima_grid.py
==========================

Phase 6 Step 1 — SARIMA baseline grid search per D-048.

Grid specification (D-048):
    non-seasonal:  p ∈ [0, 4], d = 0, q ∈ [0, 4]
    seasonal:      P ∈ [0, 2], D ∈ {0, 1}, Q ∈ [0, 2], s = 12
    total:         5 × 1 × 5 × 3 × 2 × 3 = 450 orders per variant

Five variants (per D-031 + USA dual-form requirement for D-053):
    USA_yoy_pct         USA CPI in D-031 canonical form
    USA_first_diff      USA CPI in alternative first-diff form (dual-form comparison)
    JAPAN_first_diff    D-031 canonical (revised from yoy_pct per D-031)
    UK_log_diff_pct     D-031 canonical
    GERMANY_first_diff  D-031 canonical

Selection:
    Primary:   AIC
    Secondary: BIC (reported for model-selection robustness)
    Tertiary:  HQIC (audit only)
    Tie-break: parsimony (p + q + P + Q)

Train / test split per D-005:
    Train: series start ── 2019-12-01 inclusive
    Test:  2020-01-01 onward

Forecast protocol:
    Expanding-window 1-step-ahead refit at each test timestep.
    RMSE / MAE reported for full test window + sub-windows
    {COVID: 2020-01 .. 2021-12, ENERGY+: 2022-01 .. end}.

Outputs (data/documentation/):
    phase6_step1_arima_grid_{variant_id}.csv    450 rows, all grid fits
    phase6_step1_arima_selection.csv            5 rows × best orders + agreement flags
    phase6_step1_arima_residuals.csv            5 rows × diagnostic p-values
    phase6_step1_arima_forecast.csv             test-window 1-step-ahead predictions

Decision references: D-005, D-031, D-048.

Runtime (indicative): grid search ≈ 30–60 min; expanding refit ≈ 10–20 min.
Invoke from project root: ``python scripts/phase6_step1_arima_grid.py``.
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Path bootstrap — make the repo-root importable regardless of
# whether this script is invoked via ``python scripts/...``,
# double-clicked, or executed from an IDE with cwd set to scripts/.
# Matches the pattern used by sibling phase{3,4,5}_*.py scripts.
# ─────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Project package
from src import (
    build_all_features,
    find_project_root,
    first_difference,
    load_processed_main,
)


# ─────────────────────────────────────────────────────────────
# D-048 Grid specification
# ─────────────────────────────────────────────────────────────
GRID_P:  Tuple[int, ...] = (0, 1, 2, 3, 4)
GRID_D:  Tuple[int, ...] = (0,)                # D-031: inputs already stationary
GRID_Q:  Tuple[int, ...] = (0, 1, 2, 3, 4)
GRID_SP: Tuple[int, ...] = (0, 1, 2)
GRID_SD: Tuple[int, ...] = (0, 1)              # seasonal random walk permitted
GRID_SQ: Tuple[int, ...] = (0, 1, 2)
SEASONAL_PERIOD: int = 12

# ─────────────────────────────────────────────────────────────
# D-005 Train / test split
# ─────────────────────────────────────────────────────────────
TRAIN_END:  pd.Timestamp = pd.Timestamp('2019-12-01')
TEST_START: pd.Timestamp = pd.Timestamp('2020-01-01')

# Sub-windows for stratified error reporting
COVID_START:  pd.Timestamp = pd.Timestamp('2020-01-01')
COVID_END:    pd.Timestamp = pd.Timestamp('2021-12-01')
ENERGY_START: pd.Timestamp = pd.Timestamp('2022-01-01')    # end = series end

# ─────────────────────────────────────────────────────────────
# Diagnostic configuration
# ─────────────────────────────────────────────────────────────
LJUNG_BOX_LAGS: Tuple[int, ...] = (12, 24)     # seasonal-harmonic spacing
ARCH_LAGS: int = 12
SARIMAX_METHOD: str = 'lbfgs'
SARIMAX_MAXITER: int = 200


# ─────────────────────────────────────────────────────────────
# Variant construction
# ─────────────────────────────────────────────────────────────
VariantSeries = Tuple[str, pd.Series]  # (variant_id, y)


def _force_monthly_freq(s: pd.Series) -> pd.Series:
    """Ensure DatetimeIndex has freq='MS' for statsmodels compatibility."""
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).to_period('M').to_timestamp(how='start')
    out = out.asfreq('MS')
    return out


def build_variants(
    project_root: Optional[Path] = None,
) -> List[VariantSeries]:
    """Build the five (variant_id, y) pairs per D-048 §3.

    USA yoy_pct / JPN / UK / GER canonical forms come from
    `build_all_features()` (already D-031 corrected).  USA first_diff
    is constructed on the fly from level CPI via `first_difference`.
    """
    features = build_all_features(project_root=project_root)
    variants: List[VariantSeries] = []

    # USA canonical (yoy_pct) — 12-month overlap artifact flagged in Phase 5 S4
    usa_yoy = features['USA']['USA_CPI'].dropna().copy()
    variants.append(('USA_yoy_pct', _force_monthly_freq(usa_yoy)))

    # USA dual-form (first_diff) — from level CPI, D-053 candidate
    usa_level = load_processed_main('USA', project_root=project_root)['USA_CPI']
    usa_diff = first_difference(usa_level)
    variants.append(('USA_first_diff', _force_monthly_freq(usa_diff)))

    # JAPAN (first_diff per D-031 revision)
    jpn = features['JAPAN']['JAPAN_CPI'].dropna().copy()
    variants.append(('JAPAN_first_diff', _force_monthly_freq(jpn)))

    # UK (log_diff_pct per D-031)
    uk = features['UK']['UK_CPI'].dropna().copy()
    variants.append(('UK_log_diff_pct', _force_monthly_freq(uk)))

    # GERMANY (first_diff per D-031)
    ger = features['GERMANY']['GERMANY_CPI'].dropna().copy()
    variants.append(('GERMANY_first_diff', _force_monthly_freq(ger)))

    return variants


# ─────────────────────────────────────────────────────────────
# Single-fit wrapper
# ─────────────────────────────────────────────────────────────
def fit_sarimax(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    trend: str = 'c',
) -> Dict:
    """Fit one SARIMAX model.  Never raises; failures recorded in result dict.

    Returns a flat dict safe for pd.DataFrame construction.  The fitted
    result object itself is included under key ``_fit`` for downstream
    residual / forecast work; callers must pop it before serialising.
    """
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
                y,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mod.fit(method=SARIMAX_METHOD, maxiter=SARIMAX_MAXITER, disp=False)
        row['aic']  = float(fit.aic)
        row['bic']  = float(fit.bic)
        row['hqic'] = float(fit.hqic)
        row['llf']  = float(fit.llf)
        retvals = getattr(fit, 'mle_retvals', {}) or {}
        row['converged'] = bool(retvals.get('converged', False))
        row['_fit'] = fit
    except Exception as exc:                   # noqa: BLE001
        row['error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
    row['runtime_sec'] = time.perf_counter() - t0
    return row


# ─────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────
def run_grid(
    variant_id: str,
    y_train: pd.Series,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Run the full 450-model grid; returns tidy DataFrame."""
    orders = list(itertools.product(
        GRID_P, GRID_D, GRID_Q, GRID_SP, GRID_SD, GRID_SQ
    ))
    total = len(orders)
    print(f'\n=== [{variant_id}] grid search: {total} orders over '
          f'n_train={len(y_train)} ({y_train.index.min().date()}..'
          f'{y_train.index.max().date()}) ===', flush=True)

    rows: List[Dict] = []
    t_start = time.perf_counter()
    for i, (p, d, q, P, D, Q) in enumerate(orders, start=1):
        fit_row = fit_sarimax(y_train, (p, d, q), (P, D, Q, SEASONAL_PERIOD))
        fit_row.pop('_fit', None)              # drop heavy object before storing
        rows.append(fit_row)
        if i % progress_every == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            best_aic = np.nanmin([r['aic'] for r in rows])
            conv = sum(1 for r in rows if r['converged'])
            print(f'  [{variant_id}] {i:>3}/{total}  '
                  f'elapsed={elapsed:6.1f}s  '
                  f'converged={conv}/{i}  '
                  f'best_AIC={best_aic:8.3f}', flush=True)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────
def _select_by(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return the row minimising `metric`, with parsimony tie-break."""
    valid = df[df['converged'] & df[metric].notna()].copy()
    if valid.empty:
        # fall back to any non-NaN row (convergence flag unreliable sometimes)
        valid = df[df[metric].notna()].copy()
    valid = valid.sort_values([metric, 'n_params'], ascending=[True, True])
    return valid.iloc[0]


def select_best(variant_id: str, grid_df: pd.DataFrame) -> Dict:
    """Pick AIC / BIC / HQIC best orders; record agreement flags."""
    best_aic  = _select_by(grid_df, 'aic')
    best_bic  = _select_by(grid_df, 'bic')
    best_hqic = _select_by(grid_df, 'hqic')

    def _ord_str(row):
        return (f'({int(row.p)},{int(row.d)},{int(row.q)})'
                f'({int(row.P)},{int(row.D)},{int(row.Q)},{int(row.s)})')

    aic_bic_agree  = _ord_str(best_aic) == _ord_str(best_bic)
    aic_hqic_agree = _ord_str(best_aic) == _ord_str(best_hqic)

    return {
        'variant_id':        variant_id,
        'aic_best_order':    _ord_str(best_aic),
        'aic_value':         float(best_aic['aic']),
        'bic_best_order':    _ord_str(best_bic),
        'bic_value':         float(best_bic['bic']),
        'hqic_best_order':   _ord_str(best_hqic),
        'hqic_value':        float(best_hqic['hqic']),
        'aic_bic_agree':     aic_bic_agree,
        'aic_hqic_agree':    aic_hqic_agree,
        'n_converged':       int(grid_df['converged'].sum()),
        'n_total':           int(len(grid_df)),
        # auxiliary for downstream refit
        '_best_order':         (int(best_aic.p), int(best_aic.d), int(best_aic.q)),
        '_best_seasonal_order': (int(best_aic.P), int(best_aic.D),
                                 int(best_aic.Q), int(best_aic.s)),
    }


# ─────────────────────────────────────────────────────────────
# Residual diagnostics
# ─────────────────────────────────────────────────────────────
def residual_diagnostics(variant_id: str, fit_result) -> Dict:
    """Ljung-Box, Jarque-Bera, ARCH-LM on training residuals."""
    resid = pd.Series(fit_result.resid).dropna()
    lb = acorr_ljungbox(resid, lags=list(LJUNG_BOX_LAGS), return_df=True)
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    try:
        arch_lm, arch_p, *_ = het_arch(resid, nlags=ARCH_LAGS)
    except Exception:                           # noqa: BLE001
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
    return out


# ─────────────────────────────────────────────────────────────
# Expanding-window 1-step-ahead forecast
# ─────────────────────────────────────────────────────────────
def expanding_forecast(
    variant_id: str,
    y_full: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> pd.DataFrame:
    """Refit SARIMAX at each test timestep with data up to t-1,
    forecast y_t, record (actual, predicted, residual).

    Uses the same order/seasonal_order across all refits (chosen on
    training data only; this is Step 1 baseline -- Phase 7 formal
    evaluation may revisit).
    """
    test_idx = y_full.index[y_full.index >= TEST_START]
    print(f'\n--- [{variant_id}] expanding-refit 1-step forecast: '
          f'{len(test_idx)} steps ---', flush=True)

    records: List[Dict] = []
    t_start = time.perf_counter()
    for i, t in enumerate(test_idx, start=1):
        y_hist = y_full.loc[:t].iloc[:-1]       # strictly before t
        actual = float(y_full.loc[t])
        pred   = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mod = SARIMAX(
                    y_hist,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = mod.fit(method=SARIMAX_METHOD,
                              maxiter=SARIMAX_MAXITER, disp=False)
            f = fit.get_forecast(steps=1)
            pred = float(f.predicted_mean.iloc[0])
        except Exception:                       # noqa: BLE001
            pred = np.nan
        records.append({
            'variant_id': variant_id,
            'date': t,
            'actual': actual,
            'predicted': pred,
            'residual': actual - pred if not np.isnan(pred) else np.nan,
        })
        if i % 12 == 0 or i == len(test_idx):
            elapsed = time.perf_counter() - t_start
            n_ok = sum(1 for r in records if not np.isnan(r['predicted']))
            print(f'  [{variant_id}] {i:>3}/{len(test_idx)}  '
                  f'elapsed={elapsed:6.1f}s  refit_ok={n_ok}/{i}',
                  flush=True)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Sub-window error metrics
# ─────────────────────────────────────────────────────────────
def window_errors(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE / MAE over full test, COVID, and ENERGY+ sub-windows."""
    def _rmse_mae(sub: pd.DataFrame, label: str) -> Dict:
        resid = sub['residual'].dropna()
        if len(resid) == 0:
            return {'window': label, 'n': 0,
                    'rmse': np.nan, 'mae': np.nan,
                    'bias': np.nan}
        return {
            'window': label,
            'n':      int(len(resid)),
            'rmse':   float(np.sqrt((resid ** 2).mean())),
            'mae':    float(resid.abs().mean()),
            'bias':   float(resid.mean()),
        }

    vid = forecast_df['variant_id'].iloc[0]
    rows: List[Dict] = []
    dates = forecast_df['date']

    full = forecast_df
    covid = forecast_df[(dates >= COVID_START) & (dates <= COVID_END)]
    energy = forecast_df[dates >= ENERGY_START]

    for df_sub, label in [(full, 'full_test'),
                          (covid, 'covid_2020_2021'),
                          (energy, 'energy_2022_plus')]:
        rec = _rmse_mae(df_sub, label)
        rec['variant_id'] = vid
        rows.append(rec)
    return pd.DataFrame(rows)[['variant_id', 'window', 'n', 'rmse', 'mae', 'bias']]


# ─────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────
def main() -> int:
    t_script_start = time.perf_counter()
    root = find_project_root()
    out_dir = root / 'data' / 'documentation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Project root: {root}')
    print(f'Output dir:   {out_dir}')
    print(f'Grid: {len(GRID_P)*len(GRID_D)*len(GRID_Q)*len(GRID_SP)*len(GRID_SD)*len(GRID_SQ)} '
          f'orders/variant × 5 variants')

    variants = build_variants(project_root=root)
    print(f'\nVariants built: {[v[0] for v in variants]}')
    for vid, y in variants:
        print(f'  {vid:<22} n={len(y):>3}  '
              f'[{y.index.min().date()} .. {y.index.max().date()}]  '
              f'mean={y.mean():+.4f}  std={y.std(ddof=1):.4f}')

    selection_rows:  List[Dict] = []
    residual_rows:   List[Dict] = []
    forecast_frames: List[pd.DataFrame] = []
    window_frames:   List[pd.DataFrame] = []

    for vid, y_full in variants:
        y_full = y_full.dropna().asfreq('MS')
        y_train = y_full.loc[:TRAIN_END]

        # 1. Grid search on training data
        grid_df = run_grid(vid, y_train)
        grid_csv = out_dir / f'phase6_step1_arima_grid_{vid}.csv'
        grid_df.to_csv(grid_csv, index=False)
        print(f'  wrote {grid_csv.name}')

        # 2. Select best order
        sel = select_best(vid, grid_df)
        selection_rows.append({k: v for k, v in sel.items()
                               if not k.startswith('_')})
        print(f'  [{vid}] AIC best: {sel["aic_best_order"]}  '
              f'AIC={sel["aic_value"]:.3f} | '
              f'BIC best: {sel["bic_best_order"]}  '
              f'BIC={sel["bic_value"]:.3f} | '
              f'AIC-BIC agree: {sel["aic_bic_agree"]}')

        # 3. Refit the AIC-best on the full training sample for diagnostics
        best_order = sel['_best_order']
        best_sorder = sel['_best_seasonal_order']
        refit_row = fit_sarimax(y_train, best_order, best_sorder)
        if refit_row['_fit'] is None:
            print(f'  [{vid}] WARNING: AIC-best refit failed; '
                  f'diagnostics skipped.')
        else:
            diag = residual_diagnostics(vid, refit_row['_fit'])
            diag['aic_best_order'] = sel['aic_best_order']
            residual_rows.append(diag)
            print(f'  [{vid}] resid: LB_Q12_p={diag["ljungbox_q12_p"]:.4f}  '
                  f'JB_p={diag["jb_p"]:.4f}  '
                  f'ARCH_LM_p={diag["arch_lm_p"]:.4f}')

        # 4. Expanding-window 1-step-ahead forecast on test window
        fc_df = expanding_forecast(vid, y_full, best_order, best_sorder)
        forecast_frames.append(fc_df)

        # 5. Window error metrics
        win_df = window_errors(fc_df)
        window_frames.append(win_df)
        print(f'  [{vid}] RMSE/MAE:')
        for _, row in win_df.iterrows():
            print(f'    {row["window"]:<18}  n={row["n"]:>3}  '
                  f'RMSE={row["rmse"]:.4f}  MAE={row["mae"]:.4f}  '
                  f'bias={row["bias"]:+.4f}')

    # ── Consolidated outputs ──────────────────────────────────
    sel_df  = pd.DataFrame(selection_rows)
    res_df  = pd.DataFrame(residual_rows)
    fc_df   = pd.concat(forecast_frames, ignore_index=True)
    win_df  = pd.concat(window_frames,   ignore_index=True)

    sel_path = out_dir / 'phase6_step1_arima_selection.csv'
    res_path = out_dir / 'phase6_step1_arima_residuals.csv'
    fc_path  = out_dir / 'phase6_step1_arima_forecast.csv'
    win_path = out_dir / 'phase6_step1_arima_window_errors.csv'

    sel_df.to_csv(sel_path, index=False)
    res_df.to_csv(res_path, index=False)
    fc_df.to_csv(fc_path,   index=False)
    win_df.to_csv(win_path, index=False)

    print(f'\nWrote:')
    for p in (sel_path, res_path, fc_path, win_path):
        print(f'  {p.name}')

    # Summary preview
    print('\n' + '=' * 72)
    print('SELECTION SUMMARY')
    print('=' * 72)
    print(sel_df.to_string(index=False))

    print('\n' + '=' * 72)
    print('RESIDUAL DIAGNOSTICS')
    print('=' * 72)
    if not res_df.empty:
        preview_cols = ['variant_id', 'aic_best_order',
                        'ljungbox_q12_p', 'ljungbox_q24_p',
                        'jb_p', 'arch_lm_p']
        print(res_df[preview_cols].to_string(index=False))

    print('\n' + '=' * 72)
    print('WINDOW ERRORS (expanding-refit 1-step-ahead)')
    print('=' * 72)
    print(win_df.to_string(index=False))

    total_sec = time.perf_counter() - t_script_start
    print(f'\nTotal runtime: {total_sec / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
