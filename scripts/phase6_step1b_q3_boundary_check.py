"""
phase6_step1b_q3_boundary_check.py
==================================

Phase 6 Step 1b — seasonal-Q boundary sensitivity check (D-048 sensitivity).

Motivation
----------
Step 1 grid used ``Q ∈ [0, 2]``.  Three of five variants selected the
AIC-best order at the boundary ``Q = 2``:

    * USA_yoy_pct      (2,0,3)(2,0,2,12)   — BOTH P=2 AND Q=2 at boundary
    * USA_first_diff   (0,0,3)(0,0,2,12)   — Q=2 at boundary
    * UK_log_diff_pct  (3,0,0)(1,0,2,12)   — Q=2 at boundary

This script tests whether ``Q = 3`` (and, for completeness, a few
surrounding (p,q,P,D) neighbours with Q=3) meaningfully improves AIC.

Decision rule (Burnham & Anderson 2002 threshold)
-------------------------------------------------
For each boundary-hit variant:

* ΔAIC < 2  → current ``Q=2`` bound accepted; extension has "essentially
              equivalent support" and Step 1 selection is locked
              (D-048 final, ``"sensitivity verified"``).
* ΔAIC ≥ 2  → Q=3 extension is meaningfully better; full grid re-run
              with ``Q ∈ [0, 3]`` is triggered (D-048 amendment).

Japan and Germany are NOT re-tested: neither hit any seasonal boundary
in Step 1, so extending the grid would only add null-result orders.

Outputs (data/documentation/)
-----------------------------
phase6_step1b_boundary_check_{variant}.csv
    Per-variant: all candidate orders evaluated (incl. the original
    AIC-best as a sanity-check reproduction), columns match Step 1
    grid CSVs plus ``delta_aic_vs_original`` and ``is_original``.

phase6_step1b_boundary_check_summary.csv
    One row per variant: original AIC-best, best Q=3 extension, ΔAIC,
    and verdict ('accept_Q2' | 'extend_to_Q3').

Decision references: D-033 (precedent: Quandt-Andrews trim robustness),
D-048 (parent grid spec), D-049 (candidate: Japan ARIMA uniqueness —
unrelated to boundary sensitivity but recorded adjacently).

Runtime estimate: ~21 fits at 15-30 s each ≈ 6-10 min.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Path bootstrap (same pattern as Step 1)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import (
    build_all_features,
    find_project_root,
    first_difference,
    load_processed_main,
)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
TRAIN_END:       pd.Timestamp = pd.Timestamp('2019-12-01')
SEASONAL_PERIOD: int          = 12
SARIMAX_METHOD:  str          = 'lbfgs'
SARIMAX_MAXITER: int          = 200

#: ΔAIC threshold above which the Q=3 extension is judged to offer
#: "meaningfully better" support than the Q=2-bounded grid.
#: 2.0 = Burnham & Anderson (2002) convention.
DELTA_AIC_ACCEPT_THRESHOLD: float = 2.0


# ─────────────────────────────────────────────────────────────
# Boundary-neighborhood specification
#
# For each boundary-hit variant, we evaluate:
#   (a) The original Step 1 AIC-best order (sanity-check reproduction)
#   (b) The same (p,q,P,D) with Q=3 to test pure-Q extension
#   (c) Q=3 with P swept over {0, 1, 2}
#   (d) Q=3 with D=1 (seasonal difference)
#   (e) Q=3 with a small (p or q)±1 perturbation
#
# Each order tuple: (p, d, q, P, D, Q).
# ─────────────────────────────────────────────────────────────
NEIGHBORHOOD_SPEC: Dict[str, Dict] = {
    'USA_yoy_pct': {
        'original': (2, 0, 3, 2, 0, 2),
        'original_aic': 61.750913,       # reproduced from Step 1 CSV
        'extensions': [
            (2, 0, 3, 2, 0, 3),          # pure Q=3
            (2, 0, 3, 1, 0, 3),          # Q=3, lower P
            (2, 0, 3, 0, 0, 3),          # Q=3, P=0
            (2, 0, 3, 2, 1, 3),          # Q=3, D=1
            (3, 0, 3, 2, 0, 3),          # Q=3, p+1
            (2, 0, 4, 2, 0, 3),          # Q=3, q+1
            (1, 0, 3, 2, 0, 3),          # Q=3, p-1
        ],
    },
    'USA_first_diff': {
        'original': (0, 0, 3, 0, 0, 2),
        'original_aic': 340.105984,
        'extensions': [
            (0, 0, 3, 0, 0, 3),
            (0, 0, 3, 1, 0, 3),
            (0, 0, 3, 2, 0, 3),
            (0, 0, 3, 0, 1, 3),
            (0, 0, 4, 0, 0, 3),
            (1, 0, 3, 0, 0, 3),
        ],
    },
    'UK_log_diff_pct': {
        'original': (3, 0, 0, 1, 0, 2),
        'original_aic': -119.146293,
        'extensions': [
            (3, 0, 0, 1, 0, 3),
            (3, 0, 0, 2, 0, 3),
            (3, 0, 0, 0, 0, 3),
            (3, 0, 0, 1, 1, 3),
            (4, 0, 0, 1, 0, 3),
            (3, 0, 1, 1, 0, 3),
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# Variant construction (duplicate of Step 1 for script independence;
# intentional — D-038 parallel, scripts are self-contained artifacts)
# ─────────────────────────────────────────────────────────────
def _force_monthly_freq(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).to_period('M').to_timestamp(how='start')
    return out.asfreq('MS')


def build_variants_for_check(project_root: Path) -> Dict[str, pd.Series]:
    """Return {variant_id: y_full} for only the boundary-hit variants."""
    features = build_all_features(project_root=project_root)
    out: Dict[str, pd.Series] = {}

    out['USA_yoy_pct'] = _force_monthly_freq(
        features['USA']['USA_CPI'].dropna()
    )
    usa_level = load_processed_main('USA', project_root=project_root)['USA_CPI']
    out['USA_first_diff'] = _force_monthly_freq(first_difference(usa_level))
    out['UK_log_diff_pct'] = _force_monthly_freq(
        features['UK']['UK_CPI'].dropna()
    )
    return out


# ─────────────────────────────────────────────────────────────
# Single-fit wrapper (Step 1 logic, verbatim)
# ─────────────────────────────────────────────────────────────
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
            fit = mod.fit(method=SARIMAX_METHOD,
                          maxiter=SARIMAX_MAXITER, disp=False)
        row['aic']  = float(fit.aic)
        row['bic']  = float(fit.bic)
        row['hqic'] = float(fit.hqic)
        row['llf']  = float(fit.llf)
        retvals = getattr(fit, 'mle_retvals', {}) or {}
        row['converged'] = bool(retvals.get('converged', False))
    except Exception as exc:                       # noqa: BLE001
        row['error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
    row['runtime_sec'] = time.perf_counter() - t0
    return row


# ─────────────────────────────────────────────────────────────
# Per-variant check
# ─────────────────────────────────────────────────────────────
def check_variant(
    variant_id: str,
    y_full: pd.Series,
    spec: Dict,
    out_dir: Path,
) -> Dict:
    """Fit original + all extensions; return summary row."""
    y_train = y_full.loc[:TRAIN_END].dropna().asfreq('MS')
    n_train = len(y_train)

    print(f'\n=== [{variant_id}] boundary check, n_train={n_train} ===',
          flush=True)

    orders_to_fit: List[Tuple[bool, Tuple[int, ...]]] = [
        (True, spec['original']),
        *[(False, order) for order in spec['extensions']],
    ]

    rows: List[Dict] = []
    for is_original, order6 in orders_to_fit:
        p, d, q, P, D, Q = order6
        fit_row = fit_sarimax(
            y_train, (p, d, q), (P, D, Q, SEASONAL_PERIOD)
        )
        fit_row['is_original'] = is_original
        rows.append(fit_row)
        order_str = f'({p},{d},{q})({P},{D},{Q},12)'
        tag = 'ORIG' if is_original else '    '
        print(f'  {tag} {order_str:<22}  '
              f'AIC={fit_row["aic"]:>10.3f}  '
              f'BIC={fit_row["bic"]:>10.3f}  '
              f'conv={fit_row["converged"]}  '
              f'{fit_row["runtime_sec"]:5.1f}s',
              flush=True)

    df = pd.DataFrame(rows)

    # Sanity check: original AIC should match the Step 1 CSV value
    orig_row = df[df['is_original']].iloc[0]
    step1_aic = spec['original_aic']
    orig_aic  = orig_row['aic']
    aic_repro_diff = abs(orig_aic - step1_aic)
    if aic_repro_diff > 0.01:
        print(f'  ⚠ reproduction mismatch: Step1 AIC={step1_aic:.3f}, '
              f'here AIC={orig_aic:.3f}, Δ={aic_repro_diff:.4f}',
              flush=True)
    else:
        print(f'  ✓ original AIC reproduced (Δ={aic_repro_diff:.4f})',
              flush=True)

    # Compute ΔAIC vs original for each row
    df['delta_aic_vs_original'] = df['aic'] - orig_aic

    # Reorder for readability
    col_order = [
        'is_original', 'p', 'd', 'q', 'P', 'D', 'Q', 's',
        'n_params', 'aic', 'bic', 'hqic', 'llf',
        'delta_aic_vs_original', 'converged', 'runtime_sec', 'error',
    ]
    df = df[col_order]

    # Write per-variant CSV
    csv_path = out_dir / f'phase6_step1b_boundary_check_{variant_id}.csv'
    df.to_csv(csv_path, index=False)
    print(f'  wrote {csv_path.name}')

    # Summary: best extension vs original
    ext_df = df[~df['is_original'] & df['converged']].copy()
    if ext_df.empty:
        ext_df = df[~df['is_original']].copy()          # fallback
    best_ext = ext_df.sort_values('aic').iloc[0]
    best_ext_order = (f'({int(best_ext.p)},{int(best_ext.d)},{int(best_ext.q)})'
                      f'({int(best_ext.P)},{int(best_ext.D)},'
                      f'{int(best_ext.Q)},{int(best_ext.s)})')
    orig_order_str = (f'({int(orig_row.p)},{int(orig_row.d)},{int(orig_row.q)})'
                      f'({int(orig_row.P)},{int(orig_row.D)},'
                      f'{int(orig_row.Q)},{int(orig_row.s)})')
    delta = best_ext['aic'] - orig_aic
    verdict = ('extend_to_Q3'
               if delta <= -DELTA_AIC_ACCEPT_THRESHOLD
               else 'accept_Q2')

    print(f'  → best extension: {best_ext_order}  '
          f'AIC={best_ext["aic"]:.3f}  ΔAIC={delta:+.3f}  '
          f'verdict={verdict}', flush=True)

    return {
        'variant_id':               variant_id,
        'original_order':           orig_order_str,
        'original_aic':             float(orig_aic),
        'best_extension_order':     best_ext_order,
        'best_extension_aic':       float(best_ext['aic']),
        'delta_aic':                float(delta),
        'threshold':                DELTA_AIC_ACCEPT_THRESHOLD,
        'verdict':                  verdict,
        'n_extensions_tested':      int(len(ext_df)),
        'n_extensions_converged':   int(ext_df['converged'].sum()),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    t_start = time.perf_counter()
    root    = find_project_root()
    out_dir = root / 'data' / 'documentation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Project root: {root}')
    print(f'Output dir:   {out_dir}')
    print(f'ΔAIC acceptance threshold: {DELTA_AIC_ACCEPT_THRESHOLD}')
    print(f'Variants under check: {list(NEIGHBORHOOD_SPEC.keys())}')

    variants = build_variants_for_check(root)

    summary_rows: List[Dict] = []
    for variant_id, spec in NEIGHBORHOOD_SPEC.items():
        y_full = variants[variant_id]
        summary = check_variant(variant_id, y_full, spec, out_dir)
        summary_rows.append(summary)

    # Consolidated summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / 'phase6_step1b_boundary_check_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\nWrote {summary_path.name}')

    print('\n' + '=' * 78)
    print('BOUNDARY CHECK SUMMARY')
    print('=' * 78)
    print(summary_df.to_string(index=False))

    # Decision aggregate
    verdicts = summary_df['verdict'].tolist()
    if all(v == 'accept_Q2' for v in verdicts):
        aggregate = ('ALL VARIANTS accept_Q2 — Step 1 selection LOCKED; '
                     'D-048 final with sensitivity verified.')
    elif any(v == 'extend_to_Q3' for v in verdicts):
        who = [r['variant_id'] for _, r in summary_df.iterrows()
               if r['verdict'] == 'extend_to_Q3']
        aggregate = (f'ESCALATE — {who} require Q=3 extension; '
                     'full grid re-run triggered (D-048 amendment).')
    else:
        aggregate = 'UNEXPECTED — inspect verdicts manually.'

    print(f'\n>>> DECISION: {aggregate}')

    total = time.perf_counter() - t_start
    print(f'\nTotal runtime: {total / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
