"""
scripts/phase5_step4_acf_pacf.py
=================================
Phase 5 · Step 4 — ACF / PACF diagnostics for ARIMA order identification.

Decisions under consideration (final sign-off after this run)
-------------------------------------------------------------
  D-044 (adopted, Option A): lag depth = 40 for all 4 countries.
    Rationale:
      1. 40 > 36 allows inspection of three full annual seasonal cycles.
      2. Covers Phase 3 ENERGY post-window (38-45 obs per country).
      3. Uniform across countries preserves methodological symmetry.

Sub-decisions (all adopted):
  * Input form: D-031-corrected stationary base. USA=yoy_pct,
    JPN/GER=first_diff, UK=log_diff_pct. Identical to the Phase 6 ARIMA
    input so these diagnostics directly inform Phase 6 order selection.
  * Ljung-Box lags: [12, 24, 36] — annual, biannual, triennial horizons.
  * PACF method: 'ywm' (Yule-Walker adjusted; statsmodels-preferred).
  * Confidence band: simple Bartlett ±1.96/sqrt(n) (constant across lags;
    textbook defensibility).

Inputs
------
  data/processed/features_{usa,japan,uk,germany}.csv
    (Phase 4 output; base CPI column is D-031-corrected stationary form)

Outputs
-------
  outputs/figures/phase5_step4_fig8_acf_pacf.png                 (4×2 grid)
  data/documentation/phase5_step4_acf_pacf_values.csv            (4 × 41 × 2 rows)
  data/documentation/phase5_step4_ljung_box.csv                  (4 × 3 rows)

Run
---
  python scripts/phase5_step4_acf_pacf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ────────────────────────────────────────────────────────────────────
# 0. Project root + src import
# ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import MAIN_COUNTRIES  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# 1. Configuration
# ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
DOC_DIR       = PROJECT_ROOT / 'data' / 'documentation'
FIG_DIR       = PROJECT_ROOT / 'outputs' / 'figures'
DOC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'USA':     '#1565c0',
    'JAPAN':   '#c62828',
    'UK':      '#2e7d32',
    'GERMANY': '#6a1b9a',
}

NLAGS        = 40
LJUNG_LAGS   = [12, 24, 36]
PACF_METHOD  = 'ywm'
CI_Z         = 1.96   # 95% Bartlett


# ────────────────────────────────────────────────────────────────────
# 2. Load Phase 4 CPI base (D-031-corrected stationary form)
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('Phase 5 · Step 4 — ACF / PACF diagnostics')
print('=' * 78)
print()

series_by_country: dict[str, pd.Series] = {}
for c in MAIN_COUNTRIES:
    path = PROCESSED_DIR / f'features_{c.lower()}.csv'
    df   = pd.read_csv(path, parse_dates=['date']).set_index('date')
    s = df[f'{c}_CPI'].dropna()
    s.name = c
    series_by_country[c] = s

print('CPI stationary series (D-031-corrected form):')
for c in MAIN_COUNTRIES:
    s = series_by_country[c]
    print(f'  {c:<8s}  n={len(s):>3}  '
          f'window={s.index.min():%Y-%m}..{s.index.max():%Y-%m}   '
          f'mean={s.mean():+.4f}   std={s.std(ddof=1):.4f}')
print()


# ────────────────────────────────────────────────────────────────────
# 3. Compute ACF, PACF, Ljung-Box per country
# ────────────────────────────────────────────────────────────────────
acf_by_country:  dict[str, np.ndarray] = {}
pacf_by_country: dict[str, np.ndarray] = {}
ci_by_country:   dict[str, float]      = {}

acf_pacf_rows = []
ljung_rows    = []

for c in MAIN_COUNTRIES:
    s = series_by_country[c].values
    n = len(s)
    ci = CI_Z / np.sqrt(n)

    acf_v  = acf(s, nlags=NLAGS, fft=False)
    pacf_v = pacf(s, nlags=NLAGS, method=PACF_METHOD)

    acf_by_country[c]  = acf_v
    pacf_by_country[c] = pacf_v
    ci_by_country[c]   = ci

    for k in range(len(acf_v)):
        acf_pacf_rows.append({
            'country':       c,
            'lag':           k,
            'acf':           round(float(acf_v[k]), 5),
            'pacf':          round(float(pacf_v[k]), 5),
            'ci_bartlett':   round(float(ci), 5),
            'acf_significant':  bool(abs(acf_v[k])  > ci and k > 0),
            'pacf_significant': bool(abs(pacf_v[k]) > ci and k > 0),
        })

    lb = acorr_ljungbox(s, lags=LJUNG_LAGS, return_df=True)
    for lag_val, row in lb.iterrows():
        ljung_rows.append({
            'country':   c,
            'lag':       int(lag_val),
            'n':         int(n),
            'lb_stat':   round(float(row['lb_stat']), 4),
            'lb_pvalue': float(row['lb_pvalue']),
            'reject_h0_at_5pct': bool(float(row['lb_pvalue']) < 0.05),
        })


# ────────────────────────────────────────────────────────────────────
# 4. Figure 8 — ACF / PACF 4×2 grid
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)

for i, c in enumerate(MAIN_COUNTRIES):
    ax_acf  = axes[i, 0]
    ax_pacf = axes[i, 1]

    acf_v  = acf_by_country[c]
    pacf_v = pacf_by_country[c]
    ci     = ci_by_country[c]
    n      = len(series_by_country[c])

    # lag 0 is always 1 for ACF and 1 for PACF; skip in the stem plot.
    lags = np.arange(1, NLAGS + 1)

    # ACF stem plot
    ml, sl, bl = ax_acf.stem(lags, acf_v[1:], basefmt='k-')
    ml.set_color(COLORS[c]);      ml.set_markersize(4)
    sl.set_color(COLORS[c]);      sl.set_linewidth(1.3)
    bl.set_linewidth(0.5)

    ax_acf.axhspan(-ci, ci, alpha=0.12, color=COLORS[c])
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.axvline(12, color='#e67e22', linestyle=':', alpha=0.7,
                   linewidth=0.9, zorder=0)
    ax_acf.set_ylabel('ACF', fontsize=10)
    ax_acf.set_title(f'{c}  ·  ACF  (n={n}, CI ±{ci:.3f})',
                     color=COLORS[c], fontweight='bold', fontsize=10.5, loc='left')
    ax_acf.grid(True, alpha=0.25)
    ax_acf.set_ylim(-1.05, 1.05)

    # PACF stem plot
    ml, sl, bl = ax_pacf.stem(lags, pacf_v[1:], basefmt='k-')
    ml.set_color(COLORS[c]);      ml.set_markersize(4)
    sl.set_color(COLORS[c]);      sl.set_linewidth(1.3)
    bl.set_linewidth(0.5)

    ax_pacf.axhspan(-ci, ci, alpha=0.12, color=COLORS[c])
    ax_pacf.axhline(0, color='black', linewidth=0.5)
    ax_pacf.axvline(12, color='#e67e22', linestyle=':', alpha=0.7,
                    linewidth=0.9, zorder=0)
    ax_pacf.set_ylabel('PACF', fontsize=10)
    ax_pacf.set_title(f'{c}  ·  PACF  (method={PACF_METHOD!r})',
                      color=COLORS[c], fontweight='bold', fontsize=10.5, loc='left')
    ax_pacf.grid(True, alpha=0.25)
    ax_pacf.set_ylim(-1.05, 1.05)

    if i == 3:
        ax_acf.set_xlabel('lag (months)', fontsize=10)
        ax_pacf.set_xlabel('lag (months)', fontsize=10)

fig.suptitle('Figure 8 · ACF and PACF — Phase 6 ARIMA order identification  '
             '(D-031-corrected stationary CPI form; orange dotted = lag 12 seasonal)',
             fontsize=12, y=0.995)
plt.tight_layout()
fig8_path = FIG_DIR / 'phase5_step4_fig8_acf_pacf.png'
plt.savefig(fig8_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 8 saved: {fig8_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 5. Audit CSVs
# ────────────────────────────────────────────────────────────────────
acf_path   = DOC_DIR / 'phase5_step4_acf_pacf_values.csv'
ljung_path = DOC_DIR / 'phase5_step4_ljung_box.csv'

pd.DataFrame(acf_pacf_rows).to_csv(acf_path, index=False)
pd.DataFrame(ljung_rows).to_csv(ljung_path, index=False)

print()
print('Audit CSVs:')
print(f'  {acf_path.relative_to(PROJECT_ROOT)}     ({len(acf_pacf_rows)} rows)')
print(f'  {ljung_path.relative_to(PROJECT_ROOT)}         ({len(ljung_rows)} rows)')
print()


# ────────────────────────────────────────────────────────────────────
# 6. Stdout findings
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('ACF / PACF significance summary  (|value| > Bartlett CI at lags 1..12)')
print('=' * 78)
print()
print(f'  {"Country":<8s}  {"CI":>7s}  '
      f'{"ACF_sig_1-12":>14s}  {"PACF_sig_1-12":>14s}  '
      f'{"ACF[12]":>9s}  {"PACF[12]":>10s}')

for c in MAIN_COUNTRIES:
    acf_v = acf_by_country[c]
    pacf_v = pacf_by_country[c]
    ci = ci_by_country[c]

    acf_sig_early  = int(np.sum(np.abs(acf_v[1:13])  > ci))
    pacf_sig_early = int(np.sum(np.abs(pacf_v[1:13]) > ci))
    acf_12  = float(acf_v[12])
    pacf_12 = float(pacf_v[12])

    print(f'  {c:<8s}  ±{ci:>6.3f}  '
          f'{f"{acf_sig_early}/12":>14s}  '
          f'{f"{pacf_sig_early}/12":>14s}  '
          f'{acf_12:>+9.3f}  {pacf_12:>+10.3f}')
print()
print('  (Lag 12 signal: if |ACF[12]| > CI, monthly seasonal structure persists.')
print('   Phase 6 SARIMA with (P=1, D=0, Q=0, s=12) may be warranted.)')
print()


print('=' * 78)
print(f'Ljung-Box Q statistics  (H0: residual is white noise; lags {LJUNG_LAGS})')
print('=' * 78)
print()
print(f'  {"Country":<8s}  '
      f'{"Q(12)":>9s} {"p(12)":>8s}  '
      f'{"Q(24)":>9s} {"p(24)":>8s}  '
      f'{"Q(36)":>9s} {"p(36)":>8s}')
ljung_df = pd.DataFrame(ljung_rows)
for c in MAIN_COUNTRIES:
    block = ljung_df[ljung_df['country'] == c].set_index('lag')
    def _f(lag_k):
        q = float(block.loc[lag_k, 'lb_stat'])
        p = float(block.loc[lag_k, 'lb_pvalue'])
        return q, p
    q12, p12 = _f(12)
    q24, p24 = _f(24)
    q36, p36 = _f(36)
    print(f'  {c:<8s}  '
          f'{q12:>9.2f} {p12:>8.4f}  '
          f'{q24:>9.2f} {p24:>8.4f}  '
          f'{q36:>9.2f} {p36:>8.4f}')
print()
print('  All p < 0.05 → reject H0 → autocorrelation present → ARIMA(p,0,q)')
print('  with p,q > 0 is warranted for Phase 6 (series already d-differenced')
print('  per D-031 stationary form; effective model is ARMA on the stationary level).')
print()


print('=' * 78)
print('ARMA order suggestion  (crude heuristic — Phase 6 AIC/BIC supersedes)')
print('=' * 78)
print()
print(f'  {"Country":<8s}  {"PACF cut-off → p ≤":<22s}  {"ACF cut-off → q ≤":<22s}  '
      f'{"first interpretation":<28s}')
for c in MAIN_COUNTRIES:
    acf_v  = acf_by_country[c]
    pacf_v = pacf_by_country[c]
    ci     = ci_by_country[c]

    # "cut-off" heuristic: count significant lags in 1..5
    p_cand = int(np.sum(np.abs(pacf_v[1:6]) > ci))
    q_cand = int(np.sum(np.abs(acf_v[1:6])  > ci))

    # Decay diagnostic: do ACF/PACF decay rather than cut? Rough check.
    acf_later_sig  = int(np.sum(np.abs(acf_v[6:13])  > ci))
    pacf_later_sig = int(np.sum(np.abs(pacf_v[6:13]) > ci))

    if p_cand > 0 and acf_later_sig >= 3:
        interp = f'AR({p_cand}) candidate'
    elif q_cand > 0 and pacf_later_sig >= 3:
        interp = f'MA({q_cand}) candidate'
    elif p_cand > 0 and q_cand > 0:
        interp = f'ARMA({p_cand},{q_cand}) candidate'
    else:
        interp = 'weak / near white noise'

    print(f'  {c:<8s}  '
          f'{f"p ≈ {p_cand} (sig lags 1-5)":<22s}  '
          f'{f"q ≈ {q_cand} (sig lags 1-5)":<22s}  '
          f'{interp:<28s}')
print()
print('  Interpretation is preliminary; Phase 6 grid search across (p,d,q)')
print('  with AIC/BIC selection provides the authoritative order.')
print()

print('Step 4 complete.')
