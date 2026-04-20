"""
scripts/phase5_step2_correlation_structure.py
==============================================
Phase 5 · Step 2 — Per-country time series panel + correlation heatmaps
                   + cross-lag correlation (N2 preview).

Decisions under consideration (final sign-off after this run)
-------------------------------------------------------------
  D-042 (adopted, Option B): heatmap scope is two-tier —
    Tier 1: per-country base 5×5 Pearson matrix on the D-031-corrected
            feature form (shows in-sample co-movement across the five
            indicators within each country).
    Tier 2: per-country cross-lag Pearson matrix with CPI(t) as anchor
            and {POLICY_RATE, UNEMPLOYMENT, GDP, M2} × {0,1,3,6,12} as
            columns. This previews N2 Monetary Policy Lag without
            pre-empting Phase 6 VAR Granger/IRF inference.

Scope intentionally excluded
----------------------------
  * 50×50 full-feature dendrogram (Option C): over-scope for Phase 5 EDA;
    Phase 6 Ridge regularisation handles high-dim multicollinearity
    natively per D-040.
  * Cross-country correlation panel: CPI forms differ across countries
    (USA=yoy_pct, JPN/GER=first_diff, UK=log_diff_pct per D-031), so
    cross-country Pearson between raw CPI forms is not a clean metric.

Inputs
------
  data/processed/features_{usa,japan,uk,germany}.csv   (Phase 4 output)

Outputs
-------
  outputs/figures/phase5_step2_fig3_indicator_panel.png   4×5 time series grid
  outputs/figures/phase5_step2_fig4_base_heatmap.png      2×2 country grid
  outputs/figures/phase5_step2_fig5_lag_heatmap.png       2×2 country grid
  data/documentation/phase5_step2_base_correlation.csv    4×25 = 100 rows
  data/documentation/phase5_step2_lag_correlation.csv     4×20 = 80 rows
  data/documentation/phase5_step2_window_summary.csv      4 rows

Run
---
  python scripts/phase5_step2_correlation_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ────────────────────────────────────────────────────────────────────
# 0. Project root + src import
# ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    MAIN_COUNTRIES,
    INDICATORS,
    KNOWN_BREAKS,
    load_effective_registry,
    LAG_PERIODS,
)


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

# Lag-heatmap configuration (Tier 2 of D-042)
LAG_HEATMAP_INDS: tuple[str, ...] = ('POLICY_RATE', 'UNEMPLOYMENT', 'GDP', 'M2')
LAG_HEATMAP_KS:   tuple[int, ...] = (0, 1, 3, 6, 12)   # k=0 uses base col

# Heatmap colour range for the lag matrix. Monetary macro cross-lag
# correlations rarely exceed |0.6|; clipping to ±0.6 maximises visual
# dynamic range without distorting the sign structure.
LAG_VLIM = 0.6
BASE_VLIM = 1.0


# ────────────────────────────────────────────────────────────────────
# 2. Load Phase 4 features + effective registry
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('Phase 5 · Step 2 — Correlation structure + N2 cross-lag preview')
print('=' * 78)
print()

features: dict[str, pd.DataFrame] = {}
for c in MAIN_COUNTRIES:
    path = PROCESSED_DIR / f'features_{c.lower()}.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    features[c] = df

eff_reg = load_effective_registry(PROJECT_ROOT)
# Build lookup: (country, indicator) -> effective_phase6_var_input label
form_lookup: dict[tuple[str, str], str] = {}
for _, r in eff_reg.iterrows():
    form_lookup[(r['country'], r['indicator'])] = r['effective_phase6_var_input']

print('Loaded Phase 4 features:')
for c in MAIN_COUNTRIES:
    df = features[c]
    jv = df[[f'{c}_{i}' for i in INDICATORS]].dropna(how='any')
    print(f'  {c:<8s}  shape={df.shape[0]:>3}x{df.shape[1]:<2}   '
          f'base-joint-valid n={len(jv):>3}   '
          f'{jv.index.min():%Y-%m}..{jv.index.max():%Y-%m}')
print()

# ────────────────────────────────────────────────────────────────────
# 3. Figure 3 — per-country 4×5 time series panel (base form)
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True)

for i, c in enumerate(MAIN_COUNTRIES):
    for j, ind in enumerate(INDICATORS):
        ax = axes[i, j]
        col = f'{c}_{ind}'
        s = features[c][col].dropna()
        ax.plot(s.index, s.values, color=COLORS[c], linewidth=0.9)
        ax.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.5)

        # Mark the three Phase 3 known breaks
        for bn, dt in KNOWN_BREAKS.items():
            ax.axvline(dt, color='black', linewidth=0.4, alpha=0.25)

        form_label = form_lookup.get((c, ind), '?')
        ax.set_title(f'{col}   ·   {form_label}', fontsize=8, loc='left')
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i == 3:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.suptitle('Figure 3 · Per-country 5-indicator time series panel '
             '(D-031-corrected base feature form; dashed lines = Phase 3 breaks)',
             fontsize=12, y=1.002)
plt.tight_layout()
fig3_path = FIG_DIR / 'phase5_step2_fig3_indicator_panel.png'
plt.savefig(fig3_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 3 saved: {fig3_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 4. Figure 4 — per-country base 5×5 Pearson heatmap
# ────────────────────────────────────────────────────────────────────
base_corr_matrices: dict[str, pd.DataFrame] = {}
base_corr_rows = []
window_rows = []

for c in MAIN_COUNTRIES:
    base_cols = [f'{c}_{ind}' for ind in INDICATORS]
    dfc = features[c][base_cols].dropna(how='any')
    corr = dfc.corr(method='pearson')
    # Rename axes for display
    corr.index   = list(INDICATORS)
    corr.columns = list(INDICATORS)
    base_corr_matrices[c] = corr

    for ind_row in INDICATORS:
        for ind_col in INDICATORS:
            base_corr_rows.append({
                'country':      c,
                'indicator_i':  ind_row,
                'indicator_j':  ind_col,
                'pearson_r':    round(float(corr.loc[ind_row, ind_col]), 5),
                'n':            int(len(dfc)),
            })

    window_rows.append({
        'country':       c,
        'base_n':        int(len(dfc)),
        'base_start':    dfc.index.min().strftime('%Y-%m'),
        'base_end':      dfc.index.max().strftime('%Y-%m'),
    })

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for idx, c in enumerate(MAIN_COUNTRIES):
    ax = axes[idx // 2, idx % 2]
    corr = base_corr_matrices[c]
    im = ax.imshow(corr.values, cmap='RdBu_r',
                   vmin=-BASE_VLIM, vmax=BASE_VLIM, aspect='auto')

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(INDICATORS, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(INDICATORS, fontsize=9)

    # Cell annotations
    for i in range(5):
        for j in range(5):
            v = float(corr.iloc[i, j])
            cell_color = 'white' if abs(v) > 0.55 else 'black'
            ax.text(j, i, f'{v:+.2f}',
                    ha='center', va='center',
                    fontsize=8.5, color=cell_color)

    n_obs = int((features[c][[f'{c}_{i}' for i in INDICATORS]]
                 .dropna(how='any')).shape[0])
    ax.set_title(f'{c}  ·  n = {n_obs}',
                 fontsize=11, color=COLORS[c], fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Figure 4 · Per-country base 5×5 Pearson correlation  '
             '(D-031 effective forms; joint-valid window)',
             fontsize=12, y=1.00)
plt.tight_layout()
fig4_path = FIG_DIR / 'phase5_step2_fig4_base_heatmap.png'
plt.savefig(fig4_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 4 saved: {fig4_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 5. Figure 5 — per-country cross-lag heatmap (N2 preview)
# ────────────────────────────────────────────────────────────────────
lag_corr_rows = []
lag_matrices:  dict[str, np.ndarray] = {}
lag_n_ranges:  dict[str, tuple[int, int]] = {}

for c in MAIN_COUNTRIES:
    cpi_col = f'{c}_CPI'
    mat = np.full((len(LAG_HEATMAP_INDS), len(LAG_HEATMAP_KS)), np.nan)
    n_min, n_max = 10**9, 0

    for i, ind in enumerate(LAG_HEATMAP_INDS):
        for j, k in enumerate(LAG_HEATMAP_KS):
            if k == 0:
                col = f'{c}_{ind}'
            else:
                col = f'{c}_{ind}_lag{k}'
            joint = features[c][[cpi_col, col]].dropna(how='any')
            if len(joint) >= 2:
                r = float(joint[cpi_col].corr(joint[col]))
            else:
                r = float('nan')
            mat[i, j] = r
            n_min = min(n_min, len(joint))
            n_max = max(n_max, len(joint))

            lag_corr_rows.append({
                'country':    c,
                'indicator':  ind,
                'lag':        k,
                'col_used':   col,
                'pearson_r':  round(r, 5) if np.isfinite(r) else np.nan,
                'n':          int(len(joint)),
            })

    lag_matrices[c] = mat
    lag_n_ranges[c] = (n_min, n_max)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, c in enumerate(MAIN_COUNTRIES):
    ax = axes[idx // 2, idx % 2]
    mat = lag_matrices[c]
    im = ax.imshow(mat, cmap='RdBu_r',
                   vmin=-LAG_VLIM, vmax=LAG_VLIM, aspect='auto')

    ax.set_xticks(range(len(LAG_HEATMAP_KS)))
    ax.set_yticks(range(len(LAG_HEATMAP_INDS)))
    ax.set_xticklabels([f't' if k == 0 else f't−{k}' for k in LAG_HEATMAP_KS],
                       fontsize=9)
    ax.set_yticklabels(LAG_HEATMAP_INDS, fontsize=9)

    for i in range(len(LAG_HEATMAP_INDS)):
        for j in range(len(LAG_HEATMAP_KS)):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            cell_color = 'white' if abs(v) > 0.35 else 'black'
            ax.text(j, i, f'{v:+.2f}',
                    ha='center', va='center',
                    fontsize=8.5, color=cell_color)

    n_min, n_max = lag_n_ranges[c]
    ax.set_title(f'{c}  ·  corr(CPI$_t$, X$_{{t−k}}$)  ·  n ∈ [{n_min}, {n_max}]',
                 fontsize=11, color=COLORS[c], fontweight='bold', loc='left')
    ax.set_xlabel('time reference of X', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Figure 5 · Cross-lag Pearson correlation — '
             'CPI(t) vs indicator(t−k)   '
             '[preview of N2 Monetary Policy Lag; Phase 6 VAR IRF supersedes]',
             fontsize=12, y=1.00)
plt.tight_layout()
fig5_path = FIG_DIR / 'phase5_step2_fig5_lag_heatmap.png'
plt.savefig(fig5_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 5 saved: {fig5_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 6. Audit CSVs
# ────────────────────────────────────────────────────────────────────
base_corr_path = DOC_DIR / 'phase5_step2_base_correlation.csv'
lag_corr_path  = DOC_DIR / 'phase5_step2_lag_correlation.csv'
window_path    = DOC_DIR / 'phase5_step2_window_summary.csv'

pd.DataFrame(base_corr_rows).to_csv(base_corr_path, index=False)
pd.DataFrame(lag_corr_rows).to_csv(lag_corr_path, index=False)
pd.DataFrame(window_rows).to_csv(window_path, index=False)

print()
print('Audit CSVs:')
print(f'  {base_corr_path.relative_to(PROJECT_ROOT)}   '
      f'({len(base_corr_rows)} rows)')
print(f'  {lag_corr_path.relative_to(PROJECT_ROOT)}    '
      f'({len(lag_corr_rows)} rows)')
print(f'  {window_path.relative_to(PROJECT_ROOT)}       '
      f'({len(window_rows)} rows)')
print()


# ────────────────────────────────────────────────────────────────────
# 7. Stdout findings — base correlations of economic interest
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('Key base 5×5 correlations (economic interest pairs)')
print('=' * 78)
print()
print(f'  {"Country":<8s}  {"Pair":<30s}  {"Pearson r":>10s}')
interest_pairs = [
    ('CPI', 'POLICY_RATE'),
    ('CPI', 'UNEMPLOYMENT'),   # N1 Phillips Curve direct preview
    ('CPI', 'GDP'),
    ('CPI', 'M2'),
    ('POLICY_RATE', 'UNEMPLOYMENT'),
    ('POLICY_RATE', 'GDP'),
]
for c in MAIN_COUNTRIES:
    corr = base_corr_matrices[c]
    for a, b in interest_pairs:
        r = float(corr.loc[a, b])
        print(f'  {c:<8s}  {a:<12s} vs {b:<14s}  {r:>+10.3f}')
    print()

# Per-country Phillips Curve single-number summary (N1 preview)
print('=' * 78)
print('N1 Phillips Curve preview — corr(CPI, UNEMPLOYMENT) in base form')
print('=' * 78)
print()
print(f'  {"Country":<8s}  {"corr":>8s}   sign interpretation')
for c in MAIN_COUNTRIES:
    r = float(base_corr_matrices[c].loc['CPI', 'UNEMPLOYMENT'])
    sign = ('negative → textbook Phillips' if r < -0.10 else
            ('positive → inverse / regime-mixed' if r > 0.10 else
             'near-zero → Phillips weak or absent'))
    print(f'  {c:<8s}  {r:>+8.3f}   {sign}')
print()


# ────────────────────────────────────────────────────────────────────
# 8. Stdout findings — N2 Monetary Policy Lag argmax per country
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('N2 preview — argmax_k |corr(CPI_t, POLICY_RATE_{t−k})|')
print('=' * 78)
print()
print(f'  {"Country":<8s}  '
      f'{"k=0":>8s}  {"k=1":>8s}  {"k=3":>8s}  {"k=6":>8s}  {"k=12":>8s}  '
      f'{"argmax_|r|":>12s}  {"peak_r":>8s}')
for c in MAIN_COUNTRIES:
    row_idx = LAG_HEATMAP_INDS.index('POLICY_RATE')
    row = lag_matrices[c][row_idx, :]
    argmax_j = int(np.nanargmax(np.abs(row)))
    argmax_k = LAG_HEATMAP_KS[argmax_j]
    peak_r   = float(row[argmax_j])
    vals = '  '.join(f'{row[j]:+8.3f}' for j in range(len(LAG_HEATMAP_KS)))
    print(f'  {c:<8s}  {vals}  {f"k={argmax_k}":>12s}  {peak_r:>+8.3f}')
print()
print('(Interpretation: largest-magnitude correlation across k ∈ {0,1,3,6,12}.')
print(' Negative sign + non-zero k is consistent with a policy-transmission lag.')
print(' Phase 6 VAR IRF will supply the directional / causal interpretation.)')
print()


print('=' * 78)
print('N2 preview — argmax_k |corr(CPI_t, UNEMPLOYMENT_{t−k})|  (N1 lag view)')
print('=' * 78)
print()
print(f'  {"Country":<8s}  '
      f'{"k=0":>8s}  {"k=1":>8s}  {"k=3":>8s}  {"k=6":>8s}  {"k=12":>8s}  '
      f'{"argmax_|r|":>12s}  {"peak_r":>8s}')
for c in MAIN_COUNTRIES:
    row_idx = LAG_HEATMAP_INDS.index('UNEMPLOYMENT')
    row = lag_matrices[c][row_idx, :]
    argmax_j = int(np.nanargmax(np.abs(row)))
    argmax_k = LAG_HEATMAP_KS[argmax_j]
    peak_r   = float(row[argmax_j])
    vals = '  '.join(f'{row[j]:+8.3f}' for j in range(len(LAG_HEATMAP_KS)))
    print(f'  {c:<8s}  {vals}  {f"k={argmax_k}":>12s}  {peak_r:>+8.3f}')
print()

print('Step 2 complete.')
