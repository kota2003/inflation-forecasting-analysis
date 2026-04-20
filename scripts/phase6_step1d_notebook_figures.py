"""
phase6_step1d_notebook_figures.py
==================================

Generate 8 portfolio figures for Phase 6 Step 1 notebook.

Aligned with Phase 5 S4 pattern (8 figures per Step culminating in notebook).
Figures consume the consolidated CSVs produced by Steps 1/1b/1c plus a
small number of training-residual refits (Fig 4, Fig 5).

Outputs: outputs/figures/phase6_step1_fig{1..8}_*.png (300 DPI).

Decision references: D-048 (Stages a/b/c + OOS saturation), D-049 (Japan
ARIMA uniqueness), D-050 (AIC-OOS divergence).

Runtime estimate: ~2-3 min (5 SARIMAX refits + plotting).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')                              # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

from src import (
    build_all_features,
    find_project_root,
    first_difference,
    load_processed_main,
)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
VARIANT_IDS = [
    'USA_yoy_pct',
    'USA_first_diff',
    'JAPAN_first_diff',
    'UK_log_diff_pct',
    'GERMANY_first_diff',
]

VARIANT_COLORS: Dict[str, str] = {
    'USA_yoy_pct':        '#1f77b4',               # tab:blue
    'USA_first_diff':     '#17becf',               # tab:cyan (USA sibling)
    'JAPAN_first_diff':   '#ff7f0e',               # tab:orange
    'UK_log_diff_pct':    '#2ca02c',               # tab:green
    'GERMANY_first_diff': '#d62728',               # tab:red
}

VARIANT_LABELS: Dict[str, str] = {
    'USA_yoy_pct':        'USA (yoy_pct)',
    'USA_first_diff':     'USA (first_diff)',
    'JAPAN_first_diff':   'Japan (first_diff)',
    'UK_log_diff_pct':    'UK (log_diff_pct)',
    'GERMANY_first_diff': 'Germany (first_diff)',
}

TRAIN_END      = pd.Timestamp('2019-12-01')
TEST_START     = pd.Timestamp('2020-01-01')
COVID_START    = pd.Timestamp('2020-01-01')
COVID_END      = pd.Timestamp('2021-12-01')
ENERGY_START   = pd.Timestamp('2022-01-01')
SEASONAL_S     = 12

DELTA_AIC_THRESHOLD = 2.0                          # D-048 Stage (b) threshold

# Stage (a) USA_first_diff OOS metrics — hardcoded from Step 1 stdout
# (in-place consolidated CSV was overwritten with Stage (c) values).
# Source: phase6_step1_arima_grid.py stdout, 2026-04-20 execution.
STAGE_A_USA_FD_OOS: Dict[str, Dict[str, float]] = {
    'full_test':        {'rmse': 0.8305, 'mae': 0.5914, 'bias': +0.3023},
    'covid_2020_2021':  {'rmse': 0.7693, 'mae': 0.6086, 'bias': +0.3310},
    'energy_2022_plus': {'rmse': 0.8608, 'mae': 0.5824, 'bias': +0.2873},
}
STAGE_A_USA_FD_AIC = 340.106

# Plot style
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        100,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
})


# ─────────────────────────────────────────────────────────────
# Variant construction (Step 1 parity)
# ─────────────────────────────────────────────────────────────
def _force_monthly_freq(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).to_period('M').to_timestamp(how='start')
    return out.asfreq('MS')


def build_variants(project_root: Path) -> Dict[str, pd.Series]:
    features = build_all_features(project_root=project_root)
    out: Dict[str, pd.Series] = {}
    out['USA_yoy_pct']      = _force_monthly_freq(features['USA']['USA_CPI'].dropna())
    usa_level               = load_processed_main('USA', project_root=project_root)['USA_CPI']
    out['USA_first_diff']   = _force_monthly_freq(first_difference(usa_level))
    out['JAPAN_first_diff'] = _force_monthly_freq(features['JAPAN']['JAPAN_CPI'].dropna())
    out['UK_log_diff_pct']  = _force_monthly_freq(features['UK']['UK_CPI'].dropna())
    out['GERMANY_first_diff'] = _force_monthly_freq(features['GERMANY']['GERMANY_CPI'].dropna())
    return out


def parse_order(order_str: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """'(p,d,q)(P,D,Q,s)' -> ((p,d,q), (P,D,Q,s))."""
    nons, seas = order_str.split(')(')
    p, d, q = map(int, nons.strip('()').split(','))
    P, D, Q, s = map(int, seas.strip('()').split(','))
    return (p, d, q), (P, D, Q, s)


# ─────────────────────────────────────────────────────────────
# Fig 1 — Variant overview (5 time-series on one axis per row)
# ─────────────────────────────────────────────────────────────
def fig1_variants_overview(variants: Dict[str, pd.Series],
                           out_path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for ax, vid in zip(axes, VARIANT_IDS):
        y = variants[vid]
        color = VARIANT_COLORS[vid]
        ax.plot(y.index, y.values, color=color, lw=0.9, label=VARIANT_LABELS[vid])
        # Shade test window
        ax.axvspan(TEST_START, y.index.max(), color='grey', alpha=0.08,
                   label='test 2020+' if vid == VARIANT_IDS[0] else None)
        ax.axvline(TRAIN_END, color='black', lw=0.5, ls='--', alpha=0.6)
        # Annotations
        n_train = (y.index <= TRAIN_END).sum()
        n_test  = (y.index >= TEST_START).sum()
        ax.text(0.01, 0.95, f'{VARIANT_LABELS[vid]}  '
                            f'n_train={n_train}  n_test={n_test}  '
                            f'σ={y.std(ddof=1):.3f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, fontweight='bold', color=color)
        ax.axhline(0, color='black', lw=0.3, alpha=0.3)
        ax.set_ylabel('value', fontsize=9)
    axes[-1].set_xlabel('Date')
    fig.suptitle('Fig 1 — Phase 6 Step 1 Variants Overview '
                 '(5 series, D-031 stationary form, D-005 train/test split)',
                 fontsize=12, y=0.995)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Fig 2 — AIC landscape per variant (p,q) at seasonal AIC-best
# ─────────────────────────────────────────────────────────────
def fig2_aic_landscape(doc_dir: Path, selection_df: pd.DataFrame,
                       out_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, vid in zip(axes, VARIANT_IDS):
        # Load grid CSV (for USA_first_diff, union Step 1 + Step 1c Q=3)
        g1_path = doc_dir / f'phase6_step1_arima_grid_{vid}.csv'
        g1 = pd.read_csv(g1_path)
        if vid == 'USA_first_diff':
            g1c_path = doc_dir / 'phase6_step1c_arima_grid_usa_first_diff_q3.csv'
            g1c = pd.read_csv(g1c_path)
            common = [c for c in g1.columns if c in g1c.columns]
            grid = pd.concat([g1[common], g1c[common]], ignore_index=True)
        else:
            grid = g1

        # Best seasonal order from selection
        sel_row = selection_df[selection_df['variant_id'] == vid].iloc[0]
        (p_b, d_b, q_b), (P_b, D_b, Q_b, s_b) = parse_order(sel_row['aic_best_order'])

        # Filter to best (P, D, Q)
        slab = grid[(grid['P'] == P_b) & (grid['D'] == D_b) & (grid['Q'] == Q_b)]
        piv = slab.pivot(index='p', columns='q', values='aic')

        im = ax.imshow(piv.values, cmap='viridis_r', aspect='auto',
                       origin='lower')
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel('q (non-seasonal MA)')
        ax.set_ylabel('p (non-seasonal AR)')

        # Mark the best (p, q)
        ax.plot(q_b, p_b, marker='*', color='red', markersize=14,
                markeredgecolor='white', markeredgewidth=1.0)

        # Annotate AIC values on each cell
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                if not np.isnan(piv.values[i, j]):
                    ax.text(j, i, f'{piv.values[i, j]:.0f}',
                            ha='center', va='center', fontsize=7,
                            color='white' if piv.values[i, j] < piv.values.mean()
                                           else 'black')
        title = (f'{VARIANT_LABELS[vid]}\n'
                 f'seasonal fixed at (P,D,Q)=({P_b},{D_b},{Q_b})\n'
                 f'AIC-best: {sel_row["aic_best_order"]}')
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.72, aspect=12, pad=0.02)
    fig.suptitle('Fig 2 — AIC landscape over (p, q) at AIC-best seasonal order — red ⋆ = AIC-best',
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Fig 3 — IC comparison across variants (D-049 Japan highlight)
# ─────────────────────────────────────────────────────────────
def fig3_ic_comparison(selection_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ic_colors = {'AIC': '#2c7fb8', 'BIC': '#fc8d62', 'HQIC': '#8da0cb'}
    x = np.arange(len(VARIANT_IDS))
    width = 0.27

    # Normalise AIC per variant (center on 0 for readability) — use param counts instead
    # Actually: plot n_params by each IC's best
    n_params_by_ic: Dict[str, List[int]] = {ic: [] for ic in ic_colors}
    agree_flags:    List[bool] = []
    for vid in VARIANT_IDS:
        row = selection_df[selection_df['variant_id'] == vid].iloc[0]
        for ic in ['aic', 'bic', 'hqic']:
            (p, d, q), (P, D, Q, s) = parse_order(row[f'{ic}_best_order'])
            n_params_by_ic[ic.upper()].append(p + q + P + Q + 1)
        agree_flags.append(bool(row['aic_bic_agree']) and bool(row['aic_hqic_agree']))

    for i, (ic, color) in enumerate(ic_colors.items()):
        bars = ax.bar(x + (i - 1) * width, n_params_by_ic[ic],
                      width, label=ic, color=color, edgecolor='black', lw=0.3)

    # Japan triple-agreement highlight
    for i, (xi, agree) in enumerate(zip(x, agree_flags)):
        if agree:
            ax.add_patch(Rectangle((xi - 1.5 * width, -0.3), 3 * width, 13,
                                    linewidth=2, edgecolor='gold',
                                    facecolor='gold', alpha=0.1,
                                    zorder=0))
            ax.annotate('triple\nagreement\n(D-049)', xy=(xi, 11.5),
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color='#b8860b')

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_IDS],
                       rotation=20, ha='right')
    ax.set_ylabel('n_params at IC-best')
    ax.set_title('Fig 3 — Information-criterion best orders: parameter counts '
                 '(Japan = AIC/BIC/HQIC triple agreement)',
                 fontsize=11)
    ax.set_ylim(0, 13)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Print AIC-best order labels on each bar group
    for i, vid in enumerate(VARIANT_IDS):
        row = selection_df[selection_df['variant_id'] == vid].iloc[0]
        ax.text(i, -0.6, row['aic_best_order'], ha='center', va='top',
                fontsize=7, style='italic', color='#555555')
    ax.set_ylim(-1.2, 13)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Refit AIC-best for each variant on training data (for Fig 4, 5)
# ─────────────────────────────────────────────────────────────
def refit_training_residuals(
    variants: Dict[str, pd.Series],
    selection_df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Return {variant_id: training_residuals} via AIC-best refit."""
    residuals: Dict[str, pd.Series] = {}
    for vid in VARIANT_IDS:
        y_train = variants[vid].loc[:TRAIN_END].dropna().asfreq('MS')
        row = selection_df[selection_df['variant_id'] == vid].iloc[0]
        order, s_order = parse_order(row['aic_best_order'])
        print(f'  refitting {vid} for residuals: {row["aic_best_order"]}',
              flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mod = SARIMAX(y_train, order=order, seasonal_order=s_order,
                          trend='c',
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            fit = mod.fit(method='lbfgs', maxiter=200, disp=False)
        residuals[vid] = pd.Series(fit.resid, index=y_train.index).dropna()
    return residuals


# ─────────────────────────────────────────────────────────────
# Fig 4 — Residual ACF (model adequacy confirmation)
# ─────────────────────────────────────────────────────────────
def fig4_residual_acf(residuals: Dict[str, pd.Series],
                      residuals_df: pd.DataFrame,
                      out_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
    MAX_LAG = 36
    for ax, vid in zip(axes, VARIANT_IDS):
        r = residuals[vid]
        n = len(r)
        acf_vals = acf(r, nlags=MAX_LAG, fft=True)
        ci = 1.96 / np.sqrt(n)
        lags = np.arange(MAX_LAG + 1)

        ax.bar(lags, acf_vals, width=0.8, color=VARIANT_COLORS[vid],
               alpha=0.85, edgecolor='black', lw=0.3)
        ax.axhline(ci, color='red', lw=0.6, ls='--', alpha=0.7)
        ax.axhline(-ci, color='red', lw=0.6, ls='--', alpha=0.7)
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.set_xlabel('Lag')
        if ax is axes[0]:
            ax.set_ylabel('ACF of residuals')

        # Ljung-Box p-value from residuals_df
        lb_row = residuals_df[residuals_df['variant_id'] == vid].iloc[0]
        lb12_p = lb_row['ljungbox_q12_p']
        lb24_p = lb_row['ljungbox_q24_p']
        ax.text(0.98, 0.95,
                f'Q(12) p={lb12_p:.3f}\nQ(24) p={lb24_p:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.85, lw=0.3))
        ax.set_title(VARIANT_LABELS[vid], fontsize=9)

    fig.suptitle('Fig 4 — Residual ACF of AIC-best SARIMA (training residuals; '
                 'all variants pass Ljung-Box at α = 0.05)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Fig 5 — Heteroscedasticity: rolling |residuals| (D-049 Japan highlight)
# ─────────────────────────────────────────────────────────────
def fig5_heteroscedasticity(residuals: Dict[str, pd.Series],
                            residuals_df: pd.DataFrame,
                            out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    WINDOW = 12                                     # 12-month rolling
    for vid in VARIANT_IDS:
        r = residuals[vid]
        rolling_abs = r.abs().rolling(window=WINDOW, min_periods=WINDOW).mean()
        lw = 2.2 if vid == 'JAPAN_first_diff' else 1.1
        alpha = 1.0 if vid == 'JAPAN_first_diff' else 0.75
        arch_p = residuals_df[residuals_df['variant_id'] == vid
                              ].iloc[0]['arch_lm_p']
        label = (f'{VARIANT_LABELS[vid]}  ARCH-LM p={arch_p:.3f}'
                 + ('  ★ D-049' if vid == 'JAPAN_first_diff' else ''))
        ax.plot(rolling_abs.index, rolling_abs.values,
                color=VARIANT_COLORS[vid], lw=lw, alpha=alpha, label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling 12-month mean |residual|')
    ax.set_title('Fig 5 — Training residual heteroscedasticity — '
                 "Japan's flat profile echoes N3 'uniqueness' (D-049)",
                 fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Fig 6 — Boundary sensitivity (Stage b)
# ─────────────────────────────────────────────────────────────
def fig6_boundary_sensitivity(doc_dir: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bs_path = doc_dir / 'phase6_step1b_boundary_check_summary.csv'
    bs = pd.read_csv(bs_path)

    variants = bs['variant_id'].tolist()
    deltas   = bs['delta_aic'].values
    verdicts = bs['verdict'].tolist()

    colors = [('#d62728' if v == 'extend_to_Q3' else '#7f7f7f')
              for v in verdicts]
    bars = ax.barh(range(len(variants)), deltas, color=colors,
                   edgecolor='black', lw=0.5)
    ax.axvline(-DELTA_AIC_THRESHOLD, color='red', lw=1.2, ls='--',
               label=f'extend threshold (ΔAIC ≤ −{DELTA_AIC_THRESHOLD})')
    ax.axvline(0, color='black', lw=0.5)

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel('ΔAIC (best Q=3 extension − original AIC-best)')
    ax.set_title('Fig 6 — D-048 Stage (b) boundary sensitivity verdicts — '
                 '1/3 escalate, 2/3 accept', fontsize=11)

    # Value labels on bars
    for i, (d, v) in enumerate(zip(deltas, verdicts)):
        ha = 'left' if d > 0 else 'right'
        pad = 0.3 if d > 0 else -0.3
        ax.text(d + pad, i, f'{d:+.2f}  →  {v}', va='center', ha=ha,
                fontsize=9, fontweight='bold',
                color='#d62728' if v == 'extend_to_Q3' else '#555555')

    ax.legend(loc='lower right', framealpha=0.9)
    xlim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])) + 3
    ax.set_xlim(-xlim, xlim)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Fig 7 — AIC-OOS divergence (D-050 key finding)
# ─────────────────────────────────────────────────────────────
def fig7_aic_oos_divergence(doc_dir: Path, window_errors_df: pd.DataFrame,
                            out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel A: AIC bar comparison
    ax = axes[0]
    delta_path = doc_dir / 'phase6_step1c_selection_delta.csv'
    delta = pd.read_csv(delta_path).iloc[0]
    stages = ['Stage (a)\n(0,0,3)(0,0,2,12)', 'Stage (c)\n(0,0,4)(2,0,3,12)']
    aics = [delta['step1_best_aic'], delta['step1c_best_aic']]
    colors_a = ['#aec7e8', '#1f77b4']
    bars = ax.bar(stages, aics, color=colors_a, edgecolor='black', lw=0.5)
    for b, a in zip(bars, aics):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f'{a:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.annotate('', xy=(1, aics[1]), xytext=(0, aics[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(0.5, (aics[0] + aics[1]) / 2,
            f'ΔAIC = {delta["delta_aic"]:+.2f}',
            ha='center', va='center', color='red',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='red', lw=0.8))
    ax.set_ylabel('Training AIC (lower = better)')
    ax.set_title('Panel A — In-sample AIC\nsubstantial improvement', fontsize=10)
    ax.set_ylim(320, 350)

    # Panel B: OOS RMSE/MAE comparison (USA_first_diff only)
    ax = axes[1]
    stage_c_oos = window_errors_df[
        (window_errors_df['variant_id'] == 'USA_first_diff')
    ].set_index('window')

    windows = ['full_test', 'covid_2020_2021', 'energy_2022_plus']
    metrics = ['rmse', 'mae', 'bias']
    x = np.arange(len(windows))
    width = 0.2

    stage_a_vals = np.array([[STAGE_A_USA_FD_OOS[w][m] for w in windows]
                             for m in metrics])
    stage_c_vals = np.array([[stage_c_oos.loc[w, m] for w in windows]
                             for m in metrics])

    colors_oos = {'rmse': '#ff9896', 'mae': '#c5b0d5', 'bias': '#98df8a'}
    for i, m in enumerate(metrics):
        offset = (i - 1) * width * 2
        ax.bar(x + offset - width / 2, stage_a_vals[i], width,
               color=colors_oos[m], edgecolor='black', lw=0.4,
               label=f'{m.upper()} Stage (a)', hatch='//')
        ax.bar(x + offset + width / 2, stage_c_vals[i], width,
               color=colors_oos[m], edgecolor='black', lw=0.4,
               label=f'{m.upper()} Stage (c)')

    ax.set_xticks(x)
    ax.set_xticklabels(['full_test', 'COVID\n2020-21', 'ENERGY+\n2022-'])
    ax.set_ylabel('OOS metric value')
    ax.set_title('Panel B — OOS forecasting metrics\n(Stage a hatched / Stage c solid) — '
                 'essentially invariant',
                 fontsize=10)
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(ncol=3, fontsize=7.5, loc='upper left', framealpha=0.9)

    fig.suptitle('Fig 7 — D-050 AIC-OOS Divergence: USA_first_diff Stage (a) → Stage (c). '
                 'ΔAIC = −10.46 while OOS metrics are statistically invariant.',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Fig 8 — Test window forecasts (5 variants × actual vs predicted)
# ─────────────────────────────────────────────────────────────
def fig8_test_forecasts(forecast_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    for ax, vid in zip(axes, VARIANT_IDS):
        sub = forecast_df[forecast_df['variant_id'] == vid].copy()
        sub['date'] = pd.to_datetime(sub['date'])
        sub = sub.sort_values('date')
        color = VARIANT_COLORS[vid]
        # Sub-window shading
        ax.axvspan(COVID_START, COVID_END, color='#a6cee3', alpha=0.25)
        ax.axvspan(ENERGY_START, sub['date'].max(), color='#fdbf6f', alpha=0.25)
        ax.plot(sub['date'], sub['actual'], color='black', lw=1.3,
                label='actual')
        ax.plot(sub['date'], sub['predicted'], color=color, lw=1.3, ls='--',
                label='predicted (1-step, expanding refit)')
        ax.axhline(0, color='black', lw=0.3, alpha=0.3)

        rmse = np.sqrt(((sub['actual'] - sub['predicted']) ** 2).mean())
        ax.text(0.01, 0.95, f'{VARIANT_LABELS[vid]}   OOS RMSE={rmse:.3f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, fontweight='bold', color=color)
        if ax is axes[0]:
            ax.text(COVID_START, ax.get_ylim()[1], '  COVID',
                    color='#1f77b4', fontsize=8, va='top')
            ax.text(ENERGY_START, ax.get_ylim()[1], '  ENERGY+',
                    color='#ff7f0e', fontsize=8, va='top')
        ax.legend(loc='lower left', framealpha=0.85, fontsize=8)
        ax.set_ylabel('value', fontsize=9)
    axes[-1].set_xlabel('Date')
    fig.suptitle('Fig 8 — Test-window expanding-refit 1-step-ahead forecasts '
                 '(Stage a for 4 variants; Stage c for USA_first_diff)',
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    import time
    t0 = time.perf_counter()
    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    fig_dir = root / 'outputs' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f'Project root: {root}')
    print(f'Figure dir:   {fig_dir}')

    # Shared inputs
    print('\nLoading data & CSVs ...')
    variants        = build_variants(root)
    selection_df    = pd.read_csv(doc_dir / 'phase6_step1_arima_selection.csv')
    residuals_df    = pd.read_csv(doc_dir / 'phase6_step1_arima_residuals.csv')
    forecast_df     = pd.read_csv(doc_dir / 'phase6_step1_arima_forecast.csv')
    window_errors_df = pd.read_csv(doc_dir / 'phase6_step1_arima_window_errors.csv')

    # Figure 1-3 (fast, from CSVs only)
    print('\nGenerating Fig 1 ...')
    fig1_variants_overview(
        variants, fig_dir / 'phase6_step1_fig1_variants_overview.png')
    print('Generating Fig 2 ...')
    fig2_aic_landscape(
        doc_dir, selection_df,
        fig_dir / 'phase6_step1_fig2_aic_landscape.png')
    print('Generating Fig 3 ...')
    fig3_ic_comparison(
        selection_df, fig_dir / 'phase6_step1_fig3_ic_comparison.png')

    # Refit for Fig 4, 5 residuals
    print('\nRefitting AIC-best models for Fig 4, 5 residuals ...')
    residuals = refit_training_residuals(variants, selection_df)

    print('Generating Fig 4 ...')
    fig4_residual_acf(
        residuals, residuals_df,
        fig_dir / 'phase6_step1_fig4_residual_acf.png')
    print('Generating Fig 5 ...')
    fig5_heteroscedasticity(
        residuals, residuals_df,
        fig_dir / 'phase6_step1_fig5_heteroscedasticity.png')

    # Fig 6-8
    print('Generating Fig 6 ...')
    fig6_boundary_sensitivity(
        doc_dir, fig_dir / 'phase6_step1_fig6_boundary_sensitivity.png')
    print('Generating Fig 7 ...')
    fig7_aic_oos_divergence(
        doc_dir, window_errors_df,
        fig_dir / 'phase6_step1_fig7_aic_oos_divergence.png')
    print('Generating Fig 8 ...')
    fig8_test_forecasts(
        forecast_df, fig_dir / 'phase6_step1_fig8_test_forecasts.png')

    print(f'\n8 figures written to {fig_dir}')
    print(f'Total runtime: {(time.perf_counter() - t0) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
