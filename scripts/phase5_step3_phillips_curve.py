"""
scripts/phase5_step3_phillips_curve.py
=======================================
Phase 5 · Step 3 — N1 Phillips Curve deep-dive with pre/post-GFC split
                   (Fig 6) and 60-month rolling regression (Fig 7).

Decisions under consideration (final sign-off after this run)
-------------------------------------------------------------
  D-043 (adopted, Option D): pre/post-GFC scatter + rolling slope time series.

Sub-decisions:
  * Variables are LEVEL-BASED — CPI YoY % (computed from Phase 2 CPI
    level via (lvl / lvl.shift(12) − 1) × 100) and UNEMPLOYMENT % (Phase 2
    level). NOT the D-031 stationary forms: Phase 5 Step 2 already showed
    stationary-form Phillips correlations are essentially zero for all 4
    countries (|r| ≤ 0.07). Classical Phillips Curve is a level-based
    relationship, and levels are the appropriate EDA lens.
  * Pre/post cutoff: KNOWN_BREAKS['GFC_2008'] = 2008-09-01.
  * Rolling window: 60 months (5-year business cycle), right-aligned,
    minimum observations = 60 (no partial windows).

Inputs
------
  data/processed/main_{usa,japan,uk,germany}.csv   (Phase 2 output)

Outputs
-------
  outputs/figures/phase5_step3_fig6_phillips_scatter.png   (2×2 country grid)
  outputs/figures/phase5_step3_fig7_rolling_slope.png      (dual panel)
  data/documentation/phase5_step3_phillips_fit.csv         (per-country
                                                            × {full, pre, post}
                                                            OLS coefficients)
  data/documentation/phase5_step3_rolling_slope.csv        (rolling results
                                                            long format)

Run
---
  python scripts/phase5_step3_phillips_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm


# ────────────────────────────────────────────────────────────────────
# 0. Project root + src import
# ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    MAIN_COUNTRIES,
    load_processed_all_main,
    KNOWN_BREAKS,
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

BREAK_STYLE = {
    'GFC_2008':    {'color': '#555555', 'linestyle': '--', 'linewidth': 0.8,
                    'label': 'GFC 2008-09'},
    'COVID_2020':  {'color': '#555555', 'linestyle': '-.', 'linewidth': 0.8,
                    'label': 'COVID 2020-03'},
    'ENERGY_2022': {'color': '#555555', 'linestyle': ':',  'linewidth': 1.0,
                    'label': 'ENERGY 2022-02'},
}

GFC_CUTOFF     = KNOWN_BREAKS['GFC_2008']   # 2008-09-01
ROLLING_WINDOW = 60                          # months


# ────────────────────────────────────────────────────────────────────
# 2. OLS helper
# ────────────────────────────────────────────────────────────────────
def fit_ols(x: pd.Series, y: pd.Series) -> dict | None:
    """OLS with intercept on joint-non-NaN subset. Returns None if the
    input cannot support estimation (n<3 or zero variance in x)."""
    x_a, y_a = x.align(y, join='inner')
    mask = x_a.notna() & y_a.notna()
    x_a, y_a = x_a[mask], y_a[mask]
    if len(x_a) < 3 or x_a.std() == 0:
        return None
    X = sm.add_constant(x_a.values, has_constant='add')
    model = sm.OLS(y_a.values, X).fit()
    return {
        'intercept':     float(model.params[0]),
        'slope':         float(model.params[1]),
        'se_intercept':  float(model.bse[0]),
        'se_slope':      float(model.bse[1]),
        'r_squared':     float(model.rsquared),
        'p_value_slope': float(model.pvalues[1]),
        'n':             int(model.nobs),
    }


# ────────────────────────────────────────────────────────────────────
# 3. Load Phase 2 and build per-country (UNEMP, CPI_YoY) frames
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('Phase 5 · Step 3 — N1 Phillips Curve deep-dive')
print('=' * 78)
print()

processed = load_processed_all_main(PROJECT_ROOT)

phillips_data: dict[str, pd.DataFrame] = {}
for c in MAIN_COUNTRIES:
    df = processed[c]
    cpi_level = df[f'{c}_CPI']
    unemp     = df[f'{c}_UNEMPLOYMENT']
    cpi_yoy   = (cpi_level / cpi_level.shift(12) - 1.0) * 100.0
    pair = pd.DataFrame({
        'UNEMPLOYMENT': unemp,
        'CPI_YoY':      cpi_yoy,
    }).dropna(how='any')
    phillips_data[c] = pair

print('Phillips pairs (UNEMP %, CPI YoY %):')
for c in MAIN_COUNTRIES:
    d = phillips_data[c]
    print(f'  {c:<8s}  n={len(d):>3}   '
          f'window={d.index.min():%Y-%m}..{d.index.max():%Y-%m}   '
          f'UNEMP=[{d["UNEMPLOYMENT"].min():.2f}, {d["UNEMPLOYMENT"].max():.2f}]   '
          f'CPI_YoY=[{d["CPI_YoY"].min():+.2f}, {d["CPI_YoY"].max():+.2f}]')
print()


# ────────────────────────────────────────────────────────────────────
# 4. OLS fits: full / pre-GFC / post-GFC
# ────────────────────────────────────────────────────────────────────
fit_rows = []
for c in MAIN_COUNTRIES:
    data = phillips_data[c]
    pre  = data[data.index < GFC_CUTOFF]
    post = data[data.index >= GFC_CUTOFF]

    for period_name, subset in [('full', data), ('pre_gfc', pre), ('post_gfc', post)]:
        res = fit_ols(subset['UNEMPLOYMENT'], subset['CPI_YoY'])
        if res is None:
            continue
        fit_rows.append({
            'country':      c,
            'period':       period_name,
            'period_start': subset.index.min().strftime('%Y-%m'),
            'period_end':   subset.index.max().strftime('%Y-%m'),
            **{k: (round(v, 5) if isinstance(v, float) else v)
               for k, v in res.items()},
        })

fit_df = pd.DataFrame(fit_rows)


# ────────────────────────────────────────────────────────────────────
# 5. Figure 6 — Phillips scatter with pre/post split
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for idx, c in enumerate(MAIN_COUNTRIES):
    ax = axes[idx // 2, idx % 2]
    data = phillips_data[c]
    pre  = data[data.index < GFC_CUTOFF]
    post = data[data.index >= GFC_CUTOFF]

    ax.scatter(pre['UNEMPLOYMENT'], pre['CPI_YoY'],
               color=COLORS[c], alpha=0.30, s=20, marker='o',
               edgecolors='none', label=f'pre-GFC (n={len(pre)})')
    ax.scatter(post['UNEMPLOYMENT'], post['CPI_YoY'],
               color=COLORS[c], alpha=0.75, s=20, marker='^',
               edgecolors='none', label=f'post-GFC (n={len(post)})')

    x_range = np.linspace(data['UNEMPLOYMENT'].min(),
                          data['UNEMPLOYMENT'].max(), 100)
    pre_fit  = fit_df[(fit_df['country'] == c) & (fit_df['period'] == 'pre_gfc')]
    post_fit = fit_df[(fit_df['country'] == c) & (fit_df['period'] == 'post_gfc')]

    if len(pre_fit):
        r = pre_fit.iloc[0]
        y_pred = r['intercept'] + r['slope'] * x_range
        ax.plot(x_range, y_pred, color=COLORS[c], linestyle='--',
                linewidth=2.2, alpha=0.8,
                label=f'pre OLS: β={r["slope"]:+.2f}, R²={r["r_squared"]:.2f}')
    if len(post_fit):
        r = post_fit.iloc[0]
        y_pred = r['intercept'] + r['slope'] * x_range
        ax.plot(x_range, y_pred, color=COLORS[c], linestyle='-',
                linewidth=2.2,
                label=f'post OLS: β={r["slope"]:+.2f}, R²={r["r_squared"]:.2f}')

    ax.axhline(0, color='black', linewidth=0.4, alpha=0.55)
    ax.set_xlabel('Unemployment rate (%)', fontsize=10)
    ax.set_ylabel('CPI YoY (%)', fontsize=10)
    ax.set_title(f'{c}', fontsize=11, color=COLORS[c], fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=8.5, frameon=True, framealpha=0.92)
    ax.grid(True, alpha=0.25)

fig.suptitle('Figure 6 · N1 Phillips Curve — pre/post-GFC split  '
             '(cutoff 2008-09; levels, not stationary forms)',
             fontsize=12, y=1.00)
plt.tight_layout()
fig6_path = FIG_DIR / 'phase5_step3_fig6_phillips_scatter.png'
plt.savefig(fig6_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 6 saved: {fig6_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 6. Rolling 60-month OLS per country
# ────────────────────────────────────────────────────────────────────
rolling_rows = []
rolling_slopes: dict[str, pd.Series] = {}
rolling_r2:     dict[str, pd.Series] = {}

for c in MAIN_COUNTRIES:
    data = phillips_data[c]
    slopes = pd.Series(np.nan, index=data.index, dtype=float)
    r2s    = pd.Series(np.nan, index=data.index, dtype=float)

    for i in range(ROLLING_WINDOW, len(data) + 1):
        window = data.iloc[i - ROLLING_WINDOW:i]
        x = window['UNEMPLOYMENT']
        y = window['CPI_YoY']
        if x.std() == 0 or y.std() == 0:
            continue
        X = sm.add_constant(x.values, has_constant='add')
        try:
            model = sm.OLS(y.values, X).fit()
        except Exception:
            continue
        end_date = data.index[i - 1]
        slopes.loc[end_date] = float(model.params[1])
        r2s.loc[end_date]    = float(model.rsquared)
        rolling_rows.append({
            'country':     c,
            'window_end':  end_date.strftime('%Y-%m'),
            'slope':       round(float(model.params[1]), 5),
            'se_slope':    round(float(model.bse[1]), 5),
            'r_squared':   round(float(model.rsquared), 5),
            'p_value':     round(float(model.pvalues[1]), 5),
            'n':           int(model.nobs),
        })

    rolling_slopes[c] = slopes.dropna()
    rolling_r2[c]     = r2s.dropna()


# ────────────────────────────────────────────────────────────────────
# 7. Figure 7 — dual-panel rolling slope + R²
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Panel A: slope
ax_a = axes[0]
for c in MAIN_COUNTRIES:
    s = rolling_slopes[c]
    ax_a.plot(s.index, s.values, color=COLORS[c], linewidth=1.4, label=c)
ax_a.axhline(0, color='black', linewidth=0.5, alpha=0.75)
for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_a.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'], label=st['label'])
ax_a.set_ylabel('60m rolling OLS slope β\n(CPI YoY ~ Unemployment)', fontsize=10)
ax_a.set_title(f'Panel A · Rolling Phillips slope  ·  window = {ROLLING_WINDOW} months, right-aligned',
               fontsize=11, loc='left')
ax_a.legend(loc='best', fontsize=8.5, frameon=False, ncol=2)
ax_a.grid(True, alpha=0.25)

# Panel B: R²
ax_b = axes[1]
for c in MAIN_COUNTRIES:
    s = rolling_r2[c]
    ax_b.plot(s.index, s.values, color=COLORS[c], linewidth=1.4, label=c)
ax_b.axhline(0, color='black', linewidth=0.4, alpha=0.5)
for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_b.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'])
ax_b.set_ylabel('60m rolling R²', fontsize=10)
ax_b.set_title('Panel B · Rolling fit quality  ·  '
               'R² → 0 indicates the pair loses linear structure',
               fontsize=11, loc='left')
ax_b.set_ylim(bottom=0)
ax_b.grid(True, alpha=0.25)
ax_b.xaxis.set_major_locator(mdates.YearLocator(2))
ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.suptitle('Figure 7 · Phillips Curve time-variation — 60-month rolling OLS',
             fontsize=12, y=0.995)
plt.tight_layout()
fig7_path = FIG_DIR / 'phase5_step3_fig7_rolling_slope.png'
plt.savefig(fig7_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 7 saved: {fig7_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 8. Audit CSVs
# ────────────────────────────────────────────────────────────────────
fit_path     = DOC_DIR / 'phase5_step3_phillips_fit.csv'
rolling_path = DOC_DIR / 'phase5_step3_rolling_slope.csv'

fit_df.to_csv(fit_path, index=False)
pd.DataFrame(rolling_rows).to_csv(rolling_path, index=False)

print()
print('Audit CSVs:')
print(f'  {fit_path.relative_to(PROJECT_ROOT)}       ({len(fit_df)} rows)')
print(f'  {rolling_path.relative_to(PROJECT_ROOT)}   ({len(rolling_rows)} rows)')
print()


# ────────────────────────────────────────────────────────────────────
# 9. Stdout findings
# ────────────────────────────────────────────────────────────────────
def _sig_star(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '.'
    return ' '


print('=' * 78)
print('Full-sample Phillips Curve fit (CPI_YoY ~ UNEMPLOYMENT, level-based)')
print('=' * 78)
print(f'  {"Country":<8s}  {"slope β":>8s}  {"SE":>6s}  '
      f'{"R²":>5s}  {"p-value":>8s}  {"":>3s}  {"n":>3s}')
for c in MAIN_COUNTRIES:
    r = fit_df[(fit_df['country']==c) & (fit_df['period']=='full')].iloc[0]
    print(f'  {c:<8s}  {r["slope"]:>+8.3f}  {r["se_slope"]:>6.3f}  '
          f'{r["r_squared"]:>5.2f}  {r["p_value_slope"]:>8.4f}  '
          f'{_sig_star(r["p_value_slope"]):>3s}  {int(r["n"]):>3d}')
print()
print('  (significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10)')
print()


print('=' * 78)
print(f'Pre/post-GFC split (cutoff {GFC_CUTOFF:%Y-%m-%d})')
print('=' * 78)
print(f'  {"Country":<8s}  {"Period":<10s}  {"slope β":>8s}  {"SE":>6s}  '
      f'{"R²":>5s}  {"p-value":>8s}  {"":>3s}  {"n":>3s}  {"Window":<17s}')
for c in MAIN_COUNTRIES:
    for period in ['pre_gfc', 'post_gfc']:
        r = fit_df[(fit_df['country']==c) & (fit_df['period']==period)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        window = f'{r["period_start"]}..{r["period_end"]}'
        print(f'  {c:<8s}  {period:<10s}  {r["slope"]:>+8.3f}  {r["se_slope"]:>6.3f}  '
              f'{r["r_squared"]:>5.2f}  {r["p_value_slope"]:>8.4f}  '
              f'{_sig_star(r["p_value_slope"]):>3s}  '
              f'{int(r["n"]):>3d}  {window:<17s}')
    print()


print('=' * 78)
print('Flattening diagnostic — |slope| reduction pre → post  (N1 centrepiece)')
print('=' * 78)
print(f'  {"Country":<8s}  {"|β_pre|":>8s}  {"|β_post|":>8s}  {"Δ|β|":>7s}  '
      f'{"reduction":>10s}  {"sign_pre":>9s}  {"sign_post":>10s}')
for c in MAIN_COUNTRIES:
    pre  = fit_df[(fit_df['country']==c) & (fit_df['period']=='pre_gfc')]
    post = fit_df[(fit_df['country']==c) & (fit_df['period']=='post_gfc')]
    if not (len(pre) and len(post)):
        continue
    b_pre  = pre.iloc[0]['slope']
    b_post = post.iloc[0]['slope']
    abs_pre, abs_post = abs(b_pre), abs(b_post)
    reduction_pct = (1 - abs_post / abs_pre) * 100 if abs_pre > 1e-6 else float('nan')
    sign_pre  = '+' if b_pre  > 0 else '−'
    sign_post = '+' if b_post > 0 else '−'
    print(f'  {c:<8s}  {abs_pre:>8.3f}  {abs_post:>8.3f}  '
          f'{abs_post - abs_pre:>+7.3f}  {reduction_pct:>9.1f}%  '
          f'{sign_pre:>9s}  {sign_post:>10s}')
print()
print('  Interpretation: textbook Phillips has β < 0 (negative slope).')
print('  "Flattening" = |β_post| < |β_pre|; a sign flip from − to + would')
print('  indicate regime breakdown.')
print()


print('=' * 78)
print('Rolling slope terminal values (most recent 60-month window)')
print('=' * 78)
print(f'  {"Country":<8s}  {"window end":<10s}  {"β":>7s}  {"R²":>5s}')
for c in MAIN_COUNTRIES:
    s = rolling_slopes[c]
    r2 = rolling_r2[c]
    if len(s):
        end = s.index[-1].strftime('%Y-%m')
        print(f'  {c:<8s}  {end:<10s}  '
              f'{float(s.iloc[-1]):>+7.3f}  {float(r2.iloc[-1]):>5.2f}')
print()

print('Step 3 complete.')
