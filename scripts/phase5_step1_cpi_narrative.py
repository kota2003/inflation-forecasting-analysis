"""
scripts/phase5_step1_cpi_narrative.py
=====================================
Phase 5 · Step 1 — 4-country CPI overlay + Japan N3 deep-dive.

v3 patch notes (vs v2)
----------------------
* Figure 1 Panel A: per-country vertical offset on terminal annotations.
  USA (184.9) and UK (185.2) terminate at near-identical y, 7 months apart
  on x — horizontal-offset-only labelling collides. Dict-driven offsets
  separate them cleanly without hiding the 'they converge' story.

v2 patch notes (vs v1)
----------------------
* Figure 2: x-axis explicitly clipped to (2000-10, 2026-03) after all plotting.
* Figure 1 Panel A: terminal-index annotations at 1-decimal precision.

Decisions
---------
  D-041 (Option C): dual panel — Panel A indexes CPI levels to
        2001-01 = 100; Panel B shows YoY %.
  D-045 (Option B): Japan defined by four phases
        Bubble aftermath (≤1998-12), Deflation era (1999-01..2012-12),
        Abenomics (2013-04..2022-01), Reversal (2022-02 onwards).

Inputs / Outputs / Run — see v2; unchanged.
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

JAPAN_PHASES_VISIBLE = [
    ('Deflation era', pd.Timestamp('1999-01-01'), pd.Timestamp('2012-12-31'),
     '#fde0dc', 0.55),
    ('Abenomics',     pd.Timestamp('2013-04-01'), pd.Timestamp('2022-01-31'),
     '#fff3cd', 0.55),
    ('Reversal',      pd.Timestamp('2022-02-01'), pd.Timestamp('2030-12-31'),
     '#d4edda', 0.55),
]

INDEX_BASE_DATE = pd.Timestamp('2001-01-01')

VIEW_XMIN = pd.Timestamp('2000-10-01')
VIEW_XMAX = pd.Timestamp('2026-03-01')

# v3 — vertical offset (in display points) for terminal annotations on
# Figure 1 Panel A. USA (184.88) and UK (185.17) converge; without this
# the labels overprint. Direction matches value order: UK > USA → UK up.
TERMINAL_LABEL_Y_OFFSET_PT = {
    'USA':     -7,
    'JAPAN':    0,
    'UK':      +7,
    'GERMANY':  0,
}


# ────────────────────────────────────────────────────────────────────
# 2. Load Phase 2 CPI and derive YoY
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('Phase 5 · Step 1 — 4-country CPI overlay + Japan N3 deep-dive  (v3)')
print('=' * 78)
print()

processed = load_processed_all_main(PROJECT_ROOT)

cpi_levels: dict[str, pd.Series] = {}
cpi_yoy:    dict[str, pd.Series] = {}
cpi_index:  dict[str, pd.Series] = {}

for c in MAIN_COUNTRIES:
    lvl = processed[c][f'{c}_CPI'].copy()
    lvl.name = c
    cpi_levels[c] = lvl

    yoy = (lvl / lvl.shift(12) - 1.0) * 100.0
    yoy.name = c
    cpi_yoy[c] = yoy

    if INDEX_BASE_DATE in lvl.index and not pd.isna(lvl.loc[INDEX_BASE_DATE]):
        base_val = float(lvl.loc[INDEX_BASE_DATE])
        base_dt  = INDEX_BASE_DATE
    else:
        base_dt  = lvl.first_valid_index()
        base_val = float(lvl.loc[base_dt])
    idx = (lvl / base_val) * 100.0
    idx.name = c
    cpi_index[c] = idx

print('Loaded Phase 2 CPI series (level → YoY → index):')
for c in MAIN_COUNTRIES:
    lvl = cpi_levels[c].dropna()
    yoy = cpi_yoy[c].dropna()
    print(f'  {c:<8s}  level n={len(lvl):>3}  '
          f'{lvl.index.min():%Y-%m}..{lvl.index.max():%Y-%m}   '
          f'YoY n={len(yoy):>3}  '
          f'{yoy.index.min():%Y-%m}..{yoy.index.max():%Y-%m}')
print()


# ────────────────────────────────────────────────────────────────────
# 3. Figure 1 — Cross-country dual-panel overlay (D-041)
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# --- Panel A: index = 100 at 2001-01 --------------------------------
ax_a = axes[0]
for c in MAIN_COUNTRIES:
    s = cpi_index[c].dropna()
    ax_a.plot(s.index, s.values, color=COLORS[c], linewidth=1.4, label=c)

for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_a.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'])

ax_a.axhline(100, color='black', linewidth=0.4, alpha=0.35)
ax_a.set_ylabel('CPI index (2001-01 = 100)', fontsize=10)
ax_a.set_title('Panel A · Cumulative price level — indexed to 2001-01 = 100',
               fontsize=11, loc='left')
ax_a.legend(loc='upper left', fontsize=9, frameon=False)
ax_a.grid(True, alpha=0.25)

# v3 — terminal annotations at 1-decimal precision + per-country vertical
# offset so nearly-coincident values (USA 184.9 vs UK 185.2) don't overprint.
for c in MAIN_COUNTRIES:
    s = cpi_index[c].dropna()
    last_val = float(s.iloc[-1])
    dy = TERMINAL_LABEL_Y_OFFSET_PT[c]
    ax_a.annotate(f'{last_val:.1f}',
                  xy=(s.index[-1], last_val),
                  xytext=(4, dy), textcoords='offset points',
                  color=COLORS[c], fontsize=8.5, va='center', fontweight='bold')

# --- Panel B: YoY --------------------------------------------------
ax_b = axes[1]
for c in MAIN_COUNTRIES:
    s = cpi_yoy[c].dropna()
    ax_b.plot(s.index, s.values, color=COLORS[c], linewidth=1.1, label=c)

for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_b.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'], label=st['label'])

ax_b.axhline(0, color='black', linewidth=0.5, alpha=0.65)
ax_b.axhline(2, color='#888888', linewidth=0.4, linestyle=':', alpha=0.7)
ax_b.text(pd.Timestamp('2001-03-01'), 2.25, '2% target',
          fontsize=7, color='#555555', style='italic')

ax_b.set_ylabel('CPI YoY (%)', fontsize=10)
ax_b.set_title('Panel B · Year-over-year inflation rate  '
               '(dotted horizontal line = 2% central-bank target)',
               fontsize=11, loc='left')
ax_b.legend(loc='upper left', fontsize=8, ncol=2, frameon=False)
ax_b.grid(True, alpha=0.25)
ax_b.xaxis.set_major_locator(mdates.YearLocator(2))
ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.suptitle('Figure 1 · Four-country CPI dynamics (2001–2026) — '
             'dual-panel cumulative-and-rate view (D-041)',
             fontsize=12, y=0.995)
plt.tight_layout()
fig1_path = FIG_DIR / 'phase5_step1_fig1_cpi_overlay.png'
plt.savefig(fig1_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 1 saved: {fig1_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 4. Figure 2 — Japan N3 three-panel deep-dive (D-045)
# ────────────────────────────────────────────────────────────────────
peers = [c for c in MAIN_COUNTRIES if c != 'JAPAN']
peer_df_all  = pd.concat([cpi_yoy[c] for c in peers], axis=1)
peer_df_strict = peer_df_all.dropna(how='any')
peer_avg = peer_df_strict.mean(axis=1)
japan_on_peer_idx = cpi_yoy['JAPAN'].reindex(peer_avg.index)
japan_gap = japan_on_peer_idx - peer_avg

fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True)


def shade_japan_phases(ax):
    for name, start, end, face, alpha in JAPAN_PHASES_VISIBLE:
        ax.axvspan(start, end, color=face, alpha=alpha, zorder=0)


def annotate_phases(ax, y_frac=0.93):
    """Place phase labels near the top of the plot area, clipped to xlim."""
    ymin, ymax = ax.get_ylim()
    y_text = ymin + (ymax - ymin) * y_frac
    xmin, xmax = ax.get_xlim()
    xmin_ts = pd.Timestamp(mdates.num2date(xmin).replace(tzinfo=None))
    xmax_ts = pd.Timestamp(mdates.num2date(xmax).replace(tzinfo=None))
    for name, start, end, _f, _a in JAPAN_PHASES_VISIBLE:
        vis_start = max(start, xmin_ts)
        vis_end   = min(end,   xmax_ts)
        if vis_end <= vis_start:
            continue
        mid = vis_start + (vis_end - vis_start) / 2
        ax.text(mid, y_text, name, ha='center', va='top',
                fontsize=8.5, color='#333333', style='italic',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.6))


# --- Panel A: Japan CPI level --------------------------------------
ax_a = axes[0]
shade_japan_phases(ax_a)
s = cpi_levels['JAPAN'].dropna()
ax_a.plot(s.index, s.values, color=COLORS['JAPAN'], linewidth=1.5)
for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_a.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'])
ax_a.set_ylabel('Japan CPI level\n(national index, 2020=100)', fontsize=10)
ax_a.set_title('Panel A · Japan CPI level — two decades near-flat, '
               'structural break at ENERGY 2022',
               fontsize=11, loc='left')
ax_a.grid(True, alpha=0.25)

# --- Panel B: Japan YoY --------------------------------------------
ax_b = axes[1]
shade_japan_phases(ax_b)
s = cpi_yoy['JAPAN'].dropna()
ax_b.plot(s.index, s.values, color=COLORS['JAPAN'], linewidth=1.25)
ax_b.axhline(0, color='black', linewidth=0.55, alpha=0.75)
ax_b.axhline(2, color='#888888', linewidth=0.4, linestyle=':', alpha=0.7)
for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_b.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'])

reversal_dt = pd.Timestamp('2022-02-01')
if reversal_dt in s.index:
    rv = float(s.loc[reversal_dt])
    ax_b.annotate(f'ENERGY 2022\nreversal onset\n({rv:+.1f}% YoY)',
                  xy=(reversal_dt, rv),
                  xytext=(-160, 45), textcoords='offset points',
                  fontsize=8.5, color='#333333',
                  arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8))

deflation_months = int((s < 0).sum())
ax_b.set_ylabel('Japan CPI YoY (%)', fontsize=10)
ax_b.set_title(f'Panel B · Japan CPI YoY — {deflation_months} months below zero '
               f'in the visible window',
               fontsize=11, loc='left')
ax_b.grid(True, alpha=0.25)

# --- Panel C: Japan minus peer-average YoY -------------------------
ax_c = axes[2]
shade_japan_phases(ax_c)
g = japan_gap.dropna()
ax_c.plot(g.index, g.values, color=COLORS['JAPAN'], linewidth=1.25,
          label='Japan YoY − mean of USA/UK/Germany YoY')
ax_c.fill_between(g.index, 0, g.values, where=(g.values < 0),
                  color=COLORS['JAPAN'], alpha=0.18, interpolate=True)
ax_c.fill_between(g.index, 0, g.values, where=(g.values >= 0),
                  color='#2e7d32', alpha=0.18, interpolate=True)
ax_c.axhline(0, color='black', linewidth=0.55, alpha=0.75)
for name, dt in KNOWN_BREAKS.items():
    st = BREAK_STYLE[name]
    ax_c.axvline(dt, color=st['color'], linestyle=st['linestyle'],
                 linewidth=st['linewidth'])

ax_c.set_ylabel('Gap (percentage points)', fontsize=10)
ax_c.set_title('Panel C · Japan divergence from peers — '
               'YoY gap; negative = Japan below peer average',
               fontsize=11, loc='left')
ax_c.grid(True, alpha=0.25)
ax_c.legend(loc='lower right', fontsize=8.5, frameon=False)
ax_c.xaxis.set_major_locator(mdates.YearLocator(2))
ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

for ax in axes:
    ax.set_xlim(VIEW_XMIN, VIEW_XMAX)
annotate_phases(axes[0], y_frac=0.95)

fig.suptitle("Figure 2 · Japan's Uniqueness (N3) — three-panel deep-dive "
             "(D-045: 3 of 4 phases visible; Bubble aftermath ≤1998 is pre-data)",
             fontsize=12, y=0.995)
plt.tight_layout()
fig2_path = FIG_DIR / 'phase5_step1_fig2_japan_deepdive.png'
plt.savefig(fig2_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'  Figure 2 saved: {fig2_path.relative_to(PROJECT_ROOT)}')


# ────────────────────────────────────────────────────────────────────
# 5. Audit CSVs
# ────────────────────────────────────────────────────────────────────
country_rows = []
for c in MAIN_COUNTRIES:
    lvl = cpi_levels[c].dropna()
    yoy = cpi_yoy[c].dropna()
    idx = cpi_index[c].dropna()
    country_rows.append({
        'country':             c,
        'level_start':         lvl.index.min().strftime('%Y-%m'),
        'level_end':           lvl.index.max().strftime('%Y-%m'),
        'level_n':             len(lvl),
        'yoy_start':           yoy.index.min().strftime('%Y-%m'),
        'yoy_end':             yoy.index.max().strftime('%Y-%m'),
        'yoy_n':               len(yoy),
        'yoy_mean':            round(float(yoy.mean()), 4),
        'yoy_std':             round(float(yoy.std(ddof=1)), 4),
        'yoy_min':             round(float(yoy.min()), 4),
        'yoy_min_date':        yoy.idxmin().strftime('%Y-%m'),
        'yoy_max':             round(float(yoy.max()), 4),
        'yoy_max_date':        yoy.idxmax().strftime('%Y-%m'),
        'deflation_months':    int((yoy < 0).sum()),
        'months_above_2pct':   int((yoy > 2).sum()),
        'index_terminal':      round(float(idx.iloc[-1]), 3),
        'cumulative_infl_pct': round(float(idx.iloc[-1] - 100), 3),
    })
country_df = pd.DataFrame(country_rows)

phase_rows = []
for name, start, end, _f, _a in JAPAN_PHASES_VISIBLE:
    s = cpi_yoy['JAPAN']
    mask = (s.index >= start) & (s.index <= end)
    w = s.loc[mask].dropna()
    phase_rows.append({
        'phase':             name,
        'start':             start.strftime('%Y-%m'),
        'end':               end.strftime('%Y-%m'),
        'n_months':          len(w),
        'yoy_mean':          round(float(w.mean()), 4)   if len(w) else np.nan,
        'yoy_std':           round(float(w.std(ddof=1)), 4) if len(w) > 1 else np.nan,
        'yoy_min':           round(float(w.min()), 4)    if len(w) else np.nan,
        'yoy_max':           round(float(w.max()), 4)    if len(w) else np.nan,
        'deflation_months':  int((w < 0).sum())          if len(w) else 0,
        'above_2pct_months': int((w > 2).sum())          if len(w) else 0,
    })
phase_df = pd.DataFrame(phase_rows)

g = japan_gap.dropna()
gap_rows = [{
    'series':             'JAPAN_minus_3peer_avg_yoy',
    'n':                  len(g),
    'window_start':       g.index.min().strftime('%Y-%m'),
    'window_end':         g.index.max().strftime('%Y-%m'),
    'gap_mean':           round(float(g.mean()), 4),
    'gap_std':            round(float(g.std(ddof=1)), 4),
    'gap_min':            round(float(g.min()), 4),
    'gap_min_date':       g.idxmin().strftime('%Y-%m'),
    'gap_max':            round(float(g.max()), 4),
    'gap_max_date':       g.idxmax().strftime('%Y-%m'),
    'months_below_zero':  int((g < 0).sum()),
    'months_above_zero':  int((g >= 0).sum()),
    'note':               'all-peers-valid intersection; ends at UK/GER data end 2025-03',
}]
gap_df = pd.DataFrame(gap_rows)

summary_path  = DOC_DIR / 'phase5_step1_cpi_summary.csv'
phases_path   = DOC_DIR / 'phase5_step1_japan_phases.csv'
peer_gap_path = DOC_DIR / 'phase5_step1_japan_peer_gap.csv'

country_df.to_csv(summary_path, index=False)
phase_df.to_csv(phases_path, index=False)
gap_df.to_csv(peer_gap_path, index=False)

print()
print('Audit CSVs:')
print(f'  {summary_path.relative_to(PROJECT_ROOT)}')
print(f'  {phases_path.relative_to(PROJECT_ROOT)}')
print(f'  {peer_gap_path.relative_to(PROJECT_ROOT)}')
print()


# ────────────────────────────────────────────────────────────────────
# 6. Stdout findings
# ────────────────────────────────────────────────────────────────────
print('=' * 78)
print('Key findings')
print('=' * 78)
print()
print('Cumulative inflation 2001-01 → terminal (index space):')
for r in country_rows:
    print(f"  {r['country']:<8s}  "
          f"cumulative={r['cumulative_infl_pct']:>7.2f}%   "
          f"terminal_index={r['index_terminal']:>7.2f}   "
          f"window_end={r['level_end']}")
print()
print('Per-country YoY summary:')
for r in country_rows:
    print(f"  {r['country']:<8s}  "
          f"mean={r['yoy_mean']:>5.2f}  "
          f"std={r['yoy_std']:>5.2f}  "
          f"min={r['yoy_min']:>6.2f} ({r['yoy_min_date']})  "
          f"max={r['yoy_max']:>5.2f} ({r['yoy_max_date']})  "
          f"defl_mo={r['deflation_months']:>3d}   "
          f"above-2%_mo={r['months_above_2pct']:>3d}")
print()
print('Japan phase decomposition (D-045; visible range only):')
for _, r in phase_df.iterrows():
    print(f"  {r['phase']:<16s}  {r['start']}..{r['end']}  "
          f"n={r['n_months']:>3d}   "
          f"mean={r['yoy_mean']:>+5.2f}  "
          f"std={r['yoy_std']:>5.2f}  "
          f"[{r['yoy_min']:>+5.2f} .. {r['yoy_max']:>+5.2f}]   "
          f"defl_mo={r['deflation_months']:>3d}")
print()
print('Japan peer-gap (Japan YoY − mean of USA/UK/Germany YoY):')
r = gap_rows[0]
print(f"  window: {r['window_start']}..{r['window_end']}   n={r['n']}")
print(f"  mean_gap  = {r['gap_mean']:+.3f} pp")
print(f"  std_gap   = {r['gap_std']:.3f} pp")
print(f"  min_gap   = {r['gap_min']:+.3f} pp   on {r['gap_min_date']}")
print(f"  max_gap   = {r['gap_max']:+.3f} pp   on {r['gap_max_date']}")
print(f"  months_below_zero = {r['months_below_zero']}   "
      f"months_above_zero = {r['months_above_zero']}")
print()
print('NOTE: peer-gap terminates 2025-03 (UK/GER data end);')
print('      Japan YoY values 2025-04..2025-10 are plotted in Panels A+B')
print('      but are outside the peer-comparison window by construction.')
print()
print('Step 1 complete (v3).')
