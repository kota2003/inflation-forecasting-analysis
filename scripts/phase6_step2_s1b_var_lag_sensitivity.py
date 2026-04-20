"""
scripts/phase6_step2_s1b_var_lag_sensitivity.py
================================================
Phase 6 · Step 2 · S1b — VAR lag boundary sensitivity check.

Purpose
-------
S1 (`phase6_step2_var_lag_selection.py`) tested `maxlag = 12` and
observed AIC boundary-hits at lag 12 for three of four countries:

    USA:      AIC=12 ★  (vs BIC=2, HQIC=2)
    UK:       AIC=12 ★  (vs BIC=2, HQIC=2)
    GERMANY:  AIC=12 ★  (vs BIC=2, HQIC=3)
    JAPAN:    AIC=5     (interior minimum — no boundary issue)

This S1b script extends the VAR lag grid to `maxlag = 18` (1.5×
original upper bound; matches the D-048 Stage (b) precedent of
"neighbourhood extension" rather than unbounded escalation) and
classifies the AIC behaviour in the extension window {lag 13..18}
against the Burnham & Anderson 2002 `ΔAIC ≤ −2.0` "meaningfully
better" convention.

Design (D-048 Stage (b) pattern echo)
-------------------------------------
The three decision steps mirror D-048 for ARIMA:

  1. Select extension maxlag (= 18 here; precedent: Q=3 for D-048).
  2. Rerun the full information-criteria table for lags 0..18.
  3. For each boundary-hit variant, compute
        Δ_min = min(AIC[13..18]) − AIC[12]
     and classify:
        Δ_min ≤ −2.0   → "extend_further"
                         (reject S1 maxlag=12; AIC has meaningful
                          interior minimum in the extension zone)
        Δ_min >  −2.0  → "accept_lag12_boundary_locked"
                         (AIC improvement in extension is not
                          meaningful by B&A convention; S1 scope
                          locked)

If verdict == "accept_lag12_boundary_locked" for all boundary-hit
countries, D-048-style "OOS saturation stopping rule" applies: AIC
is monotone but non-informative at the boundary; BIC-based p*=2 is
locked and AIC-based p*=12 is carried as a sensitivity candidate.

Scope boundaries (S1b only)
---------------------------
- Extension of the lag grid for AIC boundary verification only.
- JAPAN is rerun for completeness (interior minimum confirmation).
- No estimation, Granger, IRF, or FEVD. Those are S2 / S3 / S4 / S5.
- Full-sample selection (same as S1).

Output artefacts
----------------
data/documentation/
    phase6_step2_s1b_sensitivity_values.csv
        Per-country × lag (0..18) × AIC/BIC/HQIC/FPE values,
        with delta_from_lag12 columns for each IC and a
        boundary-zone flag. 4 countries × 19 lags = 76 rows × 11 cols.
    phase6_step2_s1b_sensitivity_verdict.csv
        One row per country × (s1_aic_selected_lag,
        s1_aic_value_at_lag12, s1b_min_aic_in_extension,
        s1b_argmin_lag_in_extension, delta_min_vs_lag12, verdict).
        4 rows × 7 cols.

Decisions referenced
--------------------
D-048  ARIMA Stage (b) precedent — "neighbourhood extension" with
       ΔAIC ≤ −2.0 threshold and OOS-saturation stopping rule.
D-050  (candidate, to be confirmed after S1b verdict) — VAR lag
       selection scope and BIC-primary / AIC-sensitivity protocol.

Usage
-----
    (p3_inflation) $ python scripts/phase6_step2_s1b_var_lag_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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
)
from statsmodels.tsa.vector_ar.var_model import VAR            # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────

#: S1 original upper bound (from phase6_step2_var_lag_selection.py).
S1_MAXLAG: int = 12

#: S1b extended upper bound. 1.5x original; matches D-048 Stage (b)
#: "neighbourhood extension" philosophy rather than unbounded escalation.
S1B_MAXLAG: int = 18

#: Burnham & Anderson 2002 "meaningfully better" threshold. Applied to
#: ΔAIC = min(AIC[extension]) − AIC[baseline_lag]. Identical to D-048.
BA_THRESHOLD: float = -2.0

#: Countries that hit AIC boundary in S1 at lag 12.
BOUNDARY_HIT: tuple[str, ...] = ('USA', 'UK', 'GERMANY')

#: Information criteria mirrored from S1.
VAR_CRITERIA: tuple[str, ...] = ('aic', 'bic', 'hqic', 'fpe')

#: Five base endogenous variables (VAR system).
BASE_INDICATORS: list[str] = list(INDICATORS)


# ── Helpers ───────────────────────────────────────────────────────────

def extract_base_block(features_df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Subset a country feature matrix to the 5-variable endogenous block."""
    cols = [f'{country}_{ind}' for ind in BASE_INDICATORS]
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise KeyError(f"{country}: missing base columns {missing}")
    base = features_df[cols].dropna(how='any')
    if base.empty:
        raise ValueError(f"{country}: 5-var base block is empty after dropna")
    return base


def select_var_order_extended(base_df: pd.DataFrame, maxlag: int) -> dict:
    """Run VAR.select_order at extended maxlag and extract full IC table."""
    model = VAR(base_df)
    lag_res = model.select_order(maxlags=maxlag)
    ic_table = pd.DataFrame(
        {crit: list(lag_res.ics[crit]) for crit in VAR_CRITERIA}
    )
    ic_table.index.name = 'lag'
    ic_table.index = ic_table.index.astype(int)
    selected = {crit: int(lag_res.selected_orders[crit]) for crit in VAR_CRITERIA}
    return {
        'ic_table': ic_table,
        'selected_s1b_maxlag': selected,
        'n_obs': len(base_df),
        'start': base_df.index.min(),
        'end':   base_df.index.max(),
    }


def classify_verdict(
    ic_table: pd.DataFrame,
    baseline_lag: int,
    threshold: float,
    is_boundary_hit: bool,
) -> dict:
    """Classify AIC behaviour in the extension window {baseline+1, ..., maxlag}.

    Returns a dict with diagnostic fields and the final verdict string.
    """
    maxlag = int(ic_table.index.max())
    baseline_aic = float(ic_table.loc[baseline_lag, 'aic'])

    extension_idx = [l for l in ic_table.index if l > baseline_lag]
    extension_aic = ic_table.loc[extension_idx, 'aic']
    argmin_lag = int(extension_aic.idxmin())
    min_aic = float(extension_aic.min())
    delta_min = min_aic - baseline_aic

    if not is_boundary_hit:
        # Japan: interior minimum already found at S1. Confirm it survives
        # extension.
        s1b_global_argmin = int(ic_table['aic'].idxmin())
        if s1b_global_argmin <= baseline_lag:
            verdict = 'interior_min_confirmed'
        elif delta_min <= threshold:
            verdict = 'interior_min_shifted_to_extension'
        else:
            verdict = 'interior_min_stable_at_s1_pick'
    else:
        if delta_min <= threshold:
            verdict = 'extend_further'
        else:
            verdict = 'accept_lag12_boundary_locked'

    return {
        'baseline_lag':          baseline_lag,
        'baseline_aic':          baseline_aic,
        'extension_argmin_lag':  argmin_lag,
        'extension_min_aic':     min_aic,
        'delta_min_vs_baseline': delta_min,
        'threshold':             threshold,
        'meets_threshold':       delta_min <= threshold,
        'is_boundary_hit':       is_boundary_hit,
        'verdict':               verdict,
        'maxlag_tested':         maxlag,
    }


def build_values_csv(results: dict[str, dict], baseline_lag: int) -> pd.DataFrame:
    """Long-form CSV: country × lag × IC values + delta-from-baseline."""
    rows: list[dict] = []
    for country, res in results.items():
        tbl = res['ic_table']
        baseline_vals = {c: float(tbl.loc[baseline_lag, c]) for c in VAR_CRITERIA}
        for lag, row in tbl.iterrows():
            rec = {'country': country, 'lag': int(lag)}
            for crit in VAR_CRITERIA:
                rec[crit] = float(row[crit])
                rec[f'{crit}_delta_from_lag{baseline_lag}'] = (
                    float(row[crit]) - baseline_vals[crit]
                )
            rec['is_extension_zone'] = bool(lag > baseline_lag)
            rows.append(rec)
    return pd.DataFrame(rows)


def build_verdict_csv(
    results: dict[str, dict],
    verdicts: dict[str, dict],
    baseline_lag: int,
) -> pd.DataFrame:
    """One-row-per-country verdict summary."""
    rows = []
    for country in MAIN_COUNTRIES:
        v = verdicts[country]
        rows.append({
            'country':                     country,
            'is_boundary_hit_at_s1':       v['is_boundary_hit'],
            's1_baseline_lag':             v['baseline_lag'],
            's1_aic_at_baseline':          round(v['baseline_aic'], 4),
            's1b_maxlag_tested':           v['maxlag_tested'],
            's1b_extension_argmin_lag':    v['extension_argmin_lag'],
            's1b_min_aic_in_extension':    round(v['extension_min_aic'], 4),
            's1b_delta_min_vs_baseline':   round(v['delta_min_vs_baseline'], 4),
            'ba_threshold':                v['threshold'],
            'meets_ba_threshold':          v['meets_threshold'],
            'verdict':                     v['verdict'],
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    bar = '=' * 72
    print(bar)
    print('Phase 6 · Step 2 · S1b — VAR Lag Boundary Sensitivity Check')
    print(bar)
    print(f'baseline (S1):        maxlag = {S1_MAXLAG}')
    print(f'extension (S1b):      maxlag = {S1B_MAXLAG}')
    print(f'boundary-hit at S1:   {list(BOUNDARY_HIT)}')
    print(f'B&A threshold:        ΔAIC ≤ {BA_THRESHOLD} (precedent: D-048)')
    print(f'precedent:            D-048 Stage (b) neighbourhood extension')
    print()

    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Part 1 — Load features.
    # ------------------------------------------------------------------
    print('>>> Loading Phase 4 feature matrices ...')
    features = build_all_features()
    print()

    # ------------------------------------------------------------------
    # Part 2 — Per-country extended selection + verdict.
    # ------------------------------------------------------------------
    results:  dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    for country in MAIN_COUNTRIES:
        print(f'>>> {country} — VAR.select_order(maxlags={S1B_MAXLAG}) ...')
        base = extract_base_block(features[country], country)
        res = select_var_order_extended(base, S1B_MAXLAG)
        results[country] = res

        is_bh = country in BOUNDARY_HIT
        v = classify_verdict(res['ic_table'], S1_MAXLAG, BA_THRESHOLD, is_bh)
        verdicts[country] = v

        print(f'    n_obs={res["n_obs"]}  '
              f'sample {res["start"].date()}..{res["end"].date()}')
        print(f'    AIC @ lag {S1_MAXLAG}:        {v["baseline_aic"]:+.4f}')
        print(f'    min AIC in extension:   {v["extension_min_aic"]:+.4f}  '
              f'(at lag {v["extension_argmin_lag"]})')
        print(f'    Δ_min vs lag {S1_MAXLAG}:       {v["delta_min_vs_baseline"]:+.4f}  '
              f'(threshold {BA_THRESHOLD:+.1f})')
        print(f'    verdict:                 {v["verdict"]}')
        print()

    # ------------------------------------------------------------------
    # Part 3 — Write CSVs.
    # ------------------------------------------------------------------
    values_df = build_values_csv(results, S1_MAXLAG)
    values_path = doc_dir / 'phase6_step2_s1b_sensitivity_values.csv'
    values_df.to_csv(values_path, index=False)

    verdict_df = build_verdict_csv(results, verdicts, S1_MAXLAG)
    verdict_path = doc_dir / 'phase6_step2_s1b_sensitivity_verdict.csv'
    verdict_df.to_csv(verdict_path, index=False)

    print(bar)
    print('Extension-zone AIC behaviour (lag 13..18, delta vs lag 12)')
    print(bar)
    for country in MAIN_COUNTRIES:
        ic = results[country]['ic_table']
        baseline = float(ic.loc[S1_MAXLAG, 'aic'])
        print(f'\n  {country}:')
        for lag in range(S1_MAXLAG, S1B_MAXLAG + 1):
            val = float(ic.loc[lag, 'aic'])
            delta = val - baseline
            marker = ''
            if lag == S1_MAXLAG:
                marker = '  (baseline)'
            elif lag == verdicts[country]['extension_argmin_lag']:
                marker = '  ← argmin in extension'
            print(f'    lag {lag:2d}  AIC = {val:+.4f}   '
                  f'Δ = {delta:+.4f}{marker}')
    print()

    # ------------------------------------------------------------------
    # Part 4 — Verdict summary table.
    # ------------------------------------------------------------------
    print(bar)
    print('Verdict summary')
    print(bar)
    with pd.option_context('display.max_columns', None,
                           'display.width', 200):
        print(verdict_df[['country', 'is_boundary_hit_at_s1',
                          's1b_delta_min_vs_baseline',
                          's1b_extension_argmin_lag',
                          'meets_ba_threshold',
                          'verdict']].to_string(index=False))
    print()
    print(f'written: data/documentation/{values_path.name}')
    print(f'         data/documentation/{verdict_path.name}')
    print()

    # ------------------------------------------------------------------
    # Part 5 — Overall protocol verdict + next step pointer.
    # ------------------------------------------------------------------
    print(bar)
    boundary_verdicts = {c: verdicts[c]['verdict'] for c in BOUNDARY_HIT}
    all_accept = all(v == 'accept_lag12_boundary_locked'
                     for v in boundary_verdicts.values())
    any_extend = any(v == 'extend_further'
                     for v in boundary_verdicts.values())

    if all_accept:
        print('Overall protocol verdict:  ACCEPT S1 scope (maxlag = 12)')
        print()
        print('All boundary-hit countries fail the B&A threshold —')
        print('AIC improvement in the extension window is not meaningfully')
        print('better than lag 12. Monotone-but-uninformative AIC behaviour')
        print('is a finding in itself (D-048 OOS-saturation analogue for VAR).')
        print()
        print('D-050 to be confirmed with:')
        print('  - primary:      BIC-based p* = 2 across all four countries')
        print('                  (GER HQIC=3 and AIC=12 carried as sensitivity)')
        print('  - sensitivity:  AIC=12 (USA/UK/GER) and AIC=5 (JPN) for')
        print('                  robustness cross-check in Phase 7 DM comparison')
    elif any_extend:
        escalate = [c for c, v in boundary_verdicts.items()
                    if v == 'extend_further']
        print('Overall protocol verdict:  ESCALATE to S1c')
        print()
        print(f'Countries requiring Stage (c) extension:  {escalate}')
        print('Action: rerun with maxlag ≥ 24 or adopt AIC-extension interior')
        print('minimum as the selected order for these variants.')
    else:
        print('Overall protocol verdict:  MIXED — case-by-case S1c decision')
        print()
        for c, v in boundary_verdicts.items():
            print(f'  {c:<8}  {v}')

    print()
    print('Next sub-step: confirm D-050, then S2 = VAR estimation at p*,')
    print('with D-030 regime-interaction exogenous terms.')
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
