"""
scripts/regenerate_phase2_audits.py
====================================
Regenerate the Phase 2 audit/documentation CSVs that are referenced by
ProjectDriven.md, README.md, and the two notebooks but may not be present
in data/documentation/ after a clean clone or cleanup.

This produces THREE reproducible audit artefacts:

  1. phase2_germany_m2_scout.csv
     Scout log of 10 candidate FRED Series IDs evaluated during D-021,
     confirming that every Germany-specific M2 variant terminated at
     euro adoption (1998-12) and that MABMM301EZM657S is the only
     FRESH Euro-area alternative. Requires FRED_API_KEY.

  2. phase2_m2_yoy_validation.csv
     Empirical validation underpinning D-012 (amended): pre-/post-
     transformation statistics for all 5 countries' M2 series, with
     peak-date economic-history cross-checks. Uses data/raw/.

  3. phase2_cleaning_log.csv
     Run-by-run audit of build_all_processed() outputs. Generated
     automatically by scripts/rebuild_processed.py; this module calls
     that underlying logic as a convenience.

Usage
-----
Run from project root with FRED credentials configured:

    python scripts/regenerate_phase2_audits.py

If FRED_API_KEY is unavailable, the Germany M2 scout is skipped with a
warning; the other two CSVs can still be generated from local raw data.

The Chow-Lin IP scout CSVs (phase2_ip_scout.csv,
phase2_ip_scout_tier1_expanded.csv, phase2_ip_native_fetch_log.csv)
are NOT regenerated here because reconstructing the full 45-candidate
scout requires significant API traffic and the analytical conclusion
(Chow-Lin rejected) is unchanged. If those files are missing, the
narrative in ProjectDriven.md D-018 and notebook 02 Appendix A still
stands on the retained UK_IP.csv and the rationale text.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable when invoked from any cwd inside the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from src.data_loader import (   # noqa: E402
    find_project_root,
    load_all_raw,
    MAIN_COUNTRIES,
    SUPPLEMENTARY_COUNTRIES,
)
from src.preprocessing import m2_to_yoy, M2_UNITS   # noqa: E402


# ──────────────────────────────────────────────────────────────────
# 1. Germany M2 scout (D-021)
# ──────────────────────────────────────────────────────────────────
GERMANY_M2_CANDIDATES = [
    # Germany-specific legacy
    ('germany-legacy',  'MYAGM2DEM189S',   'Germany-specific legacy broad money (pre-euro)'),
    # Germany-specific OECD harmonised
    ('germany-oecd',    'MABMM301DEM189S', 'Germany-specific OECD broad money, level'),
    ('germany-oecd',    'MABMM301DEM657S', 'Germany-specific OECD broad money, YoY growth'),
    ('germany-oecd',    'MANMM101DEM189S', 'Germany-specific OECD narrow money, level'),
    ('germany-oecd',    'MANMM101DEM657S', 'Germany-specific OECD narrow money, YoY growth'),
    # Euro-area aggregate
    ('euro-area',       'MABMM301EZM189S', 'Euro-area OECD broad money, level'),
    ('euro-area',       'MABMM301EZM657S', 'Euro-area OECD broad money, YoY growth'),
    ('euro-area',       'MANMM101EZM189S', 'Euro-area OECD narrow money, level'),
    ('euro-area',       'MANMM101EZM657S', 'Euro-area OECD narrow money, YoY growth'),
    # ECB M3 direct (broader than M2)
    ('euro-area-ecb',   'MYAGM3EZM196N',   'ECB M3 direct, monthly'),
]


def run_germany_m2_scout(doc_dir: Path) -> bool:
    """
    Evaluate all 10 Germany M2 candidates on FRED; write
    phase2_germany_m2_scout.csv. Returns True on success, False if
    skipped (no FRED key).
    """
    try:
        from dotenv import load_dotenv
        from fredapi import Fred
    except ImportError:
        print("  SKIP (python-dotenv / fredapi not installed)")
        return False

    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        print("  SKIP (FRED_API_KEY not set — scout requires FRED access)")
        print("  You can still review the scout outcomes in ProjectDriven.md D-021.")
        return False

    fred = Fred(api_key=fred_key)
    today = pd.Timestamp.today().normalize().replace(day=1)

    rows = []
    for group, sid, description in GERMANY_M2_CANDIDATES:
        time.sleep(0.15)   # be kind to the FRED API
        try:
            s = fred.get_series(sid, observation_start='1995-01-01')
            if s is None or s.empty:
                rows.append({
                    'group': group,
                    'series_id': sid,
                    'description': description,
                    'status': 'empty',
                    'effective_start': '',
                    'effective_end': '',
                    'months_outdated': '',
                    'freshness': 'MISSING',
                    'n_obs_2000plus': 0,
                    'note': 'FRED returned no data',
                })
                continue

            s2 = s.dropna()
            s_2000 = s2[s2.index >= '2000-01-01']
            eff_end = s2.index.max()
            months_out = (today.year - eff_end.year) * 12 + (today.month - eff_end.month)

            if months_out < 6:
                fresh = 'FRESH'
            elif months_out < 24:
                fresh = 'WARNING'
            else:
                fresh = 'CRITICAL'

            rows.append({
                'group': group,
                'series_id': sid,
                'description': description,
                'status': 'ok',
                'effective_start': s2.index.min().strftime('%Y-%m'),
                'effective_end': eff_end.strftime('%Y-%m'),
                'months_outdated': months_out,
                'freshness': fresh,
                'n_obs_2000plus': len(s_2000),
                'note': '',
            })
        except Exception as e:
            msg = str(e)[:120]
            rows.append({
                'group': group,
                'series_id': sid,
                'description': description,
                'status': 'error',
                'effective_start': '',
                'effective_end': '',
                'months_outdated': '',
                'freshness': 'MISSING',
                'n_obs_2000plus': 0,
                'note': f'Series does not exist or API error: {msg}',
            })

    df = pd.DataFrame(rows)
    out_path = doc_dir / 'phase2_germany_m2_scout.csv'
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}  ({len(df)} candidates)")
    return True


# ──────────────────────────────────────────────────────────────────
# 2. M2 YoY transformation validation (D-012 amended)
# ──────────────────────────────────────────────────────────────────
# Economic-history reference for cross-validation of the post-conversion peaks.
EXPECTED_PEAKS = {
    'USA':     ('2021-02', 'Fed COVID-era balance sheet expansion'),
    'JAPAN':   ('2021-02', 'BoJ COVID-era quantitative expansion'),
    'UK':      ('2008-12', 'BoE post-Lehman liquidity injection'),
    'GERMANY': ('2007-11', 'Pre-GFC Euro-area credit boom'),
    'CHINA':   ('2010-01', 'Post-GFC 4-trillion-yuan stimulus'),
}


def run_m2_yoy_validation(doc_dir: Path, raw: dict) -> None:
    """Write phase2_m2_yoy_validation.csv: pre/post-transformation stats."""
    rows = []
    for country in MAIN_COUNTRIES + SUPPLEMENTARY_COUNTRIES:
        s_raw = raw[country]['M2'].dropna()
        unit = M2_UNITS[country]

        # Raw descriptive stats
        raw_min = float(s_raw.min())
        raw_max = float(s_raw.max())
        raw_mean = float(s_raw.mean())
        raw_std = float(s_raw.std())

        # Transformed YoY
        s_yoy = m2_to_yoy(s_raw, unit).dropna()
        s_win = s_yoy[s_yoy.index >= '2001-01-01']

        if len(s_win) == 0:
            peak_date = ''
            peak_value = float('nan')
        else:
            peak_idx = s_win.idxmax()
            peak_date = peak_idx.strftime('%Y-%m')
            peak_value = float(s_win.loc[peak_idx])

        expected = EXPECTED_PEAKS.get(country, ('', ''))
        rows.append({
            'country': country,
            'unit_decision': unit,
            'raw_min': round(raw_min, 4),
            'raw_max': round(raw_max, 4),
            'raw_mean': round(raw_mean, 4),
            'raw_std': round(raw_std, 4),
            'yoy_min_pct': round(float(s_win.min()), 2) if len(s_win) else np.nan,
            'yoy_max_pct': round(peak_value, 2) if not np.isnan(peak_value) else np.nan,
            'yoy_peak_date': peak_date,
            'expected_peak_date': expected[0],
            'peak_match': (peak_date == expected[0]),
            'historical_event': expected[1],
        })

    df = pd.DataFrame(rows)
    out_path = doc_dir / 'phase2_m2_yoy_validation.csv'
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}  ({len(df)} countries)")


# ──────────────────────────────────────────────────────────────────
# 3. Phase 2 cleaning log (shared with rebuild_processed.py)
# ──────────────────────────────────────────────────────────────────
def run_cleaning_log(doc_dir: Path, processed_dir: Path) -> None:
    """Regenerate phase2_cleaning_log.csv from current data/processed/ files."""
    from src.data_loader import load_processed_main, load_processed_china

    rows = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for country in MAIN_COUNTRIES:
        try:
            df = load_processed_main(country)
        except FileNotFoundError:
            print(f"  SKIP {country} (no processed CSV; run rebuild_processed.py first)")
            continue
        nan_total = int(df.isna().sum().sum())
        rows.append({
            'file': f'main_{country.lower()}.csv',
            'country': country, 'category': 'main',
            'n_rows': df.shape[0], 'n_cols': df.shape[1],
            'start': df.index.min().strftime('%Y-%m'),
            'end': df.index.max().strftime('%Y-%m'),
            'nan_total': nan_total,
            'timestamp': ts,
        })

    try:
        df_china = load_processed_china()
        rows.append({
            'file': 'supplementary_china.csv',
            'country': 'CHINA', 'category': 'supplementary',
            'n_rows': df_china.shape[0], 'n_cols': df_china.shape[1],
            'start': df_china.index.min().strftime('%Y-%m'),
            'end': df_china.index.max().strftime('%Y-%m'),
            'nan_total': int(df_china.isna().sum().sum()),
            'timestamp': ts,
        })
    except FileNotFoundError:
        print("  SKIP CHINA supplementary (no processed CSV)")

    if not rows:
        print("  No processed files found; cleaning log not written")
        return

    log_df = pd.DataFrame(rows)
    out_path = doc_dir / 'phase2_cleaning_log.csv'
    if out_path.exists():
        existing = pd.read_csv(out_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows appended)")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    root = find_project_root()
    doc_dir = root / 'data' / 'documentation'
    processed_dir = root / 'data' / 'processed'
    doc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {root}")
    print("Regenerating Phase 2 audit CSVs...\n")

    print("1. Germany M2 scout (D-021):")
    scout_ok = run_germany_m2_scout(doc_dir)

    print("\n2. M2 YoY transformation validation (D-012 amended):")
    raw = load_all_raw()
    run_m2_yoy_validation(doc_dir, raw)

    print("\n3. Phase 2 cleaning log:")
    run_cleaning_log(doc_dir, processed_dir)

    print("\nDone.")
    if not scout_ok:
        print("\nNote: Germany M2 scout skipped. To regenerate it later, ensure "
              ".env contains FRED_API_KEY and re-run this script.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
