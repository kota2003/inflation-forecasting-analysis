"""
scripts/phase7_s4_covid_excluded.py

Phase 7 S4 — COVID-Origin-Excluded DM Sensitivity
=================================================

Re-runs the 25-cell DM battery from S2 with the 2020 Q1–Q3
walk-forward origins excluded, and emits a verdict-delta view
that makes the COVID-origin influence on each cell explicit.

Excluded origins: 2020-03-01 through 2020-08-01 (six months
spanning the COVID onset, lockdown, and initial reopening).
For each affected cell the paired sample drops from:

    58 → 52  (USA, JAPAN)
    51 → 45  (UK, GERMANY)

Both trimmed sizes remain above the n = 30 underpowered threshold,
so the DM asymptotic distribution remains usable.

Primary motivation (D-061 generalised): the S2 battery flagged
four cells with `std=tie, rob=significant` pattern. The largest
squared-loss differential magnitude is at USA primary h=12 VAR-Ridge
with `d_mean_squared = +296.67` — diagnostic of the same extreme-
error pattern D-061 originally identified for UK h=12 at the
2020-05-01 origin. S4 tests whether those four `std=tie` verdicts
convert to signed winners once the outlier-generating origins are
removed.

Inputs
------
    data/documentation/phase7_s1_unified_forecasts.csv
    data/documentation/phase7_s1_coverage_matrix.csv
    data/documentation/phase7_s2_dm_matrix.csv     (for delta comparison)

Outputs
-------
    data/documentation/phase7_s4_dm_trimmed_matrix.csv
        25 rows × 20 cols — same structure as S2 matrix but computed
        on origin-trimmed paired data. Includes n_excluded per cell.

    data/documentation/phase7_s4_verdict_delta.csv
        25 rows × 13 cols — compact side-by-side comparison of S2
        vs S4 winners under all three DM variants, with flip flags
        and a `change_description` column suitable for direct
        inclusion in notebook 09 §7.

Decision linkage: D-061, D-062, D-068, D-070, D-071, D-076, D-077,
D-078. Expected decision: **D-079** (renumbered from the original
plan; the original D-079 HAC slot was closed inside D-078).

Exit codes
----------
    0  clean run (two CSVs written)
    1  at least one cell produced NaN DM stat on trimmed data
    2  FATAL: S1 or S2 inputs missing
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "data").is_dir() and (cand / "src").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {Path.cwd()}")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    diebold_mariano_standard,
    diebold_mariano_hac,
    diebold_mariano_robust,
    align_matched_terms,
    __version__ as SRC_VERSION,
)

DOC_DIR         = PROJECT_ROOT / "data" / "documentation"
IN_UNIFIED      = DOC_DIR / "phase7_s1_unified_forecasts.csv"
IN_COVERAGE     = DOC_DIR / "phase7_s1_coverage_matrix.csv"
IN_S2_MATRIX    = DOC_DIR / "phase7_s2_dm_matrix.csv"
OUT_TRIMMED     = DOC_DIR / "phase7_s4_dm_trimmed_matrix.csv"
OUT_DELTA       = DOC_DIR / "phase7_s4_verdict_delta.csv"

# ──────────────────────────────────────────────────────────────────────
# COVID-origin exclusion set
# ──────────────────────────────────────────────────────────────────────
# Six walk-forward origin months spanning COVID onset, lockdown, and
# initial reopening. Consistent with D-061's flagged 2020 Q1–Q3
# "regime-transition stress test" window.

COVID_ONSET_ORIGINS = pd.to_datetime([
    "2020-03-01", "2020-04-01", "2020-05-01",
    "2020-06-01", "2020-07-01", "2020-08-01",
])

ALPHA_MAIN   = 0.05
ALPHA_STRICT = 0.01


# ──────────────────────────────────────────────────────────────────────
# Per-cell DM re-computation on trimmed origins
# ──────────────────────────────────────────────────────────────────────

def _winner_at(alpha: float, dm_stat: float, p_value: float,
               layer_1: str, layer_2: str) -> str:
    if not np.isfinite(dm_stat) or not np.isfinite(p_value):
        return "undefined"
    if p_value >= alpha:
        return "tie"
    return layer_1 if dm_stat < 0 else layer_2


def run_trimmed_dm_for_cell(cell: pd.Series,
                            unified: pd.DataFrame) -> dict[str, Any]:
    """Same as S2's per-cell DM, but on origin-trimmed data."""
    country = str(cell["country"])
    form    = str(cell["form"])
    h       = int(cell["h"])
    L1      = str(cell["layer_1"])
    L2      = str(cell["layer_2"])
    pair    = str(cell["pair"])

    def _filter(layer: str) -> pd.DataFrame:
        base = unified[
            (unified["layer"]   == layer)   &
            (unified["country"] == country) &
            (unified["form"]    == form)    &
            (unified["h"]       == h)
        ]
        # Drop rows whose walk-forward origin falls in the COVID-onset window
        return base[~base["origin_date"].isin(COVID_ONSET_ORIGINS)]

    d1_trim = _filter(L1)
    d2_trim = _filter(L2)

    # Also compute the full-origin count (from the same filter without the
    # origin-drop) to record n_excluded accurately per cell
    def _full(layer: str) -> int:
        return int(len(unified[
            (unified["layer"]   == layer)   &
            (unified["country"] == country) &
            (unified["form"]    == form)    &
            (unified["h"]       == h)
        ]))
    n_full_1 = _full(L1)
    n_full_2 = _full(L2)
    n_excluded_1 = n_full_1 - len(d1_trim)
    n_excluded_2 = n_full_2 - len(d2_trim)

    try:
        y, e1, e2 = align_matched_terms(d1_trim, d2_trim)
    except ValueError as e:
        # No matched rows after trimming (should not happen for β-option
        # scope because 45+ paired rows remain) — record an error row
        return {
            "country": country, "form": form, "h": h, "pair": pair,
            "layer_1": L1, "layer_2": L2,
            "n_excluded_1": n_excluded_1, "n_excluded_2": n_excluded_2,
            "n_paired": 0,
            "error": f"{type(e).__name__}: {e}",
            "beta_option": bool(cell["beta_option"]),
            "dm_scope": str(cell["dm_scope"]),
        }

    n = int(len(y))
    d_sq = e1 ** 2 - e2 ** 2
    d_ab = np.abs(e1) - np.abs(e2)

    dm_std, p_std = diebold_mariano_standard(e1, e2, h=h)
    dm_hac, p_hac = diebold_mariano_hac(e1, e2, h=h)
    dm_rob, p_rob = diebold_mariano_robust(e1, e2, h=h)

    return {
        "country": country, "form": form, "h": h, "pair": pair,
        "layer_1": L1, "layer_2": L2,
        "n_excluded_1": n_excluded_1, "n_excluded_2": n_excluded_2,
        "n_paired": n,
        "d_mean_squared":  float(d_sq.mean()),
        "d_mean_absolute": float(d_ab.mean()),
        "dm_standard": float(dm_std) if np.isfinite(dm_std) else np.nan,
        "p_standard":  float(p_std)  if np.isfinite(p_std)  else np.nan,
        "dm_hac":      float(dm_hac) if np.isfinite(dm_hac) else np.nan,
        "p_hac":       float(p_hac)  if np.isfinite(p_hac)  else np.nan,
        "dm_robust":   float(dm_rob) if np.isfinite(dm_rob) else np.nan,
        "p_robust":    float(p_rob)  if np.isfinite(p_rob)  else np.nan,
        "winner_standard_5pct": _winner_at(ALPHA_MAIN, dm_std, p_std, L1, L2),
        "winner_hac_5pct":      _winner_at(ALPHA_MAIN, dm_hac, p_hac, L1, L2),
        "winner_robust_5pct":   _winner_at(ALPHA_MAIN, dm_rob, p_rob, L1, L2),
        "beta_option": bool(cell["beta_option"]),
        "dm_scope":    str(cell["dm_scope"]),
    }


# ──────────────────────────────────────────────────────────────────────
# Verdict delta builder
# ──────────────────────────────────────────────────────────────────────

def _describe_flip(before: str, after: str) -> str:
    if before == after:
        return "no change"
    return f"{before} → {after}"


def build_verdict_delta(trimmed: pd.DataFrame,
                        s2_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compact side-by-side S2 vs S4 verdict comparison."""
    # Align both DataFrames on (country, form, h, pair) — this is the
    # natural cell key; we'll merge on it.
    key = ["country", "form", "h", "pair"]
    merged = s2_matrix[key + [
        "winner_standard_5pct", "winner_hac_5pct", "winner_robust_5pct",
        "p_standard", "p_hac", "p_robust",
        "dm_standard", "dm_hac", "dm_robust",
    ]].rename(columns={
        "winner_standard_5pct": "winner_std_s2",
        "winner_hac_5pct":      "winner_hac_s2",
        "winner_robust_5pct":   "winner_rob_s2",
        "p_standard":           "p_std_s2",
        "p_hac":                "p_hac_s2",
        "p_robust":             "p_rob_s2",
        "dm_standard":          "dm_std_s2",
        "dm_hac":               "dm_hac_s2",
        "dm_robust":            "dm_rob_s2",
    }).merge(trimmed[key + [
        "winner_standard_5pct", "winner_hac_5pct", "winner_robust_5pct",
        "p_standard", "p_hac", "p_robust",
        "dm_standard", "dm_hac", "dm_robust",
        "n_paired",
    ]].rename(columns={
        "winner_standard_5pct": "winner_std_s4",
        "winner_hac_5pct":      "winner_hac_s4",
        "winner_robust_5pct":   "winner_rob_s4",
        "p_standard":           "p_std_s4",
        "p_hac":                "p_hac_s4",
        "p_robust":             "p_rob_s4",
        "dm_standard":          "dm_std_s4",
        "dm_hac":               "dm_hac_s4",
        "dm_robust":            "dm_rob_s4",
        "n_paired":             "n_paired_s4",
    }), on=key, how="inner")

    # Flip flags and change descriptions per variant
    merged["flip_std"] = merged["winner_std_s2"] != merged["winner_std_s4"]
    merged["flip_hac"] = merged["winner_hac_s2"] != merged["winner_hac_s4"]
    merged["flip_rob"] = merged["winner_rob_s2"] != merged["winner_rob_s4"]
    merged["any_flip"] = merged["flip_std"] | merged["flip_hac"] | merged["flip_rob"]
    merged["change_std"] = [
        _describe_flip(a, b) for a, b in zip(merged["winner_std_s2"], merged["winner_std_s4"])
    ]
    merged["change_rob"] = [
        _describe_flip(a, b) for a, b in zip(merged["winner_rob_s2"], merged["winner_rob_s4"])
    ]

    # Select a compact column order
    out_cols = [
        "country", "form", "h", "pair", "n_paired_s4",
        "winner_std_s2", "winner_std_s4", "change_std",
        "winner_rob_s2", "winner_rob_s4", "change_rob",
        "flip_std", "flip_rob", "any_flip",
    ]
    return merged[out_cols]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

SEP = "=" * 76
SUB = "-" * 76


def main() -> int:
    print(SEP)
    print("Phase 7 S4 — COVID-Origin-Excluded DM Sensitivity")
    print(SEP)
    print(f"src version : {SRC_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Excluded walk-forward origins ({len(COVID_ONSET_ORIGINS)}):")
    for ts in COVID_ONSET_ORIGINS:
        print(f"  {ts.date().isoformat()}")

    # ── Step 1: Load S1 + S2 outputs ───────────────────────────────
    print()
    print(SUB)
    print("Step 1 · Load S1 + S2 audit CSVs")
    print(SUB)
    try:
        unified   = pd.read_csv(IN_UNIFIED,   parse_dates=["origin_date", "target_date"])
        coverage  = pd.read_csv(IN_COVERAGE)
        s2_matrix = pd.read_csv(IN_S2_MATRIX)
    except FileNotFoundError as e:
        print(f"  [FATAL] Missing input: {e}")
        print(f"  Run S1 and S2 first.")
        return 2

    print(f"  unified    shape: {unified.shape}")
    print(f"  coverage   shape: {coverage.shape}")
    print(f"  s2_matrix  shape: {s2_matrix.shape}")

    ok_cells = coverage[coverage["status"] == "OK"].reset_index(drop=True)
    print(f"  Iterable cells (status=OK): {len(ok_cells)}")

    # ── Step 2: Re-run DM battery on trimmed origins ───────────────
    print()
    print(SUB)
    print("Step 2 · Re-run DM battery on origin-trimmed data")
    print(SUB)
    rows: list[dict[str, Any]] = []
    for _, cell in ok_cells.iterrows():
        rows.append(run_trimmed_dm_for_cell(cell, unified))
    trimmed = pd.DataFrame(rows)

    # Handle error rows (no error column unless explicitly populated)
    if "error" in trimmed.columns:
        n_err = int(trimmed["error"].notna().sum())
        if n_err > 0:
            print(f"  [WARN] {n_err} cells raised during alignment after trim.")

    # Drop error column if fully empty (keeps output tidy)
    if "error" in trimmed.columns and trimmed["error"].notna().sum() == 0:
        trimmed = trimmed.drop(columns=["error"])

    trimmed.to_csv(OUT_TRIMMED, index=False)
    print(f"  Wrote {OUT_TRIMMED.name}: {len(trimmed)} cells × {len(trimmed.columns)} cols")

    # ── Step 3: Build verdict delta ────────────────────────────────
    print()
    print(SUB)
    print("Step 3 · Build verdict delta (S2 vs S4)")
    print(SUB)
    delta = build_verdict_delta(trimmed, s2_matrix)
    delta.to_csv(OUT_DELTA, index=False)
    print(f"  Wrote {OUT_DELTA.name}: {len(delta)} cells × {len(delta.columns)} cols")

    # ── Step 4: Decision-gate readout ──────────────────────────────
    print()
    print(SUB)
    print("Step 4 · Verdict changes under origin trimming")
    print(SUB)
    n_total     = len(delta)
    n_flip_std  = int(delta["flip_std"].sum())
    n_flip_rob  = int(delta["flip_rob"].sum())
    n_any_flip  = int(delta["any_flip"].sum())
    print(f"  Standard DM winner flips (S2 → S4): {n_flip_std}/{n_total}")
    print(f"  Robust   DM winner flips (S2 → S4): {n_flip_rob}/{n_total}")
    print(f"  Any-variant flips:                   {n_any_flip}/{n_total}")

    if n_any_flip > 0:
        print()
        print("  Flip detail:")
        flipped = delta[delta["any_flip"]]
        for _, r in flipped.iterrows():
            tag_std = "✓" if r["flip_std"] else " "
            tag_rob = "✓" if r["flip_rob"] else " "
            print(f"    {r['country']:8s} {r['form']:10s} h={r['h']:<2d} {r['pair']:13s}  "
                  f"std[{tag_std}]: {r['change_std']:25s}  "
                  f"rob[{tag_rob}]: {r['change_rob']}")

    # ── Step 5: Focus on S2-flagged cells (4 cells with std=tie, rob=signif) ──
    print()
    print(SUB)
    print("Step 5 · Follow-up on S2's 4 robust-flagged cells")
    print(SUB)
    # The 4 cells from S2 where winner_robust != winner_standard
    focus_cells = s2_matrix[
        s2_matrix["winner_standard_5pct"] != s2_matrix["winner_robust_5pct"]
    ][["country", "form", "h", "pair"]]
    focus = delta.merge(focus_cells, on=["country", "form", "h", "pair"], how="inner")
    if len(focus) > 0:
        print(focus[[
            "country", "form", "h", "pair",
            "winner_std_s2", "winner_std_s4", "change_std",
            "winner_rob_s2", "winner_rob_s4", "change_rob",
        ]].to_string(index=False))
    else:
        print("  (no S2-robust-flagged cells found)")

    # ── Step 6: Full trimmed matrix (compact view) ─────────────────
    print()
    print(SUB)
    print("Step 6 · Full trimmed matrix (compact)")
    print(SUB)
    compact = trimmed[[
        "country", "form", "h", "pair", "n_paired",
        "dm_standard", "p_standard",
        "dm_robust", "p_robust",
        "winner_standard_5pct", "winner_robust_5pct",
    ]].copy()
    for c in ("dm_standard", "dm_robust"):
        compact[c] = compact[c].round(3)
    for c in ("p_standard", "p_robust"):
        compact[c] = compact[c].round(4)
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(compact.to_string(index=False))

    # ── Exit verdict ───────────────────────────────────────────────
    print()
    print(SEP)
    n_nan = int(trimmed[["dm_standard", "dm_hac", "dm_robust"]].isna().any(axis=1).sum())
    if n_nan > 0:
        print(f"S4 AMBER — {n_nan} cell(s) produced NaN DM stat. Investigate before D-079.")
        print(SEP)
        return 1
    if n_any_flip == 0:
        print("S4 GREEN — 0 verdict flips under origin trimming. S2 verdicts are robust.")
    else:
        print(f"S4 GREEN — {n_any_flip} verdict change(s) under origin trimming; see delta CSV.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
