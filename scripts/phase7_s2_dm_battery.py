"""
scripts/phase7_s2_dm_battery.py

Phase 7 S2 — Diebold-Mariano Battery
====================================

Runs the full paired-Diebold-Mariano battery over the 25 cells
enumerated by S1 (`phase7_s1_coverage_matrix.csv`), applying all
three DM variants in a single pass:

  * **standard**  — squared-error loss differential, sample variance,
                    HLN small-sample correction (D-076 default)
  * **HAC**       — squared-error loss, Newey-West Bartlett long-run
                    variance with `n_lags = max(h-1, 0)`, HLN
                    correction (addresses D-051 partial-whitening)
  * **robust**    — absolute-error loss differential, sample variance,
                    HLN correction (addresses D-061 COVID-origin
                    outlier concerns)

Running all three variants in one script is an efficiency choice: each
variant is ~10 ms per cell, so 25 × 3 = 75 computations complete in
well under a second. A separate "S2b HAC sensitivity" sub-step is
therefore not required unless the post-S2 decision gate chooses to
sensitivity-analyse a specific HAC lag choice beyond the default.

Inputs
------
    data/documentation/phase7_s1_unified_forecasts.csv
    data/documentation/phase7_s1_coverage_matrix.csv

Outputs
-------
    data/documentation/phase7_s2_dm_matrix.csv
        One row per OK cell (25 rows) with raw DM results from all
        three variants, loss-differential means, α=0.01 / α=0.05
        significance flags, per-variant winners, a variant-agreement
        flag, and the ancestor-decision linkage.

    data/documentation/phase7_s2_dm_summary.csv
        Aggregated grouping metrics: per-pair, per-country,
        per-horizon, and overall. Used by the post-S2 decision gate
        to assess whether sensitivity sub-steps (S2b / S3 / S4) are
        needed, and by notebook 09 Section 4 for portfolio
        narrative-ready tables.

Sign convention
---------------
`dm_stat < 0` means layer_1 has lower loss (layer_1 is the better
forecaster); `dm_stat > 0` means layer_2 wins. The `winner_*_5pct`
columns resolve this into `{layer_1_name, layer_2_name, 'tie'}` at
α = 0.05.

Decision linkage
----------------
D-048, D-051, D-060, D-061, D-062, D-068, D-070, D-071, D-076, D-077.
Expected decision: **D-078** — DM battery verdict aggregated across
the 25 cells, with sensitivity-sub-step recommendation.

Exit codes
----------
    0  clean run (matrix + summary written)
    1  at least one cell produced NaN DM statistic (investigation
       needed before D-078 is written)
    2  FATAL: could not read S1 outputs
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

DOC_DIR        = PROJECT_ROOT / "data" / "documentation"
IN_UNIFIED     = DOC_DIR / "phase7_s1_unified_forecasts.csv"
IN_COVERAGE    = DOC_DIR / "phase7_s1_coverage_matrix.csv"
OUT_MATRIX     = DOC_DIR / "phase7_s2_dm_matrix.csv"
OUT_SUMMARY    = DOC_DIR / "phase7_s2_dm_summary.csv"

ALPHA_MAIN   = 0.05
ALPHA_STRICT = 0.01


# ──────────────────────────────────────────────────────────────────────
# Decision-linkage attribution
# ──────────────────────────────────────────────────────────────────────

def _decision_linkage(country: str, form: str, h: int, pair: str) -> str:
    """Attach ancestor-decision citations to a cell, in ID order.

    Every cell links to D-068 (walk-forward origin match — universal).
    Additional links are applied based on the cell's country/form/
    horizon/pair tuple.
    """
    links: set[str] = {"D-068"}  # walk-forward origin match — universal
    if country == "USA" and form == "primary":
        links.add("D-062")  # USA yoy_pct × VAR systematic bias
    if form == "secondary":
        links.add("D-071")  # USA dual-form resolution
    if "Ridge" in pair and "VAR" in pair:
        links.add("D-070")  # Ridge-vs-VAR 12/16 point-estimate win
    if "ARIMA" in pair:
        links.add("D-048")  # ARIMA stopping rule
    if "VAR" in pair:
        links.add("D-051")  # VAR(12) partial whitening → HAC motivation
    if h > 1:
        links.add("D-060")  # VAR MASE at AIC-selected p
    return ",".join(sorted(links))


def _winner_at(alpha: float, dm_stat: float, p_value: float,
               layer_1: str, layer_2: str) -> str:
    """Resolve (stat, p) into a winner label at the given significance."""
    if not np.isfinite(dm_stat) or not np.isfinite(p_value):
        return "undefined"
    if p_value >= alpha:
        return "tie"
    return layer_1 if dm_stat < 0 else layer_2


# ──────────────────────────────────────────────────────────────────────
# Per-cell DM computation — all three variants in one pass
# ──────────────────────────────────────────────────────────────────────

def run_dm_for_cell(cell: pd.Series, unified: pd.DataFrame) -> dict[str, Any]:
    """Run all 3 DM variants for a single cell from the coverage matrix."""
    country = str(cell["country"])
    form    = str(cell["form"])
    h       = int(cell["h"])
    L1      = str(cell["layer_1"])
    L2      = str(cell["layer_2"])
    pair    = str(cell["pair"])

    def _filter(layer: str) -> pd.DataFrame:
        return unified[
            (unified["layer"]   == layer)   &
            (unified["country"] == country) &
            (unified["form"]    == form)    &
            (unified["h"]       == h)
        ]

    d1 = _filter(L1)
    d2 = _filter(L2)

    y, e1, e2 = align_matched_terms(d1, d2)
    n = int(len(y))

    # Loss differentials (for interpretive context, not for the DM stat)
    d_sq = e1 ** 2 - e2 ** 2
    d_ab = np.abs(e1) - np.abs(e2)

    dm_std, p_std = diebold_mariano_standard(e1, e2, h=h)
    dm_hac, p_hac = diebold_mariano_hac(e1, e2, h=h)
    dm_rob, p_rob = diebold_mariano_robust(e1, e2, h=h)

    w_std = _winner_at(ALPHA_MAIN, dm_std, p_std, L1, L2)
    w_hac = _winner_at(ALPHA_MAIN, dm_hac, p_hac, L1, L2)
    w_rob = _winner_at(ALPHA_MAIN, dm_rob, p_rob, L1, L2)

    variant_agreement = (w_std == w_hac == w_rob)

    return {
        "country":                country,
        "form":                   form,
        "h":                      h,
        "pair":                   pair,
        "layer_1":                L1,
        "layer_2":                L2,
        "n_paired":               n,
        "d_mean_squared":         float(d_sq.mean()),
        "d_mean_absolute":        float(d_ab.mean()),
        "dm_standard":            float(dm_std) if np.isfinite(dm_std) else np.nan,
        "p_standard":             float(p_std)  if np.isfinite(p_std)  else np.nan,
        "dm_hac":                 float(dm_hac) if np.isfinite(dm_hac) else np.nan,
        "p_hac":                  float(p_hac)  if np.isfinite(p_hac)  else np.nan,
        "dm_robust":              float(dm_rob) if np.isfinite(dm_rob) else np.nan,
        "p_robust":               float(p_rob)  if np.isfinite(p_rob)  else np.nan,
        "winner_standard_5pct":   w_std,
        "winner_hac_5pct":        w_hac,
        "winner_robust_5pct":     w_rob,
        "sig_standard_1pct":      bool(np.isfinite(p_std) and p_std < ALPHA_STRICT),
        "sig_standard_5pct":      bool(np.isfinite(p_std) and p_std < ALPHA_MAIN),
        "sig_hac_5pct":           bool(np.isfinite(p_hac) and p_hac < ALPHA_MAIN),
        "sig_robust_5pct":        bool(np.isfinite(p_rob) and p_rob < ALPHA_MAIN),
        "variant_agreement_5pct": bool(variant_agreement),
        "beta_option":            bool(cell["beta_option"]),
        "dm_scope":               str(cell["dm_scope"]),
        "decision_linkage":       _decision_linkage(country, form, h, pair),
    }


# ──────────────────────────────────────────────────────────────────────
# Summary aggregation — by pair, by country, by horizon, overall
# ──────────────────────────────────────────────────────────────────────

def _aggregate_group(df: pd.DataFrame, group_type: str, group_value: str) -> dict[str, Any]:
    """Produce a single summary row for a group of cells."""
    n = len(df)
    winner_counts: dict[str, int] = df["winner_standard_5pct"].value_counts().to_dict()
    return {
        "grouping_type":            group_type,
        "grouping_value":           group_value,
        "n_cells":                  n,
        "n_sig_standard_1pct":      int(df["sig_standard_1pct"].sum()),
        "n_sig_standard_5pct":      int(df["sig_standard_5pct"].sum()),
        "n_sig_hac_5pct":           int(df["sig_hac_5pct"].sum()),
        "n_sig_robust_5pct":        int(df["sig_robust_5pct"].sum()),
        "arima_wins":               int(winner_counts.get("ARIMA", 0)),
        "var_wins":                 int(winner_counts.get("VAR", 0)),
        "ridge_wins":               int(winner_counts.get("Ridge", 0)),
        "ties":                     int(winner_counts.get("tie", 0)),
        "undefined":                int(winner_counts.get("undefined", 0)),
        "variant_agreement_rate":   float(df["variant_agreement_5pct"].mean()) if n > 0 else float("nan"),
    }


def build_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Per-pair
    for pair in sorted(matrix["pair"].unique()):
        rows.append(_aggregate_group(matrix[matrix["pair"] == pair], "pair", pair))

    # Per-country
    for country in sorted(matrix["country"].unique()):
        rows.append(_aggregate_group(matrix[matrix["country"] == country], "country", country))

    # Per-horizon
    for h in sorted(matrix["h"].unique()):
        rows.append(_aggregate_group(matrix[matrix["h"] == h], "horizon", str(int(h))))

    # β-option scope vs dual-form
    rows.append(_aggregate_group(matrix[matrix["beta_option"]], "scope", "beta_option"))
    rows.append(_aggregate_group(matrix[~matrix["beta_option"]], "scope", "usa_dual_form"))

    # Overall
    rows.append(_aggregate_group(matrix, "overall", "all"))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

SEP = "=" * 76
SUB = "-" * 76


def main() -> int:
    print(SEP)
    print("Phase 7 S2 — Diebold-Mariano Battery (standard + HAC + robust)")
    print(SEP)
    print(f"src version : {SRC_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")

    # ── Step 1: Load S1 outputs ────────────────────────────────────
    print()
    print(SUB)
    print("Step 1 · Load S1 outputs")
    print(SUB)
    try:
        unified  = pd.read_csv(IN_UNIFIED, parse_dates=["origin_date", "target_date"])
        coverage = pd.read_csv(IN_COVERAGE)
    except FileNotFoundError as e:
        print(f"  [FATAL] Could not read S1 outputs: {e}")
        print(f"  Run scripts/phase7_s1_forecast_integration.py first.")
        return 2

    print(f"  unified  shape: {unified.shape}")
    print(f"  coverage shape: {coverage.shape}")

    ok_cells = coverage[coverage["status"] == "OK"].reset_index(drop=True)
    print(f"  Iterable cells (status=OK): {len(ok_cells)}")

    # ── Step 2: Run DM battery ─────────────────────────────────────
    print()
    print(SUB)
    print("Step 2 · Run DM battery (standard + HAC + robust)")
    print(SUB)
    results: list[dict[str, Any]] = []
    for _, cell in ok_cells.iterrows():
        try:
            row = run_dm_for_cell(cell, unified)
            results.append(row)
        except Exception as e:
            print(f"  [ERROR] cell {cell['country']}/{cell['form']}/h={cell['h']}/{cell['pair']}: "
                  f"{type(e).__name__}: {e}")

    matrix = pd.DataFrame(results)
    matrix.to_csv(OUT_MATRIX, index=False)
    print(f"  Wrote {OUT_MATRIX.name}: {len(matrix)} cells × {len(matrix.columns)} cols")

    # Flag any NaN DM stat cells
    n_nan = int(matrix[["dm_standard", "dm_hac", "dm_robust"]].isna().any(axis=1).sum())
    if n_nan > 0:
        print(f"  [WARN] {n_nan} cell(s) produced NaN DM stat. See matrix for details.")

    # ── Step 3: Build summary ──────────────────────────────────────
    print()
    print(SUB)
    print("Step 3 · Aggregate summary")
    print(SUB)
    summary = build_summary(matrix)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"  Wrote {OUT_SUMMARY.name}: {len(summary)} grouping rows × {len(summary.columns)} cols")

    # ── Step 4: Decision-gate diagnostic (pair view) ───────────────
    print()
    print(SUB)
    print("Step 4 · Per-pair aggregate (decision-gate input)")
    print(SUB)
    pair_view = summary[summary["grouping_type"] == "pair"][
        ["grouping_value", "n_cells", "n_sig_standard_5pct", "n_sig_standard_1pct",
         "arima_wins", "var_wins", "ridge_wins", "ties", "variant_agreement_rate"]
    ].rename(columns={"grouping_value": "pair"})
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(pair_view.to_string(index=False))

    # ── Step 5: Per-country aggregate ──────────────────────────────
    print()
    print(SUB)
    print("Step 5 · Per-country aggregate")
    print(SUB)
    country_view = summary[summary["grouping_type"] == "country"][
        ["grouping_value", "n_cells", "n_sig_standard_5pct",
         "arima_wins", "var_wins", "ridge_wins", "ties", "variant_agreement_rate"]
    ].rename(columns={"grouping_value": "country"})
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(country_view.to_string(index=False))

    # ── Step 6: Per-horizon aggregate ──────────────────────────────
    print()
    print(SUB)
    print("Step 6 · Per-horizon aggregate")
    print(SUB)
    horizon_view = summary[summary["grouping_type"] == "horizon"][
        ["grouping_value", "n_cells", "n_sig_standard_5pct",
         "arima_wins", "var_wins", "ridge_wins", "ties", "variant_agreement_rate"]
    ].rename(columns={"grouping_value": "h"})
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(horizon_view.to_string(index=False))

    # ── Step 7: Full 25-cell matrix (compact view) ─────────────────
    print()
    print(SUB)
    print("Step 7 · Full 25-cell matrix (standard DM primary)")
    print(SUB)
    compact = matrix[[
        "country", "form", "h", "pair", "n_paired",
        "dm_standard", "p_standard",
        "dm_hac", "p_hac",
        "dm_robust", "p_robust",
        "winner_standard_5pct", "variant_agreement_5pct",
    ]].copy()
    # Round the numerics for readability
    for c in ("dm_standard", "dm_hac", "dm_robust"):
        compact[c] = compact[c].round(3)
    for c in ("p_standard", "p_hac", "p_robust"):
        compact[c] = compact[c].round(4)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(compact.to_string(index=False))

    # ── Step 8: Decision-gate readout ──────────────────────────────
    print()
    print(SUB)
    print("Step 8 · Post-S2 decision-gate signals")
    print(SUB)
    n_total       = len(matrix)
    n_sig_5       = int(matrix["sig_standard_5pct"].sum())
    n_sig_1       = int(matrix["sig_standard_1pct"].sum())
    n_agree       = int(matrix["variant_agreement_5pct"].sum())
    disagreement  = matrix[~matrix["variant_agreement_5pct"]]
    marginal      = matrix[
        (matrix["p_standard"] >= ALPHA_STRICT) &
        (matrix["p_standard"] <  0.10)
    ]
    hac_std_diff  = matrix[(matrix["winner_hac_5pct"] != matrix["winner_standard_5pct"])]
    rob_std_diff  = matrix[(matrix["winner_robust_5pct"] != matrix["winner_standard_5pct"])]

    print(f"  Standard DM: {n_sig_5}/{n_total} significant at α=0.05, "
          f"{n_sig_1}/{n_total} at α=0.01")
    print(f"  3-variant agreement: {n_agree}/{n_total} cells agree on winner at α=0.05")
    print(f"  Marginal cells (p_standard ∈ [0.01, 0.10)): {len(marginal)}")
    print(f"  HAC-vs-standard winner flips at α=0.05: {len(hac_std_diff)}")
    print(f"  Robust-vs-standard winner flips at α=0.05: {len(rob_std_diff)}")

    if len(disagreement) > 0:
        print()
        print("  Variant-disagreement cells:")
        for _, r in disagreement.iterrows():
            print(f"    {r['country']:8s} {r['form']:10s} h={r['h']:<2d} {r['pair']:13s}  "
                  f"std={r['winner_standard_5pct']:7s} hac={r['winner_hac_5pct']:7s} "
                  f"rob={r['winner_robust_5pct']}")

    print()
    print("  Decision-gate interpretation (suggested):")
    if n_sig_1 / n_total >= 0.80:
        print("    (a) ≥80% of cells significant at α=0.01 → S2b HAC sensitivity can be skipped")
    elif len(hac_std_diff) == 0:
        print("    (a) HAC agrees with standard on all winners → S2b can be skipped")
    else:
        print("    (a) HAC sensitivity required: ≥1 winner flip under HAC")

    if len(rob_std_diff) == 0:
        print("    (b) Robust agrees with standard on all winners → S4 COVID-origin sensitivity can be skipped")
    else:
        print(f"    (b) Robust-vs-standard flips in {len(rob_std_diff)} cell(s) → S4 COVID-origin sensitivity recommended")

    if len(marginal) >= 3:
        print(f"    (c) {len(marginal)} marginal cells → inspect individually before D-078")
    else:
        print(f"    (c) {len(marginal)} marginal cells — low sensitivity concern")

    # ── Exit verdict ───────────────────────────────────────────────
    print()
    print(SEP)
    if n_nan > 0:
        print(f"S2 AMBER — {n_nan} cell(s) with NaN DM stat. Investigate before writing D-078.")
        print(SEP)
        return 1
    print(f"S2 GREEN — DM battery complete. {n_sig_5}/{n_total} cells significant at α=0.05.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
