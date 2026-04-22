"""
scripts/phase7_s1_forecast_integration.py

Phase 7 S1 — Forecast Integration
=================================

Reads the three Phase 6 OOS forecast artefacts via
``src.evaluation.load_phase6_forecasts`` (v0.4.3), stacks them into a
single long-format unified panel, and emits a DM-cell-centric coverage
matrix that enumerates every paired Diebold-Mariano test the Phase 7
scope intends to run.

Inputs
------
    data/documentation/phase6_step1_arima_forecast.csv
    data/documentation/phase6_step2_s6_var_oos_forecasts.csv
    data/documentation/phase6_step3_s4_ridge_oos_forecasts.csv

Outputs
-------
    data/documentation/phase7_s1_unified_forecasts.csv
        Long-format stack of all three layers with an added ``layer``
        column. Columns: layer, country, form, h, origin_date,
        target_date, y_true, y_pred. Row count ≈ 2,312 (336 ARIMA +
        872 VAR CPI rows + 1,104 Ridge).

    data/documentation/phase7_s1_coverage_matrix.csv
        One row per planned DM test cell (25 rows). Columns: country,
        form, h, pair, layer_1, layer_2, n_layer_1, n_layer_2,
        n_paired, status, beta_option, dm_scope.

Scope enumeration
-----------------
  * 24 β-option primary-form DM cells (committed at Phase 7 Q#4):
      - 12 three-way cells at h=1 (ARIMA-VAR, ARIMA-Ridge, VAR-Ridge
        for each of USA / JAPAN / UK / GERMANY)
      - 12 VAR-Ridge cells at h ∈ {3, 6, 12} for each country
  *  1 D-071 USA secondary-form DM cell (ARIMA-Ridge at h=1)

Exit codes
----------
    0   all β-option cells + D-071 cell have status='OK' (n≥30)
    1   at least one cell flagged UNDERPOWERED or MISSING
    2   an adapter load failed

Decision linkage
----------------
D-068 (Ridge origins matched to VAR S6), D-071 (USA dual-form),
D-075 (split-promotion), D-076 (src/evaluation.py materialisation),
D-077 (this script's decision record, appended to ProjectDriven.md
after a clean green exit).
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
    """Walk up from the cwd until a directory containing both `data`
    and `src` is found. Mirrors the convention used in notebooks
    02–09 and the Phase 6 scratch scripts."""
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "data").is_dir() and (cand / "src").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {Path.cwd()}")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    MAIN_COUNTRIES,
    HORIZONS_PHASE7,
    UNIFIED_SCHEMA_COLUMNS,
    load_phase6_forecasts,
    align_matched_terms,
    __version__ as SRC_VERSION,
)

DOC_DIR        = PROJECT_ROOT / "data" / "documentation"
OUT_UNIFIED    = DOC_DIR / "phase7_s1_unified_forecasts.csv"
OUT_COVERAGE   = DOC_DIR / "phase7_s1_coverage_matrix.csv"

UNDERPOWERED_THRESHOLD = 30  # n<30 → DM asymptotic approximation weak


# ──────────────────────────────────────────────────────────────────────
# DM cell manifest — enumerate every paired test Phase 7 intends to run
# ──────────────────────────────────────────────────────────────────────

def build_dm_cell_manifest() -> list[dict[str, Any]]:
    """Enumerate the 25 DM test cells in scope for Phase 7.

    Returns a list of dicts in a stable order (country outer, then
    pair, then horizon), suitable for conversion to DataFrame.
    """
    cells: list[dict[str, Any]] = []

    # β-option primary-form scope (24 cells)
    for country in MAIN_COUNTRIES:
        # h=1 three-way: each pair of {ARIMA, VAR, Ridge}
        for (L1, L2) in [("ARIMA", "VAR"), ("ARIMA", "Ridge"), ("VAR", "Ridge")]:
            cells.append({
                "country":     country,
                "form":        "primary",
                "h":           1,
                "pair":        f"{L1}-{L2}",
                "layer_1":     L1,
                "layer_2":     L2,
                "beta_option": True,
                "dm_scope":    "primary_h1_3way",
            })
        # h ∈ {3, 6, 12} VAR-vs-Ridge only (ARIMA is h=1 only per D-048)
        for h in (3, 6, 12):
            cells.append({
                "country":     country,
                "form":        "primary",
                "h":           h,
                "pair":        "VAR-Ridge",
                "layer_1":     "VAR",
                "layer_2":     "Ridge",
                "beta_option": True,
                "dm_scope":    "primary_hmulti_vr",
            })

    # D-071 USA dual-form (ARIMA-Ridge at h=1; VAR has no secondary form)
    cells.append({
        "country":     "USA",
        "form":        "secondary",
        "h":           1,
        "pair":        "ARIMA-Ridge",
        "layer_1":     "ARIMA",
        "layer_2":     "Ridge",
        "beta_option": False,
        "dm_scope":    "usa_dual_form",
    })

    return cells


# ──────────────────────────────────────────────────────────────────────
# Coverage resolution — ask align_matched_terms how many rows pair up
# ──────────────────────────────────────────────────────────────────────

def resolve_cell_coverage(
    cell: dict[str, Any],
    layer_map: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Add n_layer_1, n_layer_2, n_paired, status to a cell dict."""
    d1 = layer_map[cell["layer_1"]]
    d2 = layer_map[cell["layer_2"]]

    mask_1 = (d1["country"] == cell["country"]) & (d1["form"] == cell["form"]) & (d1["h"] == cell["h"])
    mask_2 = (d2["country"] == cell["country"]) & (d2["form"] == cell["form"]) & (d2["h"] == cell["h"])
    d1_cell = d1[mask_1]
    d2_cell = d2[mask_2]

    n_layer_1 = int(len(d1_cell))
    n_layer_2 = int(len(d2_cell))

    if n_layer_1 == 0 or n_layer_2 == 0:
        n_paired = 0
        status   = "MISSING"
    else:
        try:
            y, _, _ = align_matched_terms(d1_cell, d2_cell)
            n_paired = int(len(y))
            status   = "OK" if n_paired >= UNDERPOWERED_THRESHOLD else "UNDERPOWERED"
        except ValueError as e:
            # Either zero-intersection on target_date, or y_true
            # tolerance violation. Both are hard-fail conditions for
            # this cell (DM cannot be run).
            n_paired = 0
            status   = f"ERROR: {str(e)[:80]}"

    out = dict(cell)
    out.update({
        "n_layer_1": n_layer_1,
        "n_layer_2": n_layer_2,
        "n_paired":  n_paired,
        "status":    status,
    })
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

SEP = "=" * 76
SUB = "-" * 76


def main() -> int:
    print(SEP)
    print("Phase 7 S1 — Forecast Integration")
    print(SEP)
    print(f"src version : {SRC_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Doc dir     : {DOC_DIR}")

    # ── Step 1: Load three layers via unified adapters ────────────
    print()
    print(SUB)
    print("Step 1 · Load Phase 6 OOS artefacts via src.evaluation adapters")
    print(SUB)
    try:
        df_arima = load_phase6_forecasts("arima")
        df_var   = load_phase6_forecasts("var")
        df_ridge = load_phase6_forecasts("ridge")
    except Exception as e:
        print(f"  [FATAL] Adapter load failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2

    print(f"  ARIMA : shape={df_arima.shape}, "
          f"countries={sorted(df_arima['country'].unique())}, "
          f"forms={sorted(df_arima['form'].unique())}")
    print(f"  VAR   : shape={df_var.shape}, "
          f"countries={sorted(df_var['country'].unique())}, "
          f"forms={sorted(df_var['form'].unique())}")
    print(f"  Ridge : shape={df_ridge.shape}, "
          f"countries={sorted(df_ridge['country'].unique())}, "
          f"forms={sorted(df_ridge['form'].unique())}")

    # ── Step 2: Concatenate into unified long-format panel ────────
    print()
    print(SUB)
    print("Step 2 · Assemble unified long-format panel")
    print(SUB)
    unified = pd.concat(
        [
            df_arima.assign(layer="ARIMA"),
            df_var  .assign(layer="VAR"),
            df_ridge.assign(layer="Ridge"),
        ],
        ignore_index=True,
    )
    # Desired column order: layer first, then unified schema columns
    unified = unified[["layer", *UNIFIED_SCHEMA_COLUMNS]]

    expected_total = len(df_arima) + len(df_var) + len(df_ridge)
    assert len(unified) == expected_total, (
        f"Concat row-count mismatch: got {len(unified)}, expected {expected_total}"
    )

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUT_UNIFIED, index=False)
    print(f"  Wrote {OUT_UNIFIED.name}: {len(unified)} rows × {len(unified.columns)} cols")
    print(f"  Layer breakdown:")
    for layer, n in unified.groupby("layer").size().items():
        print(f"    {layer:6s}  n={n}")

    # ── Step 3: Build coverage matrix via align_matched_terms ─────
    print()
    print(SUB)
    print("Step 3 · Resolve paired-DM coverage for 25 Phase 7 cells")
    print(SUB)
    layer_map = {"ARIMA": df_arima, "VAR": df_var, "Ridge": df_ridge}
    cells = build_dm_cell_manifest()
    resolved = [resolve_cell_coverage(c, layer_map) for c in cells]
    coverage = pd.DataFrame(resolved)
    coverage = coverage[[
        "country", "form", "h", "pair",
        "layer_1", "layer_2",
        "n_layer_1", "n_layer_2", "n_paired",
        "status", "beta_option", "dm_scope",
    ]]
    coverage.to_csv(OUT_COVERAGE, index=False)
    print(f"  Wrote {OUT_COVERAGE.name}: {len(coverage)} DM cells × {len(coverage.columns)} cols")

    # ── Step 4: Diagnostic summary ────────────────────────────────
    print()
    print(SUB)
    print("Step 4 · β-option 24-test scope verdict")
    print(SUB)
    beta = coverage[coverage["beta_option"]]
    n_beta_ok        = int((beta["status"] == "OK").sum())
    n_beta_under     = int((beta["status"] == "UNDERPOWERED").sum())
    n_beta_missing   = int((beta["status"] == "MISSING").sum())
    n_beta_error     = int(beta["status"].str.startswith("ERROR").sum())
    print(f"  OK           : {n_beta_ok}/{len(beta)}")
    print(f"  UNDERPOWERED : {n_beta_under}")
    print(f"  MISSING      : {n_beta_missing}")
    print(f"  ERROR        : {n_beta_error}")
    if n_beta_ok < len(beta):
        print()
        print("  Flagged β-option cells:")
        flagged = beta[beta["status"] != "OK"]
        for _, r in flagged.iterrows():
            print(f"    {r['country']:8s} {r['form']:10s} h={r['h']:<2d} "
                  f"{r['pair']:13s}  n={r['n_paired']:3d}  {r['status']}")

    print()
    print(SUB)
    print("Step 5 · D-071 USA dual-form cell verdict")
    print(SUB)
    d071 = coverage[coverage["dm_scope"] == "usa_dual_form"]
    for _, r in d071.iterrows():
        print(f"  {r['country']:8s} {r['form']:10s} h={r['h']:<2d} "
              f"{r['pair']:13s}  n_layer_1={r['n_layer_1']:3d}  "
              f"n_layer_2={r['n_layer_2']:3d}  n_paired={r['n_paired']:3d}  "
              f"status={r['status']}")

    # ── Step 6: Full coverage matrix table ────────────────────────
    print()
    print(SUB)
    print("Step 6 · Full coverage matrix")
    print(SUB)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 160,
    ):
        print(coverage.to_string(index=False))

    # ── Exit verdict ──────────────────────────────────────────────
    print()
    print(SEP)
    all_ok = (coverage["status"] == "OK").all()
    beta_all_ok = (beta["status"] == "OK").all()
    d071_all_ok = (d071["status"] == "OK").all()
    if all_ok:
        print("S1 GREEN — all 25 DM cells ready for S2 battery execution.")
        print(SEP)
        return 0
    if beta_all_ok and d071_all_ok:
        print("S1 GREEN — β-option 24 cells + D-071 dual-form cell all OK.")
        print(SEP)
        return 0
    print("S1 AMBER — see flagged cells above; S2 scope revision may be needed.")
    print(SEP)
    return 1


if __name__ == "__main__":
    sys.exit(main())
