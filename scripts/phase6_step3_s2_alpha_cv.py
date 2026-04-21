#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S2 — α Grid Sweep + Walk-Forward CV for Ridge Layer 3

For each (country, form) combination, sweeps α ∈ np.logspace(-3, 3, 13)
and evaluates each α via TimeSeriesSplit(n_splits=5) on the train window
(≤ 2019-12 per D-005). StandardScaler + Ridge wrapped in a Pipeline,
with scaler fitted train-only per fold (leakage guard).

Pending-formal decisions referenced (to be entered as D-065+):
  - D3 (CV): sklearn TimeSeriesSplit expanding-window, n_splits=5;
    train-only CV (2020-01+ is held-out for S4 OOS forecast)
  - D4 (α grid): logspace(-3, 3, 13); boundary-hit flag triggers
    D-048-style sensitivity extension at S2b if observed
  - D5 (standardization): train-only StandardScaler per fold via Pipeline

Usage:
    python scripts/phase6_step3_s2_alpha_cv.py

Outputs (under data/documentation/):
    phase6_step3_s2_cv_scores.csv        (~325 rows: 5 (c,f) × 13 α × 5 folds)
    phase6_step3_s2_alpha_selection.csv  (5 rows: 1 per (country, form))
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import (
    MAIN_COUNTRIES,
    build_all_features,
    build_country_features,
)
import src.feature_engineering as fe_module


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOC_DIR = PROJECT_ROOT / "data" / "documentation"
DOC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2019-12-01")
TEST_START = pd.Timestamp("2020-01-01")

ALPHA_GRID = np.logspace(-3, 3, 13)  # 0.001 .. 1000, 13 points
N_SPLITS = 5
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def target_col_name(country: str) -> str:
    return f"{country}_CPI"


def build_usa_first_diff_features() -> pd.DataFrame:
    """USA feature matrix with CPI forced to first_diff secondary form."""
    key = ("USA", "CPI")
    had_override = key in fe_module.REGISTRY_OVERRIDES
    original_value = fe_module.REGISTRY_OVERRIDES.get(key)
    try:
        fe_module.REGISTRY_OVERRIDES[key] = "first_diff"
        out = build_country_features("USA")
    finally:
        if had_override:
            fe_module.REGISTRY_OVERRIDES[key] = original_value
        else:
            fe_module.REGISTRY_OVERRIDES.pop(key, None)
    return out


def split_xy(df: pd.DataFrame, country: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop NaN, restrict to train window (≤ TRAIN_END), split into X / y."""
    target = target_col_name(country)
    df_clean = df.dropna()
    df_train = df_clean.loc[df_clean.index <= TRAIN_END]
    y = df_train[target]
    X = df_train.drop(columns=[target])
    return X, y


# ---------------------------------------------------------------------------
# Core CV routine
# ---------------------------------------------------------------------------
def run_cv_for_combination(
    country: str,
    form: str,
    df: pd.DataFrame,
    alpha_grid: np.ndarray,
    n_splits: int,
) -> Tuple[List[Dict], Dict]:
    """
    Run walk-forward CV across α for one (country, form) combination.

    Returns
    -------
    (cv_rows, summary)
        cv_rows    : list of fold-level dicts (one per α × fold)
        summary    : dict for the alpha_selection CSV
    """
    X, y = split_xy(df, country)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    alpha_val_mse:   Dict[float, List[float]] = {a: [] for a in alpha_grid}
    alpha_train_mse: Dict[float, List[float]] = {a: [] for a in alpha_grid}
    cv_rows: List[Dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        for alpha in alpha_grid:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
            ])
            pipe.fit(X_tr, y_tr)
            tr_mse = mean_squared_error(y_tr, pipe.predict(X_tr))
            va_mse = mean_squared_error(y_va, pipe.predict(X_va))

            alpha_train_mse[alpha].append(tr_mse)
            alpha_val_mse[alpha].append(va_mse)

            cv_rows.append({
                "country":      country,
                "form":         form,
                "alpha":        float(alpha),
                "log10_alpha":  float(np.log10(alpha)),
                "fold":         fold_idx,
                "train_start":  X_tr.index.min().strftime("%Y-%m-%d"),
                "train_end":    X_tr.index.max().strftime("%Y-%m-%d"),
                "val_start":    X_va.index.min().strftime("%Y-%m-%d"),
                "val_end":      X_va.index.max().strftime("%Y-%m-%d"),
                "n_train":      int(len(X_tr)),
                "n_val":        int(len(X_va)),
                "train_mse":    float(tr_mse),
                "val_mse":      float(va_mse),
            })

    # α selection: argmin mean val_mse
    alpha_means = {a: float(np.mean(alpha_val_mse[a])) for a in alpha_grid}
    alpha_stds  = {a: float(np.std(alpha_val_mse[a], ddof=1)) for a in alpha_grid}
    best_alpha = min(alpha_means, key=alpha_means.get)

    # 1-SE rule alternative (largest α within 1 SE of best mean val_mse)
    best_mean = alpha_means[best_alpha]
    best_std  = alpha_stds[best_alpha]
    threshold = best_mean + (best_std / np.sqrt(n_splits))
    within = [a for a in alpha_grid if alpha_means[a] <= threshold]
    alpha_1se = max(within) if within else best_alpha

    grid_min, grid_max = float(alpha_grid[0]), float(alpha_grid[-1])
    if np.isclose(best_alpha, grid_min):
        boundary = "lower"
    elif np.isclose(best_alpha, grid_max):
        boundary = "upper"
    else:
        boundary = "interior"

    summary = {
        "country":                country,
        "form":                   form,
        "n_train_total":          int(len(X)),
        "n_features":             int(X.shape[1]),
        "n_splits":               int(n_splits),
        "grid_min":               grid_min,
        "grid_max":               grid_max,
        "grid_n":                 int(len(alpha_grid)),
        "selected_alpha":         float(best_alpha),
        "selected_log10_alpha":   float(np.log10(best_alpha)),
        "cv_val_mse_mean":        best_mean,
        "cv_val_mse_std":         best_std,
        "alpha_1se":              float(alpha_1se),
        "log10_alpha_1se":        float(np.log10(alpha_1se)),
        "boundary_status":        boundary,
    }
    return cv_rows, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S2 — α Grid + Walk-Forward CV for Ridge")
    print("=" * 72)
    print(f"α grid (log10): {np.log10(ALPHA_GRID).round(2).tolist()}")
    print(f"n_splits:       {N_SPLITS}")
    print(
        f"Train window:   .. {TRAIN_END.date()}  "
        f"(test held out from {TEST_START.date()})"
    )

    # ---- 1. Build matrices
    print("\n[1/3] Building feature matrices...")
    primary = build_all_features()
    usa_fd = build_usa_first_diff_features()
    matrices = [(c, "primary", primary[c]) for c in MAIN_COUNTRIES]
    matrices.append(("USA", "first_diff_secondary", usa_fd))

    # ---- 2. Run CV per (country, form)
    print("\n[2/3] Running α grid × CV per (country, form)...")
    all_cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for country, form, df in matrices:
        print(f"    {country:<8} {form:<22}", end=" ", flush=True)
        cv_rows, summary = run_cv_for_combination(
            country, form, df, ALPHA_GRID, N_SPLITS
        )
        all_cv_rows.extend(cv_rows)
        summary_rows.append(summary)
        print(
            f"α* = {summary['selected_alpha']:>8.4f}  "
            f"(log10 = {summary['selected_log10_alpha']:>+5.2f})  "
            f"val_MSE = {summary['cv_val_mse_mean']:>9.4f}  "
            f"[{summary['boundary_status']}]"
        )

    # ---- 3. Write CSVs
    print("\n[3/3] Writing audit CSVs...")
    df_cv = pd.DataFrame(all_cv_rows)
    df_sel = pd.DataFrame(summary_rows)
    out_cv = DOC_DIR / "phase6_step3_s2_cv_scores.csv"
    out_sel = DOC_DIR / "phase6_step3_s2_alpha_selection.csv"
    df_cv.to_csv(out_cv, index=False)
    df_sel.to_csv(out_sel, index=False)
    print(f"    -> {out_cv.relative_to(PROJECT_ROOT)}  ({len(df_cv)} rows)")
    print(f"    -> {out_sel.relative_to(PROJECT_ROOT)}  ({len(df_sel)} rows)")

    # ---- 4. α selection preview
    print("\n" + "=" * 72)
    print("α SELECTION SUMMARY")
    print("=" * 72)
    print_df = df_sel.copy()
    for c in ["selected_alpha", "alpha_1se", "cv_val_mse_mean", "cv_val_mse_std"]:
        print_df[c] = print_df[c].round(4)
    for c in ["selected_log10_alpha", "log10_alpha_1se"]:
        print_df[c] = print_df[c].round(2)
    cols_show = [
        "country", "form", "n_train_total", "n_features",
        "selected_alpha", "selected_log10_alpha",
        "alpha_1se", "log10_alpha_1se",
        "cv_val_mse_mean", "cv_val_mse_std",
        "boundary_status",
    ]
    print(print_df[cols_show].to_string(index=False))

    # ---- 5. Alpha path preview (mean val_MSE per α, per combination)
    print("\n" + "=" * 72)
    print("α PATH (mean val_MSE per log10(α), per combination)")
    print("=" * 72)
    pivot = (
        df_cv.groupby(["country", "form", "alpha"])["val_mse"]
        .mean()
        .unstack(level=["country", "form"])
        .round(4)
    )
    pivot.index = [f"{np.log10(a):>+5.2f}" for a in pivot.index]
    pivot.index.name = "log10(α)"
    print(pivot.to_string())

    # ---- 6. Boundary flag
    boundary_hits = df_sel[df_sel["boundary_status"] != "interior"]
    print("\n" + "=" * 72)
    if len(boundary_hits):
        print("BOUNDARY HIT — α grid extension recommended (S2b) for:")
        print(
            boundary_hits[
                ["country", "form", "selected_alpha", "boundary_status"]
            ].to_string(index=False)
        )
    else:
        print("All α* are INTERIOR to the grid — no S2b extension needed.")
    print("=" * 72)

    print(
        "\nS2 complete. Proceed to S3 "
        "(coefficient stability / feature importance) after review."
    )


if __name__ == "__main__":
    main()
