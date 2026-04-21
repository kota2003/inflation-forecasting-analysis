#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S2b — Japan α Grid Extension (Boundary Sensitivity)

Follows D-048 Step 1 staged-grid pattern. S2 identified Japan primary as
the sole boundary-hit (α* = 1000 = upper edge of logspace(-3, 3, 13) grid
with monotonically-decreasing val_MSE). S2b extends the grid upward for
Japan only, leaving the 4 interior-α combinations untouched.

Interpretation prior: Japan's N3 isolation (sextuple-confirmed via ACF /
ARIMA / VAR lag / Granger / IRF / FEVD) predicts that Ridge should prefer
α → ∞ (intercept-only model) as an independent Ridge-lens confirmation.
S2b verifies whether the minimum is interior to the extended grid or
truly monotonically approaches the intercept-only theoretical bound
(val_MSE → train_var(y) = 0.29² ≈ 0.085).

Pending-formal decision (to be entered alongside D-065 as D-065a or
inside D-065):
  - Japan α grid extended to np.logspace(3.0, 6.0, 7) — JPN only

Usage:
    python scripts/phase6_step3_s2b_japan_grid_extension.py

Outputs (under data/documentation/):
    phase6_step3_s2b_japan_cv_scores.csv         (35 rows: 7 α × 5 folds)
    phase6_step3_s2b_japan_alpha_selection.csv   (1 row, JPN only)
"""
from __future__ import annotations

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

from src import build_country_features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOC_DIR = PROJECT_ROOT / "data" / "documentation"
DOC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2019-12-01")

# Extended α grid for Japan: log10 ∈ {3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}
ALPHA_GRID_EXT = np.logspace(3.0, 6.0, 7)
N_SPLITS = 5
RANDOM_STATE = 0

COUNTRY = "JAPAN"
FORM = "primary"
TARGET = f"{COUNTRY}_CPI"


# ---------------------------------------------------------------------------
# CV routine (mirrors S2's but scoped to one combination)
# ---------------------------------------------------------------------------
def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df_clean = df.dropna()
    df_train = df_clean.loc[df_clean.index <= TRAIN_END]
    y = df_train[TARGET]
    X = df_train.drop(columns=[TARGET])
    return X, y


def run_ext_cv(
    X: pd.DataFrame,
    y: pd.Series,
    alpha_grid: np.ndarray,
    n_splits: int,
) -> Tuple[List[Dict], Dict]:
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
                "country":      COUNTRY,
                "form":         FORM,
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

    alpha_means = {a: float(np.mean(alpha_val_mse[a])) for a in alpha_grid}
    alpha_stds  = {a: float(np.std(alpha_val_mse[a], ddof=1)) for a in alpha_grid}
    best_alpha = min(alpha_means, key=alpha_means.get)

    grid_min, grid_max = float(alpha_grid[0]), float(alpha_grid[-1])
    if np.isclose(best_alpha, grid_min):
        boundary = "lower"
    elif np.isclose(best_alpha, grid_max):
        boundary = "upper"
    else:
        boundary = "interior"

    summary = {
        "country":              COUNTRY,
        "form":                 FORM,
        "n_train_total":        int(len(X)),
        "n_features":           int(X.shape[1]),
        "n_splits":             int(n_splits),
        "grid_min":             grid_min,
        "grid_max":             grid_max,
        "grid_n":               int(len(alpha_grid)),
        "selected_alpha":       float(best_alpha),
        "selected_log10_alpha": float(np.log10(best_alpha)),
        "cv_val_mse_mean":      alpha_means[best_alpha],
        "cv_val_mse_std":       alpha_stds[best_alpha],
        "boundary_status":      boundary,
    }
    return cv_rows, summary


# ---------------------------------------------------------------------------
# Theoretical intercept-only bound
# ---------------------------------------------------------------------------
def compute_intercept_bound(X: pd.DataFrame, y: pd.Series, n_splits: int) -> float:
    """
    Across the same TimeSeriesSplit folds, compute the val_MSE of an
    intercept-only predictor (ŷ = mean(y_train)). This is the theoretical
    upper-bound Ridge α → ∞ will converge to.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    vals: List[float] = []
    for tr_idx, va_idx in splitter.split(X):
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        yhat = np.full_like(y_va.values, fill_value=y_tr.mean(), dtype=float)
        vals.append(mean_squared_error(y_va, yhat))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S2b — Japan α Grid Extension")
    print("=" * 72)
    print(f"Extended grid (log10): {np.log10(ALPHA_GRID_EXT).round(2).tolist()}")
    print(f"n_splits: {N_SPLITS}")
    print(f"Train window: .. {TRAIN_END.date()}")

    # ---- 1. Japan features + X/y
    print("\n[1/3] Building Japan feature matrix...")
    df_jpn = build_country_features(COUNTRY)
    X, y = split_xy(df_jpn)
    print(
        f"    JPN shape = {df_jpn.shape}  "
        f"X_train = {X.shape}  y_train n = {len(y)}"
    )
    print(
        f"    y_train mean = {y.mean():.4f}  std = {y.std(ddof=1):.4f}  "
        f"var = {y.var(ddof=1):.4f}"
    )

    # ---- 2. Intercept-only theoretical bound (across matched folds)
    intercept_bound = compute_intercept_bound(X, y, N_SPLITS)
    print(f"    Intercept-only val_MSE (theoretical α→∞ limit) = {intercept_bound:.4f}")

    # ---- 3. Extended CV
    print("\n[2/3] Running extended α CV on Japan primary...")
    cv_rows, summary = run_ext_cv(X, y, ALPHA_GRID_EXT, N_SPLITS)

    # ---- 4. Write CSVs
    print("\n[3/3] Writing audit CSVs...")
    df_cv = pd.DataFrame(cv_rows)
    df_sel = pd.DataFrame([summary])
    out_cv = DOC_DIR / "phase6_step3_s2b_japan_cv_scores.csv"
    out_sel = DOC_DIR / "phase6_step3_s2b_japan_alpha_selection.csv"
    df_cv.to_csv(out_cv, index=False)
    df_sel.to_csv(out_sel, index=False)
    print(f"    -> {out_cv.relative_to(PROJECT_ROOT)}  ({len(df_cv)} rows)")
    print(f"    -> {out_sel.relative_to(PROJECT_ROOT)}  ({len(df_sel)} rows)")

    # ---- 5. α path preview
    print("\n" + "=" * 72)
    print("JAPAN α PATH (extended grid, mean val_MSE per fold)")
    print("=" * 72)
    pivot = (
        df_cv.groupby("alpha")["val_mse"].mean().reset_index()
        .rename(columns={"val_mse": "val_MSE_mean"})
    )
    pivot["log10_alpha"] = np.log10(pivot["alpha"]).round(2)
    pivot["val_MSE_mean"] = pivot["val_MSE_mean"].round(4)
    pivot["gap_to_intercept_bound"] = (pivot["val_MSE_mean"] - intercept_bound).round(4)
    pivot = pivot[["log10_alpha", "alpha", "val_MSE_mean", "gap_to_intercept_bound"]]
    print(pivot.to_string(index=False))
    print(f"\n    Intercept-only theoretical bound = {intercept_bound:.4f}")

    # ---- 6. Selection verdict
    print("\n" + "=" * 72)
    print("JAPAN SELECTION VERDICT")
    print("=" * 72)
    print(f"    selected_alpha        = {summary['selected_alpha']:.4g}")
    print(f"    selected_log10_alpha  = {summary['selected_log10_alpha']:+.2f}")
    print(f"    val_MSE               = {summary['cv_val_mse_mean']:.4f}")
    print(f"    intercept_bound       = {intercept_bound:.4f}")
    print(
        f"    gap                   = "
        f"{summary['cv_val_mse_mean'] - intercept_bound:+.4f}"
    )
    print(f"    boundary_status       = {summary['boundary_status']}")

    if summary["boundary_status"] == "upper":
        print("\n    >>> UPPER BOUNDARY STILL HIT — extend further or accept")
        print("        α → ∞ (intercept-only dominance) as verdict.")
    else:
        print("\n    >>> INTERIOR saturation confirmed. N3 Ridge-lens result recorded.")

    print("\nS2b complete. Proceed to S3 (coefficient stability) after review.")


if __name__ == "__main__":
    main()
