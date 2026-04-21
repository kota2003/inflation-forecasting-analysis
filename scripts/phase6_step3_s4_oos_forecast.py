#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S4 — OOS Walk-Forward Ridge Forecast for Phase 7 DM

For each (country, form) combination and each horizon h ∈ {1, 3, 6, 12},
perform expanding-window walk-forward direct-h forecasting:

    At each origin t ∈ [2020-01, last_obs - 12m]:
        (a) Form training pairs (X_s, y_{s+h}) for s ≤ t - h
        (b) Fit Ridge(α*) on (X_s, y_{s+h})
        (c) Predict ŷ_{t+h} = β̂_h · X_t
        (d) Record actual y_{t+h}

α* is shared across horizons per combination (sourced from S2 / S2b).
Origin window is restricted so all 4 horizons are evaluable at every
origin — this matches VAR Step 2 S6 methodology for paired Phase 7 DM.

Pending-formal decisions (to be entered D-066+):
  - α* shared across horizons (D-050-analog philosophy)
  - Direct multi-step Ridge (vs recursive); canonical ML convention
  - Origin window 2020-01 .. (last_obs - h_max months)
  - Naive baseline: ŷ_{t+h} = y_t (random walk, Phase 7 DM standard)

Usage:
    python scripts/phase6_step3_s4_oos_forecast.py

Outputs (under data/documentation/):
    phase6_step3_s4_ridge_oos_forecasts.csv     (~1104 rows)
    phase6_step3_s4_ridge_oos_metrics.csv       (20 rows: 5 combos × 4 h)
    phase6_step3_s4_ridge_oos_cpi_summary.csv   (20 rows, compact)
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

TEST_START = pd.Timestamp("2020-01-01")
HORIZONS = (1, 3, 6, 12)
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Helpers (replicate S1 / S3)
# ---------------------------------------------------------------------------
def target_col_name(country: str) -> str:
    return f"{country}_CPI"


def build_usa_first_diff_features() -> pd.DataFrame:
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


def load_selected_alphas() -> Dict[Tuple[str, str], float]:
    """S2 baseline + S2b override."""
    sel_s2 = pd.read_csv(DOC_DIR / "phase6_step3_s2_alpha_selection.csv")
    sel_s2b = pd.read_csv(DOC_DIR / "phase6_step3_s2b_japan_alpha_selection.csv")
    out = {}
    for _, r in sel_s2.iterrows():
        out[(r["country"], r["form"])] = float(r["selected_alpha"])
    for _, r in sel_s2b.iterrows():
        out[(r["country"], r["form"])] = float(r["selected_alpha"])
    return out


def split_full_clean(
    df: pd.DataFrame, country: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """NaN-dropped full (train + test) X and y."""
    target = target_col_name(country)
    df_clean = df.dropna()
    y = df_clean[target]
    X = df_clean.drop(columns=[target])
    return X, y


# ---------------------------------------------------------------------------
# Origin set computation
# ---------------------------------------------------------------------------
def compute_origins(
    index: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    horizons: Tuple[int, ...],
) -> pd.DatetimeIndex:
    """
    Origins where every horizon is evaluable at the same origin.
    Enforces: origin >= test_start and origin + max(h) <= last_observed_date.
    """
    h_max = max(horizons)
    last_date = index[-1]
    last_origin = last_date - pd.DateOffset(months=h_max)
    return index[(index >= test_start) & (index <= last_origin)]


# ---------------------------------------------------------------------------
# Walk-forward Ridge direct-h forecaster
# ---------------------------------------------------------------------------
def walk_forward_direct_h(
    country: str,
    form: str,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    alpha: float,
    horizons: Tuple[int, ...],
    origins: pd.DatetimeIndex,
) -> List[Dict]:
    """
    Return a list of dicts, one per (origin, horizon) forecast event.
    """
    rows: List[Dict] = []

    for h in horizons:
        # Pre-compute the shifted target once per horizon
        z = y_full.shift(-h)  # z_t = y_{t+h}

        for origin in origins:
            # Training pairs: s <= origin - h months
            train_last = origin - pd.DateOffset(months=h)
            X_train_raw = X_full.loc[:train_last]
            z_train_raw = z.loc[:train_last].dropna()
            common = X_train_raw.index.intersection(z_train_raw.index)
            X_train = X_train_raw.loc[common]
            y_train = z_train_raw.loc[common]

            if len(X_train) < 20:
                # Too few training obs — skip, emit NaN forecast
                forecast = np.nan
                n_train = int(len(X_train))
            else:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge",  Ridge(alpha=alpha, random_state=RANDOM_STATE)),
                ])
                pipe.fit(X_train, y_train)
                forecast = float(pipe.predict(X_full.loc[[origin]])[0])
                n_train = int(len(X_train))

            target_date = origin + pd.DateOffset(months=h)
            actual = (
                float(y_full.loc[target_date])
                if target_date in y_full.index else np.nan
            )
            naive = float(y_full.loc[origin])  # y_t as random-walk forecast
            err = actual - forecast if not np.isnan(actual) else np.nan
            naive_err = actual - naive if not np.isnan(actual) else np.nan

            rows.append({
                "country":         country,
                "form":            form,
                "alpha":           float(alpha),
                "origin_date":     origin.strftime("%Y-%m-%d"),
                "target_date":     target_date.strftime("%Y-%m-%d"),
                "horizon":         int(h),
                "n_train_origin":  n_train,
                "actual":          actual,
                "forecast":        forecast,
                "naive_forecast":  naive,
                "error":           err,
                "abs_error":       abs(err) if not np.isnan(err) else np.nan,
                "sq_error":        err * err if not np.isnan(err) else np.nan,
                "naive_error":     naive_err,
                "naive_abs_error": (
                    abs(naive_err) if not np.isnan(naive_err) else np.nan
                ),
                "naive_sq_error": (
                    naive_err * naive_err if not np.isnan(naive_err) else np.nan
                ),
            })

    return rows


# ---------------------------------------------------------------------------
# Metric computation (RMSE / MAE / bias / MedASE / MASE)
# ---------------------------------------------------------------------------
def compute_metrics_for_group(
    fc_df: pd.DataFrame, y_train_full: pd.Series,
) -> Dict:
    """
    fc_df        : rows for one (country, form, horizon) group
    y_train_full : pre-2020 training target series (for MASE denominator)
    """
    valid = fc_df.dropna(subset=["actual", "forecast"])
    n = len(valid)
    if n == 0:
        return {
            "n_origins": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan,
            "med_abs_error": np.nan, "naive_rmse": np.nan, "naive_mae": np.nan,
            "rmse_ratio_vs_naive": np.nan, "mase": np.nan,
        }
    err = valid["error"].values
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    med_abs = float(np.median(np.abs(err)))

    naive_err = valid["naive_error"].values
    naive_rmse = float(np.sqrt(np.mean(naive_err ** 2)))
    naive_mae = float(np.mean(np.abs(naive_err)))
    rmse_ratio = rmse / naive_rmse if naive_rmse > 0 else np.nan

    # MASE denominator: in-sample 1-step naive MAE on training data
    y_train_diff = y_train_full.diff().abs().dropna()
    mase_denom = float(y_train_diff.mean()) if len(y_train_diff) > 0 else np.nan
    mase = mae / mase_denom if mase_denom and mase_denom > 0 else np.nan

    return {
        "n_origins":           n,
        "rmse":                rmse,
        "mae":                 mae,
        "bias":                bias,
        "med_abs_error":       med_abs,
        "naive_rmse":          naive_rmse,
        "naive_mae":           naive_mae,
        "rmse_ratio_vs_naive": rmse_ratio,
        "mase":                mase,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S4 — OOS Walk-Forward Ridge Forecast")
    print("=" * 72)
    print(f"Horizons:    {list(HORIZONS)}")
    print(f"Test start:  {TEST_START.date()}")

    # ---- 1. Load α*
    print("\n[1/4] Loading α* from S2 + S2b...")
    alphas = load_selected_alphas()
    for (c, f), a in sorted(alphas.items()):
        print(f"    ({c:<8}, {f:<22}) α* = {a:>10.4g}")

    # ---- 2. Build matrices
    print("\n[2/4] Building feature matrices...")
    primary = build_all_features()
    usa_fd = build_usa_first_diff_features()
    matrices = [(c, "primary", primary[c]) for c in MAIN_COUNTRIES]
    matrices.append(("USA", "first_diff_secondary", usa_fd))

    # ---- 3. Walk-forward forecasting per combination
    print("\n[3/4] Running walk-forward Ridge forecasts...")
    all_rows: List[Dict] = []
    for country, form, df in matrices:
        alpha = alphas[(country, form)]
        X_full, y_full = split_full_clean(df, country)
        origins = compute_origins(X_full.index, TEST_START, HORIZONS)
        print(
            f"    {country:<8} {form:<22}  α*={alpha:>10.4g}  "
            f"X_full={X_full.shape}  origins={len(origins)}"
            f"  ({origins[0].date()} .. {origins[-1].date()})"
        )
        rows = walk_forward_direct_h(
            country, form, X_full, y_full, alpha, HORIZONS, origins
        )
        all_rows.extend(rows)

    df_fc = pd.DataFrame(all_rows)
    out_fc = DOC_DIR / "phase6_step3_s4_ridge_oos_forecasts.csv"
    df_fc.to_csv(out_fc, index=False)
    print(f"    -> {out_fc.relative_to(PROJECT_ROOT)}  ({len(df_fc)} rows)")

    # ---- 4. Metrics per (country, form, horizon)
    print("\n[4/4] Computing metrics...")
    metric_rows: List[Dict] = []
    for country, form, df in matrices:
        X_full, y_full = split_full_clean(df, country)
        y_train = y_full.loc[y_full.index < TEST_START]
        alpha = alphas[(country, form)]
        for h in HORIZONS:
            grp = df_fc[
                (df_fc["country"] == country)
                & (df_fc["form"] == form)
                & (df_fc["horizon"] == h)
            ]
            m = compute_metrics_for_group(grp, y_train)
            metric_rows.append({
                "country": country,
                "form":    form,
                "horizon": h,
                "alpha":   float(alpha),
                **m,
            })

    df_metrics = pd.DataFrame(metric_rows)
    out_metrics = DOC_DIR / "phase6_step3_s4_ridge_oos_metrics.csv"
    df_metrics.to_csv(out_metrics, index=False)
    print(f"    -> {out_metrics.relative_to(PROJECT_ROOT)}  ({len(df_metrics)} rows)")

    # Compact CPI summary (same rows, selected columns)
    compact_cols = [
        "country", "form", "horizon", "n_origins",
        "rmse", "mae", "rmse_ratio_vs_naive", "mase",
    ]
    df_compact = df_metrics[compact_cols].copy()
    out_compact = DOC_DIR / "phase6_step3_s4_ridge_oos_cpi_summary.csv"
    df_compact.to_csv(out_compact, index=False)
    print(f"    -> {out_compact.relative_to(PROJECT_ROOT)}  ({len(df_compact)} rows)")

    # -----------------------------------------------------------------------
    # Terminal previews
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("OOS METRICS — ALL COMBINATIONS × HORIZONS")
    print("=" * 72)
    disp = df_metrics.copy()
    for c in [
        "rmse", "mae", "bias", "med_abs_error",
        "naive_rmse", "naive_mae", "rmse_ratio_vs_naive", "mase",
    ]:
        disp[c] = disp[c].round(4)
    print(disp.to_string(index=False))

    # MASE matrix: rows = (country, form), cols = horizon
    print("\n" + "=" * 72)
    print("MASE MATRIX (MAE / in-sample 1-step naive MAE; <1 beats random-walk)")
    print("=" * 72)
    mase_pivot = df_metrics.pivot_table(
        index=["country", "form"],
        columns="horizon",
        values="mase",
    ).round(4)
    print(mase_pivot.to_string())

    # RMSE ratio matrix
    print("\n" + "=" * 72)
    print("RMSE / NAIVE_RMSE RATIO (<1 beats random-walk at each horizon)")
    print("=" * 72)
    ratio_pivot = df_metrics.pivot_table(
        index=["country", "form"],
        columns="horizon",
        values="rmse_ratio_vs_naive",
    ).round(4)
    print(ratio_pivot.to_string())

    print(
        "\nS4 complete. Forecast CSV ready for Phase 7 DM. "
        "Proceed to S5 (cross-country summary + Phase 7 handoff) after review."
    )


if __name__ == "__main__":
    main()
