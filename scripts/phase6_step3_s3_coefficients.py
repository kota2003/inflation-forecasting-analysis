#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S3 — Ridge Coefficient Stability + Feature Importance

For each (country, form) combination, loads the S2 / S2b-selected α and:
  1. Fits Ridge(α*) on the full train window (≤ 2019-12) → main coefficient
  2. Fits Ridge(α*) on each TimeSeriesSplit(5) fold → fold-wise coefs
  3. Extracts standardized-space coefficients for cross-feature magnitude
     comparison, ranks them, and computes sign stability

Pending-formal decisions referenced (to be entered with D-065+):
  - α* source hierarchy: S2 is baseline, S2b overrides JPN primary
  - Coefficients reported in STANDARDIZED feature space (scaler-normalised),
    giving direct magnitude comparability across features and countries
  - Stability metric: mean / std / min / max / sign_stable across 5 folds

Usage:
    python scripts/phase6_step3_s3_coefficients.py

Outputs (under data/documentation/):
    phase6_step3_s3_ridge_coefficients.csv     (~254 rows)
    phase6_step3_s3_top_features.csv           (50 rows: 10 × 5 combos)
    phase6_step3_s3_category_contribution.csv  (~30 rows: 6 cats × 5 combos)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
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
N_SPLITS = 5
RANDOM_STATE = 0
TOP_K = 10

RE_INTERACTION = re.compile(r"_x_")
RE_SPLIT = re.compile(r"_D_(GFC_2008|COVID_2020|ENERGY_2022)$")
RE_PERIOD = re.compile(r"_P_(GFC|COVID|ENERGY|2008|2020|2022)")
RE_LAG = re.compile(r"_lag\d+$")
RE_ROLLING = re.compile(r"_roll\d+_(mean|std)$")

CATEGORY_ORDER = ["base", "lag", "rolling", "split", "period", "interaction"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_feature(col: str) -> str:
    if RE_INTERACTION.search(col):
        return "interaction"
    if RE_SPLIT.search(col):
        return "split"
    if RE_PERIOD.search(col):
        return "period"
    if RE_LAG.search(col):
        return "lag"
    if RE_ROLLING.search(col):
        return "rolling"
    return "base"


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


def split_xy(df: pd.DataFrame, country: str) -> Tuple[pd.DataFrame, pd.Series]:
    target = target_col_name(country)
    df_clean = df.dropna()
    df_train = df_clean.loc[df_clean.index <= TRAIN_END]
    y = df_train[target]
    X = df_train.drop(columns=[target])
    return X, y


def load_selected_alphas() -> Dict[Tuple[str, str], float]:
    """S2 baseline + S2b override for JPN primary."""
    path_s2 = DOC_DIR / "phase6_step3_s2_alpha_selection.csv"
    path_s2b = DOC_DIR / "phase6_step3_s2b_japan_alpha_selection.csv"
    if not path_s2.exists():
        raise FileNotFoundError(f"S2 selection CSV not found: {path_s2}")
    if not path_s2b.exists():
        raise FileNotFoundError(f"S2b selection CSV not found: {path_s2b}")

    sel_s2 = pd.read_csv(path_s2)
    sel_s2b = pd.read_csv(path_s2b)

    out: Dict[Tuple[str, str], float] = {}
    for _, r in sel_s2.iterrows():
        out[(r["country"], r["form"])] = float(r["selected_alpha"])
    for _, r in sel_s2b.iterrows():
        out[(r["country"], r["form"])] = float(r["selected_alpha"])  # overrides
    return out


# ---------------------------------------------------------------------------
# Core coefficient extraction
# ---------------------------------------------------------------------------
def fit_ridge_extract_coefs(
    X: pd.DataFrame, y: pd.Series, alpha: float,
) -> np.ndarray:
    """Fit StandardScaler + Ridge pipeline, return coefs in standardized space."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X, y)
    return pipe.named_steps["ridge"].coef_.copy()


def compute_coefficient_stability(
    country: str, form: str, X: pd.DataFrame, y: pd.Series, alpha: float,
) -> pd.DataFrame:
    """
    Return a per-feature DataFrame with full-train coef and fold stats.

    Columns:
        country, form, feature_name, category, selected_alpha,
        coef_full_train,
        coef_fold_mean, coef_fold_std, coef_fold_min, coef_fold_max,
        sign_stable (bool), n_folds_same_sign_as_full,
        rank_abs_full (1 = largest magnitude)
    """
    features = list(X.columns)
    coef_full = fit_ridge_extract_coefs(X, y, alpha)

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_coefs: List[np.ndarray] = []
    for tr_idx, _ in splitter.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        fold_coefs.append(fit_ridge_extract_coefs(X_tr, y_tr, alpha))
    fold_arr = np.vstack(fold_coefs)  # shape (n_splits, n_features)

    fold_mean = fold_arr.mean(axis=0)
    fold_std = fold_arr.std(axis=0, ddof=1)
    fold_min = fold_arr.min(axis=0)
    fold_max = fold_arr.max(axis=0)

    # sign stability: all 5 folds AND full-train share the same sign
    sign_full = np.sign(coef_full)
    sign_folds = np.sign(fold_arr)
    all_signs = np.vstack([sign_folds, sign_full])
    sign_stable = np.all(all_signs == sign_full, axis=0)
    n_same_sign = (sign_folds == sign_full).sum(axis=0)

    abs_full = np.abs(coef_full)
    rank_full = (-abs_full).argsort().argsort() + 1  # 1 = largest

    rows = []
    for i, feat in enumerate(features):
        rows.append({
            "country":                  country,
            "form":                     form,
            "feature_name":             feat,
            "category":                 classify_feature(feat),
            "selected_alpha":           float(alpha),
            "coef_full_train":          float(coef_full[i]),
            "coef_fold_mean":           float(fold_mean[i]),
            "coef_fold_std":            float(fold_std[i]),
            "coef_fold_min":            float(fold_min[i]),
            "coef_fold_max":            float(fold_max[i]),
            "abs_coef_full":            float(abs_full[i]),
            "sign_stable":              bool(sign_stable[i]),
            "n_folds_same_sign_as_full": int(n_same_sign[i]),
            "rank_abs_full":            int(rank_full[i]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S3 — Ridge Coefficient Stability + Importance")
    print("=" * 72)

    # ---- 1. Load selected α per (country, form)
    print("\n[1/4] Loading α* from S2 + S2b selection CSVs...")
    alphas = load_selected_alphas()
    for (c, f), a in sorted(alphas.items()):
        print(f"    ({c:<8}, {f:<22}) α* = {a:>10.4g}  (log10 = {np.log10(a):+.2f})")

    # ---- 2. Build feature matrices
    print("\n[2/4] Building feature matrices...")
    primary = build_all_features()
    usa_fd = build_usa_first_diff_features()
    matrices = [(c, "primary", primary[c]) for c in MAIN_COUNTRIES]
    matrices.append(("USA", "first_diff_secondary", usa_fd))

    # ---- 3. Compute coefficients per combination
    print("\n[3/4] Fitting Ridge and extracting coefficients...")
    all_rows: List[pd.DataFrame] = []
    for country, form, df in matrices:
        alpha = alphas.get((country, form))
        if alpha is None:
            raise KeyError(f"No α* found for ({country}, {form})")
        X, y = split_xy(df, country)
        print(
            f"    {country:<8} {form:<22}  α*={alpha:>10.4g}  "
            f"X={X.shape}  y={y.shape}"
        )
        stab = compute_coefficient_stability(country, form, X, y, alpha)
        all_rows.append(stab)
    df_coef = pd.concat(all_rows, ignore_index=True)

    # ---- 4. Derived CSVs: top-K per combination, category contribution
    print("\n[4/4] Assembling top-features and category-contribution CSVs...")

    # Top-K per (country, form)
    top_rows = []
    for (country, form), grp in df_coef.groupby(["country", "form"], sort=False):
        top = grp.nsmallest(TOP_K, "rank_abs_full").sort_values("rank_abs_full")
        for _, r in top.iterrows():
            top_rows.append({
                "country":         country,
                "form":            form,
                "rank":            int(r["rank_abs_full"]),
                "feature_name":    r["feature_name"],
                "category":        r["category"],
                "coef_full_train": float(r["coef_full_train"]),
                "coef_fold_mean":  float(r["coef_fold_mean"]),
                "coef_fold_std":   float(r["coef_fold_std"]),
                "sign_stable":     bool(r["sign_stable"]),
                "selected_alpha":  float(r["selected_alpha"]),
            })
    df_top = pd.DataFrame(top_rows)

    # Category contribution: sum / mean / max |coef| per (country, form, category)
    cat_rows = []
    for (country, form), grp in df_coef.groupby(["country", "form"], sort=False):
        for cat in CATEGORY_ORDER:
            sub = grp[grp["category"] == cat]
            if len(sub) == 0:
                cat_rows.append({
                    "country":             country,
                    "form":                form,
                    "category":            cat,
                    "n_features":          0,
                    "sum_abs_coef":        0.0,
                    "mean_abs_coef":       0.0,
                    "max_abs_coef":        0.0,
                    "top_feature":         "",
                    "top_feature_coef":    0.0,
                })
                continue
            idx_top = sub["abs_coef_full"].idxmax()
            cat_rows.append({
                "country":             country,
                "form":                form,
                "category":            cat,
                "n_features":          int(len(sub)),
                "sum_abs_coef":        float(sub["abs_coef_full"].sum()),
                "mean_abs_coef":       float(sub["abs_coef_full"].mean()),
                "max_abs_coef":        float(sub["abs_coef_full"].max()),
                "top_feature":         str(sub.loc[idx_top, "feature_name"]),
                "top_feature_coef":    float(sub.loc[idx_top, "coef_full_train"]),
            })
    df_cat = pd.DataFrame(cat_rows)

    out_coef = DOC_DIR / "phase6_step3_s3_ridge_coefficients.csv"
    out_top = DOC_DIR / "phase6_step3_s3_top_features.csv"
    out_cat = DOC_DIR / "phase6_step3_s3_category_contribution.csv"
    df_coef.to_csv(out_coef, index=False)
    df_top.to_csv(out_top, index=False)
    df_cat.to_csv(out_cat, index=False)
    print(f"    -> {out_coef.relative_to(PROJECT_ROOT)}  ({len(df_coef)} rows)")
    print(f"    -> {out_top.relative_to(PROJECT_ROOT)}  ({len(df_top)} rows)")
    print(f"    -> {out_cat.relative_to(PROJECT_ROOT)}  ({len(df_cat)} rows)")

    # -----------------------------------------------------------------------
    # Terminal previews
    # -----------------------------------------------------------------------
    # Overall magnitude summary per combination (gives the Japan-dwarf signal)
    print("\n" + "=" * 72)
    print("COEFFICIENT MAGNITUDE SUMMARY (full-train, per combination)")
    print("=" * 72)
    summary_rows = []
    for (country, form), grp in df_coef.groupby(["country", "form"], sort=False):
        summary_rows.append({
            "country":       country,
            "form":          form,
            "n_features":    int(len(grp)),
            "max_abs_coef":  float(grp["abs_coef_full"].max()),
            "mean_abs_coef": float(grp["abs_coef_full"].mean()),
            "sum_abs_coef":  float(grp["abs_coef_full"].sum()),
            "p50_abs_coef":  float(grp["abs_coef_full"].median()),
            "sign_stable_count": int(grp["sign_stable"].sum()),
        })
    df_overall = pd.DataFrame(summary_rows)
    for c in ["max_abs_coef", "mean_abs_coef", "sum_abs_coef", "p50_abs_coef"]:
        df_overall[c] = df_overall[c].round(4)
    print(df_overall.to_string(index=False))

    # Top-5 per combination
    print("\n" + "=" * 72)
    print("TOP-5 FEATURES PER COMBINATION (by |coef_full_train|)")
    print("=" * 72)
    for (country, form), grp in df_top.groupby(["country", "form"], sort=False):
        top5 = grp.head(5)
        print(f"\n  {country} · {form}  (α* = {top5['selected_alpha'].iloc[0]:.4g})")
        disp = top5[[
            "rank", "feature_name", "category",
            "coef_full_train", "coef_fold_std", "sign_stable",
        ]].copy()
        disp["coef_full_train"] = disp["coef_full_train"].round(4)
        disp["coef_fold_std"]   = disp["coef_fold_std"].round(4)
        print(disp.to_string(index=False))

    # Category contribution matrix: sum_abs_coef pivot
    print("\n" + "=" * 72)
    print("CATEGORY CONTRIBUTION (sum_abs_coef per category × combination)")
    print("=" * 72)
    pivot = df_cat.pivot_table(
        index="category",
        columns=["country", "form"],
        values="sum_abs_coef",
        fill_value=0.0,
    ).reindex(CATEGORY_ORDER)
    print(pivot.round(4).to_string())

    print(
        "\nS3 complete. Proceed to S4 (OOS walk-forward forecast for Phase 7 DM)."
    )


if __name__ == "__main__":
    main()
