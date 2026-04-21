#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S1 — Data Preparation for Ridge Layer 3

Prepares the four-country (+ USA secondary) feature matrices and target
vectors for Ridge regression. Produces three audit CSVs covering matrix
shape, feature category counts, and target distribution statistics.

Pending-formal decisions referenced (to be entered as D-064+):
  - D1 (feature set): Phase 4 full superset 50-53 cols; no pre-pruning
    (D-040 deferred selection to Ridge L2 regularisation)
  - D2 (target): per-country CPI; USA dual-form (yoy_pct primary +
    first_diff secondary per D-048 / D-062 alignment for Phase 7 DM)
  - D6 (OOS alignment): train = 2000-01 .. 2019-12,
                        test  = 2020-01 .. present
    (D-005 matched with ARIMA Step 1 and VAR Step 2 S6)

Usage:
    python scripts/phase6_step3_s1_data_preparation.py

Outputs (under data/documentation/):
    phase6_step3_s1_feature_matrix_summary.csv   (5 rows)
    phase6_step3_s1_feature_categories.csv       (5 rows)
    phase6_step3_s1_target_summary.csv           (5 rows)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path injection — ensure project root is importable before 'src' is used
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from typing import Dict

import numpy as np
import pandas as pd

from src import (
    MAIN_COUNTRIES,
    build_all_features,
    build_country_features,
    load_effective_registry,
)
import src.feature_engineering as fe_module


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOC_DIR = PROJECT_ROOT / "data" / "documentation"
DOC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2019-12-01")
TEST_START = pd.Timestamp("2020-01-01")

# Feature category regex patterns (applied in priority order).
# Interaction check must come first because interaction cols often contain
# the split-dummy substring (e.g. USA_D_COVID_2020_x_USA_M2).
RE_INTERACTION = re.compile(r"_x_")
RE_SPLIT = re.compile(r"_D_(GFC_2008|COVID_2020|ENERGY_2022)$")
RE_PERIOD = re.compile(r"_P_(GFC|COVID|ENERGY|2008|2020|2022)")
RE_LAG = re.compile(r"_lag\d+$")
RE_ROLLING = re.compile(r"_roll\d+_(mean|std)$")


def classify_feature(col: str) -> str:
    """Map a feature column name to its Phase 4 category."""
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
    """CPI target column name (identical string for primary / secondary form)."""
    return f"{country}_CPI"


# ---------------------------------------------------------------------------
# USA secondary form builder (first_diff)
# ---------------------------------------------------------------------------
def build_usa_first_diff_features() -> pd.DataFrame:
    """
    Build USA feature matrix with CPI in first_diff secondary form.

    Strategy: temporarily patch ``src.feature_engineering.REGISTRY_OVERRIDES``
    to force ``('USA', 'CPI')`` → ``first_diff``, rebuild the feature matrix
    end-to-end so that all CPI-derived lag / rolling / interaction columns
    are consistent with the first_diff base, then restore the original
    dictionary state.

    Note
    ----
    ``REGISTRY_OVERRIDES`` uses **tuple keys** ``(country, indicator)``,
    not string keys — see feature_engineering.py line 68 and the
    tuple-based lookup in ``load_effective_registry()`` line 137.
    """
    key = ("USA", "CPI")

    # Snapshot original state
    had_override = key in fe_module.REGISTRY_OVERRIDES
    original_value = fe_module.REGISTRY_OVERRIDES.get(key)

    # Diagnostic: effective form BEFORE patch
    reg_before = load_effective_registry()
    row_before = reg_before[
        (reg_before["country"] == "USA") & (reg_before["indicator"] == "CPI")
    ].iloc[0]
    print(
        f"    [pre-patch ] USA_CPI effective form = "
        f"{row_before['effective_phase6_var_input']!r}  "
        f"(override_applied = {row_before['override_applied']})"
    )

    try:
        fe_module.REGISTRY_OVERRIDES[key] = "first_diff"

        # Diagnostic: effective form AFTER patch
        reg_after = load_effective_registry()
        row_after = reg_after[
            (reg_after["country"] == "USA") & (reg_after["indicator"] == "CPI")
        ].iloc[0]
        print(
            f"    [post-patch] USA_CPI effective form = "
            f"{row_after['effective_phase6_var_input']!r}  "
            f"(override_applied = {row_after['override_applied']})"
        )

        usa_fd = build_country_features("USA")
    finally:
        if had_override:
            fe_module.REGISTRY_OVERRIDES[key] = original_value
        else:
            fe_module.REGISTRY_OVERRIDES.pop(key, None)
    return usa_fd


# ---------------------------------------------------------------------------
# Summary computations
# ---------------------------------------------------------------------------
def summarise_matrix(country: str, form: str, df: pd.DataFrame) -> Dict:
    """Per-(country, form) matrix shape / window summary."""
    target = target_col_name(country)
    if target not in df.columns:
        raise KeyError(
            f"Target column {target} missing in {country}/{form} matrix"
        )

    df_clean = df.dropna()
    n_rows_all = len(df)
    n_rows_clean = len(df_clean)

    first_valid = df_clean.index.min() if n_rows_clean else pd.NaT
    last_valid = df_clean.index.max() if n_rows_clean else pd.NaT

    train_mask = df_clean.index <= TRAIN_END
    test_mask = df_clean.index >= TEST_START

    return {
        "country": country,
        "form": form,
        "n_cols_total": int(df.shape[1]),
        "n_features_X": int(df.shape[1] - 1),
        "target_col": target,
        "n_rows_pre_dropna": n_rows_all,
        "n_rows_joint_valid": n_rows_clean,
        "first_valid_date": (
            first_valid.strftime("%Y-%m-%d") if pd.notna(first_valid) else ""
        ),
        "last_valid_date": (
            last_valid.strftime("%Y-%m-%d") if pd.notna(last_valid) else ""
        ),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_end": TRAIN_END.strftime("%Y-%m-%d"),
        "test_start": TEST_START.strftime("%Y-%m-%d"),
    }


def categorise_features(country: str, form: str, df: pd.DataFrame) -> Dict:
    """Count Phase 4 feature categories for one (country, form) matrix."""
    target = target_col_name(country)
    features = [c for c in df.columns if c != target]
    cats = {
        "base": 0,
        "lag": 0,
        "rolling": 0,
        "split": 0,
        "period": 0,
        "interaction": 0,
    }
    for col in features:
        cats[classify_feature(col)] += 1
    return {
        "country": country,
        "form": form,
        **cats,
        "total_features": len(features),
    }


def summarise_target(country: str, form: str, df: pd.DataFrame) -> Dict:
    """Target distribution split across train / test windows."""
    target = target_col_name(country)
    df_clean = df.dropna()
    y = df_clean[target]

    train_y = y.loc[y.index <= TRAIN_END]
    test_y = y.loc[y.index >= TEST_START]

    def _stats(s: pd.Series) -> Dict:
        if len(s) == 0:
            return dict(n=0, mean=np.nan, std=np.nan, minv=np.nan, maxv=np.nan)
        return dict(
            n=int(len(s)),
            mean=float(s.mean()),
            std=float(s.std(ddof=1)),
            minv=float(s.min()),
            maxv=float(s.max()),
        )

    tr = _stats(train_y)
    te = _stats(test_y)
    return {
        "country": country,
        "form": form,
        "target_col": target,
        "train_n": tr["n"],
        "train_mean": tr["mean"],
        "train_std": tr["std"],
        "train_min": tr["minv"],
        "train_max": tr["maxv"],
        "test_n": te["n"],
        "test_mean": te["mean"],
        "test_std": te["std"],
        "test_min": te["minv"],
        "test_max": te["maxv"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S1 — Data Preparation for Ridge Layer 3")
    print("=" * 72)

    # ---- 1. Primary-form matrices for 4 countries via standard pipeline
    print("\n[1/3] Building primary-form feature matrices (4 countries)...")
    primary: Dict[str, pd.DataFrame] = build_all_features()
    for c in MAIN_COUNTRIES:
        raw_shape = primary[c].shape
        clean_shape = primary[c].dropna().shape
        print(f"    {c:<8} shape = {raw_shape}  dropna = {clean_shape}")

    # ---- 2. USA secondary form (first_diff) via REGISTRY_OVERRIDES patch
    print("\n[2/3] Building USA secondary-form feature matrix (first_diff)...")
    usa_fd = build_usa_first_diff_features()
    print(f"    USA_fd   shape = {usa_fd.shape}  dropna = {usa_fd.dropna().shape}")

    # Collect matrices under a unified (country, form) labelling
    matrices = [(c, "primary", primary[c]) for c in MAIN_COUNTRIES]
    matrices.append(("USA", "first_diff_secondary", usa_fd))

    # ---- 3. Assemble audit CSVs
    print("\n[3/3] Assembling audit CSVs...")
    summary_rows = []
    category_rows = []
    target_rows = []
    for country, form, df in matrices:
        summary_rows.append(summarise_matrix(country, form, df))
        category_rows.append(categorise_features(country, form, df))
        target_rows.append(summarise_target(country, form, df))

    df_summary = pd.DataFrame(summary_rows)
    df_categories = pd.DataFrame(category_rows)
    df_target = pd.DataFrame(target_rows)

    out_summary = DOC_DIR / "phase6_step3_s1_feature_matrix_summary.csv"
    out_categories = DOC_DIR / "phase6_step3_s1_feature_categories.csv"
    out_target = DOC_DIR / "phase6_step3_s1_target_summary.csv"

    df_summary.to_csv(out_summary, index=False)
    df_categories.to_csv(out_categories, index=False)
    df_target.to_csv(out_target, index=False)

    print(f"    -> {out_summary.relative_to(PROJECT_ROOT)}  ({len(df_summary)} rows)")
    print(f"    -> {out_categories.relative_to(PROJECT_ROOT)}  ({len(df_categories)} rows)")
    print(f"    -> {out_target.relative_to(PROJECT_ROOT)}  ({len(df_target)} rows)")

    # ---- 4. Terminal preview (for conversational review with Kota)
    print("\n" + "=" * 72)
    print("MATRIX SUMMARY")
    print("=" * 72)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("FEATURE CATEGORIES")
    print("=" * 72)
    print(df_categories.to_string(index=False))

    print("\n" + "=" * 72)
    print("TARGET SUMMARY")
    print("=" * 72)
    print_df = df_target.copy()
    for c in [
        "train_mean", "train_std", "train_min", "train_max",
        "test_mean", "test_std", "test_min", "test_max",
    ]:
        print_df[c] = print_df[c].round(4)
    print(print_df.to_string(index=False))

    # ---- 5. Sanity check: USA primary vs secondary must differ on target scale
    usa_primary = df_target[
        (df_target["country"] == "USA") & (df_target["form"] == "primary")
    ].iloc[0]
    usa_secondary = df_target[
        (df_target["country"] == "USA") & (df_target["form"] == "first_diff_secondary")
    ].iloc[0]
    same_mean = np.isclose(usa_primary["train_mean"], usa_secondary["train_mean"])
    same_std = np.isclose(usa_primary["train_std"], usa_secondary["train_std"])
    print("\n" + "=" * 72)
    print("SANITY CHECK — USA dual-form differentiation")
    print("=" * 72)
    print(
        f"    primary   train: mean = {usa_primary['train_mean']:.4f}, "
        f"std = {usa_primary['train_std']:.4f}"
    )
    print(
        f"    secondary train: mean = {usa_secondary['train_mean']:.4f}, "
        f"std = {usa_secondary['train_std']:.4f}"
    )
    if same_mean and same_std:
        raise RuntimeError(
            "USA primary and secondary have identical target statistics. "
            "The REGISTRY_OVERRIDES patch did not take effect."
        )
    print("    PASS: primary and secondary differ (dual-form patch effective)")

    print("\nS1 complete. Proceed to S2 (α grid + walk-forward CV) after review.")


if __name__ == "__main__":
    main()
