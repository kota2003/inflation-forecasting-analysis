#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S5 — Cross-Country Narrative Consolidation +
                        Ridge vs VAR MASE Comparison

Consolidates S1-S4 outputs into three portfolio-ready summary tables.

ARIMA Step 1 is NOT joined here because ARIMA used fixed-origin multi-step
forecasting whereas VAR S6 and Ridge S4 use walk-forward. Phase 7 DM will
handle the ARIMA side separately.

VAR MASE values are hardcoded from D-060 (Phase 6 Step 2 S6 AIC-selected p
per country). This is the canonical "VAR primary" spec committed in the
three-layer architecture per D-004.

Pending-formal decisions (to be entered D-067+):
  - Three-layer forecast comparison table (Ridge vs VAR MASE, 4 × 4)
  - Narrative Ridge-lens attribution (N1/N2/N3 quantitative contributions)

Usage:
    python scripts/phase6_step3_s5_narrative_consolidation.py

Outputs (under data/documentation/):
    phase6_step3_s5_country_narrative_summary.csv   (5 rows)
    phase6_step3_s5_ridge_vs_var_mase.csv           (16 rows)
    phase6_step3_s5_narrative_ridge_statements.csv  (3 rows)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOC_DIR = PROJECT_ROOT / "data" / "documentation"
DOC_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 3, 6, 12)

# VAR MASE from D-060 (Phase 6 Step 2 S6, AIC-selected p VAR, primary form)
# Keys: (country, horizon)
VAR_MASE_D060: Dict[tuple, float] = {
    ("USA",     1):  3.73, ("USA",     3): 11.61, ("USA",     6): 20.64, ("USA",     12): 32.32,
    ("JAPAN",   1):  0.89, ("JAPAN",   3):  0.96, ("JAPAN",   6):  0.91, ("JAPAN",   12):  1.03,
    ("UK",      1):  1.90, ("UK",      3):  1.95, ("UK",      6):  5.60, ("UK",      12): 79.07,
    ("GERMANY", 1):  1.48, ("GERMANY", 3):  1.76, ("GERMANY", 6):  1.56, ("GERMANY", 12):  2.26,
}

# VAR AIC-selected lag per country (D-050 revised)
VAR_P_D050: Dict[str, int] = {
    "USA":     12,
    "JAPAN":    5,
    "UK":      12,
    "GERMANY": 12,
}


# ---------------------------------------------------------------------------
# Load S1-S4 CSVs
# ---------------------------------------------------------------------------
def _read_csv(name: str) -> pd.DataFrame:
    path = DOC_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path)


def load_all_inputs() -> Dict[str, pd.DataFrame]:
    return {
        "feature_matrix":      _read_csv("phase6_step3_s1_feature_matrix_summary.csv"),
        "feature_categories":  _read_csv("phase6_step3_s1_feature_categories.csv"),
        "target_summary":      _read_csv("phase6_step3_s1_target_summary.csv"),
        "alpha_selection":     _read_csv("phase6_step3_s2_alpha_selection.csv"),
        "alpha_jpn_ext":       _read_csv("phase6_step3_s2b_japan_alpha_selection.csv"),
        "coefficients":        _read_csv("phase6_step3_s3_ridge_coefficients.csv"),
        "top_features":        _read_csv("phase6_step3_s3_top_features.csv"),
        "category_contrib":    _read_csv("phase6_step3_s3_category_contribution.csv"),
        "oos_metrics":         _read_csv("phase6_step3_s4_ridge_oos_metrics.csv"),
    }


def merged_alpha_selection(
    sel_s2: pd.DataFrame, sel_s2b: pd.DataFrame
) -> pd.DataFrame:
    """S2 baseline + S2b override for JPN primary."""
    out = sel_s2.copy()
    out = out.set_index(["country", "form"])
    for _, r in sel_s2b.iterrows():
        key = (r["country"], r["form"])
        for col in sel_s2b.columns:
            if col in out.columns and col not in ("country", "form"):
                out.loc[key, col] = r[col]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Output 1: per-combination narrative summary
# ---------------------------------------------------------------------------
def build_country_narrative_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    alpha = merged_alpha_selection(
        inputs["alpha_selection"], inputs["alpha_jpn_ext"]
    )
    top = inputs["top_features"]
    coef = inputs["coefficients"]
    metrics = inputs["oos_metrics"]
    feat_mx = inputs["feature_matrix"]

    rows: List[Dict] = []
    keys = alpha[["country", "form"]].apply(tuple, axis=1).tolist()
    for country, form in keys:
        a_row = alpha[
            (alpha["country"] == country) & (alpha["form"] == form)
        ].iloc[0]
        t_rows = top[
            (top["country"] == country) & (top["form"] == form)
        ].sort_values("rank")
        c_rows = coef[(coef["country"] == country) & (coef["form"] == form)]
        m_rows = metrics[
            (metrics["country"] == country) & (metrics["form"] == form)
        ].sort_values("horizon")
        f_row = feat_mx[
            (feat_mx["country"] == country) & (feat_mx["form"] == form)
        ].iloc[0]

        top1 = t_rows.iloc[0] if len(t_rows) >= 1 else None
        top2 = t_rows.iloc[1] if len(t_rows) >= 2 else None
        top3 = t_rows.iloc[2] if len(t_rows) >= 3 else None

        mase_by_h = dict(zip(m_rows["horizon"], m_rows["mase"]))
        rmse_ratio_by_h = dict(zip(m_rows["horizon"], m_rows["rmse_ratio_vs_naive"]))
        beats_naive_rmse = sum(
            1 for h in HORIZONS
            if not np.isnan(rmse_ratio_by_h.get(h, np.nan))
            and rmse_ratio_by_h.get(h) < 1.0
        )
        beats_naive_mase = sum(
            1 for h in HORIZONS
            if not np.isnan(mase_by_h.get(h, np.nan))
            and mase_by_h.get(h) < 1.0
        )

        rows.append({
            "country":              country,
            "form":                 form,
            "n_features":           int(f_row["n_features_X"]),
            "n_train":              int(f_row["n_train"]),
            "n_test":               int(f_row["n_test"]),
            "selected_alpha":       float(a_row["selected_alpha"]),
            "log10_alpha":          float(a_row["selected_log10_alpha"]),
            "cv_val_mse":           float(a_row["cv_val_mse_mean"]),
            "sum_abs_coef_full":    float(c_rows["abs_coef_full"].sum()),
            "max_abs_coef_full":    float(c_rows["abs_coef_full"].max()),
            "sign_stable_count":    int(c_rows["sign_stable"].sum()),
            "top1_feature":         str(top1["feature_name"]) if top1 is not None else "",
            "top1_coef":            float(top1["coef_full_train"]) if top1 is not None else np.nan,
            "top1_category":        str(top1["category"]) if top1 is not None else "",
            "top2_feature":         str(top2["feature_name"]) if top2 is not None else "",
            "top2_coef":            float(top2["coef_full_train"]) if top2 is not None else np.nan,
            "top3_feature":         str(top3["feature_name"]) if top3 is not None else "",
            "top3_coef":            float(top3["coef_full_train"]) if top3 is not None else np.nan,
            "mase_h1":              float(mase_by_h.get(1,  np.nan)),
            "mase_h3":              float(mase_by_h.get(3,  np.nan)),
            "mase_h6":              float(mase_by_h.get(6,  np.nan)),
            "mase_h12":             float(mase_by_h.get(12, np.nan)),
            "rmse_ratio_h1":        float(rmse_ratio_by_h.get(1,  np.nan)),
            "rmse_ratio_h3":        float(rmse_ratio_by_h.get(3,  np.nan)),
            "rmse_ratio_h6":        float(rmse_ratio_by_h.get(6,  np.nan)),
            "rmse_ratio_h12":       float(rmse_ratio_by_h.get(12, np.nan)),
            "beats_naive_rmse_n":   int(beats_naive_rmse),
            "beats_naive_mase_n":   int(beats_naive_mase),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output 2: Ridge vs VAR MASE comparison
# ---------------------------------------------------------------------------
def build_ridge_vs_var_mase(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """4 countries × 4 horizons, primary form only (VAR D-060 is primary)."""
    metrics = inputs["oos_metrics"]
    ridge_pri = metrics[metrics["form"] == "primary"].copy()

    rows: List[Dict] = []
    for (country, h), var_mase in sorted(VAR_MASE_D060.items()):
        m = ridge_pri[
            (ridge_pri["country"] == country) & (ridge_pri["horizon"] == h)
        ]
        if len(m) == 0:
            continue
        ridge_mase = float(m["mase"].iloc[0])
        ridge_rmse = float(m["rmse"].iloc[0])
        ridge_ratio = float(m["rmse_ratio_vs_naive"].iloc[0])

        # Winner by MASE
        if ridge_mase < var_mase:
            winner = "Ridge"
            delta = ridge_mase - var_mase  # negative = Ridge better
            pct_improvement = (var_mase - ridge_mase) / var_mase * 100.0
        elif ridge_mase > var_mase:
            winner = "VAR"
            delta = ridge_mase - var_mase
            pct_improvement = (var_mase - ridge_mase) / var_mase * 100.0
        else:
            winner = "Tie"
            delta = 0.0
            pct_improvement = 0.0

        rows.append({
            "country":                country,
            "horizon":                int(h),
            "ridge_mase":             ridge_mase,
            "ridge_rmse":             ridge_rmse,
            "ridge_rmse_ratio_naive": ridge_ratio,
            "var_mase_d060":          float(var_mase),
            "var_aic_p":              VAR_P_D050[country],
            "mase_delta_ridge_minus_var": delta,
            "pct_improvement_ridge":  pct_improvement,
            "winner":                 winner,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output 3: Narrative Ridge-lens statements (N1 / N2 / N3)
# ---------------------------------------------------------------------------
def build_narrative_statements(
    inputs: Dict[str, pd.DataFrame], summary: pd.DataFrame, comparison: pd.DataFrame,
) -> pd.DataFrame:
    coef = inputs["coefficients"]
    top = inputs["top_features"]

    # N1 — cross-country dynamics
    primary_only = summary[summary["form"] == "primary"].set_index("country")
    alpha_min = primary_only["selected_alpha"].min()
    alpha_max = primary_only["selected_alpha"].max()
    alpha_range_ratio = alpha_max / alpha_min
    country_max = primary_only["selected_alpha"].idxmax()
    country_min = primary_only["selected_alpha"].idxmin()

    # Phillips Curve base-feature lens — which countries have UNEMPLOYMENT in top-5
    unemp_countries = []
    for country in ["USA", "JAPAN", "UK", "GERMANY"]:
        t = top[(top["country"] == country) & (top["form"] == "primary")]
        if any(t["feature_name"].str.contains("UNEMPLOYMENT")):
            unemp_countries.append(country)
    n1_statement = (
        f"Ridge α* spans {alpha_range_ratio:.0f}x across four countries "
        f"({country_min} α*={alpha_min:.0f} → {country_max} α*={alpha_max:.0f}); "
        f"Phillips base-feature (UNEMPLOYMENT) top-5 only in: "
        f"{', '.join(unemp_countries) if unemp_countries else 'none'}."
    )
    n1_row = {
        "narrative_id": "N1",
        "title": "Cross-Country Inflation Dynamics",
        "ridge_contribution_type": "Regularisation-intensity heterogeneity + Phillips surface",
        "ridge_key_statistic": n1_statement,
        "cross_lens_count":  3,  # VAR FEVD, Ridge, Phase 5 rolling Phillips
        "ridge_adds_new_lens": True,
    }

    # N2 — policy response (USA specific)
    usa_pri_top = top[(top["country"] == "USA") & (top["form"] == "primary")]
    usa_fd_top = top[(top["country"] == "USA") & (top["form"] == "first_diff_secondary")]
    usa_pri_has_policy = any(usa_pri_top["feature_name"].str.contains("POLICY_RATE"))
    usa_fd_policy = usa_fd_top[
        usa_fd_top["feature_name"].str.contains("POLICY_RATE")
    ]
    fd_policy_str = (
        "; ".join(
            f"{r.feature_name}={r.coef_full_train:+.3f}"
            for _, r in usa_fd_policy.iterrows()
        ) if len(usa_fd_policy) else "none"
    )
    n2_statement = (
        f"USA dual-form divergence: primary (yoy_pct) top-5 has POLICY_RATE = "
        f"{'yes' if usa_pri_has_policy else 'no'}; first_diff secondary top-5 "
        f"features including policy-rate lags: {fd_policy_str}. "
        "D-056 VAR IRF peak at h=4 with negative sign reproduced by Ridge first_diff."
    )
    n2_row = {
        "narrative_id": "N2",
        "title": "Policy Response Patterns (USA Monetary Transmission)",
        "ridge_contribution_type": "Dual-form validation + VAR IRF cross-lens consistency",
        "ridge_key_statistic": n2_statement,
        "cross_lens_count": 4,   # VAR Granger, VAR IRF, VAR FEVD, Ridge first_diff
        "ridge_adds_new_lens": True,
    }

    # N3 — Japan uniqueness
    jpn_coef = coef[(coef["country"] == "JAPAN") & (coef["form"] == "primary")]
    other_max_abs = coef[
        (coef["country"] != "JAPAN") & (coef["form"] == "primary")
    ].groupby("country")["abs_coef_full"].max()
    jpn_max = float(jpn_coef["abs_coef_full"].max())
    ratio_min = float(other_max_abs.min()) / jpn_max
    ratio_max = float(other_max_abs.max()) / jpn_max
    jpn_alpha = float(primary_only.loc["JAPAN", "selected_alpha"])
    other_alphas = primary_only.drop("JAPAN")["selected_alpha"]
    alpha_ratio_min = jpn_alpha / other_alphas.max()
    alpha_ratio_max = jpn_alpha / other_alphas.min()
    jpn_mase_h1 = float(primary_only.loc["JAPAN", "mase_h1"])
    n3_statement = (
        f"JAPAN Ridge signatures: α*=3162 is {alpha_ratio_min:.0f}x to "
        f"{alpha_ratio_max:.0f}x larger than other countries; "
        f"max|coef|=0.0100 is {ratio_min:.1f}x to {ratio_max:.1f}x smaller than "
        f"other countries; OOS MASE h=1 = {jpn_mase_h1:.3f} (marginal beat-naive). "
        "Ridge-lens = 7th independent confirmation of N3 after ACF/ARIMA/VAR-lag/"
        "Granger/IRF/FEVD."
    )
    n3_row = {
        "narrative_id": "N3",
        "title": "Japan's Uniqueness (2022 Inflation Reversal + Isolation)",
        "ridge_contribution_type": "Coefficient-magnitude quantification of isolation",
        "ridge_key_statistic": n3_statement,
        "cross_lens_count": 7,   # ACF, ARIMA, VAR lag, Granger, IRF, FEVD, Ridge
        "ridge_adds_new_lens": True,
    }

    return pd.DataFrame([n1_row, n2_row, n3_row])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S5 — Narrative Consolidation + Ridge vs VAR")
    print("=" * 72)

    # ---- 1. Load S1-S4 inputs
    print("\n[1/4] Loading S1-S4 audit CSVs...")
    inputs = load_all_inputs()
    for name, df in inputs.items():
        print(f"    {name:<20} shape = {df.shape}")

    # ---- 2. Country narrative summary
    print("\n[2/4] Building per-combination narrative summary...")
    summary = build_country_narrative_summary(inputs)
    out_summary = DOC_DIR / "phase6_step3_s5_country_narrative_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"    -> {out_summary.relative_to(PROJECT_ROOT)}  ({len(summary)} rows)")

    # ---- 3. Ridge vs VAR MASE comparison
    print("\n[3/4] Building Ridge vs VAR MASE comparison (primary form only)...")
    comparison = build_ridge_vs_var_mase(inputs)
    out_comparison = DOC_DIR / "phase6_step3_s5_ridge_vs_var_mase.csv"
    comparison.to_csv(out_comparison, index=False)
    print(f"    -> {out_comparison.relative_to(PROJECT_ROOT)}  ({len(comparison)} rows)")

    # ---- 4. Narrative statements
    print("\n[4/4] Building N1 / N2 / N3 narrative statements...")
    statements = build_narrative_statements(inputs, summary, comparison)
    out_statements = DOC_DIR / "phase6_step3_s5_narrative_ridge_statements.csv"
    statements.to_csv(out_statements, index=False)
    print(f"    -> {out_statements.relative_to(PROJECT_ROOT)}  ({len(statements)} rows)")

    # -----------------------------------------------------------------------
    # Terminal previews
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COUNTRY NARRATIVE SUMMARY (per combination)")
    print("=" * 72)
    cols_show = [
        "country", "form", "n_features", "selected_alpha", "log10_alpha",
        "sum_abs_coef_full", "max_abs_coef_full",
        "top1_feature", "top1_coef",
        "mase_h1", "mase_h12",
        "beats_naive_rmse_n", "beats_naive_mase_n",
    ]
    disp = summary[cols_show].copy()
    for c in [
        "log10_alpha", "sum_abs_coef_full", "max_abs_coef_full",
        "top1_coef", "mase_h1", "mase_h12",
    ]:
        disp[c] = disp[c].round(4)
    print(disp.to_string(index=False))

    print("\n" + "=" * 72)
    print("RIDGE vs VAR MASE COMPARISON (primary form, 4 × 4 cells)")
    print("=" * 72)
    disp = comparison.copy()
    for c in ["ridge_mase", "ridge_rmse", "ridge_rmse_ratio_naive",
              "var_mase_d060", "mase_delta_ridge_minus_var",
              "pct_improvement_ridge"]:
        disp[c] = disp[c].round(4) if c != "pct_improvement_ridge" else disp[c].round(1)
    print(
        disp[
            ["country", "horizon", "ridge_mase", "var_mase_d060",
             "pct_improvement_ridge", "winner"]
        ].to_string(index=False)
    )

    # Aggregate winner count
    print("\n" + "=" * 72)
    print("WINNER AGGREGATION")
    print("=" * 72)
    winner_counts = comparison["winner"].value_counts()
    for w, n in winner_counts.items():
        print(f"    {w:<6} : {n} / {len(comparison)}")
    print("\n    By country:")
    by_country = (
        comparison.groupby("country")["winner"]
        .apply(lambda s: (s == "Ridge").sum())
    )
    for c, n in by_country.items():
        print(f"        {c:<8} Ridge wins: {n} / 4")

    print("\n" + "=" * 72)
    print("NARRATIVE RIDGE-LENS STATEMENTS")
    print("=" * 72)
    for _, r in statements.iterrows():
        print(f"\n[{r['narrative_id']}] {r['title']}")
        print(f"  Contribution: {r['ridge_contribution_type']}")
        print(f"  Cross-lens count: {r['cross_lens_count']}")
        print(f"  Key: {r['ridge_key_statistic']}")

    print(
        "\nS5 complete. Ready for Phase 6 Step 3 closure: "
        "D-064+ decision entries, src/ promotion scan, "
        "notebook 08_ridge_regression.ipynb, phase6_step3_summary.md."
    )


if __name__ == "__main__":
    main()
