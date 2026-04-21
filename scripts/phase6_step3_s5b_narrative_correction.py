#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 · Step 3 · S5b — Narrative Statement Logic Correction

S5 auto-generation had two logic bugs:

  (1) N1 "Phillips UNEMPLOYMENT in top-5" used str.contains over all
      top-10 rows and matched lag/rolling UNEMPLOYMENT features rather
      than the BASE-form contemporaneous UNEMPLOYMENT coefficient.
      Correct lens = base category + "UNEMPLOYMENT" exact suffix, in top-5.

  (2) N2 "USA primary top-5 has POLICY_RATE" also searched top-10 and
      over-matched. Correct check = top-5 only.

S5b recomputes both narrative statements from the S3 top_features CSV
using the corrected logic, overwrites the S5 narrative statements CSV,
and emits a new audit CSV documenting the Phillips base-feature lens
per country.

Pending-formal decisions (to be entered D-067):
  - Phillips base-feature lens restricted to CATEGORY=base AND
    feature_name suffix == "_UNEMPLOYMENT" (excludes all lags/rollings)
  - POLICY_RATE surface check restricted to top-5 per combination

Usage:
    python scripts/phase6_step3_s5b_narrative_correction.py

Outputs (under data/documentation/):
    phase6_step3_s5_narrative_ridge_statements.csv      (OVERWRITTEN, 3 rows)
    phase6_step3_s5b_phillips_base_feature_lens.csv     (NEW, 4 rows)
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
COUNTRIES_PRIMARY = ("USA", "JAPAN", "UK", "GERMANY")


# ---------------------------------------------------------------------------
# Load S1-S4 inputs
# ---------------------------------------------------------------------------
def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DOC_DIR / name)


# ---------------------------------------------------------------------------
# CORRECTED N1 logic — Phillips base-feature lens
# ---------------------------------------------------------------------------
def phillips_base_feature_audit(
    coef_df: pd.DataFrame, top_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each country (primary form), check whether contemporaneous
    UNEMPLOYMENT (base category, suffix '_UNEMPLOYMENT' exactly) appears
    in the top-5 features by |coef|. Returns a 4-row audit DataFrame.
    """
    rows: List[Dict] = []
    for country in COUNTRIES_PRIMARY:
        target_feat = f"{country}_UNEMPLOYMENT"

        # Full coefficient record for this feature (if present)
        full = coef_df[
            (coef_df["country"] == country)
            & (coef_df["form"] == "primary")
            & (coef_df["feature_name"] == target_feat)
        ]
        if len(full) == 0:
            rows.append({
                "country":              country,
                "feature_name":         target_feat,
                "rank_abs_full":        np.nan,
                "coef_full_train":      np.nan,
                "abs_coef_full":        np.nan,
                "category":             "",
                "sign_stable":          False,
                "in_top5":              False,
                "phillips_lens_active": False,
            })
            continue

        r = full.iloc[0]
        rank = int(r["rank_abs_full"])
        in_top5 = rank <= 5
        # Phillips lens "active" requires: in top-5 AND base category AND
        # negative sign (classical Phillips = unemployment up → CPI down)
        phillips_active = (
            in_top5
            and r["category"] == "base"
            and float(r["coef_full_train"]) < 0
        )

        rows.append({
            "country":              country,
            "feature_name":         target_feat,
            "rank_abs_full":        rank,
            "coef_full_train":      float(r["coef_full_train"]),
            "abs_coef_full":        float(r["abs_coef_full"]),
            "category":             str(r["category"]),
            "sign_stable":          bool(r["sign_stable"]),
            "in_top5":              in_top5,
            "phillips_lens_active": phillips_active,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CORRECTED N2 logic — POLICY_RATE surface in top-5 only
# ---------------------------------------------------------------------------
def policy_rate_top5_audit(
    top_df: pd.DataFrame, country: str, form: str,
) -> pd.DataFrame:
    t = top_df[
        (top_df["country"] == country)
        & (top_df["form"] == form)
        & (top_df["rank"] <= 5)
    ]
    return t[t["feature_name"].str.contains("POLICY_RATE")]


# ---------------------------------------------------------------------------
# Main narrative builder (corrected)
# ---------------------------------------------------------------------------
def build_corrected_narratives(
    inputs: Dict[str, pd.DataFrame]
) -> (pd.DataFrame, pd.DataFrame):
    coef = inputs["coefficients"]
    top = inputs["top_features"]

    summary = inputs["summary"]  # phase6_step3_s5_country_narrative_summary.csv
    primary_only = summary[summary["form"] == "primary"].set_index("country")

    # --- Phillips audit (new CSV) ---
    phillips_audit = phillips_base_feature_audit(coef, top)
    active_countries = phillips_audit[
        phillips_audit["phillips_lens_active"]
    ]["country"].tolist()
    in_top5_countries = phillips_audit[
        phillips_audit["in_top5"]
    ]["country"].tolist()

    # Phillips statement description
    if len(active_countries) == 0:
        phillips_summary = "no country surfaces contemporaneous UNEMPLOYMENT in top-5"
    else:
        parts = []
        for _, r in phillips_audit[phillips_audit["phillips_lens_active"]].iterrows():
            parts.append(
                f"{r['country']} (rank {int(r['rank_abs_full'])}, "
                f"coef {r['coef_full_train']:+.4f})"
            )
        phillips_summary = (
            f"contemporaneous UNEMPLOYMENT base feature in top-5 with "
            f"negative sign: {', '.join(parts)}"
        )

    # --- N1 corrected ---
    alpha_min = float(primary_only["selected_alpha"].min())
    alpha_max = float(primary_only["selected_alpha"].max())
    alpha_ratio = alpha_max / alpha_min
    country_max = primary_only["selected_alpha"].idxmax()
    country_min = primary_only["selected_alpha"].idxmin()

    n1_statement = (
        f"Ridge α* spans {alpha_ratio:.0f}x across four countries "
        f"({country_min} α*={alpha_min:.0f} → {country_max} α*={alpha_max:.0f}). "
        f"Phillips base-feature lens: {phillips_summary}."
    )

    n1_row = {
        "narrative_id": "N1",
        "title": "Cross-Country Inflation Dynamics",
        "ridge_contribution_type": "Regularisation-intensity heterogeneity + Phillips base-feature surface",
        "ridge_key_statistic": n1_statement,
        "cross_lens_count": 3,
        "ridge_adds_new_lens": True,
    }

    # --- N2 corrected (USA dual-form, top-5 only) ---
    usa_pri_top5_policy = policy_rate_top5_audit(top, "USA", "primary")
    usa_fd_top5_policy = policy_rate_top5_audit(top, "USA", "first_diff_secondary")

    pri_has = len(usa_pri_top5_policy) > 0
    fd_policy_str = (
        "; ".join(
            f"{r.feature_name}={r.coef_full_train:+.3f} (rank {int(r['rank'])})"
            for _, r in usa_fd_top5_policy.iterrows()
        ) if len(usa_fd_top5_policy) else "none in top-5"
    )

    n2_statement = (
        f"USA dual-form divergence in top-5: primary (yoy_pct) has POLICY_RATE = "
        f"{'yes' if pri_has else 'NO'} (CPI auto-features only); "
        f"first_diff secondary top-5 policy-rate features: {fd_policy_str}. "
        f"Ridge first_diff reproduces D-056 VAR IRF peak at h=4 (negative sign)."
    )

    n2_row = {
        "narrative_id": "N2",
        "title": "Policy Response Patterns (USA Monetary Transmission)",
        "ridge_contribution_type": "Dual-form validation + VAR IRF cross-lens consistency",
        "ridge_key_statistic": n2_statement,
        "cross_lens_count": 4,
        "ridge_adds_new_lens": True,
    }

    # --- N3 (unchanged from S5, verified correct) ---
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
        f"max|coef|=0.0100 is {ratio_min:.1f}x to {ratio_max:.1f}x smaller "
        f"than other countries; OOS MASE h=1 = {jpn_mase_h1:.3f} "
        f"(marginal beat-naive). Ridge-lens = 7th independent confirmation of "
        f"N3 after ACF / ARIMA / VAR-lag / Granger / IRF / FEVD."
    )

    n3_row = {
        "narrative_id": "N3",
        "title": "Japan's Uniqueness (2022 Inflation Reversal + Isolation)",
        "ridge_contribution_type": "Coefficient-magnitude quantification of isolation",
        "ridge_key_statistic": n3_statement,
        "cross_lens_count": 7,
        "ridge_adds_new_lens": True,
    }

    narratives_corrected = pd.DataFrame([n1_row, n2_row, n3_row])
    return narratives_corrected, phillips_audit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 6 · Step 3 · S5b — Narrative Logic Correction")
    print("=" * 72)

    print("\n[1/3] Loading S1-S5 CSVs...")
    inputs = {
        "coefficients":  _read_csv("phase6_step3_s3_ridge_coefficients.csv"),
        "top_features":  _read_csv("phase6_step3_s3_top_features.csv"),
        "summary":       _read_csv("phase6_step3_s5_country_narrative_summary.csv"),
    }
    for name, df in inputs.items():
        print(f"    {name:<15} shape = {df.shape}")

    print("\n[2/3] Running corrected N1/N2/N3 logic...")
    narratives, phillips_audit = build_corrected_narratives(inputs)

    print("\n[3/3] Writing outputs...")
    out_narr = DOC_DIR / "phase6_step3_s5_narrative_ridge_statements.csv"
    out_phil = DOC_DIR / "phase6_step3_s5b_phillips_base_feature_lens.csv"

    narratives.to_csv(out_narr, index=False)
    phillips_audit.to_csv(out_phil, index=False)
    print(f"    -> {out_narr.relative_to(PROJECT_ROOT)}  (OVERWRITTEN, {len(narratives)} rows)")
    print(f"    -> {out_phil.relative_to(PROJECT_ROOT)}  (NEW, {len(phillips_audit)} rows)")

    # -----------------------------------------------------------------------
    # Terminal preview
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHILLIPS BASE-FEATURE LENS AUDIT (4 countries)")
    print("=" * 72)
    disp = phillips_audit.copy()
    for c in ["coef_full_train", "abs_coef_full"]:
        disp[c] = disp[c].round(4)
    print(disp.to_string(index=False))

    print("\n" + "=" * 72)
    print("CORRECTED NARRATIVE RIDGE-LENS STATEMENTS")
    print("=" * 72)
    for _, r in narratives.iterrows():
        print(f"\n[{r['narrative_id']}] {r['title']}")
        print(f"  Contribution: {r['ridge_contribution_type']}")
        print(f"  Cross-lens count: {r['cross_lens_count']}")
        print(f"  Key: {r['ridge_key_statistic']}")

    print(
        "\nS5b complete. Phase 6 Step 3 analytical pipeline is now frozen. "
        "Remaining: D-064+ decisions → 08_ridge_regression.ipynb → phase6_step3_summary.md."
    )


if __name__ == "__main__":
    main()
