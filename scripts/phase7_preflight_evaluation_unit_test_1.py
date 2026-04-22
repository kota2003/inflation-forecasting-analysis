"""
scripts/phase7_preflight_evaluation_unit_test.py

Acceptance gate for ``src/evaluation.py`` v0.4.3 (D-076).

Verifies:
    1. Package imports cleanly at v0.4.3 and all 10 new exports are present.
    2. Harvey-Leybourne-Newbold correction matches textbook values.
    3. Loss primitives (RMSE, MAE, MASE) are correct on toy data.
    4. Diebold-Mariano variants behave sensibly:
       - identical forecasts → (0.0, 1.0)
       - clearly dominated forecast → large-magnitude stat, p << 0.05
       - HAC at h=1 agrees with standard up to (T-1)/T divisor scaling
       - robust is demonstrably different from standard under outlier
    5. CSV adapters load all three Phase 6 OOS files with the unified schema.
    6. align_matched_terms() successfully pairs the three layers on
       (country, form, h, target_date) with y_true agreement within 1e-6.
    7. End-to-end smoke: USA primary h=1 ARIMA-vs-VAR, ARIMA-vs-Ridge,
       VAR-vs-Ridge DM each run and return finite (dm_stat, p_value).

Stdout is structured as a set of ``[pass] …`` / ``[FAIL] …`` lines
followed by a final verdict. Exit code 0 iff all tests pass.

Decision linkage: D-048, D-051, D-060, D-061, D-068, D-074, D-075, D-076.
"""

from __future__ import annotations

import sys
import traceback
from math import sqrt
from pathlib import Path

import numpy as np

# ── Ensure src/ is importable regardless of CWD ────────────────────
try:
    from src import __version__ as src_version
    from src import (
        HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT,
        rmse, mae, mase,
        diebold_mariano_standard, diebold_mariano_hac, diebold_mariano_robust,
        load_phase6_forecasts, align_matched_terms,
        UNIFIED_SCHEMA_COLUMNS,
        VAR_MASE_D060,
    )
except ImportError as e:
    # Add project root to sys.path
    here = Path(__file__).resolve().parent
    root = here.parent
    sys.path.insert(0, str(root))
    from src import __version__ as src_version
    from src import (
        HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT,
        rmse, mae, mase,
        diebold_mariano_standard, diebold_mariano_hac, diebold_mariano_robust,
        load_phase6_forecasts, align_matched_terms,
        UNIFIED_SCHEMA_COLUMNS,
        VAR_MASE_D060,
    )


RESULTS: list[tuple[str, bool, str]] = []
SEP = "=" * 76
SUB = "-" * 76


def check(label: str, condition: bool, detail: str = "") -> None:
    tag = "pass" if condition else "FAIL"
    RESULTS.append((label, condition, detail))
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print()
    print(SUB)
    print(title)
    print(SUB)


# ──────────────────────────────────────────────────────────────────────
# Test 1: Package metadata
# ──────────────────────────────────────────────────────────────────────
section("Test 1 · Package version + export surface")

check("src.__version__ == '0.4.3'", src_version == "0.4.3",
      detail=f"actual={src_version!r}")
check("UNIFIED_SCHEMA_COLUMNS has expected 7 fields",
      UNIFIED_SCHEMA_COLUMNS == ("country", "form", "h", "origin_date",
                                 "target_date", "y_true", "y_pred"),
      detail=str(UNIFIED_SCHEMA_COLUMNS))


# ──────────────────────────────────────────────────────────────────────
# Test 2: HLN correction
# ──────────────────────────────────────────────────────────────────────
section("Test 2 · Harvey-Leybourne-Newbold small-sample correction")

# HLN(T, 1) = sqrt((T - 1) / T). At T=58 → sqrt(57/58) ≈ 0.991342.
T, h = 58, 1
expected = sqrt((T - 1) / T)
actual = HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT(T, h)
check(f"HLN(T={T}, h={h}) = sqrt((T-1)/T) ≈ {expected:.6f}",
      abs(actual - expected) < 1e-10,
      detail=f"actual={actual:.10f}")

# HLN(T, h) full formula spot check: T=100, h=3
T, h = 100, 3
expected = sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
actual = HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT(T, h)
check(f"HLN(T={T}, h={h}) = sqrt({(T + 1 - 2*h + h*(h-1)/T)/T:.6f}) ≈ {expected:.6f}",
      abs(actual - expected) < 1e-10,
      detail=f"actual={actual:.10f}")

# HLN is monotone decreasing in h (for fixed T)
vals = [HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT(60, k) for k in (1, 3, 6, 12)]
check("HLN(60, h) strictly decreasing in h ∈ {1,3,6,12}",
      all(vals[i] > vals[i+1] for i in range(3)),
      detail=f"h=1..12: {[round(v,4) for v in vals]}")


# ──────────────────────────────────────────────────────────────────────
# Test 3: Loss primitives
# ──────────────────────────────────────────────────────────────────────
section("Test 3 · Loss primitives (RMSE, MAE, MASE)")

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
# Errors: [0.1, -0.1, 0.2, -0.2, 0.1]  →  RMSE = sqrt((0.01+0.01+0.04+0.04+0.01)/5) = sqrt(0.022)
# MAE   = (0.1+0.1+0.2+0.2+0.1)/5 = 0.14
expected_rmse = sqrt(0.022)
expected_mae = 0.14
check("rmse() matches hand-computed value on 5-point toy series",
      abs(rmse(y_true, y_pred) - expected_rmse) < 1e-10,
      detail=f"expected={expected_rmse:.6f}, actual={rmse(y_true,y_pred):.6f}")
check("mae() matches hand-computed value",
      abs(mae(y_true, y_pred) - expected_mae) < 1e-10,
      detail=f"expected={expected_mae:.6f}, actual={mae(y_true,y_pred):.6f}")
check("mase(scale=0.5) = MAE / 0.5",
      abs(mase(y_true, y_pred, 0.5) - expected_mae / 0.5) < 1e-10,
      detail=f"expected={expected_mae/0.5:.6f}, actual={mase(y_true,y_pred,0.5):.6f}")

# MASE rejects invalid scale
try:
    mase(y_true, y_pred, 0.0)
    check("mase() rejects scale=0", False, "no ValueError raised")
except ValueError:
    check("mase() rejects scale=0", True)
try:
    mase(y_true, y_pred, -1.0)
    check("mase() rejects negative scale", False, "no ValueError raised")
except ValueError:
    check("mase() rejects negative scale", True)


# ──────────────────────────────────────────────────────────────────────
# Test 4: Diebold-Mariano behaviour (synthetic)
# ──────────────────────────────────────────────────────────────────────
section("Test 4 · Diebold-Mariano variants (synthetic)")

rng = np.random.default_rng(seed=42)
e_base = rng.normal(scale=1.0, size=60)

# 4.1 Identical errors → (0.0, 1.0)
dm, p = diebold_mariano_standard(e_base, e_base, h=1)
check("DM standard(e, e) = (0.0, 1.0)",
      dm == 0.0 and p == 1.0,
      detail=f"dm={dm}, p={p}")
dm, p = diebold_mariano_hac(e_base, e_base, h=3)
check("DM HAC(e, e, h=3) = (0.0, 1.0)",
      dm == 0.0 and p == 1.0,
      detail=f"dm={dm}, p={p}")
dm, p = diebold_mariano_robust(e_base, e_base, h=1)
check("DM robust(e, e) = (0.0, 1.0)",
      dm == 0.0 and p == 1.0,
      detail=f"dm={dm}, p={p}")

# 4.2 Clearly worse e2 → large |stat|, p << 0.05, negative sign
e_worse = 2.0 * e_base + rng.normal(scale=0.1, size=60)
dm, p = diebold_mariano_standard(e_base, e_worse, h=1)
check("DM standard(e, 2e+ε) detects dominance: |dm|>2 and p<0.05 and dm<0",
      abs(dm) > 2.0 and p < 0.05 and dm < 0.0,
      detail=f"dm={dm:.4f}, p={p:.6f}")

# 4.3 HAC at h=1, n_lags=0 reduces to γ_0 (sample second moment / T),
#     whereas standard uses sample variance (ddof=1) = γ_0 * T/(T-1).
#     So HAC and standard DM at h=1 differ only by sqrt((T-1)/T) ≈ 1.
dm_std, _ = diebold_mariano_standard(e_base, e_worse, h=1)
dm_hac, _ = diebold_mariano_hac(e_base, e_worse, h=1)
ratio = dm_hac / dm_std
expected_ratio = sqrt(60 / 59)
check(f"DM HAC(h=1) / DM standard(h=1) ≈ sqrt(T/(T-1)) = {expected_ratio:.6f}",
      abs(ratio - expected_ratio) < 1e-6,
      detail=f"actual ratio={ratio:.6f}")

# 4.4 Robust DM under a single extreme outlier: standard loses power
#     (outlier-inflated variance), robust retains rejection power.
e_outlier = e_base.copy()
e_outlier[0] = 1000.0
dm_std_o, p_std_o = diebold_mariano_standard(e_outlier, e_worse, h=1)
dm_rob_o, p_rob_o = diebold_mariano_robust(e_outlier, e_worse, h=1)
# Robust should declare e_worse lower-loss (dm > 0, model 1 with outlier is worse)
check("DM robust under outlier: stat > 0 (outlier model is worse)",
      dm_rob_o > 0.0,
      detail=f"robust dm={dm_rob_o:.4f}, p={p_rob_o:.4f}")
# Standard power should be reduced by the outlier (higher p than robust typical)
check("DM standard vs robust under outlier: behaviour differs",
      abs(dm_std_o - dm_rob_o) > 1e-3 or abs(p_std_o - p_rob_o) > 1e-3,
      detail=f"std=({dm_std_o:.4f}, p={p_std_o:.4f}), "
             f"rob=({dm_rob_o:.4f}, p={p_rob_o:.4f})")


# ──────────────────────────────────────────────────────────────────────
# Test 5: CSV adapter — schema conformance
# ──────────────────────────────────────────────────────────────────────
section("Test 5 · CSV adapters (load_phase6_forecasts)")

try:
    df_arima = load_phase6_forecasts("arima")
    df_var   = load_phase6_forecasts("var")
    df_ridge = load_phase6_forecasts("ridge")
except Exception as e:
    print(f"  [FATAL] Could not load Phase 6 CSVs: {e}")
    traceback.print_exc()
    sys.exit(1)

for name, df in [("ARIMA", df_arima), ("VAR", df_var), ("Ridge", df_ridge)]:
    check(f"{name} DataFrame has UNIFIED_SCHEMA_COLUMNS",
          list(df.columns) == list(UNIFIED_SCHEMA_COLUMNS),
          detail=f"shape={df.shape}, columns={list(df.columns)}")
    check(f"{name} 'country' values are UPPERCASE country names",
          set(df["country"].unique()).issubset({"USA", "JAPAN", "UK", "GERMANY"}),
          detail=f"countries={sorted(df['country'].unique().tolist())}")
    check(f"{name} 'form' values in {{primary, secondary}}",
          set(df["form"].unique()).issubset({"primary", "secondary"}),
          detail=f"forms={sorted(df['form'].unique().tolist())}")
    check(f"{name} 'h' values subset of {{1, 3, 6, 12}}",
          set(df["h"].unique()).issubset({1, 3, 6, 12}),
          detail=f"horizons={sorted(df['h'].unique().tolist())}")
    check(f"{name} origin_date / target_date are datetime64",
          all(str(df[c].dtype).startswith("datetime64") for c in ("origin_date", "target_date")),
          detail=f"origin_date={df['origin_date'].dtype}, target_date={df['target_date'].dtype}")
    check(f"{name} y_true + y_pred have no NaN",
          not df[["y_true", "y_pred"]].isna().any().any(),
          detail=f"nan_counts={df[['y_true','y_pred']].isna().sum().to_dict()}")

# ARIMA specifics: only h=1, only {USA primary, USA secondary, JAPAN primary,
# UK primary, GERMANY primary}
check("ARIMA is h=1 only (per D-048)",
      (df_arima["h"] == 1).all(),
      detail=f"unique h={sorted(df_arima['h'].unique().tolist())}")
check("ARIMA has USA secondary (first_diff per D-048)",
      ((df_arima["country"] == "USA") & (df_arima["form"] == "secondary")).any())

# VAR specifics: primary form only (D-031)
check("VAR has primary form only (D-031)",
      (df_var["form"] == "primary").all(),
      detail=f"unique form={sorted(df_var['form'].unique().tolist())}")

# Ridge specifics: USA has both primary and secondary after schema
# normalisation (`_RIDGE_FORM_MAP` collapses `first_diff_secondary` →
# `secondary` for consistency with D-048 / D-064 / D-071 role labels).
usa_ridge_forms = set(df_ridge.loc[df_ridge["country"] == "USA", "form"].unique())
check("Ridge USA has both primary and secondary (D-071 dual-form)",
      usa_ridge_forms == {"primary", "secondary"},
      detail=f"USA forms={sorted(usa_ridge_forms)}")
# The raw compound label MUST NOT leak into the unified output.
check("Ridge form column does not contain raw 'first_diff_secondary'",
      "first_diff_secondary" not in set(df_ridge["form"].unique()),
      detail=f"all forms after normalisation={sorted(df_ridge['form'].unique())}")


# ──────────────────────────────────────────────────────────────────────
# Test 6: Walk-forward origin integrity via target_date match
# ──────────────────────────────────────────────────────────────────────
section("Test 6 · Matched-terms alignment (align_matched_terms)")

# 6.1 VAR ↔ Ridge at h=1, primary — should match 4 × 58 or 51 rows
try:
    y, e_var, e_ridge = align_matched_terms(
        df_var[(df_var["h"] == 1) & (df_var["form"] == "primary")],
        df_ridge[(df_ridge["h"] == 1) & (df_ridge["form"] == "primary")],
    )
    n_match = len(y)
    check(f"VAR vs Ridge (h=1, primary) alignment: {n_match} paired rows",
          n_match > 0 and n_match == (58 + 58 + 51 + 51),
          detail=f"expected 218, actual {n_match}")
except Exception as e:
    check("VAR vs Ridge (h=1, primary) alignment", False, detail=f"raised {type(e).__name__}: {e}")

# 6.2 ARIMA ↔ VAR (h=1, primary) — ARIMA's 2020-01 target has no VAR
#     counterpart (VAR first target is 2020-02), so intersect < ARIMA n.
try:
    y, e_a, e_v = align_matched_terms(
        df_arima[df_arima["form"] == "primary"],
        df_var[(df_var["h"] == 1) & (df_var["form"] == "primary")],
    )
    check(f"ARIMA vs VAR (h=1, primary) alignment: {len(y)} paired rows",
          len(y) > 0,
          detail=f"paired={len(y)}")
except Exception as e:
    check("ARIMA vs VAR (h=1, primary) alignment", False, detail=f"raised {type(e).__name__}: {e}")

# 6.3 ARIMA ↔ Ridge (h=1, primary) — same logic
try:
    y, e_a, e_r = align_matched_terms(
        df_arima[df_arima["form"] == "primary"],
        df_ridge[(df_ridge["h"] == 1) & (df_ridge["form"] == "primary")],
    )
    check(f"ARIMA vs Ridge (h=1, primary) alignment: {len(y)} paired rows",
          len(y) > 0,
          detail=f"paired={len(y)}")
except Exception as e:
    check("ARIMA vs Ridge (h=1, primary) alignment", False, detail=f"raised {type(e).__name__}: {e}")

# 6.3b ARIMA ↔ Ridge (h=1, USA secondary) — the D-071 dual-form DM
# pair. Requires _RIDGE_FORM_MAP normalisation of
# `first_diff_secondary` → `secondary` to succeed. If this is 0 rows,
# the Ridge form normalisation is not being applied.
try:
    y, e_a, e_r = align_matched_terms(
        df_arima[(df_arima["country"] == "USA") & (df_arima["form"] == "secondary")],
        df_ridge[(df_ridge["country"] == "USA") & (df_ridge["h"] == 1) & (df_ridge["form"] == "secondary")],
    )
    check(f"ARIMA vs Ridge (h=1, USA secondary): {len(y)} paired rows (D-071 dual-form)",
          len(y) > 0,
          detail=f"paired={len(y)}; if 0, check _RIDGE_FORM_MAP normalisation")
except Exception as e:
    check("ARIMA vs Ridge (h=1, USA secondary) alignment", False,
          detail=f"raised {type(e).__name__}: {e}")

# 6.4 y_true tolerance violation detection (negative test)
df_var_perturbed = df_var.copy()
df_var_perturbed.loc[df_var_perturbed.index[0], "y_true"] += 1.0  # perturb one row
try:
    _ = align_matched_terms(
        df_var_perturbed[(df_var_perturbed["h"] == 1) & (df_var_perturbed["form"] == "primary")],
        df_ridge[(df_ridge["h"] == 1) & (df_ridge["form"] == "primary")],
    )
    check("align_matched_terms() raises on y_true mismatch", False,
          detail="expected ValueError, got silent success")
except ValueError:
    check("align_matched_terms() raises on y_true mismatch", True)


# ──────────────────────────────────────────────────────────────────────
# Test 7: End-to-end DM smoke test on real USA primary h=1 data
# ──────────────────────────────────────────────────────────────────────
section("Test 7 · End-to-end DM smoke test (USA primary h=1)")

usa_arima = df_arima[(df_arima["country"] == "USA") & (df_arima["form"] == "primary")]
usa_var   = df_var[(df_var["country"] == "USA") & (df_var["form"] == "primary") & (df_var["h"] == 1)]
usa_ridge = df_ridge[(df_ridge["country"] == "USA") & (df_ridge["form"] == "primary") & (df_ridge["h"] == 1)]

print(f"  USA primary rows: ARIMA={len(usa_arima)}, VAR={len(usa_var)}, Ridge={len(usa_ridge)}")

for (name1, df1), (name2, df2) in [
    (("ARIMA", usa_arima), ("VAR",   usa_var)),
    (("ARIMA", usa_arima), ("Ridge", usa_ridge)),
    (("VAR",   usa_var),   ("Ridge", usa_ridge)),
]:
    y, e1, e2 = align_matched_terms(df1, df2)
    dm_s, p_s = diebold_mariano_standard(e1, e2, h=1)
    dm_h, p_h = diebold_mariano_hac(e1, e2, h=1)
    dm_r, p_r = diebold_mariano_robust(e1, e2, h=1)
    label = f"{name1} vs {name2} (T={len(y)})"
    ok = all(np.isfinite(x) for x in (dm_s, p_s, dm_h, p_h, dm_r, p_r))
    check(f"{label}: all three DM variants return finite values",
          ok,
          detail=f"std=({dm_s:+.3f}, p={p_s:.3f}) "
                 f"hac=({dm_h:+.3f}, p={p_h:.3f}) "
                 f"rob=({dm_r:+.3f}, p={p_r:.3f})")


# ──────────────────────────────────────────────────────────────────────
# Final verdict
# ──────────────────────────────────────────────────────────────────────
print()
print(SEP)
n_total = len(RESULTS)
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
n_fail = n_total - n_pass
print(f"Summary: {n_pass}/{n_total} passed, {n_fail} failed")
if n_fail:
    print()
    print("Failures:")
    for label, ok, detail in RESULTS:
        if not ok:
            print(f"  - {label}  — {detail}")
    print(SEP)
    sys.exit(1)
print("All tests passed. src/evaluation.py v0.4.3 is acceptance-ready.")
print(SEP)
