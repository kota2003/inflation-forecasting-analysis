# Inflation Prediction and Economic Signal Analysis

A multi-country econometric study forecasting consumer price inflation across USA, Japan, UK, and Germany (2000-01 to present) using ARIMA, VAR, and Ridge Regression. The analysis combines classical time-series rigour with modern data-engineering practice and documents every design decision in a living decision log.

This is **Portfolio Project 3 (P3)** in a three-project series. P1 demonstrated machine-learning engineering on structured customer data; P2 covered feature engineering and classification with interpretability tools; this P3 demonstrates **classical econometric rigour combined with modern data engineering**. The project deliberately prioritises decision documentation, source   auditing, and reproducibility over algorithmic novelty — the skills most valued in consulting contexts where analytical defensibility matters more than headline accuracy.

---

## Project Status

| Phase | Focus | Status |
|---|---|---|
| Phase 0 | Project scoping, country selection, narrative definition | ✅ Complete |
| Phase 1 | Data collection — 25 series, 5 countries × 5 indicators, multi-source rebuild | ✅ Complete |
| Phase 2 | Data cleaning, unit harmonisation, temporal alignment | ✅ Complete |
| Phase 3 | Stationarity testing (ADF+KPSS), structural-break testing (Chow, Quandt-Andrews) | ✅ Complete |
| Phase 4 | Feature engineering (lags, rolling statistics, regime dummies) | ✅ Complete |
| Phase 5 | Exploratory data analysis & cross-country narrative visualisation | ✅ Complete |
| Phase 6 | Model estimation — ARIMA, VAR, Ridge | ⏳ **Step 2 of 3 complete** |
| Phase 7 | Evaluation — Diebold-Mariano, walk-forward validation | Pending |
| Phase 8 | Interpretation — Granger maps, IRF plots, narrative synthesis | Pending |

As of this writing, the project has completed the five-phase analytical foundation through exploratory data analysis plus the first two of three modelling layers in Phase 6. The `data/processed/` directory contains four main-country feature matrices of 50–53 columns each (USA, Japan, UK, Germany — D-031-corrected stationary forms with lags, rolling statistics, and regime dummies) plus a supplementary China dataset. Phase 3 classified every series' stationarity status and characterised three pre-specified structural breaks (2008-09, 2020-03, 2022-02) via Chow and Quandt-Andrews tests. Phase 4 built the per-country feature matrices (joint-valid from 2002-02 or 2003-01) via `src.feature_engineering`. Phase 5 produced eight portfolio figures and twelve audit CSVs spanning cross-country CPI dynamics, correlation structure, N1 Phillips Curve deep-dive, and ACF/PACF diagnostics — with seven signature findings flagged in `ProjectDriven.md` that directly informed the Phase 6 ARIMA/VAR/Ridge estimation. **Phase 6 Step 1 (Layer 1 SARIMA baseline) is complete**: five variants estimated via a three-stage grid search protocol (450-order initial grid • boundary sensitivity check • targeted Q = 3 extension), 8 portfolio figures delivered in `notebooks/06_arima_baseline.ipynb`, and two decisions recorded (D-048 three-stage protocol with OOS-saturation stopping rule; D-049 Japan ARIMA uniqueness as an N3 narrative echo at the ARIMA layer). **Phase 6 Step 2 (Layer 2 VAR core model) is now complete**: four country-specific VARs estimated across nine scratch orchestrators (S1 / S1b / S2 / S2b / S3 / S4 / S5 / S6 / S6b) covering lag selection with boundary sensitivity, BIC→AIC criterion revision via residual diagnostics, 5×5 Granger causality battery (100 tests), orthogonalised IRFs with asymptotic 95 % CIs, Forecast Error Variance Decomposition, and walk-forward out-of-sample forecasts for Phase 7 Diebold-Mariano; 25 audit CSVs written to `data/documentation/phase6_step2_*.csv`; 14 new decisions logged (D-050 through D-063) covering the VAR lag-selection protocol, Cholesky ordering, statsmodels API robustness patches, the inference-primary-vs-forecast-auxiliary positioning, and the Step 2 closeout `src/modelling_utils` promotion at v0.4.1. Phase 6 Step 3 (Layer 3 Ridge regression) is the immediate next work; `notebooks/07_var_model.ipynb` consolidates the Step 2 narrative in parallel.

---

## Three Narratives — State through Phase 6 Step 2

The project is organised around three named economic narratives (ProjectScope §4). Phase 5 EDA supplied the single-number correlational evidence; Phase 6 Step 1 added ARIMA residual-structure evidence; Phase 6 Step 2 now supplies the multivariate causal / directional interpretation via Granger tests, IRFs, and FEVD.

- **N1 · Phillips Curve — Anglo-specific, methodology-dependent.** Phase 5 rolling slopes reach |β| ≈ 5–9 with R² ∈ [0.60, 0.75] post-2022; UK alone shows a pre/post-GFC sign flip (β = +1.68 → −0.27). Phase 6 Step 2 Granger tests localise significance to the Anglo axis (UK p = 0.002 ★★; USA p = 0.017 ★; Japan p = 0.39; Germany p = 0.30). Step 2 IRFs further deliver an **unexpected positive sign** for UK and USA UE → CPI peak responses (+0.042 at h = 1 and +0.267 at h = 5 respectively) — a stagflation-era co-movement echo that regime dummies do not fully absorb. FEVD at h = 12 corroborates: USA unemployment explains 26.8 % of CPI variance (#2 driver), UK 6.7 %, Japan and Germany below 4 %.
- **N2 · Monetary Policy Lag — USA-specific via the interest-rate channel, not the money-supply channel.** Phase 5 previewed USA `corr(CPI_t, M2_{t−12}) = +0.41` as a Quantity Theory signature. Phase 6 Step 2 definitively separates the two channels: the **interest-rate channel** survives only in USA (Granger p = 0.004 ★★; IRF peak −0.149 at h = 4 with CI excluding zero; FEVD share growing to 14.1 % at h = 24 — five times the share in any other country), while the **money-supply channel (M2 → CPI) is universally null** across Granger (p > 0.12 all four countries), IRF (all CIs straddle zero), and FEVD (< 5 % share at every horizon, every country). N2 consequently survives as a USA-anchored narrative rather than a universal monetary-economic claim.
- **N3 · Japan's Uniqueness — sextuple-confirmed structural isolation.** (1) Level peer-gap: Japan CPI YoY is below the mean of USA/UK/Germany in 253 of 279 monthly observations (90.7 %; mean gap −1.80 pp). (2) Phase monotone progression: Deflation −0.20 % → Abenomics +0.64 % → Reversal +2.99 % (Reversal phase shows 0 / 45 deflation months). Phase 5 S1 (Fig 1, Fig 2). (3) **ARIMA simplicity**: Japan is the only variant among five Step 1 candidates with triple AIC / BIC / HQIC agreement on a 4-parameter sparse order and ARCH-LM p ≈ 0.9999 (D-049). (4) **VAR lag interior minimum**: Japan is the only country whose AIC argmin (p* = 5) is interior rather than at the maxlag = 12 boundary (D-050 / S1b extension confirms). (5) **Granger null across all causers**: 0 / 4 Granger tests for CPI-receivers are significant in Japan (p ∈ {0.21, 0.42, 0.39, 0.26}), while Japan's VAR shows active causation elsewhere (GDP → UE p = 0.003; M2 → GDP p = 0.000) — causation structure exists, but none flows to CPI. (6) **FEVD self-share plateau**: Japan CPI's own-shock share is 92.1 % at h = 12 and 92.0 % at h = 24 — no external driver ever catches up. This sextuple confirmation across six independent inferential lenses establishes N3 as the project's most robust signature finding.

Five methodology findings are recorded separately in the decision log and shape how the three narratives are reported:

- **D-046** — the Phillips Curve is visible in level-based EDA (Phase 5 S3) but invisible under stationary-form correlation (S2). This is a deliberate methodology choice, not an inconsistency: the classical Phillips Curve is a level relationship, and both lenses play legitimate roles across Phase 5 (EDA), Phase 6 Step 1 (ARIMA), and Phase 6 Step 2 (VAR).
- **D-048 stopping rule** — AIC in-sample improvement from SARIMA grid extension does not translate to out-of-sample performance at the orders evaluated in Phase 6 Step 1; USA_first_diff Stage (a) → Stage (c) ΔAIC = −10.46 produced OOS RMSE Δ = −0.003. OOS saturation is adopted as the principled stopping criterion, obligating Phase 7 Diebold-Mariano to compare OOS loss differentials rather than AIC rankings.
- **D-057 Phillips Methodology Trilogy** — the Phillips Curve manifests in three mutually compatible but visually different ways across analytical lenses: classical negative slope in level form (Phase 5 D-043), invisibility in stationary correlation (D-046), and positive sign in stationary IRF for Anglo countries (D-057). The phenomenon is real but lens-dependent — a project-centerpiece methodology meta-finding rather than a contradiction.
- **D-058 Four-Lens Disconfirmation of Quantity Theory of Money** — Phase 5's +0.41 USA CPI-M2 cross-lag correlation fails across four independent Phase 6 inferential lenses (Granger, IRF, FEVD, cross-country consistency). This definitive negative result extends D-046's methodology asymmetry into a generalised principle: **correlation in stationary form does not imply any form of inferential causation**.
- **D-060 VAR inference-primary vs forecast-auxiliary positioning** — the D-050 revision from BIC p = 2 to AIC-selected p per country optimised for residual whitening and inference quality at the explicit cost of OOS forecast accuracy. Only Japan VAR(5) beats the random-walk naive benchmark in aggregate MASE; USA, UK, and Germany under-perform. Under robust median-absolute-error metric (D-061 / S6b), Japan wins at every horizon and UK / Germany are near-competitive. The portfolio records VAR as "inference-primary, forecast-auxiliary" in the three-layer architecture — Phase 7 DM battery will evaluate ARIMA / VAR / Ridge on matched terms.

---

## Repository Structure

```
inflation-forecasting-analysis/
├── data/
│   ├── raw/                                      # 25 source series (final Phase 1 v2 + D-021 state)
│   │   ├── {COUNTRY}_{INDICATOR}.csv            # 5 × 5 grid of country × indicator
│   │   ├── _archive_v1/{timestamp}/             # Phase 1 v1 state archived pre-rebuild
│   │   ├── _archive_d021/{timestamp}/           # Germany M2 placeholder archived pre-resolution
│   │   ├── _manual/                             # Manual government CSVs (Japan CPI)
│   │   └── UK_IP.csv                            # Retained from Chow-Lin due diligence
│   │
│   ├── processed/                                # Phase 2 + Phase 4 output
│   │   ├── main_usa.csv                         # 298 rows × 5 cols, 2001-01 to 2025-10
│   │   ├── main_japan.csv                       # 298 rows × 5 cols, 2001-01 to 2025-10
│   │   ├── main_uk.csv                          # 291 rows × 5 cols, 2001-01 to 2025-03
│   │   ├── main_germany.csv                     # 291 rows × 5 cols, 2001-01 to 2025-03
│   │   ├── supplementary_china.csv              # 300 rows × 5 cols, VAR-excluded
│   │   ├── schema.md                            # Auto-generated schema (Phase 2)
│   │   ├── features_usa.csv                     # Phase 4: 298 × 53
│   │   ├── features_japan.csv                   # Phase 4: 298 × 50
│   │   ├── features_uk.csv                      # Phase 4: 291 × 51
│   │   ├── features_germany.csv                 # Phase 4: 291 × 52
│   │   └── features_schema.md                   # Auto-generated feature schema
│   │
│   └── documentation/                            # Audit logs from every pipeline stage
│       ├── phase1v2_rebuild_log.csv
│       ├── phase2_cleaning_log.csv
│       ├── phase2_germany_m2_scout.csv
│       ├── phase2_m2_yoy_validation.csv
│       ├── phase2_ip_scout*.csv                 # Chow-Lin due diligence (rejected)
│       ├── phase3_adf_kpss_levels.csv           # 20 rows
│       ├── phase3_differencing_log.csv          # 16 rows
│       ├── phase3_conflict_ct_retest.csv        # 2 rows
│       ├── phase3_cpi_transform_comparison.csv  # 16 rows
│       ├── phase3_transformation_registry_final.csv  # 20 rows — D-027/D-031 source of truth
│       ├── phase3_subperiod_stationarity.csv    # 60 rows
│       ├── phase3_cpi_deep_dive.csv             # 36 rows
│       ├── phase3_break_window_stationarity.csv # 120 rows
│       ├── phase3_chow_tests_classical.csv      # 12 rows
│       ├── phase3_chow_tests_hac.csv            # 12 rows
│       ├── phase3_chow_tests_covid_dummy.csv    # 8 rows
│       ├── phase3_chow_coefficient_decomposition.csv # 60 rows
│       ├── phase3_chow_bonferroni_summary.csv   # 32 rows
│       ├── phase3_quandt_andrews_supwald.csv    # 4 rows — π₀ = 0.15
│       ├── phase3_quandt_andrews_curve.csv      # 815 rows — π₀ = 0.15 curve
│       ├── phase3_quandt_andrews_supwald_trim10.csv  # 4 rows — π₀ = 0.10
│       ├── phase3_quandt_andrews_curve_trim10.csv    # 933 rows — π₀ = 0.10 curve
│       ├── phase4_step1_effective_registry.csv  # 20 rows — D-031 applied
│       ├── phase4_step1_base_features_summary.csv    # 20 rows
│       ├── phase4_step1_base_features_preview.csv    # long-form head+tail
│       ├── phase4_step2_lag_{country}.csv       # × 4
│       ├── phase4_step2_lag_summary.csv         # 80 rows (first-valid-date proof)
│       ├── phase4_step3_rolling_{country}.csv   # × 4
│       ├── phase4_step3_rolling_summary.csv     # 80 rows (1e-10 spot check)
│       ├── phase4_step4_regime_{country}.csv    # × 4
│       ├── phase4_step4_regime_summary.csv      # 26 rows
│       ├── phase4_step4_regime_specification.csv # 12 rows — D-030 matrix echo
│       ├── phase4_step5_category_counts.csv
│       ├── phase4_step5_joint_valid_summary.csv
│       ├── phase4_step5_consistency_check.csv   # 12 rows — all passed
│       ├── phase5_step1_cpi_summary.csv         # 4 rows — per-country CPI stats
│       ├── phase5_step1_japan_phases.csv        # 3 rows — Japan phase decomposition (D-045)
│       ├── phase5_step1_japan_peer_gap.csv      # 1 row  — Japan vs 3-peer gap
│       ├── phase5_step2_base_correlation.csv    # 100 rows — base 5×5 per country
│       ├── phase5_step2_lag_correlation.csv     # 80 rows — cross-lag matrix
│       ├── phase5_step2_window_summary.csv      # 4 rows — joint-valid windows
│       ├── phase5_step3_phillips_fit.csv        # 12 rows — OLS {full, pre, post}
│       ├── phase5_step3_rolling_slope.csv       # 894 rows — 60-month rolling OLS
│       ├── phase5_step4_acf_pacf_values.csv     # 164 rows — ACF/PACF/CI per lag
│       ├── phase5_step4_ljung_box.csv           # 12 rows — Q(12/24/36) per country
│       ├── phase6_step1_arima_selection.csv     # 5 rows — AIC/BIC/HQIC best orders per variant
│       ├── phase6_step1_arima_residuals.csv     # 5 rows — Ljung-Box, JB, ARCH-LM diagnostics
│       ├── phase6_step1_arima_forecast.csv      # 340 rows — expanding-refit 1-step test forecasts
│       ├── phase6_step1_arima_window_errors.csv # 15 rows — RMSE/MAE/bias × 3 test sub-windows
│       ├── phase6_step1_arima_grid_*.csv        # 5 × 450 rows — Stage (a) grid per variant
│       ├── phase6_step1b_boundary_check_*.csv   # 4 files — Stage (b) verdicts + per-variant detail
│       ├── phase6_step1c_*.csv                  # 2 files — Stage (c) Q=3 extension + selection delta
│       ├── phase6_step2_var_lag_selection_*.csv         # 2 files — S1 IC grid per country + summary (D-050)
│       ├── phase6_step2_s1b_sensitivity_*.csv           # 2 files — S1b B&A threshold verdicts (D-050)
│       ├── phase6_step2_s2_var_*.csv                    # 5 files — BIC p=2 baseline + diagnostics
│       ├── phase6_step2_s2b_var_*.csv                   # 5 files — AIC refit + whiteness comparison (D-050, D-051)
│       ├── phase6_step2_s3_granger_*.csv                # 3 files — full matrix + CPI receivers + country summary (D-052)
│       ├── phase6_step2_s4_irf_*.csv                    # 3 files — full IRF + CPI-target + peak summary (D-054..D-057)
│       ├── phase6_step2_s5_fevd_*.csv                   # 4 files — full matrix + CPI-target + summary + top contributors (D-058, D-059)
│       ├── phase6_step2_s6_var_oos_*.csv                # 3 files — walk-forward forecasts + metrics + CPI summary (D-060)
│       └── phase6_step2_s6b_*.csv                       # 3 files — worst origins + robust metrics + CPI robust summary (D-061, D-062)
│
├── notebooks/                                    # Portfolio-grade narrative layer
│   ├── 01_data_collection.ipynb                 # Phase 1 collection + v2 rebuild
│   ├── 02_cleaning_alignment.ipynb              # Phase 2 unit harmonisation
│   ├── 03_stationarity_structural_breaks.ipynb  # Phase 3 ADF/KPSS + Chow/Q-A
│   ├── 04_feature_engineering.ipynb             # Phase 4 lag/rolling/regime assembly
│   ├── 05_eda.ipynb                             # Phase 5 cross-country narrative
│   ├── 06_arima_baseline.ipynb                  # Phase 6 Step 1 SARIMA baseline (D-048, D-049)
│   └── 07_var_model.ipynb                       # Phase 6 Step 2 VAR narrative (pending; D-050..D-062)
│
├── scripts/                                      # Scratch orchestrators (pre-notebook stage)
│   ├── phase1v2_candidate_scout.py              # Multi-source rebuild scout
│   ├── phase1v2_execute_rebuild.py              # Targeted series replacement
│   ├── phase3_step1_level_adf_kpss.py
│   ├── phase3_step2_differencing.py
│   ├── phase3_step3_cpi_registry.py
│   ├── phase3_step4_chow_structural_breaks.py
│   ├── phase3_step5_quandt_andrews.py           # π₀ = 0.15
│   ├── phase3_step5b_quandt_andrews_trim10.py   # π₀ = 0.10
│   ├── phase4_step1_base_features.py
│   ├── phase4_step2_lag_matrix.py
│   ├── phase4_step3_rolling_matrix.py
│   ├── phase4_step4_regime_dummies.py
│   ├── phase4_step5_assemble.py                 # Module-vs-scratch regression test
│   ├── phase5_step1_cpi_narrative.py            # Fig 1, Fig 2 + audit CSVs (D-041, D-045)
│   ├── phase5_step2_correlation_structure.py    # Fig 3, Fig 4, Fig 5 (D-042)
│   ├── phase5_step3_phillips_curve.py           # Fig 6, Fig 7 (D-043, D-046)
│   ├── phase5_step4_acf_pacf.py                 # Fig 8 (D-044)
│   ├── phase6_step1_arima_grid.py               # Stage (a) 2,250 grid fits (D-048)
│   ├── phase6_step1b_q3_boundary_check.py       # Stage (b) 22 boundary fits (D-048)
│   ├── phase6_step1c_usa_firstdiff_q3_extension.py  # Stage (c) 150 targeted fits (D-048)
│   ├── phase6_step1d_notebook_figures.py        # 8 portfolio figures consolidated
│   ├── phase6_step2_var_lag_selection.py        # S1  — IC grid at maxlag=12 (D-050)
│   ├── phase6_step2_s1b_var_lag_sensitivity.py  # S1b — B&A threshold at maxlag=18 (D-050)
│   ├── phase6_step2_s2_var_estimation.py        # S2  — BIC p=2 baseline + diagnostics
│   ├── phase6_step2_s2b_var_estimation_aic.py   # S2b — AIC refit + whiteness comparison (D-050, D-051)
│   ├── phase6_step2_s3_granger_causality.py     # S3  — 5×5 × 4 Granger battery (D-052, D-053)
│   ├── phase6_step2_s4_irf.py                   # S4  — orthogonalised IRF with asymptotic CI (D-054..D-057)
│   ├── phase6_step2_s5_fevd.py                  # S5  — FEVD via manual IRF-based computation (D-054, D-055, D-058, D-059)
│   ├── phase6_step2_s6_oos_forecast.py          # S6  — walk-forward OOS for Phase 7 DM (D-060)
│   ├── phase6_step2_s6b_robust_metrics.py       # S6b — robust-metric diagnostic (D-060..D-062)
│   └── rebuild_processed.py                     # Canonical CLI for Phase 2 regeneration
│
├── src/                                          # Reusable module architecture (v0.4.1)
│   ├── __init__.py                              # Package meta + 84 re-exports
│   ├── data_loader.py                           # I/O helpers for raw + processed
│   ├── preprocessing.py                         # Phase 2 transformations
│   ├── stationarity.py                          # Phase 3 ADF+KPSS + 4-quadrant protocol
│   ├── structural_breaks.py                     # Phase 3 Chow + Quandt-Andrews
│   ├── feature_engineering.py                   # Phase 4 feature construction
│   └── modelling_utils.py                       # Phase 6 shared utilities (D-063)
│
├── outputs/
│   └── figures/
│       ├── phase3_level_adf_kpss_quadrant.png
│       ├── phase3_transformation_registry.png
│       ├── phase3_chow_pvalues.png
│       ├── phase3_energy2022_forest.png
│       ├── phase3_quandt_andrews_curves.png
│       ├── phase3_break_dates_alignment.png
│       ├── phase4_transform_distribution.png
│       ├── phase4_base_features_panel.png
│       ├── phase4_lag_illustration.png
│       ├── phase4_rolling_volatility.png
│       ├── phase4_regime_timeline_usa.png
│       ├── phase4_feature_nan_landscape.png
│       ├── phase5_step1_fig1_cpi_overlay.png    # 4-country dual-panel (D-041)
│       ├── phase5_step1_fig2_japan_deepdive.png # N3 three-panel (D-045)
│       ├── phase5_step2_fig3_indicator_panel.png # 4×5 time-series grid
│       ├── phase5_step2_fig4_base_heatmap.png   # 5×5 per country (D-042 Tier 1)
│       ├── phase5_step2_fig5_lag_heatmap.png    # 4×5 cross-lag (D-042 Tier 2)
│       ├── phase5_step3_fig6_phillips_scatter.png # Pre/post-GFC OLS (D-043)
│       ├── phase5_step3_fig7_rolling_slope.png  # 60m rolling dual panel
│       ├── phase5_step4_fig8_acf_pacf.png       # 4×2 ACF+PACF grid (D-044)
│       ├── phase6_step1_fig1_variants_overview.png       # 5 variants train/test split
│       ├── phase6_step1_fig2_aic_landscape.png           # AIC surface (p,q) at best seasonal
│       ├── phase6_step1_fig3_ic_comparison.png           # AIC/BIC/HQIC params (D-049 triple agreement)
│       ├── phase6_step1_fig4_residual_acf.png            # Residual ACF + Ljung-Box p-values
│       ├── phase6_step1_fig5_heteroscedasticity.png      # Rolling |residual| (D-049 Japan signature)
│       ├── phase6_step1_fig6_boundary_sensitivity.png    # Stage (b) ΔAIC verdicts (D-048)
│       ├── phase6_step1_fig7_aic_oos_divergence.png      # D-048 stopping rule 2-panel figure
│       ├── phase6_step1_fig8_test_forecasts.png          # Expanding-refit OOS forecasts × 5 variants
│       └── phase6_step2_*.png                            # Pending — 07_var_model.ipynb will add ~8 figures (IRF, FEVD heatmap, OOS panel, N3 sextuple mosaic)
│
├── ProjectDriven.md                              # Living decision log (D-001..D-063)
├── ProjectScope_v1.md                            # Immutable project scope
├── README.md                                     # This document
├── phase6_step2_summary.md                       # Phase 6 Step 2 PK compact summary
├── requirements.txt                              # Python dependencies
├── .env.example                                  # FRED API key template (never commit .env)
└── .gitignore
```

*Note: scratch script filenames shown above are illustrative; actual filenames may differ slightly based on local rebuild decisions. All canonical logic lives in `src/`; scratch scripts are preserved as reproducible implementation traces. Phase 6 Step 2 ultimately promoted one narrow module — `src/modelling_utils.py` at the v0.4.1 patch bump per D-063 — after empirical observation of 4–6× duplication of exog-assembly helpers and constant definitions across the nine Step 2 scratch scripts; model-fitting logic remains in scratch per D-047 spirit. The nine Step 2 scripts are deliberately left unrefactored (audit CSVs already produced; DRY refactor carries regression risk without output benefit); only new code from this point forward — `notebooks/07_var_model.ipynb`, Phase 7, and Step 3 — imports from `src.modelling_utils`. `src/` v0.5.0 is reserved for Phase 6 closure module assembly.*

---

## Reusable Module Architecture (`src/` v0.4.1)

| Module | Purpose | Exports |
|---|---|---:|
| `data_loader.py` | I/O helpers for raw and processed datasets | 9 |
| `preprocessing.py` | Phase 2 unit/frequency harmonisation, NaN handling, wide-format assembly | 14 |
| `stationarity.py` | Phase 3 ADF + KPSS joint protocol, 4-quadrant classification, transformation dispatch | 20 |
| `structural_breaks.py` | Phase 3 Chow (classical / HAC / COVID-dummy), per-coefficient decomposition, Quandt-Andrews sup-Wald | 16 |
| `feature_engineering.py` | Phase 4 base transform → lags → rolling stats → regime dummies → feature matrix assembly | 17 |
| `modelling_utils.py` | Phase 6 shared utilities — Cholesky ordering (D-054), AIC / BIC lag-order dicts (D-050), D-030 exog-assembly helpers. Narrow scope per D-063: pure utilities and constants only; model-fitting logic remains in scratch scripts pending Step 3 assessment. | 7 |
| `__init__.py` (v0.4.1) | Package meta + re-exports | 84 total |

**Version history:**

- `v0.1.0` — initial package scaffolding (Phase 1)
- `v0.2.0` — `preprocessing` module and expanded `data_loader` (Phase 2)
- `v0.3.0` — `stationarity` and `structural_breaks` modules (Phase 3)
- `v0.4.0` — `feature_engineering` module (Phase 4); consumed by Phase 5 EDA, Phase 6 Step 1 SARIMA, and Phase 6 Step 2 VAR with no API changes
- `v0.4.1` — `modelling_utils` module (Phase 6 Step 2 closeout per D-063); patch bump promoting seven items (two lag-order dicts, three constant lists, two pure helpers) duplicated 4–6× across the nine Step 2 scratch scripts. No regression risk — promoted items are byte-identical to scratch-script versions. Step 2 scratch scripts deliberately left unrefactored (audit CSVs already produced; DRY refactor carries regression risk without output benefit); only new code from this point forward — `notebooks/07_var_model.ipynb`, Phase 7, Step 3 — imports from `src.modelling_utils`
- `v0.5.0` — reserved for Phase 6 closure module assembly (likely `src/modelling.py` wrapping VAR / Ridge / walk-forward / Diebold-Mariano); the conservative v0.4.1 patch preserves room for this larger bump once Step 3 reveals stable model-fitting patterns

---

## Reproduction

The pipeline can be regenerated from source data via either the CLI scripts or the narrated notebooks. Both paths import from `src/` — the single source of truth for all transformation logic.

```bash
# Prerequisites
conda create -n p3_inflation python=3.10
conda activate p3_inflation
pip install -r requirements.txt

# Set FRED API key
cp .env.example .env
# Then edit .env and populate FRED_API_KEY=<your_key>

# Canonical CLI path (Phase 2 regeneration)
python scripts/rebuild_processed.py

# Phase 3 tests (can be re-run independently after Phase 2 output exists)
python scripts/phase3_step1_level_adf_kpss.py
python scripts/phase3_step2_differencing.py
python scripts/phase3_step3_cpi_registry.py
python scripts/phase3_step4_chow_structural_breaks.py
python scripts/phase3_step5_quandt_andrews.py
python scripts/phase3_step5b_quandt_andrews_trim10.py

# Phase 4 feature engineering (5 scratch orchestrators OR one module call)
python scripts/phase4_step5_assemble.py                        # CLI
# Alternatively, from Python:
#   from src import build_all_features, write_features_schema_md
#   features = build_all_features()
#   write_features_schema_md(features)

# Phase 5 exploratory data analysis (4 scratch orchestrators)
python scripts/phase5_step1_cpi_narrative.py
python scripts/phase5_step2_correlation_structure.py
python scripts/phase5_step3_phillips_curve.py
python scripts/phase5_step4_acf_pacf.py

# Phase 6 Step 1 SARIMA baseline (4 scratch orchestrators; total ~75 min single CPU)
python scripts/phase6_step1_arima_grid.py                      # ~61 min — 2,250 grid fits + 350 expanding refits
python scripts/phase6_step1b_q3_boundary_check.py              # ~1.5 min — 22 boundary fits
python scripts/phase6_step1c_usa_firstdiff_q3_extension.py     # ~12.5 min — 150 targeted fits + 70 refits
python scripts/phase6_step1d_notebook_figures.py               # ~0.2 min — 8 portfolio figures consolidated

# Phase 6 Step 2 VAR layer (9 scratch orchestrators; total ~5-10 min; S6 walk-forward dominates)
python scripts/phase6_step2_var_lag_selection.py               # S1  — IC grid at maxlag=12
python scripts/phase6_step2_s1b_var_lag_sensitivity.py         # S1b — boundary check at maxlag=18
python scripts/phase6_step2_s2_var_estimation.py               # S2  — BIC p=2 baseline
python scripts/phase6_step2_s2b_var_estimation_aic.py          # S2b — AIC refit + whiteness comparison
python scripts/phase6_step2_s3_granger_causality.py            # S3  — 5×5 × 4 Granger battery
python scripts/phase6_step2_s4_irf.py                          # S4  — orthogonalised IRF + asymptotic CI
python scripts/phase6_step2_s5_fevd.py                         # S5  — FEVD (manual IRF-based computation)
python scripts/phase6_step2_s6_oos_forecast.py                 # S6  — walk-forward OOS for Phase 7 DM
python scripts/phase6_step2_s6b_robust_metrics.py              # S6b — robust-metric diagnostic

# Narrated notebook path (Portfolio narrative, reproduces same outputs)
jupyter lab notebooks/
```

All notebooks import from `src` rather than re-implementing logic, so any future change to `src/` propagates through the entire stack. `notebooks/06_arima_baseline.ipynb` consumes the pre-computed CSVs in `data/documentation/phase6_step1_*.csv` to avoid re-running the 75-minute grid inside the notebook; figures are regenerated inline at Run All (≈ 1–2 min). `notebooks/07_var_model.ipynb` follows the same convention against `data/documentation/phase6_step2_*.csv` and consolidates the nine scratch orchestrators into a single portfolio narrative.

---

## Decision Log Pointer

Every analytical choice — country selection, data-source trade-offs, registry overrides, lag grids, regime specifications, EDA methodology, ARIMA grid protocol, VAR lag selection, Cholesky ordering, forecast robustness, `src/` promotion rules — is logged in **`ProjectDriven.md`** with rationale, alternatives considered, and implementation references. The log currently holds **63 decisions (D-001 through D-063)** covering Phases 0 through 6 Step 2 (Phase 6 is a three-step sub-phase; ARIMA Step 1 is complete with D-048 / D-049, VAR Step 2 is complete with D-050 through D-062 plus the D-063 module-architecture closeout, Ridge Step 3 pending with D-064+). This log is the primary portfolio artefact: the quality of the decisions matters more than the sophistication of the models, and documenting them explicitly is the core differentiator of this project.

For anyone reviewing this work, the recommended reading order is:

1. **`README.md`** (this document) — 5-minute overview
2. **`ProjectScope_v1.md`** — immutable project scope (written before implementation)
3. **`ProjectDriven.md`** — 62-decision log with full rationale
4. **`phase6_step2_summary.md`** — Phase 6 Step 2 VAR signature-findings compact summary (most recent deliverable)
5. **`notebooks/07_var_model.ipynb`** — Phase 6 Step 2 VAR narrative (pending assembly; consolidates D-050..D-062)
6. **`notebooks/06_arima_baseline.ipynb`** — Phase 6 Step 1 SARIMA baseline (D-048, D-049)
7. **`notebooks/05_eda.ipynb`** — Phase 5 cross-country narrative
8. **Upstream notebooks** (`03`, `04`) — stationarity, structural breaks, feature engineering

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Core language |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical operations |
| `statsmodels` | latest | ARIMA, VAR, ADF, KPSS, Chow, Ljung-Box, ACF/PACF, OLS, Granger, IRF, FEVD |
| `scikit-learn` | latest | Ridge regression, train/test split, metrics (Phase 6 Step 3) |
| `matplotlib` | latest | All figures |
| `fredapi` | latest | FRED API data collection |
| `python-dotenv` | latest | `.env` key management |
| `jupyter` | latest | Notebook environment |

Environment reproducibility: see `requirements.txt`. Conda environment name: `p3_inflation`.

---

## Next Steps

**Phase 6 Step 3 (immediate next) — Layer 3 Ridge regression** with walk-forward cross-validation on the full 50–53-feature matrices under L2 regularisation; handles the multicollinearity Phase 4 deliberately did not prune (D-040). Bridges back to P2 methodology. Delivered as `notebooks/08_ridge_regression.ipynb`. Expected decisions from D-064 onwards covering the regularisation-path selection, validation-fold structure, and coefficient-stability interpretation; following D-063's evidence-driven promotion threshold, any Ridge helpers duplicated 4+ times may be added to `src/modelling_utils` (v0.4.2 patch) or folded into the larger v0.5.0 module assembly at Phase 6 closure.

**Phase 6 closure** — `src/` v0.5.0 module assembly consolidating the Ridge modelling logic alongside (potentially) a full VAR wrapper that folds `modelling_utils` in — decision deferred until Step 3 reveals stable model-fitting patterns; `phase6_summary.md` PK handoff file consolidating all three layers; `notebooks/07_var_model.ipynb` finalisation if not already completed; README promotion of Phase 6 row to ✅ Complete.

**Phase 7** — Diebold-Mariano pairwise forecast comparison across ARIMA (Step 1) / VAR (Step 2) / Ridge (Step 3) for all four countries at horizons {1, 3, 6, 12} months. Both standard squared-error loss and robust / HAC sensitivities per D-051 (partial whitening caveat) and D-060 (inference-vs-forecast trade-off). USA yoy_pct vs first_diff dual-form comparison pre-committed per D-048 / D-062. Expected ~20 pairwise tests across the four countries.

**Phase 8** — Interpretation synthesis; `findings.md` (six-signature-finding centerpiece consolidating N1 / N2 / N3 with the Phillips Methodology Trilogy and Four-Lens Disconfirmation of Quantity Theory as the project's methodology meta-findings); `methodology.md` generation; final portfolio assembly.

---

*Portfolio Project 3 · Phase 6 Step 2 complete (Layers 1 + 2 of 3 modelling layers; ARIMA ✅ SARIMA baseline + VAR ✅ Granger / IRF / FEVD / OOS) · 63 decisions logged (D-001 through D-063) · `src/` at v0.4.1 (`modelling_utils` added per D-063; v0.5.0 reserved for Phase 6 closure module assembly) · 6 portfolio notebooks + `07_var_model.ipynb` pending · 32 portfolio figures (Phase 3 △ 6, Phase 4 △ 6, Phase 5 △ 8, Phase 6 Step 1 △ 8, Phase 6 Step 2 △ ~4 to be added with notebook assembly) · 25 Phase 6 Step 2 audit CSVs.*
