# Inflation Prediction and Economic Signal Analysis

A multi-country econometric study forecasting consumer price inflation across USA, Japan, UK, and Germany (2000-01 to present) using ARIMA, VAR, and Ridge Regression. The analysis combines classical time-series rigour with modern data-engineering practice and documents every design decision in a living decision log.

This is **Portfolio Project 3 (P3)** in a three-project series. P1 demonstrated machine-learning engineering on structured customer data; P2 covered feature engineering and classification with interpretability tools; this P3 demonstrates **classical econometric rigour combined with modern data engineering**. The project deliberately prioritises decision documentation, source auditing, and reproducibility over algorithmic novelty — the skills most valued in consulting contexts where analytical defensibility matters more than headline accuracy.

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
| Phase 6 | Model estimation — ARIMA, VAR, Ridge | ✅ **Complete (3 / 3 layers)** |
| Phase 7 | Evaluation — Diebold-Mariano, walk-forward validation | Pending |
| Phase 8 | Interpretation — Granger maps, IRF plots, narrative synthesis | Pending |

As of this writing, the project has completed the five-phase analytical foundation through exploratory data analysis plus all three modelling layers in Phase 6. The `data/processed/` directory contains four main-country feature matrices of 50–53 columns each (USA, Japan, UK, Germany — D-031-corrected stationary forms with lags, rolling statistics, and regime dummies) plus a supplementary China dataset. Phase 3 classified every series' stationarity status and characterised three pre-specified structural breaks (2008-09, 2020-03, 2022-02) via Chow and Quandt-Andrews tests. Phase 4 built the per-country feature matrices (joint-valid from 2002-02 or 2003-01) via `src.feature_engineering`. Phase 5 produced eight portfolio figures and twelve audit CSVs spanning cross-country CPI dynamics, correlation structure, N1 Phillips Curve deep-dive, and ACF/PACF diagnostics — with seven signature findings flagged in `ProjectDriven.md` that directly informed the Phase 6 ARIMA/VAR/Ridge estimation. **Phase 6 Step 1 (Layer 1 SARIMA baseline) is complete**: five variants estimated via a three-stage grid search protocol (450-order initial grid • boundary sensitivity check • targeted Q = 3 extension), 8 portfolio figures delivered in `notebooks/06_arima_baseline.ipynb`, and two decisions recorded (D-048 three-stage protocol with OOS-saturation stopping rule; D-049 Japan ARIMA uniqueness as an N3 narrative echo at the ARIMA layer). **Phase 6 Step 2 (Layer 2 VAR core model) is complete**: four country-specific VARs estimated across nine scratch orchestrators (S1 / S1b / S2 / S2b / S3 / S4 / S5 / S6 / S6b) covering lag selection with boundary sensitivity, BIC→AIC criterion revision via residual diagnostics, 5×5 Granger causality battery (100 tests), orthogonalised IRFs with asymptotic 95 % CIs, Forecast Error Variance Decomposition, and walk-forward out-of-sample forecasts for Phase 7 Diebold-Mariano; 25 audit CSVs written to `data/documentation/phase6_step2_*.csv`; 13 new decisions logged (D-050 through D-062) covering the VAR lag-selection protocol, Cholesky ordering, statsmodels API robustness patches, the inference-primary-vs-forecast-auxiliary positioning; plus the D-063 Step 2 closeout `src/modelling_utils` promotion at v0.4.1; 8 portfolio figures and the consolidated `notebooks/07_var_model.ipynb` narrative assembled at closure. **Phase 6 Step 3 (Layer 3 Ridge regression) is complete**: five (country, form) Ridge combinations fit under a leakage-guarded `Pipeline(StandardScaler, Ridge)` with α log-grid `np.logspace(-3, 3, 13)` × TimeSeriesSplit(5) CV, Japan grid extended to `logspace(3, 6, 7)` at S2b after the α = 1000 boundary hit; direct-h walk-forward OOS delivered to matched origins for Phase 7 DM; 15 audit CSVs; 8 portfolio figures in `notebooks/08_ridge_regression.ipynb`; 11 new decisions (D-064 through D-074, with D-073 amended in place) including the N3 septuple formalisation (D-072) and the `src/modelling_utils` v0.4.2 extension (D-074). **Phase 6 closure** adds D-075 — the `src/` v0.5.0 architectural assessment that resolves the tension between ProjectScope §12's blueprint specification and D-063's empirical promotion rule via a split plan (`src/evaluation.py` promoted at v0.4.3 before Phase 7; `src/models/` subdirectory deferred to Phase 7 closeout for re-assessment). Phase 7 Diebold-Mariano is the immediate next work.

---

## Three Narratives — State through Phase 6 (all three layers)

The project is organised around three named economic narratives (ProjectScope §4). Phase 5 EDA supplied the single-number correlational evidence; Phase 6 Step 1 added ARIMA residual-structure evidence; Phase 6 Step 2 supplied the multivariate causal / directional interpretation via Granger tests, IRFs, and FEVD; Phase 6 Step 3 completes the triangulation by adding high-dimensional regularised-regression evidence through Ridge α-selection and standardised coefficients.

- **N1 · Phillips Curve — Anglo-specific, methodology-dependent (Quadrilogy confirmed).** Phase 5 rolling slopes reach |β| ≈ 5–9 with R² ∈ [0.60, 0.75] post-2022; UK alone shows a pre/post-GFC sign flip (β = +1.68 → −0.27). Phase 6 Step 2 Granger tests localise significance to the Anglo axis (UK p = 0.002 ★★; USA p = 0.017 ★; Japan p = 0.39; Germany p = 0.30). Step 2 IRFs further deliver an **unexpected positive sign** for UK and USA UE → CPI peak responses (+0.042 at h = 1 and +0.267 at h = 5 respectively) — a stagflation-era co-movement echo that regime dummies do not fully absorb. FEVD at h = 12 corroborates: USA unemployment explains 26.8 % of CPI variance (#2 driver), UK 6.7 %, Japan and Germany below 4 %. Phase 6 Step 3 Ridge adds a **fourth lens**: under standardised coefficients with 5-fold TimeSeriesSplit sign-stability, Germany is the only country to surface a classical-signed Phillips base-feature in the top-5 drivers — a Germany-only active Phillips lens that extends D-057's Trilogy into the D-067 Quadrilogy and sharpens the lens-dependence claim.
- **N2 · Monetary Policy Lag — USA-specific via the interest-rate channel, cross-lens-confirmed.** Phase 5 previewed USA `corr(CPI_t, M2_{t−12}) = +0.41` as a Quantity Theory signature. Phase 6 Step 2 definitively separates the two channels: the **interest-rate channel** survives only in USA (Granger p = 0.004 ★★; IRF peak −0.149 at h = 4 with CI excluding zero; FEVD share growing to 14.1 % at h = 24 — five times the share in any other country), while the **money-supply channel (M2 → CPI) is universally null** across Granger (p > 0.12 all four countries), IRF (all CIs straddle zero), and FEVD (< 5 % share at every horizon, every country). Phase 6 Step 3 Ridge resolves the USA dual-form ambiguity (D-071): the Ridge first_diff fit delivers `POLICY_RATE_lag3` standardised coefficient = −0.136 with sign stability across CV folds — a **cross-lens match** with VAR IRF peak −0.149 at h = 4 (D-056). Ridge first_diff is therefore the preferred N2 primary form for Phase 7 DM, with Ridge yoy_pct retained as sensitivity. N2 now carries two-method agreement (VAR inference + Ridge regularised coefficient) on a USA-anchored interest-rate channel.
- **N3 · Japan's Uniqueness — septuple-confirmed structural isolation.** (1) Level peer-gap: Japan CPI YoY is below the mean of USA/UK/Germany in 253 of 279 monthly observations (90.7 %; mean gap −1.80 pp). (2) Phase monotone progression: Deflation −0.20 % → Abenomics +0.64 % → Reversal +2.99 % (Reversal phase shows 0 / 45 deflation months). Phase 5 S1 (Fig 1, Fig 2). (3) **ARIMA simplicity**: Japan is the only variant among five Step 1 candidates with triple AIC / BIC / HQIC agreement on a 4-parameter sparse order and ARCH-LM p ≈ 0.9999 (D-049). (4) **VAR lag interior minimum**: Japan is the only country whose AIC argmin (p* = 5) is interior rather than at the maxlag = 12 boundary (D-050 / S1b extension confirms). (5) **Granger null across all causers**: 0 / 4 Granger tests for CPI-receivers are significant in Japan (p ∈ {0.21, 0.42, 0.39, 0.26}), while Japan's VAR shows active causation elsewhere (GDP → UE p = 0.003; M2 → GDP p = 0.000) — causation structure exists, but none flows to CPI. (6) **FEVD self-share plateau**: Japan CPI's own-shock share is 92.1 % at h = 12 and 92.0 % at h = 24 — no external driver ever catches up. (7) **Ridge α-magnitude and coefficient-magnitude stratification** (D-066, D-067): under the identical leakage-guarded CV protocol applied to all four main countries, Japan selects α* = 3162 versus α* ∈ {10, 100, 31.6} for USA / UK / Germany — three orders of magnitude higher regularisation pressure; Japan's resulting max|coefficient| = 0.010 is 9.9 × to 71.4 × smaller than any peer country's maximum. These two Ridge-layer fingerprints are independent of each other (α selection is CV-driven; coefficient magnitude is post-fit) and together deliver the seventh inferential lens. The cross-project septuple status is formally declared in D-072. N3 is now the single most robustly-evidenced finding in the project — six methodologies over Phases 5–6 plus the Ridge α + coefficient magnitude, drawing on fundamentally different mathematical objects (correlation, likelihood, hypothesis testing, dynamic response, variance share, penalised coefficient magnitude) and all converging on the same conclusion.

Six methodology findings are recorded separately in the decision log and shape how the three narratives are reported:

- **D-046** — the Phillips Curve is visible in level-based EDA (Phase 5 S3) but invisible under stationary-form correlation (S2). This is a deliberate methodology choice, not an inconsistency: the classical Phillips Curve is a level relationship, and both lenses play legitimate roles across Phase 5 (EDA), Phase 6 Step 1 (ARIMA), and Phase 6 Step 2 (VAR).
- **D-048 stopping rule** — AIC in-sample improvement from SARIMA grid extension does not translate to out-of-sample performance at the orders evaluated in Phase 6 Step 1; USA_first_diff Stage (a) → Stage (c) ΔAIC = −10.46 produced OOS RMSE Δ = −0.003. OOS saturation is adopted as the principled stopping criterion, obligating Phase 7 Diebold-Mariano to compare OOS loss differentials rather than AIC rankings.
- **D-057 / D-067 Phillips Methodology Quadrilogy** — the Phillips Curve manifests in four mutually compatible but visually different ways across analytical lenses: classical negative slope in level form (Phase 5 D-043), invisibility in stationary correlation (D-046), positive sign in stationary IRF for Anglo countries (D-057 Trilogy), and Germany-only activation in Ridge standardised base-feature coefficients (D-067 Quadrilogy extension). No single country is Phillips-active across all four lenses — the emergent lens-dependence is itself a project-centerpiece methodology meta-finding rather than a contradiction.
- **D-058 Four-Lens Disconfirmation of Quantity Theory of Money** — Phase 5's +0.41 USA CPI-M2 cross-lag correlation fails across four independent Phase 6 inferential lenses (Granger, IRF, FEVD, cross-country consistency). This definitive negative result extends D-046's methodology asymmetry into a generalised principle: **correlation in stationary form does not imply any form of inferential causation**.
- **D-060 / D-070 VAR inference-primary vs forecast-auxiliary, with Ridge 12/16 forecast dominance** — the D-050 revision from BIC p = 2 to AIC-selected p per country optimised for residual whitening and inference quality at the explicit cost of OOS forecast accuracy. Only Japan VAR(5) beats the random-walk naive benchmark in aggregate MASE; USA, UK, and Germany under-perform. Under robust median-absolute-error metric (D-061 / S6b), Japan wins at every horizon and UK / Germany are near-competitive. Phase 6 Step 3 Ridge then improves on VAR in 12 / 16 (country × horizon) cells — Germany 4 / 4, UK 4 / 4, USA 4 / 4, Japan 0 / 4 but all within noise — with the UK h = 12 MASE reduction from 79.07 to 1.02 (77 ×) as L2 absorbs the D-061 COVID-origin outlier. Absolute-vs-relative honesty is preserved (D-070): Ridge beats the naive benchmark in only 2 / 16 cells, so the three-layer architecture delivers *relative* improvement rather than dominance over the null model. The portfolio records VAR as "inference-primary, forecast-auxiliary" and Ridge as "forecast-improved over VAR, still naive-non-dominant" — Phase 7 DM will evaluate ARIMA / VAR / Ridge on matched terms.
- **D-069 Regime-interaction zero-information** — 5 of 6 D-030-gated regime interaction columns are structurally zero in the 2000–2019 train window (COVID and ENERGY breaks post-date the training cut). Ridge L2 correctly shrinks these to near-zero; this is methodology transparency rather than a bug, and documents that the D-030 regime specification is a post-COVID test-window construct reconstructed on training data by design.

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
│       ├── phase6_step2_s6b_*.csv                       # 3 files — worst origins + robust metrics + CPI robust summary (D-061, D-062)
│       ├── phase6_step3_s1_*.csv                        # 3 files — feature matrix summary + categories + target (D-064)
│       ├── phase6_step3_s2_*.csv                        # 2 files — CV scores + α selection (D-065)
│       ├── phase6_step3_s2b_japan_*.csv                 # 2 files — Japan grid extension CV + α selection (D-066)
│       ├── phase6_step3_s3_*.csv                        # 3 files — Ridge coefficients + top features + category contribution (D-067)
│       ├── phase6_step3_s4_ridge_oos_*.csv              # 3 files — OOS forecasts + metrics + CPI summary (D-068)
│       ├── phase6_step3_s5_*.csv                        # 3 files — country narrative summary + Ridge vs VAR MASE + Ridge narrative statements (D-070)
│       └── phase6_step3_s5b_phillips_base_feature_lens.csv # 4 rows — Phillips Quadrilogy audit (D-067)
│
├── notebooks/                                     # Portfolio-grade narrative layer
│   ├── 01_data_collection.ipynb                 # Phase 1 collection + v2 rebuild
│   ├── 02_cleaning_alignment.ipynb              # Phase 2 unit harmonisation
│   ├── 03_stationarity_structural_breaks.ipynb  # Phase 3 ADF/KPSS + Chow/Q-A
│   ├── 04_feature_engineering.ipynb             # Phase 4 lag/rolling/regime assembly
│   ├── 05_eda.ipynb                             # Phase 5 cross-country narrative
│   ├── 06_arima_baseline.ipynb                  # Phase 6 Step 1 SARIMA baseline (D-048, D-049)
│   ├── 07_var_model.ipynb                       # Phase 6 Step 2 VAR narrative (D-050..D-063 consolidated; N3 sextuple at Step 2 extended to septuple at Step 3 per D-072)
│   ├── 08_ridge_regression.ipynb                # Phase 6 Step 3 Ridge layer (D-064..D-074)
│   └── 09_evaluation_interpretation.ipynb       # Phase 7/8 narrative (pending; per ProjectScope §12)
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
│   ├── phase6_step2_var_lag_selection.py        # S1  — IC grid at maxlag=12
│   ├── phase6_step2_s1b_var_lag_sensitivity.py  # S1b — boundary check at maxlag=18
│   ├── phase6_step2_s2_var_estimation.py        # S2  — BIC p=2 baseline
│   ├── phase6_step2_s2b_var_estimation_aic.py   # S2b — AIC refit + whiteness comparison
│   ├── phase6_step2_s3_granger_causality.py     # S3  — 5×5 × 4 Granger battery
│   ├── phase6_step2_s4_irf.py                   # S4  — orthogonalised IRF + asymptotic CI
│   ├── phase6_step2_s5_fevd.py                  # S5  — FEVD (manual IRF-based computation)
│   ├── phase6_step2_s6_oos_forecast.py          # S6  — walk-forward OOS for Phase 7 DM
│   ├── phase6_step2_s6b_robust_metrics.py       # S6b — robust-metric diagnostic
│   ├── phase6_step3_s1_feature_matrix.py        # S1  — feature-matrix scope + CPI target (D-064)
│   ├── phase6_step3_s2_cv_alpha_selection.py    # S2  — α grid × TimeSeriesSplit(5) CV (D-065)
│   ├── phase6_step3_s2b_japan_grid_extension.py # S2b — Japan boundary-escape extension (D-066)
│   ├── phase6_step3_s3_coefficients.py          # S3  — standardised coefficients + category contribution (D-067)
│   ├── phase6_step3_s4_oos_forecast.py          # S4  — direct-h walk-forward OOS (D-068)
│   ├── phase6_step3_s5_narrative_synthesis.py   # S5  — Ridge vs VAR MASE + country narrative (D-070)
│   └── phase6_step3_s5b_narrative_correction.py # S5b — Phillips base-feature lens (D-067 quadrilogy)
│
├── src/                                          # Reusable module architecture (v0.4.2)
│   ├── __init__.py                              # Package meta + re-exports (97 total)
│   ├── data_loader.py                           # Phase 1 I/O helpers
│   ├── preprocessing.py                         # Phase 2 transformations
│   ├── stationarity.py                          # Phase 3 ADF/KPSS + 4-quadrant classification
│   ├── structural_breaks.py                     # Phase 3 Chow + Quandt-Andrews
│   ├── feature_engineering.py                   # Phase 4 base/lag/rolling/regime assembly
│   └── modelling_utils.py                       # Phase 6 shared utilities (D-063 + D-074)
│
├── outputs/
│   └── figures/
│       ├── phase3_*.png                         # 6 figures
│       ├── phase4_*.png                         # 6 figures
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
│       ├── phase6_step2_fig1..8_*.png                    # 8 figures — IRF, FEVD heatmap, OOS panel, N3 sextuple mosaic (notebook 07 Run All)
│       └── phase6_step3_fig1..8_*.png                    # 8 figures — α selection, coefficient stratification, Ridge OOS, Ridge vs VAR MASE (notebook 08 Run All)
│
├── ProjectDriven.md                              # Living decision log (D-001..D-075; D-020 historical vacancy)
├── ProjectScope_v1.md                            # Immutable project scope
├── README.md                                     # This document
├── phase6_step1_summary.md                       # Phase 6 Step 1 PK compact summary
├── phase6_step2_summary.md                       # Phase 6 Step 2 PK compact summary (sextuple → septuple amended per D-072)
├── phase6_step3_summary.md                       # Phase 6 Step 3 PK compact summary
├── phase6_summary.md                             # Phase 6 top-level consolidated summary (nine cross-phase signature findings)
├── requirements.txt                              # Python dependencies
├── .env.example                                  # FRED API key template (never commit .env)
└── .gitignore
```

*Note: scratch script filenames shown above are illustrative; actual filenames may differ slightly based on local rebuild decisions. All canonical logic lives in `src/`; scratch scripts are preserved as reproducible implementation traces. Phase 6 Step 2 ultimately promoted one narrow module — `src/modelling_utils.py` at the v0.4.1 patch bump per D-063 — after empirical observation of 4–6× duplication of exog-assembly helpers and constant definitions across the nine Step 2 scratch scripts; Phase 6 Step 3 extended that module at v0.4.2 per D-074 with Ridge-layer helpers and constants. Model-fitting logic remains in scratch per D-047 spirit. Both the nine Step 2 scripts and the seven Step 3 scripts are deliberately left unrefactored (audit CSVs already produced; DRY refactor carries regression risk without output benefit); only new code from this point forward — `notebooks/07_var_model.ipynb`, `notebooks/08_ridge_regression.ipynb`, Phase 7 DM, and any future reproducibility work — imports from `src.modelling_utils`. Per D-075, `src/evaluation.py` is scheduled for promotion at v0.4.3 immediately preceding Phase 7 DM pre-flight; the `src/models/` subdirectory (ProjectScope §12) is deferred to Phase 7 closeout pending empirical promotion evidence under D-063.*

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

# Phase 6 Step 3 Ridge layer (7 scratch orchestrators; total ~2-5 min)
python scripts/phase6_step3_s1_feature_matrix.py               # S1  — feature-matrix scope + CPI target (D-064)
python scripts/phase6_step3_s2_cv_alpha_selection.py           # S2  — α log-grid × TSS(5) CV; leakage-guard (D-065)
python scripts/phase6_step3_s2b_japan_grid_extension.py        # S2b — Japan boundary escape to logspace(3,6,7) (D-066)
python scripts/phase6_step3_s3_coefficients.py                 # S3  — standardised coefficients + 5-fold stability (D-067)
python scripts/phase6_step3_s4_oos_forecast.py                 # S4  — direct-h walk-forward OOS (D-068)
python scripts/phase6_step3_s5_narrative_synthesis.py          # S5  — Ridge vs VAR MASE + country narrative (D-070)
python scripts/phase6_step3_s5b_narrative_correction.py        # S5b — Phillips base-feature lens (D-067 quadrilogy)

# Narrated notebook path (Portfolio narrative, reproduces same outputs)
jupyter lab notebooks/
```

All notebooks import from `src` rather than re-implementing logic, so any future change to `src/` propagates through the entire stack. `notebooks/06_arima_baseline.ipynb` consumes the pre-computed CSVs in `data/documentation/phase6_step1_*.csv` to avoid re-running the 75-minute grid inside the notebook; figures are regenerated inline at Run All (≈ 1–2 min). `notebooks/07_var_model.ipynb` follows the same convention against `data/documentation/phase6_step2_*.csv` and consolidates the nine scratch orchestrators into a single portfolio narrative. `notebooks/08_ridge_regression.ipynb` consumes `data/documentation/phase6_step3_*.csv` and delivers the Layer 3 Ridge narrative with α-selection diagnostics, standardised-coefficient stratification (D-067 Quadrilogy lens), Ridge-vs-VAR forecast differential (D-070 twelve-of-sixteen positioning), and the N3 septuple-confirmation mosaic (D-066, D-067, D-072).

---

## Decision Log Pointer

Every analytical choice — country selection, data-source trade-offs, registry overrides, lag grids, regime specifications, EDA methodology, ARIMA grid protocol, VAR lag selection, Cholesky ordering, forecast robustness, Ridge α-selection protocol, `src/` promotion rules — is logged in **`ProjectDriven.md`** with rationale, alternatives considered, and implementation references. The log currently holds **74 decisions (D-001 through D-075, with D-020 as a historical vacancy in numbering — discovered during Phase 6 Step 3 documentation cleanup; accepted as-is per D-075 rationale on evidence-driven restraint vis-à-vis retroactive renumbering)** covering Phases 0 through 6 (Phase 6 is a three-step sub-phase: ARIMA Step 1 complete with D-048 / D-049, VAR Step 2 complete with D-050 through D-062 plus the D-063 module-architecture closeout, Ridge Step 3 complete with D-064 through D-074 and D-073 amended in place; D-075 is the Phase 6 closure architectural assessment pre-committing `src/evaluation.py` at v0.4.3 for Phase 7). This log is the primary portfolio artefact: the quality of the decisions matters more than the sophistication of the models, and documenting them explicitly is the core differentiator of this project.

For anyone reviewing this work, the recommended reading order is:

1. **`README.md`** (this document) — 5-minute overview
2. **`ProjectScope_v1.md`** — immutable project scope (written before implementation)
3. **`ProjectDriven.md`** — 74-decision log with full rationale
4. **`phase6_summary.md`** — Phase 6 top-level consolidated summary (nine cross-phase signature findings centrepiece)
5. **`phase6_step3_summary.md`** — Phase 6 Step 3 Ridge signature-findings compact summary (most recent deliverable)
6. **`phase6_step2_summary.md`** — Phase 6 Step 2 VAR signature-findings compact summary (septuple amendment applied per D-072)
7. **`phase6_step1_summary.md`** — Phase 6 Step 1 SARIMA compact summary
8. **`notebooks/08_ridge_regression.ipynb`** — Phase 6 Step 3 Ridge narrative (D-064..D-074)
9. **`notebooks/07_var_model.ipynb`** — Phase 6 Step 2 VAR narrative (D-050..D-063)
10. **`notebooks/06_arima_baseline.ipynb`** — Phase 6 Step 1 SARIMA baseline (D-048, D-049)
11. **`notebooks/05_eda.ipynb`** — Phase 5 cross-country narrative
12. **Upstream notebooks** (`01`, `02`, `03`, `04`) — data collection, cleaning, stationarity / structural breaks, feature engineering

---

## Reusable Module Architecture (`src/` v0.4.2)

| Module | Purpose | Exports |
|---|---|---:|
| `data_loader.py` | I/O helpers for raw and processed datasets | 9 |
| `preprocessing.py` | Phase 2 unit/frequency harmonisation, NaN handling, wide-format assembly | 14 |
| `stationarity.py` | Phase 3 ADF + KPSS joint protocol, 4-quadrant classification, transformation dispatch | 20 |
| `structural_breaks.py` | Phase 3 Chow (classical / HAC / COVID-dummy), per-coefficient decomposition, Quandt-Andrews sup-Wald | 16 |
| `feature_engineering.py` | Phase 4 base transform → lags → rolling stats → regime dummies → feature matrix assembly | 17 |
| `modelling_utils.py` | Phase 6 shared utilities — Cholesky ordering (D-054), AIC / BIC lag-order dicts (D-050), D-030 exog-assembly helpers, Ridge α-grid constant + pipeline factory (D-065), selected-α loader (D-065/D-066), VAR MASE denominator (D-060). Narrow scope per D-063: pure utilities and constants only; model-fitting logic remains in scratch. | 20 |
| `__init__.py` (v0.4.2) | Package meta + re-exports | 97 total |

**Version history:**

- `v0.1.0` — initial package scaffolding (Phase 1)
- `v0.2.0` — `preprocessing` module and expanded `data_loader` (Phase 2)
- `v0.3.0` — `stationarity` and `structural_breaks` modules (Phase 3)
- `v0.4.0` — `feature_engineering` module (Phase 4); consumed by Phase 5 EDA, Phase 6 Step 1 SARIMA, and Phase 6 Step 2 VAR with no API changes
- `v0.4.1` — `modelling_utils` module (Phase 6 Step 2 closeout per D-063); patch bump promoting seven items (two lag-order dicts, three constant lists, two pure helpers) duplicated 4–6× across the nine Step 2 scratch scripts. No regression risk — promoted items are byte-identical to scratch-script versions.
- `v0.4.2` — `modelling_utils` extension (Phase 6 Step 3 closeout per D-074); patch bump promoting thirteen additional exports (seven constants covering the α log-grid / TimeSeriesSplit defaults / feature-category regexes, five helpers covering the leakage-guarded Pipeline factory / selected-α loader / direct-h walk-forward OOS helpers / Phillips base-feature audit, and one reference dict for the VAR MASE denominator per D-060). Step 2 and Step 3 scratch scripts deliberately left unrefactored; only new code from this point forward — Phase 7 DM, Phase 8 interpretability work, future reproducibility — imports from `src.modelling_utils`.
- `v0.4.3` — **scheduled** for Phase 7 DM pre-flight per D-075: new module `src/evaluation.py` containing RMSE / MAE / MASE / Diebold-Mariano (standard + HAC + robust) primitives. Pre-committed rather than retrospectively promoted because Phase 7's four-plus sub-step scripts are projected to exercise these primitives 4 × under D-063's threshold rule.
- `v0.5.0` — **deferred** to Phase 7 closeout per D-075. Contingent on empirical duplication evidence accumulating for model-fitting logic. Candidate scope: `src/models/{arima_model,var_model,ridge_model}.py` subdirectory per ProjectScope §12 blueprint.

The current v0.4.2 state implements four of the eight files specified in the ProjectScope §12 `src/` blueprint (`data_loader.py`, `preprocessing.py`, `stationarity.py` — split with `structural_breaks.py` per D-032 — and `feature_engineering.py`) plus `modelling_utils.py` as a Phase 6 shared-utilities convenience module not named in §12. The remaining blueprint files — `src/models/arima_model.py`, `src/models/var_model.py`, `src/models/ridge_model.py`, and `src/evaluation.py` — are subject to D-075's split promotion plan: `src/evaluation.py` precedes Phase 7 as a v0.4.3 patch bump because Phase 7's DM sub-step scripts predictively satisfy D-063's 4× threshold; the `src/models/` subdirectory is deferred to Phase 7 closeout pending empirical promotion evidence (current model-fitting duplication is ≤ 2× per pattern across Phase 6 Step {1, 2, 3} scratch orchestrators, and Phase 7 DM consumes pre-computed forecast CSVs without re-fitting models). This split honours ProjectScope §12's pre-implementation blueprint and D-063's evidence-driven promotion rule simultaneously rather than subordinating one to the other.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Core language |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical operations |
| `statsmodels` | latest | ARIMA, VAR, ADF, KPSS, Chow, Ljung-Box, ACF/PACF, OLS, Granger, IRF, FEVD |
| `scikit-learn` | latest | Ridge regression, TimeSeriesSplit, Pipeline, StandardScaler, metrics (Phase 6 Step 3) |
| `matplotlib` | latest | All figures |
| `fredapi` | latest | FRED API data collection |
| `python-dotenv` | latest | `.env` key management |
| `jupyter` | latest | Notebook environment |

Environment reproducibility: see `requirements.txt`. Conda environment name: `p3_inflation`.

---

## Next Steps

**Phase 6 complete (layers 1 + 2 + 3 of 3).** The three-layer modelling architecture mandated by ProjectScope §9 and D-004 is analytically closed: ARIMA provides the univariate baseline; VAR provides the multivariate inference-primary lens through Granger / IRF / FEVD; Ridge provides the high-dimensional regularised-regression lens on the 50–53-feature superset. Nine cross-phase signature findings are consolidated in `phase6_summary.md` for direct handoff to Phase 8 `findings.md` — the N3 septuple confirmation (D-072) takes top billing, with the Phillips Quadrilogy (D-067) and the Ridge-vs-VAR 12/16 relative-dominance (D-070) as methodology meta-findings. Twenty-four portfolio figures, fifty-five audit CSVs, and eight narrated notebooks cover the Phase 6 implementation; D-075 at Phase 6 closure formalises the `src/` v0.5.0 architectural decision as a split plan against ProjectScope §12.

**Phase 7 (immediate next) — Diebold-Mariano pairwise forecast comparison** across ARIMA (Step 1) / VAR (Step 2) / Ridge (Step 3) for all four countries at horizons {1, 3, 6, 12} months. The pre-flight opens with Tranche 1 of D-075: `src/evaluation.py` is created at v0.4.3 containing RMSE / MAE / MASE / Diebold-Mariano primitives (standard, HAC-robust, robust-metric variants), then Phase 7's DM sub-step scripts import from it cleanly from the first commit. Both standard squared-error loss and robust / HAC sensitivities per D-051 (partial whitening caveat) and D-060 / D-061 (inference-vs-forecast trade-off; UK h = 12 COVID-origin instability). USA yoy_pct vs first_diff dual-form comparison pre-committed per D-048 (ARIMA layer) and D-062 / D-071 (Ridge layer resolves first_diff as preferred for N2). Expected approximately 48 DM tests across the four countries — 3 model-pairs × 4 countries × 4 horizons with USA dual-form expanded as sensitivity. `notebooks/09_evaluation_interpretation.ipynb` (ProjectScope §12 specified) assembles the DM narrative with cross-lens match highlights for the Phase 7 closeout.

**Phase 8 — Interpretation synthesis.** `findings.md` centred on the nine cross-phase signature findings (N3 septuple, N1 Phillips Quadrilogy, N2 two-method cross-lens match, Four-Lens Disconfirmation of Quantity Theory, Ridge-vs-VAR relative dominance, absolute-vs-relative positioning, AIC-OOS divergence stopping rule, regime-interaction zero-information, cross-phase audit trail integrity). `methodology.md` generated from the decision progression across D-046 → D-053 → D-058 → D-070. Final portfolio assembly per ProjectScope §14 README outline. The Phase 7 closeout will revisit D-075 Tranche 2 (`src/models/` v0.5.0) against accumulated evidence; if no new duplication signal has emerged by then, the `src/models/` blueprint diff is accepted as an intentional architectural decision and recorded as such.

---

*Portfolio Project 3 · Phase 6 complete (3 / 3 modelling layers — ARIMA ✅ SARIMA baseline + VAR ✅ Granger / IRF / FEVD + Ridge ✅ high-dim regularised) · 74 decisions logged (D-001..D-075; D-020 historical vacancy) · `src/` at v0.4.2 (ProjectScope §12 blueprint partially implemented; v0.4.3 `src/evaluation.py` scheduled for Phase 7 pre-flight per D-075) · 8 portfolio notebooks + `09_evaluation_interpretation.ipynb` pending for Phase 7/8 · 24 Phase 6 portfolio figures · 55 Phase 6 audit CSVs.*
