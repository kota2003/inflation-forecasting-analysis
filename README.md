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
| Phase 5 | Exploratory data analysis & cross-country narrative visualisation | ✅ **Complete** |
| Phase 6 | Model estimation — ARIMA, VAR, Ridge | ⏳ Next |
| Phase 7 | Evaluation — Diebold-Mariano, walk-forward validation | Pending |
| Phase 8 | Interpretation — Granger maps, IRF plots, narrative synthesis | Pending |

As of this writing, the project has completed the five-phase analytical foundation through exploratory data analysis. The `data/processed/` directory contains four main-country feature matrices of 50–53 columns each (USA, Japan, UK, Germany — D-031-corrected stationary forms with lags, rolling statistics, and regime dummies) plus a supplementary China dataset. Phase 3 classified every series' stationarity status and characterised three pre-specified structural breaks (2008-09, 2020-03, 2022-02) via Chow and Quandt-Andrews tests. Phase 4 built the per-country feature matrices (joint-valid from 2002-02 or 2003-01) via `src.feature_engineering` v0.4.0. Phase 5 produced eight portfolio figures and twelve audit CSVs spanning cross-country CPI dynamics, correlation structure, N1 Phillips Curve deep-dive, and ACF/PACF diagnostics — with seven signature findings now flagged in `ProjectDriven.md` that directly inform Phase 6 ARIMA/VAR/Ridge estimation.

---

## Three Narratives — Phase 5 State

The project is organised around three named economic narratives (ProjectScope §4). Phase 5 EDA supplies the single-number evidence for each; Phase 6 modelling will supply the causal / directional interpretation.

- **N1 · Phillips Curve — shock-activated, not dead.** Cross-country post-2022 rolling Phillips slopes reach |β| ≈ 5–9 with R² ∈ [0.60, 0.75] after a 2014–2020 quiescence. The UK exhibits a unique pre/post-GFC sign flip (β = +1.68 → −0.27) absent in the other three economies. Phase 5 S3 (Fig 6, Fig 7).
- **N2 · Monetary Policy Lag — M2 leads inflation by 12 months.** USA `corr(CPI_t, M2_{t−12}) = +0.41` contrasts with `corr(CPI_t, M2_t) = −0.17` — a sign-flip pattern consistent with the Quantity Theory of Money. Phase 5 S2 (Fig 5 cross-lag heatmap) previews what Phase 6 VAR IRF will quantify as directional causation.
- **N3 · Japan's Uniqueness — structural divergence in 90.7 % of months.** Japan CPI YoY is below the mean of USA/UK/Germany in 253 of 279 monthly observations (mean gap −1.80 pp). Four-phase decomposition shows monotone progression: Deflation era −0.20 % → Abenomics +0.64 % → Reversal +2.99 %, with the Reversal phase showing zero deflation months (0/45). Phase 5 S1 (Fig 1, Fig 2).

A seventh finding — recorded as **D-046** in the decision log — notes that the Phillips Curve is visible in level-based EDA (S3) but invisible under stationary-form correlation (S2). This is a deliberate methodology choice, not an inconsistency: the classical Phillips Curve is a level relationship, and both lenses play legitimate roles across Phase 5 (EDA) and Phase 6 (VAR).

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
│       └── phase5_step4_ljung_box.csv           # 12 rows — Q(12/24/36) per country
│
├── notebooks/                                    # Portfolio-grade narrative layer
│   ├── 01_data_collection.ipynb                 # Phase 1 collection + v2 rebuild
│   ├── 02_cleaning_alignment.ipynb              # Phase 2 unit harmonisation
│   ├── 03_stationarity_structural_breaks.ipynb  # Phase 3 ADF/KPSS + Chow/Q-A
│   ├── 04_feature_engineering.ipynb             # Phase 4 lag/rolling/regime assembly
│   └── 05_eda.ipynb                             # Phase 5 cross-country narrative
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
│   └── rebuild_processed.py                     # Canonical CLI for Phase 2 regeneration
│
├── src/                                          # Reusable module architecture (v0.4.0)
│   ├── __init__.py                              # Package meta + 77 re-exports
│   ├── data_loader.py                           # I/O helpers for raw + processed
│   ├── preprocessing.py                         # Phase 2 transformations
│   ├── stationarity.py                          # Phase 3 ADF+KPSS + 4-quadrant protocol
│   ├── structural_breaks.py                     # Phase 3 Chow + Quandt-Andrews
│   └── feature_engineering.py                   # Phase 4 feature construction
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
│       └── phase5_step4_fig8_acf_pacf.png       # 4×2 ACF+PACF grid (D-044)
│
├── ProjectDriven.md                              # Living decision log (D-001..D-047)
├── ProjectScope_v1.md                            # Immutable project scope
├── README.md                                     # This document
├── requirements.txt                              # Python dependencies
├── .env.example                                  # FRED API key template (never commit .env)
└── .gitignore
```

*Note: scratch script filenames shown above are illustrative; actual filenames may differ slightly based on local rebuild decisions. All canonical logic lives in `src/`; scratch scripts are preserved as reproducible implementation traces.*

---

## Reusable Module Architecture (`src/` v0.4.0)

| Module | Purpose | Exports |
|---|---|---:|
| `data_loader.py` | I/O helpers for raw and processed datasets | 9 |
| `preprocessing.py` | Phase 2 unit/frequency harmonisation, NaN handling, wide-format assembly | 14 |
| `stationarity.py` | Phase 3 ADF + KPSS joint protocol, 4-quadrant classification, transformation dispatch | 20 |
| `structural_breaks.py` | Phase 3 Chow (classical / HAC / COVID-dummy), per-coefficient decomposition, Quandt-Andrews sup-Wald | 16 |
| `feature_engineering.py` | Phase 4 base transform → lags → rolling stats → regime dummies → feature matrix assembly | 17 |
| `__init__.py` (v0.4.0) | Package meta + re-exports | 77 total |

**Version history:**

- `v0.1.0` — initial package scaffolding (Phase 1)
- `v0.2.0` — `preprocessing` module and expanded `data_loader` (Phase 2)
- `v0.3.0` — `stationarity` and `structural_breaks` modules (Phase 3)
- `v0.4.0` — `feature_engineering` module (Phase 4)
- `v0.5.0` — reserved for Phase 6 ARIMA / VAR / Ridge modelling modules

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

# Narrated notebook path (Portfolio narrative, reproduces same outputs)
jupyter lab notebooks/
```

All notebooks import from `src` rather than re-implementing logic, so any future change to `src/` propagates through the entire stack.

---

## Decision Log Pointer

Every analytical choice — country selection, data-source trade-offs, registry overrides, lag grids, regime specifications, EDA methodology — is logged in **`ProjectDriven.md`** with rationale, alternatives considered, and implementation references. The log currently holds **47 decisions (D-001 through D-047)** covering Phases 0 through 5. This log is the primary portfolio artefact: the quality of the decisions matters more than the sophistication of the models, and documenting them explicitly is the core differentiator of this project.

For anyone reviewing this work, the recommended reading order is:

1. **`README.md`** (this document) — 3-minute overview
2. **`ProjectScope_v1.md`** — immutable project scope (written before implementation)
3. **`ProjectDriven.md`** — 47-decision log with full rationale
4. **`notebooks/05_eda.ipynb`** — Phase 5 cross-country narrative (most recent deliverable)
5. **Upstream notebooks** (`03`, `04`) — stationarity, structural breaks, feature engineering

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Core language |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical operations |
| `statsmodels` | latest | ARIMA, VAR, ADF, KPSS, Chow, Ljung-Box, ACF/PACF, OLS |
| `scikit-learn` | latest | Ridge regression, train/test split, metrics (Phase 6+) |
| `matplotlib` | latest | All figures |
| `fredapi` | latest | FRED API data collection |
| `python-dotenv` | latest | `.env` key management |
| `jupyter` | latest | Notebook environment |

Environment reproducibility: see `requirements.txt`. Conda environment name: `p3_inflation`.

---

## Next Steps

**Phase 6 (immediate next)** — ARIMA / VAR / Ridge estimation on the Phase 4 feature matrices. The Phase 5 findings directly inform Phase 6 specification:

- **ARIMA baseline.** Ljung-Box Q(12) rejects white noise at p < 0.001 for all four countries (D-044); ARIMA modelling is statistically required. ACF[12] significance across all four countries argues for **SARIMA with seasonal order s=12**. Preliminary ARMA order candidates: USA AR(3), Japan ARMA(1,2), UK AR(2), Germany ARMA(2,2) — AIC/BIC grid search supersedes.
- **VAR.** Five-variable system (CPI + POLICY_RATE + UNEMPLOYMENT + GDP + M2) per country, on D-031-corrected stationary inputs. Regime-dummy interactions follow the D-030 dominant-driver matrix. Granger causality + Impulse Response Functions will quantify N2 Monetary Policy Lag.
- **Ridge.** 50–53 features per country under L2 regularisation; handles the multicollinearity Phase 4 deliberately did not prune (D-040). Bridges back to P2 methodology.

**Phase 7** — Diebold-Mariano pairwise model comparison; walk-forward validation (train 2002–2019, test 2020+).

**Phase 8** — Interpretation synthesis; `findings.md` and `methodology.md` generation; final portfolio assembly.

---

*Portfolio Project 3 · Phase 5 complete · 47 decisions logged · `src/` at v0.4.0 · 8 figures and 12 audit CSVs produced in Phase 5.*
