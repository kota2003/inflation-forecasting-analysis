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
| Phase 3 | Stationarity testing (ADF+KPSS), structural-break testing (Chow, Quandt-Andrews) | ✅ **Complete** |
| Phase 4 | Feature engineering (lags, rolling statistics, regime dummies) | ⏳ Next |
| Phase 5 | Exploratory data analysis & cross-country narrative visualisation | Pending |
| Phase 6 | Model estimation — ARIMA, VAR, Ridge | Pending |
| Phase 7 | Evaluation — Diebold-Mariano, walk-forward validation | Pending |
| Phase 8 | Interpretation — Granger maps, IRF plots, narrative synthesis | Pending |

As of this writing, the `data/processed/` directory contains four fully-observed main-country datasets (USA, Japan, UK, Germany — NaN-free, 2001-01 onwards) and one supplementary dataset (China, sparse by design). All are VAR-ingestion-ready via `src.data_loader.load_processed_main()`. Phase 3 has classified every series' stationarity status, characterised three pre-specified structural breaks (2008-09, 2020-03, 2022-02) via three Chow variants, and independently confirmed the ENERGY_2022 break via Quandt-Andrews sup-Wald scan at two trim fractions.

---

## Repository Structure

```
inflation-forecasting-analysis/
├── data/
│   ├── raw/                              # 25 source series (final Phase 1 v2 + D-021 state)
│   │   ├── {COUNTRY}_{INDICATOR}.csv    # 5 × 5 grid of country × indicator
│   │   ├── _archive_v1/{timestamp}/     # Phase 1 v1 state archived pre-rebuild
│   │   ├── _archive_d021/{timestamp}/   # Germany M2 placeholder archived pre-resolution
│   │   ├── _manual/                     # Manual government CSVs (Japan CPI)
│   │   └── UK_IP.csv                    # Retained from Chow-Lin due diligence
│   ├── processed/                        # Phase 2 output — VAR-ready datasets
│   │   ├── main_usa.csv                 # 298 rows × 5 cols, 2001-01 to 2025-10
│   │   ├── main_japan.csv               # 298 rows × 5 cols, 2001-01 to 2025-10
│   │   ├── main_uk.csv                  # 291 rows × 5 cols, 2001-01 to 2025-03
│   │   ├── main_germany.csv             # 291 rows × 5 cols, 2001-01 to 2025-03
│   │   ├── supplementary_china.csv      # 300 rows × 5 cols, VAR-excluded
│   │   └── schema.md                    # Auto-generated schema specification
│   └── documentation/                    # Audit logs from every pipeline stage
│       ├── phase1v2_rebuild_log.csv
│       ├── phase2_cleaning_log.csv
│       ├── phase2_germany_m2_scout.csv
│       ├── phase2_m2_yoy_validation.csv
│       ├── phase2_ip_scout*.csv         # Chow-Lin due diligence (rejected)
│       ├── phase3_adf_kpss_levels.csv   # Phase 3 Task 1 audit artefacts
│       ├── phase3_differencing_log.csv
│       ├── phase3_cpi_deep_dive.csv
│       ├── phase3_subperiod_stationarity.csv
│       ├── phase3_transformation_registry_final.csv    # source of truth for D-027/D-031
│       ├── phase3_chow_tests_{classical,hac,covid_dummy}.csv
│       ├── phase3_chow_coefficient_decomposition.csv   # input to D-030
│       ├── phase3_chow_bonferroni_summary.csv
│       └── phase3_quandt_andrews_*.csv  # Task 2 sup-Wald scans (π₀ = 0.15 & 0.10)
├── src/                                  # Reusable Python modules (imported by notebooks)
│   ├── __init__.py                       # Package v0.3.0
│   ├── data_loader.py                    # I/O helpers for raw & processed datasets
│   ├── preprocessing.py                  # Phase 2 transformation functions
│   ├── stationarity.py                   # Phase 3 Task 1 — ADF/KPSS, 4-quadrant, transforms
│   └── structural_breaks.py              # Phase 3 Task 2 — Chow, decomposition, Quandt-Andrews
├── scripts/
│   ├── rebuild_processed.py              # CLI orchestrator — regenerates data/processed/
│   ├── regenerate_phase2_audits.py       # Regenerates Phase 2 audit CSVs
│   └── phase3_step*.py                   # Phase 3 scratch scripts (S1–S5b, audit trail)
├── notebooks/
│   ├── 00_environment_test.ipynb         # Environment verification
│   ├── 01_data_collection.ipynb          # Phase 1 — data collection & quality assurance
│   ├── 02_cleaning_alignment.ipynb       # Phase 2 — cleaning, alignment, harmonisation
│   └── 03_stationarity_structural_breaks.ipynb  # Phase 3 — stationarity & structural breaks
├── outputs/
│   └── figures/                          # Phase-specific visualisations (Phase 2 + Phase 3 panels)
├── .env.example                          # Template for FRED API key
├── requirements.txt                      # Python dependencies (v1.1, scipy pinned)
├── README.md                             # This file
├── ProjectScope_v1.md                    # Full analytical scope (§1–§14)
└── ProjectDriven.md                      # Living decision log (D-001 through D-033)
```

---

## How to Reproduce

### Prerequisites

- Python 3.10 or later (developed against 3.10.20)
- Conda environment recommended; dependencies in `requirements.txt`
- FRED API key: register at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) and place in `.env` as `FRED_API_KEY=...`

### Setup

```bash
git clone <repo-url> inflation-forecasting-analysis
cd inflation-forecasting-analysis

# Create environment
conda create -n p3_inflation python=3.10 -y
conda activate p3_inflation
pip install -r requirements.txt

# Configure FRED credentials
cp .env.example .env
# Edit .env and paste your FRED_API_KEY
```

### Regenerate `data/raw/`

The Phase 1 collection is fully automated. Run end-to-end via the notebook:

```bash
jupyter lab notebooks/01_data_collection.ipynb
# Run All → takes approximately 2–3 minutes (FRED API + one D-021 Euro-area M2 fetch)
```

The notebook archives any prior `data/raw/` contents, performs the Phase 1 v2 rebuild, executes the D-021 Germany M2 resolution (§8.5), and re-runs the freshness diagnostic. One manual step remains: downloading `zmi2020s.csv` from Japan's Statistics Bureau ([https://www.stat.go.jp/data/cpi/](https://www.stat.go.jp/data/cpi/)) into `data/raw/_manual/` per D-016. The notebook displays clear instructions when this file is missing.

### Regenerate `data/processed/`

Phase 2 cleaning and alignment is reproducible via either the notebook or the CLI orchestrator:

```bash
# Option A: notebook narrative
jupyter lab notebooks/02_cleaning_alignment.ipynb

# Option B: CLI
python scripts/rebuild_processed.py
```

Both paths call `src.preprocessing.build_all_processed()` and write identical outputs.

### Regenerate Phase 3 artefacts

Phase 3 audit CSVs are produced by the `scripts/phase3_step*.py` series (S1 → S5b), each focused on a single analytical question. All logic also lives in the reusable modules (`src/stationarity.py`, `src/structural_breaks.py`); the Portfolio notebook exercises the same functions from a single narrative layer:

```bash
# Run scratch scripts in sequence (optional — audit trail)
python scripts/phase3_step1_adf_kpss_levels.py
python scripts/phase3_step2_differencing.py
python scripts/phase3_step3_cpi_decision_and_registry.py
python scripts/phase3_step4_chow_structural_breaks.py
python scripts/phase3_step5_quandt_andrews.py
python scripts/phase3_step5b_quandt_andrews_trim10.py

# Or simply run the Portfolio notebook (Run All, ~2–3 minutes)
jupyter lab notebooks/03_stationarity_structural_breaks.ipynb
```

---

## Phase 3 Highlights

Four interwoven signature findings:

1. **Statistical robustness.** Classical vs HAC Chow agree 12/12 on reject/non-reject; HAC vs COVID-dummy HAC agree 8/8. Autocorrelation, heteroskedasticity, and COVID outliers do not drive the conclusions. All nine HAC rejects survive Bonferroni at family-wise α = 0.05/12.

2. **GFC_2008 is USA-specific.** Only USA rejects at α = 0.05 at the Global Financial Crisis break. Japan, UK, and Germany show p-values between 0.06 and 0.53 — consistent with differential ECB/BOJ liquidity responses dampening the CPI–macro relationship's response to 2008.

3. **Break channel differs by country at the same date.** At ENERGY_2022, the dominant regressor is `POLICY_RATE` for USA (z = +5.95), the `const` for Japan (z = +4.98), and `GDP` for UK and Germany (z = +3.58, +2.82). The same calendar-month event operated through different economic channels in different economies — the finding that links project narratives N1 (Phillips Curve), N2 (Monetary Policy Lag), and N3 (Japan's Uniqueness) to one statistical event.

4. **Data-driven break detection confirms the ex-ante specification.** Quandt-Andrews argmax at π₀ = 0.10 falls within ±1 month of 2022-02 for all four countries. USA sup-W = 37.73 exceeds the Andrews 1% critical value (23.04). The data independently pinpoint the break date that ProjectScope identified from economic reasoning alone — the Phase 3 signature finding.

See `notebooks/03_stationarity_structural_breaks.ipynb` for the full narrative and `ProjectDriven.md` entries D-024 through D-033 for decision rationale.

---

## References

- **Andrews, D. W. K.** (1993). *Tests for Parameter Instability and Structural Change with Unknown Change Point.* Econometrica, 61(4), 821–856.
- **Chow, G. C.** (1960). *Tests of Equality Between Sets of Coefficients in Two Linear Regressions.* Econometrica, 28(3), 591–605.
- **Chow, G. C., & Lin, A. L.** (1971). *Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series.* Review of Economics and Statistics, 53(4), 372–375.
- **Diebold, F. X., & Mariano, R. S.** (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics, 13(3), 253–263.
- **Fernald, J. G., Hsu, E., & Spiegel, M. M.** (2021). *Is China fudging its GDP figures? Evidence from trading partner data.* Journal of International Money and Finance.
- **Hansen, B. E.** (1997). *Approximate Asymptotic P Values for Structural-Change Tests.* Journal of Business & Economic Statistics, 15(1), 60–67.
- **Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y.** (1992). *Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root.* Journal of Econometrics, 54(1–3), 159–178.
- **Newey, W. K., & West, K. D.** (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708.
- **Schwert, G. W.** (1989). *Tests for Unit Roots: A Monte Carlo Investigation.* Journal of Business & Economic Statistics, 7(2), 147–159.

---

## Author

Portfolio author: **Kouta** (Melbourne, Australia). This project documents analytical decisions in English for reviewability by an international audience; commentary on workflow and iteration is maintained separately in Japanese.

See [`ProjectDriven.md`](ProjectDriven.md) for the complete decision log and [`ProjectScope_v1.md`](ProjectScope_v1.md) for the full analytical scope.

---

*Last updated: Phase 3 complete — four main-country datasets classified, Chow/Quandt-Andrews breaks characterised, reusable `src/` module architecture extended to v0.3.0 with 4 modules and 60 total exports. Next: Phase 4 feature engineering.*
