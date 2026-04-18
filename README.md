# Inflation Prediction and Economic Signal Analysis

A Multi-Country Time-Series Study of Central-Bank Policy, Unemployment, and Money Supply as Drivers of Inflation

---

## Overview

This project builds multi-country vector autoregression (VAR) models to decompose inflation dynamics into their contributing channels — central-bank policy rates, unemployment, real activity, and money supply — for four advanced economies with structurally distinct inflation histories: **USA, Japan, UK, and Germany**. China is included as a supplementary descriptive comparison with explicit reliability caveats.

Three narrative questions drive the analysis:

- **N1 — The Phillips Curve in Practice.** Did the post-GFC decade genuinely break the inflation–unemployment relationship, and did the 2022 shock restore it?
- **N2 — Monetary Policy Lag.** How many months after a central bank tightens does inflation actually respond? Do impulse response functions match the "6-to-18 months" textbook claim?
- **N3 — Japan's Uniqueness.** Why did 30 years of low-zero inflation end abruptly in 2022? Is the mechanism pass-through from global commodity prices, or a domestic structural break?

The analysis combines classical econometrics (ARIMA baselines, VAR core, Granger causality, impulse response functions, forecast error variance decomposition) with a regularised machine-learning comparator (Ridge regression on engineered lag features). Every design decision is recorded in [`ProjectDriven.md`](ProjectDriven.md); the full analytical scope is in [`ProjectScope_v1.md`](ProjectScope_v1.md).

---

## Why Portfolio Project 3 Takes This Angle

This is the third project in a seven-project data-science portfolio targeting Japanese technology-consulting roles. P1 demonstrated machine-learning engineering on structured customer data; P2 covered feature engineering and classification with interpretability tools; this P3 demonstrates **classical econometric rigour combined with modern data engineering**. The project deliberately prioritises decision documentation, source auditing, and reproducibility over algorithmic novelty — the skills most valued in consulting contexts where analytical defensibility matters more than headline accuracy.

---

## Project Status

| Phase | Focus | Status |
|---|---|---|
| Phase 0 | Project scoping, country selection, narrative definition | ✅ Complete |
| Phase 1 | Data collection — 25 series, 5 countries × 5 indicators, multi-source rebuild | ✅ Complete |
| Phase 2 | Data cleaning, unit harmonisation, temporal alignment | ✅ Complete |
| Phase 3 | Stationarity testing (ADF), structural-break testing (Chow) | ⏳ Next |
| Phase 4 | Feature engineering (lags, rolling statistics, regime dummies) | Pending |
| Phase 5 | Exploratory data analysis & cross-country narrative visualisation | Pending |
| Phase 6 | Model estimation — ARIMA, VAR, Ridge | Pending |
| Phase 7 | Evaluation — Diebold-Mariano, walk-forward validation | Pending |
| Phase 8 | Interpretation — Granger maps, IRF plots, narrative synthesis | Pending |

As of this writing, the `data/processed/` directory contains four fully-observed main-country datasets (USA, Japan, UK, Germany — NaN-free, 2001-01 onwards) and one supplementary dataset (China, sparse by design). All are VAR-ingestion-ready via `src.data_loader.load_processed_main()`.

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
│       └── phase2_ip_scout*.csv         # Chow-Lin due diligence (rejected)
├── src/                                  # Reusable Python modules (imported by notebooks)
│   ├── __init__.py                       # Package v0.2.0
│   ├── data_loader.py                    # I/O helpers for raw & processed datasets
│   └── preprocessing.py                  # Phase 2 transformation functions
├── scripts/
│   └── rebuild_processed.py              # CLI orchestrator — regenerates data/processed/
├── notebooks/
│   ├── 00_environment_test.ipynb         # Environment verification
│   ├── 01_data_collection.ipynb          # Phase 1 — data collection & quality assurance
│   └── 02_cleaning_alignment.ipynb       # Phase 2 — cleaning, alignment, harmonisation
├── outputs/
│   └── figures/                          # Phase-specific visualisations
├── .env.example                          # Template for FRED API key
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── ProjectScope_v1.md                    # Full analytical scope (§1–§14)
└── ProjectDriven.md                      # Living decision log (D-001 through D-023)
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

Two equivalent paths are available:

**Path A — CLI (fastest, non-interactive):**
```bash
python scripts/rebuild_processed.py
# ~15 seconds; reads data/raw/*.csv, writes 5 CSVs + audit log
```

**Path B — Notebook (narrated, portfolio-readable):**
```bash
jupyter lab notebooks/02_cleaning_alignment.ipynb
# Run All → ~30 seconds; same transformations, plus mathematical commentary,
# economic-history validation tables, and 3 portfolio-grade figures
```

Both paths import from `src/preprocessing.py` — the single source of truth for all transformation logic.

### Regenerate audit/documentation CSVs

If `data/documentation/` is missing Phase 2 audit CSVs (e.g., after a fresh clone or cleanup), regenerate them via:

```bash
python scripts/regenerate_phase2_audits.py
```

This rebuilds (a) `phase2_m2_yoy_validation.csv` — M2 YoY transformation validation with economic-history peak-date checks, (b) `phase2_cleaning_log.csv` — per-file audit of the processed/ outputs, and (c) `phase2_germany_m2_scout.csv` — D-021 scout log (requires `FRED_API_KEY` in `.env`; skipped gracefully otherwise).

---

## Data Sources

| Source | Role | Series used |
|---|---|---|
| [FRED](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis) | Primary — 23 of 25 series | CPI, policy rates, unemployment, GDP, M2 across USA/JPN/UK/GER/CHN |
| [Statistics Bureau of Japan](https://www.stat.go.jp/data/cpi/) (総務省統計局) | Phase 1 v2 rebuild | Japan CPI (`zmi2020s.csv`, 2020-base, per D-016) |
| [World Bank](https://data.worldbank.org/) | Phase 1 | China unemployment (annual, SL.UEM.TOTL.ZS) |
| [UK ONS](https://www.ons.gov.uk/) | Chow-Lin due diligence (retained, unused) | UK Index of Production K222 |

---

## Honest Data Integrity Caveats

Reviewers and future collaborators should read these before interpreting the results:

1. **Germany M2 is Euro-area-wide, not German national.** Following 1999 euro adoption, Germany does not maintain a national monetary aggregate; D-021 adopts `MABMM301EZM657S` (Euro-area M2) as the institutionally-correct substitute. Cross-country M2 comparisons for Germany must be read as national-vs-currency-union, not national-vs-national.

2. **UK and Germany CPI end at 2025-03.** OECD harmonised publication for these two lags approximately 13 months. Their effective windows end 2025-03 and their datasets contain 291 rows (vs 298 for USA/JPN). Sufficient for VAR estimation but the latest inflation cycle (Q2-2025 onwards) is out-of-sample for these two.

3. **Monthly GDP is linearly interpolated from quarterly.** D-018 adopts linear interpolation on the quarterly level; within-quarter GDP variation in the processed data is an interpolation artefact, not a measurement. The Chow-Lin due diligence archive (`data/documentation/phase2_ip_scout*.csv`) documents why a more sophisticated method was evaluated and rejected as disproportionate to GDP's role in the VAR (one of 5 regressors under short lags).

4. **M2 YoY was computed from MoM % source for 3 of 4 main countries.** The `MABMM301...657S` FRED series documentation labels them "growth rate same period previous year," but their empirical distributions (max ~2-5 %, std < 1 %) are incompatible with YoY growth. D-012 (amended) corrects this via a cumulative-product conversion from MoM. Post-conversion peak dates align with known monetary-policy episodes across all four countries — the empirical validation is in `notebooks/02_cleaning_alignment.ipynb` §5.5.

5. **China is supplementary only.** China's CPI (WARNING freshness), policy rate (WARNING), M2 (CRITICAL, ends 2018-12), and annual unemployment are incorporated for descriptive cross-country context only. D-001 excludes China from the main VAR due to documented reliability concerns (GDP targets, CPI basket opacity, urban-only unemployment coverage).

6. **Structural breaks at 2008-09, 2020-03, and 2022-02 are untreated in Phase 2.** The VAR coefficients estimated on 2001–2019 training data (per D-005) may not generalise to post-2020 test data without explicit regime-dummy or split-sample treatment. Phase 3 will test for structural breaks at these three candidate dates via Chow tests, and Phase 6 will adjust the VAR specification accordingly.

---

## Portfolio Context

This project is the third in a seven-project data-science portfolio prepared for Japanese technology-consulting and data-platform roles.

| # | Project | Analytical focus | Status |
|---|---|---|---|
| P1 | Supermarket Price Analysis and ML Classification (Product Categorization) | Data engineering, cleaning, and baseline ML classification | Complete |
| P2 | Customer Segmentation and Business Insights (Marketing Analytics) | Feature engineering, segmentation, interpretability | Complete |
| **P3** | **Inflation Prediction and Economic Signal Analysis** | **Classical econometrics + data engineering** | **In progress — Phase 2 complete** |
| P4 | Bank Churn Prediction | Imbalanced-class ML, SHAP interpretability | Planned |
| P5 | Time-Series Sales Forecasting (Retail) | Prophet / LSTM / tree-based forecasting | Planned |
| P6 | Deep Learning Classification (Image Recognition) | CNN architectures, transfer learning | Planned |
| P7 | End-to-End ML Pipeline with Deployment | MLOps, cloud deployment, API serving | Planned |

P3 deliberately focuses on econometric methodology and analytical defensibility. P5 will cover applied time-series forecasting from a different angle (demand forecasting with tree-based and deep models); P3's VAR framework is complementary, emphasising causal interpretation over forecast accuracy.

---

## Key References

- **Phillips, A. W.** (1958). *The Relation between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom, 1861–1957.* Economica.
- **Friedman, M.** (1956). *The Quantity Theory of Money — A Restatement.* Studies in the Quantity Theory of Money.
- **Taylor, J. B.** (1993). *Discretion versus Policy Rules in Practice.* Carnegie-Rochester Conference Series on Public Policy.
- **Sims, C. A.** (1980). *Macroeconomics and Reality.* Econometrica, 48(1), 1–48.
- **Chow, G. C., & Lin, A. L.** (1971). *Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series.* Review of Economics and Statistics, 53(4), 372–375.
- **Diebold, F. X., & Mariano, R. S.** (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics, 13(3), 253–263.
- **Fernald, J. G., Hsu, E., & Spiegel, M. M.** (2021). *Is China fudging its GDP figures? Evidence from trading partner data.* Journal of International Money and Finance.

---

## Author

Portfolio author: **Kouta** (Melbourne, Australia). This project documents analytical decisions in English for reviewability by an international audience; commentary on workflow and iteration is maintained separately in Japanese.

See [`ProjectDriven.md`](ProjectDriven.md) for the complete decision log and [`ProjectScope_v1.md`](ProjectScope_v1.md) for the full analytical scope.

---

*Last updated: Phase 2 complete — 4 main + 1 supplementary datasets in `data/processed/`, VAR-ready, reusable `src/` module architecture established. Next: Phase 3 stationarity & structural-break testing.*
