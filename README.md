# Inflation Prediction and Economic Signal Analysis

A multi-country time-series study of inflation dynamics across five economies (USA, Japan, UK, Germany, and China), combining classical econometric methods with modern data-engineering practice.

**Status:** Phase 1 complete (data collection & quality assurance with multi-source rebuild) · Phase 2 in progress (cleaning & alignment)

---

## Why This Project

Inflation is one of the most consequential macroeconomic indicators, yet remains notoriously hard to predict. The dynamics differ sharply across countries — the same model cannot explain both USA's 2022 surge and Japan's thirty-year deflation.

This project is intentionally positioned at the intersection of:

- **Statistical rigour** — proper time-series handling, structural break testing, stationarity treatment
- **Economic interpretability** — three named narratives, policy lag quantification, cross-country contrast
- **Data engineering maturity** — multi-source architecture, automated diagnostic, resilient API layer

---

## Three Named Economic Narratives

The project is organised around three testable narratives, each corresponding to a school of macroeconomic thought:

### Narrative 1 — The Phillips Curve
> *Does lower unemployment still predict higher inflation?*

Classical relationship (Phillips 1958; Samuelson & Solow 1960). Tested across all four main countries, with explicit examination of whether the relationship broke down post-2008 (the "flattening" debate).

### Narrative 2 — Monetary Policy Lag Effects
> *How many months after a rate hike does inflation actually respond?*

Taylor rule operationalisation (Taylor 1993). Quantified via VAR Impulse Response Functions per country. Comparative: does the Fed's rate hike affect inflation faster than the BOJ's?

### Narrative 3 — Japan's Uniqueness
> *Why 30 years of deflation — and why did the pattern reverse in 2022?*

Japan is the only developed economy with sustained deflation (1990s–2010s). The 2022 inflation was structurally different from the West — yen depreciation plus energy-cost pass-through rather than demand-pull. This narrative alone differentiates the project for Japan-focused employers.

---

## Data

### Countries and Indicators

Five economies × five indicators = 25 monthly/quarterly time series from 2000-01 to present.

| Country | Role | CPI | Policy Rate | Unemployment | GDP | M2 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 🇺🇸 USA | Main (benchmark) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 🇯🇵 Japan | Main (structural contrast) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 🇬🇧 UK | Main (post-Brexit) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 🇩🇪 Germany | Main (Euro-area proxy) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 🇨🇳 China | Supplementary | ✓ | ✓ | ✓ (annual) | ✓ | ✓ |

China is **excluded from the main VAR models** due to documented reliability concerns (see `ProjectDriven.md` D-001) and included only as supplementary descriptive comparison.

### Multi-Source Architecture

Phase 1 v1 retrieved all 25 series via the FRED API. A Phase 2 diagnostic subsequently identified six series as critically stale (>24 months outdated). The Phase 1 v2 rebuild adopted a three-tier source hierarchy:

```
Tier 1 — FRED API (19 series)
            ↓ (if FRED stale or missing)
Tier 2 — FRED alternative Series IDs (4 series, scout-tested)
            ↓ (if no FRED alternative fresh enough)
Tier 3 — Direct from primary statistical agency
            (Japan Statistics Bureau for Japan CPI: 1 series)
```

Special handling: China unemployment retrieved from the World Bank API (annual only, supplementary per D-001).

### Final Data Quality (post Phase 1 v2)

| Freshness | Count | Detail |
|---|---|---|
| 🟢 FRESH (<6 months) | 15 | |
| 🟡 WARNING (6–24 months) | 8 | Normal publication cadence |
| 🔴 CRITICAL (>24 months) | 2 | Both China (Supplementary accepted) |
| ⚫ MISSING | 0 | |

**All three narratives confirmed viable** (N1 / N2 / N3: ✅ Ready).

See `ProjectDriven.md` D-006 through D-017 for the complete diagnostic, decision, and remediation record.

---

## Methodology

A three-layer modelling progression designed to showcase both econometric depth and ML literacy:

### Layer 1 — ARIMA / SARIMA (Baseline)
Univariate inflation forecast per country. Establishes time-series fundamentals (ACF/PACF, stationarity, order selection) and sets an accuracy benchmark.

### Layer 2 — VAR (Core Model)
Multivariate: CPI × Policy Rate × Unemployment × GDP × M2. Enables Granger causality testing, Impulse Response Functions, and Forecast Error Variance Decomposition. **This is the primary analytical contribution.**

### Layer 3 — Ridge Regression with Lag Features (ML Comparison)
Structured feature matrix with engineered lags. Regularisation for multicollinearity. Feature-importance mapping back to economic interpretation.

### Evaluation
Time-based split (train 2000–2019, test 2020–present). RMSE / MAE primary, AIC/BIC for model selection, Diebold-Mariano test for pairwise model comparison.

See `docs/ProjectScope_v1.md` for the full methodology specification.

---

## Reproducing the Pipeline

### Prerequisites

| # | Requirement | How |
|---|---|---|
| 1 | Python ≥ 3.10 | Use conda or venv |
| 2 | Dependencies | `pip install -r requirements.txt` |
| 3 | FRED API key | Free at https://fred.stlouisfed.org/docs/api/api_key.html → add `FRED_API_KEY=xxx` to `.env` |
| 4 | Japan CPI CSV | Download `zmi2020s.csv` from [stat.go.jp](https://www.stat.go.jp/data/cpi/) (長期時系列データ → 中分類指数 2020基準 全国 月次) → place at `data/raw/_manual/zmi2020s.csv` |

### Run

```bash
# 1. Environment check (verifies Python version, library versions, API connectivity)
jupyter notebook notebooks/00_environment_test.ipynb

# 2. Full data collection pipeline (runs automatically once zmi2020s.csv is in place)
jupyter notebook notebooks/01_data_collection.ipynb
# → fetches all 25 series from FRED + World Bank, runs diagnostic,
#   executes v2 rebuild (FRED replacements + manual Japan CPI import),
#   generates visualisations, writes audit log
#   Total runtime: ~30-60 seconds
```

The notebook handles transient API failures automatically via retry logic with exponential backoff (see D-017). Only the `zmi2020s.csv` step requires human action — structural features of Japan's government statistics pipeline make this unavoidable (documented in D-016).

### Outputs

```
data/
├── raw/                              # 25 series, final v2 state
├── raw/_archive_v1/{timestamp}/      # Archived v1 versions
├── raw/_manual/                      # User-placed external files
├── processed/                        # Phase 2 output (forthcoming)
└── documentation/
    └── phase1v2_rebuild_log.csv      # Complete audit trail

outputs/figures/
├── data_collection_staleness_bar.png
├── data_collection_cpi_comparison.png
└── data_collection_jpn_cpi_v1_vs_v2.png
```

---

## Repository Structure

```
inflation-forecasting-analysis/
├── README.md                         ← This file
├── ProjectDriven.md                  ← Living decision log (D-001 through D-017)
├── requirements.txt
├── .env                              ← FRED_API_KEY (gitignored)
│
├── docs/
│   └── ProjectScope_v1.md            ← Full methodology specification
│
├── notebooks/
│   ├── 00_environment_test.ipynb
│   ├── 01_data_collection.ipynb      ← Full pipeline (Phase 1 complete)
│   ├── 02_cleaning_alignment.ipynb   ← Phase 2 (in progress)
│   ├── 03_stationarity_structural_breaks.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_eda.ipynb
│   ├── 06_arima_baseline.ipynb
│   ├── 07_var_model.ipynb
│   ├── 08_ridge_regression.ipynb
│   └── 09_evaluation_interpretation.ipynb
│
├── data/
│   ├── raw/                          ← .gitignored
│   ├── processed/                    ← .gitignored
│   └── documentation/
│
└── outputs/
    ├── figures/
    └── forecasts/
```

---

## Honest Data Integrity Caveats

Responsible portfolio work documents limitations. This project's residual issues:

1. **UK GDP (WARNING, 6 months outdated)** — Reflects normal ONS quarterly publication cadence, not a pipeline defect
2. **M2 unit heterogeneity** — USA/Germany in level form, Japan/UK in YoY growth. Harmonised in Phase 2 cleaning (D-012)
3. **Germany M2 uses USA Series ID as proxy** — Documented placeholder; Germany-specific scout scheduled in Phase 2
4. **China series (3 CRITICAL including annual unemployment)** — All consistent with supplementary framing per D-001; no impact on main VAR

Full accounting is in `ProjectDriven.md` §Phase 1 Final State.

---

## Key References

- **Phillips, A. W.** (1958). *The Relation between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom, 1861–1957.* Economica.
- **Friedman, M.** (1956). *The Quantity Theory of Money — A Restatement.* Studies in the Quantity Theory of Money.
- **Taylor, J. B.** (1993). *Discretion versus Policy Rules in Practice.* Carnegie-Rochester Conference Series on Public Policy.
- **Fernald, J. G., Hsu, E., & Spiegel, M. M.** (2021). *Is China fudging its GDP figures? Evidence from trading partner data.* Journal of International Money and Finance.
- **Federal Reserve Bank of St. Louis** — FRED Economic Data. https://fred.stlouisfed.org/
- **Statistics Bureau of Japan (総務省統計局)** — Consumer Price Index. https://www.stat.go.jp/data/cpi/
- **World Bank** — SL.UEM.TOTL.ZS Unemployment Total. https://data.worldbank.org/

---

## Project Context

This is **Project 3** of a broader data-science portfolio designed to demonstrate complementary skills:

| Project | Focus |
|---|---|
| P1 | Foundation — SQL, pandas |
| P2 | Classical ML — feature engineering, model selection |
| **P3 (this project)** | **Econometrics — time-series, interpretability, multi-source data** |
| P5 | Data engineering — pipelines, productionisation |
| P6 | Deep learning — sequential models |
| P7 | Analytical critique — hypothesis testing, research quality |

The distinctive contribution of P3 is the combination of **classical econometric rigour** (VAR, Granger, IRF) with **modern data engineering** (multi-source architecture, retry logic, reproducible pipelines). The Phase 1 v2 rebuild — in which a seemingly-complete dataset was audited, found wanting, and strategically remediated — is the clearest showcase of data-literacy depth.
