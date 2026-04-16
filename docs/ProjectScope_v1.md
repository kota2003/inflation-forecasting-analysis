# Project Scope — Version 2.0
## Project Title

**Inflation Prediction and Economic Signal Analysis: A Multi-Country Time-Series Study**

> **Version note:** Version 2.0 reflects full scope decisions made in design phase.
> Countries: USA, Japan, UK, Germany (main) + China (supplementary).
> Period: 2000–present. Core model: VAR with Granger causality.
> All sections supersede the original v1.0 draft.

---

## 1. Objective

Develop a multi-country time-series framework to forecast inflation, identify key economic drivers,
and provide interpretable, policy-relevant insights grounded in macroeconomic theory.

This project is intentionally positioned at the intersection of:

- **Statistical rigour** — proper time-series handling, structural break testing, stationarity
- **Economic interpretability** — named narratives, lag effects, cross-country comparisons
- **Portfolio differentiation** — distinct from P2 (ML), P6 (DL), P5 (pipeline), P7 (critique)

---

## 2. Problem Statement

Inflation is one of the most consequential macroeconomic indicators, yet it is notoriously
difficult to predict due to complex interactions between multiple economic variables,
structural regime changes, and external shocks.

This project asks:

- Can historical economic data reliably forecast inflation across different economies?
- Which indicators drive inflation — and does this vary by country?
- How do lag effects operate? When does a rate hike actually slow inflation?
- What happens to models when structural breaks occur (GFC, COVID, 2022 energy shock)?

---

## 3. Target Users

- Thinktank analysts (NRI and equivalents)
- Policy makers and economic consultants
- Financial analysts
- Data science recruiters evaluating economic domain knowledge

---

## 4. Economic Narratives (Core of the Project)

This project is organised around **three named economic narratives**.
Each narrative is tested empirically and discussed critically.

### Narrative 1 — The Phillips Curve
> *Does lower unemployment still predict higher inflation?*

- Classic macroeconomic relationship: unemployment ↓ → inflation ↑
- Test whether this holds across all four countries
- Examine whether the relationship broke down post-2008 (flattening of the curve)

### Narrative 2 — Monetary Policy Lag Effects
> *How many months after a rate hike does inflation actually respond?*

- Central banks raise rates to control inflation — but with a delay
- Use VAR Impulse Response Functions (IRF) to quantify the lag
- Compare: does the Fed's rate hike affect inflation faster than the BOJ's?

### Narrative 3 — Japan's Uniqueness
> *Why did Japan experience 30 years of deflation — and why did that change in 2022?*

- Japan is the only developed economy with sustained deflation (1990s–2010s)
- Abenomics and yield curve control as a natural experiment
- 2022 inflation driven by yen depreciation and energy costs — structurally different from the West
- This narrative alone differentiates the project for Japan-focused employers (NRI etc.)

---

## 5. Country Selection

### Main Analysis (4 Countries)

| Country | Role | Key Analytical Value |
|---------|------|----------------------|
| 🇺🇸 USA | Benchmark | FRED home country; richest data; global reference |
| 🇯🇵 Japan | Structural contrast | Deflation → inflation transition; BOJ policy experiments |
| 🇬🇧 UK | Structural change | Brexit impact; BOE independent policy |
| 🇩🇪 Germany | Euro area proxy | ECB constraint; energy shock exposure |

### Supplementary (1 Country)

| Country | Role | Handling |
|---------|------|----------|
| 🇨🇳 China | Supplementary reference | Included with explicit data reliability caveat |

> **China Data Note (to be documented in ProjectDriven.md and README):**
> Official Chinese macroeconomic data carries known reliability concerns:
> GDP figures align closely with political targets; CPI basket composition is opaque;
> unemployment statistics exclude rural and migrant workers.
> China is therefore excluded from the main statistical models
> and included only as a supplementary descriptive comparison.
> This decision reflects data literacy, not analytical avoidance.

---

## 6. Analysis Period

**2000 — Present (monthly frequency)**

Rationale:
- Captures three major structural regimes: pre-GFC, post-GFC low-inflation era, post-COVID inflation surge
- Includes the 2008 Global Financial Crisis (structural break)
- Includes the 2020 COVID shock (structural break)
- Includes the 2022 energy/inflation shock (structural break)
- Long enough for VAR and ARIMA to be statistically valid

---

## 7. Data Sources

### Primary: FRED API (`fredapi` Python library)

All main indicators for USA, Japan, UK, and Germany are available via FRED API.

| Indicator | USA | Japan | UK | Germany |
|-----------|-----|-------|----|---------|
| CPI (YoY) | CPIAUCSL | JPNCPIALLMINMEI | GBRCPIALLMINMEI | DEUCPIALLMINMEI |
| Policy Rate | FEDFUNDS | IRSTCB01JPM156N | IRSTCB01GBM156N | IRSTCB01DEM156N |
| Unemployment | UNRATE | LRUNTTTTJPM156S | LRUNTTTTGBM156S | LRUNTTTTDEM156S |
| GDP Growth | GDP | JPNRGDPEXP | UKNGDP | CLVMNACSCAB1GQDE |
| M2 Supply | M2SL | MYAGM2JPM189S | — | — |

### Supplementary: OECD API or manual download
- Used where FRED coverage is incomplete (e.g. UK M2, China indicators)
- China CPI: National Bureau of Statistics (NBS) — used with explicit reliability caveat

---

## 8. Key Variables

### Target Variable
- **CPI YoY (%)** — monthly, per country

### Feature Variables

| Variable | Economic Role | Lag Hypothesis |
|----------|--------------|----------------|
| Policy Interest Rate | Monetary policy tool | t-3 to t-12 |
| Unemployment Rate | Phillips Curve proxy | t-1 to t-6 |
| GDP Growth Rate | Demand-side pressure | t-1 to t-3 |
| M2 Money Supply | Monetary quantity theory | t-3 to t-9 |
| Consumer Sentiment (optional) | Expectations channel | t-1 to t-3 |

---

## 9. Methodology

### Phase 0 — Environment & Repository Setup
- Create GitHub repository with full folder structure
- Set up `fredapi` connection and API key management (`.env`)
- Document all design decisions in `ProjectDriven.md`

---

### Phase 1 — Data Collection
- Fetch all indicators via FRED API for all 4 main countries
- Collect China data from NBS/OECD with source documentation
- Store raw data as CSV with timestamps
- Inspect: shape, frequency, missing values, coverage

---

### Phase 2 — Data Cleaning & Alignment
- Convert all series to monthly frequency
- Align time index: 2000-01 to present
- Handle missing values:
  - Short gaps (≤3 months): linear interpolation
  - Structural gaps: document and exclude
- Standardise country naming and column conventions

---

### Phase 3 — Stationarity & Structural Break Analysis

#### Stationarity Testing (Required for VAR/ARIMA)
- **ADF Test (Augmented Dickey-Fuller)** for each series per country
- If non-stationary: apply first differencing
- Document all transformation decisions

#### Structural Break Detection
- **Chow Test** at known break points:
  - 2008-09 (Global Financial Crisis)
  - 2020-03 (COVID shock)
  - 2022-02 (Energy/inflation shock)
- If breaks confirmed: introduce **dummy variables** or **split sample analysis**
- This is critical — ignoring structural breaks produces misleading models

---

### Phase 4 — Feature Engineering

- **Lag features**: t-1, t-3, t-6, t-12 for all indicators
- **Rolling statistics**: 3-month and 12-month rolling mean
- **Percentage changes**: MoM and YoY transformations
- **Anomaly flags**: COVID period dummy, GFC period dummy

---

### Phase 5 — Exploratory Analysis

- Time series plots: CPI trends across all 4 countries (one combined chart)
- Correlation heatmaps per country
- Cross-country comparison: Phillips Curve scatter plots
- ACF / PACF plots for ARIMA order identification
- Japan-specific: deflation period highlighted separately

---

### Phase 6 — Modelling

#### Model 1: ARIMA / SARIMA (Baseline)
- Per-country univariate inflation forecast
- Demonstrates time-series fundamentals
- Establishes benchmark accuracy

#### Model 2: VAR — Vector Autoregression (Core Model)
- Multivariate: CPI + Interest Rate + Unemployment + GDP + M2
- Per-country estimation
- **Granger Causality Tests**: does interest rate Granger-cause inflation?
- **Impulse Response Functions (IRF)**: visualise lag effect of rate hike on CPI
- **Forecast Error Variance Decomposition (FEVD)**: which variable explains most CPI variance?

#### Model 3: Ridge Regression with Lag Features (ML comparison)
- Structured feature matrix with engineered lags
- Regularisation to handle multicollinearity
- Feature importance for economic interpretation
- Connects methodologically to P2

---

### Phase 7 — Evaluation

| Metric | Purpose |
|--------|---------|
| RMSE | Primary forecast accuracy |
| MAE | Interpretable error in % points |
| AIC / BIC | Model selection for ARIMA/VAR |
| Diebold-Mariano Test | Statistical test: is Model A significantly better than Model B? |
| Train vs Test split | Time-based (no leakage): train 2000–2019, test 2020–present |

---

### Phase 8 — Interpretation & Economic Discussion

- **Narrative 1**: Phillips Curve — does it hold? Where does it break?
- **Narrative 2**: IRF analysis — quantify monetary policy lag per country
- **Narrative 3**: Japan deep-dive — deflation causes, 2022 reversal
- **China supplementary**: descriptive comparison with reliability caveat
- **Structural break discussion**: how COVID changed inflation dynamics
- **Limitations**: external shocks, model assumptions, China data issues

---

## 10. Outputs

### Analytical Outputs

| Output | Description |
|--------|-------------|
| Model comparison table | RMSE / MAE across ARIMA, VAR, Ridge per country |
| Granger causality matrix | Heatmap: which indicators predict CPI in which country |
| Lag effect summary | Quantified delay (months) from rate hike to CPI response |
| Phillips Curve analysis | Scatter + regression line per country, pre/post GFC split |

### Visual Outputs

| Visual | Description |
|--------|-------------|
| 4-country CPI comparison | One chart, 2000–present, all countries overlaid |
| Prediction vs Actual | Per model, per country |
| Impulse Response Functions | Rate shock → CPI response, per country |
| Granger Causality Heatmap | Country × Indicator matrix |
| Structural break chart | CPI with GFC / COVID / 2022 shaded |
| Residual plots | Diagnostics per model |
| Japan highlight chart | Deflation period + Abenomics + 2022 shift |

---

## 11. Tech Stack

| Library | Purpose |
|---------|---------|
| Python ≥ 3.10 | Core language |
| `fredapi` | FRED API data collection |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `statsmodels` | ARIMA, VAR, ADF test, Granger test, IRF |
| `scikit-learn` | Ridge Regression, train/test split, metrics |
| `matplotlib` / `seaborn` | Visualisation |
| `python-dotenv` | API key management |
| `jupyter` | Notebook environment |

---

## 12. Repository Structure

```
project_inflation_analysis/
│
├── README.md
├── requirements.txt
├── ProjectDriven.md          ← Living decision log
├── .gitignore
├── .env.example              ← API key template (never commit .env)
│
├── docs/
│   ├── ProjectScope.md       ← This document
│   ├── methodology.md        ← Completed in Phase 8
│   └── findings.md           ← Completed in Phase 8
│
├── data/
│   ├── raw/                  ← Not tracked by Git
│   └── processed/            ← Not tracked by Git
│
├── notebooks/
│   ├── 00_environment_test.ipynb
│   ├── 01_data_collection.ipynb
│   ├── 02_cleaning_alignment.ipynb
│   ├── 03_stationarity_structural_breaks.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_eda.ipynb
│   ├── 06_arima_baseline.ipynb
│   ├── 07_var_model.ipynb
│   ├── 08_ridge_regression.ipynb
│   └── 09_evaluation_interpretation.ipynb
│
├── src/
│   ├── data_loader.py         ← FRED API functions
│   ├── preprocessing.py       ← Cleaning, alignment, differencing
│   ├── stationarity.py        ← ADF test, Chow test wrappers
│   ├── feature_engineering.py ← Lag features, rolling stats
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── var_model.py
│   │   └── ridge_model.py
│   └── evaluation.py          ← RMSE, MAE, Diebold-Mariano
│
├── models/
│   └── saved_models/
│
├── outputs/
│   ├── figures/
│   └── forecasts/
```

---

## 13. ProjectDriven.md — Key Decisions to Document

The following decisions must be logged with rationale:

| Decision | Rationale |
|----------|-----------|
| 4 main countries selected | Diverse inflation regimes; all available via FRED API |
| China excluded from main analysis | Data reliability concerns; documented as data literacy |
| Period: 2000–present | Captures 3 structural regimes; sufficient length for VAR |
| VAR as core model | Captures inter-variable dynamics; enables IRF and Granger |
| Structural breaks treated explicitly | Ignoring them invalidates model assumptions |
| Train: 2000–2019 / Test: 2020–present | Time-based split; no leakage; COVID period as stress test |

---

## 14. README Outline

### 1. Introduction
- Why inflation matters
- Why multi-country comparison

### 2. Economic Narratives
- Phillips Curve
- Monetary policy lag
- Japan's uniqueness

### 3. Data
- FRED API
- Countries and indicators
- China data caveat

### 4. Methodology
- Stationarity and structural breaks
- ARIMA → VAR → Ridge progression

### 5. Results
- Model comparison
- IRF and Granger findings
- Country-by-country highlights

### 6. Interpretation
- What the data says — and what it cannot say
- Policy implications

### 7. Limitations
- External shocks
- China data reliability
- Model assumptions

---

## 15. Success Criteria

| Criterion | Description |
|-----------|-------------|
| Statistical rigour | ADF tests, structural break handling, no data leakage |
| Economic interpretability | Three named narratives supported by results |
| Model progression | ARIMA → VAR → Ridge, each adding analytical value |
| Cross-country comparison | Meaningful insights from 4-country analysis |
| Japan narrative | Clearly explained deflation and 2022 reversal |
| China handling | Honest, documented, data-literate treatment |
| Reproducibility | Full pipeline from FRED API to outputs via documented code |
| Portfolio fit | Clearly differentiated from P2, P5, P6, P7 |

---

## 16. Risks & Limitations

| Risk | Mitigation |
|------|-----------|
| External shocks not in model | Structural break dummies; documented in limitations |
| China data reliability | Excluded from main models; supplementary only |
| VAR overfitting (too many variables) | Lag order selection via AIC/BIC; parsimony principle |
| COVID period distorts test set | Explicitly discussed; stress test framing |
| FRED API changes / downtime | Cache raw data locally after first pull |
| UK M2 data gap in FRED | Use OECD supplementary source; document substitution |

---

## 17. Portfolio Positioning

This project demonstrates:

| Skill | Evidence |
|-------|---------|
| Time-series fundamentals | ADF, ARIMA, train/test without leakage |
| Advanced econometrics | VAR, Granger causality, IRF, FEVD |
| Economic domain knowledge | Three named narratives grounded in theory |
| Data sourcing via API | FRED API with key management |
| Critical data thinking | China caveat; structural break discussion |
| Cross-country analysis | 4 economies, distinct regimes |
| Reproducible workflow | Full pipeline, documented decisions |

**Strongest fit for:** NRI, Nomura Research, Deloitte / McKinsey economics practice,
policy-oriented data science roles, economic consulting.

---

## 18. Phase Plan

| Phase | Description | Key Output |
|-------|-------------|-----------|
| Phase 0 | Environment & Git setup | Repo, `.env`, `ProjectDriven.md` |
| Phase 1 | Data collection via FRED API | `raw/*.csv` |
| Phase 2 | Cleaning & alignment | `processed/*.csv` |
| Phase 3 | Stationarity & structural breaks | ADF results, Chow test results |
| Phase 4 | Feature engineering | Lag matrix, rolling features |
| Phase 5 | EDA | 4-country CPI chart, correlation heatmaps |
| Phase 6 | Modelling (ARIMA → VAR → Ridge) | Trained models, forecasts |
| Phase 7 | Evaluation | Model comparison table, IRF, Granger heatmap |
| Phase 8 | Interpretation & packaging | README, findings.md, methodology.md |

---

## 19. Immediate Next Steps (START HERE)

1. Create GitHub repository: `project_inflation_analysis`
2. Create folder structure as defined above
3. Create `.env.example` and add `FRED_API_KEY=your_key_here`
4. Register for FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
5. Open `00_environment_test.ipynb`
6. Run: `pip install fredapi pandas numpy statsmodels scikit-learn matplotlib seaborn python-dotenv`
7. Test FRED API connection: fetch `CPIAUCSL` for USA
8. Confirm data returns correctly before proceeding to Phase 1

---

# END OF SCOPE — Version 2.0
