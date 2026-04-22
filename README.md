# 📈 Multi-Country CPI Inflation Forecasting

> Three-layer econometric and machine-learning study comparing ARIMA, VAR, and Ridge
> on 25 years of CPI data across USA, Japan, UK, and Germany — built for econometric
> researchers, policy-oriented data scientists, and quantitative hiring managers.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-e8743b)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-f7931e?logo=scikit-learn&logoColor=white)
![FRED-API](https://img.shields.io/badge/FRED--API-live-2e7d32)
![Decisions](https://img.shields.io/badge/Decisions-80-1565c0)
![DM--tests](https://img.shields.io/badge/DM--tests-125-6a1b9a)
![Countries](https://img.shields.io/badge/Countries-4-00838f)
![Years](https://img.shields.io/badge/Years-2000--2025-8e44ad)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why Multi-Country Econometrics?](#-why-multi-country-econometrics)
3. [Pipeline](#️-pipeline)
4. [Three-Layer Model Architecture](#-three-layer-model-architecture)
5. [Results](#-results)
6. [Key Findings](#-key-findings)
7. [Limitations](#️-limitations)
8. [Reproducibility](#-reproducibility)
9. [Tech Stack](#️-tech-stack)
10. [Docs](#-docs)
11. [Repository Structure](#-repository-structure)

---

## 📌 Project Overview

Inflation forecasting is one of the oldest and hardest problems in applied
econometrics — and the one central bankers, pension fund managers, and sovereign
debt analysts still get wrong. This project builds and compares **three model
layers** — univariate ARIMA, multivariate VAR, and high-dimensional Ridge
regression — on monthly CPI year-on-year growth across four major economies
over 2000–2025, and evaluates them under **125 paired Diebold-Mariano tests**
across three loss-function variants plus a COVID-origin sensitivity re-run.

> **Goal:** Demonstrate a portfolio-grade end-to-end forecasting pipeline with
> econometric rigour (Granger, IRF, FEVD, structural breaks) and ML discipline
> (walk-forward CV, leakage-guarded pipelines, regularisation paths) — with
> every analytical decision logged, every narrative claim cross-lens-verified,
> and one headline finding revised in place by subsequent evidence.

### Target Users
- 🏦 **Econometric researchers** — multi-country inflation dynamics with policy-transmission lens
- 📊 **Data science recruiters** — end-to-end pipeline with 80 logged decisions and honest narrative revision
- ⚙️ **ML engineers** — walk-forward validation, leakage-guarded pipelines, and reusable `src/` architecture

### Dataset at a Glance

| | |
|---|---|
| **Countries** | USA, Japan, UK, Germany (+ China supplementary) |
| **Period** | January 2000 – early 2025 (monthly frequency) |
| **Target** | `CPI_YOY` — year-on-year CPI growth rate (%) |
| **Source series** | 25 (CPI, policy rate, M2, unemployment, oil, GDP, ...) |
| **Primary sources** | FRED API · World Bank API · Japan Statistics Bureau · IMF |
| **Features after Phase 4** | 50+ (lags, rolling, regime dummies, cross-country panels) |
| **Walk-forward origins** | 58 (USA, JAPAN) / 51 (UK, GERMANY) |
| **Forecast horizons** | h ∈ {1, 3, 6, 12} months |
| **Paired-DM cells** | 25 (24 β-option + 1 USA dual-form) |
| **Decisions logged** | **80** in [`ProjectDriven.md`](ProjectDriven.md) |

---

## 🌏 Why Multi-Country Econometrics?

Most public inflation-forecasting portfolios stop at a single-country ARIMA
on US CPI. This project deliberately chose a harder, multi-dimensional scope:

| | Typical ML portfolio | This project |
|---|---|---|
| Countries | 1 (usually USA) | **4** (USA / Japan / UK / Germany) |
| Models | 1 (usually ARIMA or LSTM) | **3 layers** (ARIMA + VAR + Ridge) |
| Evaluation | Point-estimate RMSE on one test split | **125 paired Diebold-Mariano tests** + COVID sensitivity |
| Policy lens | Rarely examined | **VAR IRF + Granger + FEVD + Phillips quadrilogy** |
| Documentation | README + notebooks | **80 logged decisions + findings.md + methodology.md** |
| Revision discipline | Retrofitted after the fact | **Honest in-place revision** (D-070 caveated post-Phase-7) |

The multi-country scope creates real econometric challenges absent from
single-country studies: Japan CPI's multi-source data pipeline (FRED stopped
updating mid-2021, requiring Japan Statistics Bureau manual CSV integration),
cross-country heterogeneity in structural breaks (2008 GFC vs 2022 energy
shock timing differs by country), and the N3 finding that Japan's
forecasting topography is fundamentally different from the others across
nine independent methodological lenses.

---

## ⚙️ Pipeline

```
FRED API + World Bank API + Japan Statistics Bureau CSV + IMF
│
├── 01_data_collection.ipynb
│       Multi-source extraction with retry logic → 25 series × 4 countries (+ 1 supplementary)
│
├── 02_cleaning_alignment.ipynb
│       Frequency harmonisation, M2 heterogeneity resolution, content-level NaN audit
│
├── 03_stationarity_structural_breaks.ipynb
│       ADF/KPSS battery, Chow + Quandt-Andrews breaks, transformation registry
│
├── 04_feature_engineering.ipynb
│       Lags, rolling, regime dummies (D-030), 50+ features per country
│
├── 05_eda.ipynb
│       Phillips Curve scan, cross-country correlation heatmaps, N3 structural uniqueness preview
│
├── 06_arima_baseline.ipynb
│       Layer 1 — SARIMA grid search with OOS-saturation stopping rule (D-048)
│
├── 07_var_model.ipynb
│       Layer 2 — 4-country VAR with Granger, orthogonalised IRF, FEVD
│
├── 08_ridge_regression.ipynb
│       Layer 3 — leakage-guarded Ridge pipeline, coefficient stability, Phillips quadrilogy
│
└── 09_evaluation_interpretation.ipynb
        Phase 7 — 25-cell DM battery × 3 variants + COVID-origin sensitivity
```

All notebooks run in sequence (01 → 09). Every cleaning, modelling, and
evaluation decision is logged with rationale, alternatives considered, and
propagation trace in [`ProjectDriven.md`](ProjectDriven.md).

---

## 🎯 Three-Layer Model Architecture

Each layer answers a different forecasting question:

| Layer | Model | Purpose | Key decisions |
|---|---|---|---|
| **Layer 1** | ARIMA / SARIMA | Univariate baseline — captures purely autoregressive dynamics | D-048 three-stage grid search · D-049 Japan uniqueness |
| **Layer 2** | 4-country VAR | Multivariate system with policy-transmission inference | D-050 BIC→AIC lag revision · D-056 orthogonalised IRF |
| **Layer 3** | Ridge regression | High-dimensional ML with leakage-guarded CV | D-065 logspace × TSS(5) · D-068 direct-h walk-forward |

Layer 2 produces inferential evidence (Granger, IRF, FEVD) that Layer 3
cannot; Layer 3 produces forecast-level coefficient stratification (D-067)
that Layer 2 cannot. The **paired Diebold-Mariano battery in Phase 7**
(notebook 09) is where the three layers are head-to-head compared under
matched walk-forward origins.

---

## 📊 Results

### Phase 7 Diebold-Mariano Battery — 25 cells × 3 loss variants

![DM Heatmap](https://raw.githubusercontent.com/kota2003/inflation-forecasting-analysis/main/outputs/figures/phase7_fig1_dm_heatmap.png)

*Figure 1. 25 cells (country × horizon × layer-pair) × 3 DM variants
(standard squared-loss · HAC Newey-West · robust absolute-loss).
Blue = layer 1 (ARIMA or VAR) wins; red = layer 2 (VAR or Ridge) wins.
Bold p-values are significant at α = 0.05.*

### Headline post-COVID-trim finding — 3 signed cells, all ARIMA h=1 wins

After excluding the 2020-03 through 2020-08 walk-forward origins
(D-079 sensitivity), **only three of 25 cells remain significant
at α = 0.05 under standard Diebold-Mariano — and all three are
univariate ARIMA wins at the shortest horizon**:

| Country | Horizon | Layer 1 vs Layer 2 | DM p-value (std) | DM p-value (robust) | Winner |
|---|:---:|---|:---:|:---:|:---:|
| **USA** | h=1 | ARIMA vs VAR   | **0.044** | **0.001** | ⭐ ARIMA |
| **USA** | h=1 | ARIMA vs Ridge | **0.001** | **0.000** | ⭐ ARIMA |
| **UK**  | h=1 | ARIMA vs VAR   | **0.024** | **0.017** | ⭐ ARIMA |

*Post-trim paired sample: n = 52 (USA) / n = 45 (UK). Neither VAR nor
Ridge retains a single significant win under post-trim standard DM.*

### Cross-lens methodology match — USA policy transmission

Three mathematically-independent lenses converge on the same signal:

| Lens | Object | Value | Decision |
|---|---|:---:|:---:|
| VAR orthogonalised IRF | Peak at h = 4 (policy rate → CPI) | **−0.149** | D-056 |
| Ridge first-difference | `POLICY_RATE_lag3` standardised coef | **−0.136** | D-067, D-071 |
| Paired DM USA h=1 | ARIMA-VAR standard / robust | **p=0.014 / p=0.0001** | D-078 |

Two estimators from two model families agreeing within 10% of each other
on magnitude, sign, and lag is the project's strongest
monetary-transmission finding — and is how the "three-lens match
methodology" is operationalised in `methodology.md`.

---

## 💡 Key Findings

### Finding 1 — ARIMA h=1 Univariate Dominance Survives Three Robustness Tests
**3 of 25 post-trim standard-DM-significant cells, all ARIMA h=1 wins.**

Under three independent robustness lenses (standard DM, HAC DM, COVID-origin
trim), the univariate ARIMA baseline beats both VAR and Ridge at h=1 for
USA and UK. The USA h=1 ARIMA-Ridge p-value shifts from 0.135 (full sample)
to **0.001** (COVID-trimmed) — a nine-order-of-magnitude change driven
entirely by removing six walk-forward origins. This is the project's most
portfolio-memorable "simplicity sometimes wins" finding.

---

### Finding 2 — Ridge Architectural Advantage is COVID-Era-Specific
**D-070 MASE claim preserved; DM-significance reclassified.**

D-070 recorded that Ridge beats VAR in 12 of 16 country × horizon cells
on point-estimate MASE. This finding remains factually correct and is
**not retracted**. What Phase 7 revised is the inferential extension: the
three cells where Ridge's MASE edge translated to Diebold-Mariano
significance (USA h=3, UK h=1, UK h=3) all lost significance under
COVID-origin trimming. Ridge has a measurable MASE edge that translates
to statistical significance only when COVID-era origins are retained.

---

### Finding 3 — Japan's Structural Uniqueness, Nine-Lens Triangulated
**0 of 6 Japan cells significant under standard DM; 9 independent lenses agree.**

Japan's inflation series exhibits a **uniform forecasting difficulty** —
no model layer systematically out-forecasts another — supported by nine
independent methodological lenses across Phases 3–7: ACF autocorrelation
(D-044), ARIMA order (D-049), VAR lag (D-050), Granger (D-052), IRF
(D-056), FEVD (D-058/059), Ridge α-boundary regime (D-066), coefficient
magnitude stratification with a **70× gap against USA** (D-067), and the
Phase 7 DM-null pattern (D-078, D-079). This is the project's most
comprehensively triangulated empirical finding.

---

### Finding 4 — Honest Revision as Methodology Signal
**D-079 recast D-070 in place without retraction.**

The Phase 7 evidence rebalanced the pre-Phase-7 architectural narrative
from "Ridge dominates at the forecast level" to "Ridge has a COVID-era
MASE edge; univariate ARIMA wins at h=1 under robust testing." The
revision is logged transparently in [`ProjectDriven.md`](ProjectDriven.md)
as D-079 and D-080 with rationale, alternatives considered, and
propagation — rather than retroactively editing the pre-Phase-7
narrative. An honest revision is portfolio-stronger than a coerced
narrative.

---

### Finding 5 — USA `yoy_pct × VAR` Systematic Bias (D-062)
**The only Phase-6 point-estimate finding to survive all four Phase 7 lenses unchanged.**

USA VAR systematically under-predicts the 2022 energy-shock CPI spike
when the target is expressed in year-on-year percentage form. This
finding survives standard DM (p=0.014), HAC DM, robust DM (p=0.0001),
and COVID-origin trim (p=0.044) without any verdict change — a rare
cross-lens-immune result.

---

## ⚠️ Limitations

| Limitation | Impact |
|---|---|
| Walk-forward paired samples (n = 45–58 per cell) | Moderate DM power; sub-10 cells would be underpowered (n = 30 threshold observed) |
| COVID-onset origin treatment | Two verdict regimes (full vs trimmed) reported side-by-side; the "true" regime depends on use case (structural vs crisis forecasting) |
| China series quality | CRITICAL status in Phase 1; supplementary role only — not included in the primary N1/N2/N3 analysis |
| Forecast horizon cap at h=12 | Longer horizons (h=24, h=36) would require larger walk-forward samples than available |
| No bootstrap confidence intervals | DM battery uses asymptotic p-values only; stationary-bootstrap variants are a natural extension |
| 2025 partial-year data | Most recent walk-forward origins use partial 2024 Q4 / 2025 Q1 CPI data; final release revisions may shift edge cells |
| `src/models/` Tranche 2 deferred | ProjectScope §12 blueprint entry for model-wrapper classes deferred per D-075 / D-080 (no active callers) |

> ✅ **Intended use:** Portfolio artefact demonstrating multi-model inflation
> forecasting methodology — **not a production forecasting system** for central
> bank or financial use.

---

## 🚀 Reproducibility

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/kota2003/inflation-forecasting-analysis.git
cd inflation-forecasting-analysis

# Create conda environment
conda create -n p3_inflation python=3.11
conda activate p3_inflation

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Register for a FRED API key at
   [FRED API Keys](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Copy `.env.example` to `.env` and set `FRED_API_KEY=your_key_here`
3. Download Japan CPI manual CSV (`zmi2020s.csv`) from the
   [Japan Statistics Bureau](https://www.stat.go.jp/english/)
   and place it in `data/raw/_manual/`
4. Run notebooks `01` → `09` in sequence

> **Note on data files:** Raw and processed CSVs are excluded from Git
> (see `.gitignore`). All data is fully reproducible from FRED / World Bank
> APIs plus the single manual Japan CPI CSV. Walk-forward forecast audit
> CSVs in `data/documentation/` are tracked for downstream-test auditability.

All random seeds are fixed at `random_state=42`. Every decision is
documented in [`ProjectDriven.md`](ProjectDriven.md).

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|:-------:|---------|
| Python | 3.11 | Core language |
| pandas | ≥ 2.0 | Data loading, alignment, panel construction |
| numpy | ≥ 1.26 | Numerical operations |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisation |
| statsmodels | ≥ 0.14 | ARIMA, VAR, Granger, IRF, FEVD, Diebold-Mariano |
| scikit-learn | ≥ 1.4 | Ridge, Pipeline, TimeSeriesSplit, StandardScaler |
| scipy | ≥ 1.11 | ADF, KPSS, Chow, Quandt-Andrews, bootstrap |
| fredapi | ≥ 0.5 | FRED API client |
| wbdata | ≥ 0.3 | World Bank API client |
| python-dotenv | ≥ 1.0 | Environment variable management |

---

## 📄 Docs

| Document | Description |
|----------|-------------|
| [`ProjectDriven.md`](ProjectDriven.md) | Living project log — 80 decisions (D-001 through D-081) with rationale, alternatives, propagation |
| [`docs/findings.md`](docs/findings.md) | Three cross-country findings (N1 / N2 / N3) with decision anchors — portfolio-grade narrative |
| [`docs/methodology.md`](docs/methodology.md) | Four-phase iteration pattern, `src/` promotion discipline, cross-lens match methodology |
| [`outputs/portfolio/P3_onepager.pdf`](outputs/portfolio/P3_onepager.pdf) | **Portfolio one-pager** — 30-second read, DM heatmap + 3-bullet summary |
| [`ProjectScope_v1.md`](ProjectScope_v1.md) | Original project specification (Phase 0 baseline) |

---

## 📁 Repository Structure

```
inflation-forecasting-analysis/
│
├── README.md
├── requirements.txt
├── ProjectDriven.md              # Living project log — 80 decisions
├── ProjectScope_v1.md            # Original scope document
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                      # FRED / World Bank / IMF pulls (not tracked by Git)
│   │   └── _manual/              # Japan Statistics Bureau zmi2020s.csv (not tracked)
│   ├── processed/                # Cleaned and aligned panels (not tracked by Git)
│   └── documentation/            # Audit CSVs for phases 1–7 — tracked for reproducibility
│
├── docs/
│   ├── findings.md               ✅ Complete
│   └── methodology.md            ✅ Complete
│
├── src/                          # Reusable module architecture — v0.4.3 (5 of 8 modules)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── stationarity.py
│   ├── structural_breaks.py
│   ├── feature_engineering.py
│   ├── modelling_utils.py        # D-063 (v0.4.1) + D-074 (v0.4.2)
│   └── evaluation.py             # D-076 (v0.4.3)
│
├── scripts/                      # Scratch orchestrators for each sub-step
│   ├── phase3_step*.py
│   ├── phase6_step{1,2,3}_*.py
│   └── phase7_{s1,s2,s4}_*.py
│
├── notebooks/
│   ├── 01_data_collection.ipynb                ✅
│   ├── 02_cleaning_alignment.ipynb             ✅
│   ├── 03_stationarity_structural_breaks.ipynb ✅
│   ├── 04_feature_engineering.ipynb            ✅
│   ├── 05_eda.ipynb                            ✅
│   ├── 06_arima_baseline.ipynb                 ✅
│   ├── 07_var_model.ipynb                      ✅
│   ├── 08_ridge_regression.ipynb               ✅
│   └── 09_evaluation_interpretation.ipynb      ✅
│
└── outputs/
    ├── figures/                  # Phase 3–7 figures (25 audit + 8 portfolio)
    └── portfolio/
        └── P3_onepager.pdf       # Portfolio one-pager (A4 landscape)
```

---

*Data: FRED · World Bank · Japan Statistics Bureau · IMF · 2000–2025*
*Python 3.11 · `src/` v0.4.3 · 80 decisions · 125 paired Diebold-Mariano tests*
*Figures saved to `outputs/figures/` · Full methodology in [`docs/methodology.md`](docs/methodology.md)*
