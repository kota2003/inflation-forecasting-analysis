# ProjectDriven.md
## Living Decision Log — Inflation Prediction and Economic Signal Analysis

> This document records all key design decisions made throughout the project.
> Each entry explains **what** was decided, **why**, and any **alternatives considered**.
> Updated continuously as the project progresses.

---

## How to Use This Document

- Every non-trivial decision gets an entry here
- Be specific: bad → "chose ARIMA"; good → "chose ARIMA over ETS because..."
- Decisions are never rewritten; they may be amended (see D-006) to preserve audit integrity
- This document is part of the portfolio — it shows analytical thinking

---

## Phase 0 — Design Decisions

---

### D-001 | Country Selection

**Date:** Phase 0
**Decision:** Use USA, Japan, UK, Germany as main countries. China as supplementary only.

**Rationale:**
- USA: FRED home country; richest data availability; global inflation benchmark
- Japan: Unique 30-year deflation period followed by sudden 2022 inflation reversal;
  BOJ policy as natural experiment; highly relevant for NRI and Japan-focused analysis
- UK: Post-Brexit structural change provides an interesting case of domestically driven
  inflation diverging from EU patterns
- Germany: Largest EU economy; constrained by ECB monetary policy;
  highly exposed to 2022 energy shock

**China (Supplementary):**
Chinese macroeconomic data carries documented reliability concerns:
- GDP figures consistently align with government targets
- CPI basket composition and weighting methodology are not fully transparent
- Official unemployment statistics cover only urban registered workers,
  excluding rural and migrant labour (estimated 300M+ workers)
China is therefore excluded from the main statistical models to preserve
analytical integrity. A descriptive comparison using available NBS/OECD data
is included in the interpretation section with explicit caveats.
This decision reflects data literacy and responsible analysis.

**Alternatives Considered:**
- Australia: Relevant given Melbourne base, but adds limited analytical contrast
- France: Possible EU alternative to Germany, but ECB constraint is the same
- India / Brazil: Interesting emerging market contrast, but data gaps too significant

---

### D-002 | Analysis Period

**Date:** Phase 0
**Decision:** 2000-01 to present (monthly frequency)

**Rationale:**
- Captures three structurally distinct inflation regimes:
  1. 2000–2007: Pre-GFC moderate inflation
  2. 2008–2019: Post-GFC low-inflation, near-deflation era
  3. 2020–present: COVID shock and post-pandemic inflation surge
- Sufficient data length for VAR model stability (T > 200 observations)
- Includes all three named narrative periods (Phillips Curve breakdown, BOJ experiments, 2022 shock)

**Structural Break Points to Test:**
- 2008-09: Global Financial Crisis
- 2020-03: COVID-19 shock
- 2022-02: Energy/inflation shock (Russia-Ukraine, supply chain)

**Alternatives Considered:**
- 1990–present: Longer, but Japan data gaps pre-2000 are problematic
- 2010–present: Too short; misses GFC which is central to Phillips Curve narrative

---

### D-003 | Data Source Strategy

**Date:** Phase 0
**Decision:** FRED API as primary source via `fredapi` Python library

**Rationale:**
- FRED hosts USA, Japan, UK, and Germany indicators in a single API
- API access ensures reproducibility — no manual CSV downloads
- Demonstrates engineering practice valued by technical employers
- Data quality is equivalent to manual CSV download (same underlying source)
- API keys are free and easy to obtain

**Supplementary Sources:**
- OECD: For indicators not available via FRED (e.g. UK M2, some harmonised series)
- NBS (China): For supplementary China descriptive data only

**API Key Management:**
- Key stored in `.env` file (not tracked by Git)
- `.env.example` provided as template for reproducibility

**Note (Phase 1 v2 amendment):** This decision was extended in D-016 when a subset of
FRED/OECD-harmonised series were found to have stopped updating. The final architecture
adopts a three-tier source hierarchy: FRED primary → FRED alternative Series IDs →
direct retrieval from primary statistical agency (stat.go.jp for Japan CPI).

---

### D-004 | Model Architecture

**Date:** Phase 0
**Decision:** Three-layer modelling approach: ARIMA → VAR → Ridge Regression

**Layer 1 — ARIMA/SARIMA (Baseline)**
- Univariate inflation forecast per country
- Establishes time-series fundamentals
- Sets accuracy benchmark for comparison
- Demonstrates: stationarity handling, ACF/PACF, model selection

**Layer 2 — VAR (Core Model)**
- Multivariate: CPI + Interest Rate + Unemployment + GDP + M2
- Enables Granger causality testing
- Enables Impulse Response Functions (IRF) → key for Narrative 2
- Enables Forecast Error Variance Decomposition (FEVD)
- This layer is the primary analytical contribution of the project

**Layer 3 — Ridge Regression with Lag Features (ML Comparison)**
- Structured feature matrix using engineered lag variables
- Regularisation handles multicollinearity between lagged features
- Feature importance connects to economic interpretation
- Bridges to ML methodology demonstrated in P2

**Alternatives Considered:**
- LSTM / Deep Learning: Covered in P6; intentionally excluded here to avoid overlap
- Prophet: Useful but adds limited economic interpretability
- ARIMAX: Considered, but VAR is more symmetric and enables Granger testing

---

### D-005 | Train / Test Split Strategy

**Date:** Phase 0
**Decision:** Time-based split — Train: 2000–2019, Test: 2020–present

**Rationale:**
- Time-series data must never use random split (data leakage)
- 2020–present serves as a genuine out-of-sample stress test
- COVID and 2022 shock in the test set makes the evaluation more informative:
  models that fail here can be discussed critically (limitation of historical models)
- ~20 years of training data; ~4-5 years of test data

**No cross-validation:**
- Standard k-fold CV is invalid for time series
- If needed: walk-forward (expanding window) validation in Phase 6

---

## Phase 1 Decisions

---

### D-006 | Phase 1 v1 Collection — Initial Assessment (Superseded; see amendment)

**Date:** Phase 1 v1 completion
**Decision (original):** Successfully collected 24/24 target series via FRED API with nominal 100% data retrieval rate.

**Original Summary (preserved for audit):**
- **USA**: 5/5 indicators — CPI (315 obs), Policy Rate (315 obs), Unemployment (315 obs), GDP (104 obs), M2 (314 obs)
- **Japan**: 5/5 indicators — CPI (268 obs), Policy Rate (288 obs), Unemployment (314 obs), GDP (104 obs), M2 (206 obs)
- **UK**: 5/5 indicators — CPI (303 obs), Policy Rate (315 obs), Unemployment (311 obs), GDP (95 obs), M1 (287 obs)
- **Germany**: 5/5 indicators — CPI (303 obs), Policy Rate (315 obs), Unemployment (313 obs), GDP (104 obs), M2 (314 obs)
- **China**: 5/5 indicators (Supplementary) — CPI (304 obs), Policy Rate (306 obs), Unemployment (24 annual), GDP (95 obs), M2 (228 obs)

**Multi-API Architecture:** FRED API (primary, 19/24 series) + World Bank API (China unemployment).

#### AMENDMENT (Phase 1 v2, after Phase 2 diagnostic)

The "100% completion" claim above reflects successful *download* of all 24 target series but **did not assess data freshness or trailing-NaN padding**. A subsequent Phase 2 data-quality diagnostic (D-007) revealed that six series had ceased updating months-to-years before the Phase 1 v1 run date:

| Series | v1 effective end | Staleness at v1 time | Narrative impact |
|---|---|---|---|
| JAPAN_CPI (FRED `JPNCPIALLMINMEI`) | 2021-06 | 58 months + 10 trailing NaN | **N3 Japan's Uniqueness invalidated** |
| JAPAN_M2 (`MYAGM2JPM189S`) | 2017-02 | 110 months | N2 Japan channel broken |
| JAPAN_POLICY_RATE (`IRSTCB01JPM156N`) | 2023-12 | 28 months | N2 Japan response truncated |
| UK_M2 (`MANMM101GBM189S`) | 2023-11 | 29 months | UK test window truncated |
| UK_GDP (`NAEXKP01GBQ652S`) | 2023-07 | 33 months | UK test window truncated |
| CHINA_M2 (`MANMM101CNM189S`) | 2018-12 | 88 months | Supplementary — accepted |

The raw observation counts in the original summary above remain factually correct; what they mask is that the trailing observations in several series were NaN (for `JPNCPIALLMINMEI`) or that the last valid observation dated from years prior (for the others). Detecting this required analysing the *effective end* (last non-NaN date) rather than the nominal index range.

**This amendment is recorded to preserve decision-log integrity.** The superseding Phase 1 v2 rebuild is documented in D-013 through D-017. The final dataset state is summarised at the bottom of this document.

---

## Phase 2 Decisions (Data Quality Diagnostic and Phase 1 v2 Rebuild)

---

### D-007 | Data Quality Diagnostic Methodology

**Date:** Phase 2 diagnostic (post Phase 1 v1)
**Decision:** Classify every series by effective date range, gap topology, and freshness tier before any cleaning or modelling.

**Metrics computed per series:**
- **Effective start / end** — first and last non-NaN observations (not the nominal index range)
- **Trailing-NaN count** — number of NaN observations between the effective end and the nominal index end
- **Internal-gap topology** — number and length of NaN runs classified as *leading*, *trailing*, or *internal*
- **Months outdated** — months between effective end and today's first-of-month
- **Freshness tier**: FRESH (<6mo), WARNING (6–24mo), CRITICAL (>24mo)

**Rationale:**
A naive file-existence or observation-count check would have reported Phase 1 v1 as "100% successful," as indeed it initially was. The effective-end metric is the key — it distinguishes "series downloaded with 268 rows" from "series with valid data through 2021-06 followed by 10 trailing NaN." This diagnostic methodology is itself a portfolio-level contribution: it demonstrates that responsible data work begins with *auditing* the data before *using* it.

**Alternatives Considered:**
- Simple file-exists check: insufficient; would have missed all six stale series
- Manual inspection of each CSV: not scalable; error-prone

**Implementation:** `phase2_diagnostics_v2_1.py` (superseded the initial v2 version which did not distinguish trailing NaN from internal gaps). The diagnostic is re-run automatically in `01_data_collection.ipynb` after the Phase 1 v2 rebuild.

---

### D-008 | Strategy D — Targeted Multi-Source Rebuild

**Date:** Phase 1 v2
**Decision:** Respond to the Phase 2 diagnostic findings with a targeted rebuild using a three-tier multi-source architecture (FRED primary → FRED alternatives → external manual source), rather than a full re-collection or a period truncation.

**Four options were considered:**

| Option | Approach | Decision |
|---|---|---|
| A | Complete rebuild — all 25 series re-fetched from scratch via different sources | Rejected — excessive scope; 19 series were fine |
| B | Truncate all series to earliest common end date (2021-06) | Rejected — discards 4+ years of US/EU/UK test-window data |
| C | Country-specific end dates, no common analysis window | Rejected — breaks cross-country narrative alignment required by ProjectScope §4 |
| **D** | **Targeted rebuild: replace only the stale series; use multi-source strategy** | **Adopted** |

**Rationale for D:**
- Minimises waste (19 FRED series are FRESH and re-fetching them is pointless)
- Preserves common analysis window for cross-country VAR
- Forces explicit decisions about each stale series (see D-013 through D-016)
- The decision process itself — documented in this log — demonstrates analytical maturity

---

### D-009 | Structural Gap Treatment Policy

**Date:** Phase 1 v2 / Phase 2 entry
**Decision:** Apply linear interpolation to single-month NaN gaps; exclude any series with internal NaN runs exceeding three months from the main analysis.

**Finding from diagnostic:**
Across all 25 series, **zero internal NaN runs exceed three months**. All remaining NaN after rebuild are single-month missings (three total across USA CPI / USA Unemployment / CHINA Policy Rate).

**Rationale for the threshold:**
- ≤3 months: linear interpolation is standard practice for macroeconomic monthly data and has negligible impact on VAR coefficient estimates
- >3 months: risks introducing spurious structure; better to document a limitation than to fabricate data

**Impact:** The policy is effectively a non-binding safety net — no series triggers the exclusion. This is itself a positive finding from the rebuild.

---

### D-010 | China Unemployment — Annual Frequency Supplementary Only

**Date:** Phase 1 v1, reconfirmed in Phase 1 v2
**Decision:** Accept China unemployment as annual data (World Bank series SL.UEM.TOTL.ZS) and use only for supplementary descriptive comparison.

**Rationale:**
- Monthly Chinese unemployment statistics are not published in a form compatible with OECD/ILO standards
- Annual World Bank data (24+ observations, 2000–2025) is sufficient for the supplementary framing established in D-001
- No interpolation to monthly is performed; this would introduce spurious precision

**Note:** Post retry-logic upgrade in D-017, the v2 run successfully retrieved annual data through 2025-12 (FRESH).

---

### D-011 | File Naming and Data Interchange Convention

**Date:** Phase 1 v1, preserved in v2
**Decision:** Every series is stored as a single-column CSV at `data/raw/{COUNTRY}_{INDICATOR}.csv` with:
- Column 1: `date` (YYYY-MM-DD first-of-month for monthly, first-of-quarter for quarterly)
- Column 2: `{COUNTRY}_{INDICATOR}` (value column, same naming as filename)

**Rationale:**
- Enables trivial merging across series via `pd.concat([df1, df2], axis=1)`
- Stable filename contract means downstream notebooks do not need re-wiring when Series IDs change (the v2 rebuild changed FRED Series IDs but filenames were preserved)
- Matches the filename → column-name → variable-name chain expected by model training code

**Scope:** This convention is honoured by both Phase 1 v1 and v2 collection code; any future source replacements must preserve it.

---

### D-012 | Money Supply Unit Harmonisation (Phase 2 Action)

**Date:** Phase 1 v2 specification; execution deferred to Phase 2 cleaning
**Decision:** All M2 series will be harmonised to **year-over-year growth rate (%)** in Phase 2 regardless of source format.

**Current state after Phase 1 v2:**

| Country | Source | Unit |
|---|---|---|
| USA | `M2SL` | Level ($ billions) |
| Germany | `M2SL` (proxy) | Level ($ billions) |
| Japan | `MABMM301JPM657S` | YoY % growth |
| UK | `MABMM301GBM657S` | YoY % growth |

**Rationale:**
The unit heterogeneity arose because only YoY-growth variants of Japan/UK M2 had FRESH freshness in the scout. Converting USA/Germany to YoY in Phase 2 achieves three goals simultaneously:
1. **Cross-country comparability**: VAR coefficients across countries become directly interpretable
2. **Theoretical alignment**: The Quantity Theory of Money explicitly relates money *growth* to inflation, not money *levels*. The YoY harmonisation is theoretically preferable on its own merits
3. **Stationarity**: Level M2 is strongly non-stationary; YoY growth typically has a stable mean — simplifies D-018 (ADF/stationarity treatment, Phase 3)

**Method:** `(M2[t] − M2[t−12]) / M2[t−12]`, applied to USA/Germany series; Japan/UK retained as-is. Produces a common `M2_YOY_GROWTH` column across all four countries in `data/processed/`.

---

### D-013 | Phase 1 v2 Strategic Rebuild — Scope and Execution

**Date:** Phase 1 v2
**Decision:** Execute a scoped rebuild targeting exactly six stale series with a mix of FRED replacements (4), external-source retrieval (1 — Japan CPI), and documented acceptance (1 — China M2 as supplementary).

**Replacements executed:**

| Country | Indicator | Old Series ID | New Series ID | Source | Post-rebuild end | Freshness |
|---|---|---|---|---|---|---|
| JAPAN | M2 | `MYAGM2JPM189S` | `MABMM301JPM657S` | FRED | 2025-11 | FRESH |
| JAPAN | POLICY_RATE | `IRSTCB01JPM156N` | `IRSTCI01JPM156N` | FRED | 2026-03 | FRESH |
| JAPAN | CPI | `JPNCPIALLMINMEI` | `MANUAL_STATSBUREAU` | stat.go.jp | 2026-02 | FRESH |
| UK | M2 | `MANMM101GBM189S` | `MABMM301GBM657S` | FRED | 2026-02 | FRESH |
| UK | GDP | `NAEXKP01GBQ652S` | `NGDPRSAXDCGBQ` | FRED | 2025-10 | WARNING (6mo) |
| CHINA | M2 | `MANMM101CNM189S` | *(unchanged)* | — | 2018-12 | CRITICAL (supplementary — accepted) |

**Candidate scout process:**
`phase1v2_candidate_scout.py` systematically tested 31 alternative FRED Series IDs across the six stale targets and nominated a winner per target based on effective end and observation coverage. Two scout winners were subsequently overridden on economic-semantics grounds (D-014 Japan Policy Rate; D-015 UK GDP).

**Final result (post-rebuild):**
- FRESH 15, WARNING 8, CRITICAL 2, MISSING 0
- CRITICAL reduced from 7 to 2 (reduction: 71%)
- Both remaining CRITICAL series are Chinese and accepted as supplementary per D-001
- All three project narratives (N1, N2, N3) confirmed `✅ Ready`

**Audit trail:** `data/documentation/phase1v2_rebuild_log.csv` with timestamped records of every replacement, the old/new Series IDs, fetch status, observation counts, and rationale strings.

---

### D-014 | Japan Policy Rate — Semantic Override of Scout Winner

**Date:** Phase 1 v2
**Decision:** Override the scout's statistical winner (`IRLTLT01JPM156N`, 10-year JGB yield) with `IRSTCI01JPM156N` (immediate call money rate) on economic-semantics grounds. Freshness is equal between the two; the override is purely about variable meaning.

**The two candidates:**

| Series ID | Description | Economic meaning |
|---|---|---|
| `IRLTLT01JPM156N` | 10-year JGB yield | *Market* interest rate — reflects private-sector inflation expectations and risk premia |
| `IRSTCI01JPM156N` | Immediate interbank call money rate | *Policy* interest rate — BOJ's actual operational target under ZIRP, QQE, and YCC regimes |

**Rationale:**
Narrative N2 (*Monetary Policy Lag Effects*) asks how long after a central bank raises rates does inflation respond. The *independent variable* must therefore be the policy variable — the rate the central bank directly controls — not a market rate that itself responds to inflation expectations (which would cause endogeneity). The 10-year JGB yield correlates with BOJ policy but behaves differently during regime shifts (e.g. 2013 QQE launch, 2016 YCC introduction), which are precisely the episodes of greatest analytical interest for N2.

Using the 10-year yield as "policy rate" would silently corrupt all VAR coefficients, IRF analyses, and Granger tests involving Japan monetary policy.

**Illustrative cost if not overridden:** The IRF for "Japan rate shock → CPI response" (a headline deliverable for ProjectScope §8) would measure how the yield curve's reaction to inflation expectations affects CPI, not how BOJ action affects CPI. The two are confusingly similar but fundamentally different.

**Portfolio value:** This override demonstrates that technical judgement (scout picks the statistically optimal candidate) must be checked against domain judgement (economist's understanding of what the variable is *for*). The dual-check is a transferable pattern.

---

### D-015 | UK GDP — Real over Nominal Override

**Date:** Phase 1 v2
**Decision:** Override the scout's winner (`UKNGDP`, nominal GDP) with `NGDPRSAXDCGBQ` (Real GDP, seasonally adjusted, quarterly). Both have identical freshness (end 2025-10, WARNING 6mo); the override is for cross-country consistency.

**Rationale:**
- USA uses `GDP` (real), Japan uses `JPNRGDPEXP` (real), Germany uses `CPMNACSCAB1GQDE` (real, chain-linked)
- The VAR requires each country's GDP to mean the same thing. A nominal UK GDP would:
  1. **Conflate signal**: Nominal GDP = Real GDP × Price Index. Since the price index includes CPI (the dependent variable), using nominal GDP would partially regress the target on itself
  2. **Produce uninterpretable coefficients**: UK GDP coefficient would not be comparable to USA/Japan/Germany GDP coefficients

**Alternatives:**
- Deflating nominal GDP manually using UK CPI: feasible but adds a computation step with no analytical benefit; the FRED real GDP series already exists
- Accepting nominal for UK only: explicitly rejected because ProjectScope §7 requires cross-country comparability

---

### D-016 | Japan CPI — External Source Decision (Statistics Bureau Manual Retrieval)

**Date:** Phase 1 v2
**Decision:** For Japan CPI, bypass FRED entirely and retrieve the series directly from the Japan Statistics Bureau (総務省統計局) via manual CSV download (`zmi2020s.csv`, 2020-base middle-category nationwide monthly index).

**Full evidence chain that FRED and IMF were exhausted:**

**FRED/OECD-harmonised family** — six candidates tested via scout:
- `JPNCPIALLMINMEI` (original) — ends 2021-06
- `CPALTT01JPM657N` — ends 2021-06
- `CPALTT01JPM659N` — ends 2021-06
- `CPALCY01JPM661N` — ends 2022-04
- `JPNCPICORMINMEI` — ends 2021-06
- Core CPI variant — ends 2021-06

All six stopped updating at or before 2022-04. This is a structural feature of the OECD harmonisation pipeline for Japan — not a Phase 1 v1 bug. FRED mirrors OECD's harmonised series; OECD's pipeline for Japanese CPI appears to have stalled.

**IMF International Financial Statistics (SDMX)**: attempted automated retrieval failed with 3 consecutive 60-second timeouts.

**IMF DataMapper API**: returns only annual WEO data now (API structure changed since project inception); does not serve monthly IFS data.

**Resolution:** `zmi2020s.csv` downloaded from https://www.stat.go.jp/data/cpi/ (長期時系列データ → 中分類指数 2020基準 全国 月次) and integrated via a dedicated robust CSV parser (D-016a below).

---

### D-016a | Robust CSV Parser for External Japanese Government Data

**Date:** Phase 1 v2
**Decision:** Implement an encoding-detecting and header-detecting CSV parser capable of handling the three characteristics that naive `pd.read_csv` fails on:

1. **Encoding**: `zmi2020s.csv` is cp932 (Shift-JIS), not UTF-8
2. **Metadata preamble**: Real header row is preceded by metadata lines describing the dataset
3. **Japanese date formats**: Dates appear as `2000年1月`, not ISO format

**Parser capabilities:**
- Tries encoding candidates in order: `utf-8-sig`, `cp932`, `utf-8`, `shift_jis`, `cp1252`
- Auto-detects header row by scanning for lines containing both a comma AND a known hint word (`時点`, `年月`, `total`, `total inflation`, etc.)
- Parses three date formats via dedicated regex: `YYYY年M月`, `YYYY-M` / `YYYY/M`, `YYYYMM`
- Column-name matching uses Japanese hints (`総合` for overall CPI) and English hints (`all items`, `cpi`, `overall`) with case-insensitive fallback
- Graceful error reporting if parsing fails — identifies which step failed

**Rationale:**
A brittle parser would require manual intervention any time stat.go.jp adjusts its CSV format. The robust parser absorbs typical variations (column renames, encoding changes, metadata reorderings) and fails loudly only on genuine structural changes. This matters because Phase 1 v2's reliance on manual retrieval makes the parser the single point of failure for Japan CPI — it must be defensible.

---

### D-017 | Retry Logic with Exponential Backoff for Transient API Failures

**Date:** Phase 1 v2 (post first successful run, after observing one WB timeout)
**Decision:** Wrap all FRED and World Bank API calls in a retry loop with exponential backoff (1s → 2s → 4s, max 3 attempts) and differentiated error handling.

**Behaviour:**

| Failure type | Action |
|---|---|
| Timeout, connection reset, 502/503/504, 5xx | Retry up to 3 times with exponential backoff |
| 404, invalid Series ID, 4xx | Abort immediately (no point retrying a permanent failure) |
| JSON parse error, unexpected payload shape | Abort immediately (structural incompatibility) |

**Timeout parameters increased:**
- FRED: 30s → 45s
- World Bank: 30s → 60s (WB is known to be intermittently slow)

**Rationale:**
- Transient failures are common with public APIs — a single run failure is not a defect but an expected occurrence
- The first production run failed CHINA_UNEMPLOYMENT due to a 30s WB timeout; the retry logic resolved this on the second production run
- Permanent failures (e.g. deprecated Series IDs) must fail loudly and quickly, not burn through retry budget silently
- The progress indicator (`[retry in 1s]`) surfaces the retry to the user so they understand what is happening

**Portfolio value:** Demonstrates engineering maturity — the difference between "code that works when everything is fine" and "code that degrades gracefully under realistic conditions." Both are acceptable; only the second scales to production.

---

## Phase 1 Final State — Summary

**After Phase 1 v2 rebuild (25/25 series collected):**

| Metric | v1 state | v2 state |
|---|---|---|
| FRESH | 11 | **15** |
| WARNING | 7 | **8** |
| CRITICAL | 7 | **2** (both CHINA, supplementary per D-001) |
| MISSING | 0 | **0** |
| Narrative N1 | ⚠️ partial | ✅ Ready |
| Narrative N2 | ❌ blocked (Japan) | ✅ Ready |
| Narrative N3 | ❌ blocked (Japan CPI 2021-06) | ✅ Ready |

**Structural integrity:** zero internal NaN runs >3 months across all 25 series. Phase 2 cleaning reduces to linear interpolation of three single-month missings.

**Artifacts produced:**
- `data/raw/*.csv` — 25 series, final v2 state
- `data/raw/_archive_v1/{timestamp}/` — archived v1 versions for traceability
- `data/documentation/phase1v2_rebuild_log.csv` — complete audit trail
- `outputs/figures/` — staleness bar, CPI comparison, Japan v1-vs-v2 plots
- `notebooks/01_data_collection.ipynb` — self-contained reproducible pipeline

---

## Phase 2 Cleaning Decisions

*These decisions concern the transformation of the 25 Phase 1 v2 raw series into analysis-ready processed datasets. They are implemented as pure functions in `src/preprocessing.py` and narrated in `notebooks/02_cleaning_alignment.ipynb`.*

---

### D-018 | GDP Quarterly → Monthly Linear Interpolation

**Date:** Phase 2
**Decision:** Convert quarterly GDP levels to monthly via linear interpolation on the level, then compute YoY % growth on the interpolated monthly level:

```
monthly_level[t] = level[q_prev] + (t - t_{q_prev})/(t_q - t_{q_prev}) * (level[q] - level[q_prev])
YoY[t] = (monthly_level[t] / monthly_level[t-12] - 1) * 100
```

**Method alternatives considered:**

| Option | Summary | Verdict |
|---|---|---|
| Forward fill | Step-function (quarterly value held constant for 3 months) | Rejected — violates VAR innovation-i.i.d. assumptions |
| **Linear interpolation** | Smooth within-quarter variation | **Adopted** — matches ProjectScope §9 specification |
| Cubic spline | Third-order continuity | Rejected — introduces spurious oscillation |
| Chow-Lin / Fernandez temporal disaggregation | Regression-based using a high-frequency auxiliary | Considered and rejected after due diligence (see below) |

**Chow-Lin due diligence (rejection rationale):**
A systematic scout was conducted for a monthly auxiliary (Industrial Production) usable in Chow-Lin disaggregation. OECD MEI industrial-production series for Japan, UK, and Germany all terminate at 2024-03 due to a systemic publication lag. An expanded scout across 45 alternative FRED candidates (non-OECD-MEI) found only OECD Business Tendency confidence indices as FRESH — these are theoretically weaker Chow-Lin auxiliaries (lead rather than contemporaneous with GDP). The UK ONS K222 Index of Production was successfully fetched via native API; Japan (METI) and Germany (Destatis) would require manual native download. At this juncture cost–value analysis concluded that the marginal VAR-accuracy gain from Chow-Lin relative to linear interpolation is disproportionate to the integration complexity: GDP enters the VAR as one of five regressors under lag depths t-1 to t-3, for which linear-interpolation error is absorbed by the regression.

**Retained artefacts (audit trail):**
- `data/documentation/phase2_ip_scout.csv` — initial OECD IP scout (16 candidates)
- `data/documentation/phase2_ip_scout_tier1_expanded.csv` — expanded non-OECD scout (45 candidates)
- `data/raw/UK_IP.csv` — UK ONS Index of Production (retained for potential Phase 5 EDA overlays)

**Implementation:** `gdp_quarterly_to_monthly_yoy()` in `src/preprocessing.py`.

**Portfolio value:** The Chow-Lin option is documented, not silently discarded. A reviewer can verify that an advanced alternative was genuinely evaluated and the simpler method chosen on proportionality grounds. This is the difference between "we used linear interpolation" and "we knowingly used linear interpolation because it was proportionate to the downstream use".

---

### D-019 | Country-Wise Effective Window (Option b)

**Date:** Phase 2
**Decision:** Each country's processed dataset is trimmed to [2001-01, t_max(c)] where t_max(c) is the last month at which all five indicators for country c are non-NaN.

**Option compared:**

| Option | Strategy | Verdict |
|---|---|---|
| (a) Intersection | Trim all countries to max(start)–min(end); one shared window | Rejected — discards ~7 months of USA/JAPAN data to accommodate UK/Germany CPI's earlier end |
| **(b) Country-wise** | Each country keeps its own full effective window | **Adopted** — preserves information; per-country VAR design does not require shared sample |

**Rationale:**
`ProjectScope_v1.md` §9 Phase 6 specifies per-country VAR estimation. There is no structural requirement for a shared sample across countries. Option (b) preserves information; the Phase 5 EDA can concat on a common index when cross-country overlay is required.

**Resulting effective windows:**

| Country | Start | End | Binding variable |
|---|---|---|---|
| USA | 2001-01 | 2025-10 | GDP (quarterly, last observation Q3 2025) |
| JAPAN | 2001-01 | 2025-10 | GDP (quarterly) |
| UK | 2001-01 | 2025-03 | CPI (OECD publication WARNING ~13mo) |
| GERMANY | 2001-01 | 2025-03 | CPI (OECD publication WARNING ~13mo) |

The 2001-01 start reflects the 12-month YoY lookback window loss inherent to D-012-amended M2 and D-018 GDP transformations.

**Implementation:** `trim_effective_window()` in `src/preprocessing.py`.

---

### D-021 | Germany M2 — Euro-Area Broad Money as Proxy

**Date:** Phase 2
**Decision:** Replace the Phase 1 v2 placeholder `M2SL` (inherited from USA) with `MABMM301EZM657S` (OECD harmonised Euro-area broad money, YoY %).

**Empirical evidence from Phase 2 scout:**

| Candidate | Scope | Status |
|---|---|---|
| `MYAGM2DEM189S` | Germany-specific legacy | Terminated **1998-12** (euro adoption) |
| `MABMM301DEM189S`, `MABMM301DEM657S` | Germany-specific OECD | Do not exist on FRED |
| `MANMM101DEM189S`, `MANMM101DEM657S` | Germany-specific (narrow money) | Do not exist on FRED |
| `MABMM301EZM189S` | Euro-area OECD (level) | CRITICAL (ends 2023-11) |
| **`MABMM301EZM657S`** | **Euro-area OECD (YoY growth)** | **FRESH, ends 2025-12** — adopted |
| `MANMM101EZM189S` | Euro-area narrow money | CRITICAL |

**Rationale:**
The empirical termination of every Germany-specific M2 series at precisely 1998-12 is the signature of a deep institutional fact: following 1999 euro adoption, Germany no longer maintains a national monetary aggregate. The European Central Bank manages broad money at the currency-union level. The theoretically coherent substitute for a (non-existent) national German M2 is the Euro-area aggregate — the monetary quantity to which Germany is actually exposed.

**Limitation (documented):** The monetary variable's level of aggregation differs from the national scope of Germany's other four indicators. VAR coefficients on Germany's M2 therefore reflect Euro-area monetary transmission, attenuated by Germany's share of the currency union. This is institutionally correct but must be remembered when interpreting cross-country M2 comparisons.

**Implementation:**
- Fetched directly via `notebooks/01_data_collection.ipynb` §8.5
- Saved to `data/raw/GERMANY_M2.csv` (overwriting v1 placeholder, which is archived to `data/raw/_archive_d021/{timestamp}/`)

**Audit trail:** `data/documentation/phase2_germany_m2_scout.csv` records all 10 candidate outcomes.

---

### D-012 (Amendment) | M2 YoY Conversion — MoM Source Unit Corrected

**Date:** Phase 2 (amends the Phase 1 v2 D-012)

**Original D-012 assumption:** `MABMM301...657S` series from FRED represent year-over-year % growth. Under this assumption, Japan/UK/Germany M2 required no transformation in Phase 2 and only USA (level) needed conversion.

**Empirical audit (Phase 2):** The assumption was falsified. Descriptive statistics of the raw `MABMM301...657S` series revealed a distinctive signature incompatible with YoY growth:

| Country | max | std | Signature |
|---|---|---|---|
| JAPAN `MABMM301JPM657S` | 1.76 % | 0.18 % | MoM (YoY would peak ~10 % during COVID) |
| UK `MABMM301GBM657S` | 4.58 % | 0.74 % | MoM |
| GERMANY `MABMM301EZM657S` | 2.00 % | 0.42 % | MoM |

**Cross-validation:** A 12-month rolling sum of each series produced peak dates that aligned with known monetary-policy episodes (UK 2008-12 BoE post-Lehman liquidity; Japan 2021-02 BoJ COVID expansion; Euro-area 2007-11 pre-GFC credit boom). This convergence with economic history confirmed MoM identification.

**Amended conversion method:**

```
For level source (USA, CHINA):
    YoY[t] = (level[t] / level[t-12] - 1) * 100

For MoM % source (JAPAN, UK, GERMANY):
    YoY[t] = ( prod_{i=0..11}(1 + MoM[t-i]/100) - 1 ) * 100

The MoM-to-YoY conversion is implemented via log-sum for numerical stability:
    YoY[t] = ( exp( sum_{i=0..11} ln(1 + MoM[t-i]/100) ) - 1 ) * 100
```

**Economic-history validation (post-conversion):** Computed peak dates align with the informally known monetary-policy history of each jurisdiction — USA 2021-02 (+26.78 %), Japan 2021-02 (+8.08 %), UK 2008-12 (+17.66 %), Germany 2007-11 (+12.48 %). This validation was the decisive empirical confirmation of the amendment.

**Implementation:** `m2_to_yoy()` in `src/preprocessing.py`, with the `M2_UNITS` dict encoding each country's source-unit classification.

**Audit trail:** `data/documentation/phase2_m2_yoy_validation.csv` records the pre-/post-conversion distributions and peak dates.

**Portfolio value:** A silent assumption (that `...657S` = YoY) would have produced a miscalibrated VAR: the Japan/UK/Germany M2 coefficients would be 12x smaller in magnitude than USA's, creating a spurious "Japan monetary transmission is weak" conclusion. The empirical audit caught this before any VAR estimation. This is a representative example of why data-quality auditing must precede modelling.

---

### D-022 | Residual NaN — Single-Month Linear Interpolation

**Date:** Phase 2
**Decision:** Internal NaN runs of length exactly 1 month are linearly interpolated. Trailing NaN (the most recent month yet unpublished) is handled downstream by the D-019 effective-window trim.

**Finding:** Across all 25 Phase 1 v2 series, internal NaN gaps exist only in three locations, all of length 1 month:

| Country | Indicator | Gap location |
|---|---|---|
| USA | CPI | 2026-03 (may resolve after FRED refresh) |
| USA | UNEMPLOYMENT | 2026-03 (same) |
| CHINA | POLICY_RATE | Mid-period (supplementary) |

China unemployment, being annual, has 11-month gaps between observations; these are handled separately via forward-fill within the supplementary pipeline (per D-010).

**Rationale:**
The Phase 1 v2 diagnostic (D-009) established that no internal NaN run exceeds three months. Consistent with that finding, D-022 applies the least invasive treatment: linear interpolation for single-month gaps only. This is standard practice for macroeconomic monthly data.

**Implementation:** `interpolate_single_gaps(max_gap=1)` in `src/preprocessing.py`.

---

### D-023 | Processed Output Format — Wide CSV per Country

**Date:** Phase 2
**Decision:** One wide-format CSV per country, with the main / supplementary split reflecting D-001.

**File layout:**

```
data/processed/
├── main_usa.csv                  # 298 rows × 5 cols, 2001-01 to 2025-10
├── main_japan.csv                # 298 rows × 5 cols, 2001-01 to 2025-10
├── main_uk.csv                   # 291 rows × 5 cols, 2001-01 to 2025-03
├── main_germany.csv              # 291 rows × 5 cols, 2001-01 to 2025-03
├── supplementary_china.csv       # 300 rows × 5 cols (sparse, VAR-excluded)
└── schema.md                     # auto-generated schema specification
```

**Per-file schema:**

| Column | Type | Description |
|---|---|---|
| `date` | DatetimeIndex (MS freq) | First-of-month monthly index |
| `{COUNTRY}_CPI` | float | Consumer Price Index (base year varies; normalised in EDA) |
| `{COUNTRY}_POLICY_RATE` | float | Central-bank policy rate, % |
| `{COUNTRY}_UNEMPLOYMENT` | float | Harmonised unemployment rate, % |
| `{COUNTRY}_GDP` | float | GDP YoY % growth (from linearly-interpolated monthly level, D-018) |
| `{COUNTRY}_M2` | float | M2 YoY % growth (D-012 amended) |

**Alternatives considered:**

| Option | Verdict |
|---|---|
| Single long-format CSV (all countries) | Rejected — requires pivot before every VAR fit; less Git-friendly |
| Single wide-format CSV (all 25 cols) | Rejected — mixes main and supplementary, violates D-001 structural separation |
| Parquet / Pickle | Rejected — Portfolio-unfriendly (binary, not Git-diffable, not reviewable) |

**Rationale for adopted format:**
- VAR ingestion is `pd.read_csv('data/processed/main_usa.csv').set_index('date')` — one line
- `load_processed_main(country)` and `load_processed_all_main()` helpers in `src/data_loader.py`
- Main/supplementary split is structural (not a column flag) — cannot be accidentally violated
- Column names follow D-011 filename-to-column convention

**Implementation:** Orchestration via `scripts/rebuild_processed.py` (canonical entry point) and `notebooks/02_cleaning_alignment.ipynb` (narrated version).

---

## Phase 2 Final State — Summary

**After Phase 2 cleaning & alignment (4 main + 1 supplementary datasets):**

| Metric | Phase 1 v2 | Phase 2 |
|---|---|---|
| Germany M2 placeholder | `M2SL` (USA) | **`MABMM301EZM657S` (Euro area)** — D-021 resolved |
| M2 unit heterogeneity | Mixed level/MoM | **All YoY % growth** — D-012 amended |
| GDP frequency | Quarterly | **Monthly (linear interp)** — D-018 |
| Effective windows | Per-indicator | **Per-country (D-019 option b)** |
| NaN in main 4 | ≤ 3 singletons | **0** — D-022 applied |
| Output format | — | **`main_{country}.csv` × 4 + `supplementary_china.csv`** — D-023 |
| VAR readiness | ❌ (heterogeneous) | ✅ **Ready (NaN-free, harmonised, common schema)** |

**Reusable module architecture introduced:**
- `src/__init__.py` — package v0.2.0
- `src/data_loader.py` — I/O helpers (`load_raw_series`, `load_all_raw`, `load_processed_main`, `load_processed_all_main`, `load_processed_china`)
- `src/preprocessing.py` — Phase 2 transformation functions (`m2_to_yoy`, `gdp_quarterly_to_monthly_yoy`, `interpolate_single_gaps`, `trim_effective_window`, `build_processed`, `build_all_processed`)
- `scripts/rebuild_processed.py` — canonical CLI orchestrator

All Phase 3 through Phase 7 notebooks will import from `src/` rather than duplicating logic, per `ProjectScope_v1.md` §12.

**Artifacts produced:**
- `data/processed/main_usa.csv`, `main_japan.csv`, `main_uk.csv`, `main_germany.csv` (4 main-country datasets, NaN-free, 2001-01 onwards)
- `data/processed/supplementary_china.csv`
- `data/processed/schema.md` (auto-generated schema specification)
- `data/documentation/phase2_cleaning_log.csv` (run-by-run audit)
- `data/documentation/phase2_germany_m2_scout.csv`, `phase2_m2_yoy_validation.csv`, `phase2_ip_scout*.csv`, `phase2_ip_native_fetch_log.csv` (Phase 2 diagnostic / audit artefacts)
- `notebooks/02_cleaning_alignment.ipynb` — Portfolio-grade narrative of the six Phase 2 decisions
- `outputs/figures/phase2_m2_yoy_4countries.png`, `phase2_gdp_interpolation_usa.png`, `phase2_processed_4countries_panel.png`

---

## Phase 3 Decisions

*(To be added during Phase 3 — anticipated topics: per-variable ADF test results, first-differencing decisions, Chow test specifications at 2008-09 / 2020-03 / 2022-02 break points, structural-break treatment via regime dummies vs split samples.)*

---

*Last updated: Phase 2 complete — 4 main + 1 supplementary datasets in `data/processed/`, VAR-ready, reusable `src/` module architecture established.*
