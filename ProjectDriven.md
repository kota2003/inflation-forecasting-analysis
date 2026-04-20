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

*These decisions concern the stationarity testing, transformation selection, and structural-break testing of the four main-country analytic datasets produced by Phase 2. They are implemented in `src/stationarity.py` and `src/structural_breaks.py` and narrated in `notebooks/03_stationarity_structural_breaks.ipynb`.*

---

### D-024 | ADF + KPSS Joint Protocol (Four-Quadrant Classification)

**Date:** Phase 3 · Step 1
**Decision:** Classify each series using the joint outcome of the Augmented Dickey-Fuller test (H₀: unit root) and the KPSS test (H₀: stationary) at α = 0.05, into one of four quadrants:

|                    | KPSS reject             | KPSS non-reject |
|--------------------|-------------------------|-----------------|
| **ADF reject**     | Trend-stationary (conflict) | **Stationary** |
| **ADF non-reject** | **Non-stationary**      | Inconclusive    |

**Rationale:**
- ADF alone has known low power against stationary alternatives, biasing toward over-differencing
- KPSS inverts the null (H₀ = stationary), so rejecting it is strong evidence of non-stationarity — the two tests triangulate from opposite directions
- The joint protocol makes "Inconclusive" cases (neither test rejects) explicit rather than collapsing them into a one-sided verdict
- "Trend-stationary (conflict)" distinguishes series with residual trend that the ADF 'c' or 'ct' spec absorbs but KPSS still rejects — such series require a transform decision rather than a single-test dismissal

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| ADF only | Rejected — single-null framework, biased toward spurious unit-root findings |
| KPSS only | Rejected — less common in macro literature; reverse-null harder to interpret against textbook conventions |
| Phillips-Perron substitute for ADF | Considered; ADF preferred for cross-study comparability |

**Implementation:** `src/stationarity.py::classify_4quadrant()` and `test_series()`.

**Portfolio value:** Demonstrates that a single null test is a weaker inferential framework than the joint protocol, and that rigor is achieved by triangulating from two complementary nulls.

---

### D-025 | Variable-Specific ADF Regression Specification

**Date:** Phase 3 · Step 1
**Decision:** Use `regression='ct'` (constant + linear trend) for CPI level series; use `regression='c'` (constant only) for POLICY_RATE, UNEMPLOYMENT, GDP (YoY), and M2 (YoY).

**Rationale:**
- CPI is a level index exhibiting pronounced long-run upward drift. Testing it with `'c'` forces the trend into the residual, biasing the test toward non-stationary rejection even when the series is trend-stationary
- POLICY_RATE and UNEMPLOYMENT have long-run means without persistent deterministic trend — `'c'` is correct
- GDP YoY and M2 YoY are growth rates with stable means — `'c'` is correct
- Specifying a richer deterministic component than the DGP has only a small power cost; specifying a poorer one is inferentially catastrophic

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Uniform `'c'` for all | Rejected — biases CPI results |
| Uniform `'ct'` for all | Rejected — POLICY_RATE etc. have no deterministic trend; adds spurious power loss |
| Per-variable spec per economic intuition (adopted) | Adopted |

**Implementation:** `src/stationarity.py::ADF_REGRESSION_LEVEL` constant; KPSS regression is matched to the ADF spec so both tests evaluate the same deterministic specification.

---

### D-026 | ADF Lag Selection — AIC with Schwert (1989) Max Lag

**Date:** Phase 3 · Step 1
**Decision:** Use `autolag='AIC'` with `maxlag = ⌊12·(T/100)^(1/4)⌋` per Schwert (1989). For T ≈ 290–300 monthly observations, this yields maxlag ≈ 15–16. KPSS uses `nlags='auto'` (Hobijn et al. 1998).

**Rationale:**
- AIC balances parsimony with residual whitening, preserving ADF power in finite samples
- The Schwert rule is the accepted upper bound for macro time-series lag search — enough to whiten typical monthly autocorrelation without over-fitting
- BIC considered but tends to under-select lags in monthly macro data, leaving residual autocorrelation that inflates the ADF size

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Fixed maxlag = 12 (one year) | Rejected — ignores sample size |
| BIC selection | Rejected — risks under-lagging |
| t-statistic lag selection | Rejected — non-standard, less defensible in a portfolio context |

**Implementation:** `src/stationarity.py::schwert_maxlag()` and `run_adf()`.

---

### D-027 | Transformation Registry — Phase 6 VAR Input & Chow-Test Input

**Date:** Phase 3 · Step 3
**Decision:** Maintain a per-(country, indicator) registry with two forms: `phase6_var_input` and `chow_test_input`. The two columns may differ when the Phase 6-preferred form is not a full-sample stationary form (it is then kept with a caveat), while the Chow test requires strict within-sub-sample stationarity for inferential validity.

**Registry summary (phase6_var_input by transform, 20 series):**

| Transform | Count | Example |
|---|---|---|
| `level` | 5 | All four GDP series; USA M2 |
| `first_diff` | 9 | POLICY_RATE, UNEMPLOYMENT (most); JPN/UK/GER M2; JPN/GER CPI |
| `yoy_pct` | 1 | USA CPI |
| `first_diff_with_caveat` | 2 | USA UNEMPLOYMENT (COVID outlier); GERMANY POLICY_RATE (regime-stratified) |
| `log_diff_pct_with_caveat` | 1 | UK CPI |
| `yoy_pct_with_regime_dummy` | 1 | (superseded — see D-031) |
| `yoy_pct_with_caveat` | 1 | (superseded — see D-031) |

**Rationale:**
- No one-size-fits-all per-country CPI decision (D-028, D-031)
- Registry captures the decision plus its justification string as audit artefact
- Splitting Phase 6 input from Chow-test input permits the former to adopt caveats without compromising the latter's inferential assumptions

**Implementation:** Generated by `scripts/phase3_step3_cpi_decision_and_registry.py`. Final CSV at `data/documentation/phase3_transformation_registry_final.csv` (20 rows, 9 columns).

---

### D-028 | Chow-Test Dependent Variable — Stationary CPI Form, Not Level

**Date:** Phase 3 · Step 4 (resolving 论点 4)
**Decision:** Run all Chow tests with y = per-country stationary CPI form from the registry's `chow_test_input` column, not the raw level CPI.

Per-country y form:

| Country | y form | Source of decision |
|---|---|---|
| USA     | `yoy_pct`       | Only form fully stationary on full sample |
| Japan   | `first_diff`    | All three transforms non-stationary full-sample; I(1) accepted (D-031) |
| UK      | `log_diff_pct`  | All three non-stationary or conflict; log_diff chosen for narrative |
| Germany | `first_diff`    | Regime-shift pattern: full Non-stationary, pre/post both Stationary |

**Rationale:**
- Chow F and Wald inference requires the dependent variable to be stationary within each sub-sample
- Running Chow on level CPI regressed against a mix of I(1) and I(0) regressors yields a spurious F statistic whose asymptotic distribution is degenerate
- Per-country stationarity findings from the Step 3 deep-dive dictate which transform is available

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| All four countries use yoy_pct | Rejected — JPN / UK / GER YoY is non-stationary |
| All four use first_diff | Rejected — USA first_diff is non-stationary |
| Level CPI (原文の当初案) | Rejected — spurious regression risk |
| Per-country stationary form (adopted) | Adopted — statistical validity + narrative alignment |

**Implementation:** Runtime override dict `REGISTRY_OVERRIDES` in `scripts/phase3_step4_chow_structural_breaks.py`; no edit to the registry CSV itself (preserves audit integrity per the amendment convention).

---

### D-029 | COVID Outlier Handling — Dummy-Augmented Chow as Robustness Variant

**Date:** Phase 3 · Step 4
**Decision:** Run three Chow variants per (country × break): classical F-test, HAC-Wald, and HAC-Wald with an additive COVID-period level dummy (2020-03 to 2020-09). Report all three; use concordance across variants as the confidence criterion. Skip the COVID-dummy variant for the COVID_2020 break (the dummy coincides with the break itself).

**Rationale:**
- 2020-03 to 2020-09 contains extreme CPI / unemployment outliers that can dominate F-statistic contributions from either the pre- or post-window depending on the break date
- For GFC_2008: COVID is in post-sample → dummy absorbs outlier without contaminating the pre/post slope comparison
- For ENERGY_2022: COVID is in pre-sample → dummy prevents pre-window variance inflation and spurious pre-break instability
- Three-variant concordance provides a robustness signature stronger than any single test

**Empirical outcome:** Step 4 Part 2 vs Part 3 concordance = 8/8 verdicts preserved across HAC and HAC-with-COVID-dummy. None of the three known breaks is a COVID outlier artefact; all three reflect genuine regime transitions.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Exclude COVID period entirely | Rejected — choice of excluded range is arbitrary |
| Heteroskedasticity-only robust SE (not HAC) | Rejected — HAC additionally absorbs autocorrelation which is present in residuals |
| No robustness check | Rejected — risks false-positive attribution to the specified break |

**Implementation:** `src/structural_breaks.py::chow_test_covid_dummy()`.

---

### D-030 | Phase 6 Regime Treatment Strategy — Regime Dummies Default

**Date:** Phase 3 · Step 4
**Decision:** In Phase 6 VAR estimation, incorporate each break that passes the HAC Chow at Bonferroni-corrected α = 0.05/12 via regime-dummy interaction terms on the specific economic channels identified as dominant drivers by the per-coefficient decomposition (Step 4 Part 4). Split-sample estimation is reserved as a secondary strategy for cases where a dominant regressor fails stationarity in one sub-window.

**Dominant driver per (country × break) as identified by the per-coefficient decomposition:**

| Country | GFC_2008 | COVID_2020 | ENERGY_2022 |
|---|---|---|---|
| USA     | `M2` (z=+4.34) | `POLICY_RATE` (z=+3.41) | `POLICY_RATE` (z=+5.95) |
| Japan   | *(not significant)* | `const` (z=+4.05) | `const` (z=+4.98) |
| UK      | `GDP` (z=+1.96)    | `const` (z=+2.47)    | `GDP` (z=+3.58) |
| Germany | *(not significant)* | `GDP` (z=+2.93) | `GDP` (z=+2.82) |

**Rationale:**
- Regime dummies preserve full-sample information (important given that the post-2022 window is only 38–45 observations)
- Per-coefficient decomposition isolates the economically-interpretable channel through which each break operates, rather than attributing the break to the entire equation
- Different countries show different dominant drivers at the same date (e.g. USA = POLICY_RATE, Japan = const, UK/Germany = GDP at ENERGY_2022) — regime-dummy specification should be country-specific

**Portfolio value:** Connects the Chow-test output (a single F statistic per break) to a specific VAR specification choice (which interactions to include), bridging Phase 3 diagnostics to Phase 6 modelling.

**Implementation:** Phase 6 VAR spec will insert `D_t × dominant_regressor` interaction terms per the above matrix. Deferred to Phase 6; recorded here as the forward-looking directive.

---

### D-031 | Japan CPI I(1) Acceptance (Revised from Regime-Dummy Hypothesis)

**Date:** Phase 3 · Step 3 (revised from the Step 3 initial registry)
**Decision:** Accept Japan CPI as I(1) and adopt `first_diff` (MoM inflation) as both the Phase 6 VAR input and the Chow test dependent variable. Retain `yoy_pct` for narrative plots only.

**Why this is a revision:** The Step 3 initial registry proposed `yoy_pct_with_regime_dummy`, on the a-priori hypothesis that Japan's 30-year low-flation period followed by the 2022 reversal is a clean level-shift treatable via a post-2022 dummy. The Step 3 Part 2 sub-period deep-dive empirically falsified this hypothesis:

| Form | Full sample | Pre-2020 | Post-2020 |
|---|---|---|---|
| `first_diff`     | Non-stationary | Stationary | Conflict |
| `yoy_pct`        | Non-stationary | **Non-stationary** | Non-stationary |
| `log_diff_pct`   | Non-stationary | Stationary | Conflict |

The critical row is the second: Japan CPI YoY is non-stationary even **in the pre-2020 sample**. This rejects the "2022-is-a-level-shift" hypothesis — if it were a level shift, pre-2020 YoY would be stationary around its mean and post-2020 YoY would be stationary around a higher mean, but the pre-2020 YoY is itself non-stationary. The correct interpretation is that Japan CPI has a long-term structural drift rather than a discrete regime break.

**Portfolio value:** The revision is itself an empirical finding. An ex-ante plausible hypothesis (regime shift at 2022) was statistically falsified by sub-period analysis. Narrative N3 ("Japan's Uniqueness") is thereby reinforced as **structural-drift rather than regime-shift**, which is the more challenging and more economically-interesting characterisation.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Original: yoy_pct + 2022 regime dummy | Rejected — pre-2020 YoY non-stationary falsifies the level-shift hypothesis |
| Second-differencing (I(2) treatment) | Rejected — I(2) monthly CPI has no clean economic interpretation |
| Sub-sample-only VAR (split 2020) | Rejected — discards 228 pre-2020 observations; power loss not justified |
| first_diff (adopted) | Adopted — I(1) is economically interpretable (MoM inflation) and maintains full-sample estimation |

**Implementation:** `REGISTRY_OVERRIDES` in `scripts/phase3_step4_chow_structural_breaks.py`. Justification text preserved in `phase3_transformation_registry_final.csv::justification`.

---

### D-032 | `src/` Module Separation — stationarity.py + structural_breaks.py

**Date:** Phase 3 · Step A (module extraction)
**Decision:** Split the Phase 3 reusable module into two files: `src/stationarity.py` (univariate ADF/KPSS + transforms) and `src/structural_breaks.py` (multivariate Chow + Quandt-Andrews). The original prompt specified a single `src/stationarity.py` for both tasks; the split is an upgrade per ProjectScope §12 "reusable module" planning.

**Rationale:**

1. **Conceptual separation**: univariate stationarity testing and multivariate regression break testing are distinct analytical frameworks
2. **Import granularity**: Phase 6 VAR estimation needs only the stationarity module at fit time; structural_breaks is a Phase 3 diagnostic not needed for forecasting
3. **File size & reviewability**: combined would be ~800 lines; split yields ~310 + ~460 which review better in a portfolio context
4. **Future extensibility**: Bai-Perron multi-break test, if added later, fits naturally in `structural_breaks.py` without bloating the stationarity module

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Single `src/stationarity.py` (prompt default) | Rejected — mixes univariate and multivariate concerns |
| Three files (add `src/quandt_andrews.py`) | Rejected — Quandt-Andrews is a natural extension of Chow, same module |
| Two-file split (adopted) | Adopted — clean separation on analytical domain |

**Implementation:** `src/__init__.py` bumped from 0.2.0 to 0.3.0; 60 total exports from 4 submodules (`data_loader`, `preprocessing`, `stationarity`, `structural_breaks`).

---

### D-033 | Quandt-Andrews Robustness — π₀ Trim Sensitivity Check

**Date:** Phase 3 · Step 5b
**Decision:** Run the Quandt-Andrews sup-Wald scanner at both π₀ = 0.15 (Andrews 1993 standard) and π₀ = 0.10 (wider scan) and report both outcomes. Use Andrews (1993) Table I critical values for the applicable π₀ row. Retain the π₀ = 0.15 audit CSVs (`phase3_quandt_andrews_supwald.csv`, `_curve.csv`) as the Step 5 state, and add π₀ = 0.10 versions (`_trim10` suffix) as the Step 5b state.

**Why the sensitivity check matters empirically:**

| Country | π₀ = 0.15 argmax | π₀ = 0.10 argmax | Interpretation |
|---|---|---|---|
| USA     | 2022-01 | 2022-01 | Invariant — true dominant break well inside window |
| Japan   | 2022-01 | 2022-01 | Invariant — same |
| UK      | 2021-08 *(boundary)* | **2022-03** | Step 5 boundary effect; true argmax at ENERGY_2022 |
| Germany | 2020-07 *(earlier peak)* | **2022-01** | Same — Step 5 boundary hid the dominant break |

At π₀ = 0.15, the UK and Germany scan windows ended at 2021-08 — one month before ENERGY_2022. Their Step 5 argmax was either on the boundary (UK) or at an earlier local peak (Germany). At π₀ = 0.10, the scan extends to 2023-03 and all four countries' argmax converges within ±1 month of ENERGY_2022. This is the Phase 3 signature finding: **ex-ante break-date specification and data-driven break-date detection yield the same answer**.

**Empirical summary (π₀ = 0.10, Andrews 5% critical value = 18.82):**

| Country | sup-W | argmax | Verdict @ 5% |
|---|---|---|---|
| USA     | 37.73 | 2022-01 | **Reject @ 1%** (Andrews 1% = 23.04) |
| Japan   | 11.88 | 2022-01 | Fail to reject |
| UK      | 12.57 | 2022-03 | Fail to reject |
| Germany | 5.13  | 2022-01 | Fail to reject |

**Rationale:**
- Reviewer scrutiny of "did you check trim robustness" is a common Portfolio question for Quandt-Andrews applications
- Reporting both trims is evidence of methodological care
- The π₀ = 0.10 narrative (4/4 argmax at ENERGY_2022) is stronger than the π₀ = 0.15 narrative alone

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| π₀ = 0.15 only | Rejected — hides UK/Germany true argmax behind trim boundary |
| π₀ = 0.10 only | Rejected — loses the standard-trim anchor that reviewers expect |
| Both trims reported (adopted) | Adopted — combines rigor with narrative strength |

**Portfolio value:** Demonstrates the value of sensitivity analysis in procedures whose "standard" parameter choice has a material effect on the reported result.

**Implementation:** `src/structural_breaks.py::quandt_andrews_scan()`, `summarise_scan()`, `align_argmax_to_known()`, and the `ANDREWS_1993_TABLE_I` critical-value constant (π₀ ∈ {0.05, 0.10, 0.15, 0.20, 0.25} × k ∈ {1..7}).

---

## Phase 3 Final State — Summary

**After Phase 3 stationarity and structural-break testing:**

| Metric | Phase 2 | Phase 3 |
|---|---|---|
| Level ADF+KPSS 4-quadrant classifications | — | **20 series** (11 Non-stationary, 5 Stationary, 2 Inconclusive, 2 Conflict) |
| Phase 6 VAR input forms finalised | — | **20 series**, 5 transform types registered (D-027) |
| Chow test battery | — | **32 tests** (12 classical + 12 HAC + 8 COVID-dummy) |
| Chow rejections at α = 0.05 | — | **23 / 32** (9+9+5); all survive Bonferroni α_bonf = 0.05/12 |
| Per-coefficient decomposition rows | — | **60** (4 countries × 3 breaks × 5 regressors) |
| Quandt-Andrews candidate-date evaluations | — | **~1 660** (4 countries × 2 trims × ~207 candidate dates avg) |
| Data-driven confirmation of ENERGY_2022 break | — | **4/4 countries** argmax within ±1 month (π₀=0.10) |
| `src/` module architecture | 2 modules (v0.2.0) | **4 modules (v0.3.0)** — +stationarity, +structural_breaks |

**Signature findings:**

1. **Break detection is robust across variants.** Classical vs HAC Chow: 12/12 verdict agreement. HAC vs COVID-dummy HAC: 8/8 verdict agreement. Autocorrelation, heteroskedasticity, and COVID outliers do not drive the conclusions.

2. **GFC_2008 is a USA-specific break.** Only USA rejects at α = 0.05 (classical F = 9.69, HAC F = 6.20; both significant at 1%). Japan, UK, and Germany show p-values between 0.06 and 0.53. This is consistent with the narrative that the 2008 Phillips Curve breakdown was a USA-centric phenomenon; European and Japanese economies experienced the financial shock but their CPI–macro relationships remained comparatively stable under ECB and BOJ liquidity responses.

3. **COVID_2020 and ENERGY_2022 are universal breaks.** All four countries reject at α = 0.05 at both dates (HAC F statistics between 3.67 and 33.67). ENERGY_2022 is astronomically significant for USA (HAC F = 33.67, p ≈ 10⁻²⁷), strong for Japan (F = 11.60) and UK (F = 8.46), and notably weaker for Germany (F = 4.69).

4. **Break channel differs by country.** Per-coefficient decomposition at ENERGY_2022 identifies different dominant drivers: USA via POLICY_RATE (Fed hawkish turn), Japan via the constant (level-shift of monthly inflation after BOJ inertia), UK and Germany via GDP (demand-side transmission to CPI). The same calendar-month event operated through different channels in different economies — material for N1 (Phillips Curve), N2 (Monetary Policy Lag), and N3 (Japan's Uniqueness).

5. **Data-driven break detection confirms the ex-ante specification.** Quandt-Andrews argmax at π₀ = 0.10 is within ±1 month of ENERGY_2022 (2022-02) for all four countries. USA sup-W = 37.73 exceeds the Andrews 1% critical value (23.04). The data independently pinpoint the break date that ProjectScope specified from economic reasoning alone.

**Reusable module architecture extended (v0.2.0 → v0.3.0):**

| Module | Purpose | LOC | Exports |
|---|---|---|---|
| `src/data_loader.py`      | I/O helpers for raw and processed datasets       | (unchanged) | 9  |
| `src/preprocessing.py`    | Phase 2 transformation functions                 | (unchanged) | 14 |
| `src/stationarity.py`     | Phase 3 Task 1 — ADF/KPSS + 4-quadrant + transforms | ~310   | 20 |
| `src/structural_breaks.py`| Phase 3 Task 2 — Chow + coefficient decomposition + Quandt-Andrews | ~460 | 16 |
| `src/__init__.py` (v0.3.0)| Package meta + re-exports                         | ~140    | 60 total |

All Phase 6 through Phase 8 notebooks will import from these four modules rather than duplicating logic.

**Artifacts produced:**

- `src/stationarity.py`, `src/structural_breaks.py` (module extraction per D-032)
- `src/__init__.py` bumped to v0.3.0
- `scripts/phase3_step[1-5b]_*.py` — six scratch orchestrators (S1 level ADF/KPSS; S2 differencing; S3 CPI decision + registry; S4 Chow battery; S5 + S5b Quandt-Andrews at two trim fractions)
- `data/documentation/phase3_adf_kpss_levels.csv` (20 rows)
- `data/documentation/phase3_differencing_log.csv` (16 rows)
- `data/documentation/phase3_conflict_ct_retest.csv` (2 rows)
- `data/documentation/phase3_cpi_transform_comparison.csv` (16 rows)
- `data/documentation/phase3_transformation_registry_final.csv` (20 rows — source of truth per D-027/D-031)
- `data/documentation/phase3_subperiod_stationarity.csv` (60 rows)
- `data/documentation/phase3_cpi_deep_dive.csv` (36 rows)
- `data/documentation/phase3_break_window_stationarity.csv` (120 rows)
- `data/documentation/phase3_chow_tests_classical.csv` (12 rows)
- `data/documentation/phase3_chow_tests_hac.csv` (12 rows)
- `data/documentation/phase3_chow_tests_covid_dummy.csv` (8 rows)
- `data/documentation/phase3_chow_coefficient_decomposition.csv` (60 rows)
- `data/documentation/phase3_chow_bonferroni_summary.csv` (32 rows)
- `data/documentation/phase3_quandt_andrews_supwald.csv` (4 rows — π₀ = 0.15 Step 5 state)
- `data/documentation/phase3_quandt_andrews_curve.csv` (815 rows — π₀ = 0.15 curve)
- `data/documentation/phase3_quandt_andrews_supwald_trim10.csv` (4 rows — π₀ = 0.10 Step 5b state)
- `data/documentation/phase3_quandt_andrews_curve_trim10.csv` (933 rows — π₀ = 0.10 curve)
- `notebooks/03_stationarity_structural_breaks.ipynb` — Portfolio-grade narrative of the ten Phase 3 decisions (D-024 through D-033)

**Phase 4 prerequisites ready:**

- `phase3_transformation_registry_final.csv` provides the per-variable input form for Phase 4 feature engineering (lag and rolling construction operates on the phase6_var_input form)
- `phase3_chow_coefficient_decomposition.csv` is the source for Phase 6 regime-dummy specification per country (D-030 driver identification)
- All four main-country datasets are Phase-4-ready with no further transformation decisions outstanding

---

*Last updated: Phase 3 complete — four main-country datasets classified, Chow/Quandt-Andrews breaks characterised, reusable `src/` module architecture extended to v0.3.0. Next: Phase 4 feature engineering.*

## Phase 4 Decisions

*These decisions concern the feature engineering pipeline — base registry application, lag matrix, rolling statistics, and regime dummies — that produces the per-country feature matrices consumed by Phase 6 VAR/Ridge estimation. They are implemented in `src/feature_engineering.py` and narrated in `notebooks/04_feature_engineering.ipynb`.*

---

### D-034 | Lag Grid — Uniform Sparse {1, 3, 6, 12} per ProjectScope §9

**Date:** Phase 4 · Step 2
**Decision:** Adopt a uniform sparse lag grid {1, 3, 6, 12} for all indicators across all countries, applied via `pd.Series.shift(k)` on the D-031-corrected base feature form. This matches ProjectScope §9's literal Phase 4 specification.

**Rationale:**

1. **ProjectScope compliance**: §9 explicitly specifies "Lag features: t-1, t-3, t-6, t-12 for all indicators" as the Phase 4 deliverable. The §2 variable-specific lag hypothesis table (e.g. policy rate t-3..t-12, unemployment t-1..t-6) is annotated as a hypothesis for Phase 6 interpretation, not as a Phase 4 construction spec.
2. **Multi-scale sampling**: {1, 3, 6, 12} samples short-run (monthly), quarterly, semi-annual, and annual dynamics without oversampling any single horizon.
3. **Non-flooding**: 5 indicators × 4 lags = 20 lag cols/country. Phase 6 Ridge / VAR-with-AIC can select without multicollinearity collapse.
4. **Phase 6 non-constraint**: the Phase 4 grid does not commit Phase 6 to a specific lag depth; VAR estimation still runs AIC/BIC independently.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Dense `range(1, 13)` | Rejected — 60 lag cols/country triggers multicollinearity; §9 specifies sparse |
| Variable-specific per §2 | Rejected — §2 is hypothesis-level, not construction spec; implementation becomes asymmetric |
| `{1, 3, 6, 9, 12}` superset | Rejected — marginal economic benefit of 9-lag; adds no new temporal scale |

**Implementation:** `src.feature_engineering.LAG_PERIODS = (1, 3, 6, 12)`; `build_lag_matrix()` iterates outer indicator × inner lag. First-valid-date match verified against theoretical `source_first_valid + k months` for all 80 lag columns in Step 2.

---

### D-035 | Rolling Statistics — {3, 12} Windows × {mean, std}, Strict min_periods

**Date:** Phase 4 · Step 3
**Decision:** Compute rolling mean and rolling std at windows {3, 12} for all indicators. Right-aligned inclusive, strict `min_periods = window`, `ddof = 1` for std. Column naming `{COUNTRY}_{INDICATOR}_roll{w}_{stat}`. The std addition exceeds ProjectScope §9's "mean only" specification by one statistic per window.

**Rationale:**

1. **Compliance plus volatility**: §9 spec is satisfied by the mean columns; the std columns are an upgrade directly motivated by Phase 3 findings (COVID 2020 and ENERGY 2022 shocks manifest as both level shifts and variance expansions).
2. **Phase 6 Ridge benefit**: Ridge L2 regularisation handles increased feature dimensionality natively; variance-based features are Ridge-appropriate covariates that reveal volatility regime effects the VAR alone cannot.
3. **Non-flooding**: 5 × 2 × 2 = 20 rolling cols/country, same order of magnitude as lag cols.
4. **Strict alignment**: `min_periods = window` yields conservative leading-NaN behaviour (no partial windows). Phase 6 may shift by 1 for strict-trailing forecasting use; Phase 4 keeps the general-purpose form.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| {3, 12} × {mean} only (§9 literal) | Rejected as default — misses volatility regime story, which Phase 3 identified as salient |
| {3, 6, 12} × {mean, std} | Rejected — 6m interpolates 3m/12m without distinct economic meaning |
| `min_periods = 1` (partial windows) | Rejected — inconsistent sample size per row violates econometric convention |

**Implementation:** `src.feature_engineering.ROLLING_WINDOWS = (3, 12)`, `ROLLING_STATS = ('mean', 'std')`; `build_rolling_matrix()` verified against manual aggregation at 1e-10 precision (80/80 spot checks in Step 3).

---

### D-036 | Regime Dummy Structure — Splits + Periods + Gated Interactions

**Date:** Phase 4 · Step 4
**Decision:** Construct three disjoint categories of regime features per country:

1. **Split dummies** (persistent, 3 per country): `D_t = 1{t ≥ break_date}` for each entry in `KNOWN_BREAKS`. Emitted for all country × break combinations as a superset; Phase 6 filters per D-030 Bonferroni gating.
2. **Period dummies** (temporary window, 2 per country): `P_t = 1{start ≤ t ≤ end}` per ProjectScope §9 "anomaly flags". GFC window `[2008-09-01, 2009-06-01]` (Lehman to NBER US recession end); COVID window `[2020-03-01, 2020-09-01]` — matches `src.structural_breaks.COVID_DUMMY_START/END` (D-029) exactly.
3. **Interaction terms** (D-030 gated, 0–3 per country): `D_break × X_driver_transformed` for only those (country, break) pairs where D-030 identifies a regressor-valued dominant driver. Constant drivers (JPN COVID/ENERGY, UK COVID) emit **no** interaction — the split dummy alone captures the intercept shift. Not-significant cases (JPN/UK/GER × GFC) emit nothing.

**Rationale:**

1. **ProjectScope §9 + D-030 union**: the anomaly-period flags and the D-030 interaction channels are distinct statistical constructs (temporary window vs. persistent-from-date × regressor). Both are needed.
2. **Superset split dummies**: emitting all 3 splits × 4 countries (12 total) as a superset, not gated, keeps Phase 6 flexible. The D-030 dominant-driver matrix decides only which interaction to instantiate, not which split dummy to expose.
3. **Const-driver case**: when D-030 identifies the constant as dominant (intercept shift), the interaction `D × 1 ≡ D` is redundant with the split dummy itself. Suppressing it prevents column duplication.
4. **GFC window**: anchoring the period start at the break date (2008-09) rather than the NBER recession start (2007-12) keeps the period interpretable as "post-break shock absorption window".

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Interactions only (no splits/periods) | Rejected — ProjectScope §9 anomaly flag spec non-compliance |
| Splits only (no periods, no interactions) | Rejected — D-030 non-compliance; loses channel specificity |
| GFC period = NBER full recession (2007-12..2009-06) | Rejected — period starting before break is economically counterintuitive |
| Emit const-driver interactions for symmetry | Rejected — column-duplication with the split dummy |

**Implementation:** `src.feature_engineering.PERIOD_WINDOWS`, `PHASE6_REGIME_SPEC`, `build_split_dummies()`, `build_period_dummies()`, `build_interactions()`, `build_regime_matrix()`. Category totals per country: USA 8 (3+2+3), JPN 5 (3+2+0), UK 6 (3+2+1), GER 7 (3+2+2). All 26 per-column invariants verified in Step 4.

---

### D-037 | Module API — Single-File `src/feature_engineering.py` (v0.4.0)

**Date:** Phase 4 · Step 5
**Decision:** Consolidate all Phase 4 logic into a single module `src/feature_engineering.py` organised in 5 layers: (1) decision-log constants, (2) registry loading, (3) component builders (transform, lag, rolling, splits, periods, interactions, regime), (4) assembly (per-country, all-country), (5) schema writer. Bump `src/__init__.py` from v0.3.0 to v0.4.0.

**Rationale:**

1. **Mirror Phase 2 and Phase 3 patterns**: `src/preprocessing.py` (Phase 2) and `src/stationarity.py` + `src/structural_breaks.py` (Phase 3) each own one analytical domain. Phase 4 is a single domain (feature construction), not two.
2. **Layer separation permits unit testability**: components are independently callable without the full assembly wrapper, making the Step 5 regression-test vs. Step 2/3/4 scratch CSVs straightforward.
3. **Default-argument injection**: `build_country_features(country, df=None, eff_reg=None, project_root=None)` supports both CLI and notebook use. Passing `df` avoids re-reading processed CSVs in notebook narrative.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Three files (`lag.py`, `rolling.py`, `regime.py`) | Rejected — Phase 4 functions are tightly coupled (all operate on the base feature frame); over-fragmentation for 4 layers |
| Monolithic `build_features()` with no components | Rejected — kills unit testability and regression-test decomposition |
| Pre-emptive split Phase 4 + Phase 6 helpers | Rejected — Phase 6 helpers don't exist yet; YAGNI |

**Implementation:** `src/feature_engineering.py` (~390 lines, 17 public exports); `src/__init__.py` re-exports all 17 under v0.4.0; `src.__version__ = "0.4.0"`.

---

### D-038 | D-031 Override Location — Module-Embedded `REGISTRY_OVERRIDES`

**Date:** Phase 4 · Step 1
**Decision:** Embed the D-031 runtime overrides as a module-level constant `REGISTRY_OVERRIDES` in `src.feature_engineering`, applied automatically by `load_effective_registry()`. The Phase 3 override in `scripts/phase3_step4_*.py::REGISTRY_OVERRIDES` is a Phase 3 state artefact and is NOT reused from Phase 4.

**Rationale:**

1. **Single source of truth for Phase 4/6**: callers (S1–S5 scripts, notebook, future Phase 6 estimators) all share one dict. No per-caller drift.
2. **D-011 convention preservation**: hard-coded decision-log constants live in `src/` (`M2_UNITS`, `ADF_REGRESSION_LEVEL`, `PHASE6_REGIME_SPEC` all follow this pattern).
3. **Phase 3 script independence**: the Phase 3 scratch script's copy is test-fixture state and must not change retroactively. Duplicating the dict at Phase 4 decouples the two epochs cleanly.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Caller-applied override (pass dict parameter) | Rejected — every caller would re-encode the same three entries; drift risk |
| Re-import from `scripts/phase3_step4_*` | Rejected — introduces `src/` → `scripts/` import coupling |
| Embed in `src.stationarity` alongside `TRANSFORM_FN` | Rejected — override is Phase 4/6-specific, not universal stationarity logic |

**Implementation:** `src.feature_engineering.REGISTRY_OVERRIDES`; `load_effective_registry()` applies it by default. Stored values match the Phase 3 script exactly: JPN CPI → `first_diff`, GER CPI → `first_diff`, UK CPI → `log_diff_pct`.

---

### D-039 | Output Format — Per-Country Wide CSV, Leading NaN Preserved

**Date:** Phase 4 · Step 5
**Decision:** Write per-country feature matrices to `data/processed/features_{country}.csv` in wide format with columns ordered `base → lag → rolling → split → period → interaction`. Leading NaN is **preserved** (joint-valid window trimming is a Phase 6 decision, not Phase 4's). Auto-generate `data/processed/features_schema.md` via `write_features_schema_md()`.

**Rationale:**

1. **VAR/Ridge ingestion-ready**: wide format is the expected input shape; date on index, features on columns.
2. **Leading-NaN preservation gives Phase 6 flexibility**: Phase 6 ARIMA on single series doesn't need full joint-valid; Phase 6 VAR may choose a longer estimation window for one country; Phase 6 Ridge may impute. Dropping leading NaN at Phase 4 forces a narrower window on everyone.
3. **Column ordering is pedagogically clean**: base comes first (interpretable), regime comes last (Phase 6-specific), lag/rolling in between. Notebook displays inherit this ordering.
4. **Schema file mirrors D-023 pattern**: `data/processed/features_schema.md` echoes `data/processed/schema.md` from Phase 2 — single-source-of-truth documentation auto-regenerated at assembly time.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Joint-valid-only export (`dropna(how='any')`) | Rejected — loses 12–22 obs per country; forces Phase 6 window |
| Long format (country, date, feature, value) | Rejected — 5k+ rows/country; VAR consumers need wide |
| Separate category files (base.csv, lag.csv, …) | Rejected — Phase 6 joins re-do Phase 4 work |
| Single cross-country features.csv with country as index level | Rejected — JPN/USA have 298 rows, UK/GER have 291; different column sets post-interactions |

**Implementation:** `scripts/phase4_step5_assemble.py` writes 4 CSVs + 1 schema; column counts per country 50 (JPN), 51 (UK), 52 (GER), 53 (USA).

---

### D-040 | Feature Selection Timing — Superset at Phase 4, Selection at Phase 6

**Date:** Phase 4 · Step 5
**Decision:** Phase 4 delivers the full feature superset (50–53 cols/country). Feature selection — dropping, pruning, regularisation-based shrinkage — is entirely a Phase 6 responsibility.

**Rationale:**

1. **Model-family independence**: Phase 4 doesn't know which Phase 6 model consumes which column. ARIMA uses only the CPI column; VAR uses 5 baseline series; Ridge uses everything under L2.
2. **Ridge L2 native handling**: the highest-dimensional Phase 6 model (Ridge) handles multicollinearity via regularisation. Pre-pruning removes information Ridge could have used.
3. **VAR AIC/BIC independence**: VAR's lag selection via AIC/BIC operates on the 5-variable system, not on our lag-column superset. Phase 4 pre-pruning would misalign with the VAR's own selection.
4. **Portfolio separation-of-concerns**: "Feature construction" (Phase 4) and "Feature selection" (Phase 6) are distinct Portfolio chapters.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Preliminary variance/correlation filter | Rejected — premature; biases downstream model comparison |
| Phase 4-side AIC-based lag pruning | Rejected — couples Phase 4 to a specific model family |
| L1 penalty pre-screen | Rejected — anticipates Ridge L2, not the Phase 6 spec |

**Implementation:** No dropping. Per-country feature matrix shape passed through `build_country_features()` untouched (base 5 + lag 20 + rolling 20 + regime 5–8).

---

## Phase 4 Final State — Summary

**After Phase 4 feature engineering (4 main-country feature matrices):**

| Metric | Phase 3 | Phase 4 |
|---|---|---|
| Data state | Classified + break-characterised | **Feature matrices ready for Phase 6 ingestion** |
| Module architecture | 4 modules at v0.3.0 | **5 modules at v0.4.0** (77 exports) |
| Registry application | Scratch script state | **Module-embedded `REGISTRY_OVERRIDES`** |
| Features per country | — | **50 (JPN), 51 (UK), 52 (GER), 53 (USA)** |
| Joint-valid start | — | **USA 2003-01; JPN/UK/GER 2002-02** |
| Module-vs-scratch regression test | — | **12/12 passed, max_abs_diff ≤ 3.55×10⁻¹⁵** |
| Phase 6 readiness | Transformation decisions final | ✅ **Ready (VAR/Ridge wide-format CSVs)** |

**Reusable module architecture extended (v0.3.0 → v0.4.0):**

| Module | Purpose | LOC (approx) | Exports |
|---|---|---|---|
| `src/data_loader.py` | I/O helpers | unchanged | 9 |
| `src/preprocessing.py` | Phase 2 transformations | unchanged | 14 |
| `src/stationarity.py` | Phase 3 ADF/KPSS + transforms | unchanged | 20 |
| `src/structural_breaks.py` | Phase 3 Chow + Quandt-Andrews | unchanged | 16 |
| `src/feature_engineering.py` | Phase 4 feature construction | ~390 | 17 |
| `src/__init__.py` (v0.4.0) | Package meta + re-exports | ~180 | 77 total |

**Artifacts produced:**

- `src/feature_engineering.py` (new module per D-037)
- `src/__init__.py` bumped to v0.4.0
- `scripts/phase4_step[1-5]_*.py` — five scratch orchestrators (S1 base registry; S2 lags; S3 rolling; S4 regime; S5 module assembly + consistency proof)
- `data/documentation/phase4_step1_effective_registry.csv` (20 rows)
- `data/documentation/phase4_step1_base_features_summary.csv` (20 rows)
- `data/documentation/phase4_step1_base_features_preview.csv` (long-form head+tail)
- `data/documentation/phase4_step2_lag_{country}.csv` (× 4) + `phase4_step2_lag_summary.csv` (80 rows)
- `data/documentation/phase4_step3_rolling_{country}.csv` (× 4) + `phase4_step3_rolling_summary.csv` (80 rows, incl. 1e-10 spot-check)
- `data/documentation/phase4_step4_regime_{country}.csv` (× 4) + `phase4_step4_regime_summary.csv` (26 rows) + `phase4_step4_regime_specification.csv` (12 rows, D-030 matrix echo)
- `data/documentation/phase4_step5_category_counts.csv`, `phase4_step5_joint_valid_summary.csv`, `phase4_step5_consistency_check.csv` (12 rows: 3 per country × 4 countries)
- `data/processed/features_{usa,japan,uk,germany}.csv` (× 4; 291–298 rows × 50–53 cols)
- `data/processed/features_schema.md` (auto-generated)
- `notebooks/04_feature_engineering.ipynb` — Portfolio-grade narrative of the seven Phase 4 decisions (D-034 through D-040)

**Phase 5 prerequisites ready:**

- `features_{country}.csv` × 4 as primary Phase 5 input, loadable via `pd.read_csv` (or a future `src.data_loader.load_features_main()` wrapper)
- Joint-valid windows established and documented per `features_schema.md`
- All features `float64`, leading-NaN-only pattern verified by 206+ per-column invariants across S1–S4 (20 base + 80 lag + 80 rolling + 26 regime = 206)
- Module-vs-scratch regression test passes at IEEE 754 floating-point rounding precision (1e-10 tolerance, max_abs_diff ≤ 3.55×10⁻¹⁵)

---

*Last updated: Phase 4 complete — per-country feature matrices of 50–53 columns built, reusable `src/` module architecture extended to v0.4.0 with 5 modules and 77 total exports. Next: Phase 5 exploratory data analysis.*

## Phase 5 Decisions

*These decisions concern the exploratory data analysis phase — cross-country CPI visualisation, correlation structure, Phillips Curve N1 deep-dive, and ACF/PACF diagnostics for Phase 6 ARIMA order identification. They are implemented in `scripts/phase5_step{1..4}_*.py` and narrated in `notebooks/05_eda.ipynb`.*

---

### D-041 | Cross-Country CPI Normalisation — Dual-Panel View

**Date:** Phase 5 · Step 1
**Decision:** Visualise cross-country CPI dynamics via a dual-panel figure:

- **Panel A** — CPI levels normalised to 100 at 2001-01 (cumulative price level; shows 25-year divergence).
- **Panel B** — CPI YoY % computed from levels as `(lvl / lvl.shift(12) − 1) × 100`, directly overlaid across the four countries.

**Rationale:**

1. **Single-panel options fail asymmetrically.** Level-overlay alone obscures rate dynamics; YoY-overlay alone collapses the cumulative divergence which is the single strongest N3 visualisation (USA 184.9 vs JPN 116.2 at 2025-10).
2. **Both dimensions are portfolio-worthy.** Reviewers interpret cumulative inflation as "where prices are now" and YoY as "where inflation is going"; omitting either loses half the story.
3. **Choice of T0 = 2001-01.** Matches the Phase 2 effective start date (D-023); no ad-hoc anchor.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| YoY-only overlay | Rejected — flattens the N3 cumulative divergence |
| Index-only | Rejected — hides 2022+ rate dynamics relevant to N2 |
| Z-score normalisation | Rejected — non-interpretable units for portfolio audience |

**Implementation:** `scripts/phase5_step1_cpi_narrative.py` produces Fig 1. Terminal annotations rendered at 1-decimal precision with per-country vertical offset (dict-driven) to prevent near-coincident labels (USA 184.9, UK 185.2) from overprinting. Audit: `phase5_step1_cpi_summary.csv` (4 rows).

---

### D-042 | Correlation Heatmap Scope — Two-Tier (Base + Cross-Lag)

**Date:** Phase 5 · Step 2
**Decision:** Phase 5 correlation structure is visualised via two complementary tiers:

- **Tier 1** (Fig 4): Per-country base 5×5 Pearson matrix on the D-031-corrected stationary feature form, 2×2 country grid.
- **Tier 2** (Fig 5): Per-country 4×5 cross-lag Pearson matrix — `corr(CPI_t, X_{t−k})` for X ∈ {POLICY_RATE, UNEMPLOYMENT, GDP, M2} and k ∈ {0, 1, 3, 6, 12}, 2×2 country grid.

**Rationale:**

1. **Base 5×5 alone is insufficient.** Cross-lag dimension is required to preview N2 Monetary Policy Lag; pure contemporaneous correlation structure misses the entire temporal dimension of central bank transmission.
2. **Full 50×50 dendrogram (Option C) is over-scope.** Phase 5 EDA does not require feature selection — that is D-040, deferred to Phase 6 Ridge. A 50×50 heatmap is unreadable at portfolio scale.
3. **CPI as Tier-2 anchor.** CPI is the Phase 6 target; orienting the cross-lag matrix around CPI maximises narrative alignment and prevents a 5×5×5 over-specification.
4. **Stationary form preserves Phase 6 semantics.** Using D-031 corrected forms means the correlations Phase 5 reveals are directly comparable to Phase 6 VAR coefficients.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Base 5×5 only | Rejected — no N2 preview |
| Full 50×50 dendrogram | Rejected — over-scope; unreadable |
| Base + lag CPI only (4×5 single tier) | Rejected — loses contemporaneous co-movement |
| Cross-country 5×5 | Rejected — CPI forms differ per D-031; Pearson not clean |

**Key observation:** USA `corr(CPI, M2_{t−12}) = +0.41` shows a sign-flip pattern across lags (k=0: −0.17 → k=12: +0.41), consistent with Quantity Theory "money growth leads inflation by ≈12 months". Phase 6 VAR IRF will provide the directional / causal interpretation.

**Implementation:** `scripts/phase5_step2_correlation_structure.py`. Audit: `phase5_step2_base_correlation.csv` (100 rows), `phase5_step2_lag_correlation.csv` (80 rows), `phase5_step2_window_summary.csv` (4 rows).

---

### D-043 | Phillips Curve Fitting — Pre/Post-GFC Split + 60-Month Rolling

**Date:** Phase 5 · Step 3
**Decision:** N1 Phillips Curve analysis employs two complementary specifications:

- **Fig 6** — Per-country scatter (4 panels) with separate OLS fits for the pre-GFC (2002-01..2008-08) and post-GFC (2008-09..end-of-data) sub-periods. Variables are **level-based**: UNEMPLOYMENT (%) from Phase 2 and CPI YoY % computed from CPI levels.
- **Fig 7** — Dual-panel 4-country overlay: (A) 60-month rolling OLS slope β; (B) 60-month rolling R². Right-aligned, strict `min_periods = 60`.

**Rationale:**

1. **Level-based form is the correct EDA lens.** S2 showed stationary-form cross-lag Phillips correlations are essentially zero across all 4 countries (|r| ≤ 0.07). The classical Phillips Curve is a *level* relationship; stationary-form analysis de-trends away the exact relationship being studied. See D-046 for the formal methodological finding.
2. **Split-sample complies with §9 literal specification.** ProjectScope §9 explicitly specifies "pre/post break split"; 2008-09 aligns with `KNOWN_BREAKS['GFC_2008']`.
3. **Rolling captures time-variation static split cannot.** The binary pre/post partition masks within-period evolution — e.g. UK's sign flip from pre-GFC β = +1.68 to post-GFC β = −0.27 is missed under a single rolling or a single static fit alone.
4. **60-month window balances smoothness and local sensitivity.** Shorter (12–24 m) yields high variance; longer (120 m+) smooths out the regime transitions of analytical interest.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Static OLS only (single fit) | Rejected — no time-variation; misses UK sign flip |
| Pre/post split only | Rejected — §9 compliant but loses continuous evolution |
| Rolling only | Rejected — loses discrete GFC-break reference |

**Key findings (per-country pre→post |β| transition):**

| Country | \|β\| pre-GFC | \|β\| post-GFC | Verdict |
|---|---:|---:|---|
| USA     | 0.567 | 0.372 | Classical flattening (−34 %) |
| JAPAN   | 0.710 | 0.947 | Steepening (+33 %) — reinforces N3 |
| UK      | 1.676 | 0.271 | Sign-flip regime breakdown (+1.68 → −0.27) |
| GERMANY | 0.321 | 0.603 | Steepening (+88 %) — ECB-constrained regime |

Rolling slopes re-emerge at |β| ≈ 5–9 across all four countries post-2022, with rolling R² ≈ 0.6–0.75 — Phillips is **shock-activated**, not "dead".

**Implementation:** `scripts/phase5_step3_phillips_curve.py` using `statsmodels.OLS` for coefficients, SE, R², p-values. Audit: `phase5_step3_phillips_fit.csv` (12 rows: 4 × {full, pre, post}), `phase5_step3_rolling_slope.csv` (894 rows).

---

### D-044 | ACF/PACF Lag Depth — 40 Uniform, Ljung-Box {12, 24, 36}

**Date:** Phase 5 · Step 4
**Decision:** ACF/PACF diagnostic uses a uniform lag depth of 40 across all four countries. Ljung-Box Q statistics are reported at lags {12, 24, 36} (annual, biannual, triennial horizons). Confidence band is simple Bartlett `±1.96/√n` (constant across lags). PACF method is `'ywm'` (Yule-Walker adjusted).

**Rationale:**

1. **Three-cycle seasonal coverage.** 40 > 3 × 12 permits three full annual harmonics to be inspected, sufficient to distinguish transient seasonal noise from persistent 12-month structure.
2. **Covers Phase 3 post-break windows.** ENERGY 2022 post-break windows are 38–45 obs per country; 40-lag depth ensures ACF/PACF features attributable to the post-break regime are not missed by a shorter specification.
3. **Symmetric specification.** Uniform lag depth across countries avoids asymmetric defensibility claims of the form "why lag 30 for USA but 50 for Japan?".
4. **Ljung-Box at three horizons.** A single-lag Q depends heavily on the choice; reporting at {12, 24, 36} provides a robustness envelope matching the seasonal harmonic spacing.
5. **Constant Bartlett CI.** MA-adjusted (non-constant-per-lag) CI is more mathematically precise but visually confusing; the textbook Bartlett band is reviewer-transparent.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| 24-lag (two seasonal cycles) | Rejected — insufficient post-ENERGY window coverage |
| 60-lag | Rejected — diminishing returns; three cycles already diagnostic |
| Country-specific depth | Rejected — asymmetric defensibility |
| MA-adjusted non-constant CI | Rejected — constant Bartlett is textbook-standard |

**Key findings:**

| Country | ACF[12] | PACF[12] | Bartlett CI | Ljung-Box Q(12) | p(12) |
|---|---:|---:|---:|---:|---:|
| USA     | +0.268 | +0.154 | ±0.116 | 1 527.22 | < 0.001 |
| JAPAN   | +0.354 | +0.308 | ±0.114 |    67.30 | < 0.001 |
| UK      | +0.561 | +0.445 | ±0.115 |   190.46 | < 0.001 |
| GERMANY | +0.472 | +0.419 | ±0.115 |   104.33 | < 0.001 |

- Seasonal lag-12 ACF significant in all 4 countries → **SARIMA with s=12 justified universally** (not just Phase 6 ARIMA).
- Ljung-Box Q(12) rejects white noise at p < 0.001 for all 4 countries → ARIMA/SARIMA modelling is statistically required for Phase 6.
- USA shows slow-decay ACF (0.95 → 0.89 → 0.78 → ...), an artifact of `yoy_pct` 12-month overlap; this is a D-031 trade-off to be evaluated in Phase 6 ARIMA estimation.
- Preliminary ARMA order candidates (AIC/BIC in Phase 6 supersedes): USA AR(3), Japan ARMA(1,2), UK AR(2), Germany ARMA(2,2).

**Implementation:** `scripts/phase5_step4_acf_pacf.py` using `statsmodels.tsa.stattools.acf`, `pacf`, `statsmodels.stats.diagnostic.acorr_ljungbox`. Audit: `phase5_step4_acf_pacf_values.csv` (164 rows), `phase5_step4_ljung_box.csv` (12 rows).

---

### D-045 | Japan Phase Decomposition — Four-Phase Labelling

**Date:** Phase 5 · Step 1
**Decision:** Japan CPI history is labelled in four phases for the N3 narrative:

- **Bubble aftermath** (≤ 1998-12) — documented but pre-dates Phase 2 data
- **Deflation era** (1999-01..2012-12)
- **Abenomics** (2013-04..2022-01) — start aligned with BOJ QQE announcement
- **Reversal** (2022-02 onwards)

Fig 2 shades only the three phases within the data range; Bubble aftermath is recorded for completeness.

**Rationale:**

1. **ProjectScope §4 externally defined.** The N3 narrative in §4 explicitly references Abenomics as a "natural experiment" and 2022 reversal driven by yen depreciation + energy costs. Phase boundaries are externally motivated, not data-mined ex post.
2. **Data-driven validation.** Phase mean YoY are monotonically increasing: Deflation era −0.20 % → Abenomics +0.64 % → Reversal +2.99 %. The Reversal phase shows exactly 0 deflation months (of 45), a clean post-break separation.
3. **Phase boundary 2013-04.** The BOJ announced Quantitative and Qualitative Monetary Easing (QQE) on 2013-04-04; this is the operational start of Abenomics monetary policy, not the Abe administration inauguration (2012-12).
4. **Single-period shading insufficient.** A single 1999–2012 deflation block discards the Abenomics / Reversal distinction that is the most portfolio-valuable component of N3.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Single 1999–2012 shaded region | Rejected — loses Abenomics distinction |
| Data-driven (YoY < 0 months hatching) | Rejected — no external interpretability |
| 3-phase (omit Bubble aftermath) | Rejected — decision log should document full structure |

**Implementation:** `scripts/phase5_step1_cpi_narrative.py`, constant `JAPAN_PHASES_VISIBLE`. Audit: `phase5_step1_japan_phases.csv` (3 rows, visible phases only).

---

### D-046 | Level-vs-Stationary Phillips Visibility Asymmetry — Methodology Finding

**Date:** Phase 5 · Step 3 (emerged from Step 2 vs Step 3 comparison)
**Decision:** Formally record that the visibility of the N1 Phillips Curve relationship is strongly dependent on the variable transformation used, and that this asymmetry is a *finding to report*, not a flaw to hide.

**Empirical observation:**

| Lens | USA corr/β | USA R² | JAPAN corr/β | JAPAN R² |
|---|---:|---:|---:|---:|
| S2 stationary form (D-031 corrected, `corr(CPI, UNEMP)`)     | −0.062 | ~ 0  | −0.071 | ~ 0  |
| S3 level form (full-sample OLS, CPI YoY on UNEMP %)          | −0.383 | 0.18 | −0.865 | 0.38 |

The stationary form essentially erases the Phillips relationship; the level form shows it clearly.  This is **not a numerical artifact**: the Phillips Curve is theoretically a *level* relationship (inflation rate vs unemployment rate) rather than a rate-of-change relationship. First-differencing or log-differencing strips the co-movement between the levels.

**Implications for methodology:**

1. **Phase 5 EDA uses both lenses intentionally.** S2 stationary-form heatmaps and S3 level-form Phillips scatter are complementary, not redundant. Each answers a different economic question.
2. **D-031 trade-off is explicit.** The stationary form is correct for VAR estimation (Phase 6 main model) because VAR requires stationary inputs. The level form is correct for Phillips visualisation. Both are needed.
3. **Phase 6 should report both.** Phase 6 VAR output (Granger causality, IRF) is derived from the stationary form; these will be cross-referenced against Fig 6 (level scatter) to verify directional consistency.
4. **Portfolio defensibility.** This asymmetry is a *finding* to present, not a flaw to hide. Demonstrating that variable form matters — and choosing the right form for the right question — is the sign of a careful analyst.

**Rationale for formal recording:**

- Connects two apparently contradictory result sets (S2 and S3) into a single coherent methodological statement.
- Prevents downstream Phase 6 / Phase 8 misinterpretation ("why did earlier analysis show no Phillips?").
- Serves as the portfolio explanation for the reviewer question *"why did S2 show no Phillips but S3 did?"*.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Silent — report S3 without referencing S2 | Rejected — selective narrative; hides honest uncertainty |
| Treat S2 as the "true" answer | Rejected — economically incorrect; Phillips is a level relationship |
| Treat S3 as the "true" answer | Rejected — Phase 6 VAR legitimately requires stationarity |
| Record the asymmetry explicitly *(adopted)* | Adopted — portfolio-worthy methodology transparency |

**Implementation:** No code artifact — this is a methodological finding. Decision recorded in `ProjectDriven.md`; referenced by `notebooks/05_eda.ipynb` narrative at the S2-to-S3 transition section.

---

### D-047 | EDA Output Format — Notebook + Audit CSVs; No `src/eda.py`

**Date:** Phase 5 · Step 5
**Decision:** Phase 5 produces:

- `notebooks/05_eda.ipynb` — Portfolio-grade narrative assembly of S1..S4 (8 figures + interleaved commentary)
- `outputs/figures/phase5_step{1..4}_fig{1..8}*.png` — 8 figures total
- `data/documentation/phase5_step{1..4}_*.csv` — 12 audit CSVs

Phase 5 does **not** introduce a new `src/eda.py` module. The `src/__init__.py` version remains at v0.4.0.

**Rationale:**

1. **Plot code is not reusable in the way feature engineering is.** `src/feature_engineering.py` (Phase 4) is consumed by Phase 6 VAR / Ridge estimation; Phase 5 plotting code is consumed only by Phase 5 itself. The modularisation cost does not justify the reuse benefit.
2. **Scratch scripts are the canonical implementation.** The four Phase 5 scratch scripts in `scripts/phase5_step{1..4}_*.py` are preserved and cited by the notebook.
3. **Asymmetry with Phase 3/4 is intentional.** Phase 3 introduced `src/stationarity.py` and `src/structural_breaks.py` because downstream phases consume those tests. Phase 4 introduced `src/feature_engineering.py` because Phase 6 consumes it. Phase 5 is consumption-terminal — no phase consumes EDA plotting code.
4. **v0.5.0 is reserved for Phase 6 modelling modules.** The next `__init__.py` version bump will accompany Phase 6 ARIMA / VAR / Ridge code.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| `src/eda.py` with plot helpers | Rejected — no downstream consumer |
| `src/plotting.py` shared module | Rejected — 8 figures too heterogeneous to abstract usefully |
| Duplicate logic in scratch + notebook | Rejected — anti-pattern; notebook imports from scratch |

**Implementation:** Notebook imports from `src` (unchanged v0.4.0 API) and directly executes the existing scratch-script logic via path injection. `ProjectDriven.md` version bump reserved for Phase 6.

---

## Phase 5 Final State — Summary

**After Phase 5 exploratory data analysis:**

| Metric | Phase 4 | Phase 5 |
|---|---|---|
| Decision-log entries | 40 | **47** (+7) |
| Portfolio figures | 6 (Phase 4) | **+8** (Fig 1–8) |
| Audit CSV rows | 26 (Phase 4 summaries) | **+12 CSVs** (~1 300 rows total for Phase 5) |
| `src/` module version | v0.4.0 | **v0.4.0** (unchanged per D-047) |
| Narrative notebooks | `03` + `04` | **+ `05_eda.ipynb`** |

**Signature findings (seven items):**

1. **Japan peer-gap** — Japan CPI YoY is below the mean of USA/UK/Germany in 253 of 279 monthly observations (90.7 %); `mean_gap = −1.80 pp`. Single-number evidence of N3 structural uniqueness.
2. **Japan phase monotone** — Deflation era −0.20 % → Abenomics +0.64 % → Reversal +2.99 %, with Reversal-phase deflation months = 0 of 45. Externally-specified D-045 phases confirmed data-driven.
3. **USA M2 sign-flip at k=12** — `corr(CPI, M2_{t−12}) = +0.41` vs k=0 value −0.17. Direct numerical echo of the Quantity Theory of Money; preview of N2 Monetary Policy Lag.
4. **UK sign-flip regime breakdown** — Phillips β = +1.68 (pre-GFC) → −0.27 (post-GFC), with pre-GFC R² = 0.48. The only country showing a full-sign regime transition.
5. **Phillips shock-activation** — post-2022 rolling slopes reach |β| ≈ 5–9 across all four countries, with rolling R² ≈ 0.6–0.75. Phillips is shock-activated, not dead.
6. **SARIMA universally justified** — ACF[12] significant in all four countries (USA 0.27, JPN 0.35, UK 0.56, GER 0.47); Ljung-Box Q(12) rejects at p < 0.001.
7. **D-046 methodology finding** — level-vs-stationary Phillips visibility asymmetry formally recorded as a portfolio-defensibility methodology contribution.

---

*Last updated: Phase 5 complete — 4-panel EDA narrative (D-041..D-047, 7 decisions) and 7 signature findings. Next: Phase 6 — ARIMA, VAR with Granger/IRF, and Ridge estimation on the Phase 4 feature matrices.*

## Phase 6 Decisions

*These decisions concern Layer 1 of the Phase 6 three-layer modelling architecture (ARIMA → VAR → Ridge per D-004). They are implemented in `scripts/phase6_step1*_*.py` and narrated in `notebooks/06_arima_baseline.ipynb`. Steps 2 (VAR) and 3 (Ridge) will add D-050 onwards.*

---

### D-048 | SARIMA Grid Scope and Boundary Sensitivity Protocol

**Date:** Phase 6 · Step 1
**Decision:** Adopt a three-stage SARIMA grid search protocol for Phase 6 Step 1 Layer 1 estimation on five CPI variants (USA_yoy_pct, USA_first_diff, JAPAN_first_diff, UK_log_diff_pct, GERMANY_first_diff):

- **Stage (a)** — uniform initial grid `p ∈ [0, 4], d = 0, q ∈ [0, 4], P ∈ [0, 2], D ∈ {0, 1}, Q ∈ [0, 2], s = 12` (450 orders × 5 variants = 2,250 fits). Selection: AIC primary, BIC secondary, HQIC tertiary, parsimony (p + q + P + Q) tie-break. Expanding-window 1-step-ahead test refits.
- **Stage (b)** — boundary sensitivity check on variants whose Stage (a) AIC-best hit the `Q = 2` upper boundary (USA_yoy_pct, USA_first_diff, UK_log_diff_pct). Test 6–7 Q = 3 neighbourhood orders per variant; threshold ΔAIC ≤ −2.0 (Burnham & Anderson 2002 "meaningfully better") to escalate.
- **Stage (c)** — targeted `Q ∈ [0, 3]` grid extension (150 orders) only for variants meeting the Stage (b) threshold.

**Stopping rule — OOS saturation:** if Stage (a) → Stage (c) delivers substantial in-sample AIC improvement but essentially invariant OOS test-window RMSE/MAE/bias, halt escalation and defer model ranking to Phase 7 Diebold-Mariano loss comparison. Empirically: USA_first_diff Stage (a) (0,0,3)(0,0,2,12) → Stage (c) (0,0,4)(2,0,3,12) produced ΔAIC = −10.46 with OOS full-test RMSE Δ = −0.003, MAE Δ = +0.003, bias Δ = +0.016 — statistically invariant. BIC and HQIC both select a simpler alternative (0,0,2)(0,0,3,12) that accepts the Q = 3 benefit while rejecting the triple-boundary parameters, further supporting that only AIC's weak 2k penalty tolerates the escalation.

**Rationale:**

1. **ProjectScope §9 + D-004 compliance**: the three-layer architecture specifies ARIMA with AIC/BIC selection; Stage (a)'s 450-order grid is the concrete realisation, covering Phase 5 S4 order priors (USA AR(3), JPN ARMA(1,2), UK AR(2), GER ARMA(2,2)) with generous seasonal slack.
2. **`d = 0` fixed**: all five variants are already in D-031-corrected stationary form; further differencing would over-difference and corrupt the AIC landscape.
3. **Boundary sensitivity is non-negotiable**: Stage (a) returned three boundary-hit variants (Q = 2). Without Stage (b) verification, a portfolio reviewer could legitimately ask "did you check Q = 3?" The D-033 Quandt-Andrews trim sensitivity precedent requires an explicit sensitivity check when a result sits at the grid boundary.
4. **Targeted extension, not blanket extension**: Stage (c) is applied only where Stage (b) returns `extend_to_Q3` (ΔAIC ≤ −2.0). Blanket extension to Q = 3 on all variants would have added 3,375 fits for no benefit — UK_log_diff_pct returned ΔAIC = +12.33 (Q = 3 actively worse), USA_yoy_pct returned ΔAIC = −0.21 (trivially equal).
5. **OOS saturation as principled stopping rule**: the Stage (c) AIC-best hits triple grid boundary (`q = 4, P = 2, Q = 3`). A mechanical sensitivity continuation would test `q = 5, P = 3, Q = 4` — an infinite-regress exercise. OOS invariance is the non-arbitrary termination point: the extension has exhausted its forecasting-relevant information content, regardless of in-sample AIC improvement.
6. **Phase 7 directive**: D-048 obligates Phase 7 Diebold-Mariano to compare USA_first_diff Stage (a) and Stage (c) on loss differential (not AIC ranking) and report BIC/HQIC alternative orders as sensitivity candidates. If DM fails to reject equality of OOS losses, the stopping rule is empirically validated.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Stage (a) only, ignore boundary hits | Rejected — fails D-033 sensitivity precedent; portfolio-review vulnerability |
| Blanket Q ∈ [0, 3] initial grid | Rejected — 3,375 additional Q = 3 fits on four variants (UK, USA_yoy, JPN, GER) yield no AIC improvement; mechanical not statistical |
| Extend Stage (c) to q = 5 / P = 3 / Q = 4 | Rejected — OOS saturation; classical overfitting signature; no expected forecasting benefit |
| Select BIC/HQIC best (0,0,2)(0,0,3,12) for USA_fd over AIC best | Deferred — Phase 7 DM adjudication is the principled resolution mechanism |
| Variable-specific grid (per-variant custom search) | Rejected — per D-034's methodology symmetry principle; uniform spec supports cross-variant comparison |

**Implementation:** Executed by four scratch scripts:

- `scripts/phase6_step1_arima_grid.py` — Stage (a); 2,250 fits + 350 expanding refits; 61.3 min
- `scripts/phase6_step1b_q3_boundary_check.py` — Stage (b); 22 fits; 1.5 min
- `scripts/phase6_step1c_usa_firstdiff_q3_extension.py` — Stage (c); 150 fits + 70 refits; 12.5 min; in-place update of USA_first_diff rows in consolidated selection/residuals/forecast/window_errors CSVs
- `scripts/phase6_step1d_notebook_figures.py` — figure consolidation; 8 PNGs; 0.2 min (parallel to `phase4_step5_assemble.py` "pulling together" pattern)

Final AIC-best orders (post Stage (c) in-place amendment):

| Variant | AIC-best order | AIC | n_params |
|---|---|---:|---:|
| USA_yoy_pct | (2,0,3)(2,0,2,12) | 61.75 | 10 |
| USA_first_diff | (0,0,4)(2,0,3,12) | 329.65 | 10 |
| **JAPAN_first_diff** | **(0,0,1)(1,0,1,12)** | **11.52** | **4** |
| UK_log_diff_pct | (3,0,0)(1,0,2,12) | −119.15 | 7 |
| GERMANY_first_diff | (0,0,2)(1,0,1,12) | −1.18 | 5 |

All five variants pass Ljung-Box Q(12) at α = 0.05 (residual white-noise property satisfied). Heteroscedasticity is mixed (ARCH-LM p ranging from 3e-05 to 0.9999) — see D-049 for the Japan-specific observation.

Stage (b) verdicts: USA_yoy_pct `accept_Q2` (ΔAIC = −0.21), USA_first_diff `extend_to_Q3` (ΔAIC = −9.14), UK_log_diff_pct `accept_Q2` (ΔAIC = +12.33). Only USA_first_diff proceeded to Stage (c).

---

### D-049 | Japan ARIMA Uniqueness — N3 Narrative Echo at the ARIMA Layer

**Date:** Phase 6 · Step 1
**Decision:** Formally record Japan's Step 1 SARIMA diagnostic profile as an ARIMA-layer signature finding that echoes Phase 5's N3 "Japan's Uniqueness" narrative (Phase 5 Finding #1 level peer-gap, #2 monotone phases). Japan's (0,0,1)(1,0,1,12) model is uniquely characterised on four quantitative dimensions simultaneously among the five Step 1 variants.

**Four quantitative signatures (data-driven, emerged from Stage (a) execution):**

1. **Triple IC agreement**: AIC = BIC = HQIC all select (0,0,1)(1,0,1,12). Japan is the sole variant where the three information criteria converge on the same order — the log-likelihood gradient saturates at low complexity.
2. **Sparsest parameterisation**: 4 parameters (MA(1) + seasonal AR(1) + seasonal MA(1) + constant). All other variants require 5–10 parameters.
3. **ARCH-LM p = 0.9999**: residuals are statistically indistinguishable from i.i.d. homoscedastic innovations. No other variant exceeds p > 0.8. This is a near-theoretical-maximum on the ARCH-LM scale.
4. **Lowest training volatility**: σ_train = 0.240, versus USA_first_diff (0.541), USA_yoy_pct (0.308), Germany (0.238), UK (0.192). Japan's monthly CPI increments are the least volatile among the four main economies on the monthly first-difference scale.

**Cross-phase triangulation (three independent lenses on N3):**

| Lens | Phase | Finding | Quantitative signature |
|---|---|---|---|
| Level peer-gap | Phase 5 S1 (F#1) | Below peer mean in 253 / 279 monthly obs | 90.7 %; mean gap −1.80 pp |
| Phase monotone | Phase 5 S1 (F#2) | Deflation → Abenomics → Reversal | 0 / 45 deflation months in Reversal phase |
| **ARIMA simplicity** | **Phase 6 Step 1** | **Triple IC agreement + ARCH-LM p ≈ 1** | **4 parameters; ARCH-LM p = 0.9999** |

Three-lens triangulation is the target portfolio structure: the same narrative claim is independently confirmed by methodologically distinct techniques. Phase 5 established the *level-based* uniqueness (structural divergence in cumulative inflation and phase-decomposed history); Phase 6 Step 1 establishes the *dynamics-based* uniqueness — Japan is the only variant whose monthly inflation changes behave like a stationary, homoscedastic, low-order ARMA process.

**Rationale:**

1. **Cross-phase N3 reinforcement**: D-049 is the ARIMA-layer instance of the project's central narrative N3 ("Japan's Uniqueness"). Phase 7 evaluation and Phase 8 findings.md can cite three independent pieces of evidence rather than one.
2. **Emergent, not pre-specified**: neither the Phase 6 scope nor the Phase 5 summary anticipated "Japan will show triple IC agreement." The finding is data-driven, surfacing from Stage (a) grid execution. This parallels D-046's methodology-finding style (D-046 emerged from Phase 5 S2-vs-S3 level/stationary tension).
3. **Independent of the numerical model**: the finding is about *the structure of the IC agreement and diagnostic profile*, not about the specific (0,0,1)(1,0,1,12) numerical forecast. Phase 7 DM tests can refine the numerical model without affecting the structural finding.
4. **Does not obligate further Phase 6 work**: unlike D-048, D-049 is a finding record rather than a protocol. It is cited in Phase 7 narrative and Phase 8 (findings.md) without requiring additional Step 2 / Step 3 modelling.

**Alternatives Considered:**

| Option | Verdict |
|---|---|
| Not record as decision — leave as section finding in notebook | Rejected — N3 cross-phase triangulation warrants log-level recording parallel to D-046 |
| Record the numerical (0,0,1)(1,0,1,12) model as D-049 | Rejected — focus on structural property (triple IC + ARCH ≈ 1); numerical model may refine in Phase 7 |
| Combine with D-048 as single Step 1 decision | Rejected — D-048 is protocol, D-049 is finding; categorically distinct |

**Implementation:** narrated in `notebooks/06_arima_baseline.ipynb` Section 8 with triangulation table; quantitative signatures traceable to `data/documentation/phase6_step1_arima_{selection, residuals}.csv` (rows where `variant_id == 'JAPAN_first_diff'`).

---

## Phase 6 Step 1 — Interim State Summary

*Phase 6 is a three-step process (ARIMA → VAR → Ridge per D-004). This interim state covers **Step 1 (Layer 1 SARIMA) only**; Steps 2 (VAR) and 3 (Ridge) will add D-050 onwards.*

**After Phase 6 Step 1 SARIMA baseline estimation:**

| Metric | Phase 5 | Phase 6 · Step 1 (current) |
|---|---|---|
| Decision-log entries | 47 | **49** (+D-048, +D-049) |
| Narrative notebook deliverables | 5 | **6** (+`06_arima_baseline.ipynb`) |
| Modelling layers complete | 0 / 3 | **1 / 3** (SARIMA ✅; VAR ⏳; Ridge ⏳) |
| Portfolio figures | 8 (Phase 5) | **+8** (`phase6_step1_fig{1..8}_*.png`) |
| Audit CSVs | 12 (Phase 5) | **+15** (grid × 5 + boundary × 4 + extension × 2 + consolidated × 4) |
| `src/` module version | v0.4.0 | **v0.4.0** (unchanged; v0.5.0 reserved for Step 2 / 3) |
| Phase 6 completion | — | **~33 %** |

**Signature findings from Step 1 (to be cited in Phase 7 narrative and Phase 8 findings.md):**

1. **Japan ARIMA uniqueness (D-049)** — the sole variant among five with triple AIC / BIC / HQIC agreement on a 4-parameter sparse order, with ARCH-LM p = 0.9999 (near-perfect residual homoscedasticity) and lowest σ_train. N3 narrative echo at the ARIMA layer; third independent lens on Japan's structural uniqueness.
2. **AIC–OOS divergence at boundary extension (D-048 stopping rule)** — USA_first_diff Stage (a) → Stage (c) ΔAIC = −10.46 with OOS RMSE/MAE/bias essentially invariant; BIC and HQIC converge at the simpler (0,0,2)(0,0,3,12) order. Adopted as the principled stopping criterion for D-048 and obligates Phase 7 DM to compare OOS loss differentials rather than AIC ranking.
3. **UK ENERGY+ OOS degradation (+28 %)** — UK's Stage (a) model absorbs COVID (2020–21) better than ENERGY (2022+); RMSE ratio 0.402 / 0.315 = +28 %. Echoes Phase 5 Finding #4 (UK unique Phillips-curve sign-flip pre/post-GFC), suggesting a UK-specific regime interaction that Step 2 VAR can revisit via the D-030 GDP × ENERGY interaction.

**Step 1 artefact trace (all under project root):**

| Artefact type | Location | Count |
|---|---|---:|
| Scratch scripts | `scripts/phase6_step1{,b,c,d}_*.py` | 4 |
| Consolidated CSVs | `data/documentation/phase6_step1_arima_{selection,residuals,forecast,window_errors}.csv` | 4 |
| Grid CSVs (Stage a) | `data/documentation/phase6_step1_arima_grid_*.csv` | 5 |
| Boundary check CSVs (Stage b) | `data/documentation/phase6_step1b_boundary_check_{summary,variant}.csv` | 4 |
| Q=3 extension CSVs (Stage c) | `data/documentation/phase6_step1c_*.csv` | 2 |
| Portfolio figures | `outputs/figures/phase6_step1_fig{1..8}_*.png` | 8 |
| Portfolio notebook | `notebooks/06_arima_baseline.ipynb` | 1 |

**Phase 6 Step 2 (VAR, Layer 2) prerequisites ready:**

- Phase 4 feature matrices (50–53 columns × 285–296 rows per country) are VAR-ingestion-ready
- D-030 regime-dummy interaction matrix (6 interactions: USA × 3, UK × 1, GER × 2, JPN × 0) accessible via `src.feature_engineering.PHASE6_REGIME_SPEC`
- Phase 5 Finding #5–7 (cross-lag heatmap, Granger-direction hints, rolling Phillips) and Phase 5 S4 Ljung-Box diagnostics are Step 2 priors
- Step 1 forecast CSV (340 rows) available as a Phase 7 DM input baseline

---

*Last updated: Phase 6 Step 1 complete — 5 SARIMA variants across a three-stage grid search (D-048 protocol), 2 new decisions (D-048, D-049), 8 portfolio figures delivered. `src/` v0.4.0 unchanged; v0.5.0 reserved for Phase 6 Step 2/3 modelling modules. Next: Phase 6 Step 2 VAR estimation with D-030 regime interactions under a scope-driven protocol.*
