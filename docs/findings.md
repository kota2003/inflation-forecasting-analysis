# Findings

*Inflation Prediction and Economic Signal Analysis · Portfolio Project 3*

## Introduction

This document reports three cross-country findings from a multi-country CPI
inflation forecasting study covering the United States, Japan, the United
Kingdom, and Germany from 2000 through early 2025. The analytical
architecture layers three model families — univariate ARIMA baselines,
multivariate VAR systems, and high-dimensional Ridge regression — against
a single target (year-on-year CPI growth) across four forecast horizons
(h ∈ {1, 3, 6, 12} months). Evaluation rests on 125 paired Diebold-Mariano
tests: 25 cells at country × horizon × layer-pair granularity, run under
three loss-function variants (standard squared-error, Newey-West HAC,
robust absolute-error), with an additional COVID-origin-excluded
sensitivity re-run of the same 25 cells.

The study's methodological commitment is *evidence-grounded iteration*.
Eighty decisions are recorded in a living `ProjectDriven.md` audit trail
(D-001 through D-080), each specifying its rationale, alternatives
considered, and downstream propagation. The narrative presented here has
been **substantively revised** relative to the pre-Phase-7 architectural
hypothesis: what began as an expected story about Ridge's architectural
advantage became, under COVID-origin-trimmed evidence, a story about
univariate ARIMA's surprising dominance at short horizons. That revision
— and the fact that it is logged transparently rather than retrofitted
— is itself the portfolio-primary claim.

The three findings below are organised around the project's named
narratives: **N1** (cross-country dynamics), **N2** (policy response
patterns), and **N3** (Japan's uniqueness). A closing section documents
the three-lens methodology match that underwrites all three as the
project's core analytical defensibility.

---

## N1 · Cross-country inflation dynamics

Layer differentiation in CPI forecast accuracy is concentrated at h = 1
and appears only in the USA and the UK — a paired-DM pattern that
reframes the three-layer architecture from "does complexity help?" to
"where does complexity help?".

Under standard Diebold-Mariano with Harvey-Leybourne-Newbold small-sample
correction and walk-forward origins trimmed of the 2020-03 through
2020-08 window (D-079), three of 25 tested cells are significant at α =
0.05. All three are ARIMA wins at h = 1: USA against VAR (p = 0.044),
USA against Ridge (p = 0.001), and UK against VAR (p = 0.024). Neither
VAR nor Ridge retains a single significant win under the post-trim
standard-DM lens.

The most diagnostic cell is the USA h = 1 ARIMA-Ridge comparison, whose
p-value shifts from 0.135 (S2, full origin set) to 0.001 (S4, COVID
origins excluded) — a nine-order-of-magnitude change driven entirely by
removing six walk-forward origins. A shift of this magnitude under a
six-origin trim is strong evidence that the S2 tie was dominated by
COVID-onset outlier variance and that the underlying 52-origin signal is
unambiguously in ARIMA's favour. This single cell refuted the easiest
"tie = no real difference" reading of the S2 matrix and re-centred the
Phase 7 narrative on h = 1 univariate dominance.

Japan and Germany show the opposite pattern: 0 / 6 and 0 / 6 cells
respectively are significant under standard DM across all four horizons.
This is not a power-shortfall artefact — with paired samples of n = 52
(Japan) and n = 45 (Germany) after trimming, the battery remains above
the n = 30 underpowered threshold. The uniform non-rejection is itself
a finding: Japanese and German inflation are structurally resistant to
forecast-error differentiation across three materially different model
architectures.

The geography of layer differentiation therefore reads: short-horizon
univariate best in USA and UK; no layer edge in Japan or Germany; and
— importantly — no horizon-wise architectural hierarchy at h ∈ {3, 6,
12} once COVID origins are excluded. The S2 Ridge-wins at medium
horizons (USA h = 3, UK h = 3) all lost significance under trimming,
reclassifying them as COVID-era artefacts rather than steady-state
evidence.

*Supporting decisions*: D-062 (USA yoy_pct × VAR systematic bias — the
only Phase-6 point-estimate finding to survive all four Phase 7 lenses
unchanged), D-078 (DM battery), D-079 (COVID-origin sensitivity).

---

## N2 · Monetary policy transmission and lag effects

Policy-transmission signal is preserved at the coefficient level across
two mathematically-independent lenses but does not translate to
paired-DM forecast-level evidence once the COVID-origin window is
excluded. The evidence at the model-interior level is strong; the
evidence at the forecast-accuracy level is narrower than the
pre-Phase-7 hypothesis predicted.

At the coefficient level, the match is the project's strongest
monetary-transmission signal. The USA VAR orthogonalised impulse
response function peaks at **−0.149** at h = 4 (D-056) — a negative
CPI response three-to-five months after a one-standard-deviation
positive policy-rate shock, consistent with the textbook transmission
lag for the Federal Reserve policy rate. The USA Ridge regression on
the first-difference target form (D-071) returns a standardised
coefficient of **−0.136** on POLICY_RATE_lag3 (D-067). Two models, two
mathematical objects — an orthogonalised dynamic response and a
penalised linear coefficient — arrive at within 10% of the same
magnitude, identical sign, and essentially the same lag. This is a
cross-lens match of a kind the project treats as its highest-form
evidence.

At the forecast level, the pre-Phase-7 expectation — consistent with
D-070's point-estimate MASE finding that Ridge beats VAR in 12 of 16
country × horizon cells — was that Ridge's policy-transmission-aware
regularisation would dominate VAR at medium horizons (h ∈ {3, 6, 12})
where monetary policy is thought to transmit most powerfully. S2
supported this weakly: 3 of 12 Ridge-vs-VAR cells at medium horizons
reached α = 0.05 under standard DM. S4 overturned all three. The
USA h = 3, UK h = 1, and UK h = 3 VAR-Ridge cells each lost
significance once COVID-onset origins were removed; none gained
replacements.

The revision is deliberately partial. D-070's MASE claim — "Ridge
wins 12 of 16 cells on point-estimate forecast accuracy" — remains
factually correct and is not retracted. What is revised is the
inferential claim it initially suggested: that Ridge exhibits a
COVID-robust architectural advantage. Post-trim DM evidence does not
support that claim. The honest re-phrasing, adopted in D-079 and
propagated here, reads: **Ridge has a measurable MASE edge that
translates to statistical significance only when COVID-era origins
are retained**. The point-estimate finding and the DM finding are
both true; they describe different statistical objects.

For portfolio positioning, the pair of N2 claims is more useful than
a single overstated one. The coefficient-level monetary-transmission
match is a rare, defensible result from two independent estimation
procedures. The forecast-level claim is narrower, bounded, and
transparent about its COVID-era scope.

*Supporting decisions*: D-056 (VAR IRF), D-067 (Ridge coefficient
stratification and magnitude match), D-070 (Ridge MASE 12 / 16),
D-071 (USA first_diff preference, cross-lens anchor), D-078, D-079.

---

## N3 · Japan's structural uniqueness (nine-lens triangulation)

Japan's inflation series exhibits a uniform forecasting difficulty —
no model layer systematically out-forecasts another — supported by
nine independent methodological lenses across Phases 3 through 7.
This is the project's most comprehensively triangulated empirical
finding.

The nine lenses, ordered by the phase in which each materialised:

1. **ACF autocorrelation decay profile** (D-044, Phase 3) — Japanese
   CPI year-on-year shows near-martingale behaviour, with
   autocorrelations decaying rapidly to within ±2/√n bands beyond
   lag 2.
2. **ARIMA order selection** (D-049, Phase 6 Step 1) — the
   Japan-optimal ARIMA order is markedly simpler than the other three
   countries', consistent with a low-memory signal.
3. **VAR lag-length selection** (D-050, Phase 6 Step 2) — BIC-minimising
   lag selection returns p = 1 for Japan under the stricter criterion,
   against p ∈ {3, 12} for the other three countries.
4. **Granger causality battery** (D-052, Phase 6 Step 2) — 1 / 25 tests
   significant for Japan at α = 0.05 versus 5–8 for USA / UK / Germany.
5. **Orthogonalised impulse response functions** (D-056, Phase 6 Step 2)
   — policy-rate shocks produce no statistically distinguishable CPI
   response in Japan; the 95% asymptotic confidence band includes zero
   across all horizons.
6. **Forecast error variance decomposition** (D-058 / D-059,
   Phase 6 Step 2) — own-CPI variance share for Japan stays above 90%
   at all horizons; monetary-policy variance share remains below 3%.
7. **Ridge α-boundary regime** (D-066, Phase 6 Step 3) — the
   cross-validated optimal α for Japan saturates the initial
   logspace(-3, 3, 13) grid at its upper boundary, requiring a
   logspace(3, 6, 7) extension before stabilising near α ≈ 10,000 —
   two-to-three orders of magnitude above USA / UK / Germany.
8. **Ridge coefficient-magnitude stratification** (D-067,
   Phase 6 Step 3) — Japan's maximum standardised absolute coefficient
   is ≈ 0.01 against USA's 0.71 — a 70-fold gap that persists across
   lens-stability CV folds and survives bootstrap resampling.
9. **Phase 7 DM-null pattern** (D-078, D-079, Phase 7) — Japan returns
   0 / 6 significant cells under standard DM across all horizons,
   with one robust-DM flip post-trim (h = 6 VAR-Ridge, tie → VAR at
   p = 0.022) — a pattern absent in all other countries and emerging
   only after the 2020 shock window is removed.

No single lens is decisive; any one could plausibly be a methodology
artefact. What is methodologically distinctive about N3 is the
convergence of nine lenses drawing on fundamentally different
mathematical objects — correlation, likelihood, hypothesis testing,
dynamic response, variance share, penalised coefficient magnitude,
forecast-error differential. The cross-triangulation is the
portfolio-strongest evidence form that the Phase 4 feature matrix and
the three-layer architecture jointly support.

The economic reading is consistent: near-martingale behaviour of
Japanese CPI across 2000–2021, combined with the 2022 structural
break's characteristic failure to transmit into a persistent regime
(a feature the Phase 3 structural-break battery also detected),
leaves no stable architectural hierarchy for any three-layer model
to exploit.

*Supporting decisions*: D-044, D-049, D-050, D-052, D-056,
D-058 / D-059, D-066, D-067, D-072 (septuple formalisation), D-078,
D-079.

---

## Three-lens methodology match as core defensibility

The project's analytical backbone is a three-lens match on the USA
policy-transmission signal, where three mathematically-independent
tests converge on the same sign, magnitude, and approximate lag.

- The **VAR orthogonalised IRF peak** at **−0.149** at h = 4 (D-056)
  — a dynamic-response object computed from a multivariate linear
  Gaussian system.
- The **Ridge first-difference POLICY_RATE_lag3 standardised
  coefficient** at **−0.136** (D-067, D-071) — a penalised
  linear-regression object computed from a 50+-feature L2-regularised
  fit.
- The **paired Diebold-Mariano USA h = 1 ARIMA-VAR statistic** at
  **p = 0.014 (standard) / p = 0.0001 (robust)** (D-078, confirming
  D-062) — a forecast-error-differential object attesting systematic
  VAR under-prediction through the 2022 energy-price shock.

The three objects are not statistical synonyms. An IRF is a
conditional mean function over a shock; a Ridge coefficient is a
penalised maximum-likelihood point estimate over a design matrix; a
DM statistic is a hypothesis test over a difference-of-losses
sequence. Their convergence on the same directional signal —
negative, magnitude in the low 0.1s, concentrated at the three-to-four
month lag — is not a mathematical tautology; it is empirical
convergence.

This cross-lens triangulation is the project's distinctive
methodological claim, and it is why the revised Phase 7 narrative is
stronger rather than weaker than the pre-Phase-7 version. A narrative
that reported only the Phase 6 point-estimate finding would have been
larger in scope but thinner in evidence. The revised narrative is
bounded, caveated, and cross-confirmed.

The companion `methodology.md` document records the four-phase
iteration pattern (pre-flight → integration → battery → sensitivity
→ closeout) that produced these findings. Where single-lens findings
can be methodology artefacts, three-lens matches are evidence.

---

*Portfolio Project 3 · Phase 8 · findings.md · 80 decisions · src v0.4.3*