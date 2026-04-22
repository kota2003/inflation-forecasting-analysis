# Methodology

*Inflation Prediction and Economic Signal Analysis · Portfolio Project 3*

## Introduction — Evidence-grounded iteration as template

This document records the methodological template that produced the
findings in `findings.md`: an **evidence-grounded iteration pattern** in
which every analytical decision is pre-committed with alternatives
considered, executed with an audit artefact, and revised in place when
subsequent evidence warrants — without retroactive erasure. The pattern
sits deliberately between pure exploratory analysis (which leaves no
defensible trail) and rigid pre-registration (which cannot accommodate
what the data teach). It aims to make revision cheap and misreading
expensive.

Quantitatively, the template produced **80 decisions** in a living
`ProjectDriven.md` audit log (D-001 through D-080), **125 paired
Diebold-Mariano computations** across Phase 7, **5 of 8 planned `src/`
modules** materialised under an empirical promotion rule, and **25
audit CSVs** preserving intermediate results at every non-trivial
inflection point. One decision slot (D-020) is marked vacant as an
artefact of an earlier consolidation; three decisions (D-006, D-073,
D-081) are amended in place with the original text preserved.

The sections below document seven elements of the template: the
four-phase iteration pattern that structures each phase, the
`ProjectDriven.md` decision-log philosophy, the evidence-grounded
`src/` promotion discipline, the multi-source data strategy established
in Phase 1, the cross-lens match methodology that underwrites the core
findings, and — as a case study — the honest-revision episode in which
Phase 7 evidence re-characterised a Phase 6 headline claim. A closing
section argues for the template's portability.

---

## The four-phase iteration pattern

Every non-trivial phase of this project — most conspicuously Phases 6
and 7 — decomposed into the same five-sub-step structure: **pre-flight,
integration, battery, sensitivity, closeout**. The pattern is not
methodological ornamentation; each sub-step answers a distinct question
that the next sub-step cannot answer on its behalf. Pre-flight asks "is
the environment ready?"; integration asks "is the data assembly
correct?"; battery asks "what does the primary test say?"; sensitivity
asks "how fragile is that?"; closeout asks "what, taken together, do
we now claim?".

**Phase 6 Step 2 (VAR)** is the pattern's clearest early expression.
Nine sub-steps (S1 / S1b / S2 / S2b / S3 / S4 / S5 / S6 / S6b) map to
the five phases as follows. S1 and S1b constitute the pre-flight: lag
selection under BIC with a B&A-threshold sensitivity extension,
producing D-050's preliminary criterion choice. S2 and S2b are
integration under a methodology revision — S2's BIC-fitted VAR failed
Ljung-Box whiteness diagnostics at the 55% pass rate, triggering S2b's
AIC refit and the D-051 partial-whitening caveat. S3 (Granger causality
battery, D-052) and S4 (orthogonalised IRFs with asymptotic 95%
confidence bands, D-056) constitute the primary-inferential battery. S5
(FEVD, D-058 and D-059) and S6 (walk-forward OOS forecasts with the
D-061 COVID-onset-window pre-flag) execute the forecast battery. S6b
and the closeout then produced D-062 (USA yoy_pct × VAR systematic
bias) and D-063 (the `src/modelling_utils` v0.4.1 promotion triggered
by accumulated 4× duplication across Step 2 scratch scripts).

**Phase 7** is the pattern's most compressed expression. What the
original plan budgeted as seven sub-steps collapsed to five when
empirical evidence during S2 made two originally-separate sub-steps
redundant. The executed structure is: pre-flight (D-076, materialising
`src/evaluation.py` at v0.4.3); integration (D-077, unifying 336 ARIMA
+ 872 VAR + 1,104 Ridge walk-forward forecast rows into a 2,312-row
panel with a DM-cell-centric coverage manifest); battery (D-078,
executing all three DM variants over 25 cells = 75 computations);
sensitivity (D-079, re-running the same 25 cells with 2020-03 through
2020-08 origins excluded per D-061's pre-flagged COVID-onset window);
and closeout (D-080, aggregating S1 + S2 + S4 into the N1 / N2 / N3
revised verdicts and deferring the `src/models/` Tranche 2 promotion
under D-075's evidence rule).

The compression is itself a methodology-quality signal. Phase 7's
S2-variant HAC computation was originally scheduled as its own S2b
sub-step with its own decision (D-079-original); the three-variant DM
execution in S2 consumed HAC at zero marginal cost and returned
zero winner-flips against standard DM, closing D-051 in place and
making a separate S2b sub-step redundant. Similarly, Phase 7's
originally-planned USA dual-form S3 sub-step was absorbed when the D-071
resolution plus Ridge first_diff's inclusion in S2 made separate
execution unnecessary. The audit trail reflects the compression with
D-078 absorbing the two would-have-been-separate decisions; the
decision-number sequence (D-076..D-080) therefore contains no vacancies.

The five-sub-step structure is not a recipe. Its use-value is that it
forces each sub-step to produce an audit CSV before the next begins,
which is what makes a later revision affordable.

---

## The ProjectDriven.md decision-log philosophy

Every decision is recorded in `ProjectDriven.md` at the moment it is
committed, with a fixed four-part structure: **decision statement,
rationale, alternatives considered, and implementation / propagation
trace**. The structure is not aesthetic — each part disciplines a
different failure mode. The decision statement prevents later narrative
drift; the rationale prevents retroactive justification; the
alternatives-considered table prevents the appearance of inevitability
(decisions without rejected alternatives read as foregone conclusions,
which they almost never are); the propagation trace prevents silent
cross-file inconsistency.

The log observes two conventions that distinguish it from a commit
history. First, **amendment in place**: when new evidence alters a
decision, the amendment is logged within the original decision's entry
with an explicit date and trigger, rather than deleting and rewriting.
D-006 was amended during Phase 1 when the manual-CSV integration path
for Japan CPI was added; D-073 was amended at Phase 6 Step 3 closeout
when Kota elected to execute the `src/modelling_utils` v0.4.2 patch
immediately rather than defer it. The amendment preserves the original
decision's text for audit; the revision is a diff, not a replacement.

Second, **vacancy preservation**: when a planned decision slot is not
needed (because evidence during execution made it redundant or
absorbed it into a neighbouring decision), the slot is marked vacant
rather than renumbered. D-020 is the project's sole such vacancy — an
artefact of an earlier sub-step consolidation. Vacancy preservation
costs nothing but preserves 1-to-1 correspondence between the decision
sequence and the executed history, which is what enables a portfolio
reader to audit evidence-to-conclusion alignment without access to
the analyst's working memory.

What the log deliberately avoids is retrospective narrative
construction — writing decisions after the fact to fit a chosen story.
The Phase 7 revision case (next section) is the cleanest demonstration:
D-070's Ridge 12/16 MASE claim remained unedited even after D-079
reframed its portfolio interpretation, because D-070's factual content
was unchanged.

---

## Evidence-grounded `src/` promotion

Reusable code promotion from notebooks into `src/` follows a strict
**four-times-duplication threshold** recorded in D-063. No module is
created speculatively; every extraction consolidates at least four
independent call sites. The threshold is not a heuristic — it is an
empirical rule that protects against two common failure modes: the
premature-abstraction tax (creating wrappers nobody calls) and the
under-refactoring tax (copy-pasted helpers drifting out of sync).

The current `src/` state implements five of the eight modules named
in the ProjectScope §12 blueprint: `data_loader`, `preprocessing`,
`stationarity` (with `structural_breaks` split per D-032),
`feature_engineering`, and two Phase 6+ promotions — `modelling_utils`
at v0.4.2 (D-074, thirteen exports) and `evaluation` at v0.4.3 (D-076,
ten exports). The remaining three modules in the ProjectScope §12
blueprint — `src/models/{arima_model, var_model, ridge_model}.py` —
are **deferred**. This is not incidental slippage; it is explicit
D-075 architecture and D-080 re-assessment recording the decision.

The re-assessment logic is the template's didactic moment. D-075
committed that `src/models/` promotion would happen if empirical
evidence warranted and not otherwise. At Phase 7 closeout, five Phase
7 scripts ran consuming `src.evaluation` primitives without re-fitting
any ARIMA / VAR / Ridge model; zero refit call sites across five
scripts against a four-times-duplication threshold is decisive evidence
that the blueprint entry is not yet warranted. D-080 therefore defers
Tranche 2 again, with a revised rationale: the project has empirically
converged on a "model fits live in notebooks 06–08; evaluation lives
in notebook 09 plus `src.evaluation`" pattern, and forcing an
`src/models/` subdirectory would require model-wrapper classes that
no current caller requests.

The portfolio consequence is that every file in `src/` has active
callers. A `src/` directory with unused abstractions is expensive; a
`src/` directory that is three modules shy of the aspirational
blueprint with three logged deferrals is inexpensive. The preserved
ProjectScope §12 slot is available for future project iterations that
may introduce additional model families or cross-project re-use.

---

## Multi-source data strategy

Phase 1 established a precedent that **file-existence checks are
insufficient — content-level validation is mandatory** — and that
targeted single-series replacements preserve audit integrity better
than wholesale re-acquisition.

The precedent was set by the Japan CPI case. The FRED and OECD
derivative series for Japanese CPI year-on-year appeared complete:
file present, column headers correct, date range spanning 2000–2024.
Only content-level validation — checking for trailing NaN after the
nominal end-of-series — revealed that the upstream feed had stopped
updating mid-2021. A pipeline that tested only for file existence
would have accepted the series and invalidated the N3 narrative
entirely, since the 2022 inflation reversal would have been
unobservable. The D-013 remediation was targeted: only the Japan CPI
series was replaced, via a manual CSV download from the Japan
Statistics Bureau in Shift-JIS encoding with Japanese-format date
headers. The other 24 series were untouched, preserving the
`data_loader` unit-test baseline and the audit trail of the first 24
series' provenance.

The four-path architecture this established — FRED API, World Bank
API, Japan Statistics Bureau manual CSV, IMF — is portable. Its key
property is that failure of any one path degrades gracefully: the
others remain authoritative for their assigned series, and the
replacement path requires only a targeted override in the loader, not
a pipeline rebuild.

Two secondary conventions emerged from Phase 1 that the template
retains. **Economic-semantics overrides** are logged as explicit
judgment calls — Japan's policy rate is the call money rate (not the
ten-year JGB yield); UK GDP is the real series (not nominal) —
because the alternative (silent semantic substitution) corrupts
cross-country comparability without raising a diagnostic flag.
**Retry logic with exponential backoff** resolved World Bank API
timeouts that initially blocked China unemployment data; the
investment is one-off but the payoff scales across projects.

---

## Cross-lens match methodology

The project's analytical backbone is a **cross-lens match
methodology**: a finding achieves portfolio-primary status only when
two or more mathematically-independent tests converge on the same
sign, magnitude, and lag structure. The single-lens-promoted-to-claim
failure mode is the one this methodology exists to prevent.

The canonical three-lens match, reprised from `findings.md § 5`, is
the USA monetary-transmission signal. The VAR orthogonalised impulse
response function peaks at −0.149 at h = 4 (D-056) — a dynamic
response object. The Ridge first-difference POLICY_RATE_lag3
standardised coefficient is −0.136 (D-067, D-071) — a penalised
linear-regression object. The paired Diebold-Mariano USA h = 1
ARIMA-VAR statistic is p = 0.014 standard and p = 0.0001 robust
(D-078, confirming D-062) — a forecast-error-differential object.
Three objects from three mathematical families (multivariate Gaussian
system, L2-regularised OLS, hypothesis test) converging on the same
directional signal cannot be a statistical tautology.

The methodology's maximal application is N3, where nine independent
lenses across Phases 3–7 converge on Japan's uniform forecasting
difficulty. No single N3 lens is decisive — ACF could be
methodology artefact, Granger could be power-limited, FEVD could
reflect ordering choice, Ridge magnitude could be feature-scale
artefact, the DM-null pattern could be sample-size-limited. The
convergence of all nine is what the cross-triangulation methodology
treats as evidence of architectural fact rather than estimation
artefact.

The counter-pattern the methodology explicitly avoids is promoting a
single-lens finding to a portfolio-primary claim. D-070's Ridge 12/16
MASE win, read in isolation, would have been such a promotion. The
Phase 7 evidence check — promoted to primary status precisely because
the cross-lens methodology required it — reframed D-070 as narrower
than its initial apparent scope, without retraction.

---

## Honest revision — the D-070 caveat episode

Phase 7's most instructive methodological moment was not a new
finding but a **structured revision** of a pre-existing one: D-079
recast D-070's Ridge MASE dominance as COVID-era-specific, without
retracting the underlying point-estimate claim.

The pre-Phase-7 narrative state was that D-070 (12 / 16 Ridge MASE
wins against VAR) was the natural Phase 6 headline; the architectural
hypothesis, consistent with D-071, was that Ridge dominated at the
forecast level. Phase 7 S4 evidence overturned the inferential
extension of this claim. Three of three S2-significant Ridge wins
(USA h = 3, UK h = 1, UK h = 3) lost Diebold-Mariano significance
under COVID-origin trimming; concurrently, three ARIMA h = 1 cells
strengthened or gained significance. The directional asymmetry (four
cells lost significance, three gained) is what ruled out a pure
noise-reduction interpretation of the trim and made the revision
substantive.

D-079's revision logic is deliberately surgical. D-070's MASE claim
is a point-estimate statement over 16 cells and remains factually
correct — it is not retracted. What is revised is the inferential
extension: that Ridge exhibits a COVID-robust architectural
advantage. Post-trim DM evidence does not support that extension.
The honest re-phrasing reads: "Ridge has a measurable MASE edge that
translates to statistical significance only when COVID-era origins
are retained." The MASE claim and the DM claim describe different
statistical objects; both are true.

The portfolio-strength argument for this revision is straightforward.
A portfolio that reports narrow, evidence-bounded conclusions with
logged revisions is strictly preferable, for technical-hiring
audiences, to one that over-claims and hopes the audit trail is not
examined. The template's expectation is that a D-079-class revision
will occur in most non-trivial phases, and the decision-log structure
is built to accommodate such revisions without retroactive erasure.

---

## Closing — Portability of the template

The methodology recorded here is not P3-specific. Each of its elements
— the five-sub-step phase pattern, the four-part decision-log
structure with amendment-in-place and vacancy preservation, the
four-times-duplication `src/` promotion threshold, the multi-source
data strategy with targeted remediation, the cross-lens match
requirement for portfolio-primary claims, and the honest-revision
convention — is portable to any multi-model empirical study.

The template's distinctive value is that it makes revision cheap and
misreading expensive. Revision is cheap because the decision log
preserves the original claim while recording the amendment; the
reader can trace what changed and why. Misreading is expensive
because no single-lens finding is promoted to primary status without
a cross-lens match, and every audit artefact (CSVs, decision log,
`src/` version tags) is explicit about its scope.

`findings.md` records the empirical findings that this methodology
produced. `ProjectDriven.md` records the full 80-decision audit
trail. This document exists as the bridge between them — the
template that can be re-applied to a different dataset, a different
question, and a different set of models, with the expectation that
the same discipline will produce a comparably defensible portfolio
artefact.

---

*Portfolio Project 3 · Phase 8 · methodology.md · 80 decisions · src v0.4.3*