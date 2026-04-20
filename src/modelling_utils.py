"""
src.modelling_utils
===================
Shared utilities for Phase 6 modelling layers (VAR · Ridge · future).

Introduced at v0.4.1 per D-063 after Phase 6 Step 2 revealed 6× duplication
of ``build_exog_column_list()`` and 5× duplication of the ``P_PER_COUNTRY``
lag-order dict across nine scratch scripts. The D-047 "scratch-only"
principle was designed for single-use exploratory code; once a helper is
copied into 4+ scripts its single-use status has been empirically
falsified and promotion to ``src/`` improves both correctness (single
source of truth) and readability (intent visible via import).

Scope (deliberately narrow):
    * Cross-script constants used by 4+ scripts (lag orders, Cholesky
      ordering, structural-break / period dummy keys).
    * Pure helper functions with no fitting logic (exog column assembly,
      endog / exog block extraction).

Deliberately excluded:
    * ``VAR(...).fit()`` / ``Ridge.fit()`` / any model-fitting call.
    * IRF / FEVD / Granger post-processing.
    * Walk-forward / expanding-window refit orchestration.

These remain in scratch scripts until Phase 6 Step 3 (Ridge) reveals
whether a stable model-fitting API is worth promoting. If so, a future
v0.5.0 ``src/modelling.py`` can absorb them at Phase 6 closure — at
which point ``modelling_utils`` may also be folded in.

Decision linkage
----------------
D-027 / D-029 / D-030  Structural-break / period / regime-interaction spec.
D-050                 VAR lag selection protocol (BIC→AIC revision).
D-054                 Cholesky ordering [GDP, UE, CPI, PR, M2].
D-063                 This module (src/ promotion rationale).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from .structural_breaks import KNOWN_BREAKS
from .feature_engineering import PHASE6_REGIME_SPEC


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

#: VAR lag orders per D-050 (AIC-primary, revised from BIC).
#: Inferential primary: Granger / IRF / FEVD use these.
P_PER_COUNTRY_AIC: dict[str, int] = {
    'USA':     12,
    'JAPAN':    5,
    'UK':      12,
    'GERMANY': 12,
}

#: BIC parsimony reference per D-050 — retained for Phase 7 Diebold-
#: Mariano benchmark (forecast-primary, not inference-primary).
P_PER_COUNTRY_BIC: dict[str, int] = {c: 2 for c in P_PER_COUNTRY_AIC}

#: Cholesky ordering for orthogonalised IRF / FEVD (D-054).
#: Slow-to-fast macroeconomic predetermination: GDP → UE → CPI → PR → M2.
#: Rationale: real-economy variables predetermined within the month,
#: inflation responds to slack, policy reacts to observed π and y,
#: money supply endogenously adjusts last.
CHOLESKY_ORDER: list[str] = [
    'GDP', 'UNEMPLOYMENT', 'CPI', 'POLICY_RATE', 'M2',
]

#: Structural-break dummy name stems (D-027 / D-029).
#: Full column names: ``{country}_D_{break}`` e.g. 'USA_D_GFC_2008'.
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())

#: Period-regime dummy keys (D-029 four-phase structure).
#: Full column names: ``{country}_P_{key}`` e.g. 'USA_P_COVID'.
#: NB the four phases (pre-GFC / GFC / COVID / Energy2022) are encoded
#: via two dummies — the baseline (pre-GFC) is the omitted category.
PERIOD_KEYS: list[str] = ['GFC', 'COVID']


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def build_regime_exog_columns(country: str,
                              features_cols: list[str]) -> list[str]:
    """Compose the D-030 regime-interaction exog column list for a country.

    Returns the ordered list of column names to select as the ``exog``
    block in VAR / Ridge fitting. The list concatenates three groups in
    deterministic order:

    1. **Split dummies** — one per structural break in :data:`SPLIT_BREAK_NAMES`,
       with prefix ``{country}_D_`` (e.g. ``USA_D_GFC_2008``).
    2. **Period dummies** — one per key in :data:`PERIOD_KEYS`, with prefix
       ``{country}_P_`` (e.g. ``USA_P_COVID``). Encodes the 4-phase regime
       structure using 2 dummies + omitted baseline (D-029).
    3. **Regime-interaction columns** — per :data:`PHASE6_REGIME_SPEC`,
       one column per (country, break) where a non-trivial dominant
       driver was identified in D-030. Format:
       ``{country}_{driver}_x_D_{break_name}``.

    Parameters
    ----------
    country : str
        One of 'USA', 'JAPAN', 'UK', 'GERMANY'.
    features_cols : list[str]
        Columns available in the country's feature matrix — used to verify
        every computed exog column is present.

    Returns
    -------
    list[str]
        Ordered exog column names.

    Raises
    ------
    KeyError
        If any composed column is missing from ``features_cols``.

    Notes
    -----
    Previously duplicated 6× across Phase 6 Step 2 scripts
    (S2 / S2b / S3 / S4 / S5 / S6). Promoted to ``src/`` per D-063.
    """
    split_cols  = [f'{country}_D_{b}' for b in SPLIT_BREAK_NAMES]
    period_cols = [f'{country}_P_{p}' for p in PERIOD_KEYS]

    interaction_cols: list[str] = []
    for (c, break_name), driver in PHASE6_REGIME_SPEC.items():
        if c != country:
            continue
        if driver is None or driver == 'const':
            continue
        interaction_cols.append(f'{country}_{driver}_x_D_{break_name}')

    all_cols = split_cols + period_cols + interaction_cols
    missing = [c for c in all_cols if c not in features_cols]
    if missing:
        raise KeyError(
            f"{country}: missing exog columns in feature matrix: {missing}"
        )
    return all_cols


def extract_endog_exog_cholesky(
    features_df: 'pd.DataFrame',
    country: str,
) -> tuple['pd.DataFrame', 'pd.DataFrame']:
    """Extract Cholesky-ordered endog and D-030 exog blocks for VAR fitting.

    Returns the country's five core stationary indicators reordered per
    :data:`CHOLESKY_ORDER` (D-054) alongside the D-030 regime-interaction
    exog block, both restricted to the joint-valid window via
    ``pd.DataFrame.dropna(how='any')``.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Country-specific feature matrix — typically from
        ``src.build_all_features()[country]``. Must contain the five
        Cholesky endog columns (``{country}_{indicator}``) and all D-030
        exog columns per :func:`build_regime_exog_columns`.
    country : str
        One of 'USA', 'JAPAN', 'UK', 'GERMANY'.

    Returns
    -------
    endog : pandas.DataFrame
        Shape ``(n_obs, 5)``. Columns ordered per :data:`CHOLESKY_ORDER`.
    exog : pandas.DataFrame
        Shape ``(n_obs, n_exog)``. D-030 regime-interaction block.

    Raises
    ------
    ValueError
        If the joint-valid block is empty after dropna.
    KeyError
        If any expected column is missing (propagates from
        :func:`build_regime_exog_columns`).

    Notes
    -----
    Previously duplicated 4× across Phase 6 Step 2 scripts
    (S4 / S5 / S6 / S6b). Promoted to ``src/`` per D-063.
    """
    endog_cols = [f'{country}_{ind}' for ind in CHOLESKY_ORDER]
    exog_cols = build_regime_exog_columns(country, list(features_df.columns))
    joint = features_df[endog_cols + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    return joint[endog_cols].copy(), joint[exog_cols].copy()


__all__ = [
    # Constants
    'P_PER_COUNTRY_AIC',
    'P_PER_COUNTRY_BIC',
    'CHOLESKY_ORDER',
    'SPLIT_BREAK_NAMES',
    'PERIOD_KEYS',
    # Helpers
    'build_regime_exog_columns',
    'extract_endog_exog_cholesky',
]
