"""
src.modelling_utils
===================
Shared utilities for Phase 6 modelling layers (VAR · Ridge · future).

Version history
---------------
v0.4.1 (D-063, Phase 6 Step 2 closeout) — initial module creation:
    6 Step-2-duplicated items promoted (2 constant dicts, 3 constant
    lists, 2 pure helper functions).
v0.4.2 (D-074, Phase 6 Step 3 closeout) — extension for Ridge layer:
    6 Step-3-duplicated items promoted (6 constants, 5 helper functions,
    1 cross-layer comparison dict). No API changes to v0.4.1 exports.

D-063 principle retained: "scratch-only" was designed for single-use
exploratory code; once a helper is copied into 4+ scripts its single-use
status has been empirically falsified and promotion to ``src/`` improves
both correctness (single source of truth) and readability (intent
visible via import).

Scope (still deliberately narrow):
    * Cross-script constants used by 4+ scripts (lag orders, Cholesky
      ordering, structural-break / period dummy keys, Phase 6 Step 3
      train-window / α-grid / horizon / category conventions).
    * Pure helper functions with no model-fitting call in their body
      (exog assembly, endog/exog extraction, USA dual-form construction,
      feature-category classification, α selection I/O, walk-forward
      origin construction, unfitted Pipeline factory).

Deliberately excluded (still):
    * ``VAR(...).fit()`` / ``Ridge.fit()`` — model-fitting calls
      remain in scratch scripts and notebooks.
    * IRF / FEVD / Granger post-processing.
    * Walk-forward refit loop orchestration (``compute_walk_forward_origins``
      returns the origin set only; the refit loop is caller-side).

Decision linkage
----------------
D-005                  Train / test split (2000-01 .. 2019-12 / 2020-01+).
D-027 / D-029 / D-030  Structural-break / period / regime-interaction spec.
D-031                  Per-series transform registry (USA CPI = yoy_pct
                       primary; first_diff secondary per D-048 / D-062).
D-050                  VAR lag selection protocol (BIC→AIC revision).
D-054                  Cholesky ordering [GDP, UE, CPI, PR, M2].
D-060                  VAR OOS MASE at AIC-selected p (hardcoded for
                       cross-layer comparison with Ridge D-070).
D-063                  Initial module creation at v0.4.1.
D-065                  α log-grid, TimeSeriesSplit(5), Pipeline leakage-guard.
D-067                  Feature category regex patterns.
D-068                  Walk-forward OOS origin construction.
D-074                  This extension at v0.4.2 (Step 3 closeout).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from sklearn.pipeline import Pipeline as _Pipeline

from .structural_breaks import KNOWN_BREAKS
from .feature_engineering import PHASE6_REGIME_SPEC


# ════════════════════════════════════════════════════════════════════
# v0.4.1 content (D-063) — UNCHANGED
# ════════════════════════════════════════════════════════════════════

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
CHOLESKY_ORDER: list[str] = [
    'GDP', 'UNEMPLOYMENT', 'CPI', 'POLICY_RATE', 'M2',
]

#: Structural-break dummy name stems (D-027 / D-029).
SPLIT_BREAK_NAMES: list[str] = list(KNOWN_BREAKS.keys())

#: Period-regime dummy keys (D-029 four-phase structure).
PERIOD_KEYS: list[str] = ['GFC', 'COVID']


def build_regime_exog_columns(country: str,
                              features_cols: list[str]) -> list[str]:
    """Compose the D-030 regime-interaction exog column list for a country.

    See v0.4.1 docstring (unchanged).
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
) -> 'tuple[pd.DataFrame, pd.DataFrame]':
    """Extract Cholesky-ordered endog and D-030 exog blocks.

    See v0.4.1 docstring (unchanged).
    """
    endog_cols = [f'{country}_{ind}' for ind in CHOLESKY_ORDER]
    exog_cols = build_regime_exog_columns(country, list(features_df.columns))
    joint = features_df[endog_cols + exog_cols].dropna(how='any')
    if joint.empty:
        raise ValueError(f"{country}: joint endog+exog block is empty")
    return joint[endog_cols].copy(), joint[exog_cols].copy()


# ════════════════════════════════════════════════════════════════════
# v0.4.2 additions (D-074) — Phase 6 Step 3 promotions
# ════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────
# Temporal conventions (D-005)
# ────────────────────────────────────────────────────────────────────

#: Last training observation (end of month 2019-12). Train window is
#: ``(−∞, TRAIN_END]``. Project-wide per D-005. Previously duplicated
#: across Phase 6 Step 3 scripts S1 / S2 / S2b / S3 / S4.
TRAIN_END = None  # set below after pandas import to avoid import cycle

#: First test observation (2020-01). Test window is ``[TEST_START, ∞)``.
#: Project-wide per D-005. Previously duplicated across S1 / S4.
TEST_START = None  # set below

# Deferred pandas Timestamp construction to module-body; importing
# pd at top causes a minor circular-import concern in rare test
# scenarios where src is imported before pandas is on sys.path.
import pandas as pd  # noqa: E402
TRAIN_END = pd.Timestamp("2019-12-01")
TEST_START = pd.Timestamp("2020-01-01")


#: Forecast horizons evaluated for Phase 7 Diebold-Mariano (D-060, D-068).
#: Shared by VAR S6, Ridge S4, and future Phase 7 tests.
HORIZONS_PHASE7: tuple[int, ...] = (1, 3, 6, 12)


# ────────────────────────────────────────────────────────────────────
# Ridge hyperparameter conventions (D-065)
# ────────────────────────────────────────────────────────────────────

#: Default α grid for Ridge penalised regression (D-065).
#: 13 log-spaced points spanning 6 orders of magnitude.
#: Japan's extended grid at S2b is handled separately in that script.
ALPHA_GRID_DEFAULT: np.ndarray = np.logspace(-3, 3, 13)

#: Default number of TimeSeriesSplit folds for walk-forward CV (D-065).
N_SPLITS_DEFAULT: int = 5

#: Default random_state for sklearn estimators across this module.
#: Ridge is deterministic in principle; passing a fixed random_state
#: aids downstream reproducibility claims.
RANDOM_STATE_DEFAULT: int = 0


# ────────────────────────────────────────────────────────────────────
# Feature category taxonomy (D-067)
# ────────────────────────────────────────────────────────────────────

#: Canonical order of Phase 4 feature categories. Used for category-
#: contribution tables, barplot orderings, and display consistency.
CATEGORY_ORDER: list[str] = [
    'base', 'lag', 'rolling', 'split', 'period', 'interaction',
]

# Compiled regex patterns for Phase 4 feature-name category matching.
# Applied in priority order in classify_feature_category():
#   interaction (contains _x_) BEFORE split (contains _D_BREAK)
#   because interaction columns often contain the split substring.
_RE_INTERACTION = re.compile(r'_x_')
_RE_SPLIT       = re.compile(r'_D_(GFC_2008|COVID_2020|ENERGY_2022)$')
_RE_PERIOD      = re.compile(r'_P_(GFC|COVID|ENERGY|2008|2020|2022)')
_RE_LAG         = re.compile(r'_lag\d+$')
_RE_ROLLING     = re.compile(r'_roll\d+_(mean|std)$')


def classify_feature_category(col: str) -> str:
    """Map a feature column name to one of the six Phase 4 categories.

    Returns
    -------
    str
        One of ``CATEGORY_ORDER``. Fallthrough is 'base'.

    Notes
    -----
    Previously duplicated 3× across Phase 6 Step 3 scripts
    (S1 / S3 / S5b). Promoted to ``src/`` per D-074.

    Priority ordering in the regex checks is deliberate:
    interaction columns contain ``_D_{break_name}`` substrings so
    must be tested before split dummies.
    """
    if _RE_INTERACTION.search(col):
        return 'interaction'
    if _RE_SPLIT.search(col):
        return 'split'
    if _RE_PERIOD.search(col):
        return 'period'
    if _RE_LAG.search(col):
        return 'lag'
    if _RE_ROLLING.search(col):
        return 'rolling'
    return 'base'


# ────────────────────────────────────────────────────────────────────
# USA dual-form construction (D-064 / D-071)
# ────────────────────────────────────────────────────────────────────

def build_usa_first_diff_features() -> 'pd.DataFrame':
    """Build USA feature matrix with CPI forced to first_diff form.

    Implements the D-064 dual-form construction: temporarily overrides
    ``src.feature_engineering.REGISTRY_OVERRIDES[('USA', 'CPI')]`` to
    ``'first_diff'``, rebuilds the USA feature matrix end-to-end so
    that all CPI-derived lag, rolling, and interaction columns are
    form-consistent with the first_diff base, then restores the
    original dictionary state in a ``finally`` block.

    The secondary-form matrix joins Phase 7 Diebold-Mariano alongside
    the D-031 primary yoy_pct form per D-048 / D-062 / D-071.

    Returns
    -------
    pandas.DataFrame
        USA feature matrix with CPI in first_diff secondary form.
        Shape ≈ (298, 53), joint-valid start 2002-02-01 (vs 2003-01-01
        for the yoy_pct primary form).

    Notes
    -----
    Previously duplicated 4× across Phase 6 Step 3 scripts
    (S1 / S2 / S3 / S4). Promoted to ``src/`` per D-074.

    The override dictionary uses **tuple keys** ``(country, indicator)``;
    earlier scratch-script implementations occasionally mis-used string
    keys. This canonical implementation uses the tuple form.
    """
    # Local import avoids circular dependency at module-load time
    # (feature_engineering imports from us for PHASE6_REGIME_SPEC).
    from . import feature_engineering as fe_module
    from .feature_engineering import build_country_features

    key = ("USA", "CPI")
    had_override = key in fe_module.REGISTRY_OVERRIDES
    original_value = fe_module.REGISTRY_OVERRIDES.get(key)
    try:
        fe_module.REGISTRY_OVERRIDES[key] = "first_diff"
        out = build_country_features("USA")
    finally:
        if had_override:
            fe_module.REGISTRY_OVERRIDES[key] = original_value
        else:
            fe_module.REGISTRY_OVERRIDES.pop(key, None)
    return out


# ────────────────────────────────────────────────────────────────────
# α-selection provenance I/O (D-065 / D-066)
# ────────────────────────────────────────────────────────────────────

def load_selected_alphas(
    doc_dir: Optional[Path] = None,
) -> dict[tuple[str, str], float]:
    """Load S2 + S2b α selection CSVs and merge into a single dict.

    S2 supplies the initial α* for all 5 (country, form) combinations;
    S2b overrides the Japan primary entry with the extended-grid value
    per D-066. This is the canonical access point for any downstream
    Phase 6 Step 3 consumer that needs the selected α*.

    Parameters
    ----------
    doc_dir : Path, optional
        Path to ``data/documentation/`` directory. If ``None``, auto-
        locates via ``find_project_root() / 'data' / 'documentation'``.

    Returns
    -------
    dict[tuple[str, str], float]
        Keys are (country, form) tuples. Values are α* per (country, form).

    Raises
    ------
    FileNotFoundError
        If either input CSV is missing (run S2 / S2b scripts to regenerate).

    Notes
    -----
    Previously duplicated 3× across Phase 6 Step 3 scripts
    (S3 / S4 / S5). Promoted to ``src/`` per D-074.
    """
    from .data_loader import find_project_root

    if doc_dir is None:
        doc_dir = find_project_root() / 'data' / 'documentation'

    path_s2  = doc_dir / "phase6_step3_s2_alpha_selection.csv"
    path_s2b = doc_dir / "phase6_step3_s2b_japan_alpha_selection.csv"

    for p in (path_s2, path_s2b):
        if not p.exists():
            raise FileNotFoundError(
                f"α selection CSV missing: {p}. "
                f"Run the corresponding scripts/phase6_step3_s2*.py "
                f"to regenerate."
            )

    sel_s2  = pd.read_csv(path_s2)
    sel_s2b = pd.read_csv(path_s2b)

    out: dict[tuple[str, str], float] = {}
    for _, r in sel_s2.iterrows():
        out[(r["country"], r["form"])] = float(r["selected_alpha"])
    # S2b override (Japan primary)
    for _, r in sel_s2b.iterrows():
        out[(r["country"], r["form"])] = float(r["selected_alpha"])
    return out


# ────────────────────────────────────────────────────────────────────
# Ridge pipeline factory (D-065)
# ────────────────────────────────────────────────────────────────────

def make_ridge_pipeline(
    alpha: float,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> '_Pipeline':
    """Return an UNFITTED ``Pipeline(StandardScaler → Ridge(alpha))``.

    Encapsulates the D-065 leakage-guard convention: ``StandardScaler``
    is the first step of the Pipeline so ``pipe.fit(X_tr, y_tr)`` fits
    the scaler on the training fold only; subsequent ``pipe.predict``
    applies the frozen scaler transform.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength for Ridge. Typically an element of
        :data:`ALPHA_GRID_DEFAULT` at S2, or a selected α* from
        :func:`load_selected_alphas`.
    random_state : int, default :data:`RANDOM_STATE_DEFAULT`
        Passed to ``Ridge``. Deterministic in principle; fixed for
        reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted Pipeline. Caller must call ``.fit(X, y)``.

    Notes
    -----
    Previously inlined 4× across Phase 6 Step 3 scripts
    (S2 / S2b / S3 / S4). Promoted to ``src/`` per D-074.

    Scikit-learn is not a hard dependency of ``src.modelling_utils``;
    it is imported lazily inside this function so that importing
    ``src`` does not require sklearn to be installed for consumers
    who only need Phase 1–5 utilities.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha, random_state=random_state)),
    ])


# ────────────────────────────────────────────────────────────────────
# Walk-forward origin construction (D-068)
# ────────────────────────────────────────────────────────────────────

def compute_walk_forward_origins(
    index: 'pd.DatetimeIndex',
    test_start: 'pd.Timestamp' = None,
    horizons: tuple[int, ...] = HORIZONS_PHASE7,
) -> 'pd.DatetimeIndex':
    """Compute paired-DM walk-forward origin set.

    Returns the subset of ``index`` that can serve as an origin for all
    horizons in ``horizons`` simultaneously. Enforces:

        origin >= test_start  AND  origin + max(horizons) months <= index[-1]

    so that every origin has a valid target observation at every horizon.
    Matches VAR S6 (D-060) and Ridge S4 (D-068) conventions for paired
    Phase 7 Diebold-Mariano comparison.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        The observation index (typically ``X_full.index`` after
        ``dropna()``). Must be monthly.
    test_start : pandas.Timestamp, optional
        First valid origin. Defaults to :data:`TEST_START` (2020-01-01).
    horizons : tuple[int, ...]
        Forecast horizons in months. Defaults to :data:`HORIZONS_PHASE7`.

    Returns
    -------
    pandas.DatetimeIndex
        Origins satisfying the paired-DM constraint.

    Notes
    -----
    Previously inlined 1× in Phase 6 Step 3 S4. Promoted to ``src/``
    per D-074 to support Phase 7 DM reuse.
    """
    if test_start is None:
        test_start = TEST_START

    h_max = max(horizons)
    last_date = index[-1]
    last_origin = last_date - pd.DateOffset(months=h_max)
    return index[(index >= test_start) & (index <= last_origin)]


# ────────────────────────────────────────────────────────────────────
# Cross-layer comparison reference (D-060)
# ────────────────────────────────────────────────────────────────────

#: VAR OOS MASE at AIC-selected p per country (D-060).
#: Primary form only (D-031). Used by Phase 6 Step 3 S5 for the
#: Ridge-vs-VAR comparison (D-070) and by Phase 7 DM for the VAR
#: baseline. Keys: (country, horizon).
VAR_MASE_D060: dict[tuple[str, int], float] = {
    ("USA",      1):  3.73, ("USA",      3): 11.61,
    ("USA",      6): 20.64, ("USA",     12): 32.32,
    ("JAPAN",    1):  0.89, ("JAPAN",    3):  0.96,
    ("JAPAN",    6):  0.91, ("JAPAN",   12):  1.03,
    ("UK",       1):  1.90, ("UK",       3):  1.95,
    ("UK",       6):  5.60, ("UK",      12): 79.07,
    ("GERMANY",  1):  1.48, ("GERMANY",  3):  1.76,
    ("GERMANY",  6):  1.56, ("GERMANY", 12):  2.26,
}


# ════════════════════════════════════════════════════════════════════
# Module exports
# ════════════════════════════════════════════════════════════════════

__all__ = [
    # v0.4.1 (D-063) — Phase 6 Step 2 heritage
    'P_PER_COUNTRY_AIC',
    'P_PER_COUNTRY_BIC',
    'CHOLESKY_ORDER',
    'SPLIT_BREAK_NAMES',
    'PERIOD_KEYS',
    'build_regime_exog_columns',
    'extract_endog_exog_cholesky',
    # v0.4.2 (D-074) — Phase 6 Step 3 additions
    'TRAIN_END',
    'TEST_START',
    'HORIZONS_PHASE7',
    'ALPHA_GRID_DEFAULT',
    'N_SPLITS_DEFAULT',
    'RANDOM_STATE_DEFAULT',
    'CATEGORY_ORDER',
    'classify_feature_category',
    'build_usa_first_diff_features',
    'load_selected_alphas',
    'make_ridge_pipeline',
    'compute_walk_forward_origins',
    'VAR_MASE_D060',
]
