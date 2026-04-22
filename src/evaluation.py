"""
src.evaluation
==============
Phase 7 evaluation primitives — loss functions, Diebold-Mariano tests,
and CSV adapters for the Phase 6 out-of-sample forecast artefacts.

Promoted at v0.4.3 per D-076 (executing D-075 Tranche 1).

Scope
-----
Deliberately narrow, consistent with the D-063 / D-074 convention that
``src/`` houses pure helpers and no model-fitting or refit-loop logic.

Included:
    * Pointwise loss primitives (RMSE, MAE, MASE) — caller supplies
      arrays; no CSV / pandas dependency in the primitive bodies.
    * Three Diebold-Mariano variants (standard, HAC-robust,
      absolute-loss) each with Harvey-Leybourne-Newbold (1997)
      small-sample correction.
    * Harvey-Leybourne-Newbold correction factor as a named callable
      so it can be audited / reused independent of the DM variants.
    * Two CSV adapters that normalise the three Phase 6 forecast
      artefacts into a unified schema suitable for paired-DM analysis.

Deliberately excluded:
    * Any model refit / forecast re-computation (forecasts are consumed
      from pre-computed Phase 6 CSVs).
    * DM aggregation across (country, h) cells — that is Phase 7 S2
      orchestration, not a ``src/`` helper.
    * Plotting — Phase 7 notebook assembly owns figures.

Decision linkage
----------------
D-048   ARIMA Stage (a) vs Stage (c) stopping rule — Phase 7 S3 sensitivity.
D-051   VAR(12) partial-whitening caveat — HAC DM variant exists for this.
D-060   VAR OOS MASE at AIC-selected p (hardcoded in ``VAR_MASE_D060``).
D-061   COVID-origin VAR(12) instability — robust-loss DM exists for this.
D-062   USA yoy_pct × VAR(12) systematic bias — Phase 7 main DM.
D-068   Ridge walk-forward origins matched to VAR S6 (empirically
        verified at Phase 7 pre-flight: 58/58 USA, 58/58 JAPAN,
        51/51 UK, 51/51 GERMANY).
D-070   Ridge-vs-VAR 12/16 relative win — Phase 7 main DM.
D-071   USA dual-form resolution — Phase 7 S3 sensitivity.
D-075   Split-promotion architectural decision — Tranche 1 executed here.
D-076   This module's creation record.

Harvey, D.; Leybourne, S.; Newbold, P. (1997). "Testing the equality of
prediction mean squared errors." *International Journal of Forecasting*
13 (2), 281–291.

Diebold, F. X.; Mariano, R. S. (1995). "Comparing predictive accuracy."
*Journal of Business & Economic Statistics* 13 (3), 253–263.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .data_loader import find_project_root


# ──────────────────────────────────────────────────────────────────────
# Small-sample correction (exported as a "constant" callable per D-076)
# ──────────────────────────────────────────────────────────────────────

def HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT(T: int, h: int) -> float:
    """Harvey-Leybourne-Newbold (1997) small-sample correction factor.

    Multiplies the raw DM statistic to improve empirical size under
    finite ``T`` and multi-step (``h > 1``) forecasts. Reduces to
    ``sqrt((T - 1) / T)`` at ``h = 1`` — a minor shrinkage that
    prevents spurious rejections when ``T`` is small.

    Exported in ALL-CAPS because it is treated as a reference formula
    from the literature (a "constant-like callable") rather than a
    general helper function; this mirrors the D-076 API surface where
    it appears alongside named constants.

    Parameters
    ----------
    T : int
        Length of the loss-differential series.
    h : int
        Forecast horizon (steps ahead).

    Returns
    -------
    float
        Multiplicative correction factor. Positive for all valid
        ``(T, h)``; equals ``sqrt((T + 1 - 2h + h(h-1)/T) / T)``.

    Raises
    ------
    ValueError
        If ``T <= 0`` or ``h <= 0`` or the correction would be
        non-real (``T + 1 - 2h + h(h-1)/T < 0``; arises only for
        pathologically small ``T`` relative to ``h``).
    """
    if T <= 0 or h <= 0:
        raise ValueError(f"T and h must be positive; got T={T}, h={h}")
    inner = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if inner < 0:
        raise ValueError(
            f"HLN correction argument is negative (T={T}, h={h}); "
            f"need T >= 2h - 1 roughly."
        )
    return sqrt(inner)


# ──────────────────────────────────────────────────────────────────────
# Pointwise loss primitives
# ──────────────────────────────────────────────────────────────────────

def rmse(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Root mean squared error, NaN-aware.

    Parameters
    ----------
    y_true, y_pred : array-like
        Must be broadcast-compatible.
    axis : int or None
        Passed to ``np.nanmean``; default collapses everything to a scalar.

    Returns
    -------
    float or np.ndarray
        ``sqrt(nanmean((y_true - y_pred)**2))``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=axis))


def mae(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Mean absolute error, NaN-aware."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.nanmean(np.abs(y_true - y_pred), axis=axis)


def mase(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    scale_denominator: float,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Mean absolute scaled error.

    Caller-supplied ``scale_denominator`` follows the Phase 7 Q#2
    commitment (both ``VAR_MASE_D060`` and seasonal-naive scales are
    valid; reporting strategy is caller-side).

    Parameters
    ----------
    y_true, y_pred : array-like
    scale_denominator : float
        Typically the in-sample naive MAE of the same target series.
        Examples:

        * Random-walk naive: ``mean(|y_t - y_{t-1}|)`` over the
          training window.
        * Seasonal naive (12): ``mean(|y_t - y_{t-12}|)``.
        * VAR-shared: ``src.modelling_utils.VAR_MASE_D060[(country, h)]``
          (used by Phase 6 Step 3 D-070 for cross-layer comparability).

    Returns
    -------
    float or np.ndarray
        ``MAE / scale_denominator``.

    Raises
    ------
    ValueError
        If ``scale_denominator <= 0`` or not finite.
    """
    scale = float(scale_denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"scale_denominator must be a positive finite float; got {scale!r}"
        )
    return mae(y_true, y_pred, axis=axis) / scale


# ──────────────────────────────────────────────────────────────────────
# Diebold-Mariano primitives
# ──────────────────────────────────────────────────────────────────────

def _loss_differential(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: Literal["squared", "absolute"],
) -> np.ndarray:
    """Compute d_t = L(e1_t) - L(e2_t) under the specified loss."""
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError(
            f"e1.shape={e1.shape} != e2.shape={e2.shape}; "
            f"DM requires matched-terms inputs."
        )
    if loss == "squared":
        return e1 ** 2 - e2 ** 2
    if loss == "absolute":
        return np.abs(e1) - np.abs(e2)
    raise ValueError(f"Unknown loss {loss!r}; expected 'squared' or 'absolute'.")


def _newey_west_long_run_variance(d: np.ndarray, n_lags: int) -> float:
    """Newey-West long-run variance estimator with Bartlett kernel.

    ``LRV = γ_0 + 2 Σ_{l=1..n_lags} w_l γ_l``,
    where ``w_l = 1 - l / (n_lags + 1)`` (Bartlett) and ``γ_l`` is the
    sample autocovariance at lag ``l`` of the demeaned series.

    Parameters
    ----------
    d : np.ndarray
        1-D series.
    n_lags : int
        Truncation lag. Must be non-negative; ``n_lags = 0`` reduces
        to the sample variance (γ_0 only).

    Returns
    -------
    float
        Long-run variance estimate. May be non-positive in pathological
        cases; caller should check before taking square roots.
    """
    if n_lags < 0:
        raise ValueError(f"n_lags must be non-negative; got {n_lags}")
    T = len(d)
    d_bar = d.mean()
    demean = d - d_bar
    gamma_0 = float(np.sum(demean ** 2) / T)
    lrv = gamma_0
    for lag in range(1, n_lags + 1):
        w = 1.0 - lag / (n_lags + 1.0)
        gamma_lag = float(np.sum(demean[lag:] * demean[:-lag]) / T)
        lrv += 2.0 * w * gamma_lag
    return lrv


def _dm_from_variance(
    d_bar: float,
    var_estimate: float,
    T: int,
    h: int,
) -> Tuple[float, float]:
    """Assemble (dm_stat, p_value) given a variance estimate of ``d_bar * sqrt(T)``.

    Applies HLN correction and returns two-sided p-value using the
    t-distribution with ``T - 1`` degrees of freedom, following HLN.
    """
    if T <= h or var_estimate <= 0 or not np.isfinite(var_estimate):
        return float("nan"), float("nan")
    raw_dm = d_bar / sqrt(var_estimate / T)
    dm = raw_dm * HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT(T, h)
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=T - 1))
    return float(dm), float(p)


def diebold_mariano_standard(
    e1: Union[Sequence[float], np.ndarray],
    e2: Union[Sequence[float], np.ndarray],
    h: int,
) -> Tuple[float, float]:
    """Standard Diebold-Mariano test with HLN small-sample correction.

    * Loss: squared error.
    * Variance: sample variance of the loss differential divided by
      ``T`` (asymptotic; assumes ``d_t`` serially uncorrelated — valid
      at ``h = 1`` under forecast optimality).
    * Small-sample correction: HLN (1997).

    Parameters
    ----------
    e1, e2 : array-like
        Matched-terms forecast errors of models 1 and 2. Same length.
    h : int
        Forecast horizon (steps ahead).

    Returns
    -------
    (dm_stat, p_value) : tuple of float
        ``p_value`` is two-sided. ``dm_stat < 0`` means model 1 has
        lower loss (is better); ``dm_stat > 0`` means model 2 is better.
        Returns ``(nan, nan)`` if ``T <= h`` or variance is degenerate.
        Returns ``(0.0, 1.0)`` if the two models produce identical
        errors (``d_t ≡ 0``).
    """
    d = _loss_differential(e1, e2, loss="squared")
    T = len(d)
    d_bar = float(d.mean())
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    sigma2 = float(d.var(ddof=1))
    return _dm_from_variance(d_bar, sigma2, T, h)


def diebold_mariano_hac(
    e1: Union[Sequence[float], np.ndarray],
    e2: Union[Sequence[float], np.ndarray],
    h: int,
    kernel: str = "bartlett",
    n_lags: Optional[int] = None,
) -> Tuple[float, float]:
    """DM test with HAC (Newey-West) long-run variance.

    Addresses the D-051 partial-whitening caveat: at ``h > 1`` the
    optimal loss differential is serially correlated up to lag ``h-1``,
    and even at ``h = 1`` VAR(12) residuals exhibit partial whitening
    (LB(12) pass rate 55% per D-051). HAC variance absorbs that
    correlation into the denominator.

    Parameters
    ----------
    e1, e2 : array-like
    h : int
        Forecast horizon.
    kernel : {"bartlett"}
        Currently only Bartlett is supported. Placeholder parameter
        for future kernel extensions.
    n_lags : int, optional
        Truncation lag. Defaults to ``max(h - 1, 0)``, the standard
        DM convention. At ``h = 1`` defaults to ``0`` (HAC reduces
        to sample variance), so HAC and standard DM agree for h=1 up
        to the (T-1)/T divisor difference.

    Returns
    -------
    (dm_stat, p_value) : tuple of float
    """
    if kernel != "bartlett":
        raise NotImplementedError(
            f"kernel={kernel!r} not supported; only 'bartlett' is implemented."
        )
    d = _loss_differential(e1, e2, loss="squared")
    T = len(d)
    d_bar = float(d.mean())
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    if n_lags is None:
        n_lags = max(h - 1, 0)
    lrv = _newey_west_long_run_variance(d, n_lags)
    return _dm_from_variance(d_bar, lrv, T, h)


def diebold_mariano_robust(
    e1: Union[Sequence[float], np.ndarray],
    e2: Union[Sequence[float], np.ndarray],
    h: int,
) -> Tuple[float, float]:
    """DM with absolute-error loss (robust to outliers).

    Uses loss differential ``|e1_t| - |e2_t|`` instead of the squared
    difference. Rationale (D-061): the UK h=12 cell contains a
    walk-forward origin (2020-05-01) whose VAR forecast is −980.29
    against actual 0.54. Under squared-error loss this single row
    contributes ≈ 960k to the mean loss, dominating the test. Absolute
    loss scales linearly with the error, so the same row contributes
    ≈ 981 — non-trivial but not catastrophic.

    Variance: sample variance of the absolute-loss differential,
    divided by ``T``. HLN correction applied.

    Returns
    -------
    (dm_stat, p_value) : tuple of float
        Sign convention matches ``diebold_mariano_standard``.
    """
    d = _loss_differential(e1, e2, loss="absolute")
    T = len(d)
    d_bar = float(d.mean())
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    sigma2 = float(d.var(ddof=1))
    return _dm_from_variance(d_bar, sigma2, T, h)


# ──────────────────────────────────────────────────────────────────────
# CSV adapters — Phase 6 artefact → unified DM schema
# ──────────────────────────────────────────────────────────────────────
#
# Unified schema (Phase 7 Q#3 = option (b)):
#
#     country       str    — UPPERCASE; matches MAIN_COUNTRIES.
#     form          str    — 'primary' or 'secondary'. USA is the only
#                            country with 'secondary' (first_diff per
#                            D-048; all other countries use their D-031
#                            primary form exclusively).
#     h             int    — forecast horizon in months (1, 3, 6, 12).
#     origin_date   datetime64[ns]  — forecast made-at timestamp.
#     target_date   datetime64[ns]  — forecast-for timestamp.
#     y_true        float  — realised value at target_date.
#     y_pred        float  — model's forecast at target_date given
#                            information up to origin_date.
#
# ARIMA layer packs country + form into a single `variant_id`; the
# mapping below is the inverse.

_ARIMA_VARIANT_MAP: dict[str, Tuple[str, str]] = {
    "USA_yoy_pct":        ("USA",     "primary"),   # D-031 primary form
    "USA_first_diff":     ("USA",     "secondary"), # D-048 secondary form
    "JAPAN_first_diff":   ("JAPAN",   "primary"),   # D-031
    "UK_log_diff_pct":    ("UK",      "primary"),   # D-031
    "GERMANY_first_diff": ("GERMANY", "primary"),   # D-031
}

# Ridge Step 3 CSV uses a domain-specific compound label for the USA
# secondary-form rows (a Phase 6 Step 3 script-side naming choice; see
# `build_usa_first_diff_features` in `src.modelling_utils`). The unified
# schema uses the role abstraction `{primary, secondary}` that is
# consistent with the language of D-048 / D-064 / D-071, so this
# mapping normalises the Ridge label at adapter time. Unknown values
# raise `ValueError` — schema drift is a hard fault, not silent.
_RIDGE_FORM_MAP: dict[str, str] = {
    "primary":              "primary",
    "first_diff_secondary": "secondary",  # D-064 / D-071 USA dual-form
}

UNIFIED_SCHEMA_COLUMNS: Tuple[str, ...] = (
    "country", "form", "h", "origin_date", "target_date", "y_true", "y_pred",
)


def load_phase6_forecasts(
    layer: Literal["arima", "var", "ridge"],
    doc_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and normalise a Phase 6 OOS forecast CSV.

    Parameters
    ----------
    layer : {"arima", "var", "ridge"}
        Which Phase 6 artefact to load:

        * ``"arima"``  → ``phase6_step1_arima_forecast.csv``
          (5 variants × h=1 only per D-048; 336 rows)
        * ``"var"``    → ``phase6_step2_s6_var_oos_forecasts.csv``
          (4 countries × 5 variables × origins × {1,3,6,12};
          CPI rows only are retained — 872 rows of 4360)
        * ``"ridge"``  → ``phase6_step3_s4_ridge_oos_forecasts.csv``
          (5 combos × origins × {1,3,6,12}; 1104 rows)
    doc_dir : str or Path, optional
        Override the documentation directory. Defaults to
        ``<project_root>/data/documentation`` via ``find_project_root``.

    Returns
    -------
    pd.DataFrame
        Columns exactly ``UNIFIED_SCHEMA_COLUMNS``. Dtypes:
        ``country`` object, ``form`` object, ``h`` int64,
        ``origin_date`` / ``target_date`` datetime64[ns],
        ``y_true`` / ``y_pred`` float64.

    Raises
    ------
    FileNotFoundError
        If the expected CSV is not under ``doc_dir``.
    ValueError
        If ``layer`` is unknown, or if the CSV contains unexpected
        values — an ARIMA ``variant_id`` not in ``_ARIMA_VARIANT_MAP``,
        or a Ridge ``form`` not in ``_RIDGE_FORM_MAP``. Both
        dictionaries are enumerated at module level so schema drift
        surfaces at adapter time rather than silently downstream.

    Notes
    -----
    The Ridge Phase 6 Step 3 CSV uses a domain-specific compound label
    (``first_diff_secondary``) for the USA secondary-form rows. The
    unified schema uses the role abstraction ``{primary, secondary}``
    — consistent with D-048 / D-064 / D-071's language — so
    ``_load_ridge_forecasts`` normalises ``first_diff_secondary``
    to ``secondary`` via ``_RIDGE_FORM_MAP`` at load time. Downstream
    callers (``align_matched_terms``, S2 / S3 scripts) therefore
    match ARIMA ``USA_first_diff`` (which maps to ``(USA, "secondary")``)
    against the Ridge secondary-form rows on the same ``"secondary"``
    key — this is the intended dual-form DM pairing per D-071.
    """
    if doc_dir is None:
        doc_dir = find_project_root() / "data" / "documentation"
    doc_dir = Path(doc_dir)

    if layer == "arima":
        return _load_arima_forecasts(doc_dir)
    if layer == "var":
        return _load_var_forecasts(doc_dir)
    if layer == "ridge":
        return _load_ridge_forecasts(doc_dir)
    raise ValueError(
        f"Unknown layer {layer!r}; expected one of 'arima', 'var', 'ridge'."
    )


def _load_arima_forecasts(doc_dir: Path) -> pd.DataFrame:
    path = doc_dir / "phase6_step1_arima_forecast.csv"
    if not path.exists():
        raise FileNotFoundError(f"ARIMA forecast CSV not found: {path}")
    # Deliberately avoid `parse_dates=...` here: the Phase 6 Step 1 CSV
    # contains heterogeneous ISO8601 representations (`YYYY-MM-DD` for
    # the first variant block and `YYYY-MM-DD HH:MM:SS` for later
    # variants — a format drift at write-time). An explicit
    # `format="ISO8601"` on `pd.to_datetime` handles both forms
    # robustly.
    df = pd.read_csv(path)

    mapped = df["variant_id"].map(_ARIMA_VARIANT_MAP)
    unknown_mask = mapped.isna()
    if unknown_mask.any():
        bad = df.loc[unknown_mask, "variant_id"].unique().tolist()
        raise ValueError(
            f"Unknown ARIMA variant_id(s): {bad}. "
            f"Expected one of {list(_ARIMA_VARIANT_MAP.keys())}."
        )
    countries = mapped.map(lambda t: t[0])
    forms = mapped.map(lambda t: t[1])

    target = pd.to_datetime(df["date"], format="ISO8601")
    origin = target - pd.DateOffset(months=1)

    out = pd.DataFrame({
        "country":     countries.values,
        "form":        forms.values,
        "h":           1,  # expanding-refit 1-step only per D-048
        "origin_date": origin.values,
        "target_date": target.values,
        "y_true":      df["actual"].astype(float).values,
        "y_pred":      df["predicted"].astype(float).values,
    })
    return out[list(UNIFIED_SCHEMA_COLUMNS)]


def _load_var_forecasts(doc_dir: Path) -> pd.DataFrame:
    path = doc_dir / "phase6_step2_s6_var_oos_forecasts.csv"
    if not path.exists():
        raise FileNotFoundError(f"VAR forecast CSV not found: {path}")
    # Explicit ISO8601 parsing for cross-layer date-format robustness
    # (mirrors the defensive pattern used in _load_arima_forecasts).
    df = pd.read_csv(path)
    df["origin_date"] = pd.to_datetime(df["origin_date"], format="ISO8601")
    df["target_date"] = pd.to_datetime(df["target_date"], format="ISO8601")

    cpi = df[df["variable"] == "CPI"].copy()
    if cpi.empty:
        raise ValueError(
            "VAR CSV contains no rows with variable == 'CPI'; "
            "check the source file or Phase 6 Step 2 S6 output."
        )

    out = pd.DataFrame({
        "country":     cpi["country"].astype(str).values,
        "form":        "primary",  # VAR uses D-031 primary form per country
        "h":           cpi["horizon"].astype(int).values,
        "origin_date": cpi["origin_date"].values,
        "target_date": cpi["target_date"].values,
        "y_true":      cpi["actual"].astype(float).values,
        "y_pred":      cpi["forecast"].astype(float).values,
    })
    return out[list(UNIFIED_SCHEMA_COLUMNS)].reset_index(drop=True)


def _load_ridge_forecasts(doc_dir: Path) -> pd.DataFrame:
    path = doc_dir / "phase6_step3_s4_ridge_oos_forecasts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Ridge forecast CSV not found: {path}")
    # Explicit ISO8601 parsing for cross-layer date-format robustness.
    df = pd.read_csv(path)
    df["origin_date"] = pd.to_datetime(df["origin_date"], format="ISO8601")
    df["target_date"] = pd.to_datetime(df["target_date"], format="ISO8601")

    # Normalise Ridge's compound `first_diff_secondary` label to the
    # unified schema's role abstraction `secondary`. Fail loudly on
    # any unexpected form value to surface schema drift at load time.
    form_normalised = df["form"].map(_RIDGE_FORM_MAP)
    unknown_mask = form_normalised.isna()
    if unknown_mask.any():
        bad = df.loc[unknown_mask, "form"].unique().tolist()
        raise ValueError(
            f"Unknown Ridge form value(s): {bad}. "
            f"Expected one of {list(_RIDGE_FORM_MAP.keys())} "
            f"(unified schema normalises these to {{primary, secondary}})."
        )

    out = pd.DataFrame({
        "country":     df["country"].astype(str).values,
        "form":        form_normalised.values,
        "h":           df["horizon"].astype(int).values,
        "origin_date": df["origin_date"].values,
        "target_date": df["target_date"].values,
        "y_true":      df["actual"].astype(float).values,
        "y_pred":      df["forecast"].astype(float).values,
    })
    return out[list(UNIFIED_SCHEMA_COLUMNS)].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Matched-terms extractor for paired DM
# ──────────────────────────────────────────────────────────────────────

def align_matched_terms(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: Iterable[str] = ("country", "form", "h", "target_date"),
    y_true_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract same-target forecast pairs from two unified-schema DataFrames.

    Performs an inner merge on the specified key columns, verifies
    that ``y_true`` agrees between the two layers (within
    ``y_true_tol``), and returns the three arrays needed for a paired
    DM test.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Outputs of :func:`load_phase6_forecasts`. Must contain
        ``UNIFIED_SCHEMA_COLUMNS``.
    on : iterable of str
        Merge keys. Default ``("country", "form", "h", "target_date")``
        is the canonical matching grid for paired DM: two forecasts
        targeting the same country × form × horizon × date are treated
        as paired regardless of when they were made. Under expanding
        walk-forward refit with a fixed horizon ``h``, a single
        target_date maps to a single origin_date per layer
        (``origin = target - h months``), so the match is also
        origin-matched in practice; this is empirically verified by
        ``scripts/phase7_preflight_schema_check.py`` at the pre-flight
        gate.
    y_true_tol : float
        Maximum allowed absolute deviation in ``y_true`` between the
        two layers for the same merge key. Values above this raise
        ``ValueError`` — a hard signal of schema drift or a mismatch
        in country/form between the two inputs.

    Returns
    -------
    y_true : np.ndarray
        Shared realised values.
    e1, e2 : np.ndarray
        Forecast errors ``y_true - y_pred`` for the two layers,
        same length as ``y_true``.

    Raises
    ------
    ValueError
        If the merge is empty, if required columns are missing, or
        if the ``y_true`` tolerance is violated.
    """
    on = list(on)
    required = set(on) | {"y_true", "y_pred"}
    missing_1 = required - set(df1.columns)
    missing_2 = required - set(df2.columns)
    if missing_1 or missing_2:
        raise ValueError(
            f"Missing columns for alignment. "
            f"df1 missing={missing_1}; df2 missing={missing_2}. "
            f"Use load_phase6_forecasts() to produce unified-schema inputs."
        )

    merged = df1.merge(df2, on=on, suffixes=("_1", "_2"), how="inner")
    if len(merged) == 0:
        raise ValueError(f"No matched rows after inner-join on keys {on}.")

    y1 = merged["y_true_1"].to_numpy(dtype=float)
    y2 = merged["y_true_2"].to_numpy(dtype=float)
    max_abs_diff = float(np.nanmax(np.abs(y1 - y2)))
    if max_abs_diff > y_true_tol:
        # Surface a small diagnostic on which rows fail most.
        worst_idx = int(np.nanargmax(np.abs(y1 - y2)))
        worst_row = merged.iloc[worst_idx]
        raise ValueError(
            f"y_true mismatch across layers exceeds tol={y_true_tol:g}: "
            f"max |Δ|={max_abs_diff:g}. "
            f"Worst row: { {k: worst_row[k] for k in on} } "
            f"→ y_true_1={y1[worst_idx]!r}, y_true_2={y2[worst_idx]!r}. "
            f"Likely cause: country/form mix-up between df1 and df2, or "
            f"different CPI transforms at the same (country, form) key."
        )

    y_true = y1
    e1 = y_true - merged["y_pred_1"].to_numpy(dtype=float)
    e2 = y_true - merged["y_pred_2"].to_numpy(dtype=float)
    return y_true, e1, e2


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

__all__ = [
    # Small-sample correction
    "HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT",
    # Loss primitives
    "rmse",
    "mae",
    "mase",
    # Diebold-Mariano variants
    "diebold_mariano_standard",
    "diebold_mariano_hac",
    "diebold_mariano_robust",
    # CSV adapters
    "load_phase6_forecasts",
    "align_matched_terms",
    # Schema reference
    "UNIFIED_SCHEMA_COLUMNS",
]
