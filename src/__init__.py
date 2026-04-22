"""
src
===
Reusable analytical modules for the Inflation Prediction and Economic
Signal Analysis project (Portfolio Project 3).

This package is consumed by:
  - notebooks/01_data_collection.ipynb          (Phase 1 narrative)
  - notebooks/02_cleaning_alignment.ipynb        (Phase 2 narrative)
  - notebooks/03_stationarity_structural_breaks.ipynb  (Phase 3 narrative)
  - notebooks/04_feature_engineering.ipynb      (Phase 4 narrative)
  - notebooks/07_var_model.ipynb                 (Phase 6 Step 2 narrative)
  - notebooks/08_ridge_regression.ipynb          (Phase 6 Step 3 narrative)
  - notebooks/09_evaluation_interpretation.ipynb (Phase 7 narrative, pending)
  - scripts/*.py                                 (CLI orchestrators)

Modules
-------
data_loader          I/O helpers for raw and processed datasets.
preprocessing        Phase 2 transformations (unit/frequency harmonisation,
                     NaN handling, wide-format assembly, schema generation).
stationarity         Phase 3 Task 1: ADF + KPSS joint protocol, 4-quadrant
                     classification, transformation dispatch.
structural_breaks    Phase 3 Task 2: Chow (classical / HAC / COVID-dummy),
                     per-coefficient decomposition, Quandt-Andrews sup-Wald.
feature_engineering  Phase 4: base transform → lags → rolling stats →
                     regime dummies → wide feature matrix assembly.
modelling_utils      Phase 6 shared utilities. At v0.4.2 covers:
                       - VAR: Cholesky ordering, AIC/BIC lag orders,
                         D-030 exog column assembly (v0.4.1, D-063).
                       - Ridge: train/test window constants, α grid,
                         feature-category regex, USA dual-form builder,
                         Pipeline factory, walk-forward origin helper,
                         VAR-MASE reference for cross-layer comparison
                         (v0.4.2, D-074).
                     Narrow-scope principle retained (no model-fitting
                     calls); full model wrappers remain deferred to
                     Phase 7 closeout's v0.5.0 assessment per D-075.
evaluation           Phase 7 evaluation primitives (v0.4.3, D-076):
                       - Loss functions (RMSE, MAE, MASE).
                       - Diebold-Mariano variants (standard, HAC,
                         robust) with Harvey-Leybourne-Newbold (1997)
                         small-sample correction.
                       - CSV adapters that normalise Phase 6 OOS
                         forecast artefacts into a unified
                         DM-ready schema.
                     Narrow-scope: no forecast re-computation, no
                     aggregation across (country, h) cells.

Version history
---------------
0.1.0  Initial package scaffolding (Phase 1).
0.2.0  Added preprocessing module and expanded data_loader (Phase 2).
0.3.0  Added stationarity and structural_breaks modules (Phase 3).
0.4.0  Added feature_engineering module (Phase 4).
0.4.1  Added modelling_utils module (Phase 6 Step 2 closeout; D-063).
       Patch bump — no API change to existing modules; promotes
       helpers duplicated 4+ times across nine Step 2 scratch scripts.
0.4.2  Extended modelling_utils with Phase 6 Step 3 helpers (D-074).
       Patch bump — no API change to v0.4.1; adds 7 constants +
       5 helper functions + 1 reference dict. Driven by the same
       4×-duplication rule as D-063.
0.4.3  Added evaluation module for Phase 7 DM pre-flight (D-076,
       executing D-075 Tranche 1). Patch bump — no API change to
       v0.4.2 exports. Adds 1 correction callable + 3 loss primitives
       + 3 Diebold-Mariano variants + 2 CSV adapters + 1 schema
       constant = 10 new exports. ProjectScope §12 blueprint item 5
       of 8 now materialised; `src/models/` subdirectory remains
       deferred per D-075 Tranche 2.
"""
from __future__ import annotations

__version__ = "0.4.3"


# ──────────────────────────────────────────────────────────────────
# Phase 1 / Phase 2 re-exports
# ──────────────────────────────────────────────────────────────────
from .data_loader import (       # noqa: F401
    find_project_root,
    MAIN_COUNTRIES,
    SUPPLEMENTARY_COUNTRIES,
    INDICATORS,
    load_raw_series,
    load_all_raw,
    load_processed_main,
    load_processed_all_main,
    load_processed_china,
)

from .preprocessing import (     # noqa: F401
    m2_to_yoy,
    gdp_quarterly_to_monthly_yoy,
    normalise_monthly_index,
    interpolate_single_gaps,
    process_country,
    assemble_wide,
    trim_effective_window,
    build_processed,
    build_all_processed,
    write_schema_md,
    M2_UNITS,
    ANALYSIS_START,
    YOY_LOOKBACK,
    GAP_INTERP_MAX,
)

# ──────────────────────────────────────────────────────────────────
# Phase 3 re-exports
# ──────────────────────────────────────────────────────────────────
from .stationarity import (      # noqa: F401
    DEFAULT_ALPHA,
    ADF_REGRESSION_LEVEL,
    TRANSFORM_FN,
    FOUR_QUADRANT_CLASSES,
    FLAGGED_CLASSES,
    CONFLICT_CLASS,
    SMALL_SAMPLE_WARN,
    SMALL_SAMPLE_HARD,
    first_difference,
    second_difference,
    yoy_pct,
    log_first_diff_pct,
    apply_transform,
    strip_suffix,
    schwert_maxlag,
    run_adf,
    run_kpss,
    classify_4quadrant,
    test_series,
    test_all_series,
)

from .structural_breaks import (  # noqa: F401
    DEFAULT_HAC_LAG,
    KNOWN_BREAKS,
    COVID_DUMMY_START,
    COVID_DUMMY_END,
    COVID_DUMMY_BREAKS,
    ANDREWS_1993_TABLE_I,
    make_split_dummy,
    make_covid_dummy,
    chow_test_classical,
    chow_test_hac,
    chow_test_covid_dummy,
    coefficient_decomposition,
    wald_at_break,
    quandt_andrews_scan,
    summarise_scan,
    align_argmax_to_known,
)

# ──────────────────────────────────────────────────────────────────
# Phase 4 re-exports (v0.4.0)
# ──────────────────────────────────────────────────────────────────
from .feature_engineering import (  # noqa: F401
    REGISTRY_OVERRIDES,
    LAG_PERIODS,
    ROLLING_WINDOWS,
    ROLLING_STATS,
    PERIOD_WINDOWS,
    PHASE6_REGIME_SPEC,
    load_effective_registry,
    transform_country,
    build_lag_matrix,
    build_rolling_matrix,
    build_split_dummies,
    build_period_dummies,
    build_interactions,
    build_regime_matrix,
    build_country_features,
    build_all_features,
    write_features_schema_md,
)

# ──────────────────────────────────────────────────────────────────
# Phase 6 modelling utilities re-exports
#   v0.4.1 baseline (D-063) + v0.4.2 additions (D-074)
# ──────────────────────────────────────────────────────────────────
from .modelling_utils import (  # noqa: F401
    # v0.4.1 (D-063)
    P_PER_COUNTRY_AIC,
    P_PER_COUNTRY_BIC,
    CHOLESKY_ORDER,
    SPLIT_BREAK_NAMES,
    PERIOD_KEYS,
    build_regime_exog_columns,
    extract_endog_exog_cholesky,
    # v0.4.2 (D-074) — Phase 6 Step 3 additions
    TRAIN_END,
    TEST_START,
    HORIZONS_PHASE7,
    ALPHA_GRID_DEFAULT,
    N_SPLITS_DEFAULT,
    RANDOM_STATE_DEFAULT,
    CATEGORY_ORDER,
    classify_feature_category,
    build_usa_first_diff_features,
    load_selected_alphas,
    make_ridge_pipeline,
    compute_walk_forward_origins,
    VAR_MASE_D060,
)

# ──────────────────────────────────────────────────────────────────
# Phase 7 evaluation re-exports (v0.4.3, D-076)
# ──────────────────────────────────────────────────────────────────
from .evaluation import (  # noqa: F401
    # Harvey-Leybourne-Newbold correction (callable constant)
    HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT,
    # Loss primitives
    rmse,
    mae,
    mase,
    # Diebold-Mariano variants
    diebold_mariano_standard,
    diebold_mariano_hac,
    diebold_mariano_robust,
    # CSV adapters
    load_phase6_forecasts,
    align_matched_terms,
    # Unified schema reference
    UNIFIED_SCHEMA_COLUMNS,
)


__all__ = [
    # Package meta
    "__version__",
    # Data access
    "find_project_root",
    "MAIN_COUNTRIES",
    "SUPPLEMENTARY_COUNTRIES",
    "INDICATORS",
    "load_raw_series",
    "load_all_raw",
    "load_processed_main",
    "load_processed_all_main",
    "load_processed_china",
    # Phase 2 preprocessing
    "m2_to_yoy",
    "gdp_quarterly_to_monthly_yoy",
    "normalise_monthly_index",
    "interpolate_single_gaps",
    "process_country",
    "assemble_wide",
    "trim_effective_window",
    "build_processed",
    "build_all_processed",
    "write_schema_md",
    "M2_UNITS",
    "ANALYSIS_START",
    "YOY_LOOKBACK",
    "GAP_INTERP_MAX",
    # Phase 3 stationarity
    "DEFAULT_ALPHA",
    "ADF_REGRESSION_LEVEL",
    "TRANSFORM_FN",
    "FOUR_QUADRANT_CLASSES",
    "FLAGGED_CLASSES",
    "CONFLICT_CLASS",
    "SMALL_SAMPLE_WARN",
    "SMALL_SAMPLE_HARD",
    "first_difference",
    "second_difference",
    "yoy_pct",
    "log_first_diff_pct",
    "apply_transform",
    "strip_suffix",
    "schwert_maxlag",
    "run_adf",
    "run_kpss",
    "classify_4quadrant",
    "test_series",
    "test_all_series",
    # Phase 3 structural breaks
    "DEFAULT_HAC_LAG",
    "KNOWN_BREAKS",
    "COVID_DUMMY_START",
    "COVID_DUMMY_END",
    "COVID_DUMMY_BREAKS",
    "ANDREWS_1993_TABLE_I",
    "make_split_dummy",
    "make_covid_dummy",
    "chow_test_classical",
    "chow_test_hac",
    "chow_test_covid_dummy",
    "coefficient_decomposition",
    "wald_at_break",
    "quandt_andrews_scan",
    "summarise_scan",
    "align_argmax_to_known",
    # Phase 4 feature engineering
    "REGISTRY_OVERRIDES",
    "LAG_PERIODS",
    "ROLLING_WINDOWS",
    "ROLLING_STATS",
    "PERIOD_WINDOWS",
    "PHASE6_REGIME_SPEC",
    "load_effective_registry",
    "transform_country",
    "build_lag_matrix",
    "build_rolling_matrix",
    "build_split_dummies",
    "build_period_dummies",
    "build_interactions",
    "build_regime_matrix",
    "build_country_features",
    "build_all_features",
    "write_features_schema_md",
    # Phase 6 modelling utilities — v0.4.1 (D-063)
    "P_PER_COUNTRY_AIC",
    "P_PER_COUNTRY_BIC",
    "CHOLESKY_ORDER",
    "SPLIT_BREAK_NAMES",
    "PERIOD_KEYS",
    "build_regime_exog_columns",
    "extract_endog_exog_cholesky",
    # Phase 6 modelling utilities — v0.4.2 (D-074)
    "TRAIN_END",
    "TEST_START",
    "HORIZONS_PHASE7",
    "ALPHA_GRID_DEFAULT",
    "N_SPLITS_DEFAULT",
    "RANDOM_STATE_DEFAULT",
    "CATEGORY_ORDER",
    "classify_feature_category",
    "build_usa_first_diff_features",
    "load_selected_alphas",
    "make_ridge_pipeline",
    "compute_walk_forward_origins",
    "VAR_MASE_D060",
    # Phase 7 evaluation — v0.4.3 (D-076)
    "HARVEY_LEYBOURNE_NEWBOLD_ADJUSTMENT",
    "rmse",
    "mae",
    "mase",
    "diebold_mariano_standard",
    "diebold_mariano_hac",
    "diebold_mariano_robust",
    "load_phase6_forecasts",
    "align_matched_terms",
    "UNIFIED_SCHEMA_COLUMNS",
]
