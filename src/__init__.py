"""
src
===
Reusable analytical modules for the Inflation Prediction and Economic
Signal Analysis project (Portfolio Project 3).

This package is consumed by:
  - notebooks/01_data_collection.ipynb          (Phase 1 narrative)
  - notebooks/02_cleaning_alignment.ipynb        (Phase 2 narrative)
  - notebooks/03_stationarity_structural_breaks.ipynb  (Phase 3 narrative)
  - scripts/*.py                                 (CLI orchestrators)

Modules
-------
data_loader        I/O helpers for raw and processed datasets.
preprocessing      Phase 2 transformations (unit/frequency harmonisation,
                   NaN handling, wide-format assembly, schema generation).
stationarity       Phase 3 Task 1: ADF + KPSS joint protocol, 4-quadrant
                   classification, transformation dispatch.
structural_breaks  Phase 3 Task 2: Chow test (classical / HAC / COVID-dummy),
                   per-coefficient decomposition, Quandt-Andrews sup-Wald.

Version history
---------------
0.1.0  Initial package scaffolding (Phase 1).
0.2.0  Added preprocessing module and expanded data_loader (Phase 2).
0.3.0  Added stationarity and structural_breaks modules (Phase 3).
"""
from __future__ import annotations

__version__ = "0.3.0"


# ──────────────────────────────────────────────────────────────────
# Phase 1 / Phase 2 re-exports (existing)
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
# Phase 3 re-exports (new in v0.3.0)
# ──────────────────────────────────────────────────────────────────
from .stationarity import (      # noqa: F401
    # Constants
    DEFAULT_ALPHA,
    ADF_REGRESSION_LEVEL,
    TRANSFORM_FN,
    FOUR_QUADRANT_CLASSES,
    FLAGGED_CLASSES,
    CONFLICT_CLASS,
    SMALL_SAMPLE_WARN,
    SMALL_SAMPLE_HARD,
    # Transforms
    first_difference,
    second_difference,
    yoy_pct,
    log_first_diff_pct,
    apply_transform,
    strip_suffix,
    # Testing
    schwert_maxlag,
    run_adf,
    run_kpss,
    classify_4quadrant,
    test_series,
    test_all_series,
)

from .structural_breaks import (  # noqa: F401
    # Constants
    DEFAULT_HAC_LAG,
    KNOWN_BREAKS,
    COVID_DUMMY_START,
    COVID_DUMMY_END,
    COVID_DUMMY_BREAKS,
    ANDREWS_1993_TABLE_I,
    # Dummies
    make_split_dummy,
    make_covid_dummy,
    # Chow tests
    chow_test_classical,
    chow_test_hac,
    chow_test_covid_dummy,
    # Decomposition
    coefficient_decomposition,
    # Quandt-Andrews
    wald_at_break,
    quandt_andrews_scan,
    summarise_scan,
    align_argmax_to_known,
)


__all__ = [
    # Package meta
    "__version__",
    # Data access (data_loader)
    "find_project_root",
    "MAIN_COUNTRIES",
    "SUPPLEMENTARY_COUNTRIES",
    "INDICATORS",
    "load_raw_series",
    "load_all_raw",
    "load_processed_main",
    "load_processed_all_main",
    "load_processed_china",
    # Phase 2 (preprocessing)
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
]
