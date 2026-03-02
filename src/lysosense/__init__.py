"""LysoSense helper package.

This package provides tools for analyzing differential centrifugal sedimentation (DCS/CPS)
data and flow cytometry (FACS) data from E. coli homogenization campaigns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# CPS/DCS analysis
from .io import Measurement, list_dat_files, load_dat_file, parse_dat_bytes
from .analysis import AnalysisOptions, AnalysisResult, analyze_measurement

# FACS analysis (optional import)
try:
    from .facs import (
        FCSMeasurement,
        SYTOXAnalysisConfig,
        SYTOXAnalysisResult,
        FACSConfig,
        read_fcs,
        analyze_sytox_data,
        create_fsc_ssc_plots,
        create_sytox_plots,
    )

    _FACS_AVAILABLE = True
except ImportError:
    # FACS dependencies not available
    _FACS_AVAILABLE = False
    if TYPE_CHECKING:
        from .facs import (
            FCSMeasurement as FCSMeasurement,
            SYTOXAnalysisConfig as SYTOXAnalysisConfig,
            SYTOXAnalysisResult as SYTOXAnalysisResult,
            FACSConfig as FACSConfig,
            read_fcs as read_fcs,
            analyze_sytox_data as analyze_sytox_data,
            create_fsc_ssc_plots as create_fsc_ssc_plots,
            create_sytox_plots as create_sytox_plots,
        )
    else:
        FCSMeasurement: Any = None  # type: ignore[misc,assignment]
        SYTOXAnalysisConfig: Any = None  # type: ignore[misc,assignment]
        SYTOXAnalysisResult: Any = None  # type: ignore[misc,assignment]
        FACSConfig: Any = None  # type: ignore[misc,assignment]
        read_fcs: Any = None  # type: ignore[assignment]
        analyze_sytox_data: Any = None  # type: ignore[assignment]
        create_fsc_ssc_plots: Any = None  # type: ignore[assignment]
        create_sytox_plots: Any = None  # type: ignore[assignment]

__all__ = [
    # CPS/DCS analysis
    "Measurement",
    "AnalysisOptions",
    "AnalysisResult",
    "list_dat_files",
    "load_dat_file",
    "parse_dat_bytes",
    "analyze_measurement",
    # FACS analysis (if available)
    "FCSMeasurement",
    "SYTOXAnalysisConfig",
    "SYTOXAnalysisResult",
    "FACSConfig",
    "read_fcs",
    "analyze_sytox_data",
    "create_fsc_ssc_plots",
    "create_sytox_plots",
]
