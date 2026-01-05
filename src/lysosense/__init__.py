"""LysoSense helper package.

This package provides tools for analyzing differential centrifugal sedimentation (DCS/CPS)
data and flow cytometry (FACS) data from E. coli homogenization campaigns.
"""
from __future__ import annotations

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
        create_sytox_plots
    )
    _FACS_AVAILABLE = True
except ImportError:
    # FACS dependencies not available
    _FACS_AVAILABLE = False
    FCSMeasurement = None
    SYTOXAnalysisConfig = None
    SYTOXAnalysisResult = None
    FACSConfig = None
    read_fcs = None
    analyze_sytox_data = None
    create_fsc_ssc_plots = None
    create_sytox_plots = None

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
