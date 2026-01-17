"""
Data loading utilities for Raman spectroscopy.

Provides loaders for:
- ATCC bacterial Raman spectra (StrainDataset, load_data)
- Simple folder-based spectra (SimpleDataset, load_simple_data)
"""

from ramanlib.loading.strain import (
    StrainDataset,
    load_data,
    DEFAULT_STRAIN_INFO,
)

from ramanlib.loading.simple import (
    SimpleDataset,
    load_simple_data,
)

from ramanlib.loading.base import (
    detect_saturation,
    detect_dropout,
    preprocess_for_outlier_detection,
    compute_snr,
    add_normalized_intensity,
    filter_misaligned_pairs,
    detect_outliers_iqr,
    detect_outliers_mad,
    detect_outliers_zscore,
)

__all__ = [
    # Main classes
    "StrainDataset",
    "SimpleDataset",
    # Loading functions
    "load_data",
    "load_simple_data",
    # Constants
    "DEFAULT_STRAIN_INFO",
    # Utilities
    "detect_saturation",
    "detect_dropout",
    "preprocess_for_outlier_detection",
    "compute_snr",
    "add_normalized_intensity",
    "filter_misaligned_pairs",
    "detect_outliers_iqr",
    "detect_outliers_mad",
    "detect_outliers_zscore",
]
