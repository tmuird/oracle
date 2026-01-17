"""
RamanLib - Library for Raman spectroscopy data analysis.

Modules:
    core: SpectralData class and preprocessing utilities
    plot: Spectral visualization
    utils: Positional encoding and general utilities
    bleaching: Photobleaching decomposition for fluorescence removal
    loading: Data loading utilities for ATCC and simple datasets
"""

from ramanlib.core import SpectralData
from ramanlib.plot import compare_spectra
from ramanlib.utils import wavenumber_positional_encoding, add_positional_encoding

from ramanlib.loading import (
    StrainDataset,
    SimpleDataset,
    load_data,
    load_simple_data,
)

__all__ = [
    # Core
    "SpectralData",
    "compare_spectra",
    "wavenumber_positional_encoding",
    "add_positional_encoding",
    # Loading
    "StrainDataset",
    "SimpleDataset",
    "load_data",
    "load_simple_data",
]

__version__ = "0.1.0"
