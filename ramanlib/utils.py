from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass
import ramanspy as rp
import xarray as xr
from ramanlib.core import SpectralData, convert_to_spectral_data


def wavenumber_positional_encoding(
    wavenumbers: np.ndarray,
    d_model: int,
    scale: float = 0.1,
    freq_scale: float = 1000,
    normalize: bool = True,
    global_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Generate sinusoidal positional encoding from wavenumber values.

    This encodes the actual wavenumber positions (e.g., 400-1800 cm⁻¹)
    rather than array indices, preserving the physical meaning of the
    spectral axis.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber axis values, shape (n_wavenumbers,)
    d_model : int
        Embedding dimension (must be even)
    scale : float
        Output scaling factor
    normalize : bool
        If True, normalize wavenumbers to [0, 1] range before encoding.
        Recommended for typical positional encoding frequencies.
    global_range : tuple, optional
        (min, max) wavenumber range for normalization.
        IMPORTANT: Use global min/max across ALL samples to preserve
        calibration differences between samples. If None, uses local min/max.

    Returns
    -------
    np.ndarray
        Positional encoding, shape (n_wavenumbers, d_model)

    Examples
    --------
    >>> wn = np.linspace(400, 1800, 736)
    >>> pe = wavenumber_positional_encoding(wn, d_model=6, normalize=True)
    >>> print(pe.shape)
    (736, 6)

    >>> # For per-sample axes with global normalization:
    >>> global_range = (400.0, 1800.5)  # Global min/max
    >>> pe = wavenumber_positional_encoding(wn, d_model=6, global_range=global_range)

    Notes
    -----
    Uses the standard transformer positional encoding:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where pos = wavenumber (optionally normalized to [0, 1])
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    # Normalize wavenumbers to [0, 1] if requested
    if normalize:
        if global_range is not None:
            wn_min, wn_max = global_range
        else:
            wn_min, wn_max = wavenumbers.min(), wavenumbers.max()

        if wn_max == wn_min:
            position = np.zeros_like(wavenumbers)[:, np.newaxis]
        else:
            position = (wavenumbers - wn_min) / (wn_max - wn_min)
            position = position[:, np.newaxis]
    else:
        position = wavenumbers[:, np.newaxis]

    if normalize:
        position = position * freq_scale
    # Frequency scaling factors
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Sinusoidal encoding
    pe = np.zeros((len(wavenumbers), d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe * scale


def add_positional_encoding(
    data: Union[SpectralData, "rp.SpectralContainer", np.ndarray],
    wavenumbers: Optional[np.ndarray] = None,
    d_model: int = 6,
    scale: float = 0.1,
    freq_scale: float = 1000,
    normalize: bool = True,
    per_sample: bool = False,
) -> np.ndarray:
    """
    Add wavenumber positional encoding to spectral data.

    Expands spectral intensities from (n_samples, n_wavenumbers) to
    (n_samples, n_wavenumbers, 1 + d_model) by concatenating the
    intensity channel with d_model positional encoding channels.

    Parameters
    ----------
    data : SpectralData, rp.SpectralContainer, or np.ndarray
        Spectral intensities
    wavenumbers : np.ndarray, optional
        Wavenumber axis. Required if data is np.ndarray.
        Can be shape (n_wavenumbers,) for shared axis or
        (n_samples, n_wavenumbers) for per-sample axes.
    d_model : int
        Positional encoding dimension (must be even)
    scale : float
        PE scaling factor
    normalize : bool
        Normalize wavenumbers before encoding
    per_sample : bool
        If True and wavenumbers vary per sample, generate PE per sample

    Returns
    -------
    np.ndarray
        Expanded data, shape (n_samples, n_wavenumbers, 1 + d_model)
        - Channel 0: Original intensities
        - Channels 1 to d_model: Positional encoding

    Examples
    --------
    >>> # With SpectralData
    >>> data = SpectralData(intensities, wavenumbers)
    >>> data_pe = add_positional_encoding(data, d_model=6)
    >>> print(data_pe.shape)  # (n_samples, n_wavenumbers, 7)

    >>> # With numpy arrays
    >>> intensities = np.random.randn(100, 736)
    >>> wavenumbers = np.linspace(400, 1800, 736)
    >>> data_pe = add_positional_encoding(intensities, wavenumbers, d_model=6)

    >>> # With per-sample wavenumber axes
    >>> data = SpectralData(intensities, per_sample_wavenumbers)
    >>> data_pe = add_positional_encoding(data, d_model=6, per_sample=True)
    """
    # Extract intensities and wavenumbers
    if isinstance(data, SpectralData):
        intensities = data.intensities
        wn = data.wavenumbers
    elif HAS_RAMANSPY and isinstance(data, rp.SpectralContainer):
        intensities = data.spectral_data
        wn = data.spectral_axis
    elif isinstance(data, np.ndarray):
        if wavenumbers is None:
            raise ValueError("wavenumbers required when using numpy arrays")
        intensities = data
        wn = wavenumbers
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Ensure 2D intensities
    if intensities.ndim == 1:
        intensities = intensities[np.newaxis, :]

    n_samples, n_wavenumbers = intensities.shape

    # Expand intensities to (n_samples, n_wavenumbers, 1)
    expanded = intensities[..., np.newaxis]

    # Compute global range for normalization
    if normalize and wn.ndim == 2:
        global_min = np.min(wn)
        global_max = np.max(wn)
        global_range = (global_min, global_max)
        print(
            f"Computed global wavenumber range: [{global_min:.2f}, {global_max:.2f}] cm⁻¹"
        )
    else:
        global_range = None

    # Generate positional encoding
    if wn.ndim == 2 and per_sample:
        # Per-sample wavenumber axes with global normalization
        print("Generating per-sample positional encodings with global normalization...")
        pe_all = np.zeros((n_samples, n_wavenumbers, d_model))
        for i in range(n_samples):
            pe_all[i] = wavenumber_positional_encoding(
                wn[i], d_model, scale, freq_scale, normalize, global_range
            )
    else:
        # Shared wavenumber axis

        if wn.ndim == 2:
            print(
                "Using shared wavenumber axis from first sample for positional encoding."
            )
            # Use first sample's axis as representative
            wn = wn[0]
        pe = wavenumber_positional_encoding(wn, d_model, scale, freq_scale, normalize, global_range)
        pe_all = np.broadcast_to(pe[None, :, :], (n_samples, n_wavenumbers, d_model))

    # Concatenate intensity + PE
    return np.concatenate([expanded, pe_all], axis=-1)
