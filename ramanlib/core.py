"""
Flexible spectral plotting utilities supporting multiple data formats.

Includes:
- SpectralData: Unified data representation with preprocessing
- convert_to_spectral_data: Format conversion
- compare_spectra: Flexible plotting
- Positional encoding: Wavenumber-based encoding for ML models
"""

from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass

try:
    import ramanspy as rp

    HAS_RAMANSPY = True
except ImportError:
    HAS_RAMANSPY = False
try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@dataclass
class SpectralData:
    """
    Unified representation of spectral data with RamanSPy preprocessing support.

    Acts as a lightweight wrapper that can interface with RamanSPy preprocessing
    while supporting per-sample wavenumber axes (unlike RamanSPy SpectralContainer).

    Attributes
    ----------
    intensities : np.ndarray
        Shape (n_samples, n_wavenumbers) - spectral intensities
    wavenumbers : np.ndarray
        Shape (n_wavenumbers,) for shared axis, or
        Shape (n_samples, n_wavenumbers) for per-sample axes
    label : str
        Optional label for this dataset

    Examples
    --------
    >>> # Create and preprocess
    >>> data = SpectralData(intensities, wavenumbers, label='Raw')
    >>> processed = data.baseline_correction().normalize().crop(600, 1800)
    >>>
    >>> # Plot comparison
    >>> compare_spectra([data, processed], titles=['Raw', 'Processed'])
    """

    intensities: np.ndarray
    wavenumbers: np.ndarray
    label: Optional[str] = None

    def get_spectrum(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get wavenumber axis and intensity for a single sample.

        Returns
        -------
        wavenumber : np.ndarray
            Wavenumber axis for this sample
        intensity : np.ndarray
            Intensity values for this sample
        """
        intensity = self.intensities[sample_idx]

        if self.wavenumbers.ndim == 1:
            # Single shared axis
            wavenumber = self.wavenumbers
        else:
            # Per-sample axes
            wavenumber = self.wavenumbers[sample_idx]

        return wavenumber, intensity

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.intensities.shape[0]

    @property
    def n_wavenumbers(self) -> int:
        """Number of wavenumber points."""
        if self.wavenumbers.ndim == 1:
            return len(self.wavenumbers)
        else:
            return self.wavenumbers.shape[1]

    def to_ramanspy(self):
        """
        Convert to RamanSPy SpectralContainer.

        Note: Only works if all samples share the same wavenumber axis.

        Returns
        -------
        rp.SpectralContainer

        Raises
        ------
        ValueError
            If samples have different wavenumber axes
        """
        if not HAS_RAMANSPY:
            raise ImportError("RamanSPy is not installed")

        if self.wavenumbers.ndim != 1:
            raise ValueError(
                "Cannot convert to RamanSPy: samples have different wavenumber axes"
            )

        return rp.SpectralContainer(
            spectral_data=self.intensities, spectral_axis=self.wavenumbers
        )

    def apply_ramanspy_preprocessing(self, preprocessor) -> "SpectralData":
        """
        Apply any RamanSPy preprocessing and return new SpectralData.

        Parameters
        ----------
        preprocessor : rp.preprocessing object
            Any RamanSPy preprocessing (baseline correction, normalization, etc.)

        Returns
        -------
        SpectralData
            New SpectralData with preprocessed intensities
        """
        if not HAS_RAMANSPY:
            raise ImportError("RamanSPy is required for preprocessing")

        if self.wavenumbers.ndim == 1:
            # Shared axis - process all at once
            container = rp.SpectralContainer(self.intensities, self.wavenumbers)
            processed = preprocessor.apply(container)
            return SpectralData(
                intensities=processed.spectral_data,
                wavenumbers=processed.spectral_axis,
                label=f"{self.label or 'data'} (preprocessed)",
            )
        else:
            # Per-sample axes - process individually
            processed_intensities = []
            for i in range(self.n_samples):
                wn, intensity = self.get_spectrum(i)
                container = rp.SpectralContainer(intensity[np.newaxis, :], wn)
                processed = preprocessor.apply(container)
                processed_intensities.append(processed.spectral_data[0])

            return SpectralData(
                intensities=np.array(processed_intensities),
                wavenumbers=self.wavenumbers,
                label=f"{self.label or 'data'} (preprocessed)",
            )

    def baseline_correction(self, method="iarpls", **kwargs) -> "SpectralData":
        """
        Apply baseline correction using RamanSPy.

        Parameters
        ----------
        method : str
            'iarpls', 'als', 'arpls', 'drpls', 'aspls', 'polynomial', 'morphological'
        **kwargs
            Additional arguments for the baseline correction method

        Returns
        -------
        SpectralData
            Baseline-corrected data
        """
        if not HAS_RAMANSPY:
            raise ImportError("RamanSPy required for baseline correction")

        methods = {
            "iarpls": rp.preprocessing.baseline.IARPLS,
            "als": rp.preprocessing.baseline.ALS,
            "arpls": rp.preprocessing.baseline.ARPLS,
            "drpls": rp.preprocessing.baseline.DRPLS,
            "aspls": rp.preprocessing.baseline.ASPLS,
            "polynomial": rp.preprocessing.baseline.Polynomial,
            "morphological": rp.preprocessing.baseline.Morphological,
        }

        if method.lower() not in methods:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(methods.keys())}"
            )

        corrector = methods[method.lower()](**kwargs)
        return self.apply_ramanspy_preprocessing(corrector)

    def normalize(self, method="minmax", **kwargs) -> "SpectralData":
        """
        Normalize spectra using RamanSPy.

        Parameters
        ----------
        method : str
            'minmax', 'vector', 'area', 'snv'
        **kwargs
            Additional arguments

        Returns
        -------
        SpectralData
            Normalized data
        """
        if not HAS_RAMANSPY:
            raise ImportError("RamanSPy required for normalization")

        methods = {
            "minmax": rp.preprocessing.normalise.MinMax,
            "vector": rp.preprocessing.normalise.Vector,
            "area": rp.preprocessing.normalise.AUC,
            "snv": rp.preprocessing.normalise.SNV,
        }

        if method.lower() not in methods:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(methods.keys())}"
            )

        normalizer = methods[method.lower()](**kwargs)
        return self.apply_ramanspy_preprocessing(normalizer)

    def despike(self, **kwargs) -> "SpectralData":
        """
        Remove cosmic ray spikes using RamanSPy Whitaker-Hayes method.

        Returns
        -------
        SpectralData
            Despiked data
        """
        if not HAS_RAMANSPY:
            raise ImportError("RamanSPy required for despiking")

        despiker = rp.preprocessing.despike.WhitakerHayes(**kwargs)
        return self.apply_ramanspy_preprocessing(despiker)

    def crop(self, wn_min: float, wn_max: float) -> "SpectralData":
        """
        Crop wavenumber range.

        Parameters
        ----------
        wn_min : float
            Minimum wavenumber
        wn_max : float
            Maximum wavenumber

        Returns
        -------
        SpectralData
            Cropped data
        """
        if self.wavenumbers.ndim == 1:
            # Shared axis
            mask = (self.wavenumbers >= wn_min) & (self.wavenumbers <= wn_max)
            return SpectralData(
                intensities=self.intensities[:, mask],
                wavenumbers=self.wavenumbers[mask],
                label=f"{self.label or 'data'} (cropped)",
            )
        else:
            # Per-sample axes
            cropped_intensities = []
            cropped_wavenumbers = []
            for i in range(self.n_samples):
                wn, intensity = self.get_spectrum(i)
                mask = (wn >= wn_min) & (wn <= wn_max)
                cropped_intensities.append(intensity[mask])
                cropped_wavenumbers.append(wn[mask])

            return SpectralData(
                intensities=np.array(cropped_intensities),
                wavenumbers=np.array(cropped_wavenumbers),
                label=f"{self.label or 'data'} (cropped)",
            )

    def copy(self) -> "SpectralData":
        """Create a deep copy."""
        return SpectralData(
            intensities=self.intensities.copy(),
            wavenumbers=self.wavenumbers.copy(),
            label=self.label,
        )

    def normalize_for_plotting(self, method="l2") -> "SpectralData":
        """
        Normalize spectra for visualization.
        This is specifically for making spectra visually comparable in plots,
        not for analysis preprocessing. For preprocessing, use .normalize().

        Parameters
        ----------
        method : str
            'l2': Center and L2 normalize (good for comparing shapes)
            'zscore': Z-score normalization (mean=0, std=1)
            'minmax': Scale to [0, 1] range
            'center': Just center to mean=0

        Returns
        -------
        SpectralData
            Normalized data for plotting
        """
        if method == "l2":
            # Center and L2 normalize - preserves shape, removes scale
            centered = self.intensities - np.mean(
                self.intensities, axis=1, keepdims=True
            )
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            normalized = centered / (norms + 1e-8)

        elif method == "zscore":
            # Z-score: (x - mean) / std
            mean = np.mean(self.intensities, axis=1, keepdims=True)
            std = np.std(self.intensities, axis=1, keepdims=True)
            normalized = (self.intensities - mean) / (std + 1e-8)

        elif method == "minmax":
            # Scale to [0, 1]
            min_val = np.min(self.intensities, axis=1, keepdims=True)
            max_val = np.max(self.intensities, axis=1, keepdims=True)
            normalized = (self.intensities - min_val) / (max_val - min_val + 1e-8)

        elif method == "center":
            # Just remove mean
            normalized = self.intensities - np.mean(
                self.intensities, axis=1, keepdims=True
            )

        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from ['l2', 'zscore', 'minmax', 'center']"
            )

        return SpectralData(
            intensities=normalized,
            wavenumbers=self.wavenumbers,
            label=f"{self.label or 'data'} ({method}-norm)",
        )


def convert_to_spectral_data(
    data: Union[
        SpectralData,
        "rp.SpectralContainer",
        "xr.DataArray",
        "xr.Dataset",
        np.ndarray,
        Dict,
        Tuple,
    ],
    wavenumbers: Optional[np.ndarray] = None,
    intensity_var: str = "intensity_raw",
    wavenumber_var: str = "wavenumber",
    label: Optional[str] = None,
) -> SpectralData:
    """
    Convert various input formats to unified SpectralData.

    This function handles format conversion, NOT data normalization.
    For normalization, use SpectralData.normalize() or .normalize_for_plotting().

    Supports:
    - SpectralData (returns as-is or with updated label)
    - rp.SpectralContainer
    - xr.DataArray (assumes 'wavenumber' dimension)
    - xr.Dataset (extracts intensity_var and wavenumber_var)
    - np.ndarray + wavenumbers parameter
    - dict with 'intensities' and 'wavenumbers' keys
    - tuple of (intensities, wavenumbers)

    Parameters
    ----------
    data : various types
        Spectral data in any supported format
    wavenumbers : np.ndarray, optional
        Required when data is np.ndarray without other context
    intensity_var : str
        Variable name for intensity in xarray.Dataset
    wavenumber_var : str
        Coordinate name for wavenumber in xarray
    label : str, optional
        Label for this dataset

    Returns
    -------
    SpectralData
        Unified spectral data representation
    """
    # Handle SpectralData (already in correct format)
    if isinstance(data, SpectralData):
        if label is not None:
            data.label = label
        return data

    # Handle tuple: (intensities, wavenumbers)
    if isinstance(data, tuple) and len(data) == 2:
        intensities, wn = data
        return SpectralData(
            intensities=np.atleast_2d(intensities),
            wavenumbers=np.asarray(wn),
            label=label,
        )

    # Handle dict: {'intensities': ..., 'wavenumbers': ...}
    if isinstance(data, dict):
        if "intensities" not in data or "wavenumbers" not in data:
            raise ValueError("Dict must contain 'intensities' and 'wavenumbers' keys")
        return SpectralData(
            intensities=np.atleast_2d(data["intensities"]),
            wavenumbers=np.asarray(data["wavenumbers"]),
            label=label or data.get("label"),
        )

    # Handle RamanSPy SpectralContainer
    if HAS_RAMANSPY and isinstance(data, rp.SpectralContainer):
        return SpectralData(
            intensities=data.spectral_data, wavenumbers=data.spectral_axis, label=label
        )

    # Handle xarray DataArray
    if HAS_XARRAY and isinstance(data, xr.DataArray):
        if wavenumber_var not in data.dims:
            raise ValueError(f"xarray.DataArray must have '{wavenumber_var}' dimension")

        # Check if wavenumber is a coordinate or dimension
        if wavenumber_var in data.coords:
            wn = data.coords[wavenumber_var].values
        else:
            wn = data[wavenumber_var].values

        return SpectralData(
            intensities=data.values, wavenumbers=wn, label=label or data.name
        )

    # Handle xarray Dataset
    if HAS_XARRAY and isinstance(data, xr.Dataset):
        if intensity_var not in data.data_vars:
            raise ValueError(f"xarray.Dataset must contain '{intensity_var}' variable")
        if wavenumber_var not in data.coords:
            raise ValueError(f"xarray.Dataset must have '{wavenumber_var}' coordinate")

        return SpectralData(
            intensities=data[intensity_var].values,
            wavenumbers=data.coords[wavenumber_var].values,
            label=label or intensity_var,
        )

    # Handle numpy array
    if isinstance(data, np.ndarray):
        if wavenumbers is None:
            raise ValueError("wavenumbers parameter required when using numpy arrays")

        # Ensure 2D
        intensities = np.atleast_2d(data)
        if intensities.ndim > 2:
            raise ValueError(
                f"Intensity array must be 1D or 2D, got shape {data.shape}"
            )

        return SpectralData(
            intensities=intensities, wavenumbers=np.asarray(wavenumbers), label=label
        )

    raise TypeError(f"Unsupported input type: {type(data)}")


# def compare_encoding_methods(
#     wavenumbers: np.ndarray, d_model: int = 6, figsize: Tuple[int, int] = (14, 10)
# ) -> Figure:
#     """
#     Compare index-based vs wavenumber-based positional encoding.

#     Parameters
#     ----------
#     wavenumbers : np.ndarray
#         Wavenumber axis
#     d_model : int
#         Encoding dimension
#     figsize : tuple
#         Figure size

#     Returns
#     -------
#     Figure
#         Comparison plot

#     Examples
#     --------
#     >>> wn = np.linspace(400, 1800, 736)
#     >>> fig = compare_encoding_methods(wn)
#     >>> plt.show()
#     """
#     # Index-based encoding
#     n = len(wavenumbers)
#     indices = np.arange(n)
#     pe_index = wavenumber_positional_encoding(
#         indices, d_model, scale=1.0, normalize=False
#     )

#     # Wavenumber-based encoding (normalized)
#     pe_wn_norm = wavenumber_positional_encoding(
#         wavenumbers, d_model, scale=1.0, normalize=True
#     )

#     # Wavenumber-based encoding (raw)
#     pe_wn_raw = wavenumber_positional_encoding(
#         wavenumbers, d_model, scale=1.0, normalize=False
#     )

#     fig, axes = plt.subplots(d_model, 3, figsize=figsize, sharex="col")

#     for i in range(d_model):
#         # Index-based
#         axes[i, 0].plot(indices, pe_index[:, i], linewidth=1)
#         axes[i, 0].set_ylabel(f"Dim {i}", fontsize=9)
#         axes[i, 0].grid(True, alpha=0.3)
#         if i == 0:
#             axes[i, 0].set_title("Index-Based\n(0, 1, 2, ...)", fontsize=10)

#         # Wavenumber normalized
#         axes[i, 1].plot(wavenumbers, pe_wn_norm[:, i], linewidth=1)
#         axes[i, 1].grid(True, alpha=0.3)
#         if i == 0:
#             axes[i, 1].set_title("Wavenumber-Based\n(Normalized)", fontsize=10)

#         # Wavenumber raw
#         axes[i, 2].plot(wavenumbers, pe_wn_raw[:, i], linewidth=1)
#         axes[i, 2].grid(True, alpha=0.3)
#         if i == 0:
#             axes[i, 2].set_title("Wavenumber-Based\n(Raw Scale)", fontsize=10)

#     axes[-1, 0].set_xlabel("Index", fontsize=10)
#     axes[-1, 1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
#     axes[-1, 2].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)

#     fig.suptitle(f"Positional Encoding Comparison (d_model={d_model})", fontsize=13)
#     plt.tight_layout()

#     return fig
