from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass
import ramanspy as rp
import xarray as xr
from ramanlib.core import SpectralData, convert_to_spectral_data


def compare_spectra(
    data: List[
        Union[
            SpectralData,
            "rp.SpectralContainer",
            "xr.DataArray",
            "xr.Dataset",
            np.ndarray,
            Dict,
            Tuple,
        ]
    ],
    wavenumbers: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    n_samples: int = 1,
    width: int = 30,
    height: int = 20,
    overlay: bool = True,
    separate_samples: bool = False,
    legend: bool = True,
    titles: Optional[List[str]] = None,
    plot_mean: bool = False,
    alphas: Optional[List[float]] = None,
    crop: Optional[Tuple[float, float]] = None,
    intensity_var: str = "intensity_raw",
    wavenumber_var: str = "wavenumber",
) -> Union[Tuple[Figure, Axes], Tuple[Figure, np.ndarray]]:
    """
    Compare multiple spectral datasets with flexible layout options.

    Supports multiple input formats:
    - SpectralData objects (recommended - supports preprocessing chains!)
    - RamanSPy SpectralContainers
    - xarray DataArrays or Datasets
    - numpy arrays (with wavenumbers parameter)
    - dicts: {'intensities': array, 'wavenumbers': array}
    - tuples: (intensities, wavenumbers)

    Each dataset can have:
    - Shared wavenumber axis for all samples, or
    - Per-sample wavenumber axes (for datasets with varying calibration)

    Layout modes:
    - overlay=True, separate_samples=False: All datasets and samples on single axes
    - overlay=True, separate_samples=True: All datasets on one axes per sample
    - overlay=False, separate_samples=False: One subplot per dataset, all samples per subplot
    - overlay=False, separate_samples=True: One figure per sample with one subplot per dataset

    Parameters
    ----------
    data : list
        List of spectral data in any supported format
    wavenumbers : np.ndarray or list of np.ndarray, optional
        Wavenumber axis/axes. Required for numpy array inputs.
        Can be single array (shared) or list of arrays (per-dataset)
    n_samples : int
        Number of spectra to plot from each dataset
    width : int
        Figure width in inches
    height : int
        Figure height in inches
    overlay : bool
        If True, plot all datasets on same axes
    separate_samples : bool
        If True, create separate plots for each sample
    titles : list of str, optional
        Titles for each dataset
    plot_mean : bool
        If True, plot mean spectrum for each dataset
    alphas : list of float, optional
        Alpha values for each dataset
    crop : tuple of (min, max), optional
        Crop wavenumber range
    intensity_var : str
        Variable name for xarray.Dataset intensity
    wavenumber_var : str
        Coordinate name for xarray wavenumbers

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes or ndarray of Axes
        Matplotlib axes

    Examples
    --------
    # Compare SpectralData with preprocessing chain (recommended!)
    >>> raw = SpectralData(intensities, wavenumbers, label='Raw')
    >>> processed = raw.baseline_correction().normalize().crop(600, 1800)
    >>> compare_spectra([raw, processed])

    # Compare RamanSPy containers
    >>> compare_spectra([container1, container2], titles=['Raw', 'Processed'])

    # Compare numpy arrays with shared axis
    >>> compare_spectra(
    ...     [data1, data2],
    ...     wavenumbers=shared_axis,
    ...     titles=['Sample 1', 'Sample 2']
    ... )

    # Compare xarray datasets (e.g., from ATCCLoader)
    >>> compare_spectra(
    ...     [atcc_ds.sel(bleaching_time='0s'), atcc_ds.sel(bleaching_time='10s')],
    ...     intensity_var='intensity_clean',
    ...     titles=['t=0s', 't=10s']
    ... )

    # Mixed formats
    >>> compare_spectra(
    ...     [spectral_data, container, (numpy_data, numpy_axis), xr_data],
    ...     titles=['SpectralData', 'RamanSPy', 'NumPy', 'xarray']
    ... )
    """
    n_datasets = len(data)

    # Validate titles
    if titles is not None and len(titles) != n_datasets:
        raise ValueError(
            f"Number of titles ({len(titles)}) must match number of datasets ({n_datasets})"
        )

    # Handle wavenumbers parameter for numpy arrays
    if wavenumbers is not None:
        # Check if it's a list of arrays (per-dataset) or single array (shared)
        if isinstance(wavenumbers, list):
            if len(wavenumbers) != n_datasets:
                raise ValueError(
                    f"Number of wavenumber arrays ({len(wavenumbers)}) must match datasets ({n_datasets})"
                )
            wn_list = wavenumbers
        else:
            # Single shared wavenumber array for all datasets
            wn_list = [wavenumbers] * n_datasets
    else:
        wn_list = [None] * n_datasets

    # Convert all inputs to SpectralData
    spectral_data: List[SpectralData] = []
    for i, (d, wn) in enumerate(zip(data, wn_list)):
        label = titles[i] if titles else None
        spec = convert_to_spectral_data(
            d,
            wavenumbers=wn,
            intensity_var=intensity_var,
            wavenumber_var=wavenumber_var,
            label=label,
        )
        spectral_data.append(spec)

    # Apply cropping if specified
    if crop is not None:
        cropped_data = []
        for spec in spectral_data:
            # Crop based on shared or per-sample axes
            if spec.wavenumbers.ndim == 1:
                # Shared axis - crop once
                mask = (spec.wavenumbers >= crop[0]) & (spec.wavenumbers <= crop[1])
                cropped_wn = spec.wavenumbers[mask]
                cropped_intensities = spec.intensities[:, mask]
            else:
                # Per-sample axes - crop each sample
                cropped_intensities = []
                cropped_wn = []
                for sample_idx in range(spec.n_samples):
                    wn, intensity = spec.get_spectrum(sample_idx)
                    mask = (wn >= crop[0]) & (wn <= crop[1])
                    cropped_wn.append(wn[mask])
                    cropped_intensities.append(intensity[mask])
                cropped_intensities = np.array(cropped_intensities)
                cropped_wn = np.array(cropped_wn)

            cropped_data.append(
                SpectralData(
                    intensities=cropped_intensities,
                    wavenumbers=cropped_wn,
                    label=spec.label,
                )
            )
        spectral_data = cropped_data

    # Limit samples
    n_samples = min(n_samples, min(spec.n_samples for spec in spectral_data))

    # Set default alphas
    if alphas is None:
        alphas = [0.7] * n_datasets

    # --- Plotting Logic ---

    if overlay:
        if separate_samples:
            # Each sample gets its own figure with all datasets overlaid
            for sample_idx in range(n_samples):
                fig, ax = plt.subplots(figsize=(width, height))
                ax.set_title(f"Spectral Comparison - Sample {sample_idx}")
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.grid(True, alpha=0.3)

                for dataset_idx, spec in enumerate(spectral_data):
                    wn, intensity = spec.get_spectrum(sample_idx)
                    label = spec.label or f"Dataset {dataset_idx}"
                    ax.plot(wn, intensity, alpha=alphas[dataset_idx], label=label)

                if legend:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                plt.show()

            return fig, ax
        else:
            # Single figure with all datasets and all samples overlaid
            fig, ax = plt.subplots(figsize=(width, height))
            ax.set_title("Spectral Comparison")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.grid(True, alpha=0.3)

            for dataset_idx, spec in enumerate(spectral_data):
                dataset_label = spec.label or f"Dataset {dataset_idx}"

                for sample_idx in range(n_samples):
                    wn, intensity = spec.get_spectrum(sample_idx)
                    label = (
                        f"{dataset_label} - Sample {sample_idx}"
                        if n_samples > 1
                        else dataset_label
                    )
                    ax.plot(wn, intensity, label=label, alpha=alphas[dataset_idx])

                if plot_mean:
                    # Compute mean (only if all samples share same axis)
                    if spec.wavenumbers.ndim == 1:
                        mean_intensity = spec.intensities[:n_samples].mean(axis=0)
                        ax.plot(
                            spec.wavenumbers,
                            mean_intensity,
                            label=f"{dataset_label} - Mean",
                            linestyle="--",
                            linewidth=2,
                        )
            if legend:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            return fig, ax
    else:
        if separate_samples:
            # Each sample gets its own figure with one subplot per dataset
            for sample_idx in range(n_samples):
                fig, axes = plt.subplots(
                    nrows=n_datasets, ncols=1, figsize=(width, height), squeeze=False
                )

                for dataset_idx, spec in enumerate(spectral_data):
                    ax = axes[dataset_idx, 0]
                    title = spec.label or f"Dataset {dataset_idx}"
                    ax.set_title(f"{title} - Sample {sample_idx}")
                    ax.set_xlabel("Raman Shift (cm⁻¹)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.grid(True, alpha=0.3)

                    wn, intensity = spec.get_spectrum(sample_idx)
                    ax.plot(wn, intensity, alpha=alphas[dataset_idx])

                plt.tight_layout()
                plt.show()

            return fig, axes.flatten()
        else:
            # Single figure with one subplot per dataset, all samples per subplot
            fig, axes = plt.subplots(
                nrows=n_datasets, ncols=1, figsize=(width, height), squeeze=False
            )

            for dataset_idx, spec in enumerate(spectral_data):
                ax = axes[dataset_idx, 0]
                title = spec.label or f"Dataset {dataset_idx}"
                ax.set_title(title)
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.grid(True, alpha=0.3)

                for sample_idx in range(n_samples):
                    wn, intensity = spec.get_spectrum(sample_idx)
                    ax.plot(
                        wn,
                        intensity,
                        label=f"Sample {sample_idx}",
                        alpha=alphas[dataset_idx],
                    )

                if plot_mean and spec.wavenumbers.ndim == 1:
                    mean_intensity = spec.intensities[:n_samples].mean(axis=0)
                    ax.plot(
                        spec.wavenumbers,
                        mean_intensity,
                        label="Mean",
                        linestyle="--",
                        linewidth=2,
                        color="black",
                    )
                if legend:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            return fig, axes.flatten()


def visualize_positional_encoding(
    wavenumbers: np.ndarray,
    d_model: int = 6,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Visualize positional encoding patterns.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber axis
    d_model : int
        Encoding dimension
    normalize : bool
        Whether to normalize wavenumbers
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure showing all PE dimensions

    Examples
    --------
    >>> wn = np.linspace(400, 1800, 736)
    >>> fig = visualize_positional_encoding(wn, d_model=6)
    >>> plt.show()
    """
    pe = wavenumber_positional_encoding(
        wavenumbers, d_model, scale=1.0, normalize=normalize
    )

    fig, axes = plt.subplots(d_model, 1, figsize=figsize, sharex=True)

    for i in range(d_model):
        axes[i].plot(wavenumbers, pe[:, i], linewidth=1)
        axes[i].set_ylabel(f"PE dim {i}", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)

    title = "Normalized" if normalize else "Raw"
    fig.suptitle(
        f"{title} Wavenumber Positional Encoding (d_model={d_model})",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout()

    return fig
