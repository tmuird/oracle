"""
SimpleDataset for basic Raman spectroscopy data.

Handles simple folder structure with wavenumber file and spectra files
at different integration times.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from ramanlib.loading.base import (
    detect_saturation,
    detect_dropout,
    preprocess_for_outlier_detection,
    compute_snr,
    add_normalized_intensity,
)

try:
    import ramanspy as rp
    HAS_RAMANSPY = True
except ImportError:
    HAS_RAMANSPY = False


class SimpleDataset:
    """
    Container for simple Raman spectroscopy data with multiple integration times.

    Simpler structure than StrainDataset - all samples share the same wavenumber axis.
    """

    def __init__(self, spectral_axis: np.ndarray):
        """
        Parameters
        ----------
        spectral_axis : np.ndarray
            Shared wavenumber axis for all spectra
        """
        self.spectral_axis = spectral_axis
        self.integration_times: List[str] = []
        self.spectral_data: Dict[str, np.ndarray] = {}
        self.background_data: Dict[str, np.ndarray] = {}

    def add_integration_time(
        self,
        integration_time: str,
        spectra: np.ndarray,
        background: Optional[np.ndarray] = None,
    ):
        """
        Add spectra for an integration time.

        Parameters
        ----------
        integration_time : str
            Integration time label (e.g., '1s', '0.5s')
        spectra : np.ndarray
            Spectra array, shape (n_samples, n_wavenumbers)
        background : np.ndarray, optional
            Background spectra for subtraction
        """
        if integration_time in self.spectral_data:
            raise ValueError(f"Integration time {integration_time} already added")

        self.integration_times.append(integration_time)
        self.spectral_data[integration_time] = spectra.astype(np.float32)

        if background is not None:
            self.background_data[integration_time] = background.astype(np.float32)

    def apply_background_subtraction(self):
        """Subtract mean background from spectra for all integration times."""
        for int_time in self.integration_times:
            if int_time in self.background_data:
                bg_mean = self.background_data[int_time].mean(axis=0)
                self.spectral_data[int_time] = self.spectral_data[int_time] - bg_mean
                print(f"Background subtracted for {int_time}")

    def summary(self):
        """Print dataset summary."""
        print(f"\nSimple Dataset Summary:")
        print(f"  Spectral axis: {len(self.spectral_axis)} points, "
              f"range [{self.spectral_axis.min():.2f}, {self.spectral_axis.max():.2f}] cm⁻¹")
        print(f"  Integration times: {len(self.integration_times)}")
        for int_time in sorted(self.integration_times):
            n_samples = self.spectral_data[int_time].shape[0]
            has_bg = "with background" if int_time in self.background_data else "no background"
            print(f"    {int_time}: {n_samples} samples ({has_bg})")

    def to_xarray(
        self,
        remove_incomplete: bool = False,
        normalise: bool = False,
        crop: Optional[Tuple[float, float]] = None,
        remove_outliers: bool = False,
        baseline_correction: bool = False,
        despike: bool = False,
        method: str = "l2",
        outlier_mad_threshold: float = 10.0,
        aggressive_snr_filter: bool = False,
        snr_reference_time: Optional[str] = None,
        snr_quantile_filter: Tuple[float, float] = (0.25, 0.75),
        snr_silent_region: Tuple[float, float] = (1800, 1900),
    ) -> xr.Dataset:
        """
        Convert to xarray Dataset with (sample, integration_time, wavenumber) structure.

        Parameters
        ----------
        remove_incomplete : bool
            Not used (all samples have same times in SimpleDataset)
        normalise : bool
            Apply normalization
        crop : tuple, optional
            (min, max) wavenumber range to crop
        remove_outliers : bool
            Remove saturated/dropout spectra
        baseline_correction : bool
            Apply polynomial baseline correction
        despike : bool
            Apply Whittaker-Hayes despiking
        method : str
            Normalization method: 'l2' or 'percentile'
        outlier_mad_threshold : float
            MAD threshold for dropout detection
        aggressive_snr_filter : bool
            Filter by SNR quantiles
        snr_reference_time : str, optional
            Integration time for SNR calculation
        snr_quantile_filter : tuple
            (lower, upper) quantile bounds for SNR filter
        snr_silent_region : tuple
            Wavenumber range for noise estimation

        Returns
        -------
        xr.Dataset
            Dataset with intensity and coordinates
        """
        if not self.integration_times:
            raise ValueError("No data available - add integration times first")

        if aggressive_snr_filter and crop is not None:
            if snr_silent_region[0] < crop[0] or snr_silent_region[1] > crop[1]:
                raise ValueError(
                    f"SNR silent region {snr_silent_region} outside crop range {crop}."
                )

        sorted_times = sorted(self.integration_times)
        max_samples = max(self.spectral_data[t].shape[0] for t in sorted_times)

        spectral_axis = self.spectral_axis.copy()
        if crop is not None:
            print(f"\n=== Cropping Spectra: {crop[0]} - {crop[1]} cm⁻¹ ===")
            wn_mask = (spectral_axis >= crop[0]) & (spectral_axis <= crop[1])
            spectral_axis = spectral_axis[wn_mask]
        else:
            wn_mask = np.ones(len(spectral_axis), dtype=bool)

        n_wavenumbers = len(spectral_axis)
        n_times = len(sorted_times)

        sample_indices = list(range(max_samples))
        outlier_mask = np.zeros(max_samples, dtype=bool)
        snr_dict = {}

        # Quality filtering
        if remove_outliers or aggressive_snr_filter:
            print("\n=== Quality Filtering ===")
            ref_time = sorted_times[0]
            ref_data = self.spectral_data[ref_time][:, wn_mask] if crop else self.spectral_data[ref_time]

            for sample_idx in range(ref_data.shape[0]):
                spectrum_raw = ref_data[sample_idx]

                if remove_outliers:
                    if detect_saturation(spectrum_raw):
                        outlier_mask[sample_idx] = True
                        continue

                    spectrum_processed = preprocess_for_outlier_detection(
                        spectrum_raw, spectral_axis, baseline_correction
                    )
                    if detect_dropout(spectrum_processed, outlier_mad_threshold):
                        outlier_mask[sample_idx] = True
                        continue

                if aggressive_snr_filter:
                    snr_value = compute_snr(spectrum_raw, spectral_axis, snr_silent_region)
                    snr_dict[sample_idx] = snr_value

            if remove_outliers:
                n_removed = np.sum(outlier_mask)
                print(f"Outliers removed: {n_removed}/{max_samples}")

            if aggressive_snr_filter:
                snr_values = np.array([snr_dict.get(i, np.nan) for i in range(max_samples)])
                valid_snr = snr_values[~np.isnan(snr_values) & ~outlier_mask]

                if len(valid_snr) > 0:
                    q_lower_pct = snr_quantile_filter[0] * 100
                    q_upper_pct = snr_quantile_filter[1] * 100
                    q_lower = np.percentile(valid_snr, q_lower_pct)
                    q_upper = np.percentile(valid_snr, q_upper_pct)

                    snr_filter_mask = (
                        (snr_values < q_lower) |
                        (snr_values > q_upper) |
                        np.isnan(snr_values)
                    )
                    outlier_mask = outlier_mask | snr_filter_mask

                    n_snr_removed = np.sum(snr_filter_mask & ~outlier_mask)
                    print(f"SNR filtering: removed {n_snr_removed} additional samples")
                    print(f"Thresholds: Q{int(q_lower_pct)} = {q_lower:.2f}, Q{int(q_upper_pct)} = {q_upper:.2f}")

        sample_indices = [i for i in sample_indices if not outlier_mask[i]]
        n_samples = len(sample_indices)

        if despike:
            print(f"\n=== Applying Whittaker-Hayes Despiking ===")
            print(f"  Processing {n_samples} samples × {n_times} integration times")

        intensity_array = np.full((n_samples, n_times, n_wavenumbers), np.nan, dtype=np.float32)

        for time_idx, int_time in enumerate(sorted_times):
            data = self.spectral_data[int_time][:, wn_mask] if crop else self.spectral_data[int_time]

            for new_idx, orig_idx in enumerate(sample_indices):
                if orig_idx < data.shape[0]:
                    spectrum = data[orig_idx]

                    if despike and HAS_RAMANSPY:
                        container = rp.SpectralContainer(spectrum[np.newaxis, :], spectral_axis)
                        despiked = rp.preprocessing.despike.WhitakerHayes().apply(container)
                        spectrum = despiked.spectral_data[0]

                    intensity_array[new_idx, time_idx, :] = spectrum

        intensity_description = "Raw Raman intensity"
        if despike:
            intensity_description = "Raman intensity (Whittaker-Hayes despiked)"

        ds = xr.Dataset(
            data_vars={
                "intensity_raw": (
                    ["sample", "integration_time", "wavenumber"],
                    intensity_array,
                    {
                        "long_name": intensity_description,
                        "units": "counts",
                        "despiked": despike,
                    },
                )
            },
            coords={
                "sample": np.arange(n_samples),
                "integration_time": sorted_times,
                "wavenumber": spectral_axis,
            },
            attrs={
                "title": "Simple Raman Spectroscopy Dataset",
                "creation_date": pd.Timestamp.now().isoformat(),
            },
        )

        # Baseline correction
        if baseline_correction and HAS_RAMANSPY:
            print("\nApplying IARPLS baseline correction")
            raw_intensity = ds["intensity_raw"].values
            wavenumbers = ds["wavenumber"].values

            baseline_corrected_data = np.full_like(raw_intensity, np.nan)

            for s_idx in range(ds.dims["sample"]):
                for t_idx in range(ds.dims["integration_time"]):
                    spectrum = raw_intensity[s_idx, t_idx, :]
                    if not np.isnan(spectrum).all():
                        container = rp.SpectralContainer(spectrum[np.newaxis, :], wavenumbers)
                        corrected = rp.preprocessing.baseline.ModPoly().apply(container)
                        baseline_corrected_data[s_idx, t_idx, :] = corrected.spectral_data[0]

            ds["intensity_baseline_corrected"] = (
                ["sample", "integration_time", "wavenumber"],
                baseline_corrected_data.astype(np.float32),
                {"long_name": "Baseline-corrected Raman intensity"},
            )
            print("Baseline correction complete.")

        # Normalization
        if normalise:
            print("\nApplying normalisation")
            source_data = (
                ds["intensity_baseline_corrected"] if baseline_correction else ds["intensity_raw"]
            )
            ds = add_normalized_intensity(ds, source_data, method=method)

        return ds


# =============================================================================
# Loading Function
# =============================================================================

def load_simple_data(
    data_folder: str,
    single_file: Optional[str] = None,
    wavenumber_file: str = "wavenumber.txt",
    spectra_pattern: str = "*_spectra.txt",
    background_pattern: str = "*_BG_spectra.txt",
    apply_background_subtraction: bool = False,
    clip: Tuple[int, int] = (1, 0),
    delimiter: str = ",",
) -> SimpleDataset:
    """
    Load simple Raman data from folder.

    Expects:
    - wavenumber.txt: Wavenumber axis
    - *_spectra.txt: Spectra files (named by integration time)
    - *_BG_spectra.txt: Optional background files

    Parameters
    ----------
    data_folder : str
        Path to data folder
    single_file : str, optional
        Load only this specific file (not implemented)
    wavenumber_file : str
        Name of wavenumber axis file
    spectra_pattern : str
        Glob pattern for spectra files
    background_pattern : str
        Glob pattern for background files
    apply_background_subtraction : bool
        Subtract background after loading
    clip : tuple
        (start, end) points to clip from wavenumber axis
    delimiter : str
        CSV delimiter

    Returns
    -------
    SimpleDataset
        Loaded dataset

    Examples
    --------
    >>> dataset = load_simple_data('/path/to/data')
    >>> ds = dataset.to_xarray(crop=(400, 1800), normalise=True)
    """
    data_folder = Path(data_folder)

    # Find wavenumber file
    wavenumber_path = data_folder / wavenumber_file
    if not wavenumber_path.exists():
        wavenumber_path = data_folder.parent / wavenumber_file
    if not wavenumber_path.exists():
        raise FileNotFoundError(f"Wavenumber file not found: {wavenumber_file}")

    print(f"Loading wavenumber axis from: {wavenumber_path}")
    wavenumbers = np.loadtxt(wavenumber_path, delimiter=delimiter)

    if clip[0] > 0 or clip[1] > 0:
        print(f"Clipping wavenumber axis: removing first {clip[0]} and last {clip[1]} points")
        wavenumbers = wavenumbers[clip[0]:len(wavenumbers) - clip[1] if clip[1] > 0 else None]

    print(f"Wavenumber axis: {len(wavenumbers)} points, "
          f"range [{wavenumbers.min():.2f}, {wavenumbers.max():.2f}] cm⁻¹")

    dataset = SimpleDataset(wavenumbers)

    # Find spectra files
    spectra_files = [f for f in data_folder.glob(spectra_pattern) if "_BG_" not in f.name]
    if not spectra_files:
        raise FileNotFoundError(f"No spectra files found matching: {spectra_pattern}")

    print(f"\nFound {len(spectra_files)} spectra files:")

    for spectra_file in sorted(spectra_files):
        time_str = spectra_file.stem.replace("_spectra", "")
        print(f"\n  Loading {time_str}:")
        print(f"    File: {spectra_file.name}")

        spectra_data = np.loadtxt(spectra_file, delimiter=delimiter)

        if clip[0] > 0 or clip[1] > 0:
            end_idx = spectra_data.shape[1] - clip[1] if clip[1] > 0 else None
            spectra_data = spectra_data[:, clip[0]:end_idx]
            print(f"    Clipped spectra shape: {spectra_data.shape}")
        else:
            print(f"    Shape: {spectra_data.shape} (samples × wavenumbers)")

        # Check for background file
        bg_file = data_folder / f"{time_str}_BG_spectra.txt"
        background_data = None

        if bg_file.exists():
            print(f"    Background file: {bg_file.name}")
            background_data = np.loadtxt(bg_file, delimiter=delimiter)

            if clip[0] > 0 or clip[1] > 0:
                end_idx = background_data.shape[1] - clip[1] if clip[1] > 0 else None
                background_data = background_data[:, clip[0]:end_idx]

            print(f"    Background shape: {background_data.shape}")

        dataset.add_integration_time(time_str, spectra_data, background=background_data)

    if apply_background_subtraction:
        print("\n=== Applying Background Subtraction ===")
        dataset.apply_background_subtraction()

    dataset.summary()
    return dataset
