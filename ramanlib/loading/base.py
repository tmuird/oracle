"""
Base utilities for Raman spectroscopy data loading.

Provides shared functionality for quality control, normalization,
and outlier detection used by both StrainDataset and SimpleDataset.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional, List, Dict
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

try:
    import ramanspy as rp
    HAS_RAMANSPY = True
except ImportError:
    HAS_RAMANSPY = False


# =============================================================================
# Outlier Detection
# =============================================================================

def detect_saturation(spectrum: np.ndarray, threshold: float = 60000) -> bool:
    """
    Detect if spectrum has saturation artifacts.

    Parameters
    ----------
    spectrum : np.ndarray
        Raw spectrum
    threshold : float
        Saturation threshold (detector-dependent)

    Returns
    -------
    bool
        True if saturated
    """
    return np.mean(spectrum > threshold) > 0.1


def detect_dropout(
    spectrum: np.ndarray,
    mad_threshold: float = 8.0,
    nan_fraction_limit: float = 0.1,
    min_std: float = 0.01,
) -> bool:
    """
    Detect if spectrum has dropout artifacts (cosmic rays, dead pixels).

    Uses MAD (Median Absolute Deviation) for robust outlier detection.

    Parameters
    ----------
    spectrum : np.ndarray
        Preprocessed (normalized) spectrum
    mad_threshold : float
        Threshold for MAD-based outlier detection
    nan_fraction_limit : float
        Maximum allowed fraction of NaN values
    min_std : float
        Minimum standard deviation (below = flat/dead spectrum)

    Returns
    -------
    bool
        True if dropout detected
    """
    nan_fraction = np.isnan(spectrum).sum() / len(spectrum)
    if nan_fraction > nan_fraction_limit:
        return True

    spectrum_std = np.nanstd(spectrum)
    if spectrum_std < min_std:
        return True

    median_val = np.nanmedian(spectrum)
    mad = np.nanmedian(np.abs(spectrum - median_val))
    if mad == 0:
        return True

    max_deviation = np.nanmax(np.abs(spectrum - median_val))
    if max_deviation > mad_threshold * mad * 2:
        return True

    return False


def preprocess_for_outlier_detection(
    spectrum: np.ndarray,
    spectral_axis: np.ndarray,
    baseline_correction: bool = False,
) -> np.ndarray:
    """
    Preprocess spectrum for outlier detection.

    Centers and normalizes the spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        Raw spectrum
    spectral_axis : np.ndarray
        Wavenumber axis
    baseline_correction : bool
        Apply baseline correction before normalization

    Returns
    -------
    np.ndarray
        Preprocessed spectrum
    """
    processed = spectrum.copy()

    if baseline_correction and HAS_RAMANSPY:
        container = rp.SpectralContainer(processed[np.newaxis, :], spectral_axis)
        corrected = rp.preprocessing.baseline.ModPoly().apply(container)
        processed = corrected.spectral_data[0]

    mean_val = np.nanmean(processed)
    centered = processed - mean_val
    norm = np.linalg.norm(centered)
    if norm > 0:
        processed = centered / norm

    return processed


# =============================================================================
# SNR Computation
# =============================================================================

def compute_snr(
    spectrum: np.ndarray,
    wavenumbers: np.ndarray,
    silent_region: Tuple[float, float],
) -> float:
    """
    Compute signal-to-noise ratio using a silent spectral region.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum values
    wavenumbers : np.ndarray
        Wavenumber axis
    silent_region : tuple
        (min, max) wavenumber range for noise estimation

    Returns
    -------
    float
        SNR value (or NaN if cannot compute)
    """
    noise_mask = (wavenumbers >= silent_region[0]) & (wavenumbers <= silent_region[1])
    noise_region = spectrum[noise_mask]
    noise_region = noise_region[~np.isnan(noise_region)]

    if len(noise_region) < 5:
        return np.nan

    max_signal = np.nanmax(spectrum)
    avg_noise = np.mean(noise_region)
    std_noise = np.std(noise_region, ddof=1)

    if std_noise == 0:
        return np.nan

    return (max_signal - avg_noise) / std_noise


# =============================================================================
# Normalization
# =============================================================================

def add_normalized_intensity(
    ds: xr.Dataset,
    source_data: xr.DataArray,
    method: str = "l2",
) -> xr.Dataset:
    """
    Add normalized intensity variable to dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to modify
    source_data : xr.DataArray
        Source intensity data
    method : str
        Normalization method: 'l2' or 'percentile'

    Returns
    -------
    xr.Dataset
        Dataset with 'intensity_normalised' variable added
    """
    intensity = source_data.values

    if method == "l2":
        print("Applying L2 normalisation")
        means = np.nanmean(intensity, axis=2, keepdims=True)
        centered = intensity - means
        norms = np.linalg.norm(centered, axis=2, keepdims=True)
        norms[norms == 0] = 1
        normalised = centered / norms

        ds["l2_norms"] = (
            ["sample", "integration_time"],
            norms.squeeze(axis=2).astype(np.float32),
            {"long_name": "L2 normalisation scaling factors", "units": "intensity"},
        )
        ds["l2_means"] = (
            ["sample", "integration_time"],
            means.squeeze(axis=2).astype(np.float32),
            {"long_name": "Mean centering offsets", "units": "intensity"},
        )
        description = "Mean-centred L2 normalisation - unit vectors preserving peak ratios"

    elif method == "percentile":
        print("Applying percentile normalisation")
        p25 = np.nanpercentile(intensity, 25, axis=2, keepdims=True)
        p75 = np.nanpercentile(intensity, 75, axis=2, keepdims=True)
        iqr = p75 - p25
        iqr[iqr == 0] = 1
        normalised = (intensity - p25) / iqr
        description = "Robust per-spectrum normalisation using 25th-75th percentile range"

    else:
        raise ValueError(f"Unknown normalisation method: '{method}'")

    ds["intensity_normalised"] = (
        ["sample", "integration_time", "pixel"],
        normalised.astype(np.float32),
        {
            "long_name": "Normalised Raman intensity",
            "units": "normalised",
            "description": description,
            "method": method,
            "source": source_data.name,
        },
    )
    return ds


# =============================================================================
# Statistical Outlier Detection Methods
# =============================================================================

def detect_outliers_iqr(
    values: np.ndarray,
    threshold: float,
    direction: str = "both",
) -> np.ndarray:
    """
    Detect outliers using IQR method.

    Parameters
    ----------
    values : np.ndarray
        Values to check
    threshold : float
        IQR multiplier for bounds
    direction : str
        'low', 'high', or 'both'

    Returns
    -------
    np.ndarray
        Boolean mask of outliers
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    if direction == "low":
        return values < lower_bound
    elif direction == "high":
        return values > upper_bound
    return (values < lower_bound) | (values > upper_bound)


def detect_outliers_mad(
    values: np.ndarray,
    threshold: float,
    direction: str = "both",
) -> np.ndarray:
    """
    Detect outliers using MAD (Median Absolute Deviation).

    Parameters
    ----------
    values : np.ndarray
        Values to check
    threshold : float
        Modified z-score threshold
    direction : str
        'low', 'high', or 'both'

    Returns
    -------
    np.ndarray
        Boolean mask of outliers
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return np.zeros(len(values), dtype=bool)

    modified_z_scores = 0.6745 * (values - median) / mad

    if direction == "low":
        return modified_z_scores < -threshold
    elif direction == "high":
        return modified_z_scores > threshold
    return np.abs(modified_z_scores) > threshold


def detect_outliers_zscore(
    values: np.ndarray,
    threshold: float,
    direction: str = "both",
) -> np.ndarray:
    """
    Detect outliers using z-score method.

    Parameters
    ----------
    values : np.ndarray
        Values to check
    threshold : float
        Z-score threshold
    direction : str
        'low', 'high', or 'both'

    Returns
    -------
    np.ndarray
        Boolean mask of outliers
    """
    mean = np.mean(values)
    std = np.std(values)
    z_scores = (values - mean) / (std + 1e-10)

    if direction == "low":
        return z_scores < -threshold
    elif direction == "high":
        return z_scores > threshold
    return np.abs(z_scores) > threshold


# =============================================================================
# Pair Alignment Quality Control
# =============================================================================

def compute_pair_similarities(
    low_snr: np.ndarray,
    ref_snr: np.ndarray,
    metrics: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute similarity metrics between low and reference SNR spectra.

    Parameters
    ----------
    low_snr : np.ndarray
        Low SNR spectra, shape (n_samples, n_wavenumbers)
    ref_snr : np.ndarray
        Reference SNR spectra, shape (n_samples, n_wavenumbers)
    metrics : list of str
        Metrics to compute: 'pearson', 'cosine', 'sam'

    Returns
    -------
    dict
        Similarity values per metric
    """
    n_samples = low_snr.shape[0]
    similarities = {}

    for metric in metrics:
        if metric == "pearson":
            values = np.array([
                pearsonr(low_snr[i], ref_snr[i])[0]
                for i in range(n_samples)
            ])
            similarities["pearson"] = values

        elif metric == "cosine":
            values = np.array([
                1 - cosine(low_snr[i], ref_snr[i])
                for i in range(n_samples)
            ])
            similarities["cosine"] = values

        elif metric == "sam":
            # Spectral Angle Mapper
            values = np.array([
                np.arccos(np.clip(
                    np.dot(low_snr[i], ref_snr[i]) / (
                        np.linalg.norm(low_snr[i]) * np.linalg.norm(ref_snr[i]) + 1e-10
                    ), -1, 1
                ))
                for i in range(n_samples)
            ])
            similarities["sam"] = values

    return similarities


def filter_misaligned_pairs(
    ds: xr.Dataset,
    low_time: str,
    ref_time: str,
    threshold: float = 1.5,
    method: str = "iqr",
    metrics: Optional[List[str]] = None,
) -> Tuple[xr.Dataset, Dict]:
    """
    Remove samples where low and high SNR pairs are misaligned.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    low_time : str
        Low SNR integration time
    ref_time : str
        Reference integration time
    threshold : float
        Outlier detection threshold
    method : str
        'iqr', 'mad', or 'zscore'
    metrics : list of str, optional
        Similarity metrics to use. Default: ['pearson', 'cosine']

    Returns
    -------
    ds_filtered : xr.Dataset
        Filtered dataset
    qc_report : dict
        Quality control report
    """
    if metrics is None:
        metrics = ["pearson", "cosine"]

    # Determine intensity variable
    if "intensity_normalised" in ds:
        intensity_var = "intensity_normalised"
    elif "intensity_baseline_corrected" in ds:
        intensity_var = "intensity_baseline_corrected"
    else:
        intensity_var = "intensity_raw"

    low_snr = ds[intensity_var].sel(integration_time=low_time).values
    ref_snr = ds[intensity_var].sel(integration_time=ref_time).values

    n_samples = low_snr.shape[0]

    # Compute similarities
    similarities = compute_pair_similarities(low_snr, ref_snr, metrics)

    # Detect outliers for each metric
    outlier_masks = {}
    for metric_name, values in similarities.items():
        direction = "low" if metric_name != "sam" else "high"

        if method == "iqr":
            outlier_masks[metric_name] = detect_outliers_iqr(values, threshold, direction)
        elif method == "mad":
            outlier_masks[metric_name] = detect_outliers_mad(values, threshold, direction)
        elif method == "zscore":
            outlier_masks[metric_name] = detect_outliers_zscore(values, threshold, direction)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    # Combine masks (remove if anomalous in ANY metric)
    is_outlier = np.any(list(outlier_masks.values()), axis=0)

    # Filter dataset
    keep_mask = ~is_outlier
    ds_filtered = ds.sel(sample=keep_mask)

    n_removed = np.sum(is_outlier)
    n_kept = np.sum(keep_mask)

    print(f"Method: {method.upper()} (threshold={threshold})")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"\nOriginal samples: {n_samples}")
    print(f"Removed: {n_removed} ({100 * n_removed / n_samples:.1f}%)")
    print(f"Kept: {n_kept} ({100 * n_kept / n_samples:.1f}%)")

    print("\nSimilarity statistics:")
    for metric_name, values in similarities.items():
        n_flagged = np.sum(outlier_masks[metric_name])
        print(f"  {metric_name}: mean={np.mean(values):.4f}, "
              f"std={np.std(values):.4f}, flagged={n_flagged}")

    qc_report = {
        "n_original": n_samples,
        "n_removed": n_removed,
        "n_kept": n_kept,
        "similarities": similarities,
        "outlier_masks": outlier_masks,
    }

    return ds_filtered, qc_report
