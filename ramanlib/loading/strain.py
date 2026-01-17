"""
StrainDataset for bacterial Raman spectroscopy data.

Handles ATCC bacterial spectra with multiple strains, integration times,
and per-sample wavenumber calibration.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
import re

from ramanlib.loading.base import (
    detect_saturation,
    detect_dropout,
    preprocess_for_outlier_detection,
    compute_snr,
    add_normalized_intensity,
    filter_misaligned_pairs,
)

try:
    import ramanspy as rp
    HAS_RAMANSPY = True
except ImportError:
    HAS_RAMANSPY = False


class StrainDataset:
    """
    Container for bacterial Raman spectroscopy data with unified wavenumber axis.

    Supports multiple strains with per-sample calibration axes. Uses eager-loading
    where unique axes are registered immediately to save memory.
    """

    def __init__(self):
        self.strains: Dict = {}
        self.unique_axes: List[np.ndarray] = []

    def add_spectrum(self, spectrum, metadata: Dict):
        """
        Add spectrum and register its axis.

        Parameters
        ----------
        spectrum : ramanspy.Spectrum
            Loaded spectrum object
        metadata : dict
            Metadata including 'strain', 'species', 'gram', etc.
        """
        strain_id = metadata["strain"]
        axis_id = self._get_or_create_axis_id(spectrum.spectral_axis)

        if strain_id not in self.strains:
            self.strains[strain_id] = {
                "spectra": [],
                "axis_ids": [],
                "metadata": [],
                "species": metadata.get("species", "Unknown"),
                "gram": metadata.get("gram", "Unknown"),
            }

        self.strains[strain_id]["spectra"].append(spectrum.spectral_data)
        self.strains[strain_id]["axis_ids"].append(axis_id)
        self.strains[strain_id]["metadata"].append(metadata)

    def _get_or_create_axis_id(self, new_axis: np.ndarray, tolerance: float = 1e-4) -> int:
        """Check if axis exists in registry. Returns index of axis."""
        for i, existing_axis in enumerate(self.unique_axes):
            if len(existing_axis) != len(new_axis):
                continue
            if np.allclose(existing_axis, new_axis, atol=tolerance):
                return i

        self.unique_axes.append(new_axis)
        print(f"  -> New calibration set detected (Set #{len(self.unique_axes) - 1})")
        return len(self.unique_axes) - 1

    def get_unique_integration_times(self) -> List[str]:
        """Get sorted list of unique integration times across all strains."""
        times = set()
        for strain_data in self.strains.values():
            for meta in strain_data["metadata"]:
                times.add(meta["integration_time"])
        return sorted(times)

    def get_positions_by_strain(self, strain_id: int) -> List[int]:
        """Get unique position IDs for specified strain."""
        if strain_id not in self.strains:
            raise ValueError(f"Strain {strain_id} not found")

        positions = set()
        for meta in self.strains[strain_id]["metadata"]:
            positions.add(meta["position"])
        return sorted(list(positions))

    def summary(self):
        """Print dataset summary."""
        print(f"\nLoaded {len(self.strains)} strains:")
        for strain_id, data in self.strains.items():
            print(f"  Strain {strain_id} ({data['species']}, {data['gram']}): "
                  f"{len(data['spectra'])} spectra")

    def to_xarray(
        self,
        remove_incomplete: bool = True,
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
        snr_filter_per_species: bool = True,
        remove_misaligned_pairs: bool = False,
        misalignment_reference_time: Optional[str] = None,
        misalignment_low_time: Optional[str] = None,
        misalignment_threshold: float = 1.5,
        misalignment_method: str = "iqr",
        misalignment_metrics: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Convert to xarray Dataset with (sample, integration_time, pixel) structure.

        Parameters
        ----------
        remove_incomplete : bool
            Remove samples missing any integration time
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
        snr_filter_per_species : bool
            Apply SNR filter per species
        remove_misaligned_pairs : bool
            Remove misaligned integration time pairs
        misalignment_reference_time : str, optional
            Reference time for alignment check
        misalignment_low_time : str, optional
            Low time for alignment check
        misalignment_threshold : float
            Threshold for misalignment detection
        misalignment_method : str
            Method: 'iqr', 'mad', or 'zscore'
        misalignment_metrics : list, optional
            Metrics: 'pearson', 'cosine', 'sam'

        Returns
        -------
        xr.Dataset
            Dataset with intensity, metadata, and coordinates
        """
        if not self.strains:
            raise ValueError("No data available - load data first")

        # Validate axis compatibility
        pixel_counts = [len(ax) for ax in self.unique_axes]
        if not pixel_counts:
            raise ValueError("No spectral axes found in registry.")

        if len(set(pixel_counts)) > 1:
            raise ValueError(
                f"Cannot stack spectra: Found variable pixel counts {set(pixel_counts)}."
            )

        n_pixels_raw = pixel_counts[0]

        # Validate SNR silent region
        if aggressive_snr_filter:
            all_axes_stacked = np.vstack(self.unique_axes)
            global_min = all_axes_stacked.min()
            global_max = all_axes_stacked.max()

            if snr_silent_region[0] < global_min or snr_silent_region[1] > global_max:
                raise ValueError(
                    f"SNR silent region {snr_silent_region} outside spectral range "
                    f"[{global_min:.1f}, {global_max:.1f}] cm⁻¹."
                )

        all_integration_times = sorted(self.get_unique_integration_times())
        n_times = len(all_integration_times)

        sample_pairs = []
        removal_dict = {}
        snr_dict = {}
        species_dict = {}

        # Filtering and sample selection
        for strain_id in sorted(self.strains.keys()):
            strain_data_dict = self.strains[strain_id]
            metadata_df = pd.DataFrame(strain_data_dict["metadata"])

            positions = self.get_positions_by_strain(strain_id)
            n_original_positions = len(positions)
            total_removed, saturation_removal, dropout_removal = 0, 0, 0

            spectral_data = strain_data_dict["spectra"]
            axis_ids = strain_data_dict["axis_ids"]
            species = strain_data_dict["species"]

            for position_id in positions:
                position_mask = metadata_df["position"] == position_id
                position_data = metadata_df[position_mask]

                # Incomplete data check
                if remove_incomplete:
                    available_times = set(position_data["integration_time"])
                    if len(available_times) < n_times:
                        continue

                # Outlier detection
                if remove_outliers:
                    has_outlier = False
                    for int_time in all_integration_times:
                        time_mask = position_mask & (
                            metadata_df["integration_time"] == int_time
                        )

                        if time_mask.any():
                            spectrum_idx = np.where(time_mask)[0][0]
                            spectrum_raw = spectral_data[spectrum_idx]

                            current_axis_id = axis_ids[spectrum_idx]
                            current_axis = self.unique_axes[current_axis_id]

                            if detect_saturation(spectrum_raw):
                                saturation_removal += 1
                                has_outlier = True
                                break

                            spectrum_processed = preprocess_for_outlier_detection(
                                spectrum_raw, current_axis, baseline_correction
                            )

                            if detect_dropout(spectrum_processed, outlier_mad_threshold):
                                dropout_removal += 1
                                has_outlier = True
                                break

                    if has_outlier:
                        total_removed += 1
                        continue

                # SNR calculation
                if aggressive_snr_filter:
                    ref_time = snr_reference_time or all_integration_times[0]
                    ref_time_mask = position_mask & (
                        metadata_df["integration_time"] == ref_time
                    )

                    if ref_time_mask.any():
                        spectrum_idx = np.where(ref_time_mask)[0][0]
                        spectrum_raw = spectral_data[spectrum_idx]

                        current_axis_id = axis_ids[spectrum_idx]
                        current_axis = self.unique_axes[current_axis_id]

                        snr_value = compute_snr(spectrum_raw, current_axis, snr_silent_region)
                        snr_dict[(strain_id, position_id)] = snr_value
                        species_dict[(strain_id, position_id)] = species

                sample_pairs.append((strain_id, position_id))

            removal_dict[strain_id] = {
                "n_original_positions": n_original_positions,
                "total_removed": total_removed,
                "saturation_removed": saturation_removal,
                "dropout_removed": dropout_removal,
            }

        # Report outlier removal
        if remove_outliers:
            print("\n=== Outlier Removal Summary ===")
            for strain_id, stats in removal_dict.items():
                print(f"Strain {strain_id}: Orig: {stats['n_original_positions']}, "
                      f"Rm: {stats['total_removed']} (Sat: {stats['saturation_removed']}, "
                      f"Drop: {stats['dropout_removed']})")

        # SNR filtering
        if aggressive_snr_filter:
            print("\n=== SNR Filtering ===")
            ref_time = snr_reference_time or all_integration_times[0]
            print(f"Reference time: {ref_time}, silent region: {snr_silent_region[0]}-{snr_silent_region[1]} cm⁻¹")
            print(f"Filter mode: {'per-species' if snr_filter_per_species else 'global'}")

            if snr_filter_per_species:
                sample_pairs = self._apply_per_species_snr_filter(
                    sample_pairs, snr_dict, species_dict, snr_quantile_filter
                )
            else:
                snr_values = np.array([snr_dict.get(pair, np.nan) for pair in sample_pairs])
                valid_snr_mask = ~np.isnan(snr_values)
                valid_snr = snr_values[valid_snr_mask]

                if len(valid_snr) > 0:
                    q_lower_pct, q_upper_pct = snr_quantile_filter[0] * 100, snr_quantile_filter[1] * 100
                    q_lower = np.percentile(valid_snr, q_lower_pct)
                    q_upper = np.percentile(valid_snr, q_upper_pct)
                    snr_filter_mask = (snr_values >= q_lower) & (snr_values <= q_upper)

                    n_before = len(sample_pairs)
                    n_q1 = np.sum(valid_snr_mask & (snr_values < q_lower))
                    n_q4 = np.sum(valid_snr_mask & (snr_values > q_upper))
                    n_kept = np.sum(snr_filter_mask)

                    print(f"Global thresholds: Q{int(q_lower_pct)} = {q_lower:.2f}, Q{int(q_upper_pct)} = {q_upper:.2f}")
                    print(f"Removed Q1: {n_q1}, Q4: {n_q4} ({(n_q1 + n_q4) / n_before * 100:.1f}%)")
                    print(f"Kept: {n_kept}/{n_before} ({n_kept / n_before * 100:.1f}%)")

                    sample_pairs = [
                        pair for pair, keep in zip(sample_pairs, snr_filter_mask) if keep
                    ]

        n_samples = len(sample_pairs)

        # Determine cropping mask
        crop_indices_map = {}
        final_n_pixels = n_pixels_raw

        if crop is not None:
            print(f"\n=== Cropping Spectra: {crop[0]} - {crop[1]} cm⁻¹ ===")
            lengths = []
            for i, ax in enumerate(self.unique_axes):
                mask = (ax >= crop[0]) & (ax <= crop[1])
                crop_indices_map[i] = mask
                lengths.append(mask.sum())

            if len(set(lengths)) > 1:
                final_n_pixels = min(lengths)
                print(f"  Warning: Variable crop lengths {set(lengths)}. Truncating to {final_n_pixels}.")
            else:
                final_n_pixels = lengths[0]
        else:
            for i in range(len(self.unique_axes)):
                crop_indices_map[i] = slice(None)

        # Allocate arrays
        intensity_array = np.full((n_samples, n_times, final_n_pixels), np.nan, dtype=np.float32)
        wavenumber_array = np.full((n_samples, final_n_pixels), np.nan, dtype=np.float32)

        metadata = defaultdict(list)
        n_despiked = 0

        # Populate data
        for sample_idx, (strain_id, position_id) in enumerate(sample_pairs):
            strain_data_dict = self.strains[strain_id]
            metadata_df = pd.DataFrame(strain_data_dict["metadata"])

            spectra_list = strain_data_dict["spectra"]
            axis_ids_list = strain_data_dict["axis_ids"]

            position_mask = metadata_df["position"] == position_id

            meta_row = metadata_df[position_mask].iloc[0]
            metadata["species"].append(str(strain_data_dict["species"]))
            metadata["gram"].append(str(strain_data_dict["gram"]))
            metadata["strain_id"].append(strain_id)
            metadata["position_id"].append(position_id)
            metadata["date"].append(str(meta_row["date"]))
            metadata["spectrum_count"].append(position_mask.sum())

            for time_idx, int_time in enumerate(all_integration_times):
                time_mask = position_mask & (metadata_df["integration_time"] == int_time)

                if time_mask.any():
                    idx = np.where(time_mask)[0][0]

                    raw_spectrum = spectra_list[idx]
                    ax_id = axis_ids_list[idx]
                    full_axis = self.unique_axes[ax_id]

                    if despike and HAS_RAMANSPY:
                        n_despiked += 1
                        container = rp.SpectralContainer(raw_spectrum[np.newaxis, :], full_axis)
                        raw_spectrum = (
                            rp.preprocessing.despike.WhitakerHayes()
                            .apply(container)
                            .spectral_data[0]
                        )

                    mask = crop_indices_map[ax_id]
                    cropped_spectrum = raw_spectrum[mask]

                    if len(cropped_spectrum) > final_n_pixels:
                        cropped_spectrum = cropped_spectrum[:final_n_pixels]

                    intensity_array[sample_idx, time_idx, :] = cropped_spectrum

                    if time_idx == 0:
                        cropped_axis = full_axis[mask]
                        if len(cropped_axis) > final_n_pixels:
                            cropped_axis = cropped_axis[:final_n_pixels]
                        wavenumber_array[sample_idx, :] = cropped_axis

        if despike:
            print(f"  Despiked {n_despiked} spectra in total")

        intensity_description = "Raw Raman intensity"
        if despike:
            intensity_description = "Raman intensity (Whittaker-Hayes despiked)"

        # Create dataset
        ds = xr.Dataset(
            data_vars={
                "intensity_raw": (
                    ["sample", "integration_time", "pixel"],
                    intensity_array,
                    {
                        "long_name": intensity_description,
                        "units": "counts",
                        "despiked": despike,
                    },
                ),
                "species": (["sample"], metadata["species"], {"long_name": "Bacterial species"}),
                "gram": (["sample"], metadata["gram"], {"long_name": "Gram staining type"}),
                "strain_id": (["sample"], np.array(metadata["strain_id"], dtype="int32")),
                "position_id": (["sample"], np.array(metadata["position_id"], dtype="int32")),
                "date": (["sample"], metadata["date"]),
                "spectrum_count": (["sample"], np.array(metadata["spectrum_count"], dtype="int32")),
            },
            coords={
                "wavenumber": (["sample", "pixel"], wavenumber_array),
                "sample": np.arange(n_samples),
                "integration_time": all_integration_times,
                "pixel": np.arange(final_n_pixels),
            },
            attrs={
                "title": "ATCC Bacterial Raman Dataset",
                "n_strains": len(self.strains),
                "creation_date": pd.Timestamp.now().isoformat(),
            },
        )

        # Post-processing: baseline correction
        if baseline_correction and HAS_RAMANSPY:
            print("\nApplying IARPLS baseline correction...")
            raw_int = ds["intensity_raw"].values
            wns = ds["wavenumber"].values

            corrected_data = np.full_like(raw_int, np.nan)

            for s in range(n_samples):
                sample_axis = wns[s, :]
                for t in range(n_times):
                    spec = raw_int[s, t, :]
                    if not np.isnan(spec).all():
                        container = rp.SpectralContainer(spec[np.newaxis, :], sample_axis)
                        corrected_data[s, t, :] = (
                            rp.preprocessing.baseline.ModPoly().apply(container).spectral_data[0]
                        )

            ds["intensity_baseline_corrected"] = (
                ["sample", "integration_time", "pixel"],
                corrected_data.astype(np.float32),
                {"long_name": "Baseline-corrected Raman intensity"},
            )
            print("Baseline correction complete.")

        # Post-processing: normalization
        if normalise:
            print("\nApplying normalisation")
            target_var = "intensity_baseline_corrected" if baseline_correction else "intensity_raw"
            ds = add_normalized_intensity(ds, ds[target_var], method=method)

        # Post-processing: misalignment removal
        if remove_misaligned_pairs:
            if len(all_integration_times) < 2:
                print("Warning: Cannot check pair alignment with < 2 integration times")
            else:
                ref_time = misalignment_reference_time or all_integration_times[-1]
                low_time = misalignment_low_time or all_integration_times[0]

                if ref_time not in all_integration_times or low_time not in all_integration_times:
                    print(f"Warning: Specified times ({low_time}, {ref_time}) not in dataset.")
                else:
                    print(f"\n=== Quality Control: Removing Misaligned Pairs ===")
                    print(f"Comparing {low_time} vs {ref_time}")

                    if misalignment_metrics is None:
                        misalignment_metrics = ["pearson", "cosine"]

                    ds, _ = filter_misaligned_pairs(
                        ds, low_time, ref_time,
                        threshold=misalignment_threshold,
                        method=misalignment_method,
                        metrics=misalignment_metrics,
                    )

        return ds

    def _apply_per_species_snr_filter(
        self,
        sample_pairs: List,
        snr_dict: Dict,
        species_dict: Dict,
        snr_quantile_filter: Tuple[float, float],
    ) -> List:
        """Apply SNR filter separately for each species."""
        q_lower_pct = snr_quantile_filter[0] * 100
        q_upper_pct = snr_quantile_filter[1] * 100

        species_samples = defaultdict(list)
        for pair in sample_pairs:
            species = species_dict.get(pair, "Unknown")
            snr = snr_dict.get(pair, np.nan)
            species_samples[species].append((pair, snr))

        print(f"\nPer-species SNR filtering (keeping Q{int(q_lower_pct)}-Q{int(q_upper_pct)}):")
        print(f"{'Species':<30} {'Before':>8} {'After':>8} {'Q_low':>8} {'Q_high':>8} {'Removed':>8}")
        print("-" * 80)

        filtered_pairs = []
        total_before = 0
        total_after = 0

        for species in sorted(species_samples.keys()):
            samples = species_samples[species]
            n_before = len(samples)
            total_before += n_before

            snr_values = np.array([snr for _, snr in samples])
            valid_mask = ~np.isnan(snr_values)
            valid_snr = snr_values[valid_mask]

            if len(valid_snr) < 4:
                print(f"{species:<30} {n_before:>8} {n_before:>8} {'N/A':>8} {'N/A':>8} {0:>8} (too few samples)")
                filtered_pairs.extend([pair for pair, _ in samples])
                total_after += n_before
                continue

            q_lower = np.percentile(valid_snr, q_lower_pct)
            q_upper = np.percentile(valid_snr, q_upper_pct)

            kept_samples = []
            for pair, snr in samples:
                if np.isnan(snr):
                    continue
                if q_lower <= snr <= q_upper:
                    kept_samples.append(pair)

            n_after = len(kept_samples)
            n_removed = n_before - n_after
            total_after += n_after

            print(f"{species:<30} {n_before:>8} {n_after:>8} {q_lower:>8.2f} {q_upper:>8.2f} {n_removed:>8}")
            filtered_pairs.extend(kept_samples)

        print("-" * 80)
        print(f"{'TOTAL':<30} {total_before:>8} {total_after:>8} {'':<8} {'':<8} {total_before - total_after:>8}")
        print(f"\nRetained {total_after}/{total_before} samples ({total_after / total_before * 100:.1f}%)")

        return filtered_pairs


# =============================================================================
# Loading Function
# =============================================================================

# Default ATCC strain information
DEFAULT_STRAIN_INFO = {
    25923: {"species": "Staphylococcus aureus", "gram": "G+"},
    29213: {"species": "Staphylococcus aureus", "gram": "G+"},
    25922: {"species": "Escherichia coli", "gram": "G-"},
    35218: {"species": "Escherichia coli", "gram": "G-"},
    27853: {"species": "Pseudomonas aeruginosa", "gram": "G-"},
    700603: {"species": "Klebsiella Pneumoniae", "gram": "G-"},
    12228: {"species": "Staphylococcus epidermidis", "gram": "G+"},
    19606: {"species": "Acinetobacter baumannii", "gram": "G-"},
    29212: {"species": "Enterococcus faecalis", "gram": "G+"},
}


def load_data(
    data_folder: str,
    strain_info: Optional[Dict] = None,
) -> StrainDataset:
    """
    Load ATCC bacterial Raman spectra from folder.

    Expects LabSpec format files with naming convention:
    YYYYMMDD_STRAIN_INTEGRATIONs_--Position N--Spectrum--M--Spec.Data X Y.txt

    Parameters
    ----------
    data_folder : str
        Path to folder containing spectrum files
    strain_info : dict, optional
        Mapping of strain ID to species/gram info. Uses defaults if None.

    Returns
    -------
    StrainDataset
        Loaded dataset

    Examples
    --------
    >>> dataset = load_data('/path/to/atcc/data')
    >>> ds = dataset.to_xarray(crop=(400, 1800), normalise=True)
    """
    if not HAS_RAMANSPY:
        raise ImportError("ramanspy is required for loading ATCC data")

    data_folder = Path(data_folder)
    if strain_info is None:
        strain_info = DEFAULT_STRAIN_INFO

    txt_files = sorted(data_folder.glob("*.txt"))
    print(f"Found {len(txt_files)} spectrum files")

    pattern = re.compile(
        r"(\d{8})_(\d+)_([0-9.]+s)_--Position (\d+)--Spectrum--(\d+)--Spec\.Data (\d+) (\d+)\.txt"
    )

    dataset = StrainDataset()
    loaded_count = 0
    errors = []

    for filepath in txt_files:
        match = pattern.match(filepath.name)
        if not match:
            continue

        date, strain, int_time, position, spec_id, data_id, file_num = match.groups()
        strain_id = int(strain)

        if strain_id not in strain_info:
            continue

        try:
            spectrum = rp.load.labspec(str(filepath))
            metadata = {
                "date": date,
                "strain": strain_id,
                "integration_time": int_time,
                "position": int(position),
                "spectrum_id": int(spec_id),
                "file_number": int(file_num),
                "filename": filepath.name,
            }
            metadata.update(strain_info[strain_id])
            dataset.add_spectrum(spectrum, metadata)
            loaded_count += 1
        except Exception as e:
            errors.append((filepath.name, str(e)))

    if errors:
        print(f"\nWarning: {len(errors)} files failed to load")
        if len(errors) <= 3:
            for fname, error in errors:
                print(f"  {fname}: {error}")

    print(f"\nSuccessfully loaded {loaded_count} spectra")
    dataset.summary()
    return dataset
