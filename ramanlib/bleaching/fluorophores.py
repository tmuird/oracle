"""
Fluorophore spectra loading and processing.

Handles CSV format from fpbase.org and fluorophores.org.
Converts wavelength (nm) to Raman shift (cm⁻¹).
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Optional, Tuple


def nm_to_wavenumber(wavelength_nm: np.ndarray, laser_nm: float = 532.0) -> np.ndarray:
    """
    Convert wavelength (nm) to Raman shift (cm⁻¹).

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Emission wavelength in nm
    laser_nm : float
        Laser excitation wavelength in nm

    Returns
    -------
    np.ndarray
        Raman shift in cm⁻¹ (Stokes shift)
    """
    laser_wn = 1e7 / laser_nm
    emission_wn = 1e7 / wavelength_nm
    return laser_wn - emission_wn


def wavenumber_to_nm(wavenumber: np.ndarray, laser_nm: float = 532.0) -> np.ndarray:
    """
    Convert Raman shift (cm⁻¹) to wavelength (nm).

    Parameters
    ----------
    wavenumber : np.ndarray
        Raman shift in cm⁻¹
    laser_nm : float
        Laser excitation wavelength in nm

    Returns
    -------
    np.ndarray
        Emission wavelength in nm
    """
    laser_wn = 1e7 / laser_nm
    emission_wn = laser_wn - wavenumber
    return 1e7 / emission_wn


class FluorophoreLoader:
    """
    Load fluorophore spectra from fpbase.org or fluorophores.org CSV files.

    Expected CSV format:
    - First column: wavelength (nm)
    - Other columns: fluorophore spectra (marked as 'em' for emission)
    """

    def __init__(self, csv_path: str, laser_nm: float = 532.0):
        """
        Parameters
        ----------
        csv_path : str
            Path to CSV file
        laser_nm : float
            Laser wavelength in nm for converting to Raman shift
        """
        self.csv_path = Path(csv_path)
        self.laser_nm = laser_nm
        self.df = pd.read_csv(csv_path)

        self.wavelength_col = self.df.columns[0]
        self.wavelength_nm = self.df[self.wavelength_col].values
        self.fluorophore_cols = list(self.df.columns[1:])

    def get_emission_spectra(
        self,
        filter_emission: bool = True,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Extract emission spectra from CSV.

        Parameters
        ----------
        filter_emission : bool
            Only include columns with 'em' in the name
        normalize : bool
            Normalize each spectrum to [0, 1]

        Returns
        -------
        intensities : np.ndarray
            Shape (n_fluorophores, n_wavelengths)
        names : List[str]
            Fluorophore names
        wavelength_nm : np.ndarray
            Wavelength axis in nm
        """
        if filter_emission:
            cols = [col for col in self.fluorophore_cols if "em" in col.lower()]
        else:
            cols = self.fluorophore_cols

        intensities = []
        names = []

        for col in cols:
            spectrum = np.array(self.df[col].values, dtype=float)

            if np.all(np.isnan(spectrum)) or np.all(spectrum == 0):
                continue

            spectrum = np.nan_to_num(spectrum, nan=0.0)

            if normalize and spectrum.max() > 0:
                spectrum = spectrum / spectrum.max()

            intensities.append(spectrum)
            names.append(col)

        return np.array(intensities), names, np.array(self.wavelength_nm, dtype=float)

    def to_xarray(
        self,
        use_wavenumber: bool = True,
        crop_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
    ) -> xr.Dataset:
        """
        Convert to xarray Dataset compatible with SyntheticBleachingDataset.

        Parameters
        ----------
        use_wavenumber : bool
            Convert wavelength to Raman shift (cm⁻¹)
        crop_range : tuple of (min, max), optional
            Range to crop (wavenumber if use_wavenumber, else wavelength)
        normalize : bool
            Normalize each spectrum to [0, 1]

        Returns
        -------
        xr.Dataset
            Dataset with dimensions (sample, wavenumber/wavelength)
        """
        intensities, names, wavelength_nm = self.get_emission_spectra(
            filter_emission=True,
            normalize=normalize,
        )

        n_fluorophores = len(names)

        if use_wavenumber:
            axis = nm_to_wavenumber(wavelength_nm, self.laser_nm)
            axis_name = "wavenumber"
            axis_units = "cm⁻¹"

            sort_idx = np.argsort(axis)
            axis = axis[sort_idx]
            intensities = intensities[:, sort_idx]
        else:
            axis = wavelength_nm
            axis_name = "wavelength"
            axis_units = "nm"

        if crop_range is not None:
            min_val, max_val = crop_range
            mask = (axis >= min_val) & (axis <= max_val)
            axis = axis[mask]
            intensities = intensities[:, mask]

        ds = xr.Dataset(
            {
                "intensity": ((axis_name, "sample"), intensities.T),
                "fluorophore_name": (("sample",), names),
            },
            coords={
                axis_name: axis,
                "sample": np.arange(n_fluorophores),
            },
            attrs={
                "source": str(self.csv_path),
                "laser_nm": self.laser_nm,
                "axis_units": axis_units,
            },
        )

        return ds

    def list_fluorophores(self) -> List[str]:
        """List all available fluorophore names."""
        _, names, _ = self.get_emission_spectra(filter_emission=False, normalize=False)
        return names


def filter_bad_fluorophores(
    ds: xr.Dataset,
    min_max_intensity: float = 0.1,
    min_std: float = 0.05,
    min_range: float = 0.1,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Filter out low-quality fluorophore spectra.

    Parameters
    ----------
    ds : xr.Dataset
        Fluorophore dataset
    min_max_intensity : float
        Minimum peak intensity
    min_std : float
        Minimum standard deviation
    min_range : float
        Minimum dynamic range

    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    n_original = len(ds["sample"])
    good_indices = []

    for i in range(n_original):
        spectrum = ds["intensity"].isel(sample=i).values
        name = ds["fluorophore_name"].isel(sample=i).values

        max_val = spectrum.max()
        std_val = spectrum.std()
        range_val = spectrum.max() - spectrum.min()

        is_good = (
            max_val >= min_max_intensity
            and std_val >= min_std
            and range_val >= min_range
        )

        if is_good:
            good_indices.append(i)
        elif verbose:
            print(f"  Filtered: {name} (max={max_val:.3f}, std={std_val:.3f})")

    if verbose:
        print(f"Kept {len(good_indices)}/{n_original} fluorophores")

    if len(good_indices) == 0:
        raise ValueError("All fluorophores filtered out")

    return ds.isel(sample=good_indices)


def load_fluorophores(
    csv_path: str,
    laser_nm: float = 532.0,
    crop_range: Optional[Tuple[float, float]] = None,
    use_wavenumber: bool = True,
    filter_bad: bool = True,
    min_quality: float = 0.1,
) -> xr.Dataset:
    """
    Load fluorophore spectra from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV file from fpbase.org or fluorophores.org
    laser_nm : float
        Laser wavelength in nm
    crop_range : tuple of (min, max), optional
        Crop range in cm⁻¹ (if use_wavenumber) or nm
    use_wavenumber : bool
        Convert to wavenumber (Raman shift)
    filter_bad : bool
        Filter out low-quality fluorophores
    min_quality : float
        Minimum quality threshold

    Returns
    -------
    xr.Dataset
        Fluorophore emission spectra

    Examples
    --------
    >>> fluor_ds = load_fluorophores(
    ...     'fpbase_spectra.csv',
    ...     laser_nm=532.0,
    ...     crop_range=(400, 1800),
    ... )
    """
    loader = FluorophoreLoader(csv_path, laser_nm=laser_nm)
    ds = loader.to_xarray(
        use_wavenumber=use_wavenumber,
        crop_range=crop_range,
        normalize=False,
    )

    if filter_bad:
        ds = filter_bad_fluorophores(
            ds,
            min_max_intensity=min_quality,
            min_std=min_quality * 0.5,
            min_range=min_quality,
            verbose=True,
        )

    return ds
