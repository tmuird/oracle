"""
Loader for fluorophore spectra from fpbase.org and fluorophores.org.

Handles CSV format with wavelength (nm) and multiple fluorophore emission/excitation spectra.
Converts to xarray Dataset compatible with SyntheticBleachingDataset.
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Optional, List, Tuple
from pathlib import Path


def nm_to_wavenumber(wavelength_nm: np.ndarray, laser_nm: float = 532.0) -> np.ndarray:
    """
    Convert wavelength (nm) to Raman shift (cm⁻¹).

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Emission wavelength in nm
    laser_nm : float
        Laser excitation wavelength in nm (default: 532 nm)

    Returns
    -------
    np.ndarray
        Raman shift in cm⁻¹
    """
    # Convert to wavenumbers (cm⁻¹)
    laser_wn = 1e7 / laser_nm
    emission_wn = 1e7 / wavelength_nm

    # Raman shift = laser_wn - emission_wn (Stokes shift)
    raman_shift = laser_wn - emission_wn

    return raman_shift


def wavenumber_to_nm(wavenumber: np.ndarray, laser_nm: float = 532.0) -> np.ndarray:
    """
    Convert Raman shift (cm⁻¹) to wavelength (nm).

    Parameters
    ----------
    wavenumber : np.ndarray
        Raman shift in cm⁻¹
    laser_nm : float
        Laser excitation wavelength in nm (default: 532 nm)

    Returns
    -------
    np.ndarray
        Emission wavelength in nm
    """
    laser_wn = 1e7 / laser_nm
    emission_wn = laser_wn - wavenumber
    wavelength_nm = 1e7 / emission_wn
    return wavelength_nm


class FluorophoreLoader:
    """
    Load fluorophore spectra from fpbase.org or fluorophores.org CSV files.

    Expected CSV format:
    - First column: wavelength (nm)
    - Other columns: fluorophore spectra (usually marked as 'em' for emission)

    Example:
        wavelength,mOrange2 em,mEGFP em,mCherry em,...
        400,0.0,0.1,0.0,...
        401,0.0,0.12,0.0,...
    """

    def __init__(self, csv_path: str, laser_nm: float = 532.0):
        """
        Parameters
        ----------
        csv_path : str
            Path to CSV file
        laser_nm : float
            Laser wavelength in nm for converting to Raman shift (default: 532 nm)
        """
        self.csv_path = Path(csv_path)
        self.laser_nm = laser_nm
        self.df = pd.read_csv(csv_path)

        # First column should be wavelength
        self.wavelength_col = self.df.columns[0]
        self.wavelength_nm = self.df[self.wavelength_col].values

        # Get fluorophore columns (exclude wavelength)
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
            If True, only include columns with 'em' in the name
        normalize : bool
            If True, normalize each spectrum to [0, 1]

        Returns
        -------
        intensities : np.ndarray
            Shape (n_fluorophores, n_wavelengths)
        names : List[str]
            Fluorophore names
        wavelength_nm : np.ndarray
            Wavelength axis in nm
        """
        # Filter columns
        if filter_emission:
            cols = [col for col in self.fluorophore_cols if 'em' in col.lower()]
        else:
            cols = self.fluorophore_cols

        # Extract spectra
        intensities = []
        names = []

        for col in cols:
            spectrum = np.array(self.df[col].values, dtype=float)

            # Skip if all NaN or all zeros
            if np.all(np.isnan(spectrum)) or np.all(spectrum == 0):
                continue

            # Replace NaN with 0
            spectrum = np.nan_to_num(spectrum, nan=0.0)

            # Normalize if requested
            if normalize and spectrum.max() > 0:
                spectrum = spectrum / spectrum.max()

            intensities.append(spectrum)
            names.append(col)

        intensities_array = np.array(intensities)
        wavelength_array = np.array(self.wavelength_nm, dtype=float)

        return intensities_array, names, wavelength_array

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
            If True, convert wavelength to Raman shift (cm⁻¹)
            If False, keep wavelength (nm)
        crop_range : tuple of (min, max), optional
            Range to crop (in wavenumber if use_wavenumber=True, else wavelength)
        normalize : bool
            Normalize each spectrum to [0, 1]

        Returns
        -------
        xr.Dataset
            Dataset with dimensions (sample, wavenumber) and variables:
            - intensity: fluorophore emission spectra
            - fluorophore_name: names of fluorophores
            - wavenumber: wavenumber axis (cm⁻¹) or wavelength (nm)
        """
        # Get emission spectra
        intensities, names, wavelength_nm = self.get_emission_spectra(
            filter_emission=True,
            normalize=normalize,
        )

        n_fluorophores = len(names)

        # Convert to wavenumber if requested
        if use_wavenumber:
            axis = nm_to_wavenumber(wavelength_nm, self.laser_nm)
            axis_name = 'wavenumber'
            axis_units = 'cm⁻¹'

            # Sort by wavenumber (may be decreasing)
            sort_idx = np.argsort(axis)
            axis = axis[sort_idx]
            intensities = intensities[:, sort_idx]
        else:
            axis = wavelength_nm
            axis_name = 'wavelength'
            axis_units = 'nm'

        # Crop if requested
        if crop_range is not None:
            min_val, max_val = crop_range
            mask = (axis >= min_val) & (axis <= max_val)
            axis = axis[mask]
            intensities = intensities[:, mask]

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                'intensity': (('sample', axis_name), intensities),
                'fluorophore_name': (('sample',), names),
            },
            coords={
                axis_name: axis,
                'sample': np.arange(n_fluorophores),
            },
            attrs={
                'source': str(self.csv_path),
                'laser_nm': self.laser_nm,
                'axis_units': axis_units,
                'description': 'Fluorophore emission spectra from fpbase.org/fluorophores.org',
            }
        )

        return ds

    def list_fluorophores(self) -> List[str]:
        """List all available fluorophore names."""
        _, names, _ = self.get_emission_spectra(filter_emission=False, normalize=False)
        return names


def load_fluorophores(
    csv_path: str,
    laser_nm: float = 532.0,
    crop_range: Optional[Tuple[float, float]] = None,
    use_wavenumber: bool = True,
) -> xr.Dataset:
    """
    Convenience function to load fluorophore spectra.

    Parameters
    ----------
    csv_path : str
        Path to CSV file from fpbase.org or fluorophores.org
    laser_nm : float
        Laser wavelength in nm (default: 532 nm for green laser)
    crop_range : tuple of (min, max), optional
        Crop range in cm⁻¹ (if use_wavenumber=True) or nm
        Example: (400, 1800) for typical Raman range
    use_wavenumber : bool
        Convert to wavenumber (Raman shift) instead of wavelength

    Returns
    -------
    xr.Dataset
        Fluorophore emission spectra as xarray Dataset

    Examples
    --------
    >>> # Load fluorophores for 532 nm laser
    >>> fluor_ds = load_fluorophores(
    ...     'fpbase_spectra.csv',
    ...     laser_nm=532.0,
    ...     crop_range=(400, 1800),  # Typical Raman range
    ... )
    >>>
    >>> # Load for NIR laser (785 nm)
    >>> fluor_ds = load_fluorophores(
    ...     'fpbase_spectra.csv',
    ...     laser_nm=785.0,
    ...     crop_range=(400, 1800),
    ... )
    """
    loader = FluorophoreLoader(csv_path, laser_nm=laser_nm)
    return loader.to_xarray(
        use_wavenumber=use_wavenumber,
        crop_range=crop_range,
        normalize=False,
    )


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load fluorophores
    csv_path = "/Users/tom/Downloads/fpbase_spectra.csv"

    # Load with wavelength (nm)
    print("=" * 60)
    print("Loading fluorophores (wavelength)")
    print("=" * 60)

    ds_nm = load_fluorophores(
        csv_path,
        use_wavenumber=False,
        crop_range=(500, 700),  # Visible range
    )

    print(f"\nLoaded {len(ds_nm['sample'])} fluorophores")
    print(f"Wavelength range: {ds_nm['wavelength'].min().values:.1f} - {ds_nm['wavelength'].max().values:.1f} nm")
    print("\nFluorophore names:")
    for name in ds_nm['fluorophore_name'].values:
        print(f"  - {name}")

    # Plot in wavelength
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(ds_nm['sample'])):
        ax.plot(
            ds_nm['wavelength'],
            ds_nm['intensity'].isel(sample=i),
            label=ds_nm['fluorophore_name'].isel(sample=i).values,
            alpha=0.7,
        )
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Fluorophore Emission Spectra')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fluorophores_wavelength.png', dpi=150)
    print("\nSaved: fluorophores_wavelength.png")
    plt.close()

    # Load with wavenumber (Raman shift)
    print("\n" + "=" * 60)
    print("Loading fluorophores (Raman shift)")
    print("=" * 60)

    ds_wn = load_fluorophores(
        csv_path,
        laser_nm=532.0,
        use_wavenumber=True,
        crop_range=(400, 1800),  # Typical Raman range
    )

    print(f"\nLoaded {len(ds_wn['sample'])} fluorophores")
    print(f"Wavenumber range: {ds_wn['wavenumber'].min().values:.1f} - {ds_wn['wavenumber'].max().values:.1f} cm⁻¹")

    # Plot in wavenumber
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(ds_wn['sample'])):
        ax.plot(
            ds_wn['wavenumber'],
            ds_wn['intensity'].isel(sample=i),
            label=ds_wn['fluorophore_name'].isel(sample=i).values,
            alpha=0.7,
        )
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Fluorophore Emission Spectra (532 nm laser)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fluorophores_wavenumber.png', dpi=150)
    print("\nSaved: fluorophores_wavenumber.png")
    plt.close()

    print("\n" + "=" * 60)
    print("✅ Fluorophore loader example completed!")
    print("=" * 60)
