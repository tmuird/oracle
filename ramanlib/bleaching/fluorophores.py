"""Fluorophore spectra loading and generation."""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Optional, Tuple


def nm_to_wavenumber(wavelength_nm: np.ndarray, laser_nm: float = 532.0) -> np.ndarray:
    """Convert wavelength (nm) to Raman shift (cm⁻¹)."""
    laser_wn = 1e7 / laser_nm
    emission_wn = 1e7 / wavelength_nm
    return laser_wn - emission_wn


def wavenumber_to_nm(wavenumber: np.ndarray, laser_nm: float = 532.0) -> np.ndarray:
    """Convert Raman shift (cm⁻¹) to wavelength (nm)."""
    laser_wn = 1e7 / laser_nm
    emission_wn = laser_wn - wavenumber
    return 1e7 / emission_wn


class FluorophoreLoader:
    """Load fluorophore spectra from CSV files."""

    def __init__(self, csv_path: str, laser_nm: float = 532.0):
        self.csv_path = Path(csv_path)
        self.laser_nm = laser_nm
        self.df = pd.read_csv(csv_path)
        self.wavelength_col = self.df.columns[0]
        self.wavelength_nm = self.df[self.wavelength_col].values
        self.fluorophore_cols = list(self.df.columns[1:])

    def get_emission_spectra(
        self, filter_emission: bool = True, normalize: bool = True
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Extract emission spectra from CSV."""
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
        """Convert to xarray Dataset."""
        intensities, names, wavelength_nm = self.get_emission_spectra(
            filter_emission=True, normalize=normalize
        )

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

        return xr.Dataset(
            {
                "intensity": (("sample", axis_name), intensities),
                "fluorophore_name": (("sample",), names),
            },
            coords={axis_name: axis, "sample": np.arange(len(names))},
            attrs={"source": str(self.csv_path), "laser_nm": self.laser_nm, "axis_units": axis_units},
        )


def filter_bad_fluorophores(
    ds: xr.Dataset,
    min_max_intensity: float = 0.1,
    min_std: float = 0.05,
    min_range: float = 0.1,
    verbose: bool = True,
) -> xr.Dataset:
    """Filter out low-quality fluorophore spectra."""
    good_indices = []
    for i in range(len(ds["sample"])):
        spectrum = ds["intensity"].isel(sample=i).values
        if (
            spectrum.max() >= min_max_intensity
            and spectrum.std() >= min_std
            and (spectrum.max() - spectrum.min()) >= min_range
        ):
            good_indices.append(i)
        elif verbose:
            name = ds["fluorophore_name"].isel(sample=i).values
            print(f"Filtered: {name}")

    if verbose:
        print(f"Kept {len(good_indices)}/{len(ds['sample'])} fluorophores")
    if len(good_indices) == 0:
        raise ValueError("All fluorophores filtered out")
    return ds.isel(sample=good_indices)


def generate_synthetic_fluorophores(
    n_fluorophores: int = 100,
    laser_nm: float = 532.0,
    min_excitation_efficiency: float = 0.05,
    min_separation: float = 15.0,
    wavelength_range: Tuple[float, float] = (350, 750),
    n_wavelength_points: int = 400,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate realistic synthetic fluorophore spectra."""
    if seed is not None:
        np.random.seed(seed)

    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelength_points)
    all_data = []

    for i in range(n_fluorophores):
        ex_max = np.random.uniform(400, 650)
        stokes_shift = np.clip(np.random.lognormal(np.log(25), 0.5), 10, 80)
        em_max = ex_max + stokes_shift
        ex_fwhm = np.random.uniform(30, 80)
        em_fwhm = np.random.uniform(35, 90)
        quantum_yield = np.random.beta(5, 2)
        extinction = np.clip(np.random.lognormal(np.log(70000), 0.6), 10000, 250000)
        bleaching_rate = np.clip(
            np.random.lognormal(np.log(0.1), 1.5) * (1.5 - quantum_yield), 0.001, 3.0
        )

        sigma_ex = ex_fwhm / 2.355
        excitation_efficiency = np.exp(-((laser_nm - ex_max) ** 2) / (2 * sigma_ex**2))

        if excitation_efficiency < min_excitation_efficiency or abs(em_max - laser_nm) < min_separation:
            continue

        sigma_em = em_fwhm / 2.355
        emission_spectrum = np.exp(-((wavelengths - em_max) ** 2) / (2 * sigma_em**2))
        emission_spectrum = emission_spectrum / np.max(emission_spectrum)
        wavenumbers = nm_to_wavenumber(wavelengths, laser_nm)

        for wl, wn, intensity in zip(wavelengths, wavenumbers, emission_spectrum):
            all_data.append(
                {
                    "fluorophore_id": f"FL{i:03d}",
                    "wavelength_nm": wl,
                    "wavenumber_cm": wn,
                    "emission_intensity": intensity,
                    "ex_max": ex_max,
                    "em_max": em_max,
                    "stokes_shift": stokes_shift,
                    "quantum_yield": quantum_yield,
                    "extinction_coeff": extinction,
                    "bleaching_rate_s": bleaching_rate,
                    "excitation_efficiency": excitation_efficiency,
                }
            )

    df = pd.DataFrame(all_data)
    if len(df) == 0:
        raise ValueError(
            f"No suitable fluorophores generated for {laser_nm}nm. "
            "Try increasing n_fluorophores or relaxing filters."
        )
    return df


def _dataframe_to_xarray(
    df: pd.DataFrame,
    laser_nm: float,
    use_wavenumber: bool,
    crop_range: Optional[Tuple[float, float]],
) -> xr.Dataset:
    """Convert fluorophore DataFrame to xarray Dataset."""
    unique_ids = df["fluorophore_id"].unique()
    first_fl = df[df["fluorophore_id"] == unique_ids[0]]

    axis = first_fl["wavenumber_cm" if use_wavenumber else "wavelength_nm"].values
    axis_name = "wavenumber" if use_wavenumber else "wavelength"
    axis_units = "cm⁻¹" if use_wavenumber else "nm"

    if crop_range is not None:
        mask = (axis >= crop_range[0]) & (axis <= crop_range[1])
        axis = axis[mask]

    intensities = np.zeros((len(unique_ids), len(axis)))
    names = []
    ex_max_list = []
    em_max_list = []
    bleaching_rate_list = []
    quantum_yield_list = []

    for i, fl_id in enumerate(unique_ids):
        fl_data = df[df["fluorophore_id"] == fl_id]
        fl_axis = fl_data["wavenumber_cm" if use_wavenumber else "wavelength_nm"].values
        fl_intensity = fl_data["emission_intensity"].values

        if crop_range is not None:
            mask = (fl_axis >= crop_range[0]) & (fl_axis <= crop_range[1])
            fl_intensity = fl_intensity[mask]

        intensities[i, :] = fl_intensity
        names.append(fl_id)
        props = fl_data.iloc[0]
        ex_max_list.append(props["ex_max"])
        em_max_list.append(props["em_max"])
        bleaching_rate_list.append(props["bleaching_rate_s"])
        quantum_yield_list.append(props["quantum_yield"])

    return xr.Dataset(
        {
            "intensity": (("sample", axis_name), intensities),
            "fluorophore_name": (("sample",), names),
            "ex_max": (("sample",), ex_max_list),
            "em_max": (("sample",), em_max_list),
            "bleaching_rate": (("sample",), bleaching_rate_list),
            "quantum_yield": (("sample",), quantum_yield_list),
        },
        coords={axis_name: axis, "sample": np.arange(len(unique_ids))},
        attrs={"source": "synthetic", "laser_nm": laser_nm, "axis_units": axis_units},
    )


def load_fluorophores(
    csv_path: Optional[str] = None,
    laser_nm: float = 532.0,
    crop_range: Optional[Tuple[float, float]] = None,
    use_wavenumber: bool = True,
    filter_bad: bool = True,
    min_quality: float = 0.1,
    n_synthetic: Optional[int] = None,
    synthetic_seed: Optional[int] = None,
) -> xr.Dataset:
    """
    Load fluorophore spectra from CSV or generate synthetic fluorophores.

    Parameters
    ----------
    csv_path : str, optional
        Path to CSV file. If None, generates synthetic fluorophores.
    laser_nm : float
        Laser wavelength in nm
    crop_range : tuple, optional
        (min, max) range to crop
    use_wavenumber : bool
        Use wavenumber axis instead of wavelength
    filter_bad : bool
        Filter low-quality fluorophores (CSV mode only)
    min_quality : float
        Minimum quality threshold (CSV mode only)
    n_synthetic : int, optional
        Number of synthetic fluorophores to generate
    synthetic_seed : int, optional
        Random seed for synthetic generation

    Returns
    -------
    xr.Dataset
        Fluorophore emission spectra
    """
    if csv_path is None:
        if n_synthetic is None:
            raise ValueError("Either csv_path or n_synthetic must be specified")

        df = generate_synthetic_fluorophores(
            n_fluorophores=n_synthetic * 4,
            laser_nm=laser_nm,
            min_excitation_efficiency=0.05,
            min_separation=15.0,
            seed=synthetic_seed,
        )

        ds = _dataframe_to_xarray(df, laser_nm, use_wavenumber, crop_range)

        if len(ds.sample) > n_synthetic:
            indices = np.random.choice(len(ds.sample), size=n_synthetic, replace=False)
            ds = ds.isel(sample=indices)
        return ds
    else:
        loader = FluorophoreLoader(csv_path, laser_nm=laser_nm)
        ds = loader.to_xarray(use_wavenumber=use_wavenumber, crop_range=crop_range, normalize=False)
        if filter_bad:
            ds = filter_bad_fluorophores(
                ds,
                min_max_intensity=min_quality,
                min_std=min_quality * 0.5,
                min_range=min_quality,
                verbose=True,
            )
        return ds
