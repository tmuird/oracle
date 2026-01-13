# synthetic_bleaching.py

from typing_extensions import Literal
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.interpolate import UnivariateSpline


@dataclass
class SyntheticConfig:
    """Configuration for synthetic photobleaching dataset generation."""

    # Dataset size
    n_samples: int = 5000

    laser_nm: float = 532.0  # Laser wavelength in nm
    # Temporal parameters (cumulative laser exposure)
    # Option 1: Explicit time points
    bleaching_times: Optional[List[float]] = None

    # Option 2: Generate from interval (if bleaching_times is None)
    bleaching_interval: float = 0.1  # seconds
    bleaching_max_time: float = 10.0  # seconds

    # Integration time(s) to sample from ATCC data
    # These are the acquisition settings - only spectra at these integration times are used
    integration_times: List[str] = field(default_factory=lambda: ["1s"])

    # Fluorophore parameters
    n_fluorophores: int = 3

    # Decay rates: uniform sampling from [min, max] in s⁻¹
    # Default ranges chosen so fluorescence decay is visible over typical time scales
    decay_rate_min: float = 0.1  # s⁻¹ (τ = 10s, slow decay)
    decay_rate_max: float = 5.0  # s⁻¹ (τ = 0.2s, fast decay)

    # Abundances: relative weights of each fluorophore
    # Will be normalized to achieve target F/R ratio
    abundance_min: float = 0.1
    abundance_max: float = 2.0

    # Fluorescence-to-Raman ratio at t=0
    fr_ratio_min: float = 3.0
    fr_ratio_max: float = 15.0

    # Noise parameters
    poisson_noise_scale: float = 1.0  # Poisson (shot) noise scaling factor
    gaussian_noise_scale: float = (
        0.02  # Gaussian (read) noise std as fraction of signal mean
    )
    noise_type: str = "poisson_gaussian"  # 'gaussian', 'poisson_gaussian', or 'none'

    # Fluorophore basis generation
    basis_type: str = "gaussian_mixture"
    shared_bases: bool = True  # If True, same bases for all samples
    shared_axis: bool = True # If True, same wavenumber axis for all samples
    # Fluorophore variation (for real spectra)
    # Simulates variation due to pH, concentration, quenching, etc.
    fluorophore_variation: float = 0.0  # 10% random intensity variation

    interpolation_method: Literal["linear", "polynomial", "spline"] = (
        "polynomial"  # Method for interpolating real fluorophore spectra so they match wavenumber axis
    )

    polynomial_degree: int = (
        3  # Degree of polynomial for interpolation of real fluorophore spectra if using polynomial method
    )
    # Random seed
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters."""
        if self.decay_rate_min <= 0 or self.decay_rate_max <= 0:
            raise ValueError("Decay rates must be positive")
        if self.decay_rate_min > self.decay_rate_max:
            raise ValueError("decay_rate_min must be <= decay_rate_max")
        if self.fr_ratio_min > self.fr_ratio_max:
            raise ValueError("fr_ratio_min must be <= fr_ratio_max")
        if self.n_fluorophores < 1:
            raise ValueError("n_fluorophores must be at least 1")
        if self.poisson_noise_scale < 0:
            raise ValueError("poisson_noise_scale must be non-negative")
        if self.gaussian_noise_scale < 0:
            raise ValueError("gaussian_noise_scale must be non-negative")

    def __post_init__(self):
        self.validate()


class SyntheticBleachingDataset:
    """
    Generate synthetic photobleaching time series with known ground truth.

    Uses real ATCC Raman spectra (with per-sample wavenumber axes) and adds
    synthetic fluorescence decay.
    """

    def __init__(
        self,
        config: SyntheticConfig,
        atcc_xr: xr.Dataset,  # xarray Dataset from atcc_dataset.to_xarray()
        fluorophore_xr: Optional[xr.Dataset] = None,  # Real fluorophore spectra
    ):
        """
        Parameters
        ----------
        config : SyntheticConfig
            Dataset generation configuration
        atcc_xr : xr.Dataset
            xarray Dataset from atcc_dataset.to_xarray()
            User controls preprocessing (crop, despike, etc.) before passing
        fluorophore_xr : xr.Dataset, optional
            xarray Dataset of real fluorophore emission spectra
            If provided, uses real spectra as bases instead of synthetic
            If None, generates synthetic bases (default)
        """
        self.config = config
        self.atcc_xr = atcc_xr
        self.fluorophore_xr = fluorophore_xr
        self.rng = np.random.default_rng(config.seed)

        # Determine bleaching time points
        if config.bleaching_times is not None:
            self.bleaching_times = np.array(config.bleaching_times, dtype=float)
        else:
            # Generate from interval
            n_times = int(config.bleaching_max_time / config.bleaching_interval) + 1
            self.bleaching_times = np.linspace(0, config.bleaching_max_time, n_times)

        print(f"Bleaching time points: {self.bleaching_times}")

        # Filter to requested integration times and use late frames for pure Raman
        # For simplicity, use the longest integration time as "pure Raman"
        late_time = config.integration_times[-1] if config.integration_times else "1s"

        self.raman_spectra = atcc_xr.sel(integration_time=late_time)

        print(f"\nUsing integration time '{late_time}' for Raman extraction")
        print(f"Available samples: {len(self.raman_spectra['sample'])}")
        print(f"Wavenumber axes shape: {self.raman_spectra['wavenumber'].shape}")

        # Store wavenumber axes (per-sample if available)
        self.wavenumbers = self.raman_spectra["wavenumber"].values

        # Check if wavenumbers are per-sample or shared
        if self.wavenumbers.ndim == 1:
            # Shared axis: expand to (n_atcc_samples, n_wavenumbers)
            n_atcc_samples = len(self.raman_spectra["sample"])
            self.wavenumbers = np.tile(self.wavenumbers, (n_atcc_samples, 1))
            print(
                f"Wavenumber axis: shared across all ATCC samples (now expanded) to shape {self.wavenumbers.shape}"
            )
        else:
            print(
                f"Wavenumber axis: per-sample (calibration drift preserved) with shape {self.wavenumbers.shape}"
            )

        # Generate shared fluorophore bases if configured
        # Use first sample's wavenumber axis as reference for basis generation
        ref_wavenumbers = (
            self.wavenumbers[0] if self.wavenumbers.ndim == 2 else self.wavenumbers
        )

        if config.shared_bases:
            self.shared_bases = self._generate_fluorophore_bases(ref_wavenumbers)
        else:
            self.shared_bases = None

        # Storage
        self.dataset: Optional[xr.Dataset] = None

    def _generate_fluorophore_bases(self, wavenumbers: np.ndarray) -> np.ndarray:
        """
        Generate fluorophore basis spectra.

        If fluorophore_xr is provided, samples from real spectra.
        Otherwise, generates synthetic bases using Gaussian mixtures.

        Parameters
        ----------
        wavenumbers : np.ndarray
            Wavenumber axis for basis generation

        Returns
        -------
        bases : np.ndarray
            Shape (n_fluorophores, n_wavenumbers)
        """
        n_f = self.config.n_fluorophores

        # Use real fluorophore spectra if provided
        if self.fluorophore_xr is not None:
            return self._sample_real_fluorophores(wavenumbers)

        # Otherwise generate synthetic bases
        n_wn = len(wavenumbers)
        wn = wavenumbers

        bases = np.zeros((n_f, n_wn))

        # Different characteristic widths for different decay components
        width_multipliers = [1.5, 1.0, 0.7]  # Fast (broad), medium, slow (narrower)

        for i in range(n_f):
            n_gaussians = self.rng.integers(2, 5)

            for _ in range(n_gaussians):
                # Random center
                margin = 0.1 * (wn.max() - wn.min())
                center = self.rng.uniform(wn.min() + margin, wn.max() - margin)

                # Width based on fluorophore type
                base_width = self.rng.uniform(50, 150)  # cm⁻¹
                width = base_width * width_multipliers[i % len(width_multipliers)]

                amplitude = self.rng.uniform(0.5, 1.5)
                bases[i] += amplitude * np.exp(-0.5 * ((wn - center) / width) ** 2)

            # Ensure positivity and normalize to unit peak
            bases[i] = np.maximum(bases[i], 1e-6)
            if bases[i].max() > 0:
                bases[i] /= bases[i].max()

        return bases

    def _sample_real_fluorophores(self, wavenumbers: np.ndarray) -> np.ndarray:
        """
        Sample real fluorophore spectra from fluorophore_xr dataset.

        Adds realistic variation to simulate different conditions
        (pH, concentration, quenching, etc.)

        Parameters
        ----------
        wavenumbers : np.ndarray
            Target wavenumber axis (cm⁻¹)

        Returns
        -------
        bases : np.ndarray
            Shape (n_fluorophores, n_wavenumbers)
        """
        print("Sampling real fluorophore spectra for bases...")
        assert self.fluorophore_xr is not None, "fluorophore_xr must be provided"

        n_f = self.config.n_fluorophores
        fluor_ds = self.fluorophore_xr

        # Get available fluorophores
        n_available = len(fluor_ds["sample"])

        if n_f > n_available:
            # If requesting more fluorophores than available, sample with replacement
            indices = self.rng.choice(n_available, size=n_f, replace=True)
            print(
                f"  Warning: Requested {n_f} fluorophores but only {n_available} available. Sampling with replacement."
            )
        else:
            # Sample without replacement
            indices = self.rng.choice(n_available, size=n_f, replace=False)

        bases = np.zeros((n_f, len(wavenumbers)))

        # Get fluorophore wavenumber axis
        if "wavenumber" in fluor_ds.coords:
            fluor_wn = fluor_ds["wavenumber"].values
            print(f"  Fluorophore wavenumber axis found with shape {fluor_wn.shape}")
        elif "wavelength" in fluor_ds.coords:
            from fluorophore_loader import nm_to_wavenumber

            fluor_wavelength = fluor_ds["wavelength"].values
            fluor_wn = nm_to_wavenumber(fluor_wavelength, laser_nm=self.config.laser_nm)
            print(
                f"  Fluorophore wavelength axis found, converted to wavenumber (shape {fluor_wn.shape})"
            )
        else:
            raise ValueError(
                "Fluorophore dataset must have 'wavenumber' or 'wavelength' coordinate"
            )

        for i, idx in enumerate(indices):
            # Get spectrum
            spectrum = fluor_ds["intensity"].isel(sample=idx).values

            # Interpolate to target wavenumbers if needed

            if not np.array_equal(fluor_wn, wavenumbers):
                # Ensure both arrays are sorted in same direction
                if (np.diff(fluor_wn) < 0).any() != (np.diff(wavenumbers) < 0).any():
                    # Flip if directions don't match
                    spectrum = spectrum[::-1]
                    fluor_wn_sorted = fluor_wn[::-1]
                else:
                    fluor_wn_sorted = fluor_wn

                if self.config.interpolation_method == "polynomial":
                    # Polynomial fit (actual polynomial, not spline)
                    coeffs = np.polyfit(
                        fluor_wn_sorted, spectrum, deg=self.config.polynomial_degree
                    )
                    poly = np.poly1d(coeffs)
                    spectrum = poly(wavenumbers)
                elif self.config.interpolation_method == "spline":
                    # Smoothing spline with automatic smoothing
                    spline = UnivariateSpline(
                        fluor_wn_sorted, spectrum, k=3, s=0
                    )  # s=0 for interpolating spline
                    spectrum = spline(wavenumbers)
                else:  # linear
                    spectrum = np.interp(
                        wavenumbers, fluor_wn_sorted, spectrum, left=0.0, right=0.0
                    )

            # Add realistic variation to simulate different conditions
            if self.config.fluorophore_variation > 0:
                # Random intensity scaling (concentration variation)
                intensity_scale = self.rng.normal(
                    1.0, self.config.fluorophore_variation
                )
                intensity_scale = np.clip(intensity_scale, 0.5, 1.5)  # Limit range
                spectrum = spectrum * intensity_scale

                # Spectral noise (measurement variation)
                noise_scale = (
                    self.config.fluorophore_variation * 0.5
                )  # Half the intensity variation
                noise = self.rng.normal(
                    0, noise_scale * spectrum.mean(), len(wavenumbers)
                )
                spectrum = spectrum + noise

                # Small baseline offset (background variation)
                baseline_offset = self.rng.uniform(0, 0.05 * spectrum.max())
                spectrum = spectrum + baseline_offset

            # Ensure positivity
            spectrum = np.maximum(spectrum, 0)

            # Normalize to unit peak
            if spectrum.max() > 0:
                spectrum = spectrum / spectrum.max()
            else:
                spectrum = np.ones_like(spectrum) * 1e-6

            bases[i] = spectrum

        return bases

    def _generate_decay_rates(self) -> np.ndarray:
        """Sample decay rates uniformly from configured range."""
        n_f = self.config.n_fluorophores
        return self.rng.uniform(
            self.config.decay_rate_min, self.config.decay_rate_max, size=n_f
        )

    def _generate_abundances(self, raman_spectrum: np.ndarray) -> np.ndarray:
        """
        Generate abundances ensuring proper F/R ratio at t=0.

        Parameters
        ----------
        raman_spectrum : np.ndarray
            Raman spectrum (full spectrum, not just mean)

        Returns
        -------
        abundances : np.ndarray
            Shape (n_fluorophores,)
        """
        n_f = self.config.n_fluorophores

        # Sample F/R ratio
        # Since fluorophores are normalized to unit peak, use peak Raman for consistency
        fr_ratio = self.rng.uniform(self.config.fr_ratio_min, self.config.fr_ratio_max)
        raman_peak = raman_spectrum.max()
        target_fluor_total = fr_ratio * raman_peak

        # Sample raw abundances uniformly
        raw_abundances = self.rng.uniform(
            self.config.abundance_min, self.config.abundance_max, size=n_f
        )

        # Get bases (use shared or generate new)
        if self.shared_bases is not None:
            bases = self.shared_bases
        else:
            # For per-sample bases, we'll need wavenumbers
            # This will be handled in the main loop
            raise ValueError("Per-sample bases require wavenumber axis")

        # Scale abundances to achieve target F/R ratio
        # Since bases are normalized to unit peak (max=1.0), use max for scaling
        # fluorescence_total = sum(w_i * max(B_i))
        basis_maxs = bases.max(axis=1)
        current_total = np.sum(raw_abundances * basis_maxs)

        if current_total > 0:
            abundances = raw_abundances * (target_fluor_total / current_total)
        else:
            abundances = raw_abundances

        return abundances

    def _add_noise(
        self,
        raman: np.ndarray,
        fluorescence: np.ndarray,
    ) -> np.ndarray:
        """
        Add realistic noise to clean signal.

        Parameters
        ----------
        raman : np.ndarray
            Clean Raman signal
        fluorescence : np.ndarray
            Clean fluorescence signal

        Returns
        -------
        noisy : np.ndarray
            Noisy signal
        """
        signal = raman + fluorescence

        if self.config.noise_type == "none":
            return signal

        elif self.config.noise_type == "gaussian":
            # Gaussian read noise only
            noise_std = signal.mean() * self.config.gaussian_noise_scale
            noise = self.rng.normal(0, noise_std, signal.shape)
            return signal + noise

        elif self.config.noise_type == "poisson_gaussian":
            # Poisson (shot noise) + Gaussian (read noise)

            # Poisson noise (signal-dependent, photon counting)
            signal_mean = signal.mean()
            if signal_mean > 0:
                # Scale signal by poisson_noise_scale to control shot noise level
                # Higher scale = more photons = less relative noise
                scaled_signal = np.maximum(signal * self.config.poisson_noise_scale, 0)
                poisson_counts = self.rng.poisson(scaled_signal)
                poisson_signal = poisson_counts / self.config.poisson_noise_scale
            else:
                poisson_signal = signal

            # Gaussian read noise (signal-independent, detector noise)
            read_noise_std = signal.mean() * self.config.gaussian_noise_scale
            read_noise = self.rng.normal(0, read_noise_std, signal.shape)

            return poisson_signal + read_noise

        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")

    def _reconstruct_time_series(
        self,
        raman: np.ndarray,
        bases: np.ndarray,
        abundances: np.ndarray,
        decay_rates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct bleaching time series from parameters.

        The fluorescence decays according to:
            F_i(t) = w_i * B_i(ν) * exp(-λ_i * t_bleach)

        where t_bleach is the cumulative laser exposure time.

        Returns
        -------
        noisy : np.ndarray
            Noisy time series (n_times, n_wavenumbers)
        clean : np.ndarray
            Clean time series (n_times, n_wavenumbers)
        """
        n_t = len(self.bleaching_times)
        n_wn = len(raman)

        # Raman contribution (constant across all time points)
        clean = np.tile(raman, (n_t, 1))

        # Fluorescence contributions (decay over bleaching time)
        fluorescence_time_series = np.zeros((n_t, n_wn))

        for i in range(len(decay_rates)):
            decay = np.exp(-decay_rates[i] * self.bleaching_times)
            fluor_i = abundances[i] * decay[:, None] * bases[i, None, :]
            clean += fluor_i
            fluorescence_time_series += fluor_i

        # Add noise (per timepoint)
        noisy = np.zeros_like(clean)
        for t in range(n_t):
            noisy[t] = self._add_noise(raman, fluorescence_time_series[t])

        return noisy, clean

    def generate(self) -> xr.Dataset:
        """
        Generate the full synthetic dataset.

        Returns
        -------
        xr.Dataset
            Dataset with bleaching time series and ground truth parameters.

            Key dimensions:
            - 'bleaching_time': Time points in the photobleaching series
            - 'sample': Individual spectra/samples
            - 'wavenumber': Spectral axis (per-sample if calibration drift)

            Key coordinates:
            - 'bleaching_time_seconds': Actual bleaching times in seconds
        """
        n_samples = self.config.n_samples
        n_times = len(self.bleaching_times)
        n_f = self.config.n_fluorophores

        n_wn = len(
            self.wavenumbers[0] if self.wavenumbers.ndim == 2 else self.wavenumbers
        )
        intensity_noisy = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        intensity_clean = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        raman_gt = np.zeros((n_samples, n_wn), dtype=np.float32)
        wavenumbers_all = np.zeros((n_samples, n_wn), dtype=np.float32)

        # Get number of ATCC samples available
        n_atcc_samples = len(self.raman_spectra["sample"])

        # Preallocate arrays

        decay_rates_gt = np.zeros((n_samples, n_f), dtype=np.float32)
        abundances_gt = np.zeros((n_samples, n_f), dtype=np.float32)

        # Metadata
        species_list = []

        # Shared bases storage
        if self.shared_bases is not None:
            bases_storage = self.shared_bases  # (n_f, n_wn)
        else:
            bases_storage = []

        print(f"\nGenerating {n_samples} synthetic samples...")

        for i in range(n_samples):
            # Sample a random ATCC spectrum
            atcc_idx = self.rng.integers(0, n_atcc_samples)

            # Get Raman spectrum and wavenumbers
            raman = self.raman_spectra["intensity_raw"].isel(sample=atcc_idx).values

            if self.wavenumbers.ndim == 2:
                wn = self.wavenumbers[atcc_idx]
            else:
                wn = self.wavenumbers

            # Get species label if available
            if "species" in self.raman_spectra:
                species = str(
                    self.raman_spectra["species"].isel(sample=atcc_idx).values
                )
            else:
                species = "Unknown"

            # Generate fluorescence parameters
            decay_rates = self._generate_decay_rates()
            abundances = self._generate_abundances(raman)

            # Get bases
            if self.shared_bases is not None:
                bases = self.shared_bases
            else:
                bases = self._generate_fluorophore_bases(wn)
                bases_storage.append(bases)

            # Reconstruct time series
            noisy, clean = self._reconstruct_time_series(
                raman, bases, abundances, decay_rates
            )

            # Store
            intensity_noisy[i] = noisy
            intensity_clean[i] = clean
            raman_gt[i] = raman
            wavenumbers_all[i] = wn
            decay_rates_gt[i] = decay_rates
            abundances_gt[i] = abundances
            species_list.append(species)

            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{n_samples}")

        # Convert lists to arrays
        intensity_noisy = np.array(intensity_noisy, dtype=np.float32)
        intensity_clean = np.array(intensity_clean, dtype=np.float32)
        raman_gt = np.array(raman_gt, dtype=np.float32)
        wavenumbers_all = np.array(wavenumbers_all, dtype=np.float32)

        # Create formatted labels for bleaching times
        # bleaching_time_labels = [f"{t:.3g}s" for t in self.bleaching_times]

        # Build xarray Dataset
        ds = xr.Dataset(
            data_vars={
                # Main data
                "intensity_raw": (
                    ["sample", "bleaching_time", "wavenumber"],
                    intensity_noisy,
                    {
                        "long_name": "Synthetic Raman intensity (noisy)",
                        "units": "counts",
                    },
                ),
                "intensity_clean": (
                    ["sample", "bleaching_time", "wavenumber"],
                    intensity_clean,
                    {
                        "long_name": "Synthetic Raman intensity (clean)",
                        "units": "counts",
                    },
                ),
                # Ground truth parameters
                "raman_gt": (
                    ["sample", "wavenumber"],
                    raman_gt,
                    {"long_name": "Ground truth Raman spectrum"},
                ),
                "decay_rates_gt": (
                    ["sample", "fluorophore"],
                    decay_rates_gt,
                    {"long_name": "Ground truth decay rates", "units": "s⁻¹"},
                ),
                "abundances_gt": (
                    ["sample", "fluorophore"],
                    abundances_gt,
                    {"long_name": "Ground truth abundances"},
                ),
                # Per-sample wavenumber axes
                "wavenumber": (
                    ["sample", "wavenumber"],
                    wavenumbers_all,
                    {"long_name": "Wavenumber axis (per-sample)", "units": "cm⁻¹"},
                ),
                # Metadata
                "species": (["sample"], species_list),
            },
            coords={
                "sample": np.arange(n_samples),
                "bleaching_time": self.bleaching_times,
                # 'bleaching_time_seconds': ('bleaching_time', self.bleaching_times),
            },
            attrs={
                "title": "Synthetic Photobleaching Dataset",
                "n_samples": n_samples,
                "n_fluorophores": n_f,
                "shared_bases": self.config.shared_bases,
                "noise_type": self.config.noise_type,
                "poisson_noise_scale": self.config.poisson_noise_scale,
                "gaussian_noise_scale": self.config.gaussian_noise_scale,
                "fr_ratio_range": f"{self.config.fr_ratio_min}-{self.config.fr_ratio_max}",
                "decay_rate_range": f"{self.config.decay_rate_min}-{self.config.decay_rate_max} s⁻¹",
                "seed": self.config.seed,
            },
        )

        # Add fluorophore bases
        if self.config.shared_bases:
            ds["fluorophore_bases_gt"] = (
                ["fluorophore", "wavenumber"],
                bases_storage,
                {"long_name": "Shared fluorophore basis spectra"},
            )
        else:
            bases_storage = np.array(bases_storage, dtype=np.float32)
            ds["fluorophore_bases_gt"] = (
                ["sample", "fluorophore", "wavenumber"],
                bases_storage,
                {"long_name": "Per-sample fluorophore basis spectra"},
            )

        self.dataset = ds

        print("\nGenerated dataset:")
        print(f"  Samples: {n_samples}")
        print(f"  Bleaching time points: {n_times}")
        print(f"  Wavenumber axis: per-sample (shape: {wavenumbers_all.shape})")
        print(f"  Fluorophores: {n_f}")
        print(
            f"  Decay rate range: [{self.config.decay_rate_min}, {self.config.decay_rate_max}] s⁻¹"
        )
        print(
            f"  F/R ratio range: [{self.config.fr_ratio_min}, {self.config.fr_ratio_max}]"
        )

        return ds

    def save(self, path: str):
        """Save dataset to NetCDF."""
        if self.dataset is None:
            raise ValueError("No dataset generated. Call generate() first.")
        self.dataset.to_netcdf(path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path: str) -> xr.Dataset:
        """Load dataset from NetCDF."""
        return xr.open_dataset(path)
