"""
Synthetic photobleaching dataset generation.

Generates training data with known ground-truth for decomposition methods.
Uses real ATCC Raman spectra with synthetic fluorescence decay.
"""

from typing_extensions import Literal
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.interpolate import UnivariateSpline

from ramanlib.bleaching.physics import (
    fit_polynomial_bases,
    evaluate_polynomial_bases,
    l2_normalize,
    reconstruct_time_series,
)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic photobleaching dataset generation."""

    n_samples: int = 5000
    laser_nm: float = 532.0

    # Temporal parameters
    bleaching_times: Optional[List[float]] = None
    bleaching_interval: float = 0.1
    bleaching_max_time: float = 10.0

    # Integration time(s) to sample from ATCC data
    integration_times: List[str] = field(default_factory=lambda: ["1s"])

    # Fluorophore parameters
    n_fluorophores: int = 3

    # Decay rate sampling strategy
    decay_sampling: Literal["uniform", "log_uniform", "multi_component"] = (
        "multi_component"
    )
    decay_rate_min: float = 0.1
    decay_rate_max: float = 5.0

    # Multi-component decay ranges
    decay_slow_range: Tuple[float, float] = (0.05, 0.3)
    decay_medium_range: Tuple[float, float] = (0.3, 1.0)
    decay_fast_range: Tuple[float, float] = (1.0, 5.0)

    # Fluorophore mixing weights (relative)
    fluorophore_weight_min: float = 0.5
    fluorophore_weight_max: float = 2.0

    # F/R ratio at t=0
    fr_ratio_min: float = 3.0
    fr_ratio_max: float = 15.0

    # Noise parameters
    poisson_noise_scale: float = 1.0
    gaussian_noise_scale: float = 0.02
    noise_type: str = "poisson_gaussian"

    # Basis generation
    use_shared_bases: bool = True
    # shared_axis: bool = True
    fluorophore_variation: float = 0.0
    interpolation_method: Literal["linear", "spline"] = "spline"
    polynomial_degree: int = 3

    # Polynomial parameterization
    use_polynomial_fluorophores: bool = False
    fluorophore_polynomial_degree: int = 3

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

        if self.decay_sampling == "multi_component":
            if self.decay_slow_range[0] >= self.decay_slow_range[1]:
                raise ValueError("decay_slow_range must be (min, max) with min < max")
            if self.decay_medium_range[0] >= self.decay_medium_range[1]:
                raise ValueError("decay_medium_range must be (min, max) with min < max")
            if self.decay_fast_range[0] >= self.decay_fast_range[1]:
                raise ValueError("decay_fast_range must be (min, max) with min < max")

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
        atcc_xr: xr.Dataset,
        fluorophore_xr: Optional[xr.Dataset] = None,
    ):
        """
        Parameters
        ----------
        config : SyntheticConfig
            Dataset generation configuration
        atcc_xr : xr.Dataset
            xarray Dataset from atcc_dataset.to_xarray()
        fluorophore_xr : xr.Dataset, optional
            Real fluorophore emission spectra. If None, generates synthetic.
        """
        self.config = config
        self.atcc_xr = atcc_xr
        self.fluorophore_xr = fluorophore_xr
        self.rng = np.random.default_rng(config.seed)

        if config.bleaching_times is not None:
            self.bleaching_times = np.array(config.bleaching_times, dtype=float)
        else:
            n_times = int(config.bleaching_max_time / config.bleaching_interval) + 1
            self.bleaching_times = np.linspace(0, config.bleaching_max_time, n_times)

        print(f"Bleaching time points: {self.bleaching_times}")

        latest_time = (
            config.integration_times[-1] if config.integration_times else "10s"
        )
        self.raman_spectra = atcc_xr.sel(integration_time=latest_time)

        print(f"\nUsing integration time '{latest_time}' for Raman extraction")
        print(f"Available samples: {len(self.raman_spectra['sample'])}")

        self.wavenumbers = self.raman_spectra["wavenumber"].values

        # TODO Multi sample support
        if self.wavenumbers.ndim == 1:
            n_atcc_samples = len(self.raman_spectra["sample"])
            self.wavenumbers = np.tile(self.wavenumbers, (n_atcc_samples, 1))
            print(
                f"Wavenumber axis: shared (expanded to shape {self.wavenumbers.shape})"
            )
        else:
            self.wavenumbers = self.raman_spectra["wavenumber"].values
            print(f"Wavenumber axis: per-sample (shape {self.wavenumbers.shape})")

        # for now pass single master axis as similar enough
        ref_wavenumbers = (
            self.wavenumbers[0] if self.wavenumbers.ndim == 2 else self.wavenumbers
        )

        if config.use_shared_bases:

            # self.fluorophore_names = np.empty(self.config.n_fluorophores)
            self.shared_bases = self._generate_fluorophore_bases(ref_wavenumbers)
        # if fluorophore_xr is not None and "fluorophore_name" in fluorophore_xr:
        #     if self.config.use_shared_bases:
        self.fluorophore_names: list[list[str]] = []

        # Storage for polynomial normalization stats
        self.poly_norm_mean: Optional[float] = None
        self.poly_norm_std: Optional[float] = None
        self.log_poly_coeffs: Optional[np.ndarray] = None

        self.dataset: Optional[xr.Dataset] = None

    def _generate_fluorophore_bases(self, wavenumbers: np.ndarray) -> np.ndarray:
        """Generate fluorophore basis spectra."""
        n_f = self.config.n_fluorophores

        # Use real fluorophore spectra if provided
        if self.fluorophore_xr is not None:
            return self._sample_real_fluorophores(wavenumbers)

        # This code generates synthetic fluorophore spectra if no real data is provided
        n_wn = len(wavenumbers)
        wn = wavenumbers
        bases = np.zeros((n_f, n_wn))
        width_multipliers = [1.5, 1.0, 0.7]

        for i in range(n_f):
            n_gaussians = self.rng.integers(2, 5)
            for _ in range(n_gaussians):
                margin = 0.1 * (wn.max() - wn.min())
                center = self.rng.uniform(wn.min() + margin, wn.max() - margin)
                base_width = self.rng.uniform(50, 150)
                width = base_width * width_multipliers[i % len(width_multipliers)]
                amplitude = self.rng.uniform(0.5, 1.5)
                bases[i] += amplitude * np.exp(-0.5 * ((wn - center) / width) ** 2)
            bases[i] = np.maximum(bases[i], 1e-6)

        return l2_normalize(bases, axis=1)

    def _sample_real_fluorophores(self, target_wavenumbers: np.ndarray) -> np.ndarray:
        """Sample real fluorophore spectra from fluorophore_xr dataset."""
        print("Sampling real fluorophore spectra...")
        assert self.fluorophore_xr is not None

        n_f = self.config.n_fluorophores
        fluor_ds = self.fluorophore_xr
        n_available = len(fluor_ds["sample"])

        if n_f > n_available:
            print(
                f"Warning: Requested {n_f} fluorophores but only {n_available} available. Sampling with replacement."
            )
            indices = self.rng.choice(n_available, size=n_f, replace=True)
        else:
            indices = self.rng.choice(n_available, size=n_f, replace=False)
        if "fluorophore_name" in fluor_ds:
            self.fluorophore_names.append(
                fluor_ds.isel(sample=indices).fluorophore_name.values
            )

        bases = np.zeros((n_f, len(target_wavenumbers)))

        if "wavenumber" in fluor_ds.coords:
            source_wn = fluor_ds["wavenumber"].values
        elif "wavelength" in fluor_ds.coords:
            from ramanlib.bleaching.fluorophores import nm_to_wavenumber

            fluor_wavelength = fluor_ds["wavelength"].values
            source_wn = nm_to_wavenumber(
                fluor_wavelength, laser_nm=self.config.laser_nm
            )
        else:
            raise ValueError(
                "Fluorophore dataset must have 'wavenumber' or 'wavelength'"
            )
        bases = fluor_ds["intensity"].isel(sample=indices).values

        axes_match = (len(source_wn) == len(target_wavenumbers)) and np.allclose(
            source_wn, target_wavenumbers
        )
        if axes_match and not self.config.use_polynomial_fluorophores:
            # all good, no interpolation needed
            pass
        elif self.config.use_polynomial_fluorophores:
            print("Warning: source_wn and target_wavenumbers do not match.")
            print("Using polynomial fitting for fluorophore basis.")

            # fit on source axis
            log_poly_coeffs = self._fit_polynomial_fluorophores(bases, source_wn)
            self.log_poly_coeffs = log_poly_coeffs
            # log_intensity_values = vandermonde @ log_poly_coeffs
            # spectrum = np.exp(log_intensity_values)

            # Evaluate on target axis
            bases_processed = evaluate_polynomial_bases(
                log_poly_coeffs,
                target_wavenumbers,
                self.poly_norm_mean,
                self.poly_norm_std,
            )
        # else:
        #     if self.config.interpolation_method == "spline":
        #         print("Using spline interpolation for fluorophore basis.")
        #         spline = UnivariateSpline(source_wn, bases, k=3, s=0)
        #         bases_processed = spline(target_wavenumbers)
        #     else:
        #         print("Using linear interpolation for fluorophore basis.")
        #         # use linear interpolation if not spline
        #         bases_processed = np.interp(
        #             target_wavenumbers, source_wn, bases, left=0.0, right=0.0
        #             )

        # if self.config.fluorophore_variation > 0:
        #     intensity_scale = self.rng.normal(
        #         1.0, self.config.fluorophore_variation
        #     )
        #     intensity_scale = np.clip(intensity_scale, 0.5, 1.5)
        #     spectrum = spectrum * intensity_scale

        #     noise_scale = self.config.fluorophore_variation * 0.5
        #     noise = self.rng.normal(
        #         0, noise_scale * spectrum.mean(), len(wavenumbers)
        #     )
        #     spectrum = spectrum + noise

        #     baseline_offset = self.rng.uniform(0, 0.05 * spectrum.max())
        #     spectrum = spectrum + baseline_offset

        #     spectrum = np.maximum(spectrum, 0)
        #     bases[i] = spectrum

        return l2_normalize(bases_processed, axis=1)

    def _fit_polynomial_fluorophores(
        self, bases: np.ndarray, wavenumbers: np.ndarray
    ) -> np.ndarray:
        """Fit polynomials to fluorophore bases and store normalization stats."""
        # Fit on normalized wavenumbers (fit_polynomial_bases handles normalization)
        log_poly_coeffs, wn_mean, wn_std = fit_polynomial_bases(
            bases, wavenumbers, self.config.fluorophore_polynomial_degree
        )

        # Store normalization stats for later evaluation
        self.poly_norm_mean = wn_mean
        self.poly_norm_std = wn_std

        return log_poly_coeffs

    def _evaluate_polynomial_fluorophores(
        self, log_poly_coeffs: np.ndarray, wavenumbers: np.ndarray
    ) -> np.ndarray:
        """Evaluate polynomial fluorophores using stored normalization stats."""
        if self.poly_norm_mean is None or self.poly_norm_std is None:
            raise ValueError(
                "Normalization stats not set. Call _fit_polynomial_fluorophores first."
            )

        # Evaluate using the same normalization stats from fitting
        return evaluate_polynomial_bases(
            log_poly_coeffs, wavenumbers, self.poly_norm_mean, self.poly_norm_std
        )

    def _generate_decay_rates(self) -> np.ndarray:
        """Sample decay rates according to configured strategy."""
        n_f = self.config.n_fluorophores

        if self.config.decay_sampling == "uniform":
            return self.rng.uniform(
                self.config.decay_rate_min,
                self.config.decay_rate_max,
                size=n_f,
            )

        elif self.config.decay_sampling == "log_uniform":
            log_min = np.log(self.config.decay_rate_min)
            log_max = np.log(self.config.decay_rate_max)
            log_rates = self.rng.uniform(log_min, log_max, size=n_f)
            rates = np.exp(log_rates)
            print(f"Sampled log-uniform decay rates: {rates}")
            return rates

        elif self.config.decay_sampling == "multi_component":
            decay_rates = []
            components = (
                ["slow"] * ((n_f + 2) // 3)
                + ["medium"] * ((n_f + 1) // 3)
                + ["fast"] * (n_f // 3)
            )
            components = components[:n_f]

            for comp in components:
                if comp == "slow":
                    rate = self.rng.uniform(*self.config.decay_slow_range)
                elif comp == "medium":
                    rate = self.rng.uniform(*self.config.decay_medium_range)
                else:
                    rate = self.rng.uniform(*self.config.decay_fast_range)
                decay_rates.append(rate)

            decay_rates = np.array(decay_rates)
            self.rng.shuffle(decay_rates)
            return decay_rates

        else:
            raise ValueError(
                f"Unknown decay_sampling mode: {self.config.decay_sampling}"
            )

    def _generate_abundances(
        self, raman_spectrum: np.ndarray, bases: np.ndarray
    ) -> np.ndarray:
        """Generate abundances ensuring proper F/R ratio at t=0."""
        n_f = self.config.n_fluorophores
        fr_ratio = self.rng.uniform(self.config.fr_ratio_min, self.config.fr_ratio_max)
        raman_peak = raman_spectrum.max()
        target_fluor_total = fr_ratio * raman_peak

        raw_weights = self.rng.uniform(
            self.config.fluorophore_weight_min,
            self.config.fluorophore_weight_max,
            size=n_f,
        )

        basis_maxs = bases.max(axis=1)
        current_total = np.sum(raw_weights * basis_maxs)

        if current_total > 0:
            abundances = raw_weights * (target_fluor_total / current_total)
        else:
            abundances = raw_weights

        return abundances

    def _add_noise(
        self,
        raman: np.ndarray,
        fluorescence: np.ndarray,
    ) -> np.ndarray:
        """Add realistic noise to clean signal."""
        signal = raman + fluorescence

        if self.config.noise_type == "none":
            return signal

        elif self.config.noise_type == "gaussian":
            noise_std = signal.mean() * self.config.gaussian_noise_scale
            noise = self.rng.normal(0, noise_std, signal.shape)
            return signal + noise

        elif self.config.noise_type == "poisson_gaussian":
            signal_mean = signal.mean()
            if signal_mean > 0:
                scaled_signal = np.maximum(signal * self.config.poisson_noise_scale, 0)
                poisson_counts = self.rng.poisson(scaled_signal)
                poisson_signal = poisson_counts / self.config.poisson_noise_scale
            else:
                poisson_signal = signal

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
        """Reconstruct bleaching time series from parameters."""
        clean = reconstruct_time_series(
            raman, bases, abundances, decay_rates, self.bleaching_times
        )

        n_t = len(self.bleaching_times)
        fluorescence_time_series = clean - np.tile(raman, (n_t, 1))

        noisy = np.zeros_like(clean)
        for t in range(n_t):
            noisy[t] = self._add_noise(raman, fluorescence_time_series[t])

        return noisy, clean

    def generate(self) -> xr.Dataset:
        """Generate the full synthetic dataset."""
        n_samples = self.config.n_samples
        n_times = len(self.bleaching_times)
        n_f = self.config.n_fluorophores

        n_wn = len(
            self.wavenumbers[0]
            if self.wavenumbers.ndim == 2
            else self.wavenumbers  # if not shared wavenumber axis, use first sample's axis
        )
        intensity_noisy = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        intensity_clean = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        raman_gt = np.zeros((n_samples, n_wn), dtype=np.float32)
        wavenumbers_all = np.zeros((n_samples, n_wn), dtype=np.float32)

        n_atcc_samples = len(self.raman_spectra["sample"])
        decay_rates_gt = np.zeros((n_samples, n_f), dtype=np.float32)
        abundances_gt = np.zeros((n_samples, n_f), dtype=np.float32)

        species_list = []

        if self.config.use_shared_bases:
            bases_storage = self.shared_bases
        else:
            bases_storage = []

        print(f"\nGenerating {n_samples} synthetic samples...")

        for i in range(n_samples):
            atcc_idx = self.rng.integers(0, n_atcc_samples)
            raman = self.raman_spectra["intensity_raw"].isel(sample=atcc_idx).values

            if self.wavenumbers.ndim == 2:
                wn = self.wavenumbers[atcc_idx]
            else:
                wn = self.wavenumbers

            if "species" in self.raman_spectra:
                species = str(
                    self.raman_spectra["species"].isel(sample=atcc_idx).values
                )
            else:
                species = "Unknown"

            if self.config.use_shared_bases:
                bases = self.shared_bases
            else:
                bases = self._generate_fluorophore_bases(wn)
                bases_storage.append(bases)

            decay_rates = self._generate_decay_rates()
            abundances = self._generate_abundances(raman, bases)
            noisy, clean = self._reconstruct_time_series(
                raman, bases, abundances, decay_rates
            )

            intensity_noisy[i] = noisy
            intensity_clean[i] = clean
            raman_gt[i] = raman
            wavenumbers_all[i] = wn
            decay_rates_gt[i] = decay_rates
            abundances_gt[i] = abundances
            species_list.append(species)

            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{n_samples}")

        fluorophore_name = self.fluorophore_names
        intensity_noisy = np.array(intensity_noisy, dtype=np.float32)
        intensity_clean = np.array(intensity_clean, dtype=np.float32)
        raman_gt = np.array(raman_gt, dtype=np.float32)
        wavenumbers_all = np.array(wavenumbers_all, dtype=np.float32)

        ds = xr.Dataset(
            data_vars={
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
                "wavenumber": (
                    ["sample", "wavenumber"],
                    wavenumbers_all,
                    {"long_name": "Wavenumber axis (per-sample)", "units": "cm⁻¹"},
                ),
                "fluorophore_name": (
                    ["sample", "fluorophore"],
                    fluorophore_name,
                    {"long_name": "Ground Truth Fluorophore Name"},
                ),
                "species": (["sample"], species_list),
            },
            coords={
                "sample": np.arange(n_samples),
                "bleaching_time": self.bleaching_times,
            },
            attrs={
                "title": "Synthetic Photobleaching Dataset",
                "n_samples": n_samples,
                "n_fluorophores": n_f,
                "shared_bases": self.config.use_shared_bases,
                "noise_type": self.config.noise_type,
                "poisson_noise_scale": self.config.poisson_noise_scale,
                "gaussian_noise_scale": self.config.gaussian_noise_scale,
                "fr_ratio_range": f"{self.config.fr_ratio_min}-{self.config.fr_ratio_max}",
                "decay_rate_range": f"{self.config.decay_rate_min}-{self.config.decay_rate_max} s⁻¹",
                "seed": self.config.seed,
                "poly_norm_mean": (
                    self.poly_norm_mean if self.poly_norm_mean is not None else "None"
                ),
                "poly_norm_std": (
                    self.poly_norm_std if self.poly_norm_std is not None else "None"
                ),
            },
        )

        if self.config.use_shared_bases:
            ds["fluorophore_bases_gt"] = (
                ["fluorophore", "wavenumber"],
                bases_storage,
                {"long_name": "Shared fluorophore basis spectra"},
            )
            if self.log_poly_coeffs is not None:
                ds["log_poly_coeffs_gt"] = (
                    ["fluorophore", "poly_coeff"],
                    self.log_poly_coeffs,
                    {
                        "long_name": "Ground truth log polynomial coefficients",
                        "polynomial_degree": self.config.fluorophore_polynomial_degree,
                    },
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
