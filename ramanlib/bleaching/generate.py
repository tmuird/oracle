"""
Synthetic photobleaching dataset generation.

Generates training data with known ground truth for decomposition methods.
Fluorescence decay is always synthetic; Raman can be real ATCC spectra
(simulate_raman=False) or fully synthetic via ramanspy (simulate_raman=True).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from skimage.metrics import peak_signal_noise_ratio
from typing_extensions import Literal

from ramanlib.bleaching.physics import (
    interpolate_bases,
    l2_normalize,
    linf_normalize,
    reconstruct_time_series_numpy,
)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic photobleaching dataset generation."""

    n_samples: int = 5000
    physics_model: Literal["integrated", "factored", "pointsample"] = "pointsample"
    laser_nm: float = 532.0
    simulate_raman: bool = (
        False  # controls data source in train.py: False = real ATCC, True = synthetic via ramanspy
    )
    # Temporal parameters
    bleaching_times: Optional[List[float]] = None
    bleaching_interval: float = 0.1
    bleaching_max_time: float = 10.0

    # Integration time(s) to sample from ATCC data
    integration_times: List[str] = field(default_factory=lambda: ["1s"])

    # Fluorophore parameters
    n_fluorophores: int = 3
    # When set, use n_fluorophores as the bank size and randomly draw
    # n_active_per_sample of them per generated sample (abundances of the
    # remaining ones are zero). Requires use_shared_bases=True.
    n_active_per_sample: Optional[int] = None

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
    # poisson_noise_scale acts as gain (higher = less noise, not more!)
    # Physically: photon detection gain. Range: 0.1 (high noise) to 100 (low noise)
    poisson_noise_scale: float = 1.0
    # Gaussian read noise multiplier. noise_std = 5.0 * gaussian_noise_scale (counts RMS).
    # This is the ONLY noise in 'gaussian' mode, and the read-noise component in
    # 'poisson_gaussian' mode.  The noise std is constant across all frames and
    # wavenumbers (independent of signal level).
    # Example: 0.02 → 0.1 counts RMS (near noise-free); 1.0 → 5 counts RMS; 20.0 → 100 counts.
    gaussian_noise_scale: float = 0.02
    noise_type: str = "poisson_gaussian"

    # Basis generation
    use_shared_bases: bool = True
    # shared_axis: bool = True
    fluorophore_variation: float = 0.0
    interpolation_method: Literal["linear", "spline", "pchip"] = "pchip"
    smooth_sigma: float = (
        0.0  # Gaussian smoothing of interpolated bases (cm⁻¹); 0 = off
    )

    # Class-conditioned fluorophore assignment (requires n_active_per_sample and species labels)
    # Each class gets a Dirichlet-sampled probability row over the fluorophore bank.
    # Lower alpha → more peaked / specialised per class; higher → more uniform.
    use_class_conditioned_fluorophores: bool = False
    dirichlet_alpha: float = 0.15
    # Relative Gaussian noise on base τ per sample — simulates intra-class biological variance.
    # E.g. 0.15 → ±15% variation around each fluorophore's characteristic lifetime.
    tau_noise_std: float = 0.15

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
        if self.n_active_per_sample is not None:
            if self.n_active_per_sample < 1:
                raise ValueError("n_active_per_sample must be >= 1")
            if self.n_active_per_sample > self.n_fluorophores:
                raise ValueError("n_active_per_sample must be <= n_fluorophores")
            if not self.use_shared_bases:
                raise ValueError("n_active_per_sample requires use_shared_bases=True")
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

    Fluorescence decay is always synthetic. The Raman background is either
    real ATCC spectra (simulate_raman=False in config) or synthetic spectra
    generated via ramanspy (simulate_raman=True). Fluorophore bases can be
    shared across all samples or drawn per-sample.
    """

    def __init__(
        self,
        config: SyntheticConfig,
        raman_xr: Optional[xr.Dataset] = None,
        fluorophore_xr: Optional[xr.Dataset] = None,
    ):
        """
        Parameters
        ----------
        config : SyntheticConfig
            Dataset generation configuration
        raman_xr : xr.Dataset, optional
            Raman spectra dataset — real ATCC (simulate_raman=False) or
            synthetic from ramanspy (simulate_raman=True). Always required.
            Provides both the wavenumber axis and the per-sample raman signal.
        fluorophore_xr : xr.Dataset, optional
            Real fluorophore emission spectra. If None, generates synthetic.
        """
        if raman_xr is None:
            raise ValueError(
                "raman_xr must be provided — call load_data_sources() before constructing SyntheticBleachingDataset"
            )

        self.config = config
        self.raman_xr = raman_xr
        self.fluorophore_xr = fluorophore_xr
        self.rng = np.random.default_rng(config.seed)

        if config.bleaching_times is not None:
            self.bleaching_times = np.array(config.bleaching_times, dtype=float)
        else:
            n_times = int(config.bleaching_max_time / config.bleaching_interval) + 1
            self.bleaching_times = np.linspace(0, config.bleaching_max_time, n_times)

        print(f"Bleaching time points: {self.bleaching_times}")

        # Both simulate_raman=True (ramanspy) and simulate_raman=False (real ATCC)
        # use raman_xr as the source
        if "integration_time" in raman_xr.dims or "integration_time" in raman_xr.coords:
            requested = (
                config.integration_times[-1] if config.integration_times else "15s"
            )
            available = list(raman_xr.coords["integration_time"].values)
            latest_time = requested if requested in available else available[-1]
            if latest_time != requested:
                print(
                    f"Integration time '{requested}' not in dataset; using '{latest_time}'"
                )
            self.raman_spectra = raman_xr.sel(integration_time=latest_time)
            print(f"Integration time: '{latest_time}'")
        else:
            self.raman_spectra = raman_xr

        if "intensity_baseline_corrected" in self.raman_spectra:
            self.intensity_var = "intensity_baseline_corrected"
            print(f"Using baseline-corrected Raman spectra")
        else:
            self.intensity_var = "intensity_raw"
            print(f"Using raw Raman spectra (no baseline correction found)")

        print(f"Available samples: {len(self.raman_spectra['sample'])}")

        self.wavenumbers = self.raman_spectra["wavenumber"].values
        if self.wavenumbers.ndim == 1:
            n_raman_samples = len(self.raman_spectra["sample"])
            self.wavenumbers = np.tile(self.wavenumbers, (n_raman_samples, 1))
            print(
                f"Wavenumber axis: shared (expanded to shape {self.wavenumbers.shape})"
            )
        else:
            print(f"Wavenumber axis: per-sample (shape {self.wavenumbers.shape})")

        # for now pass single master axis as similar enough
        ref_wavenumbers = (
            self.wavenumbers[0] if self.wavenumbers.ndim == 2 else self.wavenumbers
        )
        # Pre-interpolate ALL fluorophore spectra onto the target wavenumber axis once.
        # For use_shared_bases=False this avoids N_samples × PCHIP calls in the loop;
        # for use_shared_bases=True it avoids the N_samples isel loop for names.
        self._all_fluor_bases: Optional[np.ndarray] = None
        self._all_fluor_names: Optional[np.ndarray] = None
        if fluorophore_xr is not None:
            self._all_fluor_bases, self._all_fluor_names = self._precompute_fluor_bases(
                ref_wavenumbers
            )

        self.fluorophore_names: list[list[str]] = []
        if config.use_shared_bases:
            # self.fluorophore_names = np.empty(self.config.n_fluorophores)
            self.shared_bases = self._generate_fluorophore_bases(ref_wavenumbers)
        # if fluorophore_xr is not None and "fluorophore_name" in fluorophore_xr:
        #     if self.config.use_shared_bases:

        # ── Class-conditioned fluorophore assignment ──────────────────────────
        self.class_probs: Optional[np.ndarray] = None
        self.class_to_idx: Optional[dict] = None
        self.base_rates: Optional[np.ndarray] = None

        if config.use_class_conditioned_fluorophores:
            if config.n_active_per_sample is None:
                print(
                    "Warning: use_class_conditioned_fluorophores=True but "
                    "n_active_per_sample is None — class conditioning has no "
                    "effect without a bank (all fluorophores are always active)."
                )
            if "species" in self.raman_spectra:
                all_species = self.raman_spectra["species"].values.astype(str)
                unique_classes = np.unique(all_species)
                self.class_to_idx = {c: int(i) for i, c in enumerate(unique_classes)}
                n_classes = len(unique_classes)
                n_f = config.n_fluorophores
                # [n_classes, n_fluorophores] — each row is a probability vector
                self.class_probs = self.rng.dirichlet(
                    alpha=[config.dirichlet_alpha] * n_f, size=n_classes
                )
                # Characteristic decay rate for each fluorophore in the bank.
                # Sampled once per run from the configured multi-component distribution
                # so the bank spans the full slow/medium/fast range.
                self.base_rates = self._generate_decay_rates(n=n_f)
                print(
                    f"Class-conditioned fluorophores: {n_classes} classes × "
                    f"{n_f} fluorophores | α={config.dirichlet_alpha} | "
                    f"τ-noise={config.tau_noise_std}"
                )
                print(f"  Base rates (s⁻¹): {np.round(self.base_rates, 3)}")
            else:
                print(
                    "Warning: use_class_conditioned_fluorophores=True but "
                    "raman_xr has no 'species' coordinate — conditioning disabled."
                )

        self.dataset: Optional[xr.Dataset] = None

    def _precompute_fluor_bases(
        self, target_wavenumbers: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Interpolate every fluorophore spectrum to target_wavenumbers once.

        Returns
        -------
        all_bases : np.ndarray [N_fluor, W_target]
            L-inf normalised, interpolated spectra for every fluorophore.
        all_names : np.ndarray or None
            Fluorophore name strings aligned with the sample axis, or None.
        """
        assert self.fluorophore_xr is not None
        fluor_ds = self.fluorophore_xr

        if "wavenumber" in fluor_ds.coords:
            source_wn = fluor_ds["wavenumber"].values
            if source_wn.ndim > 1:
                source_wn = source_wn[0]
        elif "wavelength" in fluor_ds.coords:
            from ramanlib.bleaching.fluorophores import nm_to_wavenumber

            source_wn = nm_to_wavenumber(
                fluor_ds["wavelength"].values, laser_nm=self.config.laser_nm
            )
        else:
            raise ValueError(
                "Fluorophore dataset must have 'wavenumber' or 'wavelength'"
            )

        all_intensities = fluor_ds["intensity"].values  # [N_fluor, W_src]
        all_bases = interpolate_bases(
            all_intensities,
            source_wn,
            target_wavenumbers,
            method=self.config.interpolation_method,
            smooth_sigma=self.config.smooth_sigma,
        )
        all_bases = linf_normalize(all_bases, axis=1)  # [N_fluor, W_target]

        all_names = None
        if "fluorophore_name" in fluor_ds:
            all_names = fluor_ds["fluorophore_name"].values  # [N_fluor]

        return all_bases, all_names

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

        return linf_normalize(bases, axis=1)

    def _sample_real_fluorophores(self, target_wavenumbers: np.ndarray) -> np.ndarray:
        """Sample real fluorophore spectra from the precomputed basis array.

        Uses self._all_fluor_bases (precomputed at init) — just index, no PCHIP.
        """
        assert self._all_fluor_bases is not None

        n_f = self.config.n_fluorophores
        n_available = len(self._all_fluor_bases)

        if n_f > n_available:
            print(
                f"Warning: Requested {n_f} fluorophores but only {n_available} available. Sampling with replacement."
            )
            indices = self.rng.choice(n_available, size=n_f, replace=True)
        else:
            indices = self.rng.choice(n_available, size=n_f, replace=False)

        if self._all_fluor_names is not None:
            names = self._all_fluor_names[indices]
            if self.config.use_shared_bases:
                # Replicate the same name row for every sample (names are fixed)
                for _ in range(self.config.n_samples):
                    self.fluorophore_names.append(names)
            else:
                self.fluorophore_names.append(names)

        # Already L-inf normalised and interpolated — just index
        return self._all_fluor_bases[indices]

    def _generate_decay_rates(self, n: Optional[int] = None) -> np.ndarray:
        """Sample decay rates according to configured strategy."""
        n_f = n if n is not None else self.config.n_fluorophores

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
        self, raman_spectrum: np.ndarray, bases: np.ndarray, n: Optional[int] = None
    ) -> np.ndarray:
        """Generate abundances ensuring proper F/R ratio at t=0."""
        n_f = n if n is not None else self.config.n_fluorophores
        fr_ratio = self.rng.uniform(self.config.fr_ratio_min, self.config.fr_ratio_max)
        raman_peak = raman_spectrum.max()
        # When raman is disabled (simulate_raman=False), peak is 0; use unit scale instead
        target_fluor_total = fr_ratio * raman_peak if raman_peak > 0 else fr_ratio

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

        elif self.config.noise_type == "poisson":
            # Poisson (shot) noise: variance = mean
            # Scale controls SNR: higher scale = more detected photons = less relative noise
            # Physically: scale represents detector gain or integration time
            scaled = np.maximum(signal * self.config.poisson_noise_scale, 0)

            noisy_counts = self.rng.poisson(scaled)
            return noisy_counts / self.config.poisson_noise_scale

        elif self.config.noise_type == "gaussian":
            # Fixed-variance additive read noise.
            # gaussian_noise_scale is a multiplier on the 5-count RMS baseline,
            # matching the read noise in 'poisson_gaussian' mode.
            # Noise std is constant across all frames (unlike signal.std() which
            # would incorrectly decrease as fluorescence bleaches).
            noise_std = self.config.gaussian_noise_scale
            noise = self.rng.normal(0, noise_std, signal.shape)
            # return np.maximum(signal + noise, 0)  # no negative counts

            return signal + noise  # allow for negatives

        elif self.config.noise_type == "poisson_gaussian":
            #  detector noise model: shot noise + read noise

            # 1. Shot noise (Poisson) - scales with signal
            # Higher poisson_noise_scale = more photons detected = lower relative noise
            scaled = np.maximum(signal * self.config.poisson_noise_scale, 0)
            shot_noisy = self.rng.poisson(scaled) / self.config.poisson_noise_scale

            # 2. Read noise (Gaussian) - constant, detector property
            read_noise_std = self.config.gaussian_noise_scale
            read_noise = self.rng.normal(0, read_noise_std, signal.shape)

            # return np.maximum(shot_noisy + read_noise, 0)  # No negative counts
            return shot_noisy + read_noise  # allow for negatives

        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")

    def calculate_noise_metrics(self, clean: np.ndarray, noisy: np.ndarray) -> dict:
        """
        Calculate all noise metrics between clean and noisy data.

        Uses skimage for PSNR, numpy for SNR.

        Returns:
            dict with psnr_db, snr_db, noise_std, and noise components
        """
        noise = noisy - clean

        # PSNR (standard implementation from skimage)
        data_range = clean.max() - clean.min()
        psnr_db = peak_signal_noise_ratio(clean, noisy, data_range=data_range)

        # SNR (signal power / noise power)
        signal_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        snr_db = (
            10 * np.log10(signal_power / noise_power)
            if noise_power > 0
            else float("inf")
        )

        return {
            "psnr_db": psnr_db,
            "snr_db": snr_db,
            "noise_std": np.std(noise),
            "noise_rms": np.sqrt(noise_power),
            "signal_mean": np.mean(clean),
        }

    def _reconstruct_time_series(
        self,
        raman: np.ndarray,
        bases: np.ndarray,
        abundances: np.ndarray,
        decay_rates: np.ndarray,
        physics_model: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # print(f"Reconstructing according to physics model: {physics_model}")
        clean = reconstruct_time_series_numpy(
            raman,
            bases,
            abundances,
            decay_rates,
            self.bleaching_times,
            frame_duration=self.config.bleaching_interval,
            physics_model=physics_model,
        )

        n_t = len(self.bleaching_times)
        # Raman contribution per frame is raman * frame_duration
        raman_per_frame = raman * self.config.bleaching_interval
        fluorescence_time_series = clean - raman_per_frame[None, :]  # [n_t, W]

        # _add_noise uses numpy ops throughout, so broadcasting over the full
        # [n_t, W] array is equivalent to the previous per-frame loop but
        # issues a single rng.poisson() call instead of n_t separate ones.
        noisy = self._add_noise(raman_per_frame, fluorescence_time_series)  # [n_t, W]

        return noisy, clean

    # ------------------------------------------------------------------
    # Vectorised helpers (used by _generate_vectorized)
    # ------------------------------------------------------------------

    def _sample_decay_rates_batch(self, n: int) -> np.ndarray:
        """Sample decay rates for *n* samples at once. Returns [n, F] float32."""
        n_f = self.config.n_fluorophores
        if self.config.decay_sampling == "uniform":
            return self.rng.uniform(
                self.config.decay_rate_min, self.config.decay_rate_max, (n, n_f)
            ).astype(np.float32)
        elif self.config.decay_sampling == "log_uniform":
            log_min = np.log(self.config.decay_rate_min)
            log_max = np.log(self.config.decay_rate_max)
            return np.exp(self.rng.uniform(log_min, log_max, (n, n_f))).astype(
                np.float32
            )
        else:  # multi_component
            n_slow = (n_f + 2) // 3
            n_medium = (n_f + 1) // 3
            n_fast = n_f // 3
            cols = []
            for _ in range(n_slow):
                cols.append(self.rng.uniform(*self.config.decay_slow_range, n))
            for _ in range(n_medium):
                cols.append(self.rng.uniform(*self.config.decay_medium_range, n))
            for _ in range(n_fast):
                cols.append(self.rng.uniform(*self.config.decay_fast_range, n))
            rates = np.stack(cols, axis=1).astype(np.float32)  # [n, F]
            # Per-row shuffle via argsort of random values
            perm = self.rng.random((n, n_f)).argsort(axis=1)
            return np.take_along_axis(rates, perm, axis=1)

    def _sample_abundances_batch(
        self, raman_all: np.ndarray, bases: np.ndarray, n: int
    ) -> np.ndarray:
        """Sample abundances for *n* samples at once. Returns [n, F] float32."""
        n_f = self.config.n_fluorophores
        fr_ratios = self.rng.uniform(
            self.config.fr_ratio_min, self.config.fr_ratio_max, n
        )
        raw_weights = self.rng.uniform(
            self.config.fluorophore_weight_min,
            self.config.fluorophore_weight_max,
            (n, n_f),
        )
        raman_peaks = raman_all.max(axis=1)  # [n]
        target = np.where(raman_peaks > 0, fr_ratios * raman_peaks, fr_ratios)
        basis_maxs = bases.max(axis=1)  # [F]
        current = (raw_weights * basis_maxs[np.newaxis, :]).sum(axis=1)  # [n]
        safe = np.where(current > 0, current, 1.0)
        return (raw_weights * (target / safe)[:, np.newaxis]).astype(np.float32)

    def _add_noise_torch(self, clean, device):
        import torch

        pscale = self.config.poisson_noise_scale
        gstd = self.config.gaussian_noise_scale

        # 'clean' is the full reconstruction (Raman + Fluorescence).
        # Clone to leave the stored clean array untouched.
        noisy = clean.clone()

        if self.config.noise_type == "poisson_gaussian":
            # In-place multiply and clamp (saves ~7.5 GiB of VRAM overhead)
            noisy.mul_(pscale).clamp_(min=0.0)

            # Poisson allocates one new tensor, but immediately frees the old 'noisy'
            noisy = torch.poisson(noisy)

            # In-place divide
            noisy.div_(pscale)

            # Generate noise, scale it in-place, and add it in-place
            noisy.add_(torch.randn_like(noisy).mul_(gstd))
            return noisy

        elif self.config.noise_type == "poisson":
            noisy.mul_(pscale).clamp_(min=0.0)
            noisy = torch.poisson(noisy)
            noisy.div_(pscale)
            return noisy

        elif self.config.noise_type == "gaussian":
            noisy.add_(torch.randn_like(noisy).mul_(gstd))
            return noisy

        return noisy

    def _add_noise_batch(
        self, raman_per_frame: np.ndarray, fluorescence: np.ndarray
    ) -> np.ndarray:
        """Vectorised noise for [B, T, W] arrays. Returns float32."""
        signal = raman_per_frame + fluorescence  # [B, T, W]
        if self.config.noise_type == "none":
            return signal.astype(np.float32)
        pscale = self.config.poisson_noise_scale
        gstd = self.config.gaussian_noise_scale
        if self.config.noise_type == "poisson":
            scaled = np.maximum(signal * pscale, 0)
            return (self.rng.poisson(scaled) / pscale).astype(np.float32)
        elif self.config.noise_type == "gaussian":
            return (signal + self.rng.normal(0.0, gstd, signal.shape)).astype(
                np.float32
            )
        elif self.config.noise_type == "poisson_gaussian":
            scaled = np.maximum(signal * pscale, 0)
            shot = self.rng.poisson(scaled) / pscale
            read = self.rng.normal(0.0, gstd, signal.shape)
            return (shot + read).astype(np.float32)
        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type!r}")

    def _sample_per_sample_bases(self, n_samples: int) -> np.ndarray:
        """Sample n_samples × n_fluorophores bases from the precomputed pool.

        Returns [N, F, W] float32. Fills self.fluorophore_names as a side-effect.
        """
        assert self._all_fluor_bases is not None
        n_f = self.config.n_fluorophores
        n_available = len(self._all_fluor_bases)

        if n_f <= n_available:
            # Vectorised sampling without replacement: generate [N, n_available] random
            # values, argsort each row, take first n_f columns. No Python loop.
            scores = self.rng.random((n_samples, n_available))
            all_indices = np.argpartition(scores, n_f, axis=1)[:, :n_f]  # [N, F]
        else:
            all_indices = self.rng.integers(0, n_available, size=(n_samples, n_f))

        bases_all = self._all_fluor_bases[all_indices]  # [N, F, W]

        if self._all_fluor_names is not None:
            self.fluorophore_names = list(self._all_fluor_names[all_indices].tolist())

        return bases_all.astype(np.float32)

    def _generate_vectorized(
        self,
        _all_raman: np.ndarray,
        raman_indices: np.ndarray,
        _all_species: Optional[np.ndarray],
    ) -> xr.Dataset:
        """Vectorised generation for all-active, non-class-conditioned configs.

        Works for both shared and per-sample bases. Replaces the per-sample Python
        loop with batched torch physics calls, which is 10–100× faster.
        """
        import torch
        from ramanlib.bleaching.physics import (
            reconstruct_time_series_factored_torch,
            reconstruct_time_series_integrated_torch,
            reconstruct_time_series_torch,
        )

        n_samples = self.config.n_samples
        n_times = len(self.bleaching_times)
        n_f = self.config.n_fluorophores
        n_wn = _all_raman.shape[1]
        frame_dur = self.config.bleaching_interval
        physics_model = self.config.physics_model
        use_shared = self.config.use_shared_bases

        # ── Raman spectra & species ───────────────────────────────────────────
        raman_all = _all_raman[raman_indices]  # [N, W]
        if self.wavenumbers.ndim == 2:
            wavenumbers_all = self.wavenumbers[raman_indices]  # [N, W]
        else:
            wavenumbers_all = np.tile(self.wavenumbers, (n_samples, 1))

        if _all_species is not None:
            species_list = [str(_all_species[i]) for i in raman_indices]
        else:
            species_list = ["Unknown"] * n_samples

        # ── Bases ─────────────────────────────────────────────────────────────
        if use_shared:
            # shared_bases: [F, W] — reused for every sample in torch via broadcast
            bases_np = self.shared_bases  # [F, W]
            if not self.fluorophore_names:
                self.fluorophore_names = [[""] * n_f for _ in range(n_samples)]
        else:
            # per-sample: [N, F, W]
            bases_np = self._sample_per_sample_bases(n_samples)

        # ── Random parameters (all samples at once) ───────────────────────────
        decay_rates_gt = self._sample_decay_rates_batch(n_samples)  # [N, F]
        # _sample_abundances_batch needs [F] max-per-basis for shared, [N,F] for per-sample
        if use_shared:
            abundances_gt = self._sample_abundances_batch(
                raman_all, bases_np, n_samples
            )  # [N, F]
        else:
            # basis_maxs per sample: [N, F]
            basis_maxs_per_sample = bases_np.max(axis=2)  # [N, F]
            fr_ratios = self.rng.uniform(
                self.config.fr_ratio_min, self.config.fr_ratio_max, n_samples
            )
            raw_weights = self.rng.uniform(
                self.config.fluorophore_weight_min,
                self.config.fluorophore_weight_max,
                (n_samples, n_f),
            )
            raman_peaks = raman_all.max(axis=1)  # [N]
            target = np.where(raman_peaks > 0, fr_ratios * raman_peaks, fr_ratios)
            current = (raw_weights * basis_maxs_per_sample).sum(axis=1)  # [N]
            safe = np.where(current > 0, current, 1.0)
            abundances_gt = (raw_weights * (target / safe)[:, np.newaxis]).astype(
                np.float32
            )

        # ── Physics reconstruction in GPU-friendly batches ────────────────────
        # Pick the GPU with the most free VRAM. GPU0 is often occupied by training
        # runs; checking all GPUs avoids stalling on a nearly-full device.
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            best_gpu = max(range(n_gpus), key=lambda g: torch.cuda.mem_get_info(g)[0])
            free_bytes, _ = torch.cuda.mem_get_info(best_gpu)
            bytes_per_sample = (
                n_wn * n_times * 4 * 4
            )  # 4 tensors: clean, noisy, raman, bases slice
            gpu_batch = max(1, int(free_bytes * 0.7 / bytes_per_sample))
            if gpu_batch >= 16:
                device = torch.device(f"cuda:{best_gpu}")
                batch_size = min(n_samples, gpu_batch, 20_000)
            else:
                # Not enough free VRAM on any GPU — fall back to CPU
                device = torch.device("cpu")
                batch_size = min(n_samples, 2_000)
        else:
            device = torch.device("cpu")
            batch_size = min(n_samples, 2_000)

        if use_shared:
            bases_device = torch.from_numpy(bases_np).float().to(device)  # [F, W]
        time_t = torch.from_numpy(self.bleaching_times.astype(np.float32)).to(device)
        print(f"  device={device}, batch_size={batch_size}")

        intensity_clean = np.empty((n_samples, n_times, n_wn), dtype=np.float32)
        intensity_noisy = np.empty((n_samples, n_times, n_wn), dtype=np.float32)

        print(
            f"\nGenerating {n_samples} synthetic samples "
            f"(vectorised, device={device}, bases={'shared' if use_shared else 'per-sample'})..."
        )
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            raman_t = torch.from_numpy(raman_all[start:end]).float().to(device)
            abund_t = torch.from_numpy(abundances_gt[start:end]).float().to(device)
            rates_t = torch.from_numpy(decay_rates_gt[start:end]).float().to(device)

            if use_shared:
                b_t = bases_device  # [F, W]
            else:
                b_t = (
                    torch.from_numpy(bases_np[start:end]).float().to(device)
                )  # [B, F, W]

            if physics_model == "factored":
                # physical_to_effective_amplitude uses np.exp — use torch ops instead
                eff_amp = (
                    abund_t * (1.0 - torch.exp(-rates_t * frame_dur)) / (rates_t + 1e-8)
                )
                clean_bwt = reconstruct_time_series_factored_torch(
                    raman=raman_t,
                    bases=b_t,
                    effective_amplitudes=eff_amp,
                    decay_rates=rates_t,
                    time_values=time_t,
                    frame_duration=frame_dur,
                )
            elif physics_model == "integrated":
                clean_bwt = reconstruct_time_series_integrated_torch(
                    raman=raman_t,
                    bases=b_t,
                    abundances=abund_t,
                    decay_rates=rates_t,
                    time_values=time_t,
                    frame_duration=frame_dur,
                )
            else:  # pointsample
                clean_bwt = reconstruct_time_series_torch(
                    raman=raman_t,
                    bases=b_t,
                    abundances=abund_t,
                    decay_rates=rates_t,
                    time_values=time_t,
                    frame_duration=frame_dur,
                )
            # clean_bwt: [B, W, T] → [B, T, W] (stay on device for noise)
            clean_btw_t = clean_bwt.permute(0, 2, 1)  # [B, T, W] on device

            # Noise injection on GPU — avoids transferring [B, T, W] to CPU for rng.poisson
            noisy_btw_t = self._add_noise_torch(clean_btw_t, device)

            intensity_clean[start:end] = clean_btw_t.cpu().numpy()
            intensity_noisy[start:end] = noisy_btw_t.cpu().numpy()

            print(f"  Generated {end}/{n_samples}")

        raman_gt = raman_all.astype(np.float32)

        return self._build_dataset(
            intensity_noisy=intensity_noisy,
            intensity_clean=intensity_clean,
            raman_gt=raman_gt,
            wavenumbers_all=wavenumbers_all,
            decay_rates_gt=decay_rates_gt,
            abundances_gt=abundances_gt,
            species_list=species_list,
            bases_storage=bases_np,
            shared_bases=use_shared,
        )

    def _build_dataset(
        self,
        intensity_noisy: np.ndarray,
        intensity_clean: np.ndarray,
        raman_gt: np.ndarray,
        wavenumbers_all: np.ndarray,
        decay_rates_gt: np.ndarray,
        abundances_gt: np.ndarray,
        species_list: list,
        bases_storage,
        shared_bases: bool,
    ) -> xr.Dataset:
        """Package arrays into an xr.Dataset (shared by both generation paths)."""
        n_samples = self.config.n_samples
        n_f = self.config.n_fluorophores

        fluorophore_name = self.fluorophore_names

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
            },
        )

        bases_storage_array = np.array(bases_storage, dtype=np.float32)
        if shared_bases:
            ds["fluorophore_bases_gt"] = (
                ["fluorophore", "wavenumber"],
                bases_storage_array,
                {"long_name": "Shared fluorophore basis spectra"},
            )
        else:
            ds["fluorophore_bases_gt"] = (
                ["sample", "fluorophore", "wavenumber"],
                bases_storage_array,
                {"long_name": "Per-sample fluorophore basis spectra"},
            )

        return ds

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

        n_raman_available = len(self.raman_spectra["sample"])

        if n_samples > n_raman_available:
            raman_indices = self.rng.choice(n_raman_available, n_samples, replace=True)
        elif n_samples < n_raman_available:
            raman_indices = self.rng.choice(n_raman_available, n_samples, replace=False)
        else:
            raman_indices = self.rng.permutation(n_raman_available)

        # Pre-extract arrays from xarray once to avoid per-sample isel overhead.
        _all_raman = self.raman_spectra[
            self.intensity_var
        ].values  # [N_avail, W] or [N_avail, 1, W]
        if _all_raman.ndim == 3:
            _all_raman = _all_raman[:, 0, :]  # drop integration_time dim

        _all_species: Optional[np.ndarray] = None
        if "species" in self.raman_spectra:
            _all_species = self.raman_spectra["species"].values.astype(str)

        # Fast vectorised path: all fluorophores active, no class conditioning.
        # Covers both shared and per-sample bases — falls back to sequential loop
        # only for bank-with-subset-active or class-conditioned configs.
        _can_vectorize = (
            self.config.n_active_per_sample is None
            and not self.config.use_class_conditioned_fluorophores
            and (self.config.use_shared_bases or self._all_fluor_bases is not None)
        )
        if _can_vectorize:
            ds = self._generate_vectorized(_all_raman, raman_indices, _all_species)
            self.dataset = ds
            n_wn_actual = ds["wavenumber"].shape[-1]
            print("\nGenerated dataset:")
            print(f"  Samples: {n_samples}")
            print(f"  Bleaching time points: {n_times}")
            print(
                f"  Wavenumber axis: per-sample (shape: ({n_samples}, {n_wn_actual}))"
            )
            print(f"  Fluorophores: {n_f}")
            print(
                f"  Decay rate range: [{self.config.decay_rate_min}, {self.config.decay_rate_max}] s⁻¹"
            )
            print(
                f"  F/R ratio range: [{self.config.fr_ratio_min}, {self.config.fr_ratio_max}]"
            )
            return ds

        # ── Sequential fallback (per-sample bases / active subsets / class conditioning) ──
        intensity_noisy = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        intensity_clean = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        raman_gt = np.zeros((n_samples, n_wn), dtype=np.float32)
        wavenumbers_all = np.zeros((n_samples, n_wn), dtype=np.float32)
        decay_rates_gt = np.zeros((n_samples, n_f), dtype=np.float32)
        abundances_gt = np.zeros((n_samples, n_f), dtype=np.float32)

        species_list = []

        if self.config.use_shared_bases:
            bases_storage_temp = self.shared_bases
        else:
            bases_storage_temp: List[np.ndarray] = []

        print(f"\nGenerating {n_samples} synthetic samples...")
        for i in range(n_samples):
            raman_idx = int(raman_indices[i])
            raman = _all_raman[raman_idx]

            if self.wavenumbers.ndim == 2:
                wn = self.wavenumbers[raman_idx]
            else:
                wn = self.wavenumbers

            if _all_species is not None:
                species = str(_all_species[raman_idx])
            else:
                species = "Unknown"

            if self.config.use_shared_bases:
                bases = self.shared_bases
            else:
                bases = self._generate_fluorophore_bases(wn)
                bases_storage_temp.append(bases)

            n_active = self.config.n_active_per_sample
            physics_model = self.config.physics_model
            if n_active is not None:
                # Bank-of-N, draw-K mode.
                # Only active components contribute; inactive ones have zero abundance
                # in the GT but their basis spectra remain in the shared bank.
                if (
                    self.class_probs is not None
                    and self.class_to_idx is not None
                    and species in self.class_to_idx
                ):
                    # Class-conditioned: weighted draw + per-fluorophore τ noise
                    class_idx = self.class_to_idx[species]
                    probs = self.class_probs[class_idx]  # [n_f] probability vector
                    active_idx = self.rng.choice(n_f, n_active, replace=False, p=probs)
                    active_bases = bases[active_idx]
                    # Characteristic rates + intra-class noise (relative Gaussian)
                    noise = np.clip(
                        1.0 + self.rng.normal(0, self.config.tau_noise_std, n_active),
                        0.5,
                        2.0,
                    )
                    active_rates = (self.base_rates[active_idx] * noise).astype(
                        np.float32
                    )
                else:
                    # Uniform random selection (original behaviour / fallback)
                    active_idx = self.rng.choice(n_f, n_active, replace=False)
                    active_bases = bases[active_idx]
                    active_rates = self._generate_decay_rates(n=n_active)

                active_abund = self._generate_abundances(
                    raman, active_bases, n=n_active
                )

                decay_rates = np.zeros(n_f, dtype=np.float32)
                abundances = np.zeros(n_f, dtype=np.float32)
                decay_rates[active_idx] = active_rates
                abundances[active_idx] = active_abund

                noisy, clean = self._reconstruct_time_series(
                    raman, active_bases, active_abund, active_rates, physics_model
                )
            else:
                decay_rates = self._generate_decay_rates()
                abundances = self._generate_abundances(raman, bases)
                noisy, clean = self._reconstruct_time_series(
                    raman, bases, abundances, decay_rates, physics_model
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

        ds = self._build_dataset(
            intensity_noisy=np.asarray(intensity_noisy, dtype=np.float32),
            intensity_clean=np.asarray(intensity_clean, dtype=np.float32),
            raman_gt=np.asarray(raman_gt, dtype=np.float32),
            wavenumbers_all=np.asarray(wavenumbers_all, dtype=np.float32),
            decay_rates_gt=decay_rates_gt,
            abundances_gt=abundances_gt,
            species_list=species_list,
            bases_storage=bases_storage_temp,
            shared_bases=self.config.use_shared_bases,
        )

        self.dataset = ds

        print("\nGenerated dataset:")
        print(f"  Samples: {n_samples}")
        print(f"  Bleaching time points: {n_times}")
        print(f"  Wavenumber axis: per-sample (shape: ({n_samples}, {n_wn}))")
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
