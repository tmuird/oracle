# synthetic_bleaching.py

import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import ramanspy as rp
from scipy.interpolate import interp1d


@dataclass
class SyntheticConfig:
    """Configuration for synthetic photobleaching dataset generation."""

    # Dataset size
    n_samples: int = 5000

    # Temporal parameters - BLEACHING TIME SERIES (cumulative laser exposure)
    # These are the time points where fluorescence has decayed
    bleaching_times: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 1.0, 5.0, 10.0  # seconds of cumulative exposure
    ])

    # Integration time for acquisition (affects SNR, not bleaching)
    # Can be single value or per-timepoint
    integration_time: Union[float, List[float]] = 1.0  # seconds

    # Fluorophore parameters
    n_fluorophores: int = 3
    
    # Decay rate distributions (log10 scale for λ in s⁻¹)
    # Based on literature + Mohammadrahim's findings
    decay_rate_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 2.5),   # Fast: λ = 10-316 s⁻¹ (τ = 3-100ms)
        (0.3, 0.8),   # Medium: λ = 2-6 s⁻¹ (τ = 0.15-0.5s)
        (-0.7, 0.0),  # Slow: λ = 0.2-1 s⁻¹ (τ = 1-5s)
    ])
    
    # Fluorescence-to-Raman ratio at t=0
    fr_ratio_range: Tuple[float, float] = (3.0, 15.0)
    
    # Noise parameters
    snr_range: Tuple[float, float] = (15.0, 50.0)
    noise_type: str = 'poisson_gaussian'
    
    # Fluorophore basis generation
    basis_type: str = 'gaussian_mixture'  # 'polynomial', 'gaussian_mixture'
    shared_bases: bool = True  # If True, same bases for all samples
    
    # Source weighting
    atcc_weight: float = 0.7  # Probability of sampling from ATCC vs RamanSPy
    
    # Random seed
    seed: Optional[int] = 42


class RamanLibrary:
    """
    Library of real Raman spectra from multiple sources.
    
    Extracts pure Raman from:
    1. ATCC late frames (after photobleaching)
    2. RamanSPy bacteria dataset (preprocessed)
    """
    
    def __init__(
        self,
        target_wavenumbers: np.ndarray,
        atcc_weight: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        target_wavenumbers : np.ndarray
            Target wavenumber axis for all spectra
        atcc_weight : float
            Probability of sampling from ATCC vs other sources
        seed : int, optional
            Random seed for reproducibility
        """
        self.target_wavenumbers = target_wavenumbers
        self.atcc_weight = atcc_weight
        self.rng = np.random.default_rng(seed)
        
        # Storage for spectra from different sources
        self.atcc_spectra = []
        self.atcc_species = []
        self.ramanspy_spectra = []
        self.ramanspy_labels = []
        
        # Source wavenumber axes (for interpolation)
        self.atcc_wavenumbers = None
        self.ramanspy_wavenumbers = None
    
    def add_from_atcc_xarray(
        self,
        ds: xr.Dataset,
        late_time: List[str] = ['15s'],
        intensity_var: str = 'intensity_raw',
        baseline_correct: bool = True,
    ):
        """
        Extract Raman from ATCC dataset late frames.
        
        Parameters
        ----------
        ds : xr.Dataset
            Output from StrainDataset.to_xarray()
        late_time_index : int
            Which integration time to use (-1 = last/longest)
        intensity_var : str
            Which intensity variable to use
        baseline_correct : bool
            Apply baseline correction to extract cleaner Raman
        """
        wavenumbers = ds['wavenumber'].values
        self.atcc_wavenumbers = wavenumbers
        
        
        intensity = ds[intensity_var].values  # (sample, integration_time, wavenumber)
        print(f"Loaded intensities with shape {intensity.shape}")
        # Use late frames
        print(f"Extracting {late_time} integration time for Raman spectra...")
        late_spectra =ds.sel(integration_time=late_time)[intensity_var].values
        
        # Get species labels if available
        if 'species' in ds:
            species = ds['species'].values
        else:
            species = ['Unknown'] * len(late_spectra)
        
        for i, spectrum in enumerate(late_spectra):
            if np.isnan(spectrum).all():
                continue
            
            processed = spectrum.copy()
            
            # Optional baseline correction
            if baseline_correct:
                container = rp.SpectralContainer(
                    processed[np.newaxis, :], wavenumbers
                )
                corrected = rp.preprocessing.baseline.IARPLS().apply(container)
                processed = corrected.spectral_data[0]
            
            # Ensure floor ≈ 0 (key physical constraint)
            floor = np.percentile(processed, 5)
            processed = np.maximum(processed - floor, 0)
            
            # Interpolate to target wavenumbers
            processed = self._interpolate_to_target(processed, wavenumbers)
            
            self.atcc_spectra.append(processed)
            self.atcc_species.append(species[i])
        
        print(f"Added {len(self.atcc_spectra)} Raman spectra from ATCC")
    
    def add_from_ramanspy(
        self,
        data_folder: str,
        datasets: List[str] = ['train', 'val', 'test'],
    ):
        """
        Load preprocessed Raman from RamanSPy bacteria dataset.
        
        Parameters
        ----------
        data_folder : str
            Path to downloaded RamanSPy bacteria data
        datasets : List[str]
            Which dataset splits to load
        """
        all_spectra = []
        all_labels = []
        
        for dataset_name in datasets:
            try:
                X, y = rp.datasets.bacteria(dataset_name, folder=data_folder)
                all_spectra.append(X.spectral_data)
                all_labels.extend(y.tolist())
                self.ramanspy_wavenumbers = X.spectral_axis
            except Exception as e:
                print(f"Warning: Could not load {dataset_name}: {e}")
        
        if all_spectra:
            spectra = np.vstack(all_spectra)
            
            # Interpolate each to target wavenumbers
            for i, spectrum in enumerate(spectra):
                processed = self._interpolate_to_target(
                    spectrum, self.ramanspy_wavenumbers
                )
                
                # Ensure floor ≈ 0
                floor = np.percentile(processed, 5)
                processed = np.maximum(processed - floor, 0)
                
                self.ramanspy_spectra.append(processed)
                self.ramanspy_labels.append(all_labels[i])
            
            print(f"Added {len(self.ramanspy_spectra)} Raman spectra from RamanSPy")
    
    def _interpolate_to_target(
        self,
        spectrum: np.ndarray,
        source_wavenumbers: np.ndarray,
    ) -> np.ndarray:
        """Interpolate spectrum to target wavenumber axis."""
        
        # Check if interpolation needed
        if np.allclose(source_wavenumbers, self.target_wavenumbers):
            return spectrum
        
        # Find overlap region
        wn_min = max(source_wavenumbers.min(), self.target_wavenumbers.min())
        wn_max = min(source_wavenumbers.max(), self.target_wavenumbers.max())
        
        # Interpolate
        f = interp1d(
            source_wavenumbers,
            spectrum,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0,
        )
        
        interpolated = f(self.target_wavenumbers)
        
        # Zero out extrapolated regions
        interpolated[self.target_wavenumbers < wn_min] = 0
        interpolated[self.target_wavenumbers > wn_max] = 0
        
        return interpolated
    
    def sample(
        self,
        source: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, str]:
        """
        Sample a Raman spectrum.
        
        Parameters
        ----------
        source : str, optional
            Force source: 'atcc' or 'ramanspy'. If None, uses atcc_weight.
        
        Returns
        -------
        spectrum : np.ndarray
            Raman spectrum at target wavenumbers
        source_name : str
            'atcc' or 'ramanspy'
        label : str
            Species label
        """
        # Determine source
        if source is None:
            use_atcc = self.rng.random() < self.atcc_weight
            source = 'atcc' if use_atcc else 'ramanspy'
        
        # Handle case where requested source is empty
        if source == 'atcc' and not self.atcc_spectra:
            source = 'ramanspy'
        if source == 'ramanspy' and not self.ramanspy_spectra:
            source = 'atcc'
        
        if source == 'atcc':
            idx = self.rng.integers(len(self.atcc_spectra))
            spectrum = self.atcc_spectra[idx].copy()
            label = self.atcc_species[idx]
        else:
            idx = self.rng.integers(len(self.ramanspy_spectra))
            spectrum = self.ramanspy_spectra[idx].copy()
            label = str(self.ramanspy_labels[idx])
        
        # Small augmentation
        scale = self.rng.uniform(0.9, 1.1)
        spectrum *= scale
        
        return spectrum, source, label
    
    def __len__(self) -> int:
        return len(self.atcc_spectra) + len(self.ramanspy_spectra)


class SyntheticBleachingDataset:
    """
    Generate synthetic photobleaching time series with known ground truth.
    
    Combines real Raman spectra with synthetic fluorescence decay.
    Output format compatible with ATCCLoader's xarray output.
    """
    
    def __init__(
        self,
        config: SyntheticConfig,
        raman_library: RamanLibrary,
    ):
        self.config = config
        self.raman_library = raman_library
        self.rng = np.random.default_rng(config.seed)

        # BLEACHING time series (cumulative laser exposure causing fluorescence decay)
        self.bleaching_times = np.array(config.bleaching_times, dtype=float)

        # Integration time(s) for acquisition (affects SNR, not bleaching)
        if isinstance(config.integration_time, (list, tuple)):
            self.integration_times = np.array(config.integration_time, dtype=float)
        else:
            self.integration_times = np.full(len(self.bleaching_times), config.integration_time, dtype=float)

        # Wavenumber axis from library
        self.wavenumbers = raman_library.target_wavenumbers
        self.n_wavenumbers = len(self.wavenumbers)

        # Generate shared fluorophore bases if configured
        if config.shared_bases:
            self.shared_bases = self._generate_fluorophore_bases()
        else:
            self.shared_bases = None

        # Storage
        self.dataset: Optional[xr.Dataset] = None
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '0.05s' to float seconds."""
        return float(time_str.rstrip('s'))
    
    def _generate_fluorophore_bases(self) -> np.ndarray:
        """
        Generate smooth fluorophore basis spectra.
        
        Returns
        -------
        bases : np.ndarray
            Shape (n_fluorophores, n_wavenumbers)
        """
        n_f = self.config.n_fluorophores
        wn = self.wavenumbers
        
        bases = np.zeros((n_f, self.n_wavenumbers))
        
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
            
            # Ensure positivity and normalise
            bases[i] = np.maximum(bases[i], 1e-6)
            bases[i] /= np.trapz(bases[i], wn)
        
        return bases
    
    def _generate_decay_rates(self) -> np.ndarray:
        """Sample decay rates from configured distributions."""
        n_f = self.config.n_fluorophores
        rates = np.zeros(n_f)
        
        for i in range(n_f):
            range_idx = i % len(self.config.decay_rate_ranges)
            log_min, log_max = self.config.decay_rate_ranges[range_idx]
            log_rate = self.rng.uniform(log_min, log_max)
            rates[i] = 10 ** log_rate
        
        return rates
    
    def _generate_abundances(self, raman_intensity: float) -> np.ndarray:
        """
        Generate abundances ensuring proper F/R ratio at t=0.
        """
        n_f = self.config.n_fluorophores
        
        # Target F/R ratio
        fr_ratio = self.rng.uniform(*self.config.fr_ratio_range)
        target_fluor = fr_ratio * raman_intensity
        
        # Random proportions
        proportions = self.rng.dirichlet(np.ones(n_f) * 2)
        
        # Get bases
        bases = self.shared_bases if self.shared_bases is not None else self._generate_fluorophore_bases()
        basis_means = bases.mean(axis=1)
        
        abundances = target_fluor * proportions / (basis_means + 1e-8)
        
        return abundances
    
    def _add_noise(self, signal: np.ndarray, snr: float) -> np.ndarray:
        """Add realistic noise to clean signal."""
        
        if self.config.noise_type == 'gaussian':
            noise_std = signal.mean() / snr
            return signal + self.rng.normal(0, noise_std, signal.shape)
        
        elif self.config.noise_type == 'poisson_gaussian':
            # Poisson (shot noise) + Gaussian (read noise)
            scale = (snr ** 2) / signal.mean() * 0.7
            poisson_noise = self.rng.poisson(np.maximum(signal * scale, 0)) / scale - signal
            
            gaussian_std = signal.mean() / snr * 0.3
            gaussian_noise = self.rng.normal(0, gaussian_std, signal.shape)
            
            return signal + poisson_noise + gaussian_noise
        
        else:
            return signal
    
    def _reconstruct_time_series(
        self,
        raman: np.ndarray,
        bases: np.ndarray,
        abundances: np.ndarray,
        decay_rates: np.ndarray,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct time series from parameters using bleaching times.

        The fluorescence decays according to:
            F_i(t) = w_i * B_i * exp(-λ_i * t_bleach)

        where t_bleach is the cumulative laser exposure time (bleaching time),
        NOT the integration time.

        Returns
        -------
        noisy : np.ndarray
            Noisy time series (n_times, n_wavenumbers)
        clean : np.ndarray
            Clean time series (n_times, n_wavenumbers)
        """
        n_t = len(self.bleaching_times)

        # Raman contribution (constant across all time points)
        clean = np.tile(raman, (n_t, 1))

        # Fluorescence contributions (decay over BLEACHING time)
        for i in range(len(decay_rates)):
            decay = np.exp(-decay_rates[i] * self.bleaching_times)
            fluor_i = abundances[i] * decay[:, None] * bases[i, None, :]
            clean += fluor_i

        # Add noise (could be scaled by integration time in future)
        if add_noise:
            snr = self.rng.uniform(*self.config.snr_range)
            noisy = self._add_noise(clean, snr)
        else:
            noisy = clean.copy()

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
            - 'wavenumber': Spectral axis

            Key coordinates:
            - 'bleaching_time_seconds': Actual bleaching times in seconds
            - 'integration_time_seconds': Integration times used for acquisition
        """
        n_samples = self.config.n_samples
        n_times = len(self.bleaching_times)
        n_wn = self.n_wavenumbers
        n_f = self.config.n_fluorophores

        # Preallocate arrays
        intensity_noisy = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        intensity_clean = np.zeros((n_samples, n_times, n_wn), dtype=np.float32)
        raman_gt = np.zeros((n_samples, n_wn), dtype=np.float32)
        decay_rates_gt = np.zeros((n_samples, n_f), dtype=np.float32)
        abundances_gt = np.zeros((n_samples, n_f), dtype=np.float32)

        # Metadata
        species_list = []
        source_list = []

        # Shared bases
        if self.shared_bases is not None:
            bases_storage = self.shared_bases  # (n_f, n_wn)
        else:
            bases_storage = np.zeros((n_samples, n_f, n_wn), dtype=np.float32)

        print(f"Generating {n_samples} synthetic samples...")

        for i in range(n_samples):
            # Sample real Raman
            raman, source, label = self.raman_library.sample()

            # Generate fluorescence parameters
            decay_rates = self._generate_decay_rates()
            abundances = self._generate_abundances(raman.mean())

            # Get bases
            if self.shared_bases is not None:
                bases = self.shared_bases
            else:
                bases = self._generate_fluorophore_bases()
                bases_storage[i] = bases

            # Reconstruct time series
            noisy, clean = self._reconstruct_time_series(
                raman, bases, abundances, decay_rates
            )

            # Store
            intensity_noisy[i] = noisy
            intensity_clean[i] = clean
            raman_gt[i] = raman
            decay_rates_gt[i] = decay_rates
            abundances_gt[i] = abundances
            species_list.append(label)
            source_list.append(source)

            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{n_samples}")

        # Create formatted labels for bleaching times (for compatibility with ATCC-style indexing)
        bleaching_time_labels = [f"{t:.3g}s" for t in self.bleaching_times]

        # Build xarray Dataset
        ds = xr.Dataset(
            data_vars={
                # Main data - using 'bleaching_time' dimension to reflect true meaning
                'intensity_raw': (
                    ['sample', 'bleaching_time', 'wavenumber'],
                    intensity_noisy,
                    {'long_name': 'Synthetic Raman intensity (noisy)', 'units': 'counts'},
                ),
                'intensity_clean': (
                    ['sample', 'bleaching_time', 'wavenumber'],
                    intensity_clean,
                    {'long_name': 'Synthetic Raman intensity (clean)', 'units': 'counts'},
                ),

                # Ground truth parameters
                'raman_gt': (
                    ['sample', 'wavenumber'],
                    raman_gt,
                    {'long_name': 'Ground truth Raman spectrum'},
                ),
                'decay_rates_gt': (
                    ['sample', 'fluorophore'],
                    decay_rates_gt,
                    {'long_name': 'Ground truth decay rates', 'units': 's⁻¹'},
                ),
                'abundances_gt': (
                    ['sample', 'fluorophore'],
                    abundances_gt,
                    {'long_name': 'Ground truth abundances'},
                ),

                # Metadata
                'species': (['sample'], species_list),
                'source': (['sample'], source_list),

            },
            coords={
                'sample': np.arange(n_samples),
                'bleaching_time': bleaching_time_labels,  # String labels for indexing
                'wavenumber': self.wavenumbers,
                'fluorophore': np.arange(n_f),
                'bleaching_time_seconds': ('bleaching_time', self.bleaching_times),  # Actual times
                'integration_time_seconds': ('bleaching_time', self.integration_times),  # Acquisition times
            },
            attrs={
                'title': 'Synthetic Photobleaching Dataset',
                'n_samples': n_samples,
                'n_fluorophores': n_f,
                'shared_bases': self.config.shared_bases,
                'noise_type': self.config.noise_type,
                'snr_range': str(self.config.snr_range),
                'fr_ratio_range': str(self.config.fr_ratio_range),
                'seed': self.config.seed,
            },
        )
        
        # Add fluorophore bases
        if self.config.shared_bases:
            ds['fluorophore_bases_gt'] = (
                ['fluorophore', 'wavenumber'],
                bases_storage,
                {'long_name': 'Shared fluorophore basis spectra'},
            )
        else:
            ds['fluorophore_bases_gt'] = (
                ['sample', 'fluorophore', 'wavenumber'],
                bases_storage,
                {'long_name': 'Per-sample fluorophore basis spectra'},
            )
        
        self.dataset = ds
        
        print(f"\nGenerated dataset:")
        print(f"  Samples: {n_samples}")
        print(f"  Time points: {n_times}")
        print(f"  Wavenumbers: {n_wn}")
        print(f"  Fluorophores: {n_f}")
        print(f"  ATCC samples: {sum(1 for s in source_list if s == 'atcc')}")
        print(f"  RamanSPy samples: {sum(1 for s in source_list if s == 'ramanspy')}")
        
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