# Physics-based spectral decomposition for Raman/fluorescence separation

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit


class PhysicsDecomposition(nn.Module):
    """
    Physics-constrained decomposition model.

    Model: Y(nu, t) = s(nu) + sum_i w_i * B_i(nu) * exp(-lambda_i * t)

    where:
        - s(nu): Raman spectrum (time-invariant)
        - B_i(nu): Fluorophore basis spectra (unit-normalized, non-negative)
        - w_i: Abundance/amplitude of each fluorophore
        - lambda_i: Decay rate of each fluorophore

    Basis spectra can be either:
        - 'free': Independent parameter per wavenumber (default)
        - 'polynomial': Parameterised as polynomial of specified degree
    """

    def __init__(
        self,
        n_wavenumbers: int,
        n_timepoints: int,
        n_fluorophores: int = 3,
        time_values: Optional[torch.Tensor] = None,
        initial_abundances: Optional[torch.Tensor] = None,
        initial_decay_rates: Optional[torch.Tensor] = None,
        initial_fluorophore_bases: Optional[torch.Tensor] = None,
        initial_raman_spectrum: Optional[torch.Tensor] = None,

        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        basis_type: str = "free",
        polynomial_degree: int = 8,
        wavenumber_axis: Optional[torch.Tensor] = None,
        min_decay_rate: float = 1.0,
    ):
        super().__init__()
        self.n_wavenumbers = n_wavenumbers
        self.n_timepoints = n_timepoints
        self.n_fluorophores = n_fluorophores
        self.device = device
        self.basis_type = basis_type
        self.polynomial_degree = polynomial_degree
        self.min_decay_rate = min_decay_rate  # Store for use in property

        # Time axis (buffer = not a learnable parameter)
        if time_values is not None:
            self.register_buffer("time_values", time_values.to(device))
        else:
            self.register_buffer(
                "time_values",
                torch.arange(n_timepoints, dtype=torch.float32, device=device),
            )

        # Raman spectrum s(nu) - stored in log-space, exp applied in property
        if initial_raman_spectrum is not None:
            if isinstance(initial_raman_spectrum, torch.Tensor):
                raman_tensor = initial_raman_spectrum.detach().clone().to(device)
            else:
                raman_tensor = torch.tensor(
                    initial_raman_spectrum, dtype=torch.float32, device=device
                )
            # Apply log transform (parameter stored in log-space, positivity via exp in property)
            self.raman_spectrum = nn.Parameter(torch.log(raman_tensor + 1e-8))
        else:
            # Random initialization in log-space (exp(0) = 1)
            self.raman_spectrum = nn.Parameter(
                torch.randn(n_wavenumbers, device=device) * 0.1
            )

        # Fluorophore bases - parameterisation depends on basis_type
        if basis_type == "free":
            if initial_fluorophore_bases is not None:
                # Initialize from provided bases
                self.fluorophore_bases_raw = nn.Parameter(
                    torch.log(initial_fluorophore_bases + 1e-8) # exp(log(B)) = B
                )
            else:
                # Random initialization
                self.fluorophore_bases_raw = nn.Parameter(
                    torch.randn(n_fluorophores, n_wavenumbers, device=device) * 0.1
                )
        elif basis_type == "polynomial":
            n_coeffs = polynomial_degree + 1
            if initial_fluorophore_bases is not None:
                # Fit polynomial coefficients to each provided fluorophore basis
                assert (
                    wavenumber_axis is not None
                ), "wavenumber_axis required for polynomial initialization"
                wn_np = (
                    wavenumber_axis.cpu().numpy()
                    if isinstance(wavenumber_axis, torch.Tensor)
                    else wavenumber_axis
                )
                bases_np = (
                    initial_fluorophore_bases.cpu().numpy()
                    if isinstance(initial_fluorophore_bases, torch.Tensor)
                    else initial_fluorophore_bases
                )

                # Normalize wavenumbers for numerical stability
                wn_min, wn_max = wn_np.min(), wn_np.max()
                wn_norm = 2.0 * (wn_np - wn_min) / (wn_max - wn_min + 1e-8) - 1.0

                # Fit polynomial to LOG of each fluorophore basis (since we apply exp later)
                # This ensures exp(poly(x)) ≈ original_bases
                coeffs_list = []
                for i in range(n_fluorophores):
                    # Fit to log-space (add small epsilon to avoid log(0))
                    log_bases = np.log(bases_np[i] + 1e-8)
                    coeffs = np.polyfit(wn_norm, log_bases, deg=polynomial_degree)
                    # IMPORTANT: np.polyfit returns coefficients in DESCENDING order [c_n, ..., c_0]
                    # But Vandermonde matrix uses ASCENDING powers [x^0, x^1, ..., x^n]
                    # So we must reverse the coefficient order!
                    coeffs = coeffs[::-1]  # Reverse to [c_0, c_1, ..., c_n]
                    coeffs_list.append(coeffs)

                coeffs_array = np.array(coeffs_list)  # (n_fluorophores, n_coeffs)
                self.poly_coeffs = nn.Parameter(
                    torch.tensor(coeffs_array, dtype=torch.float32, device=device)
                )
            else:
                # Random initialization: each fluorophore has (degree + 1) coefficients
                self.poly_coeffs = nn.Parameter(
                    torch.randn(n_fluorophores, n_coeffs, device=device) * 0.1
                )

            # Normalised wavenumber axis for polynomial evaluation
            # Normalise to [-1, 1] for numerical stability
            if wavenumber_axis is not None:
                wn = wavenumber_axis.to(device)
            else:
                wn = torch.arange(n_wavenumbers, dtype=torch.float32, device=device)

            wn_min, wn_max = wn.min(), wn.max()
            wn_normalised = 2.0 * (wn - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
            self.register_buffer("wn_normalised", wn_normalised)

            # Precompute Vandermonde matrix: (n_wavenumbers, n_coeffs)
            # Each column is wn^k for k = 0, 1, ..., degree
            # We use vandermonde so that we can efficiently compute polynomial values for all wavenumbers at once, for all fluorophores.
            vandermonde = torch.stack(
                [wn_normalised**k for k in range(n_coeffs)], dim=1
            )
            self.register_buffer("vandermonde", vandermonde)
        else:
            raise ValueError(
                f"Unknown basis_type: {basis_type}. Use 'free' or 'polynomial'."
            )

        # Abundances w_i - raw parameters, positivity enforced via exp
        if initial_abundances is not None:
            if isinstance(initial_abundances, torch.Tensor):
                abundances_tensor = initial_abundances.detach().clone().to(device)
            else:
                abundances_tensor = torch.tensor(
                    initial_abundances, dtype=torch.float32, device=device
                )
            self.abundances_raw = nn.Parameter(torch.log(abundances_tensor + 1e-8))
        else:
            self.abundances_raw = nn.Parameter(
                torch.zeros(n_fluorophores, device=device)
            )

        # Decay rates lambda_i - raw parameters, positivity enforced via exp
        # Initialize randomly near zero - exp(0.1*randn) ≈ 0.9-1.1
        # After adding min_decay_rate, initial τ ≈ 1/(1+min_decay_rate)
        if initial_decay_rates is not None:
            if isinstance(initial_decay_rates, torch.Tensor):
                rates_tensor = initial_decay_rates.detach().clone().to(device)
            else:
                rates_tensor = torch.tensor(
                    initial_decay_rates, dtype=torch.float32, device=device
                )

            # Check for negative values after subtracting min_decay_rate
            adjusted_rates = rates_tensor - min_decay_rate
            if torch.any(adjusted_rates <= 0):
                raise ValueError(
                    f"initial_decay_rates must be > min_decay_rate ({min_decay_rate}). "
                    f"Got rates: {rates_tensor}, adjusted: {adjusted_rates}"
                )

            self.decay_rates_raw = nn.Parameter(torch.log(adjusted_rates + 1e-8))
        else:
            self.decay_rates_raw = nn.Parameter(
                torch.randn(n_fluorophores, device=device) * 0.1
            )

    @property
    def fluorophore_bases(self) -> torch.Tensor:
        """
        Non-negative fluorophore bases.

        For 'free' basis_type: exp transformation of raw parameters.
        For 'polynomial' basis_type: evaluate polynomial and apply exp for positivity.

        Note: Using exp (not softmax) preserves absolute intensity scale, allowing
        meaningful comparison between fluorescence and Raman intensities.
        """
        if self.basis_type == "free":
            bases = torch.exp(self.fluorophore_bases_raw)
        elif self.basis_type == "polynomial":
            poly_values = torch.matmul(self.poly_coeffs, self.vandermonde.T)
            bases = torch.exp(poly_values)
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")
        
        norm = torch.norm(bases, p=2, dim=1, keepdim=True)
        bases = bases / (norm + 1e-8)
        return bases

    @property
    def raman_spectra(self) -> torch.Tensor:
        """Non-negative abundances via exp transformation."""
        return torch.exp(self.raman_spectrum)

    @property
    def abundances(self) -> torch.Tensor:
        """Non-negative abundances via exp transformation."""
        return torch.exp(self.abundances_raw)

    @property
    def decay_rates(self) -> torch.Tensor:
        """Positive decay rates via exp transformation with configurable minimum floor."""
        return torch.exp(self.decay_rates_raw) + self.min_decay_rate

    def forward(self) -> torch.Tensor:
        """
        Reconstruct the full time series.

        Returns:
            Tensor of shape (n_timepoints, n_wavenumbers)
        """
        # Raman contribution: (1, n_wavenumbers)
        raman = self.raman_spectra.unsqueeze(0)

        # Decay factors: exp(-lambda_i * t) -> (n_timepoints, n_fluorophores)
        decay_factors = torch.exp(
            -self.decay_rates.unsqueeze(0) * self.time_values.unsqueeze(1)
        )
        # This models each exponential term. It will take the form: $exp(-lambda_i * t_j)$ for each fluorophore. The above function simultaneously computes this for all timepoints and fluorophores.
        # We are left with a 2D tensor where each column corresponds to a fluorophore and each row corresponds to the value of the exponential decay at a specific timepoint.

        # Weighted bases: w_i * B_i(nu) -> (n_fluorophores, n_wavenumbers)
        # Both abundances and bases are already positive via exp transformation
        # We are obtaining here each fluorophore base spectra (B_i(nu)) scaled by its abundance (w_i).
        weighted_bases = self.abundances.unsqueeze(1) * self.fluorophore_bases

        # Fluorescence: sum_i [decay(t,i) * weighted_basis(i,nu)]
        # (n_timepoints, n_fluorophores) @ (n_fluorophores, n_wavenumbers) -> (n_timepoints, n_wavenumbers)
        # This just simultaneously computes the weighted sum of all fluorophore contributions at each timepoint across all wavenumbers.
        fluorescence = torch.matmul(decay_factors, weighted_bases)

        return raman + fluorescence

    def compute_spectral_separation_loss(self) -> torch.Tensor:
        """
        Enforce spectral separation: fluorescence must be smooth, Raman can be sharp.

        This breaks the degeneracy between s(ν) and w·B(ν) by requiring that
        fluorophore bases have LOW high-frequency content while allowing Raman
        to have sharp peaks.

        Key insight: Without this, the model can shift sharp spectral features
        between s(ν) and fluorescence bases by scaling abundances, making the
        problem underdetermined even with strict decay rate constraints.

        Returns:
            Ratio of fluorescence smoothness to Raman sharpness (minimize this)
        """
        bases = self.fluorophore_bases  # (n_fluorophores, n_wavenumbers)
        raman = self.raman_spectra

        # High-frequency content via second derivative
        bases_second_deriv = bases[:, 2:] - 2 * bases[:, 1:-1] + bases[:, :-2]
        raman_second_deriv = raman[2:] - 2 * raman[1:-1] + raman[:-2]

        # Mean absolute curvature (scale-invariant measure of "sharpness")
        # Add small epsilon to avoid division by zero
        bases_curvature = torch.mean(torch.abs(bases_second_deriv)) + 1e-8
        raman_curvature = torch.mean(torch.abs(raman_second_deriv)) + 1e-8

        # We want: bases_curvature << raman_curvature
        # Penalize if fluorescence is too sharp OR if Raman is too smooth
        curvature_ratio = bases_curvature / (raman_curvature + 1e-8)

        return curvature_ratio

    def compute_abundance_penalty(self) -> torch.Tensor:
        """
        Penalize very large abundances that could compensate for fast decay rates.

        This prevents the model from using w_i·B_i(ν) to absorb Raman signal
        by scaling abundances arbitrarily high.

        Returns:
            L2 penalty on log-abundances (encourages smaller w_i)
        """
        # Penalize deviation from abundance ~ 1.0
        # abundances_raw = 0 → abundances = 1.0
        return torch.mean(self.abundances_raw**2)

    def compute_decay_diversity_penalty(self) -> torch.Tensor:
        """
        Penalize when all decay rates cluster together (e.g., all τ → τ_max).

        This encourages temporal diversity: if you have 3 fluorophores, they
        should span fast/medium/slow timescales, not all decay identically.

        Returns:
            Penalty for low diversity (high when all rates similar)
        """
        rates = self.decay_rates  # (n_fluorophores,)

        # Compute coefficient of variation: std / mean
        # High CV = diverse rates, Low CV = clustered rates
        mean_rate = torch.mean(rates)
        std_rate = torch.std(rates)
        cv = std_rate / (mean_rate + 1e-8)

        # Target CV ~ 0.5 (rates should span at least 2× range)
        # Penalize if CV is too small
        target_cv = 0.5
        penalty = torch.relu(target_cv - cv) ** 2

        return penalty

    def compute_intensity_ratio_loss(self, t_early: float = 0.0) -> torch.Tensor:
        """
        Enforce that fluorescence dominates over Raman at early times.

        Biological constraint: At t≈0, autofluorescence should be stronger
        than Raman signal (typically 5-10×). This prevents the model from
        putting too much intensity in s(ν) and too little in fluorescence.

        Args:
            t_early: Time point to evaluate (default t=0)

        Returns:
            Penalty if Raman is too strong relative to fluorescence at t_early
        """
        # Fluorescence at t_early: Σ w_i · B_i(ν) · exp(-λ_i · t_early)
        decay_at_t = torch.exp(-self.decay_rates * t_early)  # (n_fluorophores,)
        weighted_bases = (
            self.abundances.unsqueeze(1) * self.fluorophore_bases
        )  # (n_fluorophores, n_wavenumbers)
        fluorescence_at_t = torch.sum(
            decay_at_t.unsqueeze(1) * weighted_bases, dim=0
        )  # (n_wavenumbers,)

        # Raman intensity
        raman_intensity = torch.abs(self.raman_spectra)  # (n_wavenumbers,)

        # Average intensities
        fluor_mean = torch.mean(fluorescence_at_t)
        raman_mean = torch.mean(raman_intensity)

        # We want: fluor_mean > raman_mean (ideally fluor ~ 5× raman)
        # Penalize if ratio is too small
        ratio = fluor_mean / (raman_mean + 1e-8)
        target_ratio = 6

        # Penalize if ratio < target
        penalty = torch.relu(target_ratio - ratio) ** 2

        return penalty

    def compute_late_time_consistency_loss(
        self, t1: float = 20.0, t2: float = 25.0
    ) -> torch.Tensor:
        """
        Penalize if model predicts different spectra at two late time points.

        Physics: If fluorescence has mostly decayed, late-time predictions should
        be nearly identical (both ≈ pure Raman). Large differences indicate slow
        fluorescence components still decaying.

        This is NOT cheating: uses only model's predictions, no ground truth!

        Args:
            t1, t2: Two late time points beyond training window

        Returns:
            MSE between predictions at t1 and t2
        """
        # Model predictions at two late times
        decay1 = torch.exp(-self.decay_rates * t1)
        decay2 = torch.exp(-self.decay_rates * t2)

        weighted_bases = self.abundances.unsqueeze(1) * self.fluorophore_bases

        fluor1 = torch.sum(decay1.unsqueeze(1) * weighted_bases, dim=0)
        fluor2 = torch.sum(decay2.unsqueeze(1) * weighted_bases, dim=0)

        pred1 = self.raman_spectra + fluor1
        pred2 = self.raman_spectra + fluor2

        # Should be nearly identical if fluorescence has decayed
        consistency_error = torch.mean((pred1 - pred2) ** 2)

        return consistency_error

    def compute_raman_floor_loss(self) -> torch.Tensor:
        """
        Penalize elevated floor in Raman spectrum.

        Physical basis: Raman scattering only occurs at specific molecular
        vibration frequencies. Between peaks, there should be no signal.
        An elevated floor indicates slow fluorescence absorbed into s(ν).

        Returns:
            Penalty for non-zero floor (5th percentile)
        """
        # Use percentile rather than min to be robust to noise
        floor = torch.quantile(self.raman_spectra, 0.05)

        # Floor should be near zero
        return floor**2

    def compute_raman_spikiness_loss(self) -> torch.Tensor:
        """
        Penalize if Raman spectrum is TOO SMOOTH (curve-like).

        Physical basis: Raman spectra have sharp peaks from vibrational modes.
        If s(ν) looks like a smooth curve/banana, fluorescence bases have
        absorbed the peaks, leaving only a smooth residual.

        We measure "spikiness" as the ratio of high-frequency to low-frequency
        power. Sharp peaks → high ratio. Smooth curve → low ratio.

        Returns:
            Penalty if Raman lacks high-frequency content
        """
        raman = self.raman_spectra

        # High-frequency content: second derivative (curvature)
        second_deriv = raman[2:] - 2 * raman[1:-1] + raman[:-2]
        high_freq_power = torch.mean(torch.abs(second_deriv))

        # Low-frequency content: first derivative (slope)
        first_deriv = raman[1:] - raman[:-1]
        low_freq_power = torch.mean(torch.abs(first_deriv)) + 1e-8

        # Spikiness ratio: should be high for sharp peaks
        spikiness = high_freq_power / low_freq_power

        target_spikiness = 0.5
        penalty = torch.relu(target_spikiness - spikiness) ** 2

        return penalty

    def compute_raman_curvature_loss(self) -> torch.Tensor:
        """
        Penalize if Raman spectrum has too much GLOBAL curvature (banana shape).

        If s(ν) is a smooth polynomial-like curve, it indicates fluorescence
        absorption. We fit a low-order polynomial and penalize if Raman is
        well-approximated by it.

        Returns:
            Penalty for smooth global curvature
        """
        raman = self.raman_spectra

        # Fit a cubic polynomial to Raman (represents global trend/curvature)
        # If Raman is mostly a cubic, it's too smooth
        n = raman.shape[0]
        x = torch.linspace(-1, 1, n, device=raman.device)

        # Build Vandermonde matrix for cubic fit
        X = torch.stack([torch.ones_like(x), x, x**2, x**3], dim=1)

        # Least squares fit: (X^T X)^-1 X^T y
        XtX = torch.matmul(X.T, X)
        Xty = torch.matmul(X.T, raman)
        coeffs = torch.linalg.lstsq(XtX, Xty).solution

        # Reconstruct smooth curve
        smooth_curve = torch.matmul(X, coeffs)

        # If Raman is well-fit by cubic, it's too smooth (penalize!)
        # Measure R² of cubic fit
        residuals = raman - smooth_curve
        ss_res = torch.sum(residuals**2)
        ss_tot = torch.sum((raman - torch.mean(raman)) ** 2) + 1e-8
        r_squared = 1 - ss_res / ss_tot

        # Penalize high R² (good fit to smooth curve = bad!)
        # Target: Raman should NOT be well-fit by cubic (R² < 0.5)
        target_r2 = 0.5
        penalty = torch.relu(r_squared - target_r2) ** 2

        return penalty

    def compute_fluorophore_convexity_loss(self) -> torch.Tensor:
        """
        Penalize if fluorophore bases are CONVEX (U-shaped).

        Fluorescence emission should be unimodal (broad peak) or monotonic,
        NEVER convex (minimum in middle). Convex bases are unphysical and
        can absorb sharp Raman features.

        Returns:
            Penalty proportional to positive second derivative (convexity)
        """
        bases = self.fluorophore_bases  # (n_fluorophores, n_wavenumbers)

        # Compute second derivative: d²B/dν²
        # Positive second derivative = convex (U-shaped) = BAD
        # Negative second derivative = concave (inverted U) = OK
        second_deriv = bases[:, 2:] - 2 * bases[:, 1:-1] + bases[:, :-2]

        # Penalize positive second derivative (convexity)
        # ReLU ensures we only penalize convex regions, not concave
        convexity_penalty = torch.mean(torch.relu(second_deriv) ** 2)

        return convexity_penalty

    def compute_decay_rate_prior_loss(self, target_rates: torch.Tensor) -> torch.Tensor:
        """
        Soft regularization toward estimated decay rates from scipy fit.

        This encourages the learned rates to be close to the initial estimate
        from fitting the wavenumber-averaged decay, but allows deviation if
        the spectral structure requires it.

        Args:
            target_rates: Estimated decay rates from scipy curve_fit (n_fluorophores,)

        Returns:
            MSE between learned log-rates and target log-rates
        """
        # Use log-space to handle the wide range of decay rates
        # This makes the penalty scale-invariant
        log_learned = torch.log(self.decay_rates)
        log_target = torch.log(target_rates)
        return torch.mean((log_learned - log_target) ** 2)

    def compute_extrapolation_validation_loss(
        self,
        data: torch.Tensor,
        first_times: int,
        fit_frames: int = 10,
        val_frames: int = 10,
    ) -> torch.Tensor:
        """
        Internal cross-validation within training window (no cheating!).

        Fits on first `fit_frames` of training data, validates on next `val_frames`.
        Both are within first_times, so no information leakage.

        Example: If first_times=20, fit_frames=10, val_frames=10:
          - Fit on frames 0-9
          - Validate on frames 10-19
          - Never access frames 20+

        This tests if the model can extrapolate within the training window.
        """
        if fit_frames + val_frames > first_times:
            raise ValueError(
                f"fit_frames ({fit_frames}) + val_frames ({val_frames}) = {fit_frames+val_frames} "
                f"exceeds first_times ({first_times}). Must stay within training window!"
            )

        reconstruction = self.forward()

        val_start = fit_frames
        val_end = fit_frames + val_frames

        # Only access data within training window
        val_residuals = reconstruction[val_start:val_end] - data[val_start:val_end]
        return torch.mean(val_residuals**2)

    def get_decomposition(self) -> Dict[str, np.ndarray]:
        """Return decomposed components as numpy arrays, sorted by decay rate (fast to slow)."""
        with torch.no_grad():
            rates = self.decay_rates.cpu().numpy()
            sort_idx = np.argsort(rates)[::-1]  # Descending: fast first

            result = {
                "raman": self.raman_spectra.cpu().numpy(),
                "fluorophore_bases": self.fluorophore_bases.cpu().numpy()[sort_idx],
                "abundances": self.abundances.cpu().numpy()[sort_idx],
                "decay_rates": rates[sort_idx],
                "time_constants": (1.0 / rates)[sort_idx],
            }

            # Include polynomial coefficients if using polynomial basis
            if self.basis_type == "polynomial":
                result["poly_coeffs"] = self.poly_coeffs.cpu().numpy()[sort_idx]

            return result

    def get_fluorescence_component(self, component_idx: int) -> np.ndarray:
        """
        Get time-resolved fluorescence for one component.

        Args:
            component_idx: Index in sorted order (0 = fastest decay)

        Returns:
            Array of shape (n_timepoints, n_wavenumbers)
        """
        with torch.no_grad():
            # Map sorted index to actual parameter index
            rates = self.decay_rates.cpu().numpy()
            # Sorted by decay rate
            sort_idx = np.argsort(rates)[::-1]
            actual_idx = sort_idx[component_idx]

            decay = torch.exp(-self.decay_rates[actual_idx] * self.time_values)
            amplitude = self.abundances[actual_idx]
            basis = self.fluorophore_bases[actual_idx]

            # (n_timepoints,) * scalar * (n_wavenumbers,) -> (n_timepoints, n_wavenumbers)
            component = decay.unsqueeze(1) * amplitude * basis.unsqueeze(0)
            return component.cpu().numpy()


def generate_random_mask(
    n_timepoints: int, n_wavenumbers: int, mask_fraction: float, device: str
) -> torch.Tensor:
    """
    Generate a random binary mask for masked fitting.

    Positions where mask=1 are EXCLUDED from loss computation.
    Positions where mask=0 are INCLUDED in loss computation.

    Args:
        n_timepoints: Number of time points in the data
        n_wavenumbers: Number of wavenumber positions
        mask_fraction: Fraction of positions to mask (0.0 to 1.0)
        device: Torch device for tensor creation

    Returns:
        Binary mask tensor of shape (n_timepoints, n_wavenumbers)
    """
    # torch.rand generates uniform [0, 1), positions below threshold are masked
    mask = (
        torch.rand(n_timepoints, n_wavenumbers, device=device) < mask_fraction
    ).float()
    return mask


def estimate_decay_rates_from_early_frames(
    data: np.ndarray, time_values: np.ndarray, first_times: int, n_components: int = 3
) -> Tuple[np.ndarray, dict]:
    """
    Estimate decay rates by fitting exponentials to wavenumber-averaged signal.

    Uses only the first `first_times` frames to avoid information leakage.
    This provides a physics-informed prior for the neural network optimization.

    Args:
        data: Full time series (n_timepoints, n_wavenumbers)
        time_values: Time axis in seconds
        first_times: Number of early frames to use (training window)
        n_components: Number of exponential components to fit

    Returns:
        Tuple of (decay_rates array, fit_info dict)
    """
    # Use only training frames
    train_data = data[:first_times]
    train_times = time_values[:first_times]

    # Wavenumber-averaged decay curve (removes spectral structure)
    avg_decay = train_data.mean(axis=1)

    # Define exponential models
    def single_exp(t, A, tau, offset):
        return A * np.exp(-t / tau) + offset

    def double_exp(t, A1, tau1, A2, tau2, offset):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + offset

    def triple_exp(t, A1, tau1, A2, tau2, A3, tau3, offset):
        return (
            A1 * np.exp(-t / tau1)
            + A2 * np.exp(-t / tau2)
            + A3 * np.exp(-t / tau3)
            + offset
        )

    # Select model based on n_components
    if n_components == 1:
        func = single_exp
        amp_init = avg_decay[0] - avg_decay[-1]
        p0 = [amp_init, 0.5, avg_decay[-1]]
        bounds = ([0, 1e-4, -np.inf], [np.inf, 100, np.inf])
    elif n_components == 2:
        func = double_exp
        amp_init = (avg_decay[0] - avg_decay[-1]) / 2
        p0 = [amp_init, 0.1, amp_init, 1.0, avg_decay[-1]]
        bounds = ([0, 1e-4, 0, 1e-4, -np.inf], [np.inf, 100, np.inf, 100, np.inf])
    else:  # 3 components
        func = triple_exp
        amp_init = (avg_decay[0] - avg_decay[-1]) / 3
        p0 = [amp_init, 0.05, amp_init, 0.3, amp_init, 2.0, avg_decay[-1]]
        bounds = (
            [0, 1e-4, 0, 1e-4, 0, 1e-4, -np.inf],
            [np.inf, 100, np.inf, 100, np.inf, 100, np.inf],
        )

    try:
        popt, pcov = curve_fit(
            func, train_times, avg_decay, p0=p0, bounds=bounds, maxfev=10000
        )

        # Extract tau values and convert to decay rates (lambda = 1/tau)
        if n_components == 1:
            taus = np.array([popt[1]])
        elif n_components == 2:
            taus = np.array([popt[1], popt[3]])
        else:
            taus = np.array([popt[1], popt[3], popt[5]])

        decay_rates = 1.0 / taus

        # Compute fit quality on training data only
        fitted = func(train_times, *popt)
        residuals = avg_decay - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((avg_decay - avg_decay.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        fit_info = {
            "success": True,
            "r_squared": r_squared,
            "taus": taus,
            "decay_rates": decay_rates,
            "popt": popt,
            "residual_std": np.std(residuals),
        }

        return decay_rates, fit_info

    except (RuntimeError, ValueError) as e:
        # Fitting failed - return default reasonable values
        default_taus = np.array([0.1, 0.5, 2.0])[:n_components]
        fit_info = {
            "success": False,
            "error": str(e),
            "taus": default_taus,
            "decay_rates": 1.0 / default_taus,
        }
        return 1.0 / default_taus, fit_info


def extract_initial_params_from_dataset(
    dataset: xr.Dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Extract initialization parameters from synthetic bleaching dataset.

    Args:
        dataset: xarray.Dataset with structure:
            - Coordinates: bleaching_time, wavenumber
            - Data vars: raman_gt, decay_rates_gt, abundances_gt, fluorophore_bases_gt
        device: Target device for tensors

    Returns:
        Dictionary with initialization parameters for fit_physics_model()

    Example:
        >>> # Load synthetic sample
        >>> sample_ds = synthetic_ds.isel(sample=0)
        >>> init_params = extract_initial_params_from_dataset(sample_ds)
        >>>
        >>> # Fit with ground truth initialization
        >>> model, history = fit_physics_model(
        ...     data=sample_ds['intensity_clean'].values,
        ...     time_values=sample_ds['bleaching_time'].values,
        ...     wavenumber_axis=sample_ds['wavenumber'].values,
        ...     **init_params
        ... )
    """
    init_params = {}

    # Extract fluorophore bases
    if "fluorophore_bases_gt" in dataset:
        bases = dataset["fluorophore_bases_gt"].values
        if bases.ndim == 3:
            # (sample, fluorophore, wavenumber) - need to select sample
            raise ValueError(
                f"fluorophore_bases_gt has 3 dimensions (shape={bases.shape}). "
                f"Did you forget to select a sample? Use: dataset.isel(sample=idx)"
            )
        init_params["initial_fluorophore_bases"] = torch.tensor(
            bases, dtype=torch.float32, device=device
        )

    # Extract decay rates
    if "decay_rates_gt" in dataset:
        rates = dataset["decay_rates_gt"].values
        if rates.ndim == 2:
            # (sample, fluorophore) - need to select sample
            raise ValueError(
                f"decay_rates_gt has 2 dimensions (shape={rates.shape}). "
                f"Did you forget to select a sample? Use: dataset.isel(sample=idx)"
            )
        init_params["initial_decay_rates"] = torch.tensor(
            rates, dtype=torch.float32, device=device
        )

    # Extract abundances
    if "abundances_gt" in dataset:
        abundances = dataset["abundances_gt"].values
        if abundances.ndim == 2:
            # (sample, fluorophore) - need to select sample
            raise ValueError(
                f"abundances_gt has 2 dimensions (shape={abundances.shape}). "
                f"Did you forget to select a sample? Use: dataset.isel(sample=idx)"
            )
        init_params["initial_abundances"] = torch.tensor(
            abundances, dtype=torch.float32, device=device
        )

    # Extract Raman spectrum
    if "raman_gt" in dataset:
        raman = dataset["raman_gt"].values
        if raman.ndim == 2:
            # (sample, wavenumber) - need to select sample
            raise ValueError(
                f"raman_gt has 2 dimensions (shape={raman.shape}). "
                f"Did you forget to select a sample? Use: dataset.isel(sample=idx)"
            )
        init_params["initial_raman_spectrum"] = torch.tensor(
            raman, dtype=torch.float32, device=device
        )

    return init_params


def fit_physics_model(
    data: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    n_fluorophores: int = 3,
    n_epochs: int = 5000,
    lr: float = 0.01,
    first_times: int = 10,
    initial_fluorophore_bases: Optional[torch.Tensor] = None,
    initial_decay_rates: Optional[torch.Tensor] = None,
    initial_abundances: Optional[torch.Tensor] = None,
    initial_raman_spectrum: Optional[torch.Tensor] = None,
    # reg_spectral_separation: float = 0.0,
    # reg_abundance_penalty: float = 0.0,
    # reg_decay_diversity: float = 0.0,
    # reg_intensity_ratio: float = 0.0,
    # reg_decay_prior: float = 0.0,
    # reg_late_time_consistency: float = 0.0,
    # reg_extrapolation_validation: float = 0.0,

    # reg_raman_spikiness: float = 0.0,
    # reg_raman_curvature: float = 0.0,
    # reg_fluorophore_convexity: float = 0.0,
    min_decay_rate: float = 1.0,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    basis_type: str = "free",
    freeze_bases: bool = False,
    polynomial_degree: int = 8,
    wavenumber_axis: Optional[np.ndarray] = None,
    normalize_data: bool = True,
) -> Tuple[PhysicsDecomposition, Dict]:
    """
    Fit the physics decomposition model to time-series spectral data.

    The model learns s(nu) through:
    1. Reconstruction loss on first_times frames
    2. Spectral shape constraints (fluorescence smooth, Raman sharp)
    3. Decay rate constraints ensuring fluorescence decays during measurement
    4. Diversity constraints ensuring components span different timescales

    Initialization:
        - If initial_* parameters are None: Random initialization
        - If initial_* parameters provided: Warm start from given values
        - Use extract_initial_params_from_dataset() to extract from xarray.Dataset

    Args:
        data: Time series of shape (n_timepoints, n_wavenumbers)
        time_values: Actual time values in seconds. If None, uses indices.
        n_fluorophores: Number of exponential decay components (ignored if initial_fluorophore_bases provided)
        n_epochs: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        first_times: Number of initial time points to include in reconstruction loss.
        initial_fluorophore_bases: Initial fluorophore basis spectra (n_fluorophores, n_wavenumbers)
        initial_decay_rates: Initial decay rates (n_fluorophores,)
        initial_abundances: Initial abundances (n_fluorophores,)
        initial_raman_spectrum: Initial Raman spectrum (n_wavenumbers,)
        reg_spectral_separation: Weight for enforcing fluorescence smoothness vs Raman sharpness.
                                 This breaks degeneracy between s(ν) and w·B(ν).
        reg_abundance_penalty: Weight for penalizing large abundances (prevents compensation).
        reg_decay_diversity: Weight for encouraging diverse decay rates (prevents all τ clustering).
        reg_decay_prior: Weight for soft regularization toward scipy-estimated decay rates.
                         Uses only first_times frames, so no information leakage.
        min_decay_rate: Minimum decay rate (λ_min). τ_max = 1/λ_min.
                        This is the primary constraint ensuring fluorescence decays.
        mask_fraction: Fraction of positions to exclude from loss each epoch (N2V style).
        verbose: Print progress every 1000 epochs
        device: Computation device ('cuda' or 'cpu')
        basis_type: 'free' or 'polynomial'
        polynomial_degree: Degree of polynomial for basis spectra
        wavenumber_axis: Actual wavenumber values for polynomial evaluation.

    Returns:
        Tuple of (fitted model, training history dict)
    """

    scale_factor = 1.0

    if normalize_data:
        scale_factor = np.max(data)
        if verbose:
            print(f"Data normalized by factor: {scale_factor}")

        # Normalize Data
        data = data / scale_factor

        # Normalize Initial Guesses (Crucial!)
        # We do NOT normalize bases (they are unitless shapes)
        # We do NOT normalize rates (they are time-dependent, not intensity-dependent)
        if initial_raman_spectrum is not None:
            initial_raman_spectrum = initial_raman_spectrum / scale_factor
        if initial_abundances is not None:
            initial_abundances = initial_abundances / scale_factor

    n_timepoints, n_wavenumbers = data.shape
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    # Build time tensor
    if time_values is None:
        time_values = np.arange(n_timepoints, dtype=np.float32)
    time_tensor = torch.tensor(time_values, dtype=torch.float32, device=device)

    # Convert wavenumber axis to tensor if provided
    wn_tensor = None
    if wavenumber_axis is not None:
        if isinstance(wavenumber_axis, np.ndarray):
            wn_tensor = torch.tensor(
                wavenumber_axis, dtype=torch.float32, device=device
            )
        else:
            wn_tensor = wavenumber_axis.to(device)

    # Determine n_fluorophores from initial parameters if provided
    if initial_fluorophore_bases is not None:
        if isinstance(initial_fluorophore_bases, np.ndarray):
            n_fluorophores = initial_fluorophore_bases.shape[0]
            initial_fluorophore_bases = torch.tensor(
                initial_fluorophore_bases, dtype=torch.float32, device=device
            )
        elif isinstance(initial_fluorophore_bases, torch.Tensor):
            n_fluorophores = initial_fluorophore_bases.shape[0]
            initial_fluorophore_bases = initial_fluorophore_bases.to(device)
        else:
            raise TypeError(
                f"initial_fluorophore_bases must be numpy array or torch.Tensor, got {type(initial_fluorophore_bases)}"
            )

        if verbose:
            print(
                f"✓ Using {n_fluorophores} fluorophores from initial bases (shape: {initial_fluorophore_bases.shape})"
            )

    if initial_decay_rates is not None:
        if isinstance(initial_decay_rates, np.ndarray):
            initial_decay_rates = torch.tensor(
                initial_decay_rates, dtype=torch.float32, device=device
            )
        elif isinstance(initial_decay_rates, torch.Tensor):
            initial_decay_rates = initial_decay_rates.to(device)
        else:
            raise TypeError(
                f"initial_decay_rates must be numpy array or torch.Tensor, got {type(initial_decay_rates)}"
            )

        if verbose:
            rates_np = initial_decay_rates.cpu().numpy()
            taus_np = 1.0 / rates_np
            print(f"✓ Using decay rates: λ = {rates_np} s⁻¹")
            print(f"  Time constants: τ = {taus_np} s")

    if initial_abundances is not None:
        if isinstance(initial_abundances, np.ndarray):
            initial_abundances = torch.tensor(
                initial_abundances, dtype=torch.float32, device=device
            )
        elif isinstance(initial_abundances, torch.Tensor):
            initial_abundances = initial_abundances.to(device)
        else:
            raise TypeError(
                f"initial_abundances must be numpy array or torch.Tensor, got {type(initial_abundances)}"
            )

        if verbose:
            abundances_np = initial_abundances.cpu().numpy()
            print(f"✓ Using abundances: {abundances_np}")

    if initial_raman_spectrum is not None:
        if isinstance(initial_raman_spectrum, np.ndarray):
            initial_raman_spectrum = torch.tensor(
                initial_raman_spectrum, dtype=torch.float32, device=device
            )
        elif isinstance(initial_raman_spectrum, torch.Tensor):
            initial_raman_spectrum = initial_raman_spectrum.to(device)
        else:
            raise TypeError(
                f"initial_raman_spectrum must be numpy array or torch.Tensor, got {type(initial_raman_spectrum)}"
            )

        if verbose:
            raman_np = initial_raman_spectrum.cpu().numpy()
            print(
                f"✓ Using Raman spectrum: min={raman_np.min():.3f}, max={raman_np.max():.3f}, mean={raman_np.mean():.3f}"
            )

    # Validate consistency of n_fluorophores across all initial parameters
    n_fluor_from_params = []
    if initial_fluorophore_bases is not None:
        n_fluor_from_params.append(
            ("initial_fluorophore_bases", initial_fluorophore_bases.shape[0])
        )
    if initial_decay_rates is not None:
        n_fluor_from_params.append(
            ("initial_decay_rates", initial_decay_rates.shape[0])
        )
    if initial_abundances is not None:
        n_fluor_from_params.append(("initial_abundances", initial_abundances.shape[0]))

    if len(n_fluor_from_params) > 0:
        # Check all initial parameters have consistent n_fluorophores
        n_fluor_values = [n for _, n in n_fluor_from_params]
        if len(set(n_fluor_values)) > 1:
            param_info = ", ".join([f"{name}={n}" for name, n in n_fluor_from_params])
            raise ValueError(
                f"Inconsistent number of fluorophores in initial parameters: {param_info}. "
                f"All initial parameters must have the same number of fluorophores."
            )

        # Warn if n_fluorophores parameter doesn't match initial parameters
        if n_fluorophores != n_fluor_values[0]:
            print(
                f"⚠️  WARNING: n_fluorophores parameter ({n_fluorophores}) differs from initial parameters ({n_fluor_values[0]})"
            )
            print(
                f"   Using n_fluorophores={n_fluor_values[0]} from initial parameters"
            )
            n_fluorophores = n_fluor_values[0]

    model = PhysicsDecomposition(
        n_wavenumbers=n_wavenumbers,
        n_timepoints=n_timepoints,
        n_fluorophores=n_fluorophores,
        time_values=time_tensor,
        initial_abundances=initial_abundances,
        initial_decay_rates=initial_decay_rates,
        initial_fluorophore_bases=initial_fluorophore_bases,
        initial_raman_spectrum=initial_raman_spectrum,
        device=device,
        basis_type=basis_type,
        polynomial_degree=polynomial_degree,
        wavenumber_axis=wn_tensor,
        min_decay_rate=min_decay_rate,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("MODEL INITIALIZED")
        print("=" * 70)
        print(f"n_fluorophores: {n_fluorophores}")
        print(f"n_wavenumbers: {n_wavenumbers}")
        print(f"n_timepoints: {n_timepoints}")
        print(f"basis_type: {basis_type}")
        print(f"min_decay_rate: {min_decay_rate}")
        init_tau = (1.0 / model.decay_rates).detach().cpu().numpy()
        init_abundances = model.abundances.detach().cpu().numpy()
        print(f"\nInitialized decay rates τ: {init_tau}")
        print(f"Initialized abundances: {init_abundances}")
        print("=" * 70 + "\n")

    if freeze_bases:
        # Freeze Polynomial Coefficients
        if hasattr(model, "poly_coeffs"):
            model.poly_coeffs.requires_grad = False
        # Freeze Raw Bases (if using free mode)
        if hasattr(model, "fluorophore_bases_raw"):
            model.fluorophore_bases_raw.requires_grad = False
        if verbose:
            print("Fluorophore bases frozen (fixed to initialization)")
    if initial_decay_rates is not None:
        model.decay_rates_raw.requires_grad = False
        print("Decay rates frozen (fixed to initialization)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    history = {
        "loss": [],
        "mse": [],
    }

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        reconstruction = model()
        residuals_squared = (
            reconstruction[:first_times] - data_tensor[:first_times]
        ) ** 2
        mse_loss = torch.mean(residuals_squared)
        
        history["mse"].append(mse_loss.item())
     
        total_loss = mse_loss 

        total_loss.backward()
        optimizer.step()

        history["loss"].append(total_loss.item())
        history["mse"].append(mse_loss.item())

        if verbose and (epoch + 1) % 1000 == 0:
            tau = (1.0 / model.decay_rates).detach().cpu().numpy()
            abundances = model.abundances.detach().cpu().numpy()
            # print all history for current epoch
            print(f"Epoch {epoch + 1}/{n_epochs}:")
            for key in history:
                print(f"  {key}: {history[key][-1]:.6f}")
            print(f"  Current τ: {tau}")
            print(f"  Current A: {abundances}")
            
    if normalize_data and scale_factor != 1.0:
        with torch.no_grad():
            # Since params are stored as log(value), we add log(scale)
            # log(value * scale) = log(value) + log(scale)
            log_scale = np.log(scale_factor)
            
            # Rescale Raman Spectrum
            model.raman_spectrum.add_(log_scale)
            
            # Rescale Abundances
            model.abundances_raw.add_(log_scale)
            
            if verbose:
                print(f"✓ Model parameters rescaled back to original units (x{scale_factor:.2f})")

    return model, history


def visualise_decomposition(
    model: PhysicsDecomposition,
    dataset: xr.Dataset,
    figsize: Tuple[int, int] = (16, 10),
):
    """Visualize decomposition results."""

    decomp = model.get_decomposition()
    reconstruction = model().detach().cpu().numpy()
    n_t = dataset.bleaching_time.values.shape[0]
    n_wavenumbers = dataset.wavenumber.values.shape[0]
    time_axis = dataset.bleaching_time.values
    wavenumbers = dataset.wavenumber.values
    original_data = dataset.intensity_clean.values
    true_raman = dataset.raman_gt.values

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Plot 1: Original time series
    ax = axes[0, 0]
    n_show = min(8, n_t)
    cmap = plt.cm.viridis
    for i, idx in enumerate(np.linspace(0, n_t - 1, n_show, dtype=int)):
        ax.plot(wavenumbers, original_data[idx], color=cmap(i / n_show), alpha=0.7)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Original Time Series")
    ax.grid(True, alpha=0.3)

    if true_raman is not None:
        reference_raman = true_raman
    else:
        print("No true Raman provided, using last 20 frames average as reference.")
        reference_raman = original_data[-20:].mean(axis=0)
    # Plot 2: Extracted Raman vs reference
    ax = axes[0, 1]
    ax.plot(wavenumbers, decomp["raman"], "b-", linewidth=2, label="Extracted Raman ")
    ax.plot(
        wavenumbers,
        reference_raman,
        "r--",
        alpha=0.7,
        label=f"Raman GT {'(Last 20 frames avg)' if true_raman is None else ''}",
    )
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Extracted Raman Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Fluorophore bases
    ax = axes[0, 2]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i in range(model.n_fluorophores):
        tau = decomp["time_constants"][i]
        ax.plot(
            wavenumbers,
            decomp["fluorophore_bases"][i],
            color=colors[i % len(colors)],
            label=f"B{i + 1} (τ={tau:.3f}s)",
        )
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title("Fluorophore Basis Spectra")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Decay components over time
    ax = axes[1, 0]
    total_fluor = np.zeros(n_t)
    for i in range(model.n_fluorophores):
        component = model.get_fluorescence_component(i)
        amplitude = component.mean(axis=1)
        total_fluor += amplitude
        tau = decomp["time_constants"][i]
        ax.plot(
            time_axis,
            amplitude,
            colors[i % len(colors)],
            label=f"τ={tau:.3f}s, w={decomp['abundances'][i]:.1f}",
        )

    # Plot ground truth fluorescence if available
    if true_raman is not None:
        gt_fluorescence = (original_data - true_raman[np.newaxis, :]).mean(axis=1)
        ax.plot(
            time_axis,
            gt_fluorescence,
            "k-",
            alpha=0.5,
            label="GT Fluorescence Mean",
        )

    ax.plot(time_axis, total_fluor, "k--", linewidth=2, label="Total Predicted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Fluorescence")
    ax.set_title("Decay Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Reconstruction quality (first frame)
    ax = axes[1, 1]
    ax.plot(wavenumbers, original_data[0], "b-", alpha=0.7, label="Original (t=0)")
    ax.plot(wavenumbers, reconstruction[0], "r--", alpha=0.7, label="Reconstructed")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Reconstruction (First Frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Residual over time
    ax = axes[1, 2]
    residuals = original_data - reconstruction
    ax.plot(time_axis, np.mean(residuals, axis=1), "k-", label="Mean")
    ax.fill_between(
        time_axis,
        np.mean(residuals, axis=1) - np.std(residuals, axis=1),
        np.mean(residuals, axis=1) + np.std(residuals, axis=1),
        alpha=0.3,
    )
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    mse = np.mean(residuals**2)
    r2 = 1 - np.sum(residuals**2) / np.sum((original_data - original_data.mean()) ** 2)
    print(f"\nReconstruction MSE: {mse:.6f}, R²: {r2:.6f}")
    print(f"Time constants (τ): {decomp['time_constants']}")
    print(f"Abundances (w): {decomp['abundances']}")

    return fig, axes


def visualise_decomposition_3d(
    model: PhysicsDecomposition,
    original_data: np.ndarray,
    wavenumbers: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    subsample_wn: int = 2,
    subsample_time: int = 1,
):
    """
    Interactive 3D visualisation using plotly (allows rotation/zoom).

    Args:
        model: Fitted PhysicsDecomposition model
        original_data: Original time series (n_timepoints, n_wavenumbers)
        wavenumbers: Wavenumber axis array
        time_values: Time axis array. If None, uses frame indices.
        subsample_wn: Subsample factor for wavenumber axis
        subsample_time: Subsample factor for time axis
    """

    decomp = model.get_decomposition()
    reconstruction = model().detach().cpu().numpy()
    n_t, n_wn = original_data.shape
    time_axis = time_values if time_values is not None else np.arange(n_t)

    # Subsample for performance
    wn_idx = np.arange(0, n_wn, subsample_wn)
    t_idx = np.arange(0, n_t, subsample_time)

    wn_sub = wavenumbers[wn_idx]
    t_sub = time_axis[t_idx]

    # Create figure with dropdown to select which surface to view
    fig = go.Figure()

    # Original data
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=original_data[np.ix_(t_idx, wn_idx)],
            colorscale="Viridis",
            name="Original",
            visible=True,
            colorbar=dict(title="Intensity", x=1.02),
        )
    )

    # Reconstruction
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=reconstruction[np.ix_(t_idx, wn_idx)],
            colorscale="Viridis",
            name="Reconstructed",
            visible=False,
        )
    )

    # Residual
    residual = original_data - reconstruction
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=residual[np.ix_(t_idx, wn_idx)],
            colorscale="RdBu",
            name="Residual",
            visible=False,
            cmid=0,  # Center colorscale at zero
        )
    )

    # Total fluorescence
    total_fluor = np.zeros_like(original_data)
    for i in range(model.n_fluorophores):
        total_fluor += model.get_fluorescence_component(i)
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=total_fluor[np.ix_(t_idx, wn_idx)],
            colorscale="Oranges",
            name="Fluorescence",
            visible=False,
        )
    )

    # Raman (constant surface)
    raman_surface = np.tile(decomp["raman"][wn_idx], (len(t_idx), 1))
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=raman_surface,
            colorscale="Blues",
            name="Predicted Raman",
            visible=False,
        )
    )

    # Create dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(
                        label="Original",
                        method="update",
                        args=[{"visible": [True, False, False, False, False]}],
                    ),
                    dict(
                        label="Reconstructed",
                        method="update",
                        args=[{"visible": [False, True, False, False, False]}],
                    ),
                    dict(
                        label="Residual",
                        method="update",
                        args=[{"visible": [False, False, True, False, False]}],
                    ),
                    dict(
                        label="Fluorescence",
                        method="update",
                        args=[{"visible": [False, False, False, True, False]}],
                    ),
                    dict(
                        label="Predicted Raman",
                        method="update",
                        args=[{"visible": [False, False, False, False, True]}],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
            )
        ],
        scene=dict(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Time (s)",
            zaxis_title="Intensity",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        title="3D Spectral Decomposition (use dropdown to switch views)",
        width=900,
        height=700,
    )

    return fig
