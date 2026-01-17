"""
Decomposition methods for Raman/fluorescence separation.

Implements:
- Differential Evolution (DE) for global rate optimization
- Analytical NNLS for spectra given fixed rates
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.optimize import differential_evolution, curve_fit
from scipy.linalg import lstsq

from ramanlib.core import SpectralData


@dataclass
class DecompositionResult:
    """Container for decomposition results."""

    raman: SpectralData  # (n_wavenumbers,)
    rates: np.ndarray  # (n_fluorophores,)
    fluorophore_spectra: SpectralData  # (n_fluorophores, n_wavenumbers)
    mse: float = 0.0

    @property
    def time_constants(self) -> np.ndarray:
        """Time constants τ = 1/λ."""
        return 1.0 / self.rates

    def reconstruction(self, time_points: np.ndarray) -> np.ndarray:
        """Reconstruct Y(t, ν) from decomposition parameters."""
        decay = np.exp(-self.rates[:, None] * time_points[None, :])  # (K, T)
        Y = (
            self.raman.intensities[None, :]
            + decay.T @ self.fluorophore_spectra.intensities
        )  # (T, W)
        return Y

    def to_dict(self) -> dict:
        """
        Convert to dictionary for visualization functions.

        Returns SpectralData objects which contain both intensities and wavenumbers.
        Visualization functions will extract what they need.
        """
        return {
            "raman": self.raman,  # SpectralData
            "rates": self.rates,
            "decay_rates": self.rates,
            "fluorophore_bases": self.fluorophore_spectra,  # SpectralData
            "bases": self.fluorophore_spectra,  # SpectralData
            "abundances": np.ones(len(self.rates)),  # Absorbed into spectra
            "time_constants": self.time_constants,
            "mse": self.mse,
        }


def solve_spectra_given_rates(
    data: SpectralData,
    time_values: np.ndarray,
    decay_rates: np.ndarray,
    non_negative: bool = True,
) -> Tuple[SpectralData, SpectralData]:
    """
    Solve for Raman and fluorescence spectra given fixed decay rates.

    Given known decay rates λₖ, the model:
        Y(ν, t) = s(ν) + Σₖ Cₖ(ν) · exp(-λₖ · t)

    is LINEAR in [s(ν), C₁(ν), ..., Cₖ(ν)].

    For each wavenumber: y = X @ β, solved via least squares.

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (n_timepoints, n_wavenumbers)
    time_values : np.ndarray
        Time points, shape (n_timepoints,)
    decay_rates : np.ndarray
        Decay rates λₖ, shape (n_components,)
    non_negative : bool
        Clip negative values to zero

    Returns
    -------
    raman : np.ndarray
        Estimated Raman spectrum s(ν), shape (n_wavenumbers,)
    fluorescence : np.ndarray
        Fluorescence contributions Cₖ(ν), shape (n_components, n_wavenumbers)
    """
    n_timepoints, n_wavenumbers = data.intensities.shape
    n_components = len(decay_rates)

    # Design matrix: [1, exp(-λ₁t), exp(-λ₂t), ...]
    X = np.ones((n_timepoints, 1 + n_components))
    for i, rate in enumerate(decay_rates):
        X[:, i + 1] = np.exp(-rate * time_values)

    # Solve Y = X @ beta
    beta, _, _, _ = lstsq(X, data.intensities, lapack_driver="gelsd")

    raman = SpectralData(beta[0, :], data.wavenumbers)
    fluorescence = SpectralData(beta[1:, :], data.wavenumbers)

    if non_negative:
        raman = SpectralData(np.maximum(raman.intensities, 0), data.wavenumbers)
        fluorescence = SpectralData(np.maximum(fluorescence.intensities, 0), data.wavenumbers)

    return raman, fluorescence


def decompose(
    data,  # Union[np.ndarray, SpectralData]
    time_points: Optional[np.ndarray] = None,
    wavenumbers: Optional[np.ndarray] = None,
    n_fluorophores: int = 2,
    rate_bounds: tuple = (0.01, 20),
    maxiter: int = 100,
    seed: int = 42,
    polish: bool = True,
    verbose: bool = False,
) -> DecompositionResult:
    """
    Decompose Y(t, ν) into Raman + fluorescence with exponential decay.

    Uses Differential Evolution for global rate optimization, then
    solves analytically for spectra given those rates.

    Parameters
    ----------
    data : np.ndarray or SpectralData
        If np.ndarray: Intensity array, shape (n_timepoints, n_wavenumbers)
                      Requires time_points parameter
        If SpectralData: Must have time_values set (is_time_series=True)
    time_points : np.ndarray, optional
        Time values, shape (n_timepoints,)
        Required if data is np.ndarray, ignored if data is SpectralData
    wavenumbers : np.ndarray, optional
        Wavenumber axis. Extracted from data if SpectralData
    n_fluorophores : int
        Number of decay components K
    rate_bounds : tuple
        (min, max) bounds for decay rates λ
    maxiter : int
        Maximum DE iterations
    seed : int
        Random seed for reproducibility
    polish : bool
        Apply L-BFGS-B refinement after DE
    verbose : bool
        Print optimization progress

    Returns
    -------
    DecompositionResult
        Contains raman, rates, fluorophore_spectra, mse

    Examples
    --------
    # Old way (still works)
    >>> result = decompose(Y_array, time_points=t_array, wavenumbers=wn_array)

    # New way (cleaner)
    >>> data = SpectralData(Y_array, wn_array, time_values=t_array)
    >>> result = decompose(data[:40])  # First 40 frames

    Notes
    -----
    Rate bounds of (0.01, 20) correspond to:
        τ_max = 1/0.01 = 100s (very slow decay)
        τ_min = 1/20 = 0.05s (fast decay)
    """
    # Extract arrays from SpectralData if provided
    if isinstance(data, SpectralData):
        if not data.is_time_series:
            raise ValueError(
                "SpectralData must have time_values for decomposition. "
                "Create with: SpectralData(Y, wn, time_values=t)"
            )
        spectral_data = data
        t = data.time_values
    else:
        # Legacy numpy array input
        if time_points is None:
            raise ValueError(
                "time_points required when data is np.ndarray. "
                "Use decompose(data, time_points=t) or "
                "decompose(SpectralData(data, wn, time_values=t))"
            )
        if wavenumbers is None:
            wavenumbers = np.arange(data.shape[1])

        spectral_data = SpectralData(
            intensities=data,
            wavenumbers=wavenumbers,
            time_values=time_points
        )
        t = time_points

    T, W = spectral_data.intensities.shape
    K = n_fluorophores

    def solve_given_rates(rates):
        raman, fluor = solve_spectra_given_rates(spectral_data, t, rates)
        decay = np.exp(-rates[:, None] * t[None, :])
        Y_recon = raman.intensities[None, :] + decay.T @ fluor.intensities
        mse = np.mean((spectral_data.intensities - Y_recon) ** 2)
        return raman, fluor, mse

    def objective(rates):
        _, _, mse = solve_given_rates(rates)
        return mse

    result = differential_evolution(
        objective,
        bounds=[rate_bounds] * K,
        maxiter=maxiter,
        seed=seed,
        polish=polish,
        updating="deferred",
        workers=1,
        disp=verbose,
    )

    best_rates = np.clip(result.x, rate_bounds[0], rate_bounds[1])
    raman, fluor, mse = solve_given_rates(best_rates)

    return DecompositionResult(
        raman=raman,
        rates=best_rates,
        fluorophore_spectra=fluor,
        mse=float(mse),
    )


def decompose_with_known_rates(
    Y: np.ndarray,
    time_points: np.ndarray,
    rates: np.ndarray,
) -> DecompositionResult:
    """
    Decompose with known decay rates (analytical solution only).

    Useful when rates are known from prior analysis or ground truth.

    Parameters
    ----------
    Y : np.ndarray
        Intensity array, shape (n_timepoints, n_wavenumbers)
    time_points : np.ndarray
        Time values, shape (n_timepoints,)
    rates : np.ndarray
        Known decay rates, shape (n_fluorophores,)

    Returns
    -------
    DecompositionResult
    """
    raman, fluor = solve_spectra_given_rates(Y, time_points, rates)
    decay = np.exp(-rates[:, None] * time_points[None, :])
    Y_recon = raman[None, :] + decay.T @ fluor
    mse = np.mean((Y - Y_recon) ** 2)

    return DecompositionResult(
        raman=raman,
        rates=rates,
        fluorophore_spectra=fluor,
        mse=mse,
    )


def estimate_decay_rates_from_early_frames(
    data: np.ndarray,
    time_values: np.ndarray,
    first_times: int,
    n_components: int = 3,
) -> Tuple[np.ndarray, Dict]:
    """
    Estimate decay rates by fitting exponentials to wavenumber-averaged signal.

    Uses only the first `first_times` frames to avoid information leakage.
    This provides a physics-informed prior for the neural network optimization.

    Parameters
    ----------
    data : np.ndarray
        Full time series, shape (n_timepoints, n_wavenumbers)
    time_values : np.ndarray
        Time axis in seconds
    first_times : int
        Number of early frames to use (training window)
    n_components : int
        Number of exponential components to fit

    Returns
    -------
    decay_rates : np.ndarray
        Estimated decay rates λ, shape (n_components,)
    fit_info : dict
        Dictionary with fit quality metrics and parameters
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
