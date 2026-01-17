"""
Decomposition methods for Raman/fluorescence separation.

Implements:
- Differential Evolution (DE) for global rate optimization
- Analytical NNLS for spectra given fixed rates
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import lstsq


@dataclass
class DecompositionResult:
    """Container for decomposition results."""
    raman: np.ndarray              # (n_wavenumbers,)
    rates: np.ndarray              # (n_fluorophores,)
    fluorophore_spectra: np.ndarray  # (n_fluorophores, n_wavenumbers)
    mse: float = 0.0
    
    @property
    def time_constants(self) -> np.ndarray:
        """Time constants τ = 1/λ."""
        return 1.0 / self.rates
    
    def reconstruction(self, time_points: np.ndarray) -> np.ndarray:
        """Reconstruct Y(t, ν) from decomposition parameters."""
        decay = np.exp(-self.rates[:, None] * time_points[None, :])  # (K, T)
        Y = self.raman[None, :] + decay.T @ self.fluorophore_spectra  # (T, W)
        return Y
    
    def to_dict(self) -> dict:
        """Convert to dictionary for visualization functions."""
        return {
            "raman": self.raman,
            "rates": self.rates,
            "decay_rates": self.rates,
            "fluorophore_bases": self.fluorophore_spectra,
            "bases": self.fluorophore_spectra,
            "abundances": np.ones(len(self.rates)),  # Absorbed into spectra
            "time_constants": self.time_constants,
            "mse": self.mse,
        }


def solve_spectra_given_rates(
    data: np.ndarray,
    time_values: np.ndarray,
    decay_rates: np.ndarray,
    non_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
    n_timepoints, n_wavenumbers = data.shape
    n_components = len(decay_rates)
    
    # Design matrix: [1, exp(-λ₁t), exp(-λ₂t), ...]
    X = np.ones((n_timepoints, 1 + n_components))
    for i, rate in enumerate(decay_rates):
        X[:, i + 1] = np.exp(-rate * time_values)
    
    # Solve Y = X @ beta
    beta, _, _, _ = lstsq(X, data, lapack_driver='gelsd')
    
    raman = beta[0, :]
    fluorescence = beta[1:, :]
    
    if non_negative:
        raman = np.maximum(raman, 0)
        fluorescence = np.maximum(fluorescence, 0)
    
    return raman, fluorescence


def decompose(
    Y: np.ndarray,
    time_points: np.ndarray,
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
    Y : np.ndarray
        Intensity array, shape (n_timepoints, n_wavenumbers)
    time_points : np.ndarray
        Time values, shape (n_timepoints,)
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
    
    Notes
    -----
    Rate bounds of (0.01, 20) correspond to:
        τ_max = 1/0.01 = 100s (very slow decay)
        τ_min = 1/20 = 0.05s (fast decay)
    """
    T, W = Y.shape
    K = n_fluorophores
    
    def solve_given_rates(rates):
        raman, fluor = solve_spectra_given_rates(Y, time_points, rates)
        decay = np.exp(-rates[:, None] * time_points[None, :])
        Y_recon = raman[None, :] + decay.T @ fluor
        mse = np.mean((Y - Y_recon) ** 2)
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
        updating='deferred',
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
