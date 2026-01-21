"""
Core physics utilities for bleaching spectral decomposition.

Physics model:
    Y(ν, t) = s(ν) + Σₖ wₖ · Bₖ(ν) · exp(-λₖ · t)

This module provides functions shared by:
- decompose.py (DE-based decomposition)
- models.py (NN-based decomposition)  
- generate.py (synthetic data generation)
"""

from typing import Tuple, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Wavenumber Normalization
# =============================================================================

def normalize_wavenumbers(
    wavenumbers: np.ndarray,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize wavenumbers to [-1, 1] range for numerical stability.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    wn_min, wn_max : float, optional
        Normalization bounds. If None, uses data min/max.

    Returns
    -------
    wn_norm : np.ndarray
        Normalized wavenumbers in [-1, 1]
    wn_min, wn_max : float
        Bounds used for normalization
    """
    if wn_min is None:
        wn_min = float(wavenumbers.min())
    if wn_max is None:
        wn_max = float(wavenumbers.max())

    wn_norm = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
    return wn_norm, wn_min, wn_max


# =============================================================================
# Vandermonde Matrix
# =============================================================================

def build_vandermonde(wn_norm: np.ndarray, degree: int) -> np.ndarray:
    """
    Build Vandermonde matrix for polynomial evaluation.

    Columns are [1, x, x², ..., xⁿ] (ascending powers).

    Parameters
    ----------
    wn_norm : np.ndarray
        Normalized wavenumbers in [-1, 1], shape (n_wavenumbers,)
    degree : int
        Polynomial degree

    Returns
    -------
    vandermonde : np.ndarray
        Shape (n_wavenumbers, degree+1)
    """
    n_coeffs = degree + 1
    return np.stack([wn_norm**k for k in range(n_coeffs)], axis=1)


# =============================================================================
# L2 Normalization
# =============================================================================

def l2_normalize(
    spectra: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """L2-normalize spectra along specified axis."""
    norm = np.linalg.norm(spectra, axis=axis, keepdims=True)
    return spectra / (norm + eps)


# =============================================================================
# Forward Model
# =============================================================================

def reconstruct_time_series(
    raman: np.ndarray,
    bases: np.ndarray,
    abundances: np.ndarray,
    decay_rates: np.ndarray,
    time_values: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct bleaching time series from physics parameters.

    Y(ν, t) = s(ν) + Σₖ wₖ · Bₖ(ν) · exp(-λₖ · t)

    Parameters
    ----------
    raman : np.ndarray
        Raman spectrum s(ν), shape (n_wavenumbers,)
    bases : np.ndarray
        Fluorophore bases Bₖ(ν), shape (n_fluorophores, n_wavenumbers)
    abundances : np.ndarray
        Abundances wₖ, shape (n_fluorophores,)
    decay_rates : np.ndarray
        Decay rates λₖ (s⁻¹), shape (n_fluorophores,)
    time_values : np.ndarray
        Time points t (s), shape (n_timepoints,)

    Returns
    -------
    reconstruction : np.ndarray
        Shape (n_timepoints, n_wavenumbers)
    """
    print("Shapes:", raman.shape, bases.shape, abundances.shape, decay_rates.shape, time_values.shape)
    decay_factors = np.exp(-decay_rates[np.newaxis, :] * time_values[:, np.newaxis])
    # print("decay_factors.shape:", decay_factors.shape)
    weighted_bases = abundances[:, np.newaxis] * bases # 3,1 * 3,630
    fluorescence = np.matmul(decay_factors, weighted_bases)
    return raman + fluorescence



# =============================================================================
# Polynomial Fitting
# =============================================================================

def fit_polynomial_bases(
    bases: np.ndarray,
    wavenumbers: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit polynomial coefficients to fluorophore bases in log-space.

    B(ν) = exp(Σₖ cₖ · νₙₒᵣₘᵏ)

    Parameters
    ----------
    bases : np.ndarray
        Fluorophore bases, shape (n_fluorophores, n_wavenumbers)
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    degree : int
        Polynomial degree

    Returns
    -------
    poly_coeffs : np.ndarray
        Shape (n_fluorophores, degree+1), ascending power order
    wn_mean : float
        Wavenumber mean used for normalization
    wn_std : float
        Wavenumber std used for normalization
    """
    n_fluorophores = bases.shape[0]
    n_coeffs = degree + 1

    # Compute normalization stats (z-score)
    wn_mean = float(wavenumbers.mean())
    wn_std = float(wavenumbers.std())
    wn_normalized = (wavenumbers - wn_mean) / (wn_std + 1e-8)

    log_poly_coeffs = np.zeros((n_fluorophores, n_coeffs))
    for i in range(n_fluorophores):
        log_basis = np.log(bases[i] + 1e-8)
        # np.polyfit returns descending order, reverse to ascending
        log_coeffs = np.polyfit(wn_normalized, log_basis, deg=degree)
        log_poly_coeffs[i] = log_coeffs[::-1]

    return log_poly_coeffs, wn_mean, wn_std


def evaluate_polynomial_bases(
    log_poly_coeffs: np.ndarray,
    wavenumbers: np.ndarray,
    wn_mean: float,
    wn_std: float,
) -> np.ndarray:
    """
    Evaluate polynomial fluorophore bases in log-space, then exponentiate.

    B(ν) = exp(Σₖ cₖ · νₙₒᵣₘᵏ)

    Parameters
    ----------
    log_poly_coeffs : np.ndarray
        Shape (n_fluorophores, degree+1), ascending power order
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    wn_mean : float
        Wavenumber mean for normalization (from fitting)
    wn_std : float
        Wavenumber std for normalization (from fitting)

    Returns
    -------
    bases : np.ndarray
        Shape (n_fluorophores, n_wavenumbers)
    """
    # print(f"log_poly_coeffs.shape: {log_poly_coeffs.shape}")
    if log_poly_coeffs.ndim == 1:
        log_poly_coeffs = log_poly_coeffs[None, :]
    degree = log_poly_coeffs.shape[1] - 1

    # Normalize wavenumbers using provided stats
    wn_normalized = (wavenumbers - wn_mean) / (wn_std + 1e-8)
    vandermonde = np.vander(wn_normalized, N=degree + 1, increasing=True)

    log_intensity_values = log_poly_coeffs @ vandermonde.T
    bases = np.exp(log_intensity_values)

    return bases




# =============================================================================
# PyTorch Versions (for NN models)
# =============================================================================

if TORCH_AVAILABLE:
    def normalize_wavenumbers_torch(
        wavenumbers: "torch.Tensor",
        wn_min: Optional["torch.Tensor"] = None,
        wn_max: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Normalize wavenumbers to [-1, 1] (PyTorch version)."""
        if wn_min is None:
            wn_min = wavenumbers.min()
        if wn_max is None:
            wn_max = wavenumbers.max()

        wn_norm = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
        return wn_norm, wn_min, wn_max
    
    def evaluate_polynomial_bases_torch(
        log_poly_coeffs: "torch.Tensor",
        wavenumbers: "torch.Tensor",
        wn_mean: Optional[float] = None,
        wn_std: Optional[float] = None,
    ) -> "torch.Tensor":
        """
        Evaluate polynomial fluorophore bases in log-space, then exponentiate.
        B(ν) = exp(Σₖ cₖ · νₙₒᵣₘᵏ)

        Parameters
        ----------
        log_poly_coeffs : torch.Tensor
            Shape (n_fluorophores, degree+1), ascending power order
        wavenumbers : torch.Tensor
            Wavenumber axis (cm⁻¹)
        wn_mean : float, optional
            Wavenumber mean for normalization (from fitting)
            If None, computes from wavenumbers
        wn_std : float, optional
            Wavenumber std for normalization (from fitting)
            If None, computes from wavenumbers

        Returns
        -------
        bases : torch.Tensor
            Shape (n_fluorophores, n_wavenumbers)
        """
        degree = log_poly_coeffs.shape[1] - 1

        # Normalize wavenumbers using provided stats (or compute if not provided)
        if wn_mean is None:
            wn_mean = float(wavenumbers.mean().item())
        if wn_std is None:
            wn_std = float(wavenumbers.std().item())

        wn_normalized = (wavenumbers - wn_mean) / (wn_std + 1e-8)
        vandermonde = torch.vander(wn_normalized, N=degree + 1, increasing=True)

        if log_poly_coeffs.dtype != vandermonde.dtype:
            vandermonde = vandermonde.to(log_poly_coeffs.dtype)
        log_intensity_values = torch.matmul(log_poly_coeffs, vandermonde.T)
        bases = torch.exp(log_intensity_values)

        return bases



    def build_vandermonde_torch(
        wn_norm: "torch.Tensor",
        degree: int,
    ) -> "torch.Tensor":
        """Build Vandermonde matrix (PyTorch version)."""
        n_coeffs = degree + 1
        return torch.stack([wn_norm**k for k in range(n_coeffs)], dim=1)

    def l2_normalize_torch(
        spectra: "torch.Tensor",
        dim: int = -1,
        eps: float = 1e-8,
    ) -> "torch.Tensor":
        """L2-normalize spectra (PyTorch version)."""
        norm = torch.norm(spectra, p=2, dim=dim, keepdim=True)
        return spectra / (norm + eps)

    def reconstruct_time_series_torch(
        raman: "torch.Tensor",
        bases: "torch.Tensor",
        abundances: "torch.Tensor",
        decay_rates: "torch.Tensor",
        time_values: "torch.Tensor",
    ) -> "torch.Tensor":
        """Reconstruct bleaching time series (PyTorch version)."""
        raman_expanded = raman.unsqueeze(0)
        decay_factors = torch.exp(-decay_rates.unsqueeze(0) * time_values.unsqueeze(1))
        weighted_bases = abundances.unsqueeze(1) * bases
        fluorescence = torch.matmul(decay_factors, weighted_bases)
        return raman_expanded + fluorescence


