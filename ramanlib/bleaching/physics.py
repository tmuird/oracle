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
    n_t = len(time_values)
    n_fluorophores = len(decay_rates)

    result = np.tile(raman, (n_t, 1))

    for i in range(n_fluorophores):
        decay = np.exp(-decay_rates[i] * time_values)
        fluor_i = abundances[i] * decay[:, None] * bases[i, None, :]
        result = result + fluor_i

    return result


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
    wn_min, wn_max : float
        Normalization bounds
    """
    n_fluorophores = bases.shape[0]
    n_coeffs = degree + 1

    wn_norm, wn_min, wn_max = normalize_wavenumbers(wavenumbers)

    poly_coeffs = np.zeros((n_fluorophores, n_coeffs))

    for i in range(n_fluorophores):
        log_basis = np.log(bases[i] + 1e-8)
        # np.polyfit returns descending order, reverse to ascending
        coeffs = np.polyfit(wn_norm, log_basis, deg=degree)
        poly_coeffs[i] = coeffs[::-1]

    return poly_coeffs, wn_min, wn_max


def evaluate_polynomial_bases(
    poly_coeffs: np.ndarray,
    wavenumbers: np.ndarray,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Evaluate polynomial fluorophore bases.

    B(ν) = exp(Σₖ cₖ · νₙₒᵣₘᵏ)

    Parameters
    ----------
    poly_coeffs : np.ndarray
        Shape (n_fluorophores, degree+1), ascending power order
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    wn_min, wn_max : float, optional
        Normalization bounds
    normalize : bool
        L2-normalize each basis

    Returns
    -------
    bases : np.ndarray
        Shape (n_fluorophores, n_wavenumbers)
    """
    degree = poly_coeffs.shape[1] - 1

    wn_norm, wn_min, wn_max = normalize_wavenumbers(wavenumbers, wn_min, wn_max)
    vandermonde = build_vandermonde(wn_norm, degree)

    poly_values = poly_coeffs @ vandermonde.T
    bases = np.exp(poly_values)

    if normalize:
        bases = l2_normalize(bases, axis=1)

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
