"""
Core physics utilities for Raman/fluorescence spectral decomposition.

This module provides shared functions used by both:
- flogen.py (synthetic data generation)
- decomposition.py (physics-constrained model fitting)

The physics model is:
    Y(ν, t) = s(ν) + Σᵢ wᵢ · Bᵢ(ν) · exp(-λᵢ · t)

where:
    - s(ν): Raman spectrum (time-invariant)
    - Bᵢ(ν): Fluorophore basis spectra (L2-normalized, non-negative)
    - wᵢ: Abundance/amplitude of each fluorophore
    - λᵢ: Decay rate of each fluorophore (s⁻¹)
    - t: Cumulative laser exposure time (s)

Fluorophore bases can be parameterized as polynomials in log-space:
    Bᵢ(ν) = exp(Σₖ cᵢₖ · νₙₒᵣₘᵏ) / ||·||₂
"""

from typing import Tuple, Optional
import numpy as np

# Optional torch import for GPU-accelerated versions
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
    wn_min : float, optional
        Minimum wavenumber for normalization. If None, uses wavenumbers.min()
    wn_max : float, optional
        Maximum wavenumber for normalization. If None, uses wavenumbers.max()

    Returns
    -------
    wn_norm : np.ndarray
        Normalized wavenumbers in [-1, 1]
    wn_min : float
        Minimum wavenumber used for normalization
    wn_max : float
        Maximum wavenumber used for normalization
    """
    if wn_min is None:
        wn_min = wavenumbers.min()
    if wn_max is None:
        wn_max = wavenumbers.max()

    wn_norm = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
    return wn_norm, wn_min, wn_max


if TORCH_AVAILABLE:
    def normalize_wavenumbers_torch(
        wavenumbers: "torch.Tensor",
        wn_min: Optional["torch.Tensor"] = None,
        wn_max: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Normalize wavenumbers to [-1, 1] range (PyTorch version).

        Parameters
        ----------
        wavenumbers : torch.Tensor
            Wavenumber axis (cm⁻¹)
        wn_min : torch.Tensor, optional
            Minimum wavenumber for normalization. If None, uses wavenumbers.min()
        wn_max : torch.Tensor, optional
            Maximum wavenumber for normalization. If None, uses wavenumbers.max()

        Returns
        -------
        wn_norm : torch.Tensor
            Normalized wavenumbers in [-1, 1]
        wn_min : torch.Tensor
            Minimum wavenumber used for normalization
        wn_max : torch.Tensor
            Maximum wavenumber used for normalization
        """
        if wn_min is None:
            wn_min = wavenumbers.min()
        if wn_max is None:
            wn_max = wavenumbers.max()

        wn_norm = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
        return wn_norm, wn_min, wn_max


# =============================================================================
# Vandermonde Matrix Construction
# =============================================================================

def build_vandermonde(wn_norm: np.ndarray, degree: int) -> np.ndarray:
    """
    Build Vandermonde matrix for polynomial evaluation.

    The Vandermonde matrix has columns [1, x, x², ..., xⁿ] (ascending powers).

    Parameters
    ----------
    wn_norm : np.ndarray
        Normalized wavenumbers in [-1, 1], shape (n_wavenumbers,)
    degree : int
        Polynomial degree

    Returns
    -------
    vandermonde : np.ndarray
        Vandermonde matrix, shape (n_wavenumbers, degree+1)
    """
    n_coeffs = degree + 1
    return np.stack([wn_norm**k for k in range(n_coeffs)], axis=1)


if TORCH_AVAILABLE:
    def build_vandermonde_torch(
        wn_norm: "torch.Tensor",
        degree: int,
    ) -> "torch.Tensor":
        """
        Build Vandermonde matrix for polynomial evaluation (PyTorch version).

        Parameters
        ----------
        wn_norm : torch.Tensor
            Normalized wavenumbers in [-1, 1], shape (n_wavenumbers,)
        degree : int
            Polynomial degree

        Returns
        -------
        vandermonde : torch.Tensor
            Vandermonde matrix, shape (n_wavenumbers, degree+1)
        """
        n_coeffs = degree + 1
        return torch.stack([wn_norm**k for k in range(n_coeffs)], dim=1)


# =============================================================================
# Polynomial Fitting and Evaluation
# =============================================================================

def fit_polynomial_bases(
    bases: np.ndarray,
    wavenumbers: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit polynomial coefficients to fluorophore bases in log-space.

    The polynomial parameterization models:
        B(ν) = exp(Σₖ cₖ · νₙₒᵣₘᵏ)

    Fitting is done in log-space: log(B) ≈ Σₖ cₖ · νₙₒᵣₘᵏ

    Parameters
    ----------
    bases : np.ndarray
        Fluorophore bases, shape (n_fluorophores, n_wavenumbers)
        Should be positive (will add epsilon if needed)
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    degree : int
        Polynomial degree

    Returns
    -------
    poly_coeffs : np.ndarray
        Polynomial coefficients, shape (n_fluorophores, degree+1)
        Coefficients are in ASCENDING power order: [c₀, c₁, ..., cₙ]
    wn_min : float
        Minimum wavenumber used for normalization
    wn_max : float
        Maximum wavenumber used for normalization
    """
    n_fluorophores = bases.shape[0]
    n_coeffs = degree + 1

    # Normalize wavenumbers
    wn_norm, wn_min, wn_max = normalize_wavenumbers(wavenumbers)

    # Fit polynomial to LOG of each basis
    poly_coeffs = np.zeros((n_fluorophores, n_coeffs))

    for i in range(n_fluorophores):
        log_basis = np.log(bases[i] + 1e-8)
        # np.polyfit returns DESCENDING order [cₙ, ..., c₀]
        coeffs = np.polyfit(wn_norm, log_basis, deg=degree)
        # Reverse to ASCENDING order [c₀, c₁, ..., cₙ]
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

    Computes B(ν) = exp(Σₖ cₖ · νₙₒᵣₘᵏ) and optionally L2-normalizes.

    Parameters
    ----------
    poly_coeffs : np.ndarray
        Polynomial coefficients, shape (n_fluorophores, degree+1)
        Coefficients must be in ASCENDING power order: [c₀, c₁, ..., cₙ]
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    wn_min : float, optional
        Minimum wavenumber for normalization. If None, uses wavenumbers.min()
    wn_max : float, optional
        Maximum wavenumber for normalization. If None, uses wavenumbers.max()
    normalize : bool
        If True, L2-normalize each basis (default: True)

    Returns
    -------
    bases : np.ndarray
        Evaluated fluorophore bases, shape (n_fluorophores, n_wavenumbers)
    """
    degree = poly_coeffs.shape[1] - 1

    # Normalize wavenumbers
    wn_norm, wn_min, wn_max = normalize_wavenumbers(wavenumbers, wn_min, wn_max)

    # Build Vandermonde matrix
    vandermonde = build_vandermonde(wn_norm, degree)

    # Evaluate: poly_coeffs @ vandermonde.T
    # (n_fluorophores, n_coeffs) @ (n_coeffs, n_wavenumbers)
    poly_values = poly_coeffs @ vandermonde.T

    # Apply exp for positivity
    bases = np.exp(poly_values)

    # L2 normalize each basis
    if normalize:
        for i in range(bases.shape[0]):
            l2_norm = np.linalg.norm(bases[i])
            if l2_norm > 0:
                bases[i] = bases[i] / l2_norm

    return bases


# =============================================================================
# L2 Normalization
# =============================================================================

def l2_normalize(
    spectra: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    L2-normalize spectra along specified axis.

    Parameters
    ----------
    spectra : np.ndarray
        Input spectra
    axis : int
        Axis along which to normalize (default: -1, last axis)
    eps : float
        Small value to avoid division by zero

    Returns
    -------
    normalized : np.ndarray
        L2-normalized spectra
    """
    norm = np.linalg.norm(spectra, axis=axis, keepdims=True)
    return spectra / (norm + eps)


if TORCH_AVAILABLE:
    def l2_normalize_torch(
        spectra: "torch.Tensor",
        dim: int = -1,
        eps: float = 1e-8,
    ) -> "torch.Tensor":
        """
        L2-normalize spectra along specified dimension (PyTorch version).

        Parameters
        ----------
        spectra : torch.Tensor
            Input spectra
        dim : int
            Dimension along which to normalize (default: -1, last dim)
        eps : float
            Small value to avoid division by zero

        Returns
        -------
        normalized : torch.Tensor
            L2-normalized spectra
        """
        norm = torch.norm(spectra, p=2, dim=dim, keepdim=True)
        return spectra / (norm + eps)


# =============================================================================
# Forward Model (Reconstruction)
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

    Computes Y(ν, t) = s(ν) + Σᵢ wᵢ · Bᵢ(ν) · exp(-λᵢ · t)

    Parameters
    ----------
    raman : np.ndarray
        Raman spectrum s(ν), shape (n_wavenumbers,)
    bases : np.ndarray
        Fluorophore bases Bᵢ(ν), shape (n_fluorophores, n_wavenumbers)
    abundances : np.ndarray
        Fluorophore abundances wᵢ, shape (n_fluorophores,)
    decay_rates : np.ndarray
        Decay rates λᵢ (s⁻¹), shape (n_fluorophores,)
    time_values : np.ndarray
        Time points t (s), shape (n_timepoints,)

    Returns
    -------
    reconstruction : np.ndarray
        Reconstructed time series, shape (n_timepoints, n_wavenumbers)
    """
    n_t = len(time_values)
    n_fluorophores = len(decay_rates)

    # Start with Raman (constant across time)
    result = np.tile(raman, (n_t, 1))

    # Add fluorescence contributions
    for i in range(n_fluorophores):
        # Decay factor: exp(-λᵢ · t), shape (n_timepoints,)
        decay = np.exp(-decay_rates[i] * time_values)
        # Fluorescence contribution: wᵢ · Bᵢ(ν) · exp(-λᵢ · t)
        fluor_i = abundances[i] * decay[:, None] * bases[i, None, :]
        result = result + fluor_i

    return result


if TORCH_AVAILABLE:
    def reconstruct_time_series_torch(
        raman: "torch.Tensor",
        bases: "torch.Tensor",
        abundances: "torch.Tensor",
        decay_rates: "torch.Tensor",
        time_values: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Reconstruct bleaching time series from physics parameters (PyTorch version).

        Parameters
        ----------
        raman : torch.Tensor
            Raman spectrum s(ν), shape (n_wavenumbers,)
        bases : torch.Tensor
            Fluorophore bases Bᵢ(ν), shape (n_fluorophores, n_wavenumbers)
        abundances : torch.Tensor
            Fluorophore abundances wᵢ, shape (n_fluorophores,)
        decay_rates : torch.Tensor
            Decay rates λᵢ (s⁻¹), shape (n_fluorophores,)
        time_values : torch.Tensor
            Time points t (s), shape (n_timepoints,)

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed time series, shape (n_timepoints, n_wavenumbers)
        """
        # Raman: (1, n_wavenumbers)
        raman_expanded = raman.unsqueeze(0)

        # Decay factors: exp(-λᵢ · t), shape (n_timepoints, n_fluorophores)
        decay_factors = torch.exp(-decay_rates.unsqueeze(0) * time_values.unsqueeze(1))

        # Weighted bases: wᵢ · Bᵢ(ν), shape (n_fluorophores, n_wavenumbers)
        weighted_bases = abundances.unsqueeze(1) * bases

        # Fluorescence: (n_timepoints, n_fluorophores) @ (n_fluorophores, n_wavenumbers)
        fluorescence = torch.matmul(decay_factors, weighted_bases)

        return raman_expanded + fluorescence
