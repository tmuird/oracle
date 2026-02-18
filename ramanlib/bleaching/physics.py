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
import torch.nn.functional as F
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


# L2 Normalization


def l2_normalize(
    spectra: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """L2-normalize spectra along specified axis."""
    norm = np.linalg.norm(spectra, axis=axis, keepdims=True)
    return spectra / (norm + eps)


# Forward Model


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
    decay_factors = np.exp(-decay_rates[np.newaxis, :] * time_values[:, np.newaxis])
    weighted_bases = abundances[:, np.newaxis] * bases
    fluorescence = np.matmul(decay_factors, weighted_bases)
    return raman + fluorescence


def reconstruct_time_series_integrated(
    raman: np.ndarray,
    bases: np.ndarray,
    abundances: np.ndarray,
    decay_rates: np.ndarray,
    time_values: np.ndarray,
    frame_duration: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct bleaching time series modelling CCD integration.

    Each frame integrates photons over [tₙ, tₙ + T]:

        Y(ν, n) = s(ν)·T + Σₖ wₖ · Bₖ(ν) · [exp(-λₖ·tₙ) - exp(-λₖ·(tₙ+T))] / λₖ

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
        Start time of each frame (s), shape (n_timepoints,)
    frame_duration : float
        CCD integration time per frame (seconds)

    Returns
    -------
    reconstruction : np.ndarray
        Shape (n_timepoints, n_wavenumbers)
    """
    # decay_rates: [F] -> [1, F],  time_values: [T] -> [T, 1]
    lam = decay_rates[np.newaxis, :]        # [1, F]
    t_start = time_values[:, np.newaxis]    # [T, 1]
    t_end = t_start + frame_duration        # [T, 1]

    # Analytical integral: (1/λ) * [exp(-λ·t_start) - exp(-λ·t_end)]
    # Result: [T, F]
    decay_integral = (
        np.exp(-lam * t_start) - np.exp(-lam * t_end)
    ) / (lam + 1e-8)

    # weighted_bases: [F, W]
    weighted_bases = abundances[:, np.newaxis] * bases

    # [T, F] @ [F, W] -> [T, W]
    fluorescence = np.matmul(decay_integral, weighted_bases)

    # Raman integrates linearly: s(ν) * T
    raman_integrated = raman * frame_duration

    return raman_integrated + fluorescence


def reconstruct_time_series_factored(
    raman: np.ndarray,
    bases: np.ndarray,
    effective_amplitudes: np.ndarray,
    decay_rates: np.ndarray,
    time_values: np.ndarray,
    frame_duration: float = 1.0,
) -> np.ndarray:
    """
    Factored reconstruction that decouples amplitude from decay rate.

    Y(ν, n) = s(ν)·T + Σₖ ãₖ · Bₖ(ν) · exp(-λₖ · tₙ)

    where ãₖ = wₖ · [1 - exp(-λₖ·T)] / λₖ is the effective amplitude.

    This is mathematically equivalent to the integrated model but
    reparameterizes so that ã controls the observable magnitude at t=0
    independently of the decay rate λ.

    Parameters
    ----------
    raman : np.ndarray
        Raman spectrum s(ν), shape (n_wavenumbers,)
    bases : np.ndarray
        Fluorophore bases Bₖ(ν), shape (n_fluorophores, n_wavenumbers)
    effective_amplitudes : np.ndarray
        Effective amplitudes ãₖ, shape (n_fluorophores,)
    decay_rates : np.ndarray
        Decay rates λₖ (s⁻¹), shape (n_fluorophores,)
    time_values : np.ndarray
        Start time of each frame (s), shape (n_timepoints,)
    frame_duration : float
        CCD integration time per frame (seconds)

    Returns
    -------
    reconstruction : np.ndarray
        Shape (n_timepoints, n_wavenumbers)
    """
    # Simple point-sampling exponential decay: [T, F]
    decay_matrix = np.exp(
        -decay_rates[np.newaxis, :] * time_values[:, np.newaxis]
    )

    # Weighted bases: [F, W]
    weighted_bases = effective_amplitudes[:, np.newaxis] * bases

    # [T, F] @ [F, W] -> [T, W]
    fluorescence = np.matmul(decay_matrix, weighted_bases)

    # Raman integrates linearly: s(ν) * T
    raman_integrated = raman * frame_duration

    return raman_integrated + fluorescence


def effective_to_physical_abundance(
    effective_amplitudes: np.ndarray,
    decay_rates: np.ndarray,
    frame_duration: float,
) -> np.ndarray:
    """
    Convert effective amplitudes to physical abundances.

    w = ã · λ / [1 - exp(-λ·T)]

    Well-behaved as λ→0 (w → ã/T via L'Hôpital).

    Parameters
    ----------
    effective_amplitudes : np.ndarray
        Effective amplitudes ã, shape (n_fluorophores,) or (batch, n_fluorophores)
    decay_rates : np.ndarray
        Decay rates λ (s⁻¹), same shape as effective_amplitudes
    frame_duration : float
        CCD integration time per frame (seconds)

    Returns
    -------
    abundances : np.ndarray
        Physical abundances w, same shape as input
    """
    return effective_amplitudes * decay_rates / (
        1.0 - np.exp(-decay_rates * frame_duration) + 1e-8
    )


def physical_to_effective_amplitude(
    abundances: np.ndarray,
    decay_rates: np.ndarray,
    frame_duration: float,
) -> np.ndarray:
    """
    Convert physical abundances to effective amplitudes.

    ã = w · [1 - exp(-λ·T)] / λ

    Parameters
    ----------
    abundances : np.ndarray
        Physical abundances w
    decay_rates : np.ndarray
        Decay rates λ (s⁻¹)
    frame_duration : float
        CCD integration time per frame (seconds)

    Returns
    -------
    effective_amplitudes : np.ndarray
        Effective amplitudes ã
    """
    return abundances * (1.0 - np.exp(-decay_rates * frame_duration)) / (
        decay_rates + 1e-8
    )


# Polynomial Fitting

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

        # Normalise wavenumbers using provided stats (or compute if not provided). This ensures consistency with training.
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
        raman: torch.Tensor,  # [Batch, Wavenumbers]
        bases: torch.Tensor,  # [Fluors, Wavenumbers] (Global Parameter)
        abundances: torch.Tensor,  # [Batch, Fluors]
        decay_rates: torch.Tensor,  # [Batch, Fluors]
        time_values: torch.Tensor,  # [Timepoints] (Buffer)
    ) -> torch.Tensor:
        """
        Batch-Safe Physics Reconstruction using Matrix Multiplication.

        Shapes:
        - Input Raman: [B, W]
        - Output:      [B, W, T] (Standard image format for CNNs) or [B, T, W] (Sequence format)
        """

        # 1. Create Decay Matrix [Batch, Time, Fluors]
        # We want exp(-lambda * t)

        # decay_rates: [B, F] -> [B, 1, F]
        lam = decay_rates.unsqueeze(1)

        # time_values: [T] -> [1, T, 1]
        t = time_values.view(1, -1, 1)

        # Result: [B, T, F]
        # This gives the decay curve for every fluorophore in every batch sample
        # print(f"Decay matrix : {lam}")
        decay_matrix = torch.exp(-lam * t)
        

        
        # decay_matrix = decay_matrix / (decay_matrix.mean(dim=1, keepdim=True) + 1e-8) 
        # Create Weighted Bases [Batch, Fluors, Wavenumbers]
        # abundances: [B, F] -> [B, F, 1]
        w = abundances.unsqueeze(2)

        # bases: [F, W] -> [1, F, W] (Broadcasts to Batch size)
        B = bases.unsqueeze(0)

        # Result: [B, F, W]
        weighted_bases = w * B

        # Matrix Multiplication
        # [B, T, F] @ [B, F, W] -> [B, T, W]
        # For each sample, we sum over Fluors (F) 
        fluorescence = torch.matmul(decay_matrix, weighted_bases)

        # Add Raman
        # Raman is [B, W]. We need [B, T, W]
        # We broadcast Raman across time
        raman_expanded = raman.unsqueeze(1)  # [B, 1, W]

        total_signal = fluorescence + raman_expanded  # [B, T, W]


        # return [B, W, T] for CNN
        return total_signal.transpose(1, 2)


    def reconstruct_time_series_integrated_torch(
        raman: torch.Tensor,  # [Batch, Wavenumbers]
        bases: torch.Tensor,  # [Fluors, Wavenumbers] (Global Parameter)
        abundances: torch.Tensor,  # [Batch, Fluors]
        decay_rates: torch.Tensor,  # [Batch, Fluors]
        time_values: torch.Tensor,  # [Timepoints] - start time of each frame
        frame_duration: float = 0.1,  # Integration time per frame (seconds)
    ) -> torch.Tensor:
        """
        Physically correct reconstruction modelling CCD integration.

        Each frame integrates photons over [t_n, t_n + frame_duration].
        The CCD measures the integral, not a point sample:

            Measured(n) = S(ν)·T + Σᵢ wᵢ·Bᵢ(ν) · (1/λᵢ) · [exp(-λᵢ·tₙ) - exp(-λᵢ·(tₙ+T))]

        This correctly accounts for within-frame bleaching and allows
        extraction of information about fast-decaying components.

        Shapes:
        - Input Raman: [B, W]
        - Output:      [B, W, T]

        Args:
            raman:          Raman spectrum per sample [B, W]
            bases:          Fluorophore basis spectra [F, W] (L2-normalised)
            abundances:     Fluorophore abundances per sample [B, F]
            decay_rates:    Photobleaching rates per sample [B, F] (s⁻¹)
            time_values:    Start time of each frame [T] (seconds)
            frame_duration: CCD integration time per frame (seconds)
        """
        # decay_rates: [B, F] -> [B, 1, F]
        lam = decay_rates.unsqueeze(1)

        # time_values: [T] -> [1, T, 1]
        t_start = time_values.view(1, -1, 1)
        t_end = t_start + frame_duration

        # Analytical integral of exp(-λt) from t_start to t_end:
        # ∫ exp(-λt) dt = (1/λ) · [exp(-λ·t_start) - exp(-λ·t_end)]
        #
        # Result: [B, T, F]
        decay_matrix = (
            torch.exp(-lam * t_start) - torch.exp(-lam * t_end)
        ) / (lam + 1e-8)

        # Weighted bases: [B, F, W]
        w = abundances.unsqueeze(2)       # [B, F, 1]
        B = bases.unsqueeze(0)            # [1, F, W]
        weighted_bases = w * B            # [B, F, W]

        # [B, T, F] @ [B, F, W] -> [B, T, W]
        fluorescence = torch.matmul(decay_matrix, weighted_bases)

        # Raman contribution: constant rate × integration time
        raman_integrated = raman.unsqueeze(1) * frame_duration  # [B, 1, W]

        total_signal = fluorescence + raman_integrated  # [B, T, W]

        # return [B, W, T]
        return total_signal.transpose(1, 2)

    def reconstruct_time_series_factored_torch(
        raman: torch.Tensor,              # [Batch, Wavenumbers]
        bases: torch.Tensor,              # [Fluors, Wavenumbers]
        effective_amplitudes: torch.Tensor,  # [Batch, Fluors] — ã values
        decay_rates: torch.Tensor,        # [Batch, Fluors]
        time_values: torch.Tensor,        # [Timepoints]
        frame_duration: float = 0.1,
    ) -> torch.Tensor:
        """
        Factored reconstruction that decouples amplitude from decay rate.

        Y(ν, n) = S(ν)·T + Σᵢ ãᵢ · Bᵢ(ν) · exp(-λᵢ · tₙ)

        where ãᵢ = wᵢ · [1 - exp(-λᵢ·T)] / λᵢ is the effective amplitude.

        Mathematically equivalent to the integrated model, but the decoder
        directly outputs ã (observable amplitude at frame 0) rather than w
        (physical abundance). This breaks the amplitude-rate coupling that
        causes identifiability issues.

        Shapes:
        - Input Raman: [B, W]
        - Output:      [B, W, T]
        """
        # decay_rates: [B, F] -> [B, 1, F]
        lam = decay_rates.unsqueeze(1)

        # time_values: [T] -> [1, T, 1]
        t = time_values.view(1, -1, 1)

        # Simple point-sampling exponential: [B, T, F]
        decay_matrix = torch.exp(-lam * t)

        # Weighted bases with effective amplitudes: [B, F, W]
        a = effective_amplitudes.unsqueeze(2)  # [B, F, 1]
        B = bases.unsqueeze(0)                 # [1, F, W]
        weighted_bases = a * B                 # [B, F, W]

        # [B, T, F] @ [B, F, W] -> [B, T, W]
        fluorescence = torch.matmul(decay_matrix, weighted_bases)

        # Raman contribution: s(ν) * T
        raman_integrated = raman.unsqueeze(1) * frame_duration  # [B, 1, W]

        total_signal = fluorescence + raman_integrated  # [B, T, W]

        # return [B, W, T]
        return total_signal.transpose(1, 2)

    def effective_to_physical_abundance_torch(
        effective_amplitudes: torch.Tensor,
        decay_rates: torch.Tensor,
        frame_duration: float,
    ) -> torch.Tensor:
        """
        Convert effective amplitudes to physical abundances.

        w = ã · λ / [1 - exp(-λ·T)]

        Parameters
        ----------
        effective_amplitudes : torch.Tensor
            Effective amplitudes ã, shape (..., n_fluorophores)
        decay_rates : torch.Tensor
            Decay rates λ (s⁻¹), same shape
        frame_duration : float
            CCD integration time per frame (seconds)

        Returns
        -------
        abundances : torch.Tensor
            Physical abundances w
        """
        return effective_amplitudes * decay_rates / (
            1.0 - torch.exp(-decay_rates * frame_duration) + 1e-8
        )
