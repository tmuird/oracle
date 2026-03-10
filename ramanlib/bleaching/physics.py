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
    return np.stack([wn_norm ** k for k in range(n_coeffs)], axis=1)


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


def reconstruct_time_series_numpy(
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
    lam = decay_rates[np.newaxis, :]  # [1, F]
    t_start = time_values[:, np.newaxis]  # [T, 1]
    t_end = t_start + frame_duration  # [T, 1]

    # Analytical integral: (1/λ) * [exp(-λ·t_start) - exp(-λ·t_end)]
    # Result: [T, F]
    decay_integral = (np.exp(-lam * t_start) - np.exp(-lam * t_end)) / (lam + 1e-8)

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
    decay_matrix = np.exp(-decay_rates[np.newaxis, :] * time_values[:, np.newaxis])

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
    return (
            effective_amplitudes
            * decay_rates
            / (1.0 - np.exp(-decay_rates * frame_duration) + 1e-8)
    )


def physical_to_effective_amplitude(
        abundances: torch.Tensor,
        decay_rates: torch.Tensor,
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
    return (
            abundances
            * (1.0 - np.exp(-decay_rates * frame_duration))
            / (decay_rates + 1e-8)
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

    Wavenumbers are normalized to [-1, 1] using min/max for numerical stability.

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
    wn_min : float
        Wavenumber min used for normalization
    wn_max : float
        Wavenumber max used for normalization
    """
    n_fluorophores = bases.shape[0]
    n_coeffs = degree + 1

    # Normalize to [-1, 1] using min/max
    wn_min = float(wavenumbers.min())
    wn_max = float(wavenumbers.max())
    wn_normalized = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0

    log_poly_coeffs = np.zeros((n_fluorophores, n_coeffs))
    for i in range(n_fluorophores):
        log_basis = np.log(bases[i] + 1e-8)
        # np.polyfit returns descending order, reverse to ascending
        log_coeffs = np.polyfit(wn_normalized, log_basis, deg=degree)
        log_poly_coeffs[i] = log_coeffs[::-1]

    return log_poly_coeffs, wn_min, wn_max


def evaluate_polynomial_bases(
        log_poly_coeffs: np.ndarray,
        wavenumbers: np.ndarray,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
) -> np.ndarray:
    """
    Evaluate polynomial fluorophore bases in log-space, then exponentiate.

    Wavenumbers are normalized to [-1, 1]. If wn_min/wn_max are not provided,
    they are computed from the wavenumbers (self-normalizing).

    Parameters
    ----------
    log_poly_coeffs : np.ndarray
        Shape (n_fluorophores, degree+1), ascending power order
    wavenumbers : np.ndarray
        Wavenumber axis (cm⁻¹)
    wn_min : float, optional
        Wavenumber min for normalization. If None, computed from wavenumbers.
    wn_max : float, optional
        Wavenumber max for normalization. If None, computed from wavenumbers.

    Returns
    -------
    bases : np.ndarray
        Shape (n_fluorophores, n_wavenumbers)
    """
    if log_poly_coeffs.ndim == 1:
        log_poly_coeffs = log_poly_coeffs[None, :]
    degree = log_poly_coeffs.shape[1] - 1

    if wn_min is None:
        wn_min = float(wavenumbers.min())
    if wn_max is None:
        wn_max = float(wavenumbers.max())

    wn_normalized = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
    vandermonde = np.vander(wn_normalized, N=degree + 1, increasing=True)

    log_intensity_values = log_poly_coeffs @ vandermonde.T
    bases = np.exp(log_intensity_values)

    return bases


def interpolate_bases(
        bases: np.ndarray,
        source_wn: np.ndarray,
        target_wn: np.ndarray,
        method: str = "pchip",
        smooth_sigma: float = 0.0,
) -> np.ndarray:
    """
    Interpolate fluorophore bases from one wavenumber axis onto another.

    Handles unsorted source axes. Non-negative clipping is applied after
    interpolation to remove any overshoot artifacts.

    Parameters
    ----------
    bases : np.ndarray
        Fluorophore spectra, shape (n_fluorophores, n_source)
    source_wn : np.ndarray
        Source wavenumber axis, shape (n_source,)
    target_wn : np.ndarray
        Target wavenumber axis, shape (n_target,)
    method : str
        Interpolation method:
        - 'pchip'  (default) Monotone piecewise cubic Hermite. Smooth, no
                   oscillations between data points. Best for sparse→dense
                   upsampling of smooth fluorescence spectra.
        - 'spline' Exact cubic spline (s=0). Can produce ringing/wobbles
                   when upsampling from very sparse (~36) to dense (~630)
                   grids because it is forced to pass exactly through every
                   point.
        - 'linear' Simple linear interpolation. Always stable but jagged.
    smooth_sigma : float
        If > 0, apply Gaussian smoothing to the interpolated result with
        this standard deviation (in wavenumber units, cm⁻¹). Useful when
        the source spectra themselves contain measurement noise. Default 0
        (no smoothing).

    Returns
    -------
    np.ndarray
        Interpolated bases, shape (n_fluorophores, n_target)
    """
    axes_match = len(source_wn) == len(target_wn) and np.allclose(source_wn, target_wn)
    if axes_match:
        result = bases.copy()
    else:
        try:
            from scipy.interpolate import PchipInterpolator, UnivariateSpline

            _have_scipy = True
        except ImportError:
            _have_scipy = False

        sort_idx = np.argsort(source_wn)
        source_wn_sorted = source_wn[sort_idx]
        bases_sorted = bases[:, sort_idx]

        result = np.zeros((bases.shape[0], len(target_wn)))
        for i in range(bases.shape[0]):
            if method == "pchip" and _have_scipy:
                interp = PchipInterpolator(
                    source_wn_sorted, bases_sorted[i], extrapolate=True
                )
                result[i] = interp(target_wn)
            elif method == "spline" and _have_scipy:
                spline = UnivariateSpline(source_wn_sorted, bases_sorted[i], k=3, s=0)
                result[i] = spline(target_wn)
            else:
                result[i] = np.interp(
                    target_wn, source_wn_sorted, bases_sorted[i], left=0.0, right=0.0
                )

    if smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter1d

        # Convert sigma from cm⁻¹ to pixels on the target axis
        wn_spacing = float(np.mean(np.diff(np.sort(target_wn))))
        sigma_px = smooth_sigma / wn_spacing
        result = gaussian_filter1d(result, sigma=sigma_px, axis=1)

    return np.maximum(result, 0.0)


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
            wn_min: Optional[float] = None,
            wn_max: Optional[float] = None,
    ) -> "torch.Tensor":
        """
        Evaluate polynomial fluorophore bases in log-space, then exponentiate.

        Wavenumbers are normalized to [-1, 1] using min/max. If wn_min/wn_max
        are not provided, they are computed from the wavenumbers (self-normalizing).

        Parameters
        ----------
        log_poly_coeffs : torch.Tensor
            Shape (n_fluorophores, degree+1), ascending power order
        wavenumbers : torch.Tensor
            Wavenumber axis (cm⁻¹)
        wn_min : float, optional
            Wavenumber min for normalization. If None, computed from wavenumbers.
        wn_max : float, optional
            Wavenumber max for normalization. If None, computed from wavenumbers.

        Returns
        -------
        bases : torch.Tensor
            Shape (n_fluorophores, n_wavenumbers)
        """
        degree = log_poly_coeffs.shape[1] - 1

        if wn_min is None:
            wn_min = float(wavenumbers.min().item())
        if wn_max is None:
            wn_max = float(wavenumbers.max().item())

        wn_normalized = 2.0 * (wavenumbers - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
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
        return torch.stack([wn_norm ** k for k in range(n_coeffs)], dim=1)


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
            frame_duration: float = 0.1,
    ) -> torch.Tensor:
        """
        Batch-Safe Physics Reconstruction using Matrix Multiplication.

        raman is treated as a rate (counts/second); it is multiplied by frame_duration
        to give counts/frame, consistent with reconstruct_time_series_integrated_torch
        and reconstruct_time_series_factored_torch.

        Shapes:
        - Input Raman: [B, W]
        - Output:      [B, W, T]
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

        B = bases

        # Result: [B, F, W]
        weighted_bases = w * B

        # Matrix Multiplication
        # [B, T, F] @ [B, F, W] -> [B, T, W]
        # For each sample, we sum over Fluors (F)
        fluorescence = torch.matmul(decay_matrix, weighted_bases)

        # Add Raman: rate × frame_duration = counts/frame, broadcast across time
        raman_integrated = raman.unsqueeze(1) * frame_duration  # [B, 1, W]

        total_signal = fluorescence + raman_integrated  # [B, T, W]

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
        # \int exp(-λt) dt = (1/λ) · [exp(-λ·t_start) - exp(-λ·t_end)]
        #
        # Result: [B, T, F]
        decay_matrix = (torch.exp(-lam * t_start) - torch.exp(-lam * t_end)) / (
                lam + 1e-8
        )

        # Weighted bases: [B, F, W]
        w = abundances.unsqueeze(2)  # [B, F, 1]
        B = bases.unsqueeze(0)  # [1, F, W]
        weighted_bases = w * B  # [B, F, W]

        # [B, T, F] @ [B, F, W] -> [B, T, W]
        fluorescence = torch.matmul(decay_matrix, weighted_bases)

        # Raman contribution: constant rate × integration time
        raman_integrated = raman.unsqueeze(1) * frame_duration  # [B, 1, W]

        total_signal = fluorescence + raman_integrated  # [B, T, W]

        # return [B, W, T]
        return total_signal.transpose(1, 2)


    def reconstruct_time_series_factored_torch(
            raman: torch.Tensor,  # [Batch, Wavenumbers]
            bases: torch.Tensor,  # [Fluors, Wavenumbers] or [Batch, Fluors, Wavenumbers]
            effective_amplitudes: torch.Tensor,  # [Batch, Fluors] — ã values
            decay_rates: torch.Tensor,  # [Batch, Fluors]
            time_values: torch.Tensor,  # [Timepoints]
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
        # bases may be shared [F, W] or per-sample [B, F, W]
        B = bases if bases.dim() == 3 else bases.unsqueeze(0)  # [B or 1, F, W]
        weighted_bases = a * B  # [B, F, W]

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
        return (
                effective_amplitudes
                * decay_rates
                / (1.0 - torch.exp(-decay_rates * frame_duration) + 1e-8)
        )


def reconstruct_time_series_numpy(
        raman: np.ndarray,
        bases: np.ndarray,
        abundances: np.ndarray,
        decay_rates: np.ndarray,
        time_values: np.ndarray,
        physics_model: str,
        frame_duration: float = 1.0,

) -> np.ndarray:
    """
    Wrapper to use the torch-based factored reconstruction from numpy code (visualization).
    Handles conversion from physical abundances to effective amplitudes.
    Output shape: [Time, Wavenumbers] (same as numpy version).
    """
    # Convert to torch
    raman_t = torch.from_numpy(raman).float().unsqueeze(0)  # [1, W]
    bases_t = torch.from_numpy(bases).float()  # [F, W] — torch fns add batch dim internally
    abundances_t = torch.from_numpy(abundances).float().unsqueeze(0)  # [1, F]
    decay_rates_t = torch.from_numpy(decay_rates).float().unsqueeze(0)  # [1, F]
    time_values_t = torch.from_numpy(time_values).float()  # [T]

    # # Call Factored Reconstruction -> [B, W, T]
    # recon_t = reconstruct_time_series_factored_torch(
    #     raman_t,
    #     bases_t,
    #     effective_amplitudes_t,
    #     decay_rates_t,
    #     time_values_t,
    #     frame_duration)

    if physics_model == "integrated":
        x_recon = reconstruct_time_series_integrated_torch(
            raman=raman_t,
            bases=bases_t,
            abundances=abundances_t,
            decay_rates=decay_rates_t,
            time_values=time_values_t,
            frame_duration=frame_duration,
        )
    elif physics_model == "pointsample":
        x_recon = reconstruct_time_series_torch(
            raman=raman_t,
            bases=bases_t,
            abundances=abundances_t,
            decay_rates=decay_rates_t,
            time_values=time_values_t,
            frame_duration=frame_duration,
        )
    else:  # "factored" (default)
        # Convert Physical Abundances -> Effective Amplitudes
        effective_amplitudes_t = physical_to_effective_amplitude(
            abundances_t, decay_rates_t, frame_duration
        )
        x_recon = reconstruct_time_series_factored_torch(
            raman=raman_t,
            bases=bases_t,
            effective_amplitudes=effective_amplitudes_t,
            decay_rates=decay_rates_t,
            time_values=time_values_t,
            frame_duration=frame_duration,
        )

    # Convert to [T, W] numpy: [1, W, T] -> [1, T, W] -> [T, W]
    return x_recon.transpose(1, 2).squeeze(0).numpy()
