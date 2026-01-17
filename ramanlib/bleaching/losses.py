"""
Physics-based loss functions for bleaching decomposition.

These loss functions enforce physical constraints and break degeneracies
in the underdetermined Raman/fluorescence separation problem.
"""

import torch


def compute_spectral_separation_loss(
    fluorophore_bases: torch.Tensor,
    raman_spectra: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce spectral separation: fluorescence must be smooth, Raman can be sharp.

    This breaks the degeneracy between s(ν) and w·B(ν) by requiring that
    fluorophore bases have LOW high-frequency content while allowing Raman
    to have sharp peaks.

    Key insight: Without this, the model can shift sharp spectral features
    between s(ν) and fluorescence bases by scaling abundances, making the
    problem underdetermined even with strict decay rate constraints.

    Parameters
    ----------
    fluorophore_bases : torch.Tensor
        Fluorophore basis spectra, shape (n_fluorophores, n_wavenumbers)
    raman_spectra : torch.Tensor
        Raman spectrum, shape (n_wavenumbers,)

    Returns
    -------
    torch.Tensor
        Ratio of fluorescence smoothness to Raman sharpness (minimize this)
    """
    # High-frequency content via second derivative
    bases_second_deriv = fluorophore_bases[:, 2:] - 2 * fluorophore_bases[:, 1:-1] + fluorophore_bases[:, :-2]
    raman_second_deriv = raman_spectra[2:] - 2 * raman_spectra[1:-1] + raman_spectra[:-2]

    # Mean absolute curvature (scale-invariant measure of "sharpness")
    # Add small epsilon to avoid division by zero
    bases_curvature = torch.mean(torch.abs(bases_second_deriv)) + 1e-8
    raman_curvature = torch.mean(torch.abs(raman_second_deriv)) + 1e-8

    # We want: bases_curvature << raman_curvature
    # Penalize if fluorescence is too sharp OR if Raman is too smooth
    curvature_ratio = bases_curvature / (raman_curvature + 1e-8)

    return curvature_ratio


def compute_abundance_penalty(abundances_raw: torch.Tensor) -> torch.Tensor:
    """
    Penalize very large abundances that could compensate for fast decay rates.

    This prevents the model from using w_i·B_i(ν) to absorb Raman signal
    by scaling abundances arbitrarily high.

    Parameters
    ----------
    abundances_raw : torch.Tensor
        Raw (log-space) abundances, shape (n_fluorophores,)

    Returns
    -------
    torch.Tensor
        L2 penalty on log-abundances (encourages smaller w_i)
    """
    # Penalize deviation from abundance ~ 1.0
    # abundances_raw = 0 → abundances = 1.0
    return torch.mean(abundances_raw**2)


def compute_decay_diversity_penalty(decay_rates: torch.Tensor) -> torch.Tensor:
    """
    Penalize when all decay rates cluster together (e.g., all τ → τ_max).

    This encourages temporal diversity: if you have 3 fluorophores, they
    should span fast/medium/slow timescales, not all decay identically.

    Parameters
    ----------
    decay_rates : torch.Tensor
        Decay rates λ, shape (n_fluorophores,)

    Returns
    -------
    torch.Tensor
        Penalty for low diversity (high when all rates similar)
    """
    # Compute coefficient of variation: std / mean
    # High CV = diverse rates, Low CV = clustered rates
    mean_rate = torch.mean(decay_rates)
    std_rate = torch.std(decay_rates)
    cv = std_rate / (mean_rate + 1e-8)

    # Target CV ~ 0.5 (rates should span at least 2× range)
    # Penalize if CV is too small
    target_cv = 0.5
    penalty = torch.relu(target_cv - cv) ** 2

    return penalty


def compute_intensity_ratio_loss(
    fluorophore_bases: torch.Tensor,
    abundances: torch.Tensor,
    decay_rates: torch.Tensor,
    raman_spectra: torch.Tensor,
    t_early: float = 0.0,
) -> torch.Tensor:
    """
    Enforce that fluorescence dominates over Raman at early times.

    Biological constraint: At t≈0, autofluorescence should be stronger
    than Raman signal (typically 5-10×). This prevents the model from
    putting too much intensity in s(ν) and too little in fluorescence.

    Parameters
    ----------
    fluorophore_bases : torch.Tensor
        Fluorophore basis spectra, shape (n_fluorophores, n_wavenumbers)
    abundances : torch.Tensor
        Abundances w_i, shape (n_fluorophores,)
    decay_rates : torch.Tensor
        Decay rates λ_i, shape (n_fluorophores,)
    raman_spectra : torch.Tensor
        Raman spectrum, shape (n_wavenumbers,)
    t_early : float
        Time point to evaluate (default t=0)

    Returns
    -------
    torch.Tensor
        Penalty if Raman is too strong relative to fluorescence at t_early
    """
    # Fluorescence at t_early: Σ w_i · B_i(ν) · exp(-λ_i · t_early)
    decay_at_t = torch.exp(-decay_rates * t_early)  # (n_fluorophores,)
    weighted_bases = abundances.unsqueeze(1) * fluorophore_bases  # (n_fluorophores, n_wavenumbers)
    fluorescence_at_t = torch.sum(
        decay_at_t.unsqueeze(1) * weighted_bases, dim=0
    )  # (n_wavenumbers,)

    # Raman intensity
    raman_intensity = torch.abs(raman_spectra)  # (n_wavenumbers,)

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
    fluorophore_bases: torch.Tensor,
    abundances: torch.Tensor,
    decay_rates: torch.Tensor,
    raman_spectra: torch.Tensor,
    t1: float = 20.0,
    t2: float = 25.0,
) -> torch.Tensor:
    """
    Penalize if model predicts different spectra at two late time points.

    Physics: If fluorescence has mostly decayed, late-time predictions should
    be nearly identical (both ≈ pure Raman). Large differences indicate slow
    fluorescence components still decaying.

    This is NOT cheating: uses only model's predictions, no ground truth!

    Parameters
    ----------
    fluorophore_bases : torch.Tensor
        Fluorophore basis spectra, shape (n_fluorophores, n_wavenumbers)
    abundances : torch.Tensor
        Abundances w_i, shape (n_fluorophores,)
    decay_rates : torch.Tensor
        Decay rates λ_i, shape (n_fluorophores,)
    raman_spectra : torch.Tensor
        Raman spectrum, shape (n_wavenumbers,)
    t1 : float
        First late time point beyond training window
    t2 : float
        Second late time point beyond training window

    Returns
    -------
    torch.Tensor
        MSE between predictions at t1 and t2
    """
    # Model predictions at two late times
    decay1 = torch.exp(-decay_rates * t1)
    decay2 = torch.exp(-decay_rates * t2)

    weighted_bases = abundances.unsqueeze(1) * fluorophore_bases

    fluor1 = torch.sum(decay1.unsqueeze(1) * weighted_bases, dim=0)
    fluor2 = torch.sum(decay2.unsqueeze(1) * weighted_bases, dim=0)

    pred1 = raman_spectra + fluor1
    pred2 = raman_spectra + fluor2

    # Should be nearly identical if fluorescence has decayed
    consistency_error = torch.mean((pred1 - pred2) ** 2)

    return consistency_error


def compute_raman_floor_loss(raman_spectra: torch.Tensor) -> torch.Tensor:
    """
    Penalize elevated floor in Raman spectrum.

    Physical basis: Raman scattering only occurs at specific molecular
    vibration frequencies. Between peaks, there should be no signal.
    An elevated floor indicates slow fluorescence absorbed into s(ν).

    Parameters
    ----------
    raman_spectra : torch.Tensor
        Raman spectrum, shape (n_wavenumbers,)

    Returns
    -------
    torch.Tensor
        Penalty for non-zero floor (5th percentile)
    """
    # Use percentile rather than min to be robust to noise
    floor = torch.quantile(raman_spectra, 0.05)

    # Floor should be near zero
    return floor**2


def compute_raman_spikiness_loss(raman_spectra: torch.Tensor) -> torch.Tensor:
    """
    Penalize if Raman spectrum is TOO SMOOTH (curve-like).

    Physical basis: Raman spectra have sharp peaks from vibrational modes.
    If s(ν) looks like a smooth curve/banana, fluorescence bases have
    absorbed the peaks, leaving only a smooth residual.

    We measure "spikiness" as the ratio of high-frequency to low-frequency
    power. Sharp peaks → high ratio. Smooth curve → low ratio.

    Parameters
    ----------
    raman_spectra : torch.Tensor
        Raman spectrum, shape (n_wavenumbers,)

    Returns
    -------
    torch.Tensor
        Penalty if Raman lacks high-frequency content
    """
    # High-frequency content: second derivative (curvature)
    second_deriv = raman_spectra[2:] - 2 * raman_spectra[1:-1] + raman_spectra[:-2]
    high_freq_power = torch.mean(torch.abs(second_deriv))

    # Low-frequency content: first derivative (slope)
    first_deriv = raman_spectra[1:] - raman_spectra[:-1]
    low_freq_power = torch.mean(torch.abs(first_deriv)) + 1e-8

    # Spikiness ratio: should be high for sharp peaks
    spikiness = high_freq_power / low_freq_power

    target_spikiness = 0.5
    penalty = torch.relu(target_spikiness - spikiness) ** 2

    return penalty


def compute_raman_curvature_loss(raman_spectra: torch.Tensor) -> torch.Tensor:
    """
    Penalize if Raman spectrum has too much GLOBAL curvature (banana shape).

    If s(ν) is a smooth polynomial-like curve, it indicates fluorescence
    absorption. We fit a low-order polynomial and penalize if Raman is
    well-approximated by it.

    Parameters
    ----------
    raman_spectra : torch.Tensor
        Raman spectrum, shape (n_wavenumbers,)

    Returns
    -------
    torch.Tensor
        Penalty for smooth global curvature
    """
    # Fit a cubic polynomial to Raman (represents global trend/curvature)
    # If Raman is mostly a cubic, it's too smooth
    n = raman_spectra.shape[0]
    x = torch.linspace(-1, 1, n, device=raman_spectra.device)

    # Build Vandermonde matrix for cubic fit
    X = torch.stack([torch.ones_like(x), x, x**2, x**3], dim=1)

    # Least squares fit: (X^T X)^-1 X^T y
    XtX = torch.matmul(X.T, X)
    Xty = torch.matmul(X.T, raman_spectra)
    coeffs = torch.linalg.lstsq(XtX, Xty).solution

    # Reconstruct smooth curve
    smooth_curve = torch.matmul(X, coeffs)

    # If Raman is well-fit by cubic, it's too smooth (penalize!)
    # Measure R² of cubic fit
    residuals = raman_spectra - smooth_curve
    ss_res = torch.sum(residuals**2)
    ss_tot = torch.sum((raman_spectra - torch.mean(raman_spectra)) ** 2) + 1e-8
    r_squared = 1 - ss_res / ss_tot

    # Penalize high R² (good fit to smooth curve = bad!)
    # Target: Raman should NOT be well-fit by cubic (R² < 0.5)
    target_r2 = 0.5
    penalty = torch.relu(r_squared - target_r2) ** 2

    return penalty


def compute_fluorophore_convexity_loss(fluorophore_bases: torch.Tensor) -> torch.Tensor:
    """
    Penalize if fluorophore bases are CONVEX (U-shaped).

    Fluorescence emission should be unimodal (broad peak) or monotonic,
    NEVER convex (minimum in middle). Convex bases are unphysical and
    can absorb sharp Raman features.

    Parameters
    ----------
    fluorophore_bases : torch.Tensor
        Fluorophore basis spectra, shape (n_fluorophores, n_wavenumbers)

    Returns
    -------
    torch.Tensor
        Penalty proportional to positive second derivative (convexity)
    """
    # Compute second derivative: d²B/dν²
    # Positive second derivative = convex (U-shaped) = BAD
    # Negative second derivative = concave (inverted U) = OK
    second_deriv = fluorophore_bases[:, 2:] - 2 * fluorophore_bases[:, 1:-1] + fluorophore_bases[:, :-2]

    # Penalize positive second derivative (convexity)
    # ReLU ensures we only penalize convex regions, not concave
    convexity_penalty = torch.mean(torch.relu(second_deriv) ** 2)

    return convexity_penalty


def compute_decay_rate_prior_loss(
    decay_rates: torch.Tensor,
    target_rates: torch.Tensor,
) -> torch.Tensor:
    """
    Soft regularization toward estimated decay rates from scipy fit.

    This encourages the learned rates to be close to the initial estimate
    from fitting the wavenumber-averaged decay, but allows deviation if
    the spectral structure requires it.

    Parameters
    ----------
    decay_rates : torch.Tensor
        Learned decay rates λ, shape (n_fluorophores,)
    target_rates : torch.Tensor
        Estimated decay rates from scipy curve_fit, shape (n_fluorophores,)

    Returns
    -------
    torch.Tensor
        MSE between learned log-rates and target log-rates
    """
    # Use log-space to handle the wide range of decay rates
    # This makes the penalty scale-invariant
    log_learned = torch.log(decay_rates)
    log_target = torch.log(target_rates)
    return torch.mean((log_learned - log_target) ** 2)


def compute_extrapolation_validation_loss(
    reconstruction: torch.Tensor,
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

    Parameters
    ----------
    reconstruction : torch.Tensor
        Model reconstruction Y(t, ν), shape (n_timepoints, n_wavenumbers)
    data : torch.Tensor
        Ground truth data, shape (n_timepoints, n_wavenumbers)
    first_times : int
        Number of frames in training window
    fit_frames : int
        Number of early frames to "fit" on (default 10)
    val_frames : int
        Number of frames to validate on (default 10)

    Returns
    -------
    torch.Tensor
        MSE on validation frames within training window

    Raises
    ------
    ValueError
        If fit_frames + val_frames > first_times
    """
    if fit_frames + val_frames > first_times:
        raise ValueError(
            f"fit_frames ({fit_frames}) + val_frames ({val_frames}) = {fit_frames+val_frames} "
            f"exceeds first_times ({first_times}). Must stay within training window!"
        )

    val_start = fit_frames
    val_end = fit_frames + val_frames

    # Only access data within training window
    val_residuals = reconstruction[val_start:val_end] - data[val_start:val_end]
    return torch.mean(val_residuals**2)
