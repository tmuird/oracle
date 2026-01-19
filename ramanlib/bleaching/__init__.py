"""
Bleaching utilities for Raman/fluorescence spectral decomposition.

Physics model:
    Y(ν, t) = s(ν) + Σₖ wₖ · Bₖ(ν) · exp(-λₖ · t)

where:
    - s(ν): Raman spectrum (time-invariant)
    - Bₖ(ν): Fluorophore basis spectra (L2-normalized)
    - wₖ: Abundances (amplitudes)
    - λₖ: Decay rates (τ = 1/λ)
"""

from ramanlib.bleaching.physics import (
    reconstruct_time_series,
    normalize_wavenumbers,
    build_vandermonde,
    l2_normalize,
    fit_polynomial_bases,
    evaluate_polynomial_bases,
)

from ramanlib.bleaching.decompose import (
    DecompositionResult,
    decompose,
    solve_spectra_given_rates,
    solve_spectra_with_polynomial_bases,
    decompose_with_known_rates,
    estimate_decay_rates_from_early_frames,
)

from ramanlib.bleaching.generate import (
    SyntheticConfig,
    SyntheticBleachingDataset,
)

from ramanlib.bleaching.fluorophores import (
    FluorophoreLoader,
    load_fluorophores,
    filter_bad_fluorophores,
    generate_synthetic_fluorophores,
    nm_to_wavenumber,
    wavenumber_to_nm,
)

from ramanlib.bleaching.models import (
    PhysicsDecomposition,
    fit_physics_model,
    # get_default_loss_weights,  # Commented out in models.py
)

# Losses are commented out in losses.py
# from ramanlib.bleaching.losses import (
#     compute_spectral_separation_loss,
#     compute_abundance_penalty,
#     compute_decay_diversity_penalty,
#     compute_intensity_ratio_loss,
#     compute_late_time_consistency_loss,
#     compute_raman_floor_loss,
#     compute_raman_spikiness_loss,
#     compute_raman_curvature_loss,
#     compute_fluorophore_convexity_loss,
#     compute_decay_rate_prior_loss,
#     compute_extrapolation_validation_loss,
# )

from ramanlib.bleaching.visualize import (
    plot_decomposition,
    plot_temporal_decomposition,
    get_full_decomposition,
)

__all__ = [
    # Physics
    "reconstruct_time_series",
    "normalize_wavenumbers",
    "build_vandermonde",
    "l2_normalize",
    "fit_polynomial_bases",
    "evaluate_polynomial_bases",
    # Decomposition (DE-based)
    "DecompositionResult",
    "decompose",
    "solve_spectra_given_rates",
    "solve_spectra_with_polynomial_bases",
    "decompose_with_known_rates",
    "estimate_decay_rates_from_early_frames",
    # NN Model
    "PhysicsDecomposition",
    "fit_physics_model",
    # "get_default_loss_weights",  # Commented out in models.py
    # Loss functions - commented out in losses.py
    # "compute_spectral_separation_loss",
    # "compute_abundance_penalty",
    # "compute_decay_diversity_penalty",
    # "compute_intensity_ratio_loss",
    # "compute_late_time_consistency_loss",
    # "compute_raman_floor_loss",
    # "compute_raman_spikiness_loss",
    # "compute_raman_curvature_loss",
    # "compute_fluorophore_convexity_loss",
    # "compute_decay_rate_prior_loss",
    # "compute_extrapolation_validation_loss",
    # Generation
    "SyntheticConfig",
    "SyntheticBleachingDataset",
    # Fluorophores
    "FluorophoreLoader",
    "load_fluorophores",
    "filter_bad_fluorophores",
    "generate_synthetic_fluorophores",
    "nm_to_wavenumber",
    "wavenumber_to_nm",
    # Visualization
    "plot_decomposition",
    "plot_temporal_decomposition",
    "get_full_decomposition",
]
