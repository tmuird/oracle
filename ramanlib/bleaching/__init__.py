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
    decompose_with_known_rates,
)

from ramanlib.bleaching.generate import (
    SyntheticConfig,
    SyntheticBleachingDataset,
)

from ramanlib.bleaching.fluorophores import (
    FluorophoreLoader,
    load_fluorophores,
    filter_bad_fluorophores,
    nm_to_wavenumber,
    wavenumber_to_nm,
)

from ramanlib.bleaching.models import (
    PhysicsDecomposition,
    fit_physics_model,
)

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
    "decompose_with_known_rates",
    # NN Model
    "PhysicsDecomposition",
    "fit_physics_model",
    # Generation
    "SyntheticConfig",
    "SyntheticBleachingDataset",
    # Fluorophores
    "FluorophoreLoader",
    "load_fluorophores",
    "filter_bad_fluorophores",
    "nm_to_wavenumber",
    "wavenumber_to_nm",
    # Visualization
    "plot_decomposition",
    "plot_temporal_decomposition",
    "plot_stacked_decomposition",
    "get_full_decomposition",
]
