"""
Neural network models for bleaching decomposition.

Contains PhysicsDecomposition - a PyTorch nn.Module that learns
Raman and fluorescence spectra via gradient descent.

For most use cases, the DE-based decompose() function is preferred.
Use PhysicsDecomposition when:
- You need differentiable decomposition in a larger pipeline
- You want to add custom regularization losses
- You're training on many samples with shared structure
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ramanlib.bleaching.physics import (
    fit_polynomial_bases,
    build_vandermonde_torch,
    l2_normalize_torch,
    reconstruct_time_series_torch,
)


class PhysicsDecomposition(nn.Module):
    """
    Physics-constrained decomposition model.

    Model: Y(ν, t) = s(ν) + Σₖ wₖ · Bₖ(ν) · exp(-λₖ · t)

    Parameters are stored in log-space for positivity constraints.
    """

    def __init__(
        self,
        n_wavenumbers: int,
        n_timepoints: int,
        n_fluorophores: int = 3,
        time_values: Optional[torch.Tensor] = None,
        initial_abundances: Optional[torch.Tensor] = None,
        initial_decay_rates: Optional[torch.Tensor] = None,
        initial_fluorophore_bases: Optional[torch.Tensor] = None,
        initial_raman_spectrum: Optional[torch.Tensor] = None,
        initial_poly_coeffs: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        basis_type: str = "free",
        polynomial_degree: int = 8,
        wavenumber_axis: Optional[torch.Tensor] = None,
        min_decay_rate: float = 0.01,
    ):
        super().__init__()
        self.n_wavenumbers = n_wavenumbers
        self.n_timepoints = n_timepoints
        self.n_fluorophores = n_fluorophores
        self.device = device
        self.basis_type = basis_type
        self.polynomial_degree = polynomial_degree
        self.min_decay_rate = min_decay_rate

        # Time axis
        if time_values is not None:
            self.register_buffer("time_values", time_values.to(device))
        else:
            self.register_buffer(
                "time_values",
                torch.arange(n_timepoints, dtype=torch.float32, device=device),
            )

        # Raman spectrum (log-space)
        if initial_raman_spectrum is not None:
            raman_tensor = self._to_tensor(initial_raman_spectrum, device)
            self.raman_spectrum = nn.Parameter(torch.log(raman_tensor + 1e-8))
        else:
            self.raman_spectrum = nn.Parameter(
                torch.randn(n_wavenumbers, device=device) * 0.1
            )

        # Fluorophore bases
        if basis_type == "free":
            if initial_fluorophore_bases is not None:
                bases_tensor = self._to_tensor(initial_fluorophore_bases, device)
                self.fluorophore_bases_raw = nn.Parameter(
                    torch.log(bases_tensor + 1e-8)
                )
            else:
                self.fluorophore_bases_raw = nn.Parameter(
                    torch.randn(n_fluorophores, n_wavenumbers, device=device) * 0.1
                )
        elif basis_type == "polynomial":
            n_coeffs = polynomial_degree + 1
            if initial_poly_coeffs is not None:
                poly_tensor = self._to_tensor(initial_poly_coeffs, device)
                self.poly_coeffs = nn.Parameter(poly_tensor)
            elif initial_fluorophore_bases is not None:
                assert wavenumber_axis is not None
                wn_np = (
                    wavenumber_axis.cpu().numpy()
                    if isinstance(wavenumber_axis, torch.Tensor)
                    else wavenumber_axis
                )
                bases_np = (
                    initial_fluorophore_bases.cpu().numpy()
                    if isinstance(initial_fluorophore_bases, torch.Tensor)
                    else initial_fluorophore_bases
                )
                coeffs, _, _ = fit_polynomial_bases(bases_np, wn_np, polynomial_degree)
                self.poly_coeffs = nn.Parameter(
                    torch.tensor(coeffs, dtype=torch.float32, device=device)
                )
            else:
                self.poly_coeffs = nn.Parameter(
                    torch.randn(n_fluorophores, n_coeffs, device=device) * 0.1
                )

            # Normalized wavenumber axis
            if wavenumber_axis is not None:
                wn = (
                    wavenumber_axis.to(device)
                    if isinstance(wavenumber_axis, torch.Tensor)
                    else torch.tensor(
                        wavenumber_axis, dtype=torch.float32, device=device
                    )
                )
            else:
                wn = torch.arange(n_wavenumbers, dtype=torch.float32, device=device)

            wn_min, wn_max = wn.min(), wn.max()
            wn_normalised = 2.0 * (wn - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
            self.register_buffer("wn_normalised", wn_normalised)
            vandermonde = build_vandermonde_torch(wn_normalised, polynomial_degree)
            self.register_buffer("vandermonde", vandermonde)
        else:
            raise ValueError(f"Unknown basis_type: {basis_type}")

        # Abundances (log-space)
        if initial_abundances is not None:
            abund_tensor = self._to_tensor(initial_abundances, device)
            self.abundances_raw = nn.Parameter(torch.log(abund_tensor + 1e-8))
        else:
            self.abundances_raw = nn.Parameter(
                torch.zeros(n_fluorophores, device=device)
            )

        # Decay rates (log-space, with floor)
        if initial_decay_rates is not None:
            rates_tensor = self._to_tensor(initial_decay_rates, device)
            adjusted = rates_tensor - min_decay_rate
            if torch.any(adjusted <= 0):
                raise ValueError(
                    f"initial_decay_rates must be > min_decay_rate ({min_decay_rate})"
                )
            self.decay_rates_raw = nn.Parameter(torch.log(adjusted + 1e-8))
        else:
            self.decay_rates_raw = nn.Parameter(
                torch.randn(n_fluorophores, device=device) * 0.1
            )

    @staticmethod
    def _to_tensor(x, device):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    @property
    def fluorophore_bases(self) -> torch.Tensor:
        if self.basis_type == "free":
            bases = torch.exp(self.fluorophore_bases_raw)
        else:
            poly_values = torch.matmul(self.poly_coeffs, self.vandermonde.T)
            bases = torch.exp(poly_values)
        return l2_normalize_torch(bases, dim=1)

    @property
    def raman_spectra(self) -> torch.Tensor:
        return torch.exp(self.raman_spectrum)

    @property
    def abundances(self) -> torch.Tensor:
        return torch.exp(self.abundances_raw)

    @property
    def decay_rates(self) -> torch.Tensor:
        return torch.exp(self.decay_rates_raw) + self.min_decay_rate

    def forward(self) -> torch.Tensor:
        """Reconstruct time series from parameters."""
        return reconstruct_time_series_torch(
            self.raman_spectra,
            self.fluorophore_bases,
            self.abundances,
            self.decay_rates,
            self.time_values,
        )

    def get_decomposition(self) -> Dict[str, np.ndarray]:
        """Return components as numpy arrays, sorted by decay rate."""
        with torch.no_grad():
            rates = self.decay_rates.cpu().numpy()
            sort_idx = np.argsort(rates)[::-1]

            result = {
                "raman": self.raman_spectra.cpu().numpy(),
                "fluorophore_bases": self.fluorophore_bases.cpu().numpy()[sort_idx],
                "abundances": self.abundances.cpu().numpy()[sort_idx],
                "decay_rates": rates[sort_idx],
                "rates": rates[sort_idx],
                "time_constants": (1.0 / rates)[sort_idx],
            }
            if self.basis_type == "polynomial":
                result["poly_coeffs"] = self.poly_coeffs.cpu().numpy()[sort_idx]
            return result

    def get_fluorescence_component(self, component_idx: int) -> np.ndarray:
        """Get time-resolved fluorescence for one component."""
        with torch.no_grad():
            rates = self.decay_rates.cpu().numpy()
            sort_idx = np.argsort(rates)[::-1]
            actual_idx = sort_idx[component_idx]

            decay = torch.exp(-self.decay_rates[actual_idx] * self.time_values)
            amplitude = self.abundances[actual_idx]
            basis = self.fluorophore_bases[actual_idx]

            component = decay.unsqueeze(1) * amplitude * basis.unsqueeze(0)
            return component.cpu().numpy()


#     # Physics loss functions
#     def compute_spectral_separation_loss(self) -> torch.Tensor:
#         """Enforce spectral separation: fluorescence smooth, Raman sharp."""
#         return losses.compute_spectral_separation_loss(
#             self.fluorophore_bases,
#             self.raman_spectra,
#         )

#     def compute_abundance_penalty(self) -> torch.Tensor:
#         """Penalize very large abundances."""
#         return losses.compute_abundance_penalty(self.abundances_raw)

#     def compute_decay_diversity_penalty(self) -> torch.Tensor:
#         """Penalize clustered decay rates."""
#         return losses.compute_decay_diversity_penalty(self.decay_rates)

#     def compute_intensity_ratio_loss(self, t_early: float = 0.0) -> torch.Tensor:
#         """Enforce fluorescence >> Raman at early times."""
#         return losses.compute_intensity_ratio_loss(
#             self.fluorophore_bases,
#             self.abundances,
#             self.decay_rates,
#             self.raman_spectra,
#             t_early=t_early,
#         )

#     def compute_late_time_consistency_loss(
#         self, t1: float = 20.0, t2: float = 25.0
#     ) -> torch.Tensor:
#         """Penalize inconsistent late-time predictions."""
#         return losses.compute_late_time_consistency_loss(
#             self.fluorophore_bases,
#             self.abundances,
#             self.decay_rates,
#             self.raman_spectra,
#             t1=t1,
#             t2=t2,
#         )

#     def compute_raman_floor_loss(self) -> torch.Tensor:
#         """Penalize elevated Raman baseline."""
#         return losses.compute_raman_floor_loss(self.raman_spectra)

#     def compute_raman_spikiness_loss(self) -> torch.Tensor:
#         """Penalize overly smooth Raman spectrum."""
#         return losses.compute_raman_spikiness_loss(self.raman_spectra)

#     def compute_raman_curvature_loss(self) -> torch.Tensor:
#         """Penalize banana-shaped Raman spectrum."""
#         return losses.compute_raman_curvature_loss(self.raman_spectra)

#     def compute_fluorophore_convexity_loss(self) -> torch.Tensor:
#         """Penalize U-shaped fluorophore bases."""
#         return losses.compute_fluorophore_convexity_loss(self.fluorophore_bases)

#     def compute_decay_rate_prior_loss(self, target_rates: torch.Tensor) -> torch.Tensor:
#         """Soft regularization toward estimated decay rates."""
#         return losses.compute_decay_rate_prior_loss(self.decay_rates, target_rates)

#     def compute_extrapolation_validation_loss(
#         self,
#         data: torch.Tensor,
#         first_times: int,
#         fit_frames: int = 10,
#         val_frames: int = 10,
#     ) -> torch.Tensor:
#         """Internal cross-validation within training window."""
#         reconstruction = self.forward()
#         return losses.compute_extrapolation_validation_loss(
#             reconstruction,
#             data,
#             first_times,
#             fit_frames=fit_frames,
#             val_frames=val_frames,
#         )


# def get_default_loss_weights() -> Dict[str, float]:
#     """Get default weights for physics-based loss functions."""
#     return {
#         'spectral_separation': 1.0,
#         'abundance': 0.1,
#         'decay_diversity': 0.5,
#         'intensity_ratio': 0.5,
#         'late_time_consistency': 0.2,
#         'raman_floor': 0.3,
#         'raman_spikiness': 0.4,
#         'raman_curvature': 0.3,
#         'fluorophore_convexity': 0.5,
#         'decay_rate_prior': 0.1,
#         'extrapolation_validation': 0.2,
#     }


def fit_physics_model(
    data: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    n_fluorophores: int = 3,
    n_epochs: int = 5000,
    lr: float = 0.01,
    first_times: Optional[int] = None,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # use_physics_losses: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
    target_decay_rates: Optional[np.ndarray] = None,
    **model_kwargs,
) -> Tuple["PhysicsDecomposition", Dict]:
    """
    Fit PhysicsDecomposition model to time-series spectral data.

    Parameters
    ----------
    data : np.ndarray
        Time series, shape (n_timepoints, n_wavenumbers)
    time_values : np.ndarray, optional
        Time values in seconds
    n_fluorophores : int
        Number of decay components
    n_epochs : int
        Optimization iterations
    lr : float
        Learning rate
    first_times : int, optional
        Number of frames to use. If None, uses all.
    verbose : bool
        Print progress
    device : str
        'cuda' or 'cpu'
    use_physics_losses : bool
        Enable advanced physics-based regularization (default: False)
    loss_weights : dict, optional
        Custom weights for physics losses. If None, uses defaults.
        Keys: 'spectral_separation', 'abundance', 'decay_diversity',
              'intensity_ratio', 'late_time_consistency', 'raman_floor',
              'raman_spikiness', 'raman_curvature', 'fluorophore_convexity',
              'decay_rate_prior', 'extrapolation_validation'
    target_decay_rates : np.ndarray, optional
        Target decay rates for prior loss (from estimate_decay_rates_from_early_frames)
    **model_kwargs
        Passed to PhysicsDecomposition

    Returns
    -------
    model : PhysicsDecomposition
    history : dict
        Training history with 'loss', 'mse', and optionally physics loss keys
    """
    n_timepoints, n_wavenumbers = data.shape
    if first_times is None:
        first_times = n_timepoints

    # Normalize data
    scale_factor = np.max(data)
    data_norm = data / scale_factor

    data_tensor = torch.tensor(data_norm, dtype=torch.float32, device=device)

    if time_values is None:
        time_values = np.arange(n_timepoints, dtype=np.float32)
    time_tensor = torch.tensor(time_values, dtype=torch.float32, device=device)

    model = PhysicsDecomposition(
        n_wavenumbers=n_wavenumbers,
        n_timepoints=n_timepoints,
        n_fluorophores=n_fluorophores,
        time_values=time_tensor,
        device=device,
        **model_kwargs,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Initialize history
    history = {"loss": [], "mse": []}
    # if use_physics_losses:
    #     weights = loss_weights or get_default_loss_weights()
    #     # Add keys for each physics loss
    #     for key in weights.keys():
    #         history[key] = []

    #     # Convert target decay rates to tensor if provided
    #     target_rates_tensor = None
    #     if target_decay_rates is not None:
    #         target_rates_tensor = torch.tensor(
    #             target_decay_rates, dtype=torch.float32, device=device
    #         )

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        reconstruction = model()

        # Main MSE loss
        mse_loss = torch.mean(
            (reconstruction[:first_times] - data_tensor[:first_times]) ** 2
        )
        total_loss = mse_loss

        # # Add physics losses if enabled
        # if use_physics_losses:
        #     physics_loss_values = {}

        #     # Spectral separation
        #     loss_val = model.compute_spectral_separation_loss()
        #     physics_loss_values['spectral_separation'] = loss_val.item()
        #     total_loss += weights['spectral_separation'] * loss_val

        #     # Abundance penalty
        #     loss_val = model.compute_abundance_penalty()
        #     physics_loss_values['abundance'] = loss_val.item()
        #     total_loss += weights['abundance'] * loss_val

        #     # Decay diversity
        #     loss_val = model.compute_decay_diversity_penalty()
        #     physics_loss_values['decay_diversity'] = loss_val.item()
        #     total_loss += weights['decay_diversity'] * loss_val

        #     # Intensity ratio
        #     loss_val = model.compute_intensity_ratio_loss()
        #     physics_loss_values['intensity_ratio'] = loss_val.item()
        #     total_loss += weights['intensity_ratio'] * loss_val

        #     # Late time consistency
        #     loss_val = model.compute_late_time_consistency_loss()
        #     physics_loss_values['late_time_consistency'] = loss_val.item()
        #     total_loss += weights['late_time_consistency'] * loss_val

        #     # Raman floor
        #     loss_val = model.compute_raman_floor_loss()
        #     physics_loss_values['raman_floor'] = loss_val.item()
        #     total_loss += weights['raman_floor'] * loss_val

        #     # Raman spikiness
        #     loss_val = model.compute_raman_spikiness_loss()
        #     physics_loss_values['raman_spikiness'] = loss_val.item()
        #     total_loss += weights['raman_spikiness'] * loss_val

        #     # Raman curvature
        #     loss_val = model.compute_raman_curvature_loss()
        #     physics_loss_values['raman_curvature'] = loss_val.item()
        #     total_loss += weights['raman_curvature'] * loss_val

        #     # Fluorophore convexity
        #     loss_val = model.compute_fluorophore_convexity_loss()
        #     physics_loss_values['fluorophore_convexity'] = loss_val.item()
        #     total_loss += weights['fluorophore_convexity'] * loss_val

        #     # Decay rate prior (if target rates provided)
        #     if target_rates_tensor is not None:
        #         loss_val = model.compute_decay_rate_prior_loss(target_rates_tensor)
        #         physics_loss_values['decay_rate_prior'] = loss_val.item()
        #         total_loss += weights['decay_rate_prior'] * loss_val

        #     # Extrapolation validation
        #     loss_val = model.compute_extrapolation_validation_loss(
        #         data_tensor, first_times
        #     )
        #     physics_loss_values['extrapolation_validation'] = loss_val.item()
        #     total_loss += weights['extrapolation_validation'] * loss_val

        #     # Record all losses
        #     for key, val in physics_loss_values.items():
        #         history[key].append(val)

        history["mse"].append(mse_loss.item())
        history["loss"].append(total_loss.item())

        total_loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 1000 == 0:
            tau = (1.0 / model.decay_rates).detach().cpu().numpy()
            # if use_physics_losses:
            #     print(f"Epoch {epoch + 1}/{n_epochs}: total_loss={total_loss.item():.6f}, mse={mse_loss.item():.6f}, τ={tau}")
            # else:
            print(f"Epoch {epoch + 1}/{n_epochs}: loss={mse_loss.item():.6f}, τ={tau}")

    # Rescale back
    with torch.no_grad():
        log_scale = np.log(scale_factor)
        model.raman_spectrum.add_(log_scale)
        model.abundances_raw.add_(log_scale)

    return model, history
