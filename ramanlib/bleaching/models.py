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
import torch.nn.functional as F
import torch.optim as optim

from ramanlib.bleaching.decompose import DecompositionResult
from ramanlib.bleaching.physics import (
    fit_polynomial_bases,
    build_vandermonde_torch,
    l2_normalize_torch,
    reconstruct_time_series_torch,
    evaluate_polynomial_bases_torch,
)
from ramanlib.core import SpectralData


class PhysicsDecomposition(nn.Module):
    """
    Physics-constrained decomposition model.

    Model: Y(ν, t) = s(ν) + Σₖ wₖ · Bₖ(ν) · exp(-λₖ · t)

    Parameters are stored in log-space for positivity constraints.
    """

    def __init__(
        self,
        data: torch.Tensor,
        time_values: torch.Tensor,
        wavenumber_axis: torch.Tensor,
        n_fluorophores: int = 3,
        basis_type: str = "polynomial",
        polynomial_degree: int = 3,
        min_decay_rate: float = 0.1,
        max_decay_rate: float = 10.0,
        initial_abundances: Optional[np.ndarray] = None,
        initial_rates: Optional[np.ndarray] = None,
        initial_bases: Optional[np.ndarray] = None,
        initial_raman: Optional[np.ndarray] = None,
        initial_log_poly_coeffs: Optional[np.ndarray] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.n_fluorophores = n_fluorophores
        self.n_wavenumbers = len(wavenumber_axis)
        self.n_times = len(time_values)
        self.device = device
        self.basis_type = basis_type
        self.polynomial_degree = polynomial_degree
        self.min_decay_rate = min_decay_rate
        self.max_decay_rate = max_decay_rate

        wn_mu = wavenumber_axis.mean()
        wn_std = wavenumber_axis.std()

        wn_normalised = (wavenumber_axis - wn_mu) / (wn_std + 1e-8)

        # Register buffers so they are saved with the model state
        self.register_buffer("wavenumbers", wavenumber_axis.to(device))
        self.register_buffer("wn_normalised", wn_normalised.to(device))

        # Time axis
        self.register_buffer("time_values", time_values.to(device))

        # Raman spectrum (log-space: unbounded positive via exp)
        if initial_raman is not None:
            raman_tensor = self._to_tensor(initial_raman, device)
            self.raman = nn.Parameter(
                torch.log(raman_tensor + 1e-8),
                requires_grad=False,
            )
        else:
            self.raman = nn.Parameter(
                torch.randn(len(wavenumber_axis), device=device) * 0.1
            )

        # Fluorophore bases
        if basis_type == "free":
            if initial_bases is not None:
                bases_tensor = self._to_tensor(initial_bases, device)
                self.fluorophore_bases_raw = nn.Parameter(
                    torch.log(bases_tensor + 1e-8),
                    requires_grad=False
                )
            else:
                self.fluorophore_bases_raw = nn.Parameter(
                    torch.randn(
                        n_fluorophores, self.wavenumbers.shape[0], device=device
                    )
                    * 0.1
                )
        elif basis_type == "polynomial":
            n_coeffs = polynomial_degree + 1
            if initial_log_poly_coeffs is not None:
                log_poly_tensor = self._to_tensor(initial_log_poly_coeffs, device)
                self.log_poly_coeffs = nn.Parameter(
                    log_poly_tensor, requires_grad=False
                )
                # If coeffs provided without stats, compute stats from wavenumbers
                self.register_buffer(
                    "poly_norm_mean",
                    torch.tensor(wn_mu, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "poly_norm_std",
                    torch.tensor(wn_std, dtype=torch.float32, device=device),
                )
            elif initial_bases is not None:
                # Use RAW wavenumbers (fit_polynomial_bases handles normalization)
                wn_np = (
                    wavenumber_axis.cpu().numpy()
                    if isinstance(wavenumber_axis, torch.Tensor)
                    else wavenumber_axis
                )
                bases_np = (
                    initial_bases.cpu().numpy()
                    if isinstance(initial_bases, torch.Tensor)
                    else initial_bases
                )
                log_coeffs, wn_mean, wn_std = fit_polynomial_bases(
                    bases_np, wn_np, polynomial_degree
                )
                self.log_poly_coeffs = nn.Parameter(
                    torch.tensor(log_coeffs, dtype=torch.float32, device=device),
                    requires_grad=False,
                )
                # Store normalization stats as buffers
                self.register_buffer(
                    "poly_norm_mean",
                    torch.tensor(wn_mean, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "poly_norm_std",
                    torch.tensor(wn_std, dtype=torch.float32, device=device),
                )
            else:
                self.log_poly_coeffs = nn.Parameter(
                    torch.randn(n_fluorophores, n_coeffs, device=device) * 0.1
                )
                # Initialize stats from wavenumbers
                self.register_buffer(
                    "poly_norm_mean",
                    torch.tensor(wn_mu, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "poly_norm_std",
                    torch.tensor(wn_std, dtype=torch.float32, device=device),
                )

            # vandermonde = build_vandermonde_torch(wn_normalised, polynomial_degree)
            # self.register_buffer("vandermonde", vandermonde)
        else:
            raise ValueError(f"Unknown basis_type: {basis_type}")

        # Abundances (log-space: unbounded positive via exp)
        if initial_abundances is not None:
            abund_tensor = self._to_tensor(initial_abundances, device)
            self.abundances_raw = nn.Parameter(
                torch.log(abund_tensor + 1e-8),
                requires_grad=False,
            )
        else:
            self.abundances_raw = nn.Parameter(
                torch.zeros(n_fluorophores, device=device)
            )

        # Decay rates (sigmoid-space for bounded [min, max] range)
        if initial_rates is not None:
            rates_tensor = self._to_tensor(initial_rates, device)
            if torch.any(rates_tensor < min_decay_rate) or torch.any(rates_tensor > max_decay_rate):
                raise ValueError(
                    f"initial_rates must be in [{min_decay_rate}, {max_decay_rate}]"
                )
            # Convert rates to normalized [0, 1] then to logit space
            rate_range = max_decay_rate - min_decay_rate
            normalized = (rates_tensor - min_decay_rate) / rate_range
            # Inverse sigmoid (logit): logit(p) = log(p / (1 - p))
            # Clamp to avoid inf at boundaries
            normalized_clamped = torch.clamp(normalized, 1e-6, 1 - 1e-6)
            self.decay_rates_raw = nn.Parameter(
                torch.log(normalized_clamped / (1 - normalized_clamped)),
                requires_grad=False,
            )
        else:
            # Initialize in logit space near center (logit(0.5) = 0)
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
            # Use RAW wavenumbers with stored normalization stats
            bases = evaluate_polynomial_bases_torch(
                self.log_poly_coeffs,
                self.wavenumbers,
                wn_mean=float(self.poly_norm_mean.item()),
                wn_std=float(self.poly_norm_std.item()),
            )

        # Ensure amplitude ambiguity is resolved via Normalization
        return l2_normalize_torch(bases, dim=-1)

    @property
    def raman_spectrum(self) -> torch.Tensor:
        """Raman spectrum - unbounded positive via exp."""
        return torch.exp(self.raman)

    @property
    def abundances(self) -> torch.Tensor:
        """Abundances - unbounded positive via exp."""
        return torch.exp(self.abundances_raw)

    @property
    def decay_rates(self) -> torch.Tensor:
        """Decay rates - bounded in [min, max] via sigmoid."""
        rate_range = self.max_decay_rate - self.min_decay_rate
        normalized = torch.sigmoid(self.decay_rates_raw)  # → [0, 1]
        return self.min_decay_rate + rate_range * normalized

    def forward(self) -> torch.Tensor:
        """Reconstruct time series from parameters."""

        # Add batch dimension for reconstruction function (expects [B, W])

        # Add batch dim to raman, decay rates, abundances
        raman_spectrum = self.raman_spectrum.unsqueeze(0)
        abundances = self.abundances.unsqueeze(0)
        decay_rates = self.decay_rates.unsqueeze(0)

        # print("Shapes of parameters:")
        # print(f"Raman: {raman_spectrum.shape}")
        # print(f"Bases: {self.fluorophore_bases.shape}")
        # print(f"Abundances: {abundances.shape}")
        # print(f"Decay rates: {decay_rates.shape}")

        return (
            reconstruct_time_series_torch(
                raman_spectrum,
                self.fluorophore_bases,
                abundances,
                decay_rates,
                self.time_values,
            )
            .squeeze(0)
            .T  # Remove batch dim, transpose to [T, W]
        )

    def get_decomposition(self) -> DecompositionResult:
        """Return components as numpy arrays, sorted by decay rate."""
        with torch.no_grad():
            rates = self.decay_rates.cpu().numpy()
            sort_idx = np.argsort(rates)[::-1]

            result = DecompositionResult(
                raman=SpectralData(
                    self.raman_spectrum.cpu().numpy(), self.wavenumbers.cpu().numpy()
                ),
                rates=rates[sort_idx],
                fluorophore_spectra=SpectralData(
                    self.fluorophore_bases.cpu().numpy()[sort_idx],
                    self.wavenumbers.cpu().numpy(),
                ),
                abundances=self.abundances.cpu().numpy()[sort_idx],
            )
            if self.basis_type == "polynomial":
                result.log_polynomial_coeffs = self.log_poly_coeffs.cpu().numpy()[
                    sort_idx
                ]
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
    data: SpectralData,
    n_fluorophores: int = 3,
    n_epochs: int = 5000,
    lr: float = 0.01,
    first_times: Optional[int] = None,
    min_decay_rate: float = 0.1,
    max_decay_rate: float = 10.0,
    initial_rates: Optional[np.ndarray] = None,
    initial_bases: Optional[np.ndarray] = None,
    initial_raman: Optional[np.ndarray] = None,
    initial_log_poly_coeffs: Optional[np.ndarray] = None,
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
    n_timepoints, n_wavenumbers = data.intensities.shape
    if first_times is None:
        first_times = n_timepoints

    # Normalize data
    # scale_factor = np.max(data.intensities)
    # data_norm = data / scale_factor

    data_tensor = torch.tensor(
        data.intensities[:first_times, :], dtype=torch.float32, device=device
    )
    print(f"Data tensor shape: {data_tensor.shape}")

    time_tensor = torch.tensor(
        data.time_values[:first_times], dtype=torch.float32, device=device
    )
    wn_tensor = (
        torch.tensor(data.wavenumbers, dtype=torch.float32, device=device)
        if data.wavenumbers is not None
        else None
    )
    print(
        f"Shapes: data {data_tensor.shape}, time {time_tensor.shape}, wn {wn_tensor.shape}"
    )

    model = PhysicsDecomposition(
        data=data_tensor,
        time_values=time_tensor,
        wavenumber_axis=wn_tensor,
        n_fluorophores=n_fluorophores,
        min_decay_rate=min_decay_rate,
        max_decay_rate=max_decay_rate,
        initial_rates=initial_rates,
        initial_bases=initial_bases,
        initial_raman=initial_raman,
        initial_log_poly_coeffs=initial_log_poly_coeffs,
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
        # print(f"Reconstruction shape: {reconstruction.shape}")
        # Main MSE loss
        mse_loss = torch.mean(
            (reconstruction[:first_times] - data_tensor[:first_times]) ** 2
        )
        # print(reconstruction.mean())
        # print(data_tensor.mean())

        # print(mse_loss)

        total_loss = mse_loss
        # print(f"Total loss so far: {total_loss.item()}")
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
            print(f"Epoch {epoch + 1}/{n_epochs}: loss={mse_loss.item():.3e}, τ={tau}")

    # # Rescale back
    # with torch.no_grad():
    #     log_scale = np.log(scale_factor)
    #     model.raman_spectrum.add_(log_scale)
    #     model.abundances_raw.add_(log_scale)

    return model, history
