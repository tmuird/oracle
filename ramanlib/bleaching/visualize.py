def visualize_data_3d(
    data: np.ndarray,
    time_values: 'Optional[np.ndarray]' = None,
    wavenumbers: 'Optional[np.ndarray]' = None,
    subsample_wn: int = 2,
    subsample_time: int = 1,
    title: str = "3D Dataset Visualization",
):
    """
    3D visualization of a raw data sample using plotly.

    Args:
        data: Array of shape (n_timepoints, n_wavenumbers)
        time_values: Optional time axis (default: frame indices)
        wavenumbers: Optional wavenumber axis (default: indices)
        subsample_wn: Subsample factor for wavenumber axis
        subsample_time: Subsample factor for time axis
        title: Plot title

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    n_t, n_wn = data.shape
    if time_values is None:
        time_values = np.arange(n_t, dtype=np.float32)
    if wavenumbers is None:
        wavenumbers = np.arange(n_wn)
    wn_idx = np.arange(0, n_wn, subsample_wn)
    t_idx = np.arange(0, n_t, subsample_time)
    wn_sub = wavenumbers[wn_idx]
    t_sub = time_values[t_idx]
    fig = go.Figure(
        data=[
            go.Surface(
                x=wn_sub,
                y=t_sub,
                z=data[np.ix_(t_idx, wn_idx)],
                colorscale="Viridis",
                colorbar=dict(title="Intensity", x=1.02),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Time (s)",
            zaxis_title="Intensity",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        title=title,
        width=900,
        height=700,
    )
    return fig
"""
Visualization utilities for bleaching datasets.

Provides functions to plot decomposition results and time-series data.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Optional, Tuple, Dict
from matplotlib.figure import Figure


def visualise_decomposition(
    data: np.ndarray,
    decomposition: Dict[str, np.ndarray],
    reconstruction: Optional[np.ndarray] = None,
    time_values: Optional[np.ndarray] = None,
    wavenumbers: Optional[np.ndarray] = None,
    reference_raman: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Visualize spectral decomposition results.

    Accepts decomposition results as a dictionary, making it compatible with
    both PhysicsDecomposition.get_decomposition() and the DE-based decompose().

    Args:
        data: Original time series (n_timepoints, n_wavenumbers)
        decomposition: Dictionary with keys:
            - 'raman': Extracted Raman spectrum (n_wavenumbers,)
            - 'fluorophore_bases': Fluorophore bases (n_fluorophores, n_wavenumbers)
            - 'abundances': Abundances (n_fluorophores,)
            - 'rates' or 'decay_rates': Decay rates (n_fluorophores,)
        reconstruction: Reconstructed time series (n_timepoints, n_wavenumbers).
            If None, computed from decomposition parameters.
        time_values: Time axis in seconds. If None, uses frame indices.
        wavenumbers: Wavenumber axis. If None, uses indices.
        reference_raman: Ground truth Raman for comparison. If None, uses
            last 20 frames average.
        figsize: Figure size tuple.

    Returns:
        Tuple of (figure, axes)
    """
    n_t, n_wn = data.shape

    # Handle optional axes
    if time_values is None:
        time_values = np.arange(n_t, dtype=np.float32)
    if wavenumbers is None:
        wavenumbers = np.arange(n_wn)

    # Extract decomposition parameters
    raman = decomposition["raman"]
    bases = decomposition.get("fluorophore_bases", decomposition.get("bases"))
    abundances = decomposition["abundances"]
    rates = decomposition.get("rates", decomposition.get("decay_rates"))
    time_constants = 1.0 / rates

    n_fluorophores = len(rates)

    # Compute reconstruction if not provided
    if reconstruction is None:
        reconstruction = np.tile(raman, (n_t, 1))
        for i in range(n_fluorophores):
            decay = np.exp(-rates[i] * time_values)
            reconstruction = (
                reconstruction + abundances[i] * decay[:, None] * bases[i, None, :]
            )

    # Reference Raman
    if reference_raman is None:
        reference_raman = data[-20:].mean(axis=0)
        ref_label = "Reference (last 20 frames avg)"
    else:
        ref_label = "Ground Truth Raman"

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Plot 1: Original time series
    ax = axes[0, 0]
    n_show = min(8, n_t)
    cmap = plt.cm.viridis
    for i, idx in enumerate(np.linspace(0, n_t - 1, n_show, dtype=int)):
        ax.plot(wavenumbers, data[idx], color=cmap(i / n_show), alpha=0.7)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Original Time Series")
    ax.grid(True, alpha=0.3)

    # Plot 2: Extracted Raman vs reference
    ax = axes[0, 1]
    ax.plot(wavenumbers, raman, "b-", linewidth=2, label="Extracted Raman")
    ax.plot(wavenumbers, reference_raman, "r--", alpha=0.7, label=ref_label)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Extracted Raman Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Fluorophore bases
    ax = axes[0, 2]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    if bases is not None:
        for i in range(n_fluorophores):
            tau = time_constants[i]
            ax.plot(
                wavenumbers,
                bases[i],
                color=colors[i % len(colors)],
                label=f"B{i + 1} (τ={tau:.3f}s)",
            )
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title("Fluorophore Basis Spectra")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Decay components over time
    ax = axes[1, 0]
    total_fluor = np.zeros(n_t)
    if bases is not None:
        for i in range(n_fluorophores):
            decay = np.exp(-rates[i] * time_values)
            amplitude = abundances[i] * decay * bases[i].mean()
            total_fluor += amplitude
            tau = time_constants[i]
            ax.plot(
                time_values,
                amplitude,
                colors[i % len(colors)],
                label=f"τ={tau:.3f}s, w={abundances[i]:.1f}",
            )

    ax.plot(time_values, total_fluor, "k--", linewidth=2, label="Total Predicted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Fluorescence")
    ax.set_title("Decay Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Reconstruction quality (first frame)
    ax = axes[1, 1]
    ax.plot(wavenumbers, data[0], "b-", alpha=0.7, label="Original (t=0)")
    ax.plot(wavenumbers, reconstruction[0], "r--", alpha=0.7, label="Reconstructed")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Reconstruction (First Frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Residual over time
    ax = axes[1, 2]
    residuals = data - reconstruction
    ax.plot(time_values, np.mean(residuals, axis=1), "k-", label="Mean")
    ax.fill_between(
        time_values,
        np.mean(residuals, axis=1) - np.std(residuals, axis=1),
        np.mean(residuals, axis=1) + np.std(residuals, axis=1),
        alpha=0.3,
    )
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    mse = np.mean(residuals**2)
    r2 = 1 - np.sum(residuals**2) / np.sum((data - data.mean()) ** 2)
    corr = np.corrcoef(raman, reference_raman)[0, 1]
    print(f"\nReconstruction MSE: {mse:.6f}, R²: {r2:.6f}")
    print(f"Raman correlation with reference: {corr:.4f}")
    print(f"Time constants (τ): {time_constants}")
    print(f"Abundances (w): {abundances}")

    return fig, axes


try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def get_fluorophore_contribution(
    ds: xr.Dataset,
    sample_idx: int,
    fluorophore_idx: int,
    time_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Compute contribution of a single fluorophore at a given time.

    F_i(ν,t) = wᵢ · Bᵢ(ν) · exp(-λᵢ · t)

    Parameters
    ----------
    ds : xr.Dataset
        Synthetic dataset with ground truth parameters
    sample_idx : int
        Sample index
    fluorophore_idx : int
        Fluorophore index
    time_seconds : float, optional
        Time in seconds. If None, returns at t=0.

    Returns
    -------
    np.ndarray
        Fluorophore contribution spectrum
    """
    w_i = (
        ds["abundances_gt"].isel(sample=sample_idx, fluorophore=fluorophore_idx).values
    )
    λ_i = (
        ds["decay_rates_gt"].isel(sample=sample_idx, fluorophore=fluorophore_idx).values
    )

    if "sample" in ds["fluorophore_bases_gt"].dims:
        B_i = (
            ds["fluorophore_bases_gt"]
            .isel(sample=sample_idx, fluorophore=fluorophore_idx)
            .values
        )
    else:
        B_i = ds["fluorophore_bases_gt"].isel(fluorophore=fluorophore_idx).values

    if time_seconds is None:
        time_seconds = 0.0
    decay = np.exp(-λ_i * time_seconds)

    return w_i * B_i * decay


def get_total_fluorescence(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
) -> np.ndarray:
    """
    Compute total fluorescence at a given time.

    F(ν,t) = Σᵢ wᵢ · Bᵢ(ν) · exp(-λᵢ · t)
    """
    n_fluorophores = len(ds["fluorophore"])

    if ds["wavenumber"].ndim == 2:
        n_wn = ds["wavenumber"].isel(sample=sample_idx).shape[0]
    else:
        n_wn = len(ds["wavenumber"])

    total = np.zeros(n_wn)
    for i in range(n_fluorophores):
        total += get_fluorophore_contribution(ds, sample_idx, i, time_seconds)

    return total


def get_full_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
) -> Dict:
    """
    Get all components of the decomposition at a given time.

    Returns
    -------
    dict
        Keys: raman, fluorophore_0, fluorophore_1, ..., total_fluorescence,
        reconstructed, observed_clean, observed_noisy, wavenumbers,
        decay_rates, abundances, time_constants
    """
    n_fluorophores = len(ds["fluorophore"])

    if ds["wavenumber"].ndim == 2:
        wavenumbers = ds["wavenumber"].isel(sample=sample_idx).values
    else:
        wavenumbers = ds["wavenumber"].values

    time_values = ds["bleaching_time"].values
    time_idx = np.argmin(np.abs(time_values - time_seconds))
    actual_time = time_values[time_idx]

    raman = ds["raman_gt"].isel(sample=sample_idx).values

    fluorophores = {}
    total_fluor = np.zeros_like(raman)

    for i in range(n_fluorophores):
        contrib = get_fluorophore_contribution(ds, sample_idx, i, actual_time)
        fluorophores[f"fluorophore_{i}"] = contrib
        total_fluor += contrib

    reconstructed = raman + total_fluor

    observed_clean = (
        ds["intensity_clean"].isel(sample=sample_idx, bleaching_time=time_idx).values
    )
    observed_noisy = (
        ds["intensity_raw"].isel(sample=sample_idx, bleaching_time=time_idx).values
    )

    decay_rates = ds["decay_rates_gt"].isel(sample=sample_idx).values
    abundances = ds["abundances_gt"].isel(sample=sample_idx).values

    result = {
        "raman": raman,
        "total_fluorescence": total_fluor,
        "reconstructed": reconstructed,
        "observed_clean": observed_clean,
        "observed_noisy": observed_noisy,
        "wavenumbers": wavenumbers,
        "time_seconds": actual_time,
        "decay_rates": decay_rates,
        "abundances": abundances,
        "time_constants": 1.0 / decay_rates,
    }
    result.update(fluorophores)

    return result


def plot_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
    figsize: Tuple[int, int] = (14, 10),
    show_noisy: bool = True,
) -> Figure:
    """
    Plot full decomposition for a single sample at a given time.

    Shows:
    - Top: Full spectrum with components
    - Bottom left: Individual fluorophore contributions
    - Bottom right: Residual
    """
    decomp = get_full_decomposition(ds, sample_idx, time_seconds)
    wn = decomp["wavenumbers"]
    n_fluorophores = len(ds["fluorophore"])

    fig = plt.figure(figsize=figsize)

    # Top panel: Full decomposition
    ax1 = fig.add_subplot(2, 2, (1, 2))

    if show_noisy:
        ax1.plot(
            wn,
            decomp["observed_noisy"],
            "gray",
            alpha=0.5,
            label="Observed (noisy)",
            linewidth=0.5,
        )

    ax1.plot(
        wn,
        decomp["observed_clean"],
        "k-",
        alpha=0.8,
        label="Observed (clean)",
        linewidth=1.5,
    )
    ax1.plot(
        wn,
        decomp["reconstructed"],
        "r--",
        alpha=0.8,
        label="Reconstructed",
        linewidth=1.5,
    )
    ax1.plot(wn, decomp["raman"], "b-", alpha=0.7, label="Raman (GT)", linewidth=1.5)
    ax1.plot(
        wn,
        decomp["total_fluorescence"],
        "orange",
        alpha=0.7,
        label="Total Fluorescence",
        linewidth=1.5,
    )

    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Intensity")
    ax1.set_title(f'Sample {sample_idx} at t = {decomp["time_seconds"]:.2f}s')
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom left: Individual fluorophores
    ax2 = fig.add_subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_fluorophores))

    for i in range(n_fluorophores):
        τ = decomp["time_constants"][i]
        w = decomp["abundances"][i]
        ax2.plot(
            wn,
            decomp[f"fluorophore_{i}"],
            color=colors[i],
            label=f"F{i+1}: τ={τ:.3f}s, w={w:.1f}",
            linewidth=1.5,
        )

    ax2.set_xlabel("Wavenumber (cm⁻¹)")
    ax2.set_ylabel("Intensity")
    ax2.set_title("Individual Fluorophore Contributions")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Bottom right: Residual
    ax3 = fig.add_subplot(2, 2, 4)
    residual = decomp["observed_clean"] - decomp["reconstructed"]
    ax3.plot(wn, residual, "k-", linewidth=1)
    ax3.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax3.fill_between(wn, residual, 0, alpha=0.3)

    rmse = np.sqrt(np.mean(residual**2))
    ax3.set_xlabel("Wavenumber (cm⁻¹)")
    ax3.set_ylabel("Residual")
    ax3.set_title(f"Residual (RMSE = {rmse:.4f})")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_temporal_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """
    Plot decomposition across all time points for a single sample.
    """
    time_values = ds["bleaching_time"].values

    if ds["wavenumber"].ndim == 2:
        wn = ds["wavenumber"].isel(sample=sample_idx).values
    else:
        wn = ds["wavenumber"].values

    n_times = len(time_values)
    n_fluorophores = len(ds["fluorophore"])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    time_colors = plt.cm.plasma(np.linspace(0, 0.9, n_times))
    fluor_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_fluorophores))

    # Top left: Observed spectra over time
    ax = axes[0, 0]
    for t_idx, t in enumerate(time_values):
        spectrum = (
            ds["intensity_clean"].isel(sample=sample_idx, bleaching_time=t_idx).values
        )
        ax.plot(wn, spectrum, color=time_colors[t_idx], alpha=0.8, label=f"t={t:.2f}s")

    ax.plot(
        wn,
        ds["raman_gt"].isel(sample=sample_idx).values,
        "k--",
        linewidth=2,
        label="Raman (GT)",
    )
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Observed Spectra Over Time")
    ax.grid(True, alpha=0.3)

    # Top right: Total fluorescence over time
    ax = axes[0, 1]
    for t_idx, t in enumerate(time_values):
        fluor = get_total_fluorescence(ds, sample_idx, t)
        ax.plot(wn, fluor, color=time_colors[t_idx], alpha=0.8, label=f"t={t:.2f}s")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Total Fluorescence Decay")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Individual fluorophore decay curves
    ax = axes[1, 0]
    decay_rates = ds["decay_rates_gt"].isel(sample=sample_idx).values
    abundances = ds["abundances_gt"].isel(sample=sample_idx).values

    for i in range(n_fluorophores):
        if "sample" in ds["fluorophore_bases_gt"].dims:
            B_i = (
                ds["fluorophore_bases_gt"].isel(sample=sample_idx, fluorophore=i).values
            )
        else:
            B_i = ds["fluorophore_bases_gt"].isel(fluorophore=i).values

        intensities = []
        for t in time_values:
            contrib = abundances[i] * B_i * np.exp(-decay_rates[i] * t)
            intensities.append(contrib.mean())

        τ = 1.0 / decay_rates[i]
        ax.plot(
            time_values,
            intensities,
            "o-",
            color=fluor_colors[i],
            label=f"F{i+1}: τ={τ:.3f}s",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Fluorescence Intensity")
    ax.set_title("Fluorophore Decay Curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Bottom right: Basis spectra
    ax = axes[1, 1]
    for i in range(n_fluorophores):
        if "sample" in ds["fluorophore_bases_gt"].dims:
            B_i = (
                ds["fluorophore_bases_gt"].isel(sample=sample_idx, fluorophore=i).values
            )
        else:
            B_i = ds["fluorophore_bases_gt"].isel(fluorophore=i).values

        τ = 1.0 / decay_rates[i]
        ax.plot(
            wn, B_i, color=fluor_colors[i], linewidth=1.5, label=f"B{i+1} (τ={τ:.3f}s)"
        )

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title("Fluorophore Basis Spectra")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sample {sample_idx} - Temporal Decomposition", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def visualize_decomposition_3d(
    data: np.ndarray,
    decomposition: dict,
    reconstruction: "Optional[np.ndarray]" = None,
    time_values: "Optional[np.ndarray]" = None,
    wavenumbers: "Optional[np.ndarray]" = None,
    subsample_wn: int = 2,
    subsample_time: int = 1,
):
    """
    Interactive 3D visualisation using plotly (allows rotation/zoom).

    Args:
        data: Original time series (n_timepoints, n_wavenumbers)
        decomposition: Dictionary with keys:
            - 'raman': Extracted Raman spectrum (n_wavenumbers,)
            - 'fluorophore_bases': Fluorophore bases (n_fluorophores, n_wavenumbers)
            - 'abundances': Abundances (n_fluorophores,)
            - 'rates' or 'decay_rates': Decay rates (n_fluorophores,)
        reconstruction: Reconstructed time series. If None, computed from decomposition.
        time_values: Time axis in seconds. If None, uses frame indices.
        wavenumbers: Wavenumber axis. If None, uses indices.
        subsample_wn: Subsample factor for wavenumber axis
        subsample_time: Subsample factor for time axis

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go

    n_t, n_wn = data.shape

    # Handle optional axes
    if time_values is None:
        time_values = np.arange(n_t, dtype=np.float32)
    if wavenumbers is None:
        wavenumbers = np.arange(n_wn)

    # Extract decomposition parameters
    raman = decomposition["raman"]
    bases = decomposition.get("fluorophore_bases", decomposition.get("bases"))
    abundances = decomposition["abundances"]
    rates = decomposition.get("rates", decomposition.get("decay_rates"))
    n_fluorophores = len(rates)

    # Compute reconstruction if not provided
    if reconstruction is None:
        reconstruction = np.tile(raman, (n_t, 1))
        for i in range(n_fluorophores):
            decay = np.exp(-rates[i] * time_values)
            reconstruction = (
                reconstruction + abundances[i] * decay[:, None] * bases[i, None, :]
            )

    # Compute total fluorescence
    total_fluor = np.zeros_like(data)
    if bases is not None:
        for i in range(n_fluorophores):
            decay = np.exp(-rates[i] * time_values)
            total_fluor += abundances[i] * decay[:, None] * bases[i, None, :]

    # Subsample for performance
    wn_idx = np.arange(0, n_wn, subsample_wn)
    t_idx = np.arange(0, n_t, subsample_time)

    wn_sub = wavenumbers[wn_idx]
    t_sub = time_values[t_idx]

    # Create figure with dropdown to select which surface to view
    fig = go.Figure()

    # Original data
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=data[np.ix_(t_idx, wn_idx)],
            colorscale="Viridis",
            name="Original",
            visible=True,
            colorbar=dict(title="Intensity", x=1.02),
        )
    )

    # Reconstruction
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=reconstruction[np.ix_(t_idx, wn_idx)],
            colorscale="Viridis",
            name="Reconstructed",
            visible=False,
        )
    )

    # Residual
    residual = data - reconstruction
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=residual[np.ix_(t_idx, wn_idx)],
            colorscale="RdBu",
            name="Residual",
            visible=False,
            cmid=0,
        )
    )

    # Total fluorescence
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=total_fluor[np.ix_(t_idx, wn_idx)],
            colorscale="Oranges",
            name="Fluorescence",
            visible=False,
        )
    )

    # Raman (constant surface)
    raman_surface = np.tile(raman[wn_idx], (len(t_idx), 1))
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=raman_surface,
            colorscale="Blues",
            name="Predicted Raman",
            visible=False,
        )
    )

    # Create dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(
                        label="Original",
                        method="update",
                        args=[{"visible": [True, False, False, False, False]}],
                    ),
                    dict(
                        label="Reconstructed",
                        method="update",
                        args=[{"visible": [False, True, False, False, False]}],
                    ),
                    dict(
                        label="Residual",
                        method="update",
                        args=[{"visible": [False, False, True, False, False]}],
                    ),
                    dict(
                        label="Fluorescence",
                        method="update",
                        args=[{"visible": [False, False, False, True, False]}],
                    ),
                    dict(
                        label="Predicted Raman",
                        method="update",
                        args=[{"visible": [False, False, False, False, True]}],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
            )
        ],
        scene=dict(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Time (s)",
            zaxis_title="Intensity",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        title="3D Spectral Decomposition (use dropdown to switch views)",
        width=900,
        height=700,
    )

    return fig
