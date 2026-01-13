"""
Visualization utilities for synthetic fluorescence bleaching datasets.

Updated for compatibility with the new flogen.py dataset structure:
- Uses 'bleaching_time' dimension instead of 'integration_time'
- Uses 'bleaching_time_seconds' coordinate instead of 'time_seconds'
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Optional, Tuple
from matplotlib.figure import Figure

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
    Compute the contribution of a single fluorophore at a given time.

    F_i(ν,t) = w_i · B_i(ν) · exp(-λ_i · t)

    Parameters
    ----------
    ds : xr.Dataset
        Synthetic dataset
    sample_idx : int
        Sample index
    fluorophore_idx : int
        Fluorophore index (0, 1, 2, ...)
    time_seconds : float, optional
        Time in seconds. If None, returns at t=0 (no decay)

    Returns
    -------
    np.ndarray
        Fluorophore contribution spectrum (n_wavenumbers,)
    """
    # Get parameters
    w_i = ds['abundances_gt'].isel(sample=sample_idx, fluorophore=fluorophore_idx).values
    λ_i = ds['decay_rates_gt'].isel(sample=sample_idx, fluorophore=fluorophore_idx).values

    # Get basis (shared or per-sample)
    if 'sample' in ds['fluorophore_bases_gt'].dims:
        B_i = ds['fluorophore_bases_gt'].isel(sample=sample_idx, fluorophore=fluorophore_idx).values
    else:
        B_i = ds['fluorophore_bases_gt'].isel(fluorophore=fluorophore_idx).values

    # Get wavenumbers for this sample (per-sample or shared)
    if ds['wavenumber'].ndim == 2:
        # Per-sample wavenumber axes
        wn_sample = ds['wavenumber'].isel(sample=sample_idx).values
        # Interpolate basis to this sample's wavenumbers if needed
        # For now, assume they match

    # Compute decay
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
    n_fluorophores = len(ds['fluorophore'])

    # Get wavenumber axis length for this sample
    if ds['wavenumber'].ndim == 2:
        n_wn = ds['wavenumber'].isel(sample=sample_idx).shape[0]
    else:
        n_wn = len(ds['wavenumber'])

    total = np.zeros(n_wn)

    for i in range(n_fluorophores):
        total += get_fluorophore_contribution(ds, sample_idx, i, time_seconds)

    return total


def get_full_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
) -> dict:
    """
    Get all components of the decomposition at a given time.

    Returns
    -------
    dict with keys:
        - 'raman': s(ν)
        - 'fluorophore_0', 'fluorophore_1', ...: individual F_i(ν,t)
        - 'total_fluorescence': Σ F_i(ν,t)
        - 'reconstructed': s(ν) + Σ F_i(ν,t)
        - 'observed_clean': from dataset
        - 'observed_noisy': from dataset
        - 'wavenumbers': ν axis (per-sample)
        - 'decay_rates': λ values
        - 'abundances': w values
        - 'time_constants': τ = 1/λ
    """
    n_fluorophores = len(ds['fluorophore'])

    # Get wavenumbers for this sample (per-sample or shared)
    if ds['wavenumber'].ndim == 2:
        wavenumbers = ds['wavenumber'].isel(sample=sample_idx).values
    else:
        wavenumbers = ds['wavenumber'].values

    # Get time index (UPDATED: use bleaching_time)
    time_values = ds['bleaching_time'].values
    time_idx = np.argmin(np.abs(time_values - time_seconds))
    actual_time = time_values[time_idx]

    # Raman
    raman = ds['raman_gt'].isel(sample=sample_idx).values

    # Individual fluorophore contributions
    fluorophores = {}
    total_fluor = np.zeros_like(raman)

    for i in range(n_fluorophores):
        contrib = get_fluorophore_contribution(ds, sample_idx, i, actual_time)
        fluorophores[f'fluorophore_{i}'] = contrib
        total_fluor += contrib

    # Reconstructed
    reconstructed = raman + total_fluor

    # Observed (UPDATED: use bleaching_time dimension)
    observed_clean = ds['intensity_clean'].isel(sample=sample_idx, bleaching_time=time_idx).values
    observed_noisy = ds['intensity_raw'].isel(sample=sample_idx, bleaching_time=time_idx).values

    # Parameters
    decay_rates = ds['decay_rates_gt'].isel(sample=sample_idx).values
    abundances = ds['abundances_gt'].isel(sample=sample_idx).values

    result = {
        'raman': raman,
        'total_fluorescence': total_fluor,
        'reconstructed': reconstructed,
        'observed_clean': observed_clean,
        'observed_noisy': observed_noisy,
        'wavenumbers': wavenumbers,
        'time_seconds': actual_time,
        'decay_rates': decay_rates,
        'abundances': abundances,
        'time_constants': 1.0 / decay_rates,
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
    Plot the full decomposition for a single sample at a given time.

    Shows:
    - Top: Full spectrum with components stacked
    - Bottom left: Individual fluorophore contributions
    - Bottom right: Residual (observed - reconstructed)
    """
    decomp = get_full_decomposition(ds, sample_idx, time_seconds)
    wn = decomp['wavenumbers']
    n_fluorophores = len(ds['fluorophore'])

    fig = plt.figure(figsize=figsize)

    # Top panel: Full decomposition
    ax1 = fig.add_subplot(2, 2, (1, 2))

    if show_noisy:
        ax1.plot(wn, decomp['observed_noisy'], 'gray', alpha=0.5,
                 label='Observed (noisy)', linewidth=0.5)

    ax1.plot(wn, decomp['observed_clean'], 'k-', alpha=0.8,
             label='Observed (clean)', linewidth=1.5)
    ax1.plot(wn, decomp['reconstructed'], 'r--', alpha=0.8,
             label='Reconstructed', linewidth=1.5)
    ax1.plot(wn, decomp['raman'], 'b-', alpha=0.7,
             label='Raman (GT)', linewidth=1.5)
    ax1.plot(wn, decomp['total_fluorescence'], 'orange', alpha=0.7,
             label='Total Fluorescence', linewidth=1.5)

    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'Sample {sample_idx} at t = {decomp["time_seconds"]:.2f}s')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom left: Individual fluorophores
    ax2 = fig.add_subplot(2, 2, 3)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_fluorophores))

    for i in range(n_fluorophores):
        τ = decomp['time_constants'][i]
        w = decomp['abundances'][i]
        ax2.plot(wn, decomp[f'fluorophore_{i}'], color=colors[i],
                 label=f'F{i+1}: τ={τ:.3f}s, w={w:.1f}', linewidth=1.5)

    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Individual Fluorophore Contributions')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Bottom right: Residual
    ax3 = fig.add_subplot(2, 2, 4)

    residual = decomp['observed_clean'] - decomp['reconstructed']
    ax3.plot(wn, residual, 'k-', linewidth=1)
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.fill_between(wn, residual, 0, alpha=0.3)

    rmse = np.sqrt(np.mean(residual**2))
    ax3.set_xlabel('Wavenumber (cm⁻¹)')
    ax3.set_ylabel('Residual')
    ax3.set_title(f'Residual (RMSE = {rmse:.4f})')
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

    Shows how fluorescence decays while Raman stays constant.
    """
    # UPDATED: use bleaching_time
    time_values = ds['bleaching_time'].values

    # Get wavenumbers for this sample
    if ds['wavenumber'].ndim == 2:
        wn = ds['wavenumber'].isel(sample=sample_idx).values
    else:
        wn = ds['wavenumber'].values

    n_times = len(time_values)
    n_fluorophores = len(ds['fluorophore'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Color maps
    time_colors = plt.cm.plasma(np.linspace(0, 0.9, n_times))
    fluor_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_fluorophores))

    # Top left: Observed spectra over time
    ax = axes[0, 0]
    for t_idx, t in enumerate(time_values):
        # UPDATED: use bleaching_time dimension
        spectrum = ds['intensity_clean'].isel(sample=sample_idx, bleaching_time=t_idx).values
        ax.plot(wn, spectrum, color=time_colors[t_idx], alpha=0.8,
                label=f't={t:.2f}s')

    ax.plot(wn, ds['raman_gt'].isel(sample=sample_idx).values, 'k--',
            linewidth=2, label='Raman (GT)')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('Observed Spectra Over Time')
    # ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Top right: Total fluorescence over time
    ax = axes[0, 1]
    for t_idx, t in enumerate(time_values):
        fluor = get_total_fluorescence(ds, sample_idx, t)
        ax.plot(wn, fluor, color=time_colors[t_idx], alpha=0.8,
                label=f't={t:.2f}s')

    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('Total Fluorescence Decay')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Individual fluorophore decay curves
    ax = axes[1, 0]

    decay_rates = ds['decay_rates_gt'].isel(sample=sample_idx).values
    abundances = ds['abundances_gt'].isel(sample=sample_idx).values

    # Get basis spectra means for amplitude
    for i in range(n_fluorophores):
        if 'sample' in ds['fluorophore_bases_gt'].dims:
            B_i = ds['fluorophore_bases_gt'].isel(sample=sample_idx, fluorophore=i).values
        else:
            B_i = ds['fluorophore_bases_gt'].isel(fluorophore=i).values

        # Mean intensity over time
        intensities = []
        for t in time_values:
            contrib = abundances[i] * B_i * np.exp(-decay_rates[i] * t)
            intensities.append(contrib.mean())

        τ = 1.0 / decay_rates[i]
        ax.plot(time_values, intensities, 'o-', color=fluor_colors[i],
                label=f'F{i+1}: τ={τ:.3f}s', linewidth=2, markersize=6)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Fluorescence Intensity')
    ax.set_title('Fluorophore Decay Curves')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Bottom right: Basis spectra (unscaled)
    ax = axes[1, 1]

    for i in range(n_fluorophores):
        if 'sample' in ds['fluorophore_bases_gt'].dims:
            B_i = ds['fluorophore_bases_gt'].isel(sample=sample_idx, fluorophore=i).values
        else:
            B_i = ds['fluorophore_bases_gt'].isel(fluorophore=i).values

        τ = 1.0 / decay_rates[i]
        ax.plot(wn, B_i, color=fluor_colors[i], linewidth=1.5,
                label=f'B{i+1} (τ={τ:.3f}s)')

    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity (normalized)')
    ax.set_title('Fluorophore Basis Spectra')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Sample {sample_idx} - Temporal Decomposition', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_stacked_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot stacked area chart showing how components sum to total.
    """
    decomp = get_full_decomposition(ds, sample_idx, time_seconds)
    wn = decomp['wavenumbers']
    n_fluorophores = len(ds['fluorophore'])

    fig, ax = plt.subplots(figsize=figsize)

    # Stack: Raman at bottom, then fluorophores
    colors = ['steelblue'] + list(plt.cm.Oranges(np.linspace(0.3, 0.8, n_fluorophores)))

    # Collect components for stacking
    components = [decomp['raman']]
    labels = ['Raman']

    for i in range(n_fluorophores):
        τ = decomp['time_constants'][i]
        components.append(decomp[f'fluorophore_{i}'])
        labels.append(f'Fluor {i+1} (τ={τ:.2f}s)')

    # Stack plot
    ax.stackplot(wn, *components, labels=labels, colors=colors, alpha=0.8)

    # Overlay observed
    ax.plot(wn, decomp['observed_clean'], 'k-', linewidth=2,
            label='Observed (clean)', alpha=0.8)

    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Sample {sample_idx} at t={time_seconds:.2f}s - Stacked Decomposition')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Load your synthetic dataset
    ds = xr.open_dataset('synthetic_bleaching_10k.nc')

    # 1. Plot full decomposition for sample 0 at t=0.5s
    fig = plot_decomposition(ds, sample_idx=0, time_seconds=0.5)
    plt.savefig('decomposition_example.png', dpi=150)
    plt.close()

    # 2. Plot temporal evolution for sample 0
    fig = plot_temporal_decomposition(ds, sample_idx=0)
    plt.savefig('temporal_decomposition_example.png', dpi=150)
    plt.close()

    # 3. Plot stacked decomposition
    fig = plot_stacked_decomposition(ds, sample_idx=0, time_seconds=0.1)
    plt.savefig('stacked_decomposition_example.png', dpi=150)
    plt.close()

    # 4. Get individual fluorophore contribution manually
    sample_idx = 0
    time_s = 0.5

    # Fluorophore 0 (fastest decay)
    F0 = get_fluorophore_contribution(ds, sample_idx, fluorophore_idx=0, time_seconds=time_s)

    # All components
    decomp = get_full_decomposition(ds, sample_idx, time_s)
    print(f"Decay rates (λ): {decomp['decay_rates']}")
    print(f"Time constants (τ): {decomp['time_constants']}")
    print(f"Abundances (w): {decomp['abundances']}")

    print("\n✅ All visualization examples completed!")

    # 5. 3D visualization (if plotly is available)
    if HAS_PLOTLY:
        fig_3d = visualise_decomposition_3d(ds, sample_idx=0, subsample_wn=5, subsample_time=2)
        fig_3d.write_html('decomposition_3d.html')
        print("Saved: decomposition_3d.html (open in browser for interactive 3D view)")
        # To display in Jupyter: fig_3d.show()
    else:
        print("\nSkipping 3D visualization (plotly not installed)")
        print("Install with: pip install plotly")


def visualise_decomposition_3d(
    ds: xr.Dataset,
    sample_idx: int,
    subsample_wn: int = 5,
    subsample_time: int = 2,
):
    """
    Interactive 3D visualization using plotly (allows rotation/zoom).

    Shows bleaching time series as interactive 3D surfaces with dropdown menu to switch between:
    - Noisy Data (Raman + Fluorescence + noise)
    - Clean Data (Raman + Fluorescence, no noise)
    - Fluorescence (decaying over time)
    - Raman (constant, pure Raman signal)

    Parameters
    ----------
    ds : xr.Dataset
        Synthetic bleaching dataset
    sample_idx : int
        Sample index to visualise
    subsample_wn : int
        Subsample factor for wavenumber axis (for performance)
    subsample_time : int
        Subsample factor for time axis (for performance)

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure

    Raises
    ------
    ImportError
        If plotly is not installed
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for 3D visualization. Install with: pip install plotly")

    # Get time values
    time_values = ds['bleaching_time'].values

    # Get wavenumbers for this sample
    if ds['wavenumber'].ndim == 2:
        wavenumbers = ds['wavenumber'].isel(sample=sample_idx).values
    else:
        wavenumbers = ds['wavenumber'].values

    n_times = len(time_values)
    n_wn = len(wavenumbers)
    n_fluorophores = len(ds['fluorophore'])

    # Get data
    # Noisy data: Raman + Fluorescence + noise
    noisy_data = ds['intensity_raw'].isel(sample=sample_idx).values  # (n_times, n_wn)

    # Clean data: Raman + Fluorescence (no noise)
    clean_data = ds['intensity_clean'].isel(sample=sample_idx).values  # (n_times, n_wn)

    # Get ground truth components
    raman = ds['raman_gt'].isel(sample=sample_idx).values
    decay_rates = ds['decay_rates_gt'].isel(sample=sample_idx).values
    abundances = ds['abundances_gt'].isel(sample=sample_idx).values

    # Get fluorophore bases
    if 'sample' in ds['fluorophore_bases_gt'].dims:
        bases = ds['fluorophore_bases_gt'].isel(sample=sample_idx).values  # (n_fluorophores, n_wn)
    else:
        bases = ds['fluorophore_bases_gt'].values  # (n_fluorophores, n_wn)

    # Calculate fluorescence component over time
    total_fluor = np.zeros((n_times, n_wn))

    for t_idx, t in enumerate(time_values):
        # Add fluorescence components (decaying)
        for i in range(n_fluorophores):
            decay = np.exp(-decay_rates[i] * t)
            fluor_i = abundances[i] * bases[i] * decay
            total_fluor[t_idx] += fluor_i

    # Subsample for performance
    wn_idx = np.arange(0, n_wn, subsample_wn)
    t_idx = np.arange(0, n_times, subsample_time)

    wn_sub = wavenumbers[wn_idx]
    t_sub = time_values[t_idx]

    # Create figure with dropdown to select which surface to view
    fig = go.Figure()

    # 1. Noisy Data (Raman + Fluorescence + noise)
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=noisy_data[np.ix_(t_idx, wn_idx)],
            colorscale="Viridis",
            name="Noisy Data",
            visible=True,
            colorbar=dict(title="Intensity", x=1.02),
        )
    )

    # 2. Clean Data (Raman + Fluorescence, no noise)
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=clean_data[np.ix_(t_idx, wn_idx)],
            colorscale="Viridis",
            name="Clean Data",
            visible=False,
            colorbar=dict(title="Intensity", x=1.02),
        )
    )

    # 3. Fluorescence Only (decaying over time)
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=total_fluor[np.ix_(t_idx, wn_idx)],
            colorscale="Oranges",
            name="Fluorescence",
            visible=False,
            colorbar=dict(title="Fluorescence", x=1.02),
        )
    )

    # 4. Raman Only (constant surface, pure Raman)
    raman_surface = np.tile(raman[wn_idx], (len(t_idx), 1))
    fig.add_trace(
        go.Surface(
            x=wn_sub,
            y=t_sub,
            z=raman_surface,
            colorscale="Blues",
            name="Raman",
            visible=False,
            colorbar=dict(title="Raman", x=1.02),
        )
    )

    # Create dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(
                        label="Noisy Data (Raman+Fluor+Noise)",
                        method="update",
                        args=[{"visible": [True, False, False, False]}],
                    ),
                    dict(
                        label="Clean Data (Raman+Fluor)",
                        method="update",
                        args=[{"visible": [False, True, False, False]}],
                    ),
                    dict(
                        label="Fluorescence Only (Decaying)",
                        method="update",
                        args=[{"visible": [False, False, True, False]}],
                    ),
                    dict(
                        label="Raman Only (Constant)",
                        method="update",
                        args=[{"visible": [False, False, False, True]}],
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
            yaxis_title="Bleaching Time (s)",
            zaxis_title="Intensity",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        title=f"3D Bleaching Time Series - Sample {sample_idx}<br><sub>Use dropdown to switch views</sub>",
        width=900,
        height=700,
    )

    return fig