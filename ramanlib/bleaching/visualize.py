"""Visualization utilities for bleaching decomposition."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ramanlib.bleaching.physics import (
    interpolate_bases,
    reconstruct_time_series_numpy,
)
from ramanlib.core import SpectralData


def visualize_data_3d(
    data: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    wavenumbers: Optional[np.ndarray] = None,
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
from ramanlib.bleaching.decompose import DecompositionResult


def visualise_decomposition(
    data: SpectralData,
    decomposition: DecompositionResult,
    data_clean: Optional[SpectralData] = None,
    reference_raman: Optional[np.ndarray] = None,
    reference_bases: Optional[np.ndarray] = None,
    reference_rates: Optional[np.ndarray] = None,
    reference_abundances: Optional[np.ndarray] = None,
    normalise: bool = False,
    figsize: Tuple[int, int] = (20, 10),
    n_train: Optional[int] = None,
    sample_id: Optional[str] = None,
):
    """
    Visualize spectral decomposition results.

    Args:
        data: Original time series as SpectralData (must have time_values set).
        data_clean: Cleaned time series as SpectralData (must have time_values set).
        decomposition: DecompositionResult from predict.
            physics_model and frame_duration are read from decomposition automatically.
        reference_raman: GT Raman spectrum for comparison. If None, uses last 20 frames avg.
        reference_bases: GT fluorophore basis spectra [n_gt, n_wavenumbers].
        reference_rates: GT decay rates [n_gt] in s⁻¹.
        reference_abundances: GT physical abundances [n_gt].
        normalise: If True, L-inf normalise spectra in the bases plot for visual comparison.
        figsize: Figure size.
        n_train: Number of training frames — adds a cutoff line and shades regions.
        sample_id: Optional label shown in the suptitle.

    Returns:
        Tuple of (figure, axes)
    """
    physics_model = getattr(decomposition, "physics_model", "pointsample")
    has_reference = (
        reference_bases is not None
        and reference_rates is not None
        and reference_abundances is not None
    )

    raman = decomposition.raman.intensities
    Y = data.intensities
    Y_clean = data_clean.intensities if data_clean is not None else None
    n_t, n_wn = Y.shape
    time_values = data.time_values
    bases = decomposition.fluorophore_spectra.intensities
    abundances = decomposition.abundances
    rates = decomposition.rates
    time_constants = 1.0 / rates
    n_fluorophores = len(rates)
    wavenumbers = data.wavenumbers

    # Frame duration: prefer decomposition attribute, fall back to time axis delta
    frame_dur = getattr(decomposition, "frame_duration", None)
    if frame_dur is None and time_values is not None and len(time_values) > 1:
        frame_dur = float(time_values[1] - time_values[0])
    if frame_dur is None:
        frame_dur = 1.0
    print(f"Frame duration: {frame_dur:.3f} s")
    # decomposition.raman.intensities is in counts/sec (rate).
    # Multiply by frame_dur to convert to counts/frame, matching Y and reconstruction.
    raman_per_frame = raman * frame_dur

    # decomposition.reconstruction() returns fluorescence + raman·Δt in ADU.
    reconstruction = decomposition.reconstruction(time_values)

    # Training cutoff time for vertical line — placed at the last *used* frame.
    if n_train is not None and time_values is not None and len(time_values) > 0:
        t_train_cutoff = float(time_values[min(n_train - 1, len(time_values) - 1)])
    else:
        t_train_cutoff = None

    # Plot 2 comparison strategy:
    #   Real data (no GT): compare Y[-1] vs reconstruction[-1] — both at the same
    #   last time point, both in raw ADU, so residual fluorescence cancels out.
    #   Synthetic data (GT provided): compare extracted Raman rate vs true GT Raman.
    if reference_raman is None:
        # Both observed and reconstructed at the identical last time point.
        plot2_observed = Y[-1]  # raw ADU at t_last
        plot2_predicted = reconstruction[-1]  # model ADU at t_last
        plot2_obs_label = (
            f"Observed (t={time_values[-1]:.2f}s)"
            if time_values is not None
            else "Observed (last frame)"
        )
        plot2_pred_label = (
            f"Reconstruction (t={time_values[-1]:.2f}s)"
            if time_values is not None
            else "Reconstruction (last frame)"
        )
        plot2_title_prefix = "Last-frame comparison"
        print(
            "No reference Raman provided — comparing observed vs reconstructed at last time point."
        )
    else:
        plot2_observed = reference_raman * frame_dur  # GT counts/sec → counts/frame
        plot2_predicted = raman_per_frame  # extracted Raman counts/frame
        plot2_obs_label = "Ground Truth Raman"
        plot2_pred_label = "Predicted Raman"
        plot2_title_prefix = "Extracted Raman Spectrum"

    # ── Consistent colour palette ────────────────────────────────────────────
    # Colours are keyed to GT fluorophore index so the same entity always
    # gets the same colour across all subplots.
    _COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Match each predicted basis to the best (highest-correlation) unmatched GT basis.
    # pred_to_ref[i] = (gt_index, correlation) or (None, 0) if no GT available.
    corr_matrix: Optional[np.ndarray] = None
    if has_reference:
        n_ref = reference_bases.shape[0]
        corr_matrix = np.zeros((n_fluorophores, n_ref))
        for i in range(n_fluorophores):
            for j in range(n_ref):
                c = np.corrcoef(bases[i], reference_bases[j])[0, 1]
                corr_matrix[i, j] = c if np.isfinite(c) else 0.0
        # Greedy: process predictions in descending order of their best available corr
        pred_to_ref: Dict[int, Tuple] = {}
        used_refs: set = set()
        order = np.argsort(-corr_matrix.max(axis=1))
        for p in order:
            available = [
                (j, corr_matrix[p, j]) for j in range(n_ref) if j not in used_refs
            ]
            if available:
                best_ref, best_corr = max(available, key=lambda x: x[1])
                pred_to_ref[p] = (best_ref, float(best_corr))
                used_refs.add(best_ref)
            else:
                pred_to_ref[p] = (None, 0.0)
        pred_colors = [
            (
                _COLORS[pred_to_ref[i][0] % len(_COLORS)]
                if pred_to_ref[i][0] is not None
                else "#888888"
            )
            for i in range(n_fluorophores)
        ]
        ref_colors = [_COLORS[j % len(_COLORS)] for j in range(n_ref)]
    else:
        pred_to_ref = {i: (None, 0.0) for i in range(n_fluorophores)}
        pred_colors = [_COLORS[i % len(_COLORS)] for i in range(n_fluorophores)]
        corr_matrix = None

    # ── Layout: always 2×3 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    def _add_train_cutoff(ax, orientation="v"):
        """Solid dark line marking the last training frame."""
        if t_train_cutoff is None:
            return
        if orientation == "v":
            ax.axvline(
                t_train_cutoff,
                color="#444444",
                linestyle="--",
                linewidth=1.8,
                label=f"Train cutoff (n={n_train})",
                zorder=5,
            )
        else:
            ax.axhline(
                t_train_cutoff, color="#444444", linestyle="--", linewidth=1.8, zorder=5
            )

    # ── Plot 1: Original Time Series + Reconstruction overlay ─────────────────
    ax = axes[0, 0]
    n_show = min(5, n_train)
    cmap_ts = plt.cm.viridis
    show_indices = np.arange(n_show)
    for i, idx in enumerate(show_indices):
        c = cmap_ts(i / max(n_show - 1, 1))
        t_label = (
            f"t={time_values[idx]:.2f}s" if time_values is not None else f"frame {idx}"
        )
        ax.plot(wavenumbers, Y[idx], color=c, alpha=0.8, linewidth=1.2, label=t_label)
        ax.plot(
            wavenumbers,
            reconstruction[idx],
            color=c,
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
        )
    # Dummy lines for legend
    ax.plot([], [], "k-", linewidth=1.5, label="Observed")
    ax.plot([], [], "k--", linewidth=1.0, alpha=0.6, label="Reconstructed")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (counts/frame)")
    ax.set_title(f"Time Series Summary")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Reconstruction vs observed at last time point (or GT Raman) ──
    ax = axes[0, 1]
    ax.plot(
        wavenumbers,
        plot2_predicted,
        color=_COLORS[0],
        linewidth=2,
        label=plot2_pred_label,
    )
    ax.plot(
        wavenumbers,
        plot2_observed,
        "r--",
        linewidth=1.5,
        alpha=0.8,
        label=plot2_obs_label,
    )
    plot2_corr = np.corrcoef(plot2_predicted, plot2_observed)[0, 1]
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (counts/frame)")
    ax.set_title(f"{plot2_title_prefix}  (r = {plot2_corr:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Fluorophore Basis Spectra ─────────────────────────────────────
    ax = axes[0, 2]
    if bases is not None:
        for i in range(n_fluorophores):
            tau = time_constants[i]
            b_plot = bases[i]
            if normalise:
                peak = b_plot.max()
                b_plot = b_plot / peak if peak > 0 else b_plot
            corr_str = f", r={pred_to_ref[i][1]:.2f}" if has_reference else ""
            ax.plot(
                wavenumbers,
                b_plot,
                color=pred_colors[i],
                linewidth=2,
                label=f"Pred B{i+1} (τ={tau:.3f}s{corr_str})",
            )
    if has_reference:
        for j in range(n_ref):
            tau_gt = 1.0 / reference_rates[j]
            b_ref = reference_bases[j]
            if normalise:
                peak = b_ref.max()
                b_ref = b_ref / peak if peak > 0 else b_ref
            ax.plot(
                wavenumbers,
                b_ref,
                color=ref_colors[j],
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
                label=f"GT B{j+1} (τ={tau_gt:.3f}s, w={reference_abundances[j]:.1f})",
            )
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Normalised intensity" if normalise else "Intensity")
    ax.set_title(
        "Fluorophore Basis Spectra\n(solid=pred, dashed=GT, colour=matched pair)"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Decay Components ──────────────────────────────────────────────
    ax = axes[1, 0]
    total_fluor = np.zeros(n_t)
    if bases is not None:
        for i in range(n_fluorophores):
            fluor_series = reconstruct_time_series_numpy(
                raman=np.zeros(bases.shape[1]),
                bases=bases[i : i + 1, :],
                abundances=np.array([abundances[i]]),
                decay_rates=np.array([rates[i]]),
                time_values=time_values,
                frame_duration=frame_dur,
                physics_model=physics_model,
            )
            amplitude = fluor_series.mean(axis=1)
            total_fluor += amplitude
            tau = time_constants[i]
            ax.plot(
                time_values,
                amplitude,
                color=pred_colors[i],
                linewidth=1.5,
                label=f"τ={tau:.3f}s, w={abundances[i]:.1f}",
            )
    if has_reference:
        total_gt_fluor = np.zeros(n_t)
        for j in range(n_ref):
            fluor_series = reconstruct_time_series_numpy(
                raman=np.zeros(reference_bases.shape[1]),
                bases=reference_bases[j : j + 1, :],
                abundances=np.array([reference_abundances[j]]),
                decay_rates=np.array([reference_rates[j]]),
                time_values=time_values,
                frame_duration=frame_dur,
                physics_model=physics_model,
            )
            amplitude = fluor_series.mean(axis=1)
            total_gt_fluor += amplitude
            ax.plot(
                time_values,
                amplitude,
                color=ref_colors[j],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"GT τ={1.0 / reference_rates[j]:.3f}s, w={reference_abundances[j]:.1f}",
            )
        ax.plot(
            time_values, total_gt_fluor, "r--", linewidth=2, label="Total GT fluor."
        )
    ax.plot(time_values, total_fluor, "k-", linewidth=2, label="Total pred. fluor.")

    # ── Observed decay profile vs prediction (no GT decomposition needed) ─────
    # Average over wavenumbers → 1D decay curve containing fluor + Raman.
    # Predicted Raman appears as a constant floor; subtracting it from the
    # observed profile gives an estimate of the GT fluorescence decay.
    raman_floor = float(raman_per_frame.mean())
    gt_profile = Y.mean(axis=1)  # [T] observed total (ADU)
    gt_fluor_est = gt_profile - raman_floor  # observed minus predicted Raman floor

    ax.plot(
        time_values,
        gt_profile,
        color="dimgray",
        linewidth=1.5,
        linestyle=":",
        alpha=0.85,
        label="GT total (wn avg)",
    )
    ax.axhline(
        raman_floor,
        color="steelblue",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Pred. Raman floor ({raman_floor:.1f})",
    )
    ax.plot(
        time_values,
        gt_fluor_est,
        color="tomato",
        linewidth=1.5,
        linestyle=":",
        alpha=0.85,
        label="GT − Raman (≈ fluor.)",
    )

    _add_train_cutoff(ax)
    if t_train_cutoff is not None:
        ax.axvspan(time_values[0], t_train_cutoff, alpha=0.06, color="steelblue")
        ax.axvspan(t_train_cutoff, time_values[-1], alpha=0.06, color="darkorange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean intensity (counts/frame)")
    ax.set_title("Decay Components")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 5: First Frame Reconstruction ───────────────────────────────────
    ax = axes[1, 1]
    ax.plot(wavenumbers, Y[0], "r--", linewidth=1.5, alpha=0.6, label="Observed (t=0)")
    if Y_clean is not None:
        ax.plot(
            wavenumbers,
            Y_clean[0],
            "g--",
            linewidth=1.5,
            alpha=0.8,
            label="Clean (t=0)",
        )
    ax.plot(
        wavenumbers,
        reconstruction[0],
        color=_COLORS[0],
        linewidth=1.5,
        label="Reconstructed (t=0)",
    )
    t0_mse = float(np.mean((Y[0] - reconstruction[0]) ** 2))
    title_mse = f"Reconstruction (t=0)  MSE={t0_mse:.2f}"
    if Y_clean is not None:
        t0_mse_clean = float(np.mean((Y_clean[0] - reconstruction[0]) ** 2))
        title_mse += f"  MSE Clean={t0_mse_clean:.2f}"
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (counts/frame)")
    ax.set_title(title_mse)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 6: MSE over time (log scale) ────────────────────────────────────
    ax = axes[1, 2]
    residuals = Y - reconstruction
    mse_over_time = np.mean(residuals**2, axis=1)
    mse_all = float(mse_over_time.mean())
    n_first = n_train if n_train else 20
    mse_first = float(mse_over_time[:n_first].mean())
    ax.semilogy(time_values, mse_over_time, "k-", linewidth=1.5, label="MSE per frame")
    _add_train_cutoff(ax)
    if t_train_cutoff is not None:
        ax.axvspan(
            time_values[0],
            t_train_cutoff,
            alpha=0.06,
            color="steelblue",
            label="Training region",
        )
        ax.axvspan(
            t_train_cutoff,
            time_values[-1],
            alpha=0.06,
            color="darkorange",
            label="Extrapolation region",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("MSE Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Suptitle with sample/model info ───────────────────────────────────────
    parts = []
    if sample_id is not None:
        parts.append(f"Sample: {sample_id}")
    parts.append(f"n_bases={n_fluorophores}")
    parts.append(f"physics={physics_model}")
    if n_train is not None:
        parts.append(f"n_train={n_train}")
    fig.suptitle("   |   ".join(parts), fontsize=9, y=1.01)

    plt.tight_layout()

    # ── Printed summary ───────────────────────────────────────────────────────
    print(f"\nReconstruction MSE (all frames):        {mse_all:.4f}")
    print(f"Reconstruction MSE (first {n_first:2d} frames):  {mse_first:.4f}")
    print(f"Raman correlation with reference:       {plot2_corr:.4f}")
    print(f"Time constants τ (s):  {time_constants}")
    print(f"Abundances w:          {abundances}")

    if has_reference:
        print("\nMatched basis pairs (pred → GT, by correlation):")
        for p in range(n_fluorophores):
            ref_j, corr_val = pred_to_ref[p]
            if ref_j is not None:
                tau_pred = time_constants[p]
                tau_gt = 1.0 / reference_rates[ref_j]
                rate_err_pct = (
                    100.0
                    * abs(rates[p] - reference_rates[ref_j])
                    / reference_rates[ref_j]
                )
                print(
                    f"  Pred B{p+1} (τ={tau_pred:.3f}s, w={abundances[p]:.1f})  →  "
                    f"GT B{ref_j+1} (τ={tau_gt:.3f}s, w={reference_abundances[ref_j]:.1f}):  "
                    f"r={corr_val:.4f},  λ err={rate_err_pct:.1f}%"
                )

        # Sorted rate errors (naive, for completeness).
        # When model capacity != n_gt, compare the top-n_compare components
        # by abundance on the predicted side vs the n_compare largest GT rates.
        n_gt = len(reference_rates)
        n_pred = len(decomposition.rates)
        n_compare = min(n_pred, n_gt)
        top_idx = np.argsort(decomposition.abundances)[-n_compare:]
        rates_est_sorted = np.sort(decomposition.rates[top_idx])
        rates_gt_sorted = np.sort(reference_rates)[-n_compare:]
        rate_errors_pct = (
            100.0 * np.abs(rates_est_sorted - rates_gt_sorted) / rates_gt_sorted
        )
        print(
            f"\nRate errors % (sorted, naive 1-to-1, top {n_compare} of "
            f"pred={n_pred}/gt={n_gt}): {np.round(rate_errors_pct, 1)}"
        )

    return fig, axes


def plot_parameter_detail(
    decomposition: DecompositionResult,
    reference_bases: Optional[np.ndarray] = None,
    reference_rates: Optional[np.ndarray] = None,
    reference_abundances: Optional[np.ndarray] = None,
    n_train: Optional[int] = None,
    sample_id: Optional[str] = None,
):
    """
    Two detail figures (returned as a tuple):
      fig_scatter — basis correlation heatmap + decay-rate scatter + abundance scatter (1×3)
      fig_comps   — per-component spectra: abundance × basis at t=0 pred vs GT

    Wavenumbers and metadata are read from decomposition automatically.

    Usage::
        fig_scatter, fig_comps = plot_parameter_detail(
            decomposition, ref_bases, ref_rates, ref_abundances,
            n_train=5, sample_id="7"
        )
    """
    _COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    has_reference = (
        reference_bases is not None
        and reference_rates is not None
        and reference_abundances is not None
    )
    bases = decomposition.fluorophore_spectra.intensities
    abundances = decomposition.abundances
    rates = decomposition.rates
    n_fluorophores = len(rates)
    n_ref = len(reference_rates) if has_reference else 0

    # Wavenumbers: prefer fluorophore_spectra axis, fall back to raman axis
    wavenumbers = decomposition.fluorophore_spectra.wavenumbers
    if wavenumbers is None:
        wavenumbers = decomposition.raman.wavenumbers
    if wavenumbers is None:
        wavenumbers = np.arange(bases.shape[1])

    # ── Info string for suptitles ─────────────────────────────────────────────
    physics_model = getattr(decomposition, "physics_model", "pointsample")
    parts = []
    if sample_id is not None:
        parts.append(f"Sample: {sample_id}")
    parts.append(f"n_bases={n_fluorophores}")
    if has_reference:
        parts.append(f"n_gt={n_ref}")
    parts.append(f"physics={physics_model}")
    if n_train is not None:
        parts.append(f"n_train={n_train}")
    info_str = "   |   ".join(parts)

    # ── Greedy matching by basis correlation (only when GT available) ─────────
    pred_to_ref: Dict[int, Tuple] = {i: (None, 0.0) for i in range(n_fluorophores)}
    corr_matrix = np.zeros((n_fluorophores, max(n_ref, 1)))
    if has_reference:
        corr_matrix = np.zeros((n_fluorophores, n_ref))
        for i in range(n_fluorophores):
            for j in range(n_ref):
                c = np.corrcoef(bases[i], reference_bases[j])[0, 1]
                corr_matrix[i, j] = c if np.isfinite(c) else 0.0
        used_refs: set = set()
        for p in np.argsort(-corr_matrix.max(axis=1)):
            available = [
                (j, corr_matrix[p, j]) for j in range(n_ref) if j not in used_refs
            ]
            if available:
                best_j, best_c = max(available, key=lambda x: x[1])
                pred_to_ref[p] = (best_j, float(best_c))
                used_refs.add(best_j)

    pred_colors = [
        (
            _COLORS[pred_to_ref[i][0] % len(_COLORS)]
            if pred_to_ref[i][0] is not None
            else _COLORS[i % len(_COLORS)]
        )
        for i in range(n_fluorophores)
    ]
    ref_colors = [_COLORS[j % len(_COLORS)] for j in range(n_ref)]

    # ── Figure 1: Correlation heatmap + Rate scatter + Abundance scatter ─────
    fig_scatter, (ax_corr, ax_r, ax_a) = plt.subplots(1, 3, figsize=(15, 5))

    if has_reference:
        im = ax_corr.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
        ax_corr.set_xticks(range(n_ref))
        ax_corr.set_yticks(range(n_fluorophores))
        ax_corr.set_xticklabels(
            [f"GT B{j+1}" for j in range(n_ref)], rotation=45, ha="right", fontsize=9
        )
        ax_corr.set_yticklabels(
            [f"Pred B{i+1}" for i in range(n_fluorophores)], fontsize=9
        )
        for i in range(n_fluorophores):
            for j in range(n_ref):
                text_color = "white" if abs(corr_matrix[i, j]) > 0.6 else "black"
                ax_corr.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )
        plt.colorbar(im, ax=ax_corr, shrink=0.8, label="Pearson r")
        ax_corr.set_title("Basis Correlation (Pred vs GT)")
        ax_corr.set_xlabel("GT Bases")
        ax_corr.set_ylabel("Predicted Bases")
    else:
        ax_corr.bar(
            range(n_fluorophores),
            rates,
            color=pred_colors,
            edgecolor="k",
            linewidth=0.5,
        )
        ax_corr.set_xticks(range(n_fluorophores))
        ax_corr.set_xticklabels([f"B{i+1}" for i in range(n_fluorophores)], fontsize=9)
        ax_corr.set_xlabel("Component")
        ax_corr.set_ylabel("Decay rate λ (s⁻¹)")
        ax_corr.set_title("Predicted Decay Rates")
        ax_corr.grid(True, alpha=0.3, axis="y")

    if has_reference:
        pred_r, gt_r, pred_a, gt_a, pair_cols, pair_labels = [], [], [], [], [], []
        for p in range(n_fluorophores):
            ref_j, _ = pred_to_ref[p]
            if ref_j is not None:
                pred_r.append(rates[p])
                gt_r.append(reference_rates[ref_j])
                pred_a.append(float(abundances[p]))
                gt_a.append(float(reference_abundances[ref_j]))
                pair_cols.append(pred_colors[p])
                pair_labels.append(f"B{p+1}→GT{ref_j+1}")
        for ax, xs, ys, xlabel, ylabel, title in [
            (ax_r, gt_r, pred_r, "GT λ (s⁻¹)", "Predicted λ (s⁻¹)", "Decay Rate"),
            (
                ax_a,
                gt_a,
                pred_a,
                "GT abundance w",
                "Predicted abundance w",
                "Abundance",
            ),
        ]:
            if xs:
                for x, y, c, lbl in zip(xs, ys, pair_cols, pair_labels):
                    ax.scatter(
                        x, y, color=c, s=110, zorder=5, edgecolors="k", linewidths=0.5
                    )
                    ax.annotate(
                        lbl,
                        (x, y),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                    )
                lo, hi = 0.0, float(max(max(xs), max(ys))) * 1.2
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="1:1")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} Comparison")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
    else:
        ax_r.bar(
            range(n_fluorophores),
            abundances,
            color=pred_colors,
            edgecolor="k",
            linewidth=0.5,
        )
        ax_r.set_xticks(range(n_fluorophores))
        ax_r.set_xticklabels([f"B{i+1}" for i in range(n_fluorophores)], fontsize=9)
        ax_r.set_xlabel("Component")
        ax_r.set_ylabel("Abundance w")
        ax_r.set_title("Predicted Abundances")
        ax_r.grid(True, alpha=0.3, axis="y")
        ax_a.bar(
            range(n_fluorophores),
            1.0 / rates,
            color=pred_colors,
            edgecolor="k",
            linewidth=0.5,
        )
        ax_a.set_xticks(range(n_fluorophores))
        ax_a.set_xticklabels([f"B{i+1}" for i in range(n_fluorophores)], fontsize=9)
        ax_a.set_xlabel("Component")
        ax_a.set_ylabel("Time constant τ (s)")
        ax_a.set_title("Predicted Time Constants")
        ax_a.grid(True, alpha=0.3, axis="y")

    if info_str:
        fig_scatter.suptitle(info_str, fontsize=9, y=1.02)
    fig_scatter.tight_layout()

    # ── Figure 2: Per-component spectra (abundance × basis at t=0) ───────────
    n_cols = min(n_fluorophores, 4)
    n_rows = int(np.ceil(n_fluorophores / n_cols))
    fig_comps, axes_c = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for i in range(n_fluorophores):
        row, col = divmod(i, n_cols)
        ax = axes_c[row][col]
        tau_pred = 1.0 / rates[i]
        pred_comp = float(abundances[i]) * bases[i]
        ax.plot(
            wavenumbers,
            pred_comp,
            color=pred_colors[i],
            linewidth=2,
            label=f"Pred  τ={tau_pred:.3f}s  w={abundances[i]:.2f}",
        )

        ref_j, corr_val = pred_to_ref[i]
        if ref_j is not None:
            tau_gt = 1.0 / reference_rates[ref_j]
            gt_comp = float(reference_abundances[ref_j]) * reference_bases[ref_j]
            ax.plot(
                wavenumbers,
                gt_comp,
                color=ref_colors[ref_j],
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"GT    τ={tau_gt:.3f}s  w={reference_abundances[ref_j]:.2f}",
            )
            ax.set_title(f"Component {i + 1}  (r={corr_val:.3f})")
        else:
            ax.set_title(f"Component {i + 1}  (no GT match)")

        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("w · B(ν)  at t=0")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_fluorophores, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes_c[row][col].set_visible(False)

    title_comps = (
        "Individual Components: abundance × basis at t=0  (solid=pred, dashed=GT)"
    )
    if info_str:
        title_comps += f"\n{info_str}"
    fig_comps.suptitle(title_comps, fontsize=10)
    fig_comps.tight_layout()

    return fig_scatter, fig_comps


try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def get_fluorophore_corrs(
    predicted: SpectralData, reference: SpectralData
) -> pd.DataFrame:
    """Find top k fluorophores in a reference dataset with the highest pearson correlations"""
    pred_wn = predicted.wavenumbers
    ref_wn = reference.wavenumbers

    # Interpolate reference onto predicted wavenumber axis if needed
    if pred_wn.shape[0] != ref_wn.shape[0] or not np.allclose(pred_wn, ref_wn):
        ref_interp = interpolate_bases(reference.intensities, ref_wn, pred_wn)
    else:
        ref_interp = reference.intensities
    pred_interp = predicted.intensities

    fluor_corrs = np.zeros((pred_interp.shape[0], ref_interp.shape[0]))
    for k in range(pred_interp.shape[0]):
        for i in range(ref_interp.shape[0]):
            fluor_corrs[k, i] = np.corrcoef(pred_interp[k], ref_interp[i])[0, 1]

    if reference.label is not None:
        return pd.DataFrame(data=fluor_corrs, columns=reference.label)
    else:
        return pd.DataFrame(data=fluor_corrs)


def get_fluorophore_contribution(
    ds: xr.Dataset,
    sample_idx: int,
    fluorophore_idx: int,
    physics_model: str,
    time_seconds: Optional[float] = None,
    frame_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Compute contribution of a single fluorophore at a given time using CCD integration.

    Integrated: wᵢ · Bᵢ(ν) · [exp(-λᵢ·t) - exp(-λᵢ·(t+T))] / λᵢ

    Parameters
    ----------
    ds : xr.Dataset
        Synthetic dataset with ground truth parameters
    sample_idx : int
        Sample index
    fluorophore_idx : int
        Fluorophore index
    time_seconds : float, optional
        Frame start time in seconds. If None, returns at t=0.
    frame_duration : float, optional
        CCD integration time. If None, inferred from time axis.

    Returns
    -------
    np.ndarray
        Fluorophore contribution spectrum (integrated over frame)
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

    if frame_duration is None:
        time_values = ds["bleaching_time"].values
        if len(time_values) > 1:
            frame_duration = float(time_values[1] - time_values[0])
        else:
            frame_duration = 0.1

    # Use physics function with single fluorophore, zero Raman
    result = reconstruct_time_series_numpy(
        raman=np.zeros_like(B_i),
        bases=B_i[np.newaxis, :],
        abundances=np.array([w_i]),
        decay_rates=np.array([float(λ_i)]),
        time_values=np.array([time_seconds]),
        physics_model=physics_model,
        frame_duration=frame_duration,
    )
    return result[0]  # [1, W] -> [W]


def get_total_fluorescence(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
    physics_model: str,
    frame_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Compute total fluorescence at a given time using CCD integration model.

    F(ν,t) = Σᵢ wᵢ · Bᵢ(ν) · [exp(-λᵢ·t) - exp(-λᵢ·(t+T))] / λᵢ
    """
    n_fluorophores = len(ds["fluorophore"])

    if ds["wavenumber"].ndim == 2:
        n_wn = ds["wavenumber"].isel(sample=sample_idx).shape[0]
    else:
        n_wn = len(ds["wavenumber"])

    total = np.zeros(n_wn)
    for i in range(n_fluorophores):
        total += get_fluorophore_contribution(
            ds,
            sample_idx,
            i,
            time_seconds,
            frame_duration=frame_duration,
            physics_model=physics_model,
        )

    return total


def get_full_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
    physics_model: str,
) -> Dict:
    """
    Get all components of the decomposition at a given time.

    Uses the CCD integration model to match data generation.

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

    # Infer frame duration from time axis
    if len(time_values) > 1:
        frame_duration = float(time_values[1] - time_values[0])
    else:
        frame_duration = 1.0
    print(f"Using {frame_duration} frames")
    raman = ds["raman_gt"].isel(sample=sample_idx).values
    decay_rates = ds["decay_rates_gt"].isel(sample=sample_idx).values
    abundances = ds["abundances_gt"].isel(sample=sample_idx).values

    # Get bases
    if "sample" in ds["fluorophore_bases_gt"].dims:
        bases = ds["fluorophore_bases_gt"].isel(sample=sample_idx).values
    else:
        bases = ds["fluorophore_bases_gt"].values

    # Use integrated model for single frame (matches data generation)
    t_single = np.array([actual_time])
    reconstructed_frame = reconstruct_time_series_numpy(
        raman,
        bases,
        abundances,
        decay_rates,
        t_single,
        frame_duration=frame_duration,
        physics_model=physics_model,
    )
    reconstructed = reconstructed_frame[0]  # [1, W] -> [W]

    # Raman contribution per frame
    raman_per_frame = raman * frame_duration

    # Total fluorescence = reconstructed - raman_per_frame
    total_fluor = reconstructed - raman_per_frame

    # Individual fluorophore contributions via physics function
    fluorophores = {}
    for i in range(n_fluorophores):
        contrib = reconstruct_time_series_numpy(
            raman=np.zeros(bases.shape[1]),
            bases=bases[i : i + 1, :],
            abundances=np.array([abundances[i]]),
            decay_rates=np.array([decay_rates[i]]),
            time_values=t_single,
            frame_duration=frame_duration,
            physics_model=physics_model,
        )
        fluorophores[f"fluorophore_{i}"] = contrib[0]  # [1, W] -> [W]

    observed_clean = (
        ds["intensity_clean"].isel(sample=sample_idx, bleaching_time=time_idx).values
    )
    observed_noisy = (
        ds["intensity_raw"].isel(sample=sample_idx, bleaching_time=time_idx).values
    )

    result = {
        "raman": raman_per_frame,
        "total_fluorescence": total_fluor,
        "reconstructed": reconstructed,
        "observed_clean": observed_clean,
        "observed_noisy": observed_noisy,
        "wavenumbers": wavenumbers,
        "time_seconds": actual_time,
        "decay_rates": decay_rates,
        "abundances": abundances,
        "time_constants": 1.0 / decay_rates,
        "frame_duration": frame_duration,
    }
    result.update(fluorophores)

    return result


def plot_decomposition(
    ds: xr.Dataset,
    sample_idx: int,
    time_seconds: float,
    physics_model: str,
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
    decomp = get_full_decomposition(ds, sample_idx, time_seconds, physics_model)
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
            label=f"F{i + 1}: τ={τ:.3f}s, w={w:.1f}",
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
    physics_model: str,
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

    # Infer frame duration from time axis
    if len(time_values) > 1:
        frame_duration = float(time_values[1] - time_values[0])
    else:
        frame_duration = 1.0

    # Top left: Observed spectra over time
    ax = axes[0, 0]
    for t_idx, t in enumerate(time_values):
        spectrum = (
            ds["intensity_clean"].isel(sample=sample_idx, bleaching_time=t_idx).values
        )
        ax.plot(wn, spectrum, color=time_colors[t_idx], alpha=0.8, label=f"t={t:.2f}s")

    # Raman GT scaled by frame_duration to match integrated observed spectra
    ax.plot(
        wn,
        ds["raman_gt"].isel(sample=sample_idx).values * frame_duration,
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
        fluor = get_total_fluorescence(ds, sample_idx, t, physics_model)
        ax.plot(wn, fluor, color=time_colors[t_idx], alpha=0.8, label=f"t={t:.2f}s")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Total Fluorescence Decay")
    # ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Individual fluorophore decay curves (integrated model)
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

        # Per-fluorophore contribution over all timepoints via physics function
        fluor_series = reconstruct_time_series_numpy(
            raman=np.zeros_like(B_i),
            bases=B_i[np.newaxis, :],
            abundances=np.array([abundances[i]]),
            decay_rates=np.array([decay_rates[i]]),
            time_values=time_values,
            frame_duration=frame_duration,
            physics_model=physics_model,
        )  # [T, W]
        intensities = fluor_series.mean(axis=1)  # mean across wavenumbers

        τ = 1.0 / decay_rates[i]
        ax.plot(
            time_values,
            intensities,
            "o-",
            color=fluor_colors[i],
            label=f"F{i + 1}: τ={τ:.3f}s",
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
            wn,
            B_i,
            color=fluor_colors[i],
            linewidth=1.5,
            label=f"B{i + 1} (τ={τ:.3f}s)",
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
    data: Union[np.ndarray, SpectralData],
    decomposition: DecompositionResult,
    subsample_wn: int = 2,
    subsample_time: int = 1,
):
    """
    Interactive 3D visualisation using plotly (allows rotation/zoom).

    Args:
        data: Original time series. Can be:
            - np.ndarray of shape (n_timepoints, n_wavenumbers)
            - SpectralData object with time_values
        decomposition: Dictionary with keys:
            - 'raman': Extracted Raman spectrum (n_wavenumbers,) or SpectralData
            - 'fluorophore_bases': Fluorophore bases (n_fluorophores, n_wavenumbers) or SpectralData
            - 'abundances': Abundances (n_fluorophores,)
            - 'rates' or 'decay_rates': Decay rates (n_fluorophores,)
        reconstruction: Reconstructed time series. If None, computed from decomposition.
        time_values: Time axis in seconds. If None, extracted from data or uses frame indices.
        wavenumbers: Wavenumber axis. If None, extracted from data or uses indices.
        subsample_wn: Subsample factor for wavenumber axis
        subsample_time: Subsample factor for time axis

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go

    raman = decomposition.raman.intensities
    Y = data.intensities
    n_t, n_wn = Y.shape

    # Handle optional axes
    time_values = data.time_values

    bases = decomposition.fluorophore_spectra.intensities

    # # Extract intensities (support both SpectralData and np.ndarray)
    # if isinstance(raman_obj, SpectralData):
    #     raman = raman_obj.intensities
    # else:
    #     raman = raman_obj

    # if isinstance(bases_obj, SpectralData):
    #     bases = bases_obj.intensities
    # else:
    #     bases = bases_obj

    abundances = decomposition.abundances
    rates = decomposition.rates
    time_constants = 1.0 / rates

    n_fluorophores = len(rates)
    wavenumbers = data.wavenumbers
    # Compute reconstruction if not provided
    # if reconstruction is None:
    #     reconstruction = np.tile(raman, (n_t, 1))
    #     for i in range(n_fluorophores):
    #         decay = np.exp(-rates[i] * time_values)
    #         reconstruction = (
    #             reconstruction + abundances[i] * decay[:, None] * bases[i, None, :]
    #         )
    # reconstruction = decomposition.reconstruction(time_values)
    reconstruction = decomposition.reconstruction(time_values)

    # Compute total fluorescence (integrated model)
    frame_dur = getattr(decomposition, "frame_duration", None)
    if frame_dur is None and len(time_values) > 1:
        frame_dur = float(time_values[1] - time_values[0])
    if frame_dur is None:
        frame_dur = 1.0

    # Total fluorescence = reconstruction - raman contribution (counts/sec → counts/frame)
    raman_per_frame = raman * frame_dur
    total_fluor = reconstruction - np.tile(raman_per_frame, (n_t, 1))

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
            z=Y[np.ix_(t_idx, wn_idx)],
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
    residual = np.square((Y - reconstruction))
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


def plot_uncertainty(
    ensemble: dict,
    wavenumbers: Optional[np.ndarray] = None,
    time_values: Optional[np.ndarray] = None,
    reference_raman: Optional[np.ndarray] = None,
    reference_bases: Optional[np.ndarray] = None,
    reference_abundances: Optional[np.ndarray] = None,
    sample_id: Optional[str] = None,
    ci: float = 0.95,
):
    """
    Visualise uncertainty across N stochastic VAE forward passes as heatmaps.

    Args:
        ensemble:             Output of sample_posterior() — dict with keys
                              'raman' [N,W], 'rates' [N,F], 'abundances' [N,F],
                              'bases' [N,F,W], 'abundance_times_basis' [N,F,W],
                              'reconstruction' [N,T,W].
        wavenumbers:          Wavenumber axis [W].
        time_values:          Time axis [T].
        reference_raman:      GT Raman spectrum [W] for comparison.
        reference_bases:      GT fluorophore basis spectra [n_gt, W].
        reference_abundances: GT abundances [n_gt] for scaling GT bases.
        sample_id:            Label for suptitle.
        ci:                   Credible interval width (default 0.95 → 2.5/97.5 percentiles).

    Returns:
        (fig_ens, fig_comps):
            fig_ens   — Raman ensemble heatmap [N×W] + mean/GT line + recon std [W×T]
            fig_comps — Per-component abundance×basis heatmap [N×W] with GT overlay
    """
    lo_p = 100 * (1 - ci) / 2
    hi_p = 100 - lo_p

    raman_ens = ensemble["raman"]  # [N, W]
    recon_ens = ensemble["reconstruction"]  # [N, T, W]
    ab_basis = ensemble.get("abundance_times_basis")  # [N, F, W] or None

    N = raman_ens.shape[0]
    W = raman_ens.shape[1]
    T = recon_ens.shape[1]
    F = ab_basis.shape[1] if ab_basis is not None else 0

    if wavenumbers is None:
        wavenumbers = np.arange(W)
    if time_values is None:
        time_values = np.arange(T)

    r_mean = raman_ens.mean(axis=0)

    title_base = f"N={N}   {int(ci*100)}% CI"
    if sample_id is not None:
        title_base = f"Sample: {sample_id}   |   {title_base}"

    def _density_heatmap(
        ax, fig, ens, wavenumbers, reference=None, ylabel="counts/sec", title=""
    ):
        """2D density heatmap: x=wavenumber, y=intensity, colour=sample count."""
        n_bins = max(50, N)
        # Include GT in y-range so it's never clipped
        all_vals = [ens.min(), ens.max()]
        if reference is not None:
            all_vals += [reference.min(), reference.max()]
        i_min, i_max = min(all_vals), max(all_vals)
        i_pad = (i_max - i_min) * 0.05
        i_edges = np.linspace(i_min - i_pad, i_max + i_pad, n_bins + 1)
        density = np.zeros((n_bins, len(wavenumbers)), dtype=np.float32)
        for w in range(len(wavenumbers)):
            density[:, w], _ = np.histogram(ens[:, w], bins=i_edges)
        im = ax.imshow(
            density,
            aspect="auto",
            origin="lower",
            extent=[wavenumbers[0], wavenumbers[-1], i_edges[0], i_edges[-1]],
            cmap="hot",
            interpolation="bilinear",
        )
        fig.colorbar(im, ax=ax, label="sample count", pad=0.02)
        mean_line = ens.mean(axis=0)
        ax.plot(wavenumbers, mean_line, color="cyan", linewidth=1.5, label="Mean")
        if reference is not None:
            ax.plot(
                wavenumbers,
                reference,
                color="lime",
                linewidth=1.5,
                linestyle="--",
                label="GT",
            )
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    # ─────────────────────────────────────────────────────────────────────────
    # fig_ens: Raman density heatmap + recon std heatmap
    # ─────────────────────────────────────────────────────────────────────────
    fig_ens, (ax_rdens, ax_rstd) = plt.subplots(1, 2, figsize=(14, 5))

    _density_heatmap(
        ax_rdens,
        fig_ens,
        raman_ens,
        wavenumbers,
        reference=reference_raman,
        ylabel="counts/sec",
        title=f"Raman Intensity Distribution ({N} samples)",
    )

    recon_std = recon_ens.std(axis=0)  # [T, W]
    im_std = ax_rstd.imshow(
        recon_std.T,  # [W, T]
        aspect="auto",
        origin="lower",
        extent=[time_values[0], time_values[-1], wavenumbers[0], wavenumbers[-1]],
        cmap="hot",
        interpolation="nearest",
    )
    fig_ens.colorbar(im_std, ax=ax_rstd, label="Std (counts/frame)", pad=0.02)
    ax_rstd.set_xlabel("Time (s)")
    ax_rstd.set_ylabel("Wavenumber (cm⁻¹)")
    ax_rstd.set_title(f"Reconstruction Std ({N} samples)")

    fig_ens.suptitle(title_base, fontsize=10)
    fig_ens.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # fig_comps: Per-component abundance×basis density heatmap
    # ─────────────────────────────────────────────────────────────────────────
    if ab_basis is None or F == 0:
        fig_comps = plt.figure(figsize=(6, 3))
        fig_comps.text(0.5, 0.5, "No abundance×basis data", ha="center", va="center")
        return fig_ens, fig_comps

    ncols = min(F, 3)
    nrows = -(-F // ncols)  # ceil division

    fig_comps, axes_c = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    # Greedy basis-correlation matching: mean predicted basis vs GT bases
    pred_to_gt: dict = {}
    if reference_bases is not None and reference_abundances is not None:
        mean_bases = ab_basis.mean(axis=0)  # [F, W]
        n_gt = len(reference_bases)
        corr_mat = np.zeros((F, n_gt))
        for i in range(F):
            for j in range(n_gt):
                c = np.corrcoef(mean_bases[i], reference_bases[j])[0, 1]
                corr_mat[i, j] = c if np.isfinite(c) else 0.0
        used: set = set()
        for p in np.argsort(-corr_mat.max(axis=1)):
            avail = [(j, corr_mat[p, j]) for j in range(n_gt) if j not in used]
            if avail:
                best_j, _ = max(avail, key=lambda x: x[1])
                pred_to_gt[p] = best_j
                used.add(best_j)

    for f in range(F):
        row, col = divmod(f, ncols)
        ax = axes_c[row, col]

        comp_ens = ab_basis[:, f, :]  # [N, W]
        gt_comp = None
        gt_j = pred_to_gt.get(f)
        if gt_j is not None:
            gt_comp = reference_bases[gt_j] * reference_abundances[gt_j]

        _density_heatmap(
            ax,
            fig_comps,
            comp_ens,
            wavenumbers,
            reference=gt_comp,
            ylabel="counts",
            title=f"Component {f + 1}"
            + (f"  →  GT {gt_j + 1}" if gt_j is not None else ""),
        )

    # Hide unused subplots
    for f in range(F, nrows * ncols):
        row, col = divmod(f, ncols)
        axes_c[row, col].set_visible(False)

    fig_comps.suptitle(f"Abundance × Basis Components   |   {title_base}", fontsize=10)
    fig_comps.tight_layout()

    return fig_ens, fig_comps


def plot_raman_posterior(
    ensemble: dict,
    wavenumbers: np.ndarray,
    reference_raman: Optional[np.ndarray] = None,
    frame_duration: float = 0.1,
    percentiles: Tuple[float, float, float] = (5.0, 50.0, 95.0),
    n_sigma: float = 1.0,
    sample_alpha: float = 0.15,
    figsize: Tuple[int, int] = (8, 5),
    sample_id: Optional[str] = None,
) -> Tuple[Figure, Figure, Figure]:
    """
    Three separate figures summarising the Raman posterior over N stochastic draws.

    All values are in counts/frame (ensemble['raman'] * frame_duration).

    Returns
    -------
    fig_raw : Figure
        N semi-transparent posterior traces. Shows full marginal distribution
        without any summary statistic; useful for detecting multi-modality.
    fig_overlay : Figure
        Same traces with the percentile band and posterior mean overlaid.
        Lets you see how the summary statistics relate to the individual draws.
    fig_tube : Figure
        Clean credible-tube summary: fill_between [p_lo, p_hi] band,
        ±n_sigma·std tube, posterior mean, and median on one axis.

    Parameters
    ----------
    ensemble : dict
        Output of sample_posterior(). Key 'raman' [N, W] in counts/sec.
    wavenumbers : (W,)
    reference_raman : (W,) counts/sec, optional.
    frame_duration : float
        Converts counts/sec → counts/frame.
    percentiles : (p_lo, p_mid, p_hi)
        Band edges and median line. Default (5, 50, 95) = 90% credible interval.
    n_sigma : float
        Half-width multiplier for the std tube in fig_tube.
    sample_alpha : float
        Per-trace opacity. 200 traces at 0.15 gives good density impression.
    """
    raman = ensemble["raman"] * frame_duration  # [N, W] counts/frame
    ref = reference_raman * frame_duration if reference_raman is not None else None

    p_lo, p_mid, p_hi = percentiles
    q_lo = np.percentile(raman, p_lo, axis=0)  # (W,)
    median = np.percentile(raman, p_mid, axis=0)  # (W,)
    q_hi = np.percentile(raman, p_hi, axis=0)  # (W,)
    mean = raman.mean(axis=0)  # (W,)
    std = raman.std(axis=0, ddof=1)  # (W,)

    N = len(raman)
    _BLUE = "#4477AA"
    _REF = "crimson"

    title_base = f"N={N}  |  p{p_lo:.0f}/p{p_mid:.0f}/p{p_hi:.0f}"
    if sample_id is not None:
        title_base = f"Sample {sample_id}  |  {title_base}"

    def _ref_line(ax):
        if ref is not None:
            ax.plot(
                wavenumbers,
                ref,
                color=_REF,
                linewidth=1.8,
                linestyle="--",
                label="Ground truth",
                zorder=7,
            )

    def _axis_labels(ax):
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (counts/frame)")
        ax.grid(True, alpha=0.3)

    # ── Fig 1: raw draws ──────────────────────────────────────────────────────
    fig_raw, ax = plt.subplots(figsize=figsize)
    for trace in raman:
        ax.plot(wavenumbers, trace, color=_BLUE, alpha=sample_alpha, linewidth=0.8)
    _ref_line(ax)
    _axis_labels(ax)
    ax.set_title(f"Posterior draws  |  {title_base}")
    if ref is not None:
        ax.legend(fontsize=8)
    fig_raw.tight_layout()

    # ── Fig 2: draws + band overlay ───────────────────────────────────────────
    fig_overlay, ax = plt.subplots(figsize=figsize)
    for trace in raman:
        ax.plot(wavenumbers, trace, color=_BLUE, alpha=sample_alpha, linewidth=0.8)
    ax.fill_between(
        wavenumbers,
        q_lo,
        q_hi,
        alpha=0.30,
        color=_BLUE,
        label=f"[p{p_lo:.0f}, p{p_hi:.0f}] band",
    )
    ax.plot(
        wavenumbers,
        mean,
        color=_BLUE,
        linewidth=2.0,
        label="Posterior mean",
        zorder=6,
        alpha=0.1,
    )
    _ref_line(ax)
    _axis_labels(ax)
    ax.set_title(f"Draws + credible band  |  {title_base}")
    ax.legend(fontsize=8)
    fig_overlay.tight_layout()

    # ── Fig 3: clean credible tube ────────────────────────────────────────────
    fig_tube, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        wavenumbers,
        q_lo,
        q_hi,
        alpha=0.20,
        color=_BLUE,
        label=f"[p{p_lo:.0f}, p{p_hi:.0f}] band",
    )
    ax.fill_between(
        wavenumbers,
        mean - n_sigma * std,
        mean + n_sigma * std,
        alpha=0.35,
        color=_BLUE,
        label=f"Mean ± {n_sigma:.0f}σ",
    )
    ax.plot(wavenumbers, mean, color=_BLUE, linewidth=2.0, label="Posterior mean")
    ax.plot(
        wavenumbers,
        median,
        color="darkorange",
        linewidth=1.5,
        linestyle="-.",
        label=f"Median (p{p_mid:.0f})",
        zorder=5,
    )
    _ref_line(ax)
    _axis_labels(ax)
    ax.set_title(f"Credible tube  |  {title_base}")
    ax.legend(fontsize=8)
    fig_tube.tight_layout()

    # ── Posterior summary metrics ─────────────────────────────────────────────
    print(f"\nRaman posterior (N={N}):")
    print(f"  Std range:           [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Mean CV (std/mean):  {(std.mean() / (mean.mean() + 1e-12)):.4f}")
    if ref is not None:
        r = np.corrcoef(mean, ref)[0, 1]
        rmse = np.sqrt(np.mean((mean - ref) ** 2))
        print(f"  Pearson r (mean vs GT):  {r:.4f}")
        print(f"  RMSE      (mean vs GT):  {rmse:.4f}")

    return fig_raw, fig_overlay, fig_tube


def plot_raman_credible_band(
    ensemble: dict,
    wavenumbers: np.ndarray,
    reference_raman: Optional[np.ndarray] = None,
    frame_duration: float = 0.1,
    ci: float = 0.90,
    sample_id: Optional[str] = None,
    figsize: tuple = (7, 4),
) -> "Figure":
    """
    Publication-quality Raman posterior summary: single figure, two panels.

    Left panel — credible band:
        Shaded ``ci`` interval (e.g. 90 % → 5th–95th percentile) + median.
        No individual traces. Optionally overlays a ground-truth reference.

    Right panel — uncertainty profile:
        Posterior standard deviation (σ) at each wavenumber, revealing where
        the model is most uncertain (peak positions vs. baseline).

    Parameters
    ----------
    ensemble : dict
        Output of ``sample_posterior()``. Key ``'raman'`` is [N, W] in counts/sec.
    wavenumbers : (W,)
    reference_raman : (W,) counts/sec, optional.
    frame_duration : float
        Converts counts/sec → counts/frame for display.
    ci : float
        Credible interval width in (0, 1). Default 0.90 → 5th/95th percentiles.
    sample_id : str, optional
        Added to the suptitle.
    figsize : (w, h)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    raman = ensemble["raman"] * frame_duration  # [N, W] counts/frame
    ref = reference_raman * frame_duration if reference_raman is not None else None

    p_lo = 100 * (1 - ci) / 2
    p_hi = 100 - p_lo
    q_lo = np.percentile(raman, p_lo, axis=0)
    median = np.percentile(raman, 50.0, axis=0)
    q_hi = np.percentile(raman, p_hi, axis=0)
    std = raman.std(axis=0, ddof=1)
    N = len(raman)

    _BLUE = "#4477AA"
    _REF = "#BB5566"

    def _style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)

    fig, (ax_band, ax_std) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: credible band ───────────────────────────────────────────────────

    ax_band.plot(wavenumbers, median, color=_BLUE, linewidth=1.8, label="Median")
    if ref is not None:
        ax_band.plot(
            wavenumbers,
            ref,
            color=_REF,
            linewidth=1.5,
            linestyle="--",
            label="Ground truth",
            zorder=5,
        )
    _style(ax_band)
    ax_band.fill_between(
        wavenumbers,
        q_lo,
        q_hi,
        alpha=0.25,
        color=_BLUE,
        label=f"{int(ci*100):d}% CI  (p{p_lo:.0f}–p{p_hi:.0f})",
    )
    ax_band.set_ylabel("Intensity (counts/frame)", fontsize=9)
    ax_band.set_title(f"Raman posterior  (N={N})", fontsize=9, fontweight="bold")
    ax_band.legend(fontsize=8, frameon=False)

    # ── Right: uncertainty profile ────────────────────────────────────────────
    ax_std.fill_between(wavenumbers, 0, std, alpha=0.30, color=_BLUE)
    ax_std.plot(wavenumbers, std, color=_BLUE, linewidth=1.5)
    _style(ax_std)
    ax_std.set_ylabel("Posterior σ (counts/frame)", fontsize=9)
    ax_std.set_title("Spectral uncertainty", fontsize=9, fontweight="bold")
    ax_std.set_ylim(bottom=0)

    # ── Posterior summary ─────────────────────────────────────────────────────
    cv = std.mean() / (raman.mean() + 1e-12)
    print(
        f"Raman posterior  N={N}  |  CV={cv:.3f}  |  "
        f"σ range=[{std.min():.4f}, {std.max():.4f}]"
    )
    if ref is not None:
        from scipy.stats import pearsonr

        r, _ = pearsonr(median, ref)
        print(f"Pearson r (median vs GT): {r:.4f}")

    suptitle = f"Sample {sample_id}" if sample_id else "Raman posterior"
    fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# =============================================================================
# Helpers: construct DecompositionResult / SpectralData from xarray datasets
# =============================================================================
#
# Two dataset schemas are supported:
#
#   generate  — produced by SyntheticBleachingDataset.generate()
#               intensity dims : [sample, bleaching_time, wavenumber]  → [T, W]
#               time coord     : "bleaching_time"
#               Raman GT       : "raman_gt"
#
#   pipeline  — produced by pipeline.process_and_export_dataset()
#               intensity dims : [sample, wavenumber, time]             → [W, T] → .T
#               time coord     : "time"
#               Raman GT       : "gt_raman"
#               frame duration : attrs["frame_duration_s"]
#
# Both share: coords["wavenumber"], "decay_rates_gt", "abundances_gt",
#             "fluorophore_bases_gt", "intensity_clean" (synthetic only).
# =============================================================================


def _ds_time_coord(ds: "xr.Dataset") -> np.ndarray:
    """Return the time axis regardless of whether it is called 'bleaching_time' or 'time'."""
    for name in ("bleaching_time", "time"):
        if name in ds.coords:
            return ds.coords[name].values
    raise KeyError(
        "Dataset has neither a 'bleaching_time' nor a 'time' coordinate. "
        f"Available coords: {list(ds.coords)}"
    )


def _ds_frame_duration(ds: "xr.Dataset") -> float:
    """Return frame duration from attrs (pipeline) or inferred from time coord (generate)."""
    if "frame_duration_s" in ds.attrs:
        return float(ds.attrs["frame_duration_s"])
    t = _ds_time_coord(ds)
    return float(t[1] - t[0]) if len(t) > 1 else 0.1


def _ds_intensity(ds: "xr.Dataset", sample_idx: int, use_clean: bool) -> np.ndarray:
    """
    Extract the ``[T, W]`` intensity array for one sample, handling both
    dimension orderings.

    Selection priority
    ------------------
    use_clean=True  → ``intensity_clean``  (both schemas; falls back to noisy)
    use_clean=False → ``intensity_raw``    (generate schema)
                   → ``time_series``       (pipeline schema)
                   → ``intensity_clean``   (last resort)
    """
    available = set(ds.data_vars)

    if use_clean:
        candidates = ["intensity_clean", "intensity_raw", "time_series"]
    else:
        candidates = ["intensity_raw", "time_series", "intensity_clean"]

    key = next((k for k in candidates if k in available), None)
    if key is None:
        raise KeyError(
            f"No intensity variable found in dataset. Available: {list(available)}"
        )

    var = ds[key].isel(sample=sample_idx)
    arr = var.values  # either [T, W] or [W, T]

    # Detect axis order from dimension names
    dims = var.dims
    if dims[0] in ("bleaching_time", "time"):
        return arr  # already [T, W]
    else:
        return arr.T  # [W, T] → [T, W]


def _ds_raman_gt(ds: "xr.Dataset", sample_idx: int, wn: np.ndarray) -> np.ndarray:
    """Return [W] Raman GT, trying 'raman_gt' then 'gt_raman'."""
    for key in ("raman_gt", "gt_raman"):
        if key in ds:
            return ds[key].isel(sample=sample_idx).values
    raise KeyError(
        "No Raman GT variable found. Expected 'raman_gt' (generate schema) "
        f"or 'gt_raman' (pipeline schema). Available: {list(ds.data_vars)}"
    )


def data_from_dataset(
    ds: "xr.Dataset",
    sample_idx: int = 0,
    use_clean: bool = False,
) -> "SpectralData":
    """
    Extract a single observed sample from an xarray Dataset as a
    :class:`SpectralData` object ready for plotting.

    Works with both the **generate** schema (``SyntheticBleachingDataset``)
    and the **pipeline** schema (``process_and_export_dataset``).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from either ``SyntheticBleachingDataset.generate()`` or
        ``pipeline.process_and_export_dataset()``.
    sample_idx : int
        Index along the ``sample`` dimension.
    use_clean : bool
        Prefer ``intensity_clean`` (noise-free forward model) over the noisy
        observed frames.  Falls back gracefully if not present.

    Returns
    -------
    SpectralData  shape ``(T, W)``, ``time_values`` set.
    """
    Y = _ds_intensity(ds, sample_idx, use_clean)
    t = _ds_time_coord(ds)
    wn = ds.coords["wavenumber"].values
    return SpectralData(Y, wn, time_values=t)


# Keep old name as an alias so existing notebooks don't break.
data_from_gt_dataset = data_from_dataset


def decomp_from_gt_dataset(
    ds: "xr.Dataset",
    sample_idx: int = 0,
    physics_model: str = "integrated",
) -> "DecompositionResult":
    """
    Build a :class:`DecompositionResult` from the ground-truth variables in an
    xarray Dataset — no model inference needed.

    Works with both the **generate** schema (``SyntheticBleachingDataset``)
    and the **pipeline** schema (``process_and_export_dataset`` + synthetic GT).

    Parameters
    ----------
    ds : xr.Dataset
        Must contain ``decay_rates_gt``, ``abundances_gt``,
        ``fluorophore_bases_gt``, and either ``raman_gt`` (generate) or
        ``gt_raman`` (pipeline).
    sample_idx : int
        Index along the ``sample`` dimension.
    physics_model : str
        Physics model tag (default ``"integrated"``).

    Returns
    -------
    DecompositionResult
    """
    from ramanlib.bleaching.decompose import DecompositionResult

    wn = ds.coords["wavenumber"].values
    frame_dur = _ds_frame_duration(ds)

    raman_arr = _ds_raman_gt(ds, sample_idx, wn)  # [W]
    rates_arr = ds["decay_rates_gt"].isel(sample=sample_idx).values  # [F]
    abund_arr = ds["abundances_gt"].isel(sample=sample_idx).values  # [F]

    bases_var = ds["fluorophore_bases_gt"]
    if "sample" in bases_var.dims:
        bases_arr = bases_var.isel(sample=sample_idx).values  # [F, W]
    else:
        bases_arr = bases_var.values  # [F, W] shared

    return DecompositionResult(
        raman=SpectralData(raman_arr, wn),
        rates=rates_arr,
        fluorophore_spectra=SpectralData(bases_arr, wn),
        abundances=abund_arr,
        physics_model=physics_model,
        frame_duration=frame_dur,
    )


# =============================================================================
# NeurIPS-quality 3D component decomposition figure
# =============================================================================

_NEURIPS_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "text.usetex": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "figure.facecolor": "white",
}

# =============================================================================
# LUMOS custom colormaps
# =============================================================================
# All run from a rich dark anchor at 0 → saturated mid → muted pastel at max.
# This means:
#   • Decayed regions (low intensity) → distinctive dark colour, never white
#   • Peak regions → professional pastel highlight
#   • Each component has a clearly distinct hue family
#
# Stops: list of (position 0–1, hex colour)
# =============================================================================
from matplotlib.colors import LinearSegmentedColormap as _LSC

_STOP_OBSERVED = [  # teal-green perceptual sequence (≈ viridis flavour)
    (0.00, "#0d0221"),  # near-black purple
    (0.25, "#1b4f72"),  # deep ocean blue
    (0.55, "#1abc9c"),  # medium teal
    (0.80, "#76d7c4"),  # soft mint
    (1.00, "#d5f5e3"),  # pale mint (NOT white)
]
_STOP_RAMAN = [  # slate-blue family  (cool, steady — Raman is constant)
    (0.00, "#0b1629"),  # near-black navy
    (0.30, "#1a3a6e"),  # deep navy
    (0.60, "#4a86c8"),  # mid cornflower blue
    (0.85, "#93bce0"),  # powder blue
    (1.00, "#d6e9f8"),  # pale sky (NOT white)
]
_STOP_FLUOR = [  # amber-rose family  (warm, decaying)
    (0.00, "#2b0a0a"),  # near-black burgundy
    (0.30, "#8b1a1a"),  # deep crimson
    (0.60, "#e06c3a"),  # warm amber-orange
    (0.85, "#f4a97c"),  # light peach
    (1.00, "#fde8d8"),  # pale blush (NOT white)
]
_STOP_NOISE = None  # diverging — built separately below
_STOP_F = [  # per-fluorophore: 5 distinct hue families
    [  # F1 — violet / lavender
        (0.00, "#120922"),
        (0.30, "#4a1a7a"),
        (0.60, "#8e5bbf"),
        (0.85, "#c39ee0"),
        (1.00, "#ead6f5"),
    ],
    [  # F2 — emerald / sage
        (0.00, "#041a10"),
        (0.30, "#0e6644"),
        (0.60, "#27ae7a"),
        (0.85, "#7dcfaf"),
        (1.00, "#c8f0e0"),
    ],
    [  # F3 — rose / coral
        (0.00, "#1f0510"),
        (0.30, "#7b1450"),
        (0.60, "#c94f8a"),
        (0.85, "#e89ec4"),
        (1.00, "#f8d7ea"),
    ],
    [  # F4 — gold / amber
        (0.00, "#1a0f00"),
        (0.30, "#7a4000"),
        (0.60, "#d4820a"),
        (0.85, "#f0bc6a"),
        (1.00, "#fde9b8"),
    ],
    [  # F5 — teal / cyan
        (0.00, "#011318"),
        (0.30, "#0a4f5e"),
        (0.60, "#1a9aaa"),
        (0.85, "#72ccd6"),
        (1.00, "#c5eef2"),
    ],
]


def _stops_to_mpl(name: str, stops: list) -> _LSC:
    """Build a matplotlib LinearSegmentedColormap from (pos, hex) stops."""
    colours = [c for _, c in stops]
    positions = [p for p, _ in stops]
    from matplotlib.colors import to_rgb

    nodes = list(zip(positions, [to_rgb(c) for c in colours]))
    return _LSC.from_list(name, nodes)


def _stops_to_plotly(stops: list) -> list:
    """Convert (pos, hex) stops to a Plotly colorscale list."""
    from matplotlib.colors import to_rgb

    out = []
    for pos, hexc in stops:
        r, g, b = [int(x * 255) for x in to_rgb(hexc)]
        out.append([pos, f"rgb({r},{g},{b})"])
    return out


# Register matplotlib colormaps (safe to call multiple times)
_CM_OBSERVED = _stops_to_mpl("lumos_observed", _STOP_OBSERVED)
_CM_RAMAN = _stops_to_mpl("lumos_raman", _STOP_RAMAN)
_CM_FLUOR = _stops_to_mpl("lumos_fluor", _STOP_FLUOR)
_CM_FS = [_stops_to_mpl(f"lumos_f{i+1}", s) for i, s in enumerate(_STOP_F)]

# Diverging noise map: deep teal ← 0 → muted rose  (stays off-white at centre)
_CM_NOISE = _LSC.from_list(
    "lumos_noise",
    [
        (0.0, "#0b3d40"),
        (0.35, "#2a9da8"),
        (0.5, "#e8e8ee"),
        (0.65, "#c86090"),
        (1.0, "#4a0a26"),
    ],
)

for _cm in [_CM_OBSERVED, _CM_RAMAN, _CM_FLUOR, _CM_NOISE] + _CM_FS:
    try:
        plt.colormaps.register(_cm, force=True)
    except AttributeError:
        import matplotlib as _mpl

        _mpl.cm.register_cmap(cmap=_cm)

# Plotly equivalents (built from the same stops)
_PX_OBSERVED = _stops_to_plotly(_STOP_OBSERVED)
_PX_RAMAN = _stops_to_plotly(_STOP_RAMAN)
_PX_FLUOR = _stops_to_plotly(_STOP_FLUOR)
_PX_NOISE = [
    [0.0, "rgb(11,61,64)"],
    [0.35, "rgb(42,157,168)"],
    [0.5, "rgb(232,232,238)"],
    [0.65, "rgb(200,96,144)"],
    [1.0, "rgb(74,10,38)"],
]
_PX_FS = [_stops_to_plotly(s) for s in _STOP_F]

# Name → Plotly colorscale registry (handles both custom and built-in names)
_MPL_TO_PLOTLY_SCALE: dict = {
    "lumos_observed": _PX_OBSERVED,
    "lumos_raman": _PX_RAMAN,
    "lumos_fluor": _PX_FLUOR,
    "lumos_noise": _PX_NOISE,
    **{f"lumos_f{i+1}": _PX_FS[i] for i in range(len(_PX_FS))},
    # Built-in pass-throughs (kept for backwards compat / user overrides)
    "viridis": "Viridis",
    "magma": "Magma",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "cividis": "Cividis",
    "RdBu": "RdBu",
    "Blues": "Blues",
    "Oranges": "Oranges",
}

# Names used by _FLUORO_CMAPS (cycled across fluorophores)
_FLUORO_CMAPS = [f"lumos_f{i+1}" for i in range(len(_STOP_F))]

# Shared scene styling applied to every Plotly 3-D subplot.
# Axis convention: X = Wavenumber (cm⁻¹), Y = Time (s, t=0 at front),
#                  Z = Intensity.  Camera is front-right-above so the
#                  spectrum runs left→right and the decay recedes backward.
_PLOTLY_SCENE: dict = dict(
    xaxis=dict(
        title="Wavenumber (cm⁻¹)",
        showgrid=True,
        gridcolor="rgba(180,180,190,0.5)",
        backgroundcolor="rgba(235,237,242,0.5)",
        showbackground=True,
        tickfont=dict(size=11, color="#333"),
        titlefont=dict(size=13, color="#111"),
        ticks="outside",
        ticklen=4,
    ),
    yaxis=dict(
        title="Time (s)",
        showgrid=True,
        gridcolor="rgba(180,180,190,0.5)",
        backgroundcolor="rgba(235,237,242,0.5)",
        showbackground=True,
        tickfont=dict(size=11, color="#333"),
        titlefont=dict(size=13, color="#111"),
        ticks="outside",
        ticklen=4,
    ),
    zaxis=dict(
        title="Intensity",
        showgrid=True,
        gridcolor="rgba(180,180,190,0.5)",
        backgroundcolor="rgba(242,242,246,0.6)",
        showbackground=True,
        tickfont=dict(size=11, color="#333"),
        titlefont=dict(size=13, color="#111"),
        ticks="outside",
        ticklen=4,
    ),
    # Front-right-above: spectrum runs left→right, decay goes into the page
    camera=dict(eye=dict(x=1.6, y=-1.8, z=1.1)),
    aspectmode="auto",
)


def _mpl_to_plotly_cmap(name: str):
    """Return a Plotly colorscale (string name or [[pos, colour]] list)."""
    return _MPL_TO_PLOTLY_SCALE.get(name, name)


def _plotly_scene(camera_eye: Optional[dict] = None) -> dict:
    """Return a copy of _PLOTLY_SCENE, optionally overriding the camera."""
    import copy

    s = copy.deepcopy(_PLOTLY_SCENE)
    if camera_eye is not None:
        s["camera"]["eye"] = camera_eye
    return s


def _compute_fluorophore_surface(
    bases: np.ndarray,
    abundances: np.ndarray,
    rates: np.ndarray,
    time_values: np.ndarray,
    frame_duration: float,
    idx: int,
) -> np.ndarray:
    """Return the [T, W] intensity surface for fluorophore *idx* (CCD-integrated)."""
    lam = rates[idx]
    T = frame_duration
    ccd = (1.0 - np.exp(-lam * T)) / lam if lam > 1e-12 else T
    decay = abundances[idx] * ccd * np.exp(-lam * time_values)  # [T]
    return np.outer(decay, bases[idx])  # [T, W]


def _style_3d_ax(
    ax,
    title: str,
    label_fontsize: int,
    title_fontsize: int,
) -> None:
    from matplotlib.ticker import MaxNLocator

    # 1. Disable Matplotlib's automatic 3D slanting/rotation
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    # 2. Set labels with \mathrm{} to ensure the math text is upright, not italic
    ax.set_xlabel(
        "\nWavenumber ($\mathrm{cm^{-1}}$)",
        fontsize=label_fontsize,
        labelpad=10,
        rotation=0,
    )
    ax.set_ylabel("\nTime (s)", fontsize=label_fontsize, labelpad=10, rotation=0)
    ax.set_zlabel("Intensity", fontsize=label_fontsize, labelpad=10, rotation=90)

    ax.set_title(title, fontsize=title_fontsize, pad=15, fontweight="bold")
    ax.tick_params(labelsize=label_fontsize - 1, pad=3)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    ax.grid(True, alpha=0.3, linewidth=0.5, color="#888888")


def _plot_components_3d_mpl(
    wn_sub: np.ndarray,
    t_sub: np.ndarray,
    main_panels: list,
    fluoro_panels: list,
    elev: float,
    azim: float,
    figsize_per_panel: Tuple[float, float],
    dpi: int,
    label_fontsize: int,
    title_fontsize: int,
    show_colorbar: bool,
) -> "Figure":
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_main = len(main_panels)
    n_fluoro = len(fluoro_panels)
    n_rows = 1 + (1 if n_fluoro > 0 else 0)
    n_cols = max(n_main, n_fluoro) if n_fluoro > 0 else n_main

    WN, TG = np.meshgrid(wn_sub, t_sub)
    rcount = len(t_sub)
    ccount = len(wn_sub)

    fw = figsize_per_panel[0] * n_cols
    fh = figsize_per_panel[1] * n_rows

    with plt.rc_context(_NEURIPS_RC):
        fig = plt.figure(figsize=(fw, fh), dpi=dpi, facecolor="white")

        def _add(row: int, col: int, title: str, Z: np.ndarray, cmap: str) -> None:
            ax = fig.add_subplot(
                n_rows, n_cols, (row - 1) * n_cols + col, projection="3d"
            )

            # ── THE "SOLID WALLS" TRICK ──
            z_floor = 0.0  # The baseline level where the walls will hit the floor

            # 1. Pad the intensity array with zeros around all 4 edges
            Z_pad = np.pad(
                Z, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=z_floor
            )

            # 2. Pad the coordinate axes by duplicating the first and last values
            # This ensures the drop to the floor is perfectly vertical
            wn_pad = np.pad(wn_sub, pad_width=(1, 1), mode="edge")
            t_pad = np.pad(t_sub, pad_width=(1, 1), mode="edge")

            # 3. Create the new padded meshgrid
            WN_pad, TG_pad = np.meshgrid(wn_pad, t_pad)

            # Use the padded arrays for plotting
            surf = ax.plot_surface(
                WN_pad,
                TG_pad,
                Z_pad,
                cmap=cmap,
                linewidth=0,
                antialiased=False,
                alpha=1.0,
                rcount=len(t_pad),  # Update to padded length
                ccount=len(wn_pad),  # Update to padded length
                shade=False,
                rasterized=True,  # Keep this so Inkscape stays fast!
            )
            # ──────────────────────────────

            if show_colorbar:
                cb = fig.colorbar(
                    surf,
                    ax=ax,
                    shrink=0.45,
                    pad=0.12,  # Pushes the colorbar away from the 3D plot
                    aspect=15,
                    format="%.1f",
                    ticks=plt.MaxNLocator(4),
                )
                # FIX: Force the label to rotate and sit outside the ticks
                cb.set_label(
                    "Intensity", rotation=270, labelpad=15, fontsize=label_fontsize
                )
                cb.ax.tick_params(labelsize=label_fontsize - 1)

            ax.view_init(elev=elev, azim=azim)
            _style_3d_ax(ax, title, label_fontsize, title_fontsize)

        for col, (title, Z, cmap) in enumerate(main_panels, start=1):
            _add(1, col, title, Z, cmap)

        for col, (title, Z, cmap) in enumerate(fluoro_panels, start=1):
            _add(2, col, title, Z, cmap)

        # Increase structural padding to prevent panels from overlapping each other
        plt.subplots_adjust(wspace=0.15, hspace=0.2)
        return fig


def _plot_components_3d_plotly(
    wn_sub: np.ndarray,
    t_sub: np.ndarray,
    main_panels: list,
    fluoro_panels: list,
) -> "go.Figure":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_main = len(main_panels)
    n_fluoro = len(fluoro_panels)
    n_rows = 1 + (1 if n_fluoro > 0 else 0)
    n_cols = max(n_main, n_fluoro) if n_fluoro > 0 else n_main

    specs = [[{"type": "scene"}] * n_cols for _ in range(n_rows)]
    subplot_titles = (
        [p[0] for p in main_panels]
        + [""] * (n_cols - n_main)  # pad empty slots in row 1
        + [p[0] for p in fluoro_panels]
    )

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    # X = Wavenumber (left→right), Y = Time (front→back, t=0 at front).
    # Z [T, W] maps directly to go.Surface(x=wn, y=t, z=Z).
    # Flat shading + ambient-only light = clean cartoon appearance.
    _lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)

    def _add_trace(row: int, col: int, title: str, Z: np.ndarray, cmap: str) -> None:
        fig.add_trace(
            go.Surface(
                x=wn_sub,
                y=t_sub,
                z=Z,
                colorscale=_mpl_to_plotly_cmap(cmap),
                showscale=False,
                name=title,
                opacity=1.0,
                flatshading=True,
                lighting=_lighting,
            ),
            row=row,
            col=col,
        )
        scene_idx = (row - 1) * n_cols + col
        key = "scene" if scene_idx == 1 else f"scene{scene_idx}"
        fig.update_layout(**{key: _plotly_scene()})

    for col, (title, Z, cmap) in enumerate(main_panels, start=1):
        _add_trace(1, col, title, Z, cmap)

    for col, (title, Z, cmap) in enumerate(fluoro_panels, start=1):
        _add_trace(2, col, title, Z, cmap)

    fig.update_layout(
        width=440 * n_cols,
        height=480 * n_rows,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=11, color="#222222"),
        title_font=dict(size=13, color="#111111"),
    )
    # Bold subplot annotation titles
    for ann in fig.layout.annotations:
        ann.font.update(size=12, color="#111111", family="Arial, Helvetica, sans-serif")
        ann.font["weight"] = "bold"  # type: ignore[index]
    return fig


def plot_data_3d(
    data: Union[np.ndarray, "SpectralData", "xr.Dataset"],
    sample_idx: int = 0,
    use_clean: bool = False,
    time_range: Optional[Tuple[float, float]] = None,
    subsample_wn: int = 2,
    subsample_time: int = 1,
    backend: str = "matplotlib",
    elev: float = 22.0,
    azim: float = -50.0,
    figsize: Tuple[float, float] = (5.5, 4.5),
    dpi: int = 150,
    cmap: str = "lumos_observed",
    title: str = "Observed time series",
    label_fontsize: int = 9,
    title_fontsize: int = 11,
    show_colorbar: bool = False,
) -> "Figure":
    """
    NeurIPS-styled 3D surface plot of a raw observed time series — no model or
    decomposition required.
    """
    # Unpack input
    if hasattr(data, "coords"):
        sd = data_from_dataset(data, sample_idx=sample_idx, use_clean=use_clean)
    elif hasattr(data, "intensities"):
        sd = data
    else:
        arr = np.asarray(data)
        n_t, n_wn = arr.shape
        sd = SpectralData(
            arr, np.arange(n_wn, dtype=float), time_values=np.arange(n_t, dtype=float)
        )

    Y = sd.intensities
    time_values = sd.time_values
    wavenumbers = sd.wavenumbers
    n_t, n_wn = Y.shape

    if time_range is not None:
        t_lo, t_hi = time_range
        mask = (time_values >= t_lo) & (time_values <= t_hi)
        time_values = time_values[mask]
        Y = Y[mask]
        n_t = len(time_values)

    wn_idx = np.arange(0, n_wn, subsample_wn)
    t_idx = np.arange(0, n_t, subsample_time)
    wn_sub = wavenumbers[wn_idx]
    t_sub = time_values[t_idx]
    Z = Y[np.ix_(t_idx, wn_idx)]

    if backend == "plotly":
        import plotly.graph_objects as go

        _lighting = dict(
            ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
        )
        fig = go.Figure(
            go.Surface(
                x=wn_sub,
                y=t_sub,
                z=Z,
                colorscale=_mpl_to_plotly_cmap(cmap),
                opacity=1.0,
                flatshading=True,
                lighting=_lighting,
                showscale=show_colorbar,
                colorbar=(
                    dict(
                        title=dict(text="Intensity", font=dict(size=13)),
                        tickfont=dict(size=11),
                        thickness=14,
                        len=0.65,
                    )
                    if show_colorbar
                    else None
                ),
            )
        )
        fig.update_layout(
            scene=_plotly_scene(),
            title=dict(
                text=f"<b>{title}</b>", font=dict(size=15), x=0.5, xanchor="center"
            ),
            width=750,
            height=600,
            margin=dict(l=10, r=10, t=55, b=10),
            font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#222222"),
            paper_bgcolor="white",
        )
        return fig

    # matplotlib — X = Wavenumber (left→right), Y = Time (front→back)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ── THE "SOLID WALLS" TRICK ──
    z_floor = 0.0

    # 1. Pad the intensity array with zeros around all 4 edges
    Z_pad = np.pad(
        Z, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=z_floor
    )

    # 2. Pad the coordinate axes by duplicating the first and last values
    wn_pad = np.pad(wn_sub, pad_width=(1, 1), mode="edge")
    t_pad = np.pad(t_sub, pad_width=(1, 1), mode="edge")

    # 3. Create the padded meshgrid
    WN_pad, TG_pad = np.meshgrid(wn_pad, t_pad)

    rcount = len(t_pad)
    ccount = len(wn_pad)
    # ──────────────────────────────

    with plt.rc_context(_NEURIPS_RC):
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        surf = ax.plot_surface(
            WN_pad,
            TG_pad,
            Z_pad,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            alpha=1.0,
            rcount=rcount,
            ccount=ccount,
            shade=False,
            rasterized=True,  # Keeps the SVG fast in Inkscape
        )

        if show_colorbar:
            cb = fig.colorbar(
                surf,
                ax=ax,
                shrink=0.38,
                pad=0.08,
                aspect=18,
                format="%.1f",
                ticks=plt.MaxNLocator(4),
            )
            cb.set_label(
                "Intensity", rotation=270, labelpad=15, fontsize=label_fontsize
            )
            cb.ax.tick_params(labelsize=label_fontsize - 1)

        ax.view_init(elev=elev, azim=azim)
        _style_3d_ax(ax, title, label_fontsize, title_fontsize)
        plt.tight_layout(pad=1.2)

    return fig


def plot_components_3d(
    data: Union[np.ndarray, "SpectralData"],
    decomposition: "DecompositionResult",
    time_range: Optional[Tuple[float, float]] = None,
    subsample_wn: int = 2,
    subsample_time: int = 1,
    show_noise: bool = True,
    show_individual_fluorophores: bool = True,
    show_main_panels: bool = True,
    backend: str = "matplotlib",
    elev: float = 22.0,
    azim: float = -50.0,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.6),
    dpi: int = 150,
    cmap_observed: str = "lumos_observed",
    cmap_raman: str = "lumos_raman",
    cmap_fluorescence: str = "lumos_fluor",
    cmap_noise: str = "lumos_noise",
    cmap_fluorophores: Optional[list] = None,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    show_colorbar: bool = True,
) -> "Figure":
    """
    Publication-quality 3D surface panels showing observed data and its
    decomposition into Raman, fluorescence, noise, and (optionally) individual
    fluorophore decay surfaces.

    Suitable for NeurIPS headline figures.  Each panel is a 3D surface over
    (Wavenumber × Time × Intensity).

    Parameters
    ----------
    data : np.ndarray [T, W] or SpectralData
        Observed time series.
    decomposition : DecompositionResult
        Result containing raman, fluorophore_spectra, rates, abundances,
        physics_model and frame_duration.
    time_range : (t_start, t_end) in seconds, optional
        Restrict the time axis.  Shows the full series when *None*.
    subsample_wn : int
        Wavenumber downsampling factor (performance vs. resolution).
    subsample_time : int
        Time downsampling factor.
    show_noise : bool
        Include a noise / residual panel.
    show_individual_fluorophores : bool
        Add a second row of panels for each fluorophore component.
    backend : {"matplotlib", "plotly"}
        ``"matplotlib"`` → static / vector-graphics output.
        ``"plotly"``     → interactive HTML.
    elev, azim : float
        Viewing elevation and azimuth in degrees (matplotlib only).
    figsize_per_panel : (w, h) in inches
        Size of each 3D panel (matplotlib only).
    dpi : int
        Figure resolution (matplotlib only).
    cmap_observed, cmap_raman, cmap_fluorescence, cmap_noise : str
        Matplotlib / Plotly colormap names for each component.
    cmap_fluorophores : list of str, optional
        Per-fluorophore colormaps.  Defaults to cycling ``_FLUORO_CMAPS``.
    label_fontsize, title_fontsize : int
        Font sizes for axis labels and panel titles (matplotlib only).
    show_colorbar : bool
        Attach a colorbar to every panel (matplotlib only).

    Returns
    -------
    fig : matplotlib.figure.Figure  or  plotly.graph_objects.Figure
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    if hasattr(data, "intensities"):
        Y = data.intensities  # [T, W]
        time_values = data.time_values
        wavenumbers = data.wavenumbers
    else:
        Y = np.asarray(data)
        time_values = None
        wavenumbers = None

    raman = decomposition.raman.intensities  # [W]
    bases = decomposition.fluorophore_spectra.intensities  # [F, W]
    rates = decomposition.rates  # [F]
    abundances = (
        decomposition.abundances
        if decomposition.abundances is not None
        else np.ones(len(rates))
    )
    n_t, n_wn = Y.shape
    n_fluoro = len(rates)

    n_wn_recon = raman.shape[0]
    if n_wn != n_wn_recon:
        w_crop = (n_wn - n_wn_recon) // 2
        Y = Y[:, w_crop : w_crop + n_wn_recon]
        if wavenumbers is not None and len(wavenumbers) == n_wn:
            wavenumbers = wavenumbers[w_crop : w_crop + n_wn_recon]
        n_wn = n_wn_recon
    # ─────────────────────────────────────────────────────────────────────

    if time_values is None:
        time_values = np.arange(n_t, dtype=float)
    if wavenumbers is None:
        wavenumbers = np.arange(n_wn, dtype=float)

    # ------------------------------------------------------------------
    # Frame duration
    # ------------------------------------------------------------------
    frame_dur = decomposition.frame_duration
    if frame_dur is None:
        frame_dur = float(time_values[1] - time_values[0]) if n_t > 1 else 1.0

    # ------------------------------------------------------------------
    # Time-range slice
    # ------------------------------------------------------------------
    if time_range is not None:
        t_lo, t_hi = time_range
        mask = (time_values >= t_lo) & (time_values <= t_hi)
        time_values = time_values[mask]
        Y = Y[mask]
    n_t = len(time_values)

    # ------------------------------------------------------------------
    # Compute component surfaces  [T, W]
    # ------------------------------------------------------------------
    reconstruction = decomposition.reconstruction(time_values)
    raman_surface = np.tile(raman * frame_dur, (n_t, 1))
    total_fluor = reconstruction - raman_surface
    noise = Y - reconstruction

    fluoro_surfaces = [
        _compute_fluorophore_surface(
            bases, abundances, rates, time_values, frame_dur, i
        )
        for i in range(n_fluoro)
    ]

    # ------------------------------------------------------------------
    # Subsample
    # ------------------------------------------------------------------
    wn_idx = np.arange(0, n_wn, subsample_wn)
    t_idx = np.arange(0, n_t, subsample_time)
    wn_sub = wavenumbers[wn_idx]
    t_sub = time_values[t_idx]

    def _sub(Z: np.ndarray) -> np.ndarray:
        return Z[np.ix_(t_idx, wn_idx)]

    # ------------------------------------------------------------------
    # Panel definitions
    # ------------------------------------------------------------------
    main_panels = []
    if show_main_panels:
        main_panels = [
            ("Observed Input (Raw Data)", _sub(Y), cmap_observed),
            ("Predicted Raman", _sub(raman_surface), cmap_raman),
            ("Predicted Total Fluorescence", _sub(total_fluor), cmap_fluorescence),
        ]
        if show_noise:
            main_panels.append(("Predicted Noise (Residual)", _sub(noise), cmap_noise))

    if cmap_fluorophores is None:
        cmap_fluorophores = [
            _FLUORO_CMAPS[i % len(_FLUORO_CMAPS)] for i in range(n_fluoro)
        ]

    fluoro_panels: list = []
    if show_individual_fluorophores:
        taus = 1.0 / rates
        for i, (Z, cmap_f) in enumerate(zip(fluoro_surfaces, cmap_fluorophores)):
            fluoro_panels.append(
                (
                    f"Predicted Fluorophore {i + 1}  (τ = {taus[i]:.2f} s)",
                    _sub(Z),
                    cmap_f,
                )
            )

    # ------------------------------------------------------------------
    # Dispatch to backend
    # ------------------------------------------------------------------
    if backend == "plotly":
        return _plot_components_3d_plotly(wn_sub, t_sub, main_panels, fluoro_panels)

    return _plot_components_3d_mpl(
        wn_sub,
        t_sub,
        main_panels,
        fluoro_panels,
        elev,
        azim,
        figsize_per_panel,
        dpi,
        label_fontsize,
        title_fontsize,
        show_colorbar,
    )
