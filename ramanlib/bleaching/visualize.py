"""Visualization utilities for bleaching decomposition."""

from typing import Optional, Union

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
        decomposition: DecompositionResult from predict_from_early.
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
    # decomposition.raman.intensities is in counts/sec (rate), matching the units
    # of data.intensities (Y) and the reference_raman (last-frames avg).
    raman_per_frame = raman

    reconstruction = decomposition.reconstruction(time_values)

    # Training cutoff time for vertical line — placed at the last *used* frame.
    if n_train is not None and time_values is not None and len(time_values) > 0:
        t_train_cutoff = float(time_values[min(n_train - 1, len(time_values) - 1)])
    else:
        t_train_cutoff = None

    # Reference Raman (expected in same units as Y, i.e. counts/frame)
    if reference_raman is None:
        reference_raman = Y[-20:].mean(axis=0)
        ref_raman_label = "Reference (last 20 frames avg)"
        print("Using last 20 frames average as reference Raman.")
    else:
        ref_raman_label = "Ground Truth Raman"

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
            available = [(j, corr_matrix[p, j]) for j in range(n_ref) if j not in used_refs]
            if available:
                best_ref, best_corr = max(available, key=lambda x: x[1])
                pred_to_ref[p] = (best_ref, float(best_corr))
                used_refs.add(best_ref)
            else:
                pred_to_ref[p] = (None, 0.0)
        pred_colors = [
            _COLORS[pred_to_ref[i][0] % len(_COLORS)] if pred_to_ref[i][0] is not None else "#888888"
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
            ax.axvline(t_train_cutoff, color="#444444", linestyle="--", linewidth=1.8,
                       label=f"Train cutoff (n={n_train})", zorder=5)
        else:
            ax.axhline(t_train_cutoff, color="#444444", linestyle="--", linewidth=1.8, zorder=5)

    # ── Plot 1: Original Time Series + Reconstruction overlay ─────────────────
    ax = axes[0, 0]
    n_show = min(6, n_t)
    cmap_ts = plt.cm.viridis
    show_indices = np.linspace(0, n_t - 1, n_show, dtype=int)
    for i, idx in enumerate(show_indices):
        c = cmap_ts(i / max(n_show - 1, 1))
        t_label = f"t={time_values[idx]:.2f}s" if time_values is not None else f"frame {idx}"
        ax.plot(wavenumbers, Y[idx], color=c, alpha=0.8, linewidth=1.2, label=t_label)
        ax.plot(wavenumbers, reconstruction[idx], color=c, alpha=0.5,
                linewidth=1.0, linestyle="--")
    # Dummy lines for legend
    ax.plot([], [], "k-",  linewidth=1.5, label="Observed")
    ax.plot([], [], "k--", linewidth=1.0, alpha=0.6, label="Reconstructed")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (counts/frame)")
    ax.set_title(f"Time Series Summary")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Extracted Raman vs reference ─────────────────────────────────
    ax = axes[0, 1]
    ax.plot(wavenumbers, raman_per_frame, color=_COLORS[0], linewidth=2,
            label="Predicted Raman")
    ax.plot(wavenumbers, reference_raman, "r--", linewidth=1.5, alpha=0.8,
            label=ref_raman_label)
    raman_corr = np.corrcoef(raman_per_frame, reference_raman)[0, 1]
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (counts/frame)")
    ax.set_title(f"Extracted Raman Spectrum  (r = {raman_corr:.4f})")
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
                wavenumbers, b_plot,
                color=pred_colors[i], linewidth=2,
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
                wavenumbers, b_ref,
                color=ref_colors[j], linewidth=1.5, linestyle="--", alpha=0.8,
                label=f"GT B{j+1} (τ={tau_gt:.3f}s, w={reference_abundances[j]:.1f})",
            )
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Normalised intensity" if normalise else "Intensity")
    ax.set_title("Fluorophore Basis Spectra\n(solid=pred, dashed=GT, colour=matched pair)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Decay Components ──────────────────────────────────────────────
    ax = axes[1, 0]
    total_fluor = np.zeros(n_t)
    if bases is not None:
        for i in range(n_fluorophores):
            fluor_series = reconstruct_time_series_numpy(
                raman=np.zeros(bases.shape[1]),
                bases=bases[i:i + 1, :],
                abundances=np.array([abundances[i]]),
                decay_rates=np.array([rates[i]]),
                time_values=time_values,
                frame_duration=frame_dur,
                physics_model=physics_model,
            )
            amplitude = fluor_series.mean(axis=1)
            total_fluor += amplitude
            tau = time_constants[i]
            ax.plot(time_values, amplitude, color=pred_colors[i], linewidth=1.5,
                    label=f"τ={tau:.3f}s, w={abundances[i]:.1f}")
    if has_reference:
        total_gt_fluor = np.zeros(n_t)
        for j in range(n_ref):
            fluor_series = reconstruct_time_series_numpy(
                raman=np.zeros(reference_bases.shape[1]),
                bases=reference_bases[j:j + 1, :],
                abundances=np.array([reference_abundances[j]]),
                decay_rates=np.array([reference_rates[j]]),
                time_values=time_values,
                frame_duration=frame_dur,
                physics_model=physics_model,
            )
            amplitude = fluor_series.mean(axis=1)
            total_gt_fluor += amplitude
            ax.plot(time_values, amplitude,
                    color=ref_colors[j], linestyle="--", linewidth=1.5, alpha=0.8,
                    label=f"GT τ={1.0 / reference_rates[j]:.3f}s, w={reference_abundances[j]:.1f}")
        ax.plot(time_values, total_gt_fluor, "r--", linewidth=2, label="Total GT fluor.")
    ax.plot(time_values, total_fluor, "k-", linewidth=2, label="Total Predicted")
    _add_train_cutoff(ax)
    if t_train_cutoff is not None:
        ax.axvspan(time_values[0], t_train_cutoff, alpha=0.06, color="steelblue")
        ax.axvspan(t_train_cutoff, time_values[-1], alpha=0.06, color="darkorange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean fluorescence (counts/frame)")
    ax.set_title("Decay Components")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 5: First Frame Reconstruction ───────────────────────────────────
    # ── Plot 5: First Frame Reconstruction ───────────────────────────────────
    ax = axes[1, 1]
    ax.plot(wavenumbers, Y[0], "r--", linewidth=1.5, alpha=0.8, label="Original (t=0)")
    ax.plot(wavenumbers, reconstruction[0], color=_COLORS[0], linewidth=1.5, alpha=0.9,
            label="Reconstructed (t=0)")
    t0_mse = float(np.mean((Y[0] - reconstruction[0]) ** 2))
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (counts/frame)")
    ax.set_title(f"Reconstruction (t=0)  MSE={t0_mse:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 6: MSE over time (log scale) ────────────────────────────────────
    ax = axes[1, 2]
    residuals = Y - reconstruction
    mse_over_time = np.mean(residuals ** 2, axis=1)
    mse_all = float(mse_over_time.mean())
    n_first = n_train if n_train else 20
    mse_first = float(mse_over_time[:n_first].mean())
    ax.semilogy(time_values, mse_over_time, "k-", linewidth=1.5, label="MSE per frame")
    _add_train_cutoff(ax)
    if t_train_cutoff is not None:
        ax.axvspan(time_values[0], t_train_cutoff, alpha=0.06, color="steelblue",
                   label="Training region")
        ax.axvspan(t_train_cutoff, time_values[-1], alpha=0.06, color="darkorange",
                   label="Extrapolation region")
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
    print(f"Raman correlation with reference:       {raman_corr:.4f}")
    print(f"Time constants τ (s):  {time_constants}")
    print(f"Abundances w:          {abundances}")

    if has_reference:
        print("\nMatched basis pairs (pred → GT, by correlation):")
        for p in range(n_fluorophores):
            ref_j, corr_val = pred_to_ref[p]
            if ref_j is not None:
                tau_pred = time_constants[p]
                tau_gt = 1.0 / reference_rates[ref_j]
                rate_err_pct = 100.0 * abs(rates[p] - reference_rates[ref_j]) / reference_rates[ref_j]
                print(
                    f"  Pred B{p+1} (τ={tau_pred:.3f}s, w={abundances[p]:.1f})  →  "
                    f"GT B{ref_j+1} (τ={tau_gt:.3f}s, w={reference_abundances[ref_j]:.1f}):  "
                    f"r={corr_val:.4f},  λ err={rate_err_pct:.1f}%"
                )

        # Sorted rate errors (naive, for completeness)
        n_gt = len(reference_rates)
        n_pred = len(decomposition.rates)
        if n_pred != n_gt:
            top_idx = np.argsort(decomposition.abundances)[-n_gt:]
            rates_est_sorted = np.sort(decomposition.rates[top_idx])
        else:
            rates_est_sorted = np.sort(decomposition.rates)
        rates_gt_sorted = np.sort(reference_rates)
        rate_errors_pct = 100.0 * np.abs(rates_est_sorted - rates_gt_sorted) / rates_gt_sorted
        print(f"\nRate errors % (sorted, naive 1-to-1): {np.round(rate_errors_pct, 1)}")

    return fig, axes


def plot_parameter_detail(
        decomposition: DecompositionResult,
        reference_bases: np.ndarray,
        reference_rates: np.ndarray,
        reference_abundances: np.ndarray,
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

    bases      = decomposition.fluorophore_spectra.intensities
    abundances = decomposition.abundances
    rates      = decomposition.rates
    n_fluorophores = len(rates)
    n_ref          = len(reference_rates)

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
    parts.append(f"n_gt={n_ref}")
    parts.append(f"physics={physics_model}")
    if n_train is not None:
        parts.append(f"n_train={n_train}")
    info_str = "   |   ".join(parts)

    # ── Greedy matching by basis correlation ─────────────────────────────────
    corr_matrix = np.zeros((n_fluorophores, n_ref))
    for i in range(n_fluorophores):
        for j in range(n_ref):
            c = np.corrcoef(bases[i], reference_bases[j])[0, 1]
            corr_matrix[i, j] = c if np.isfinite(c) else 0.0
    pred_to_ref: Dict[int, Tuple] = {}
    used_refs: set = set()
    for p in np.argsort(-corr_matrix.max(axis=1)):
        available = [(j, corr_matrix[p, j]) for j in range(n_ref) if j not in used_refs]
        if available:
            best_j, best_c = max(available, key=lambda x: x[1])
            pred_to_ref[p] = (best_j, float(best_c))
            used_refs.add(best_j)
        else:
            pred_to_ref[p] = (None, 0.0)

    pred_colors = [
        _COLORS[pred_to_ref[i][0] % len(_COLORS)] if pred_to_ref[i][0] is not None else "#888888"
        for i in range(n_fluorophores)
    ]
    ref_colors = [_COLORS[j % len(_COLORS)] for j in range(n_ref)]

    # ── Figure 1: Correlation heatmap + Rate scatter + Abundance scatter ─────
    fig_scatter, (ax_corr, ax_r, ax_a) = plt.subplots(1, 3, figsize=(15, 5))

    # Correlation heatmap
    im = ax_corr.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
    ax_corr.set_xticks(range(n_ref))
    ax_corr.set_yticks(range(n_fluorophores))
    ax_corr.set_xticklabels([f"GT B{j+1}" for j in range(n_ref)], rotation=45, ha="right", fontsize=9)
    ax_corr.set_yticklabels([f"Pred B{i+1}" for i in range(n_fluorophores)], fontsize=9)
    for i in range(n_fluorophores):
        for j in range(n_ref):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.6 else "black"
            ax_corr.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                         fontsize=9, color=text_color)
    plt.colorbar(im, ax=ax_corr, shrink=0.8, label="Pearson r")
    ax_corr.set_title("Basis Correlation")
    ax_corr.set_xlabel("GT Bases")
    ax_corr.set_ylabel("Predicted Bases")

    # Matched pairs for scatter plots
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
        (ax_a, gt_a, pred_a, "GT abundance w", "Predicted abundance w", "Abundance"),
    ]:
        if xs:
            for x, y, c, lbl in zip(xs, ys, pair_cols, pair_labels):
                ax.scatter(x, y, color=c, s=110, zorder=5, edgecolors="k", linewidths=0.5)
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
            lo = 0.0
            hi = float(max(max(xs), max(ys))) * 1.2
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="1:1")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} Comparison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    if info_str:
        fig_scatter.suptitle(info_str, fontsize=9, y=1.02)
    fig_scatter.tight_layout()

    # ── Figure 2: Per-component spectra (abundance × basis at t=0) ───────────
    n_cols = min(n_fluorophores, 4)
    n_rows = int(np.ceil(n_fluorophores / n_cols))
    fig_comps, axes_c = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for i in range(n_fluorophores):
        row, col = divmod(i, n_cols)
        ax = axes_c[row][col]
        tau_pred = 1.0 / rates[i]
        pred_comp = float(abundances[i]) * bases[i]
        ax.plot(wavenumbers, pred_comp, color=pred_colors[i], linewidth=2,
                label=f"Pred  τ={tau_pred:.3f}s  w={abundances[i]:.2f}")

        ref_j, corr_val = pred_to_ref[i]
        if ref_j is not None:
            tau_gt  = 1.0 / reference_rates[ref_j]
            gt_comp = float(reference_abundances[ref_j]) * reference_bases[ref_j]
            ax.plot(wavenumbers, gt_comp, color=ref_colors[ref_j],
                    linestyle="--", linewidth=1.5, alpha=0.85,
                    label=f"GT    τ={tau_gt:.3f}s  w={reference_abundances[ref_j]:.2f}")
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

    title_comps = "Individual Components: abundance × basis at t=0  (solid=pred, dashed=GT)"
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
            ds, sample_idx, i, time_seconds, frame_duration=frame_duration, physics_model=physics_model
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
        raman, bases, abundances, decay_rates, t_single,
        frame_duration=frame_duration, physics_model=physics_model
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
            bases=bases[i:i + 1, :],
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

    rmse = np.sqrt(np.mean(residual ** 2))
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
            wn, B_i, color=fluor_colors[i], linewidth=1.5, label=f"B{i + 1} (τ={τ:.3f}s)"
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

    # Total fluorescence = reconstruction - raman contribution
    raman_per_frame = raman
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
