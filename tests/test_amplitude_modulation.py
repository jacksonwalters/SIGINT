import numpy as np
import pytest

from sigint_examples.simulation import generate_amplitude_modulated_pulses
from sigint_examples.processing import (
    matched_filter,
    autocorrelation,
    estimate_PRI_from_autocorr,
    estimate_pulse_width_from_autocorr,
)
from sigint_examples.plotting import plot_time_domain, plot_autocorr, plot_histogram, SHOW_PLOTS

def test_amplitude_modulation_workflow():
    # -----------------------------
    # Generate signal
    # -----------------------------
    t, signal, pulse_times = generate_amplitude_modulated_pulses()
    assert signal.shape == t.shape
    assert len(pulse_times) > 0

    if SHOW_PLOTS:
        plot_time_domain(t, signal, title="Amplitude-Modulated Pulse Train")

    # -----------------------------
    # Matched filter
    # -----------------------------
    mf_output = matched_filter(signal, pulse_width=5e-6, fs=1e6)
    assert mf_output.shape == signal.shape
    if SHOW_PLOTS:
        plot_time_domain(t, mf_output, title="Matched Filter Output")

    # -----------------------------
    # Autocorrelation
    # -----------------------------
    auto, lags = autocorrelation(signal)
    assert auto.shape[0] == 2*signal.shape[0]-1
    if SHOW_PLOTS:
        plot_autocorr(lags, auto, title="Autocorrelation")

    # -----------------------------
    # PRI estimation
    # -----------------------------
    PRI_estimate, peak_lags, auto_pos, lags_pos = estimate_PRI_from_autocorr(auto, lags)
    # PRI should be positive and finite
    assert PRI_estimate > 0
    assert np.all(np.isfinite(peak_lags))

    # -----------------------------
    # Pulse width estimation
    # -----------------------------
    pulse_width_estimate, zoom_lags, zoom_auto, left_idx, right_idx = estimate_pulse_width_from_autocorr(auto, lags)
    assert pulse_width_estimate > 0
    assert left_idx < right_idx < len(zoom_auto)

    if SHOW_PLOTS:
        plot_autocorr(zoom_lags, zoom_auto, title="Zoomed Autocorrelation for Pulse Width Estimation")
        plot_histogram(np.diff(peak_lags), title="Histogram of Estimated PRIs")
