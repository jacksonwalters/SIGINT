"""
basic_workflow_modular.py

Demonstrates a simple pulse train workflow using the modular SIGINT library:
- Simulation: generate rectangular pulse train modulated with a carrier and optional noise
- Processing: matched filter, autocorrelation, PRI estimation
- Plotting: visualize signals, matched filter output, and autocorrelation
"""

import numpy as np
from sigint_examples.simulation import generate_basic_pulse_train
from sigint_examples.processing import matched_filter, autocorrelation, estimate_PRI_from_autocorr
from sigint_examples.plotting import plot_basic_workflow
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run basic SIGINT workflow")
parser.add_argument(
    "--show-plots",
    action="store_true",
    help="Display plots during execution"
)
args = parser.parse_args()
SHOW_PLOTS = args.show_plots

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000       # 1 MHz sampling rate
duration = 0.002     # 2 ms total time
fc = 100_000         # 100 kHz carrier frequency
PRI = 100e-6         # 100 µs pulse repetition interval
pulse_width = 5e-6   # 5 µs pulse width
SNR_dB = 0           # Noise level in dB
seed = 42            # For reproducibility

# -----------------------------
# Generate pulse train
# -----------------------------
t, clean_signal, rx_signal = generate_basic_pulse_train(
    fs, duration, PRI, pulse_width, fc, SNR_dB, seed=seed
)

# -----------------------------
# Processing
# -----------------------------
mf_output = matched_filter(rx_signal, pulse_width, fs)
auto, lags = autocorrelation(rx_signal)
PRI_estimate, peak_lags, auto_pos, lags_pos = estimate_PRI_from_autocorr(auto, lags)

# Pulse width estimation using FWHM
center_idx = np.argmax(auto_pos)
half_max = auto_pos[center_idx] / 2
left_idx = np.where(auto_pos[:center_idx] <= half_max)[0]
left_idx = 0 if len(left_idx) == 0 else left_idx[-1]
right_idx = np.where(auto_pos[center_idx:] <= half_max)[0]
right_idx = len(auto_pos)-1 if len(right_idx) == 0 else right_idx[0]+center_idx
pulse_width_estimate = lags_pos[right_idx] - lags_pos[left_idx]

# -----------------------------
# Display results
# -----------------------------
print(f"True PRI: {PRI*1e6:.2f} µs")
print(f"Estimated PRI from autocorrelation: {PRI_estimate*1e6:.2f} µs")
print(f"True pulse width: {pulse_width*1e6:.2f} µs")
print(f"Estimated pulse width (FWHM): {pulse_width_estimate*1e6:.2f} µs")

# -----------------------------
# Plotting
# -----------------------------
if SHOW_PLOTS:
    plot_basic_workflow(
        t=t,
        rx_signal=rx_signal,
        clean_signal=clean_signal,
        mf_output=mf_output,
        auto=auto,
        lags=lags,
        peak_lags=peak_lags,
        pulse_width_estimates=[pulse_width_estimate]
    )
