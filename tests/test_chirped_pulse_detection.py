import numpy as np
from sigint_examples.simulation import generate_chirped_train
from sigint_examples.processing import matched_filter_chirp, detect_peaks, estimate_pulse_widths, add_awgn

def test_chirped_pulse_detection_workflow():
    # -----------------------------
    # Parameters
    # -----------------------------
    fs = 1_000_000
    duration = 0.002
    PRI = 100e-6
    pulse_width = 5e-6
    fc = 100_000
    bandwidth = 50_000
    SNR_dB = 0

    # -----------------------------
    # Generate signal
    # -----------------------------
    t, rx_signal, pulse_times = generate_chirped_train(fs, duration, PRI, pulse_width, fc, bandwidth)
    rx_signal_noisy = add_awgn(rx_signal, SNR_dB)

    # Basic assertions
    assert len(t) == int(duration * fs) + 1
    assert len(rx_signal) == len(t)

    # -----------------------------
    # Matched filter
    # -----------------------------
    mf_output = matched_filter_chirp(rx_signal_noisy, pulse_width, fs, fc, bandwidth)
    peak_indices, peak_times, estimated_PRIs, PRI_estimate = detect_peaks(mf_output, t)

    # Ensure we detected at least one peak
    assert len(peak_indices) > 0
    assert np.all(peak_times > 0)

    # PRI estimates should be close to true value
    if PRI_estimate is not None:
        assert abs(PRI_estimate - PRI) < 20e-6  # within 20 µs

    # -----------------------------
    # Pulse width estimation
    # -----------------------------
    pulse_width_estimate = estimate_pulse_widths(mf_output, t, peak_indices)
    assert pulse_width_estimate is not None
    assert abs(pulse_width_estimate - pulse_width) < 5e-6  # within 5 µs

    # Optional: check that MF output is roughly same length as input
    assert len(mf_output) == len(rx_signal_noisy)
