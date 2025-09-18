import numpy as np
from sigint_examples.simulation import generate_gaussian_jittered_pulses
from sigint_examples.processing import add_awgn, matched_filter_detect, estimate_PRI_statistics
from sigint_examples.plotting import plot_time_domain, plot_matched_filter, plot_PRI_histogram, plot_PRI_vs_time, SHOW_PLOTS


def test_gaussian_jittered_workflow():
    fs = 1_000_000
    duration = 0.01
    base_PRI = 100e-6
    pulse_width = 5e-6
    jitter_std = 5e-6
    SNR_dB = 0

    # Generate signal
    t, rx, pulse_times = generate_gaussian_jittered_pulses(
        fs, duration, base_PRI, pulse_width, jitter_std, seed=42
    )
    rx_noisy = add_awgn(rx, SNR_dB, seed=42)

    # Detect pulses
    mf_output, t_valid, peaks, pulse_times_detected, threshold = matched_filter_detect(
        rx_noisy, pulse_width, fs, base_PRI, jitter_std
    )

    assert len(pulse_times_detected) > 5
    assert np.all(pulse_times_detected > 0)

    # PRI statistics
    estimated_PRIs_clean, mean_PRI_clean, std_PRI_clean = estimate_PRI_statistics(pulse_times_detected)

    assert estimated_PRIs_clean is not None
    assert abs(mean_PRI_clean - base_PRI) < 20e-6  # within 20 µs
    assert std_PRI_clean > 0

    if SHOW_PLOTS:
        plot_time_domain(t, rx, rx_noisy, detected_pulses=pulse_times_detected, zoom=(0, 0.005))
        plot_matched_filter(t_valid, mf_output, peaks, threshold, zoom=(0, 0.005))
        plot_PRI_histogram(estimated_PRIs_clean, base_PRI, mean_PRI_clean)
        plot_PRI_vs_time(pulse_times_detected, estimated_PRIs_clean, base_PRI, mean_PRI_clean, jitter_std, duration)
