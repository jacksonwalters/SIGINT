import numpy as np
from sigint_examples.simulation import generate_irregular_pulses
from sigint_examples.processing import matched_filter, autocorrelation, estimate_PRI_from_mf
from sigint_examples.plotting import plot_time_domain, plot_autocorr, plot_histogram, SHOW_PLOTS

def test_irregular_pulse_workflow():
    # Generate signal
    t, signal, pulse_times = generate_irregular_pulses()
    assert signal.shape == t.shape
    assert len(pulse_times) > 0

    if SHOW_PLOTS:
        plot_time_domain(t, signal, "Irregular Pulse Train")

    # Matched filter
    mf_output = matched_filter(signal, pulse_width=5e-6, fs=1e6)
    assert mf_output.shape == signal.shape
    if SHOW_PLOTS:
        plot_time_domain(t, mf_output, "Matched Filter Output")

    # Autocorrelation
    auto, lags = autocorrelation(signal)
    assert auto.shape[0] == 2*signal.shape[0]-1
    if SHOW_PLOTS:
        plot_autocorr(lags, auto, "Autocorrelation")

    # PRI estimation
    mean_PRI, std_PRI, peak_times, estimated_PRIs = estimate_PRI_from_mf(mf_output, t)
    assert mean_PRI > 0
    assert std_PRI >= 0
    assert np.all(np.isfinite(peak_times))
    if SHOW_PLOTS:
        plot_histogram(estimated_PRIs, title="Histogram of Estimated PRIs")
