import numpy as np
from sigint_examples.simulation import generate_two_pulse_train
from sigint_examples.processing import autocorr_two_pulses, estimate_fundamentals
from sigint_examples.plotting import plot_two_pulse_autocorr

def test_two_pulse_trains_basic(show_plots):
    fs = 1_000_000
    duration = 0.005
    PRI1, pulse_width1 = 100e-6, 5e-6
    PRI2, pulse_width2 = 150e-6, 8e-6

    t, rx, rx1, rx2 = generate_two_pulse_train(fs, duration, PRI1, pulse_width1, PRI2, pulse_width2)
    positive_lags, auto_positive = autocorr_two_pulses(rx, fs)
    fundamentals = estimate_fundamentals(positive_lags, auto_positive)

    # Check array lengths
    assert len(t) == int(duration * fs)
    assert len(positive_lags) == len(auto_positive)

    # Fundamental PRI estimates should include both
    PRI_est_us = fundamentals
    assert np.any(np.isclose(PRI_est_us, PRI1*1e6, atol=5.0))
    assert np.any(np.isclose(PRI_est_us, PRI2*1e6, atol=5.0))

    # Autocorrelation values should be positive
    assert np.all(auto_positive > -1e-12)  # allow tiny numerical negatives

    # -----------------------------
    # Plotting (optional, show only if debugging)
    # -----------------------------
    if show_plots:
        plot_two_pulse_autocorr(positive_lags, auto_positive, fundamentals)
