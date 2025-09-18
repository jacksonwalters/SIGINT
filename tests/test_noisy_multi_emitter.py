import numpy as np
from sigint_examples.simulation import generate_pulse_train
from sigint_examples.processing import autocorrelation, detect_fundamental_PRIs, add_awgn
from sigint_examples.plotting import plot_time_domain, plot_autocorr, SHOW_PLOTS

def test_noisy_multi_emitter_workflow(show_plots):
    fs = 1_000_000
    duration = 0.01
    emitters = [
        {"PRI": 100e-6, "pulse_width": 5e-6, "amp": 1.0},
        {"PRI": 150e-6, "pulse_width": 8e-6, "amp": 0.7}
    ]

    t = np.arange(0, duration, 1/fs)
    rx_clean = np.zeros_like(t)

    for e in emitters:
        rx, _ = generate_pulse_train(e["PRI"], e["pulse_width"], duration, fs, amplitude=e["amp"])
        rx_clean += rx

    rx = add_awgn(rx_clean, SNR_dB=-3)

    if show_plots:
        plot_time_domain(t, rx, title="Noisy Multi-Emitter Pulses")

    auto, lags = autocorrelation(rx)
    mid = len(auto)//2
    auto_pos = auto[mid+1:]
    lags_pos = lags[mid+1:]

    fundamentals, peak_lags_us, peaks = detect_fundamental_PRIs(
        auto_pos, lags_pos, N_emitters=2, min_PRI=min([e["PRI"] for e in emitters])
    )

    if show_plots:
        plot_autocorr(lags_pos, auto_pos, peaks=peaks, fundamentals=fundamentals)

    assert len(fundamentals) <= 2
    assert all(f > 0 for f in fundamentals)
