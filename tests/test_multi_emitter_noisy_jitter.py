import numpy as np
from sigint_examples.simulation import generate_multi_emitter_jittered_pulses
from sigint_examples.processing import autocorrelation, detect_fundamental_PRIs
from sigint_examples.plotting import plot_time_domain, SHOW_PLOTS

def test_multi_emitter_jitter_workflow(show_plots):
    # -----------------------------
    # Define emitters
    # -----------------------------
    emitters = [
        {"PRI": 100e-6, "pulse_width": 5e-6, "jitter": 5e-6, "amplitude": 1.0},
        {"PRI": 159e-6, "pulse_width": 8e-6, "jitter": 8e-6, "amplitude": 0.7}
    ]

    # -----------------------------
    # Generate composite jittered signal
    # -----------------------------
    t, rx, pulse_times_list = generate_multi_emitter_jittered_pulses(duration=0.01, emitters=emitters)
    assert rx.shape == t.shape
    assert len(pulse_times_list) == 2

    if show_plots:
        plot_time_domain(t, rx, title="Multi-Emitter Jittered Pulses")

    # -----------------------------
    # Compute autocorrelation
    # -----------------------------
    auto, lags = autocorrelation(rx)
    mid = len(auto)//2  # positive lags
    auto_pos = auto[mid+1:]
    lags_pos = lags[mid+1:]

    # -----------------------------
    # Detect fundamental PRIs
    # -----------------------------
    min_PRI = min([e["PRI"] for e in emitters])
    fundamentals, peak_lags_us, peaks = detect_fundamental_PRIs(
        auto_pos, lags_pos, N_emitters=2, min_PRI=min_PRI, smooth_signal=None
    )

    # -----------------------------
    # Assertions
    # -----------------------------
    assert len(fundamentals) <= 2
    assert all(f > 0 for f in fundamentals)
    assert np.all(np.isfinite(peak_lags_us))
