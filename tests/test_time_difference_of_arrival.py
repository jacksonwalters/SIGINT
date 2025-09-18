# tests/test_time_difference_of_arrival.py
import numpy as np
from sigint_examples.simulation import generate_time_delay_pulses
from sigint_examples.processing import estimate_delay
from sigint_examples.plotting import SHOW_PLOTS
import matplotlib.pyplot as plt

def test_tdoa_basic():
    fs = 1_000_000          # 1 MHz
    duration = 0.002         # 2 ms
    pulse_width = 5e-6       # 5 µs
    PRI = 100e-6             # 100 µs
    true_delay = 15e-6       # 15 µs
    SNR_dB = 0

    # Generate signals
    t, rx1, rx2 = generate_time_delay_pulses(
        fs, duration, pulse_width, PRI, delay=true_delay, SNR_dB=SNR_dB, seed=42
    )

    # Basic sanity checks
    assert len(t) == int(duration * fs) + 1
    assert np.mean(rx1**2) > 0  # signal has positive power
    assert np.mean(rx2**2) > 0

    # Estimate delay
    estimated_delay, lags, cross_corr = estimate_delay(rx1, rx2, fs)

    # Check estimated delay is close to true delay
    error = abs(estimated_delay - true_delay)
    print(f"True delay: {true_delay*1e6:.2f} µs, Estimated delay: {estimated_delay*1e6:.2f} µs, Error: {error*1e6:.2f} µs")
    assert error < 2e-6  # within 2 µs tolerance

    # Additional checks
    assert len(lags) == len(cross_corr)
    assert np.max(cross_corr) > 0

    if SHOW_PLOTS:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t*1e3, rx1, label="Receiver 1")
        plt.plot(t*1e3, rx2, label="Receiver 2")
        plt.title("TDOA Signals")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(lags*1e6, cross_corr)
        plt.axvline(estimated_delay*1e6, color='r', linestyle='--', label='Estimated delay')
        plt.title("Cross-Correlation")
        plt.xlabel("Lag (µs)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
