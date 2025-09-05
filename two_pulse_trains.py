import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000        # 1 MHz sampling rate
duration = 0.005      # 5 ms total time
t = np.arange(0, duration, 1/fs)

pulse_width1 = 5e-6    # 5 µs
PRI1 = 100e-6          # 100 µs
pulse_width2 = 8e-6    # 8 µs
PRI2 = 150e-6          # 150 µs

# -----------------------------
# Generate pulse trains
# -----------------------------
def generate_pulse_train(PRI, pulse_width):
    num_pulses = int(duration / PRI)
    pulse_times = np.arange(0, num_pulses * PRI, PRI)
    rx = np.zeros_like(t)
    for pt in pulse_times:
        mask = (t >= pt) & (t < pt + pulse_width)
        rx[mask] = 1.0
    return rx

rx1 = generate_pulse_train(PRI1, pulse_width1)
rx2 = generate_pulse_train(PRI2, pulse_width2)
rx = rx1 + rx2

# -----------------------------
# Autocorrelation
# -----------------------------
auto = correlate(rx, rx, mode='full')
lags = np.arange(-len(rx)+1, len(rx)) / fs
positive_lags = lags[len(lags)//2:]
auto_positive = auto[len(auto)//2:]

# -----------------------------
# Find peaks
# -----------------------------
peaks, _ = find_peaks(auto_positive, height=0.3*np.max(auto_positive))
peak_lags = positive_lags[peaks]  # in seconds
peak_lags_us = peak_lags * 1e6

# -----------------------------
# Extract fundamentals
# -----------------------------
fundamentals = []
tolerance = 5.0  # µs tolerance for rejecting harmonics

for lag in sorted(peak_lags_us):
    if lag < 20:  # ignore very small lags near zero
        continue
    # Check if lag is a harmonic of any already-accepted fundamental
    is_harmonic = False
    for f in fundamentals:
        ratio = lag / f
        if abs(ratio - round(ratio)) < 0.05:  # within 5% of integer multiple
            is_harmonic = True
            break
    if not is_harmonic:
        fundamentals.append(lag)

print("True PRIs: {:.1f} µs and {:.1f} µs".format(PRI1*1e6, PRI2*1e6))
print("Estimated fundamental PRIs (µs):", np.round(fundamentals, 2))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(positive_lags*1e6, auto_positive, label="Autocorrelation")
for f in fundamentals:
    plt.axvline(f, color='r', linestyle='--', alpha=0.7, label=f"Fundamental {f:.1f} µs")
plt.title("Autocorrelation with Fundamental PRI Estimates")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
