import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000        # 1 MHz sampling rate
duration = 0.002      # 2 ms total time
t = np.arange(0, duration, 1/fs)

pulse_width = 5e-6    # 5 µs pulse width
PRI = 100e-6          # 100 µs pulse repetition interval
SNR_dB = 0
true_delay = 15e-6    # 15 µs delay between receivers

# -----------------------------
# Generate pulse train at receiver 1
# -----------------------------
num_pulses = int(duration / PRI)
pulse_times = np.arange(0, num_pulses * PRI, PRI)
pulse_times = pulse_times[pulse_times < duration]

rx1 = np.zeros_like(t)
for pt in pulse_times:
    mask = (t >= pt) & (t < pt + pulse_width)
    rx1[mask] = 1.0  # unit amplitude pulses

# Add noise
signal_power = np.mean(rx1**2)
SNR_linear = 10**(SNR_dB/10) if SNR_dB != 0 else 1
noise_power = signal_power / SNR_linear
rx1_noisy = rx1 + np.random.normal(0, np.sqrt(noise_power), size=t.size)

# -----------------------------
# Generate receiver 2 signal (delayed)
# -----------------------------
delay_samples = int(true_delay * fs)
rx2_noisy = np.zeros_like(t)
if delay_samples < len(t):
    rx2_noisy[delay_samples:] = rx1_noisy[:len(t)-delay_samples]
rx2_noisy += np.random.normal(0, np.sqrt(noise_power), size=t.size)  # additional noise

# -----------------------------
# Cross-correlation
# -----------------------------
cross = correlate(rx2_noisy, rx1_noisy, mode='full')
lags = np.arange(-len(rx1_noisy)+1, len(rx1_noisy)) / fs

# Find peak in cross-correlation
peak_idx = np.argmax(cross)
estimated_delay = lags[peak_idx]

print(f"True delay: {true_delay*1e6:.2f} µs")
print(f"Estimated delay from cross-correlation: {estimated_delay*1e6:.2f} µs")

# -----------------------------
# Plot signals and cross-correlation
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)
plt.plot(t*1e3, rx1_noisy)
plt.title("Receiver 1 Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t*1e3, rx2_noisy)
plt.title("Receiver 2 Signal (Delayed)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(lags*1e6, cross)
plt.title("Cross-Correlation Between Receiver 2 and 1")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.axvline(estimated_delay*1e6, color='r', linestyle='--', label='Estimated delay')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
