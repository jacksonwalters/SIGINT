import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000        # 1 MHz sampling rate
duration = 0.002      # 2 ms total time
t = np.arange(0, duration, 1/fs)

pulse_width = 5e-6    # 5 µs pulse width
fc = 100_000          # carrier frequency
SNR_dB = 0

# -----------------------------
# Generate irregular pulse times
# -----------------------------
# Base PRI = 100 µs, add random jitter ±10 µs
base_PRI = 100e-6
num_pulses = int(duration / base_PRI)
np.random.seed(0)
PRI_jitter = np.random.uniform(-10e-6, 10e-6, num_pulses)
pulse_times = np.cumsum(base_PRI + PRI_jitter)
pulse_times = pulse_times[pulse_times < duration]

# -----------------------------
# Generate rectangular pulse train
# -----------------------------
rx_signal = np.zeros_like(t)

for pt in pulse_times:
    mask = (t >= pt) & (t < pt + pulse_width)
    rx_signal[mask] = 1.0  # rectangular pulse

# -----------------------------
# Add noise
# -----------------------------
signal_power = np.mean(rx_signal**2)
SNR_linear = 10**(SNR_dB/10) if SNR_dB != 0 else 1
noise_power = signal_power / SNR_linear
noise = np.random.normal(0, np.sqrt(noise_power), size=t.size)
rx_signal_noisy = rx_signal + noise

# -----------------------------
# Plot time-domain signal
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t*1e3, rx_signal_noisy)
plt.title("Irregular Pulse Train in Noise (Time Domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# Matched filter
# -----------------------------
pulse_samples = int(pulse_width * fs)
template = np.ones(pulse_samples)
mf_output = np.convolve(rx_signal_noisy, template[::-1], mode='same')

plt.figure(figsize=(12, 4))
plt.plot(t*1e3, mf_output)
plt.title("Matched Filter Output")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# Autocorrelation
# -----------------------------
auto = correlate(rx_signal_noisy, rx_signal_noisy, mode='full')
lags = np.arange(-len(rx_signal_noisy)+1, len(rx_signal_noisy)) / fs

plt.figure(figsize=(12, 4))
plt.plot(lags*1e6, auto)
plt.title("Autocorrelation of Irregular Pulse Train")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# Estimate PRI statistics
# -----------------------------
# Find peaks in matched filter output
peak_indices, _ = find_peaks(mf_output, height=0.5*np.max(mf_output))
peak_times = t[peak_indices]

# Compute PRI differences
estimated_PRIs = np.diff(peak_times)
mean_PRI = np.mean(estimated_PRIs)
std_PRI = np.std(estimated_PRIs)

print(f"Base PRI: {base_PRI*1e6:.2f} µs")
print(f"Estimated mean PRI: {mean_PRI*1e6:.2f} µs")
print(f"Estimated PRI standard deviation: {std_PRI*1e6:.2f} µs")

# Plot histogram of estimated PRIs
plt.figure(figsize=(8, 4))
plt.hist(estimated_PRIs*1e6, bins=10, color='skyblue', edgecolor='k')
plt.title("Histogram of Estimated PRIs")
plt.xlabel("PRI (µs)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.show()
