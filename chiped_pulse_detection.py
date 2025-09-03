import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000        # 1 MHz sampling rate
duration = 0.002      # 2 ms total time
t = np.arange(0, duration, 1/fs)

PRI = 100e-6          # 100 µs pulse repetition interval
pulse_width = 5e-6    # 5 µs pulse width

fc = 100_000          # carrier frequency
bandwidth = 50_000    # chirp bandwidth (Hz)
SNR_dB = 0

# -----------------------------
# Generate chirped pulse train
# -----------------------------
def chirp_pulse(t, t0, pulse_width, fc, bw):
    """Generate a single linear chirp pulse starting at t0"""
    pulse_mask = (t >= t0) & (t < t0 + pulse_width)
    tau = t[pulse_mask] - t0
    k = bw / pulse_width  # chirp rate (Hz/s)
    return pulse_mask, np.cos(2*np.pi*fc*tau + np.pi*k*tau**2)

rx_signal = np.zeros_like(t)

# Generate pulse times
pulse_times = np.arange(0, duration, PRI)
for pt in pulse_times:
    mask, pulse = chirp_pulse(t, pt, pulse_width, fc, bandwidth)
    rx_signal[mask] += pulse

# -----------------------------
# Add Gaussian noise
# -----------------------------
signal_power = np.mean(rx_signal**2)
SNR_linear = 10**(SNR_dB/10)
noise_power = signal_power / SNR_linear if SNR_linear != 0 else signal_power
noise = np.random.normal(0, np.sqrt(noise_power), size=t.size)
rx_signal_noisy = rx_signal + noise

# -----------------------------
# Plot time-domain signal
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t*1e3, rx_signal_noisy)
plt.title("Chirped Pulse Train in Noise (Time Domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# Matched filter using same chirp
# -----------------------------
# Create template (single pulse)
pulse_samples = int(pulse_width * fs)
tau_template = np.arange(pulse_samples) / fs
k = bandwidth / pulse_width
template = np.cos(2*np.pi*fc*tau_template + np.pi*k*tau_template**2)

# Convolve received signal with matched filter
mf_output = np.convolve(rx_signal_noisy, template[::-1], mode='same')

plt.figure(figsize=(12, 4))
plt.plot(t*1e3, mf_output)
plt.title("Matched Filter Output for Chirped Pulse Train")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# Autocorrelation of received signal
# -----------------------------
auto = correlate(rx_signal_noisy, rx_signal_noisy, mode='full')
lags = np.arange(-len(rx_signal_noisy)+1, len(rx_signal_noisy)) / fs

plt.figure(figsize=(12, 4))
plt.plot(lags*1e6, auto)
plt.title("Autocorrelation of Chirped Pulse Train")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

from scipy.signal import find_peaks

# -----------------------------
# PRI estimation
# -----------------------------
# Find peaks in matched filter output
peak_indices, _ = find_peaks(mf_output, height=0.5*np.max(mf_output))  # threshold 50% of max
peak_times = t[peak_indices]

# Compute differences between consecutive peaks
estimated_PRIs = np.diff(peak_times)
PRI_estimate = np.median(estimated_PRIs)

print(f"True PRI: {PRI*1e6:.2f} µs")
print(f"Estimated PRI from matched filter: {PRI_estimate*1e6:.2f} µs")

# -----------------------------
# Pulse width estimation
# -----------------------------
# Approximate pulse width from matched filter peak width at half maximum
pulse_widths = []

for idx in peak_indices:
    # Define small window around peak
    window = 20  # samples
    start = max(idx - window, 0)
    end = min(idx + window, len(mf_output))
    segment = mf_output[start:end]
    segment_times = t[start:end]

    half_max = mf_output[idx] / 2
    # Left side
    left_candidates = np.where(segment[:window] <= half_max)[0]
    if len(left_candidates) == 0:
        left_idx = 0
    else:
        left_idx = left_candidates[-1]
    # Right side
    right_candidates = np.where(segment[window:] <= half_max)[0]
    if len(right_candidates) == 0:
        right_idx = len(segment) - 1
    else:
        right_idx = right_candidates[0] + window

    width = segment_times[right_idx] - segment_times[left_idx]
    pulse_widths.append(width)

# Take median pulse width
pulse_width_estimate = np.median(pulse_widths)

print(f"True pulse width: {pulse_width*1e6:.2f} µs")
print(f"Estimated pulse width from matched filter: {pulse_width_estimate*1e6:.2f} µs")

# -----------------------------
# Plot matched filter with detected peaks
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t*1e3, mf_output, label="Matched Filter Output")
plt.plot(peak_times*1e3, mf_output[peak_indices], "ro", label="Detected Peaks")
plt.title("Matched Filter Output with Detected Peaks")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
