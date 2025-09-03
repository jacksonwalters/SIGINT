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

# -----------------------------
# Generate amplitude-modulated pulses
# -----------------------------
num_pulses = int(duration / PRI)
np.random.seed(0)
pulse_amplitudes = 1 + 0.5 * np.random.randn(num_pulses)  # mean=1, std=0.5

pulse_times = np.arange(0, num_pulses * PRI, PRI)
pulse_times = pulse_times[pulse_times < duration]

rx_signal = np.zeros_like(t)
for amp, pt in zip(pulse_amplitudes, pulse_times):
    mask = (t >= pt) & (t < pt + pulse_width)
    rx_signal[mask] = amp  # amplitude-modulated pulse

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
plt.title("Amplitude-Modulated Pulse Train in Noise (Time Domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# Matched filter (optional visualization)
# -----------------------------
pulse_samples = int(pulse_width * fs)
template = np.ones(pulse_samples)
mf_output = np.convolve(rx_signal_noisy, template[::-1], mode='same')

plt.figure(figsize=(12, 4))
plt.plot(t*1e3, mf_output)
plt.title("Matched Filter Output (Optional)")
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
plt.title("Autocorrelation of Amplitude-Modulated Pulse Train")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# -----------------------------
# PRI estimation using autocorrelation
# -----------------------------
# Only positive lags
pos_mask = lags > 0
auto_pos = auto[pos_mask]
lags_pos = lags[pos_mask]

# Find peaks in autocorrelation
peaks, properties = find_peaks(auto_pos, height=0.1*np.max(auto_pos))
peak_lags = lags_pos[peaks]

# Compute differences between consecutive peaks to estimate PRI
estimated_PRIs = np.diff(peak_lags)
PRI_estimate = np.median(estimated_PRIs)

print(f"True PRI: {PRI*1e6:.2f} µs")
print(f"Estimated PRI from autocorrelation: {PRI_estimate*1e6:.2f} µs")

# -----------------------------
# Plot autocorrelation with detected peaks
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(lags_pos*1e6, auto_pos, label="Autocorrelation")
plt.plot(peak_lags*1e6, auto_pos[peaks], "ro", label="Detected Peaks")
plt.title("Autocorrelation with Detected Peaks")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# Histogram of estimated PRIs
plt.figure(figsize=(8, 4))
plt.hist(estimated_PRIs*1e6, bins=10, color='skyblue', edgecolor='k')
plt.title("Histogram of Estimated PRIs")
plt.xlabel("PRI (µs)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# Pulse width estimation using autocorrelation
# -----------------------------

# Zoom around central peak (±20 µs)
zoom_range = 20e-6
center_idx = len(auto)//2
zoom_mask = (np.arange(len(auto)) >= center_idx - int(zoom_range*fs)) & \
            (np.arange(len(auto)) <= center_idx + int(zoom_range*fs))
zoom_lags = lags[zoom_mask]
zoom_auto = auto[zoom_mask]

# Find central peak
center_value = zoom_auto[np.argmax(zoom_auto)]

# Base threshold near zero
threshold = 0.01 * center_value  # 1% of peak

# Left side
left_idx_candidates = np.where(zoom_auto[:len(zoom_auto)//2] <= threshold)[0]
if len(left_idx_candidates) == 0:
    left_idx = 0
else:
    left_idx = left_idx_candidates[-1]

# Right side
right_idx_candidates = np.where(zoom_auto[len(zoom_auto)//2:] <= threshold)[0]
if len(right_idx_candidates) == 0:
    right_idx = len(zoom_auto) - 1
else:
    right_idx = right_idx_candidates[0] + len(zoom_auto)//2

# Compute pulse width estimate (base width of triangle)
pulse_width_estimate = zoom_lags[right_idx] - zoom_lags[left_idx]

print(f"True pulse width: {pulse_width*1e6:.2f} µs")
print(f"Estimated pulse width from autocorrelation: {pulse_width_estimate*1e6:.2f} µs")

# -----------------------------
# Plot zoomed autocorrelation with estimated edges
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(zoom_lags*1e6, zoom_auto, label="Autocorrelation")
plt.axvline(zoom_lags[left_idx]*1e6, color="r", linestyle="--", label="Estimated edges")
plt.axvline(zoom_lags[right_idx]*1e6, color="r", linestyle="--")
plt.title("Zoomed Autocorrelation for Pulse Width Estimation")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
