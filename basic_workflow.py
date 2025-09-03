import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1_000_000        # 1 MHz sampling rate
duration = 0.002      # 2 ms total time
t = np.arange(0, duration, 1/fs)

fc = 100_000          # 100 kHz carrier frequency
PRI = 100e-6          # 100 µs pulse repetition interval
pulse_width = 5e-6    # 5 µs pulse width
SNR_dB = 0            # SNR in dB

# Pulse train (rectangular gate)
gate = (np.mod(t, PRI) < pulse_width).astype(float)

# Modulate with carrier
clean_signal = gate * np.cos(2*np.pi*fc*t)

# Add Gaussian noise
signal_power = np.mean(clean_signal**2)
SNR_linear = 10**(SNR_dB/10)
noise_power = signal_power / SNR_linear
noise = np.random.normal(0, np.sqrt(noise_power), size=t.size)
rx_signal = clean_signal + noise

# Plot time domain
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(t*1e3, rx_signal)
plt.title("Received Pulse Train in Noise (Time Domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

# Plot frequency domain
from scipy.fft import fft, fftfreq

N = len(rx_signal)
X = fft(rx_signal)
freqs = fftfreq(N, 1/fs)

plt.subplot(2, 2, 2)
plt.plot(freqs/1e3, 20*np.log10(np.abs(X)))
plt.xlim(0, fs/2/1e3)  # Plot up to Nyquist
plt.title("Spectrum of Received Signal")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")

# CORRECTED: Create proper matched filter pulse
# Method 1: Extract one pulse from the clean signal
pulse_samples = int(pulse_width * fs)
pulse_template = np.zeros(pulse_samples)
pulse_template[:] = 1.0  # Rectangle pulse

# Method 2: Or use the same logic as the gate generation
# pulse_template = (np.arange(pulse_samples) < pulse_samples).astype(float)

# Matched filter
mf_output = np.convolve(rx_signal, pulse_template[::-1], mode='same')

plt.subplot(2, 2, 3)
plt.plot(t*1e3, mf_output)
plt.title("Matched Filter Output")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

# Autocorrelation
from scipy.signal import correlate

auto = correlate(rx_signal, rx_signal, mode='full')
lags = np.arange(-len(rx_signal)+1, len(rx_signal))

plt.subplot(2, 2, 4)
plt.plot(lags/fs*1e6, auto)
plt.title("Autocorrelation of Received Signal")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Find peaks for PRI estimation
from scipy.signal import find_peaks

peaks, properties = find_peaks(auto, height=0.1*np.max(auto))
peak_lags = lags[peaks] / fs
positive_lags = peak_lags[peak_lags > 0]
estimated_PRIs = np.diff(positive_lags)
PRI_estimate = np.median(estimated_PRIs)

print(f"True PRI: {PRI*1e6:.2f} µs")
print(f"Estimated PRI: {PRI_estimate*1e6:.2f} µs")

# CORRECTED: Pulse width estimation using autocorrelation
zoom_range = 20e-6  # 20 µs
zoom_mask = (lags/fs > -zoom_range) & (lags/fs < zoom_range)
zoom_lags = lags[zoom_mask] / fs
zoom_auto = auto[zoom_mask]

# Find central peak
center_idx = np.argmax(zoom_auto)
center_value = zoom_auto[center_idx]

# IMPORTANT FIX: For a rectangular pulse, the autocorrelation has a triangular shape
# The base width of this triangle equals the pulse width
# Method 1: Find where autocorrelation drops to near zero
threshold = 0.01 * center_value  # 1% of peak

# Left side
left_idx_candidates = np.where(zoom_auto[:center_idx] <= threshold)[0]
if len(left_idx_candidates) == 0:
    left_idx = 0
else:
    left_idx = left_idx_candidates[-1]

# Right side  
right_idx_candidates = np.where(zoom_auto[center_idx:] <= threshold)[0]
if len(right_idx_candidates) == 0:
    right_idx = len(zoom_auto) - 1
else:
    right_idx = right_idx_candidates[0] + center_idx

# Compute pulse width estimate (base width of triangular autocorrelation)
pulse_width_estimate_base = zoom_lags[right_idx] - zoom_lags[left_idx]

# Method 2: FWHM approach (should give pulse_width/2 for rectangular pulse)
half_max = center_value / 2

left_idx_half = np.where(zoom_auto[:center_idx] <= half_max)[0]
if len(left_idx_half) == 0:
    left_idx_half = 0
else:
    left_idx_half = left_idx_half[-1]

right_idx_half = np.where(zoom_auto[center_idx:] <= half_max)[0]
if len(right_idx_half) == 0:
    right_idx_half = len(zoom_auto) - 1
else:
    right_idx_half = right_idx_half[0] + center_idx

pulse_width_estimate_fwhm = zoom_lags[right_idx_half] - zoom_lags[left_idx_half]

print(f"\nTrue pulse width: {pulse_width*1e6:.2f} µs")
print(f"Estimated pulse width (base width): {pulse_width_estimate_base*1e6:.2f} µs")
print(f"Estimated pulse width (FWHM): {pulse_width_estimate_fwhm*1e6:.2f} µs")
print(f"Note: For rectangular pulses, FWHM ≈ pulse_width/2, base width ≈ pulse_width")

# Plot zoomed autocorrelation with both measurements
plt.figure(figsize=(10, 6))
plt.plot(zoom_lags*1e6, zoom_auto, 'b-', linewidth=2, label="Autocorrelation")
plt.axhline(half_max, color='orange', linestyle=':', alpha=0.7, label='Half maximum')
plt.axhline(threshold, color='red', linestyle=':', alpha=0.7, label='Base threshold')
plt.axvline(zoom_lags[left_idx]*1e6, color="r", linestyle="--", alpha=0.7, label="Base width")
plt.axvline(zoom_lags[right_idx]*1e6, color="r", linestyle="--", alpha=0.7)
plt.axvline(zoom_lags[left_idx_half]*1e6, color="orange", linestyle="--", alpha=0.7, label="FWHM")
plt.axvline(zoom_lags[right_idx_half]*1e6, color="orange", linestyle="--", alpha=0.7)
plt.axvline(0, color='black', linestyle='-', alpha=0.3)
plt.title("Autocorrelation Analysis for Pulse Width Estimation")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()