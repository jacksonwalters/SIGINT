import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000         # sampling rate (Hz)
duration = 0.010       # 10 ms total time
t = np.arange(0, duration, 1/fs)

# emitter 1
pulse_width1 = 5e-6    # 5 us
PRI1 = 100e-6          # 100 us
jitter1 = 5e-6         # standard deviation of PRI jitter

# emitter 2
pulse_width2 = 8e-6    # 8 us
PRI2 = 159e-6          # 150 us
jitter2 = 8e-6         # standard deviation of PRI jitter

# noise level (SNR in dB relative to pulse energy)
SNR_dB = -3.0          # try -3 dB (noisy)

np.random.seed(0)

# -----------------------------
# Generate jittered pulse trains
# -----------------------------
def generate_jittered_pulse_train(base_PRI, jitter_std, pulse_width, duration, fs):
    pulse_times = []
    t_curr = 0.0
    while t_curr < duration:
        pulse_times.append(t_curr)
        PRI_jittered = base_PRI + np.random.randn() * jitter_std
        t_curr += max(PRI_jittered, 1e-9)  # prevent negative/zero PRI
    t_array = np.arange(0, duration, 1/fs)
    rx = np.zeros_like(t_array)
    for pt in pulse_times:
        mask = (t_array >= pt) & (t_array < pt + pulse_width)
        rx[mask] = 1.0
    return rx, np.array(pulse_times)

rx1, times1 = generate_jittered_pulse_train(PRI1, jitter1, pulse_width1, duration, fs)
rx2, times2 = generate_jittered_pulse_train(PRI2, jitter2, pulse_width2, duration, fs)

# optionally give different amplitudes
rx2 *= 0.7

# Composite received signal
rx_clean = rx1 + rx2

# -----------------------------
# Add Gaussian noise
# -----------------------------
signal_power = np.mean(rx_clean**2)
SNR_linear = 10**(SNR_dB / 10)
noise_power = signal_power / SNR_linear
noise = np.random.normal(0, np.sqrt(noise_power), size=rx_clean.size)
rx = rx_clean + noise

# -----------------------------
# Visualize time-domain (brief)
# -----------------------------
plt.figure(figsize=(10, 3))
plt.plot(t*1e3, rx, linewidth=0.6)
plt.title(f"Composite Signal with Jitter (SNR={SNR_dB} dB)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.xlim(0, 2e-3*1e3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Autocorrelation (positive lags)
# -----------------------------
auto = correlate(rx, rx, mode='full')
lags = np.arange(-len(rx)+1, len(rx)) / fs
mid = len(auto)//2
auto_pos = auto[mid+1:]
lags_pos = lags[mid+1:]

# -----------------------------
# Smooth autocorrelation
# -----------------------------
smooth_ms = 2.0e-6
kernel_sigma = int(max(1, smooth_ms * fs))
kernel_len = kernel_sigma * 8 + 1
xi = np.arange(kernel_len) - kernel_len//2
kernel = np.exp(-0.5 * (xi / kernel_sigma)**2)
kernel = kernel / np.sum(kernel)
auto_pos_smooth = np.convolve(auto_pos, kernel, mode='same')

# -----------------------------
# Peak detection
# -----------------------------
prominence = 0.12 * np.max(auto_pos_smooth)
min_distance = int(0.3 * min(PRI1, PRI2) * fs)
peaks, props = find_peaks(auto_pos_smooth, prominence=prominence, distance=min_distance)
peak_lags = lags_pos[peaks]
peak_lags_us = peak_lags * 1e6

# -----------------------------
# Harmonic-rejection to find fundamental PRIs (fixed)
# -----------------------------
fundamentals = []
tol = 0.03  # 2% tolerance

for lag_us in np.sort(peak_lags_us):
    if lag_us < 0.5 * min(PRI1, PRI2) * 1e6:
        continue
    is_harmonic = False
    for f in fundamentals:
        ratio = lag_us / f
        # reject if it's an approximate integer multiple of a known fundamental
        if abs(ratio - round(ratio)) < tol:
            is_harmonic = True
            break
    if not is_harmonic:
        fundamentals.append(lag_us)

# Keep only the number of emitters
N_emitters = 2
fundamentals = fundamentals[:N_emitters]

# -----------------------------
# Print results
# -----------------------------
print(f"True PRIs: {PRI1*1e6:.1f} µs and {PRI2*1e6:.1f} µs")
print("Detected peak lags (µs):", np.round(peak_lags_us, 2))
print("Estimated fundamental PRIs (µs):", np.round(fundamentals, 2))

# -----------------------------
# Plot autocorrelation
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(lags_pos*1e6, auto_pos, alpha=0.35, label="autocorr (raw)")
plt.plot(lags_pos*1e6, auto_pos_smooth, linewidth=1.5, label="autocorr (smoothed)")
plt.plot(peak_lags_us, auto_pos_smooth[peaks], "ro", label="peaks")
for f in fundamentals:
    plt.axvline(f, color='k', linestyle='--', linewidth=1, alpha=0.9)
plt.xlim(0, 800)
plt.xlabel("Lag (µs)")
plt.ylabel("Autocorrelation amplitude")
plt.title("Autocorrelation with Jittered Multi-Emitter Pulses")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
