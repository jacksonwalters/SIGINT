import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks

# -----------------------------
# Parameters
# -----------------------------
fs = 1_000_000        # Sampling rate (Hz)
duration = 0.05       # Increased to 50 ms for more pulses
t = np.arange(0, duration, 1/fs)

base_PRI = 100e-6     # 100 µs base PRI
pulse_width = 5e-6    # 5 µs pulse width
jitter_std = 5e-6     # 5 µs standard deviation of PRI jitter
SNR_dB = 0            # Noise level

# -----------------------------
# Generate jittered pulse train
# -----------------------------
np.random.seed(0)
num_pulses = int(duration / base_PRI) + 5
PRIs = base_PRI + np.random.randn(num_pulses) * jitter_std
pulse_times = np.cumsum(PRIs)
pulse_times = pulse_times[pulse_times < duration]

print(f"Generated {len(pulse_times)} pulses")

rx = np.zeros_like(t)
for pt in pulse_times:
    mask = (t >= pt) & (t < pt + pulse_width)
    rx[mask] = 1.0

# -----------------------------
# Add Gaussian noise
# -----------------------------
signal_power = np.mean(rx**2)
SNR_linear = 10**(SNR_dB / 10) if SNR_dB != 0 else 1
noise_power = signal_power / SNR_linear
rx_noisy = rx + np.random.normal(0, np.sqrt(noise_power), size=rx.size)

# -----------------------------
# Improved matched filter detection
# -----------------------------
pulse_samples = int(pulse_width * fs)
template = np.ones(pulse_samples)

# Use 'valid' mode to avoid edge effects
mf_output = convolve(rx_noisy, template[::-1], mode='valid')
# Adjust time vector for valid convolution
t_valid = t[:len(mf_output)]

# Improved peak detection
# Use lower threshold and minimum distance based on expected PRI
threshold = 0.3 * np.max(mf_output)  # Lower threshold
min_PRI_samples = int((base_PRI - 3*jitter_std) * fs)  # Minimum expected PRI
min_distance_samples = max(min_PRI_samples, pulse_samples)

peaks, properties = find_peaks(mf_output, 
                              height=threshold, 
                              distance=min_distance_samples)

# Adjust detected times to account for template delay and valid mode offset
template_delay = (pulse_samples - 1) / 2 / fs
pulse_times_detected = t_valid[peaks] + template_delay

print(f"Detected {len(pulse_times_detected)} pulses")

# -----------------------------
# Compute PRI statistics
# -----------------------------
if len(pulse_times_detected) > 1:
    estimated_PRIs = np.diff(pulse_times_detected)
    mean_PRI = np.mean(estimated_PRIs)
    std_PRI = np.std(estimated_PRIs)
    
    # Remove outliers (more than 3 sigma from mean)
    valid_mask = np.abs(estimated_PRIs - mean_PRI) <= 3 * std_PRI
    estimated_PRIs_clean = estimated_PRIs[valid_mask]
    
    if len(estimated_PRIs_clean) > 1:
        mean_PRI_clean = np.mean(estimated_PRIs_clean)
        std_PRI_clean = np.std(estimated_PRIs_clean)
    else:
        mean_PRI_clean = mean_PRI
        std_PRI_clean = std_PRI
        estimated_PRIs_clean = estimated_PRIs
else:
    print("Error: Not enough pulses detected!")
    exit()

# -----------------------------
# Results
# -----------------------------
print(f"\nTrue parameters:")
print(f"Base PRI: {base_PRI*1e6:.2f} µs")
print(f"PRI jitter std: {jitter_std*1e6:.2f} µs")

print(f"\nEstimated parameters (raw):")
print(f"Mean PRI: {mean_PRI*1e6:.2f} µs")
print(f"PRI standard deviation: {std_PRI*1e6:.2f} µs")

print(f"\nEstimated parameters (outliers removed):")
print(f"Mean PRI: {mean_PRI_clean*1e6:.2f} µs")
print(f"PRI standard deviation: {std_PRI_clean*1e6:.2f} µs")

print(f"\nError analysis:")
print(f"Mean PRI error: {(mean_PRI_clean - base_PRI)*1e6:.2f} µs")
print(f"Std deviation error: {(std_PRI_clean - jitter_std)*1e6:.2f} µs")

# -----------------------------
# Plotting
# -----------------------------
# Plot a zoomed section of the pulse train
plt.figure(figsize=(12, 8))

# Subplot 1: Pulse train (zoomed)
plt.subplot(3, 1, 1)
zoom_start, zoom_end = 0, 0.005  # First 5ms
zoom_mask = (t >= zoom_start) & (t <= zoom_end)
plt.plot(t[zoom_mask]*1e3, rx_noisy[zoom_mask], 'b-', linewidth=0.7, label='Noisy signal')
plt.plot(t[zoom_mask]*1e3, rx[zoom_mask], 'r-', linewidth=1, alpha=0.7, label='Clean signal')

# Mark detected pulses in zoom window
detected_in_zoom = pulse_times_detected[(pulse_times_detected >= zoom_start) & 
                                       (pulse_times_detected <= zoom_end)]
for pt in detected_in_zoom:
    plt.axvline(pt*1e3, color='g', linestyle='--', alpha=0.7)

plt.title("Jittered PRI Pulse Train (Zoomed: first 5ms)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Matched filter output
plt.subplot(3, 1, 2)
zoom_mask_valid = (t_valid >= zoom_start) & (t_valid <= zoom_end)
plt.plot(t_valid[zoom_mask_valid]*1e3, mf_output[zoom_mask_valid], 'k-', linewidth=0.8)
plt.axhline(threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold = {threshold:.2f}')

# Mark detected peaks in zoom window
peaks_in_zoom_idx = peaks[(t_valid[peaks] >= zoom_start) & (t_valid[peaks] <= zoom_end)]
if len(peaks_in_zoom_idx) > 0:
    plt.plot(t_valid[peaks_in_zoom_idx]*1e3, mf_output[peaks_in_zoom_idx], 
             'ro', markersize=6, label='Detected peaks')

plt.title("Matched Filter Output")
plt.xlabel("Time (ms)")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Histogram of estimated PRIs
plt.subplot(3, 1, 3)
plt.hist(estimated_PRIs_clean*1e6, bins=20, color='skyblue', edgecolor='k', alpha=0.7)
plt.axvline(base_PRI*1e6, color='r', linestyle='--', linewidth=2, 
            label=f'True mean: {base_PRI*1e6:.1f} µs')
plt.axvline(mean_PRI_clean*1e6, color='g', linestyle='--', linewidth=2, 
            label=f'Estimated mean: {mean_PRI_clean*1e6:.1f} µs')
plt.title(f"Histogram of Estimated PRIs (n={len(estimated_PRIs_clean)})")
plt.xlabel("PRI (µs)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional diagnostic plot: PRI vs time
plt.figure(figsize=(10, 4))
pulse_centers = pulse_times_detected[:-1] + estimated_PRIs_clean/2
plt.plot(pulse_centers*1e3, estimated_PRIs_clean*1e6, 'bo-', markersize=4, linewidth=1)
plt.axhline(base_PRI*1e6, color='r', linestyle='--', linewidth=2, label='True PRI')
plt.axhline(mean_PRI_clean*1e6, color='g', linestyle='--', linewidth=2, label='Estimated mean PRI')
plt.fill_between([0, duration*1e3], 
                 [(base_PRI - jitter_std)*1e6]*2,
                 [(base_PRI + jitter_std)*1e6]*2,
                 alpha=0.2, color='red', label='±1σ true range')
plt.title("PRI Variation Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("PRI (µs)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()