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
plt.plot(t*1e3, rx_signal)
plt.title("Received Pulse Train in Noise (Time Domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.show()

# Plot frequency domain

from scipy.fft import fft, fftfreq

N = len(rx_signal)
X = fft(rx_signal)
freqs = fftfreq(N, 1/fs)

plt.plot(freqs/1e3, 20*np.log10(np.abs(X)))
plt.xlim(0, fs/2/1e3)  # Plot up to Nyquist
plt.title("Spectrum of Received Signal")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.show()

# Matched filter (just the pulse shape)
pulse = (np.arange(0, pulse_width, 1/fs) < pulse_width).astype(float)
mf_output = np.convolve(rx_signal, pulse[::-1], mode='same')

plt.plot(t*1e3, mf_output)
plt.title("Matched Filter Output")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.show()

from scipy.signal import correlate

auto = correlate(rx_signal, rx_signal, mode='full')
lags = np.arange(-len(rx_signal)+1, len(rx_signal))
plt.plot(lags/fs*1e6, auto)
plt.title("Autocorrelation of Received Signal")
plt.xlabel("Lag (µs)")
plt.ylabel("Amplitude")
plt.savefig("autocorrelation_plot.png", dpi=300)  # PNG with 300 dpi
plt.show()

from scipy.signal import find_peaks

# Find peaks in autocorrelation (ignore the central peak at lag 0 if desired)
peaks, properties = find_peaks(auto, height=0.1*np.max(auto))  # threshold 10% of max

# Convert peak indices to time lags
peak_lags = lags[peaks] / fs  # in seconds

# Compute differences between consecutive peaks to estimate PRI
# (Use positive lags only for convenience)
positive_lags = peak_lags[peak_lags > 0]
estimated_PRIs = np.diff(positive_lags)

# If PRI is constant, take the median
PRI_estimate = np.median(estimated_PRIs)

print(f"Estimated PRI: {PRI_estimate*1e6:.2f} µs")



