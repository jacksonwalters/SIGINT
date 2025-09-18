import numpy as np

def generate_amplitude_modulated_pulses(fs=1_000_000, duration=0.002,
                                        pulse_width=5e-6, PRI=100e-6, SNR_dB=0,
                                        seed=0):
    """
    Generate amplitude-modulated pulses with optional noise.
    
    Returns:
        t: time vector
        rx_signal_noisy: noisy signal
        pulse_times: pulse start times
    """
    t = np.arange(0, duration, 1/fs)
    num_pulses = int(duration / PRI)
    np.random.seed(seed)
    pulse_amplitudes = 1 + 0.5 * np.random.randn(num_pulses)
    pulse_times = np.arange(0, num_pulses * PRI, PRI)
    pulse_times = pulse_times[pulse_times < duration]
    
    rx_signal = np.zeros_like(t)
    for amp, pt in zip(pulse_amplitudes, pulse_times):
        mask = (t >= pt) & (t < pt + pulse_width)
        rx_signal[mask] = amp
    
    # Add noise
    signal_power = np.mean(rx_signal**2)
    SNR_linear = 10**(SNR_dB/10) if SNR_dB != 0 else 1
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=t.size)
    rx_signal_noisy = rx_signal + noise
    
    return t, rx_signal_noisy, pulse_times

import numpy as np

def generate_irregular_pulses(fs=1_000_000, duration=0.002,
                              pulse_width=5e-6, base_PRI=100e-6,
                              PRI_jitter_max=10e-6, SNR_dB=0, seed=0):
    """
    Generate a rectangular irregular pulse train with optional noise.

    Returns:
        t: time vector
        rx_signal_noisy: noisy signal
        pulse_times: pulse start times
    """
    t = np.arange(0, duration, 1/fs)
    np.random.seed(seed)
    num_pulses = int(duration / base_PRI)
    PRI_jitter = np.random.uniform(-PRI_jitter_max, PRI_jitter_max, num_pulses)
    pulse_times = np.cumsum(base_PRI + PRI_jitter)
    pulse_times = pulse_times[pulse_times < duration]

    rx_signal = np.zeros_like(t)
    for pt in pulse_times:
        mask = (t >= pt) & (t < pt + pulse_width)
        rx_signal[mask] = 1.0

    # Add noise
    signal_power = np.mean(rx_signal**2)
    SNR_linear = 10**(SNR_dB/10) if SNR_dB != 0 else 1
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=t.size)
    rx_signal_noisy = rx_signal + noise

    return t, rx_signal_noisy, pulse_times
