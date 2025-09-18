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

def generate_multi_emitter_jittered_pulses(fs=1_000_000, duration=0.01,
                                            emitters=[{"PRI": 100e-6, "pulse_width": 5e-6, "jitter": 5e-6, "amplitude": 1.0},
                                                      {"PRI": 159e-6, "pulse_width": 8e-6, "jitter": 8e-6, "amplitude": 0.7}],
                                            SNR_dB=-3.0,
                                            seed=0):
    """
    Generate composite signal of multiple jittered emitters with optional noise.
    
    Returns:
        t: time vector
        rx_noisy: composite noisy signal
        pulse_times_list: list of arrays of pulse times per emitter
    """
    np.random.seed(seed)
    t = np.arange(0, duration, 1/fs)
    rx_clean = np.zeros_like(t)
    pulse_times_list = []

    for emitter in emitters:
        PRI, jitter_std, pulse_width, amplitude = emitter["PRI"], emitter["jitter"], emitter["pulse_width"], emitter.get("amplitude", 1.0)
        pulse_times = []
        t_curr = 0.0
        while t_curr < duration:
            pulse_times.append(t_curr)
            PRI_jittered = PRI + np.random.randn() * jitter_std
            t_curr += max(PRI_jittered, 1e-9)
        pulse_times = np.array(pulse_times)
        pulse_times_list.append(pulse_times)

        # Generate rectangular pulses
        rx = np.zeros_like(t)
        for pt in pulse_times:
            mask = (t >= pt) & (t < pt + pulse_width)
            rx[mask] = amplitude
        rx_clean += rx

    # Add Gaussian noise
    signal_power = np.mean(rx_clean**2)
    SNR_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=rx_clean.size)
    rx_noisy = rx_clean + noise

    return t, rx_noisy, pulse_times_list

def generate_pulse_train(PRI, pulse_width, duration, fs, amplitude=1.0):
    """
    Generate a rectangular pulse train with fixed PRI and width.
    
    Parameters
    ----------
    PRI : float
        Pulse Repetition Interval (seconds).
    pulse_width : float
        Pulse width (seconds).
    duration : float
        Total duration of signal (seconds).
    fs : float
        Sampling rate (Hz).
    amplitude : float
        Pulse amplitude.

    Returns
    -------
    rx : np.ndarray
        The generated signal.
    pulse_times : np.ndarray
        The start times of each pulse.
    """
    num_pulses = int(duration / PRI) + 1
    pulse_times = np.arange(0, num_pulses) * PRI
    pulse_times = pulse_times[pulse_times < duration]
    t = np.arange(0, duration, 1/fs)
    rx = np.zeros_like(t)
    for pt in pulse_times:
        mask = (t >= pt) & (t < pt + pulse_width)
        rx[mask] = amplitude
    return rx, pulse_times

def generate_gaussian_jittered_pulses(fs, duration, base_PRI, pulse_width, jitter_std, seed=0):
    """Generate a pulse train with Gaussian PRI jitter."""
    np.random.seed(seed)
    num_pulses = int(duration / base_PRI) + 5
    PRIs = base_PRI + np.random.randn(num_pulses) * jitter_std
    pulse_times = np.cumsum(PRIs)
    pulse_times = pulse_times[pulse_times < duration]

    t = np.arange(0, duration, 1/fs)
    rx = np.zeros_like(t)
    for pt in pulse_times:
        mask = (t >= pt) & (t < pt + pulse_width)
        rx[mask] = 1.0

    return t, rx, pulse_times

import numpy as np

def chirp_pulse(t, t0, pulse_width, fc, bw):
    """Generate a single linear chirp pulse starting at t0."""
    pulse_mask = (t >= t0) & (t < t0 + pulse_width)
    tau = t[pulse_mask] - t0
    k = bw / pulse_width  # chirp rate (Hz/s)
    pulse = np.cos(2*np.pi*fc*tau + np.pi*k*tau**2)
    return pulse_mask, pulse

def generate_chirped_train(fs, duration, PRI, pulse_width, fc, bandwidth):
    """Generate a chirped pulse train over a duration."""
    t = np.arange(0, duration, 1/fs)
    rx_signal = np.zeros_like(t)
    pulse_times = np.arange(0, duration, PRI)
    
    for pt in pulse_times:
        mask, pulse = chirp_pulse(t, pt, pulse_width, fc, bandwidth)
        rx_signal[mask] += pulse
        
    return t, rx_signal, pulse_times