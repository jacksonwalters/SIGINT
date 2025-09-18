import numpy as np
from scipy.signal import correlate, find_peaks

def matched_filter(signal, pulse_width, fs):
    pulse_samples = int(pulse_width * fs)
    template = np.ones(pulse_samples)
    return np.convolve(signal, template[::-1], mode='same')

def autocorrelation(signal):
    auto = correlate(signal, signal, mode='full')
    lags = np.arange(-len(signal)+1, len(signal)) / 1.0  # will scale outside
    return auto, lags

def estimate_PRI_from_autocorr(auto, lags, threshold_factor=0.1):
    # Only positive lags
    pos_mask = lags > 0
    auto_pos = auto[pos_mask]
    lags_pos = lags[pos_mask]

    peaks, _ = find_peaks(auto_pos, height=threshold_factor*np.max(auto_pos))
    peak_lags = lags_pos[peaks]

    estimated_PRIs = np.diff(peak_lags)
    PRI_estimate = np.median(estimated_PRIs)
    return PRI_estimate, peak_lags, auto_pos, lags_pos

def estimate_pulse_width_from_autocorr(auto, lags, zoom_range=20e-6, fs=1e6):
    center_idx = len(auto)//2
    zoom_mask = (np.arange(len(auto)) >= center_idx - int(zoom_range*fs)) & \
                (np.arange(len(auto)) <= center_idx + int(zoom_range*fs))
    zoom_lags = lags[zoom_mask]
    zoom_auto = auto[zoom_mask]

    center_value = zoom_auto[np.argmax(zoom_auto)]
    threshold = 0.01 * center_value

    # Left side
    left_idx_candidates = np.where(zoom_auto[:len(zoom_auto)//2] <= threshold)[0]
    left_idx = left_idx_candidates[-1] if len(left_idx_candidates) > 0 else 0

    # Right side
    right_idx_candidates = np.where(zoom_auto[len(zoom_auto)//2:] <= threshold)[0]
    right_idx = right_idx_candidates[0] + len(zoom_auto)//2 if len(right_idx_candidates) > 0 else len(zoom_auto)-1

    pulse_width_estimate = zoom_lags[right_idx] - zoom_lags[left_idx]
    return pulse_width_estimate, zoom_lags, zoom_auto, left_idx, right_idx

def estimate_PRI_from_mf(mf_output, t, threshold_ratio=0.5):
    peaks, _ = find_peaks(mf_output, height=threshold_ratio*np.max(mf_output))
    peak_times = t[peaks]
    estimated_PRIs = np.diff(peak_times)
    mean_PRI = np.mean(estimated_PRIs)
    std_PRI = np.std(estimated_PRIs)
    return mean_PRI, std_PRI, peak_times, estimated_PRIs
