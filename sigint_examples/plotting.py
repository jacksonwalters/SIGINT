import matplotlib.pyplot as plt

SHOW_PLOTS = True

def plot_time_domain(t, signal, title="Signal"):
    plt.figure(figsize=(12,4))
    plt.plot(t*1e3, signal)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def plot_autocorr(lags, auto, peak_lags=None, title="Autocorrelation"):
    plt.figure(figsize=(12,4))
    plt.plot(lags*1e6, auto, label="Autocorrelation")
    if peak_lags is not None:
        plt.plot(peak_lags*1e6, auto[[int(p) for p in peak_lags]], "ro", label="Detected Peaks")
    plt.title(title)
    plt.xlabel("Lag (µs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_histogram(data, bins=10, title="Histogram", xlabel="Value", ylabel="Count"):
    plt.figure(figsize=(8,4))
    plt.hist(data*1e6, bins=bins, color='skyblue', edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_multi_emitter_autocorr(lags_pos, auto_pos, auto_pos_smooth, peak_lags_us, fundamentals, title="Autocorrelation"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(lags_pos*1e6, auto_pos, alpha=0.35, label="autocorr (raw)")
    plt.plot(lags_pos*1e6, auto_pos_smooth, linewidth=1.5, label="autocorr (smoothed)")
    plt.plot(peak_lags_us, auto_pos_smooth[peak_lags_us.astype(int)], "ro", label="peaks")
    for f in fundamentals:
        plt.axvline(f, color='k', linestyle='--', linewidth=1, alpha=0.9)
    plt.xlabel("Lag (µs)")
    plt.ylabel("Autocorrelation amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pulse_train_zoom(t, rx, rx_noisy, pulse_times_detected, zoom_start=0, zoom_end=0.005):
    plt.figure(figsize=(12, 4))
    zoom_mask = (t >= zoom_start) & (t <= zoom_end)
    plt.plot(t[zoom_mask]*1e3, rx_noisy[zoom_mask], 'b-', linewidth=0.7, label='Noisy signal')
    plt.plot(t[zoom_mask]*1e3, rx[zoom_mask], 'r-', linewidth=1, alpha=0.7, label='Clean signal')

    detected_in_zoom = pulse_times_detected[(pulse_times_detected >= zoom_start) &
                                           (pulse_times_detected <= zoom_end)]
    for pt in detected_in_zoom:
        plt.axvline(pt*1e3, color='g', linestyle='--', alpha=0.7)

    plt.title("Jittered PRI Pulse Train (Zoomed)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_matched_filter(t_valid, mf_output, peaks, threshold, zoom_start=0, zoom_end=0.005):
    plt.figure(figsize=(12, 4))
    zoom_mask_valid = (t_valid >= zoom_start) & (t_valid <= zoom_end)
    plt.plot(t_valid[zoom_mask_valid]*1e3, mf_output[zoom_mask_valid], 'k-', linewidth=0.8)
    plt.axhline(threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold = {threshold:.2f}')

    peaks_in_zoom_idx = peaks[(t_valid[peaks] >= zoom_start) & (t_valid[peaks] <= zoom_end)]
    if len(peaks_in_zoom_idx) > 0:
        plt.plot(t_valid[peaks_in_zoom_idx]*1e3, mf_output[peaks_in_zoom_idx],
                 'ro', markersize=6, label='Detected peaks')

    plt.title("Matched Filter Output")
    plt.xlabel("Time (ms)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_PRI_histogram(estimated_PRIs_clean, base_PRI, mean_PRI_clean):
    plt.figure(figsize=(8, 4))
    plt.hist(estimated_PRIs_clean*1e6, bins=20, color='skyblue', edgecolor='k', alpha=0.7)
    plt.axvline(base_PRI*1e6, color='r', linestyle='--', linewidth=2, label=f'True mean: {base_PRI*1e6:.1f} µs')
    plt.axvline(mean_PRI_clean*1e6, color='g', linestyle='--', linewidth=2, label=f'Estimated mean: {mean_PRI_clean*1e6:.1f} µs')
    plt.title(f"Histogram of Estimated PRIs (n={len(estimated_PRIs_clean)})")
    plt.xlabel("PRI (µs)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_PRI_vs_time(pulse_times_detected, estimated_PRIs_clean, base_PRI, mean_PRI_clean, jitter_std, duration):
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
    plt.tight_layout()
    plt.show()

def plot_tdoa(t, rx1, rx2, lags, cross_corr, estimated_delay):
    plt.figure(figsize=(12, 6))

    plt.subplot(3,1,1)
    plt.plot(t*1e3, rx1)
    plt.title("Receiver 1 Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t*1e3, rx2)
    plt.title("Receiver 2 Signal (Delayed)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(lags*1e6, cross_corr)
    plt.title("Cross-Correlation Between Receiver 2 and 1")
    plt.xlabel("Lag (µs)")
    plt.ylabel("Amplitude")
    plt.axvline(estimated_delay*1e6, color='r', linestyle='--', label='Estimated delay')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_two_pulse_autocorr(positive_lags, auto_positive, fundamentals):
    plt.figure(figsize=(12,4))
    plt.plot(positive_lags*1e6, auto_positive, label="Autocorrelation")
    for f in fundamentals:
        plt.axvline(f, color='r', linestyle='--', alpha=0.7, label=f"Fundamental {f:.1f} µs")
    plt.title("Autocorrelation with Fundamental PRI Estimates")
    plt.xlabel("Lag (µs)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_basic_workflow(t, rx_signal, clean_signal, mf_output, auto, lags, 
                        peak_lags=None, pulse_width_estimates=None):
    
    plt.figure(figsize=(12, 8))
    
    # Time-domain
    plt.subplot(2,2,1)
    plt.plot(t*1e3, rx_signal, label="Received")
    plt.plot(t*1e3, clean_signal, alpha=0.7, label="Clean")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Received Pulse Train")
    plt.grid(True)
    plt.legend()

    # Matched filter
    plt.subplot(2,2,2)
    plt.plot(t*1e3, mf_output)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Matched Filter Output")
    plt.grid(True)

    # Autocorrelation
    plt.subplot(2,2,3)
    plt.plot(lags*1e6, auto)
    if peak_lags is not None:
        for pl in peak_lags:
            plt.axvline(pl*1e6, color='r', linestyle='--', alpha=0.7)
    plt.xlabel("Lag (µs)")
    plt.ylabel("Amplitude")
    plt.title("Autocorrelation")
    plt.grid(True)

    # Optional: pulse width estimates overlay
    if pulse_width_estimates is not None:
        for pw in pulse_width_estimates:
            plt.axhline(pw, color='orange', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()
