import matplotlib.pyplot as plt

SHOW_PLOTS = False

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
