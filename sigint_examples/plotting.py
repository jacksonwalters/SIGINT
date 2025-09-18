import matplotlib.pyplot as plt

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