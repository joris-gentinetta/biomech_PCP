import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bessel, filtfilt

# ---- Load your data ----
raw_emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/2/raw_emg.npy")
raw_ts = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/2/raw_timestamps.npy")
raw_ts = raw_ts / 1e6
angles = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/2/angles.npy")
angle_ts = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/2/angle_timestamps.npy")

# --- Calculate sample rates ---
EMG_rate = 1/(raw_ts[-1]/raw_ts.shape[0])
angle_rate = 1/(angle_ts[-1]/angle_ts.shape[0])
print(f"EMG rate: {EMG_rate:.2f} Hz | Angle rate: {angle_rate:.2f} Hz")

fs = EMG_rate  # EMG sampling rate

# --- FILTER DESIGN HELPERS ---
def bessel_bandstop(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = bessel(order, [low, high], btype='bandstop', analog=False)
    return filtfilt(b, a, data)

def bessel_highpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = bessel(order, high, btype='highpass', analog=False)
    return filtfilt(b, a, data)

def bessel_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = bessel(order, low, btype='lowpass', analog=False)
    return filtfilt(b, a, data)

# ---- Choose signals ----
emg_raw = raw_emg[:, 14]   # Channel 15 (zero-based)
index_angle = angles[:, 2] # Adjust as needed

# ---- EMG FILTERING PIPELINE ----
# 1. Bandstop (notch) filter 58–62 Hz
emg_filtered = bessel_bandstop(emg_raw, 58, 62, fs)
# 2. Highpass 8 Hz
emg_filtered = bessel_highpass(emg_filtered, 8, fs)
# 3. Rectify (absolute value)
emg_rect = np.abs(emg_filtered)
# 4. Noise clip (estimate baseline noise, e.g. 1% quantile)
noise_level = np.percentile(emg_rect, 1)
emg_clipped = np.clip(emg_rect - noise_level, 0, None)
# 5. Lowpass 3 Hz (envelope)
emg_env = bessel_lowpass(emg_clipped, 3, fs)
# 6. Normalize to 0–1
emg_norm = np.clip(emg_env / np.max(emg_env), 0, 1)

# ---- Zero timestamps ----
emg_time = raw_ts - raw_ts[0]
angle_time = angle_ts - angle_ts[0]

# ---- Interpolate normalized EMG onto angle timebase for cross-correlation ----
emg_on_angle_time = np.interp(angle_time, emg_time, emg_norm)

# ---- Normalize for cross-correlation (z-score) ----
emg_z = (emg_on_angle_time - np.mean(emg_on_angle_time)) / np.std(emg_on_angle_time)
angle_z = (index_angle - np.mean(index_angle)) / np.std(index_angle)

# ---- Cross-correlation ----
corr = np.corrcoef(emg_z, angle_z)[0,1]

print(f"Correlation: {corr}")
lags = np.arange(-len(angle_z) + 1, len(angle_z))
lag_seconds = lags * np.mean(np.diff(angle_time))  # Convert lags to seconds

best_lag_idx = np.argmax(corr)
best_lag_sec = lag_seconds[best_lag_idx]
print(f"Best lag (EMG relative to angle): {best_lag_sec:.4f} seconds")

# ---- Plot cross-correlation ----
# plt.figure(figsize=(10, 4))
# plt.plot(lag_seconds, corr)
# plt.axvline(best_lag_sec, color='r', linestyle='--', label=f'Peak lag: {best_lag_sec:.3f} s')
# plt.xlabel("Lag (s) (EMG leads if positive)")
# plt.ylabel("Cross-correlation")
# plt.title("EMG (envelope) vs Angle Cross-correlation")
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---- Plot EMG and Angle with lag shift ----
angle_time_shifted = angle_time + best_lag_sec
angle_time_shifted = angle_time 

fig, ax1 = plt.subplots(figsize=(12, 5))
color_emg = 'tab:blue'
ax1.plot(emg_time, emg_norm, color=color_emg, label="EMG Channel 15 (env, 0-1)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("EMG Channel 15 (normalized)", color=color_emg)
ax1.tick_params(axis='y', labelcolor=color_emg)

color_angle = 'tab:orange'
ax2 = ax1.twinx()
ax2.plot(angle_time_shifted, index_angle, color=color_angle, label="Index Angle (shifted)", alpha=0.8)
ax2.set_ylabel("Index Angle", color=color_angle)
ax2.tick_params(axis='y', labelcolor=color_angle)

plt.title("Filtered & Normalized EMG Channel 15 and Index Angle")
plt.tight_layout()
plt.show()
