import numpy as np
import matplotlib.pyplot as plt

# ---- Load your data ----
raw_emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/1/raw_emg.npy")
raw_ts = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/1/raw_timestamps.npy")
angles = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/1/angles.npy")
angle_ts = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments/1/angle_timestamps.npy")

# ---- Choose signals (adjust if needed) ----
emg_ch15 = raw_emg[:, 14]  # Or raw_emg[:, 14] if you want channel 14
index_angle = angles[:, 2]  # Or another column if you prefer

# ---- Zero timestamps ----
emg_time = raw_ts - raw_ts[0]
angle_time = angle_ts - angle_ts[0]

# ---- Resample EMG onto angle timebase (so arrays are same length) ----
# Interpolation is okay for cross-correlation
emg_ch15_on_angle_time = np.interp(angle_time, emg_time, emg_ch15)

# ---- Normalize (zero mean, unit std) for fair cross-correlation ----
emg_norm = (emg_ch15_on_angle_time - np.mean(emg_ch15_on_angle_time)) / np.std(emg_ch15_on_angle_time)
angle_norm = (index_angle - np.mean(index_angle)) / np.std(index_angle)

# ---- Cross-correlation ----
corr = np.correlate(emg_norm, angle_norm, mode='full')
lags = np.arange(-len(angle_norm) + 1, len(angle_norm))
lag_seconds = lags * np.mean(np.diff(angle_time))  # Convert lags to seconds

# ---- Find the lag at which cross-corr is maximal ----
best_lag_idx = np.argmax(corr)
best_lag_sec = lag_seconds[best_lag_idx]

print(f"Best lag (EMG relative to angle): {best_lag_sec:.4f} seconds")

# ---- Plot ----
plt.figure(figsize=(10, 4))
plt.plot(lag_seconds, corr)
plt.axvline(best_lag_sec, color='r', linestyle='--', label=f'Peak lag: {best_lag_sec:.3f} s')
plt.xlabel("Lag (s) (EMG leads if positive)")
plt.ylabel("Cross-correlation")
plt.title("EMG vs Angle Cross-correlation")
plt.legend()
plt.tight_layout()
plt.show()
