import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bessel, filtfilt
import os

root = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/testt/recordings/handOpCl/experiments"
emg_channel = 14  # 0-based
angle_idx = 2     # adjust as needed

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

def process_trial(folder):
    raw_emg = np.load(os.path.join(folder, "raw_emg.npy"))
    raw_ts = np.load(os.path.join(folder, "raw_timestamps.npy")) / 1e6
    angles = np.load(os.path.join(folder, "angles.npy"))
    angle_ts = np.load(os.path.join(folder, "angle_timestamps.npy"))
    emg_time = raw_ts - raw_ts[0]
    angle_time = angle_ts - angle_ts[0]
    fs = 1 / (raw_ts[-1]/raw_ts.shape[0])
    emg_raw = raw_emg[:, emg_channel]
    emg_filtered = bessel_bandstop(emg_raw, 58, 62, fs)
    emg_filtered = bessel_highpass(emg_filtered, 8, fs)
    emg_rect = np.abs(emg_filtered)
    noise_level = np.percentile(emg_rect, 1)
    emg_clipped = np.clip(emg_rect - noise_level, 0, None)
    emg_env = bessel_lowpass(emg_clipped, 3, fs)
    emg_norm = np.clip(emg_env / np.max(emg_env), 0, 1)
    angle = angles[:, angle_idx]
    return emg_time, emg_norm, angle_time, angle

# --- Gather folders ---
folders = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.isdigit() and os.path.isdir(os.path.join(root, f))]
print(f"Found experiment folders: {folders}")

# --- Time-normalization setup ---
norm_points = 1000  # Number of points for normalized timeline
norm_time = np.linspace(0, 1, norm_points)  # 0% to 100% of trial

emg_trials = []
angle_trials = []

for folder in folders:
    emg_time, emg_norm, angle_time, angle = process_trial(folder)
    # Normalize time axis: map start to 0, end to 1
    angle_norm_time = (angle_time - angle_time[0]) / (angle_time[-1] - angle_time[0])
    emg_norm_time = (emg_time - emg_time[0]) / (emg_time[-1] - emg_time[0])
    # Interpolate onto normalized axis
    angle_interp = np.interp(norm_time, angle_norm_time, angle)
    emg_interp = np.interp(norm_time, emg_norm_time, emg_norm)
    angle_trials.append(angle_interp)
    emg_trials.append(emg_interp)

emg_trials = np.vstack(emg_trials)
angle_trials = np.vstack(angle_trials)

emg_min = np.min(emg_trials, axis=0)
emg_max = np.max(emg_trials, axis=0)
emg_mean = np.mean(emg_trials, axis=0)
angle_mean = np.mean(angle_trials, axis=0)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(12, 5))
color_emg = 'tab:blue'
ax1.plot(norm_time, emg_mean, color=color_emg, label="Mean EMG env (Ch 15)")
ax1.fill_between(norm_time, emg_min, emg_max, color=color_emg, alpha=0.2, label="EMG range")
ax1.set_xlabel("Normalized Movement (%)")
ax1.set_ylabel("EMG (env, 0-1)", color=color_emg)
ax1.tick_params(axis='y', labelcolor=color_emg)

color_angle = 'tab:orange'
ax2 = ax1.twinx()
ax2.plot(norm_time, angle_mean, color=color_angle, label="Mean Angle", alpha=0.7)
ax2.set_ylabel("Angle", color=color_angle)
ax2.tick_params(axis='y', labelcolor=color_angle)

fig.suptitle("Time-Normalized EMG Envelope Range Band and Angle Across All Trials")
fig.tight_layout()
plt.show()
