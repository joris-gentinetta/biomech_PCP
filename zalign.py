import numpy as np
import matplotlib.pyplot as plt
# Load your files (update path as needed)
base_dir = "data/GG/recordings/handOpCl/experiments/1"  # <-- CHANGE THIS

# --- Load data ---
emg = np.load(f"{base_dir}/raw_emg.npy")          # shape: [samples, channels]
emg_ts = np.load(f"{base_dir}/raw_timestamps.npy")
angles = np.load(f"{base_dir}/angles.npy")        # shape: [samples, features]
angle_ts = np.load(f"{base_dir}/angle_timestamps.npy")

# --- Load header and find index_Pos ---
with open(f"{base_dir}/angles_header.txt", "r") as f:
    headers = [h.strip() for h in f.read().split(",")]
index_pos_col = headers.index("index_Pos")

# --- End-aligned synchronization ---
shift = emg_ts[-1] - angle_ts[-1]
emg_ts_shifted = emg_ts - shift
mask = (emg_ts_shifted >= 0) & (emg_ts_shifted <= angle_ts[-1])
emg_cut = emg[mask]
emg_t_cut = emg_ts_shifted[mask]

# --- Optional: interpolate EMG to angle timestamps for perfect alignment ---
emg_interp = np.interp(angle_ts, emg_t_cut, emg_cut[:, 0])

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot EMG channel 0 (blue, left y-axis, raw & interpolated)
ax1.plot(emg_t_cut, emg_cut[:, 0], color='tab:blue', alpha=0.5, label='EMG ch 0 (shifted, raw)')
ax1.plot(angle_ts, emg_interp, color='tab:blue', linewidth=2, linestyle='--', label='EMG ch 0 (interp)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('EMG Channel 0', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot index finger angle (red, right y-axis)
ax2 = ax1.twinx()
ax2.plot(angle_ts, angles[:, index_pos_col], color='tab:red', label='Index_Pos')
ax2.set_ylabel('Index Finger Position [deg]', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Legends and grid
lines, labels = [], []
for ax in [ax1, ax2]:
    line, label = ax.get_legend_handles_labels()
    lines += line
    labels += label
ax1.legend(lines, labels, loc='upper right')
ax1.grid(True, which='both', alpha=0.3)

plt.title("EMG Channel 0 (end-aligned) vs. Index Finger Angle")
plt.tight_layout()
plt.show()
