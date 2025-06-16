import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
emg = np.load('data/GG/recordings/handOpCl/experiments/12/aligned_filtered_emg.npy')  # shape [N,16]
ts  = np.load('data/GG/recordings/handOpCl/experiments/12/aligned_timestamps.npy')
angles = pd.read_parquet('data/GG/recordings/handOpCl/experiments/12/aligned_angles.parquet')

# Pick out index finger position column
idx_cols = [c for c in angles.columns if 'index_Pos' in c]
if not idx_cols:
    raise RuntimeError("No column containing 'index_Pos' found in angles DataFrame")
angle_series = angles[idx_cols[0]].values

# Your mapped channel indices and pretty names
# mapped_channels = [0, 1, 2, 3, 4, 5, 6, 7]
mapped_channels = [0]
# labels = [
#     'EMG 0 (mapped 0)', 'EMG 1 (mapped 1)', 'EMG 2 (mapped 2)', 'EMG 4 (mapped 3)',
#     'EMG 12 (mapped 4)', 'EMG 13 (mapped 5)', 'EMG 14 (mapped 6)', 'EMG 15 (mapped 7)'
# ]
labels = ['EMG 0 (mapped 0)']

# plt.figure(figsize=(15, 6))
# for ch, lab in zip(mapped_channels, labels):
#     plt.plot(ts, emg[:, ch], label=lab, alpha=0.8)
# plt.xlabel('Time (s)')
# plt.ylabel('EMG (a.u.)')
# plt.title('Mapped EMG Channels vs Time')
# plt.legend(ncol=2, fontsize=9)
# plt.tight_layout()
# plt.show()

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(15, 6))

# Plot index finger position on left y-axis
ax1.plot(ts, angle_series, color='k', linewidth=2, label='Index_Pos')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Index_Pos (deg)', color='k')
ax1.tick_params(axis='y', labelcolor='k')

# Create a second y-axis for EMG channels
ax2 = ax1.twinx()
for ch, lab in zip(mapped_channels, labels):
    ax2.plot(ts, emg[:, ch], label=lab, alpha=0.7)
ax2.set_ylabel('EMG (a.u.)')
ax2.tick_params(axis='y')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Index Finger Position vs. Multiple EMG Channels')
plt.tight_layout()
plt.show()
