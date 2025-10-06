import numpy as np
import matplotlib.pyplot as plt
'''
# --- Load aligned data ---
aligned_emg = np.load('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/handOpCl/experiments/1/aligned_filtered_emg.npy')     # [N, 16]
# aligned_emg = np.load('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/handOpCl/experiments/1/raw_emg.npy')     # [N, 16]
aligned_angles = np.load('C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/handOpCl/experiments/1/aligned_angles.npy')        # [N, 6]

print(f"emg shape: {aligned_emg.shape}")
print(f"angle shape: {aligned_angles.shape}")

# --- Select channels/angles to plot ---
emg_channel = 0       # change to another channel if you want 0, 2, 11
angle_channel = 2     # change to another angle if you want

# --- Optional: Plot more than one angle/channel if you wish ---
# emg_channels = [0, 1, 2]   # plot 3 channels, for example
# angle_channels = [0, 1]

fig, ax1 = plt.subplots(figsize=(12, 6))

# X-axis: sample index (or create a time array if you want)
x = np.arange(aligned_emg.shape[0])

# --- Plot EMG channel (normalized: range [0,1]) ---
ax1.plot(x, aligned_emg[:, emg_channel], label=f'EMG Channel {emg_channel}', color='b')
ax1.set_ylabel('Normalized EMG', color='b')
ax1.set_xlabel('Sample (aligned)')
ax1.tick_params(axis='y', labelcolor='b')

# --- Plot joint angle on secondary y-axis ---
ax2 = ax1.twinx()
ax2.plot(x, aligned_angles[:, angle_channel], label=f'Angle {angle_channel}', color='r', alpha=0.6)
ax2.set_ylabel('Joint Angle', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# --- Optional: add more channels/angles if desired ---
# for ch in emg_channels:
#     ax1.plot(x, aligned_emg[:, ch], label=f'EMG {ch}')
# for ang in angle_channels:
#     ax2.plot(x, aligned_angles[:, ang], label=f'Angle {ang}', alpha=0.5)

# --- Titles and Legends ---
plt.title('Aligned EMG and Angle')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.show()
'''

import pandas as pd
df = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/gg/recordings/handOpCl/experiments/1/aligned_filtered_emg.npy")
# df = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/gg/recordings/handOpCl/experiments/1/raw_emg.npy")
# print(df.columns)
print(df.shape)
