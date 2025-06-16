'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Load your data as before
emg = np.load('data/Emanuel6/recordings/indexFlDigitsEx/experiments/1/aligned_filtered_emg.npy')
angles = pd.read_parquet('data/Emanuel6/recordings/indexFlDigitsEx/experiments/1/aligned_angles.parquet')
idx_cols = [c for c in angles.columns if 'index_Pos' in c]
if not idx_cols:
    raise RuntimeError("No column containing 'index_Pos' found in angles DataFrame")
angle_series = angles[idx_cols[0]].values

mapped_channels = [0, 1, 2, 3, 4, 5, 6, 7]
labels = [
    'EMG 0 (mapped 0)', 'EMG 1 (mapped 1)', 'EMG 2 (mapped 2)', 'EMG 4 (mapped 3)',
    'EMG 12 (mapped 4)', 'EMG 13 (mapped 5)', 'EMG 14 (mapped 6)', 'EMG 15 (mapped 7)'
]
n_points = 200  # Number of resample points

# --- Find minima for full cycles (min-to-min)
minima, _ = find_peaks(-angle_series, distance=50)
if len(minima) < 2:
    raise RuntimeError("Not enough minima found for cycle detection!")

# --- Slice and stack cycles
all_emg_cycles = {ch: [] for ch in mapped_channels}
all_angle_cycles = []

for i in range(len(minima)-1):
    start, end = minima[i], minima[i+1]
    # Only consider cycles of reasonable length
    if end - start < 10: continue

    # Resample
    for ch in mapped_channels:
        y = emg[start:end, ch]
        x = np.linspace(0, 1, num=len(y))
        interp_func = interp1d(x, y, kind='linear')
        y_resamp = interp_func(np.linspace(0, 1, num=n_points))
        all_emg_cycles[ch].append(y_resamp)
    y_angle = angle_series[start:end]
    interp_func_angle = interp1d(np.linspace(0, 1, num=len(y_angle)), y_angle, kind='linear')
    all_angle_cycles.append(interp_func_angle(np.linspace(0, 1, num=n_points)))

for ch in all_emg_cycles:
    all_emg_cycles[ch] = np.vstack(all_emg_cycles[ch])
all_angle_cycles = np.vstack(all_angle_cycles)

# --- Plot overlayed cycles for each channel in a grid
n_channels = len(mapped_channels)
ncols = 4
nrows = int(np.ceil(n_channels / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), sharex=True, sharey=True)

x_axis = np.linspace(0, 1, n_points)

for i, ch in enumerate(mapped_channels):
    row, col = divmod(i, ncols)
    ax = axs[row, col]
    # Overlay all cycles
    for cycle in all_emg_cycles[ch]:
        ax.plot(x_axis, cycle, color='C{}'.format(i), alpha=0.12)
    # Plot mean ± std
    mean = np.mean(all_emg_cycles[ch], axis=0)
    std = np.std(all_emg_cycles[ch], axis=0)
    ax.plot(x_axis, mean, color='C{}'.format(i), lw=2, label=labels[ch])
    ax.fill_between(x_axis, mean-std, mean+std, color='C{}'.format(i), alpha=0.22)
    ax.set_title(labels[ch])
    if col == 0:
        ax.set_ylabel("EMG (a.u.)")
    if row == nrows - 1:
        ax.set_xlabel("Normalized Cycle (%)")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)
    
# Hide any empty subplots
for j in range(i+1, nrows*ncols):
    fig.delaxes(axs.flatten()[j])

plt.figure(figsize=(6,4))
for angle_cycle in all_angle_cycles:
    plt.plot(x_axis, angle_cycle, color='k', alpha=0.3)
plt.plot(x_axis, np.mean(all_angle_cycles, axis=0), color='r', lw=2, label='Mean')
plt.xlabel('Normalized Cycle')
plt.ylabel('Angle')
plt.title('All Angle Cycles (min-to-min, normalized)')
plt.legend()
plt.show()    

plt.suptitle("Time-Normalized EMG Envelope (min-to-min cycles)\nEach subplot = one EMG channel")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# -------- Load Data --------
emg = np.load('data/Emanuel6/recordings/indexFlDigitsEx/experiments/1/aligned_filtered_emg.npy')
angles = pd.read_parquet('data/Emanuel6/recordings/indexFlDigitsEx/experiments/1/aligned_angles.parquet')
idx_cols = [c for c in angles.columns if 'index_Pos' in c]
if not idx_cols:
    raise RuntimeError("No column containing 'index_Pos' found in angles DataFrame")
angle_series = angles[idx_cols[0]].values

mapped_channels = [0, 1, 2, 3, 4, 5, 6, 7]
labels = [
    'EMG 0 (mapped 0)', 'EMG 1 (mapped 1)', 'EMG 2 (mapped 2)', 'EMG 4 (mapped 3)',
    'EMG 12 (mapped 4)', 'EMG 13 (mapped 5)', 'EMG 14 (mapped 6)', 'EMG 15 (mapped 7)'
]
n_points = 200  # Number of resample points

# -------- Detect Proper Cycles: min-to-min with exactly one max in between --------
minima, _ = find_peaks(-angle_series, distance=50)
maxima, _ = find_peaks(angle_series, distance=50)

cycle_starts = []
cycle_ends = []
for i in range(len(minima) - 1):
    start = minima[i]
    end = minima[i + 1]
    # Count how many maxima are within this segment
    max_in_segment = [m for m in maxima if start < m < end]
    if len(max_in_segment) == 1:
        cycle_starts.append(start)
        cycle_ends.append(end)

print(f"Found {len(cycle_starts)} clean cycles with min→max→min.")

# -------- Slice and Resample Each Cycle --------
all_emg_cycles = {ch: [] for ch in mapped_channels}
all_angle_cycles = []

for start, end in zip(cycle_starts, cycle_ends):
    # Only consider cycles of reasonable length
    if end - start < 10: continue
    for ch in mapped_channels:
        y = emg[start:end, ch]
        x = np.linspace(0, 1, num=len(y))
        interp_func = interp1d(x, y, kind='linear')
        y_resamp = interp_func(np.linspace(0, 1, num=n_points))
        all_emg_cycles[ch].append(y_resamp)
    y_angle = angle_series[start:end]
    interp_func_angle = interp1d(np.linspace(0, 1, num=len(y_angle)), y_angle, kind='linear')
    all_angle_cycles.append(interp_func_angle(np.linspace(0, 1, num=n_points)))

for ch in all_emg_cycles:
    all_emg_cycles[ch] = np.vstack(all_emg_cycles[ch])
all_angle_cycles = np.vstack(all_angle_cycles)
x_axis = np.linspace(0, 1, n_points)

# -------- (Optional) Plot All Angle Cycles to Check --------
plt.figure(figsize=(10,4))
for angle_cycle in all_angle_cycles:
    plt.plot(x_axis, angle_cycle, color='k', alpha=0.2)
plt.plot(x_axis, np.mean(all_angle_cycles, axis=0), color='r', lw=2, label='Mean')
plt.xlabel('Normalized Cycle')
plt.ylabel('Angle')
plt.title('All Angle Cycles (min→max→min, normalized)')
plt.legend()
plt.tight_layout()
plt.show()

# -------- Plot EMG and Angle Cycles in Grid --------
n_channels = len(mapped_channels)
ncols = 4
nrows = int(np.ceil(n_channels / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), sharex=True, sharey=True)

for i, ch in enumerate(mapped_channels):
    row, col = divmod(i, ncols)
    ax = axs[row, col]
    # Overlay all EMG cycles
    for cycle in all_emg_cycles[ch]:
        ax.plot(x_axis, cycle, color='C{}'.format(i), alpha=0.12)
    # Plot mean ± std for EMG
    mean = np.mean(all_emg_cycles[ch], axis=0)
    std = np.std(all_emg_cycles[ch], axis=0)
    ax.plot(x_axis, mean, color='C{}'.format(i), lw=2, label=labels[ch])
    ax.fill_between(x_axis, mean-std, mean+std, color='C{}'.format(i), alpha=0.22)
    # Overlay all angle cycles (in gray)
    for angle_cycle in all_angle_cycles:
        ax.plot(x_axis, angle_cycle, color='k', lw=1.0, alpha=0.13, zorder=1, label='_nolegend_')
    # Optionally overlay mean angle in bold
    ax.plot(x_axis, np.mean(all_angle_cycles, axis=0), color='k', lw=2, ls='--', label='Angle (mean)')
    ax.set_title(labels[ch])
    if col == 0:
        ax.set_ylabel("EMG (a.u.)")
    if row == nrows - 1:
        ax.set_xlabel("Normalized Cycle (%)")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

# Hide any empty subplots
for j in range(i+1, nrows*ncols):
    fig.delaxes(axs.flatten()[j])

plt.suptitle("Time-Normalized EMG Envelope (one full cycle: min→max→min)\nEach subplot = one EMG channel + angle overlays")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


