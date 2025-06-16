# analyze_live_emg.py
import numpy as np
import matplotlib.pyplot as plt
import json

# Load the data
raw_emg = np.load('data/Emanuel/recordings/handOpCl/experiments/1/raw_emg.npy')
filtered_emg = np.load('data/Emanuel/recordings/handOpCl/experiments/1/filtered_emg_live.npy')
normed_emg = np.load('data/Emanuel/recordings/handOpCl/experiments/1/normed_emg_live.npy')
timestamps = np.load('data/Emanuel/recordings/handOpCl/experiments/1/emg_live_timestamps.npy')

with open('emg_debug_info.json', 'r') as f:
    debug_info = json.load(f)

print("=== LIVE EMG ANALYSIS ===")
print(f"Channels used: {debug_info['usedChannels']}")
print(f"Config features: {debug_info['config_features']}")
print(f"Raw EMG range: [{raw_emg.min():.3f}, {raw_emg.max():.3f}]")
print(f"Filtered EMG range: [{filtered_emg.min():.3f}, {filtered_emg.max():.3f}]")
print(f"Normalized EMG range: [{normed_emg.min():.3f}, {normed_emg.max():.3f}]")

# Plot all stages for first channel
plt.figure(figsize=(15, 10))

for i, ch in enumerate(debug_info['usedChannels'][:4]):  # Plot first 4 channels
    plt.subplot(4, 3, i*3 + 1)
    plt.plot(timestamps, raw_emg[i, :])
    plt.title(f'Channel {ch} - Raw EMG')
    plt.ylabel('Raw ADC')
    
    plt.subplot(4, 3, i*3 + 2)
    plt.plot(timestamps, filtered_emg[i, :])
    plt.title(f'Channel {ch} - Filtered EMG')
    plt.ylabel('Filtered')
    
    plt.subplot(4, 3, i*3 + 3)
    plt.plot(timestamps, normed_emg[i, :])
    plt.title(f'Channel {ch} - Normalized EMG')
    plt.ylabel('Normalized [0-1]')
    plt.ylim([0, 1])

plt.tight_layout()
plt.show()

# Compare with training data ranges
print("\n=== COMPARISON WITH TRAINING ===")
print("Load some training data and compare ranges...")