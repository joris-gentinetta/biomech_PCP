import numpy as np
import matplotlib.pyplot as plt

# Load the EMG data
emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/handOpCl/experiments/1/aligned_filtered_emg.npy")  # shape: [channels, samples]
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/handOpCl/experiments/1/raw_emg.npy")  # shape: [channels, samples]
emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_freespace_filtered.npy")
maxVals = np.percentile(emg, 90, axis=1)
print(f"Max values: {maxVals}")
# Channel 15 is Python index 14
print(f"emg shape: {emg.shape}")
# for channel_idx in range(15):
channel_idx = 15
emg_ch15 = emg[channel_idx]
print(f"emg shape: {emg_ch15.shape}")

plt.figure(figsize=(18, 5))
plt.plot(np.arange(len(emg_ch15)), emg_ch15, lw=0.8)
plt.title('EMG Channel 15 (all samples)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
