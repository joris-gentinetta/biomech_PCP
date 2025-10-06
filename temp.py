import numpy as np
import matplotlib.pyplot as plt

# Load the EMG data
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/handOpCl/experiments/1/aligned_filtered_emg.npy")  # shape: [channels, samples]
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/filtered_stream.npy")  # shape: [channels, samples]
emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/raw_stream.npy")  # shape: [channels, samples]
emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/Calibration/calib_rest_emg.npy")  # shape: [channels, samples]
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/Calibration/experiments/1/calib_mvc_filtered.npy")  # shape: [channels, samples]
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel/recordings/fingersFlEx/experiments/1/aligned_filtered_emg.npy")  # shape: [channels, samples]
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_freespace_filtered.npy")
# emg = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/normed_emg_live.npy")
maxVals = np.percentile(emg, 90, axis=1)
print(f"emg shape: {emg.shape}")
print(f"Max values: {maxVals}")
# Channel 15 is Python index 14
print(f"emg shape: {emg.shape}")
# for channel_idx in range(15):
channel_idx = 0

# print(f"emg shape: {emg_ch15.shape}")
for channel_idx in range(16):
    emg_ch15 = emg[:, channel_idx]
    plt.figure(figsize=(18, 5))
    plt.plot(np.arange(len(emg_ch15)), emg_ch15, lw=0.8)
    plt.title('normed (all samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

