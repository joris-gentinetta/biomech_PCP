import numpy as np
import matplotlib.pyplot as plt
'''
# Define your filter (from earlier in the conversation)
def filter_emg_pipeline_bessel(raw_emg, fs, noise_level=None):
    from scipy.signal import bessel, sosfilt
    import numpy as np

    num_channels = raw_emg.shape[0]
    # 1. Bandstop (powerline)
    sos_notch = [bessel(8, [58/(fs/2), 62/(fs/2)], btype='bandstop', output='sos') for _ in range(num_channels)]
    emg = np.vstack([sosfilt(sos_notch[ch], raw_emg[ch]) for ch in range(num_channels)])
    # 2. Highpass (20 Hz)
    sos_hp = [bessel(4, 20/(fs/2), btype='highpass', output='sos') for _ in range(num_channels)]
    emg = np.vstack([sosfilt(sos_hp[ch], emg[ch]) for ch in range(num_channels)])
    # 3. Rectification
    emg = np.abs(emg)
    # 4. Noise subtraction and clipping
    if noise_level is not None:
        emg = np.clip(emg - noise_level[:, None], 0, None)
    # 5. Lowpass (envelope, 3 Hz)
    sos_lp = [bessel(4, 3/(fs/2), btype='lowpass', output='sos') for _ in range(num_channels)]
    emg = np.vstack([sosfilt(sos_lp[ch], emg[ch]) for ch in range(num_channels)])

    return emg  # same shape as input

# Load files
free_data = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_freespace_emg.npy")
free_ts  = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_freespace_timestamps.npy")
mvc_data = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_mvc_emg.npy")
mvc_ts = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_mvc_timestamps.npy")
rest_data = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_rest_emg.npy")
rest_ts = np.load("C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1/calib_rest_timestamps.npy")
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate fs
def get_fs(timestamps):
    # Handles edge case where only one timestamp is present
    if len(timestamps) < 2:
        return None
    duration = timestamps[-1] - timestamps[0]
    if duration == 0:
        return None
    return (len(timestamps) - 1) / duration

# Calculate fs for each segment
fs_free = get_fs(free_ts)
fs_mvc = get_fs(mvc_ts)
fs_rest = get_fs(rest_ts)

print(f"Sampling Frequency Free: {fs_free:.2f} Hz")
print(f"Sampling Frequency MVC:  {fs_mvc:.2f} Hz")
print(f"Sampling Frequency Rest: {fs_rest:.2f} Hz")

# Paths to your calibration files (adapt as needed)
base_dir = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1"  # Set to your calibration folder

rest = np.load(f"{base_dir}/calib_rest_emg.npy")   # [samples, 16]
mvc = np.load(f"{base_dir}/calib_mvc_emg.npy")
free = np.load(f"{base_dir}/calib_freespace_emg.npy")

channels = [14, 15]  # Python indices for channel 14 and 15
labels = ['Channel 14', 'Channel 15']

segments = [
    ('Rest', rest, '#1f77b4'),
    ('MVC', mvc, '#ff7f0e'),
    ('Free Space', free, '#2ca02c')
]

plt.figure(figsize=(14, 6))

for idx, ch in enumerate(channels):
    plt.subplot(1, 2, idx+1)
    for name, data, color in segments:
        y = data[:, ch]
        x = np.arange(len(y))
        plt.plot(x, y, label=name, color=color)
    plt.title(labels[idx])
    plt.xlabel('Sample')
    plt.ylabel('Raw EMG')
    plt.legend()
    plt.tight_layout()

plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from helpers.EMGClass import EMG
from helpers.BesselFilter import BesselFilterArr

# Your filtering function
def filter_emg_pipeline_bessel(raw_emg, fs, noise_level=None):
    num_channels = raw_emg.shape[0]
    notch = BesselFilterArr(numChannels=num_channels, order=8, critFreqs=[58,62], fs=fs, filtType='bandstop')
    emg = notch.filter(raw_emg)
    hp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=20, fs=fs, filtType='highpass')
    emg = hp.filter(emg)
    emg = np.abs(emg)
    if noise_level is not None:
        emg = np.clip(emg - noise_level[:, None], 0, None)
    lp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=3, fs=fs, filtType='lowpass')
    emg = lp.filter(emg)
    return emg

# Load data and timestamps
base_dir = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/ew/recordings/Calibration/experiments/1"

rest_data = np.load(f"{base_dir}/calib_rest_emg.npy").T
rest_ts = np.load(f"{base_dir}/calib_rest_timestamps.npy")

mvc_data = np.load(f"{base_dir}/calib_mvc_emg.npy").T
mvc_ts = np.load(f"{base_dir}/calib_mvc_timestamps.npy")

free_data = np.load(f"{base_dir}/calib_freespace_emg.npy").T
free_ts = np.load(f"{base_dir}/calib_freespace_timestamps.npy")

# Calculate sampling frequencies
def get_fs(timestamps):
    if len(timestamps) < 2:
        return None
    duration = timestamps[-1] - timestamps[0]
    if duration == 0:
        return None
    return (len(timestamps) - 1) / duration

fs_rest = get_fs(rest_ts)
fs_mvc = get_fs(mvc_ts)
fs_free = get_fs(free_ts)

print(f"Sampling Frequency Rest: {fs_rest:.2f} Hz")
print(f"Sampling Frequency MVC:  {fs_mvc:.2f} Hz")
print(f"Sampling Frequency Free: {fs_free:.2f} Hz")

# Assume noise level is zero or replace with actual noise if available
noise_levels = np.zeros(rest_data.shape[0])
noise_levels[14] = 8.58804052606238 
noise_levels[15] = 9.182965099316622 
maxVals = np.ones(rest_data.shape[0])
maxVals[14] = 300.58804052606238 
maxVals[15] = 300.182965099316622

# Filter each segment manually
filtered_rest_manual = filter_emg_pipeline_bessel(rest_data, fs_rest, noise_levels)
filtered_mvc_manual = filter_emg_pipeline_bessel(mvc_data, fs_mvc, noise_levels)
filtered_free_manual = filter_emg_pipeline_bessel(free_data, fs_free, noise_levels)

# Filter each segment with EMGClass pipeline
def filter_with_emgclass(raw_emg, fs):
    emg_proc = EMG(samplingFreq=fs, offlineData=raw_emg, maxVals=maxVals, noiseLevel=noise_levels, numElectrodes=raw_emg.shape[0])
    emg_proc.startCommunication()
    emg_proc.emgThread.join()
    filtered = emg_proc.emgHistory[:, emg_proc.numPackets * 100 + 1:]
    filtered = filtered[:, :raw_emg.shape[1]]
    return filtered

filtered_rest_emgclass = filter_with_emgclass(rest_data, fs_rest)
filtered_mvc_emgclass = filter_with_emgclass(mvc_data, fs_mvc)
filtered_free_emgclass = filter_with_emgclass(free_data, fs_free)

eps = 1e-9
norm_rest = np.clip(filtered_rest_manual / (maxVals[:, None] + eps), 0, 1)
norm_mvc = np.clip(filtered_mvc_manual / (maxVals[:, None] + eps), 0, 1)
norm_free = np.clip(filtered_free_manual / (maxVals[:, None] + eps), 0, 1)

# Plotting comparison for channels 14 and 15 
channels = [14, 15]
segment_names = ['Rest', 'MVC', 'Free Space']
# manual_segments = [filtered_rest_manual, filtered_mvc_manual, filtered_free_manual]
manual_segments = [norm_rest, norm_mvc, norm_free]
emgclass_segments = [filtered_rest_emgclass, filtered_mvc_emgclass, filtered_free_emgclass]

for ch in channels:
    plt.figure(figsize=(15, 10))
    for i, seg_name in enumerate(segment_names):
        plt.subplot(len(segment_names), 1, i+1)
        plt.plot(manual_segments[i][ch], label='Manual filter_emg_pipeline_bessel')
        plt.plot(emgclass_segments[i][ch], label='EMGClass', alpha=0.7)
        plt.title(f"Channel {ch+1} - {seg_name}")
        plt.xlabel("Sample")
        plt.ylabel("Filtered EMG")
        plt.legend()
    plt.tight_layout()
    plt.show()
