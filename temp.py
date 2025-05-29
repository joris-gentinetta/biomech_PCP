#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Adjust these if your files are elsewhere:
    DATA_FILE = "data/raw_stream.npy"      # shape (N,16)
    TS_FILE   = "data/raw_timestamps.npy"  # shape (N,)
    # DATA_FILE = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/1test/recordings/handOpCl/experiments/1/raw_emg.npy" # shape (N,16)
    # TS_FILE   = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/1test/recordings/handOpCl/experiments/1/raw_timestamps.npy"  # shape (N,)

    # 1) Load
    emg  = np.load(DATA_FILE)      # float32, (N,16)
    ts   = np.load(TS_FILE)        # seconds, (N,)

    # 2) If your timestamps were microsecond‐based, uncomment:
    # if ts.max() > 1000: ts = ts / 1e6

    # 3) Make a 4×4 grid of subplots
    # fig, axes = plt.subplots(1, 1, figsize=(14, 10), sharex=True)
    # axes = axes.flatten()

    # 4) Plot each channel
    # for ch in range(16):

    # ch = 5 [0,1,2,4,12,13,14,15]
    ch = 15
    for ch in range(16):
        plt.figure(figsize=(14,5))
        plt.plot(ts, emg[:, ch], linewidth=0.7)
        plt.title(f"Raw EMG Channel {ch}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# # Paths
# BASE = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/gg/recordings/handOpCl/experiments/1"
# EMG_NPY    = f"{BASE}/aligned_filtered_emg.npy"
# TS_NPY     = f"{BASE}/aligned_timestamps.npy"
# ANGLES_PQ  = f"{BASE}/aligned_angles.parquet"
# 
# # Load
# emg = np.load(EMG_NPY)
# ts = np.load(TS_NPY)
# angles_pq = pd.read_parquet(ANGLES_PQ)
# 
# # Use first EMG channel (adjust if needed)
# emg_ch = 0
# 
# # Find the correct index_Pos column string
# col = None
# for c in angles_pq.columns:
#     if "index_Pos" in c:
#         col = c
#         break
# if col is None:
#     raise ValueError("Could not find index_Pos in the parquet columns!")
# 
# # Use timestamp from parquet, or fallback to ts
# if "timestamp" in angles_pq.columns:
#     parquet_ts = angles_pq["timestamp"]
# else:
#     parquet_ts = ts
# 
# # Plot
# fig, ax1 = plt.subplots(figsize=(12, 4))
# ax1.plot(ts, emg[:, emg_ch], label='EMG ch 0', color='tab:blue')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('EMG', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# 
# ax2 = ax1.twinx()
# ax2.plot(parquet_ts, angles_pq[col], label=f'{col} (parquet)', color='tab:orange')
# ax2.set_ylabel('index_Pos', color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')
# 
# plt.title('EMG (from .npy) and index_Pos (from .parquet) over Time')
# fig.tight_layout()
# plt.show()
