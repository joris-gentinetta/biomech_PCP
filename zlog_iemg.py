import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
filename = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel6/logs/zzgg.csv"

# --- Load CSV ---
df = pd.read_csv(filename, sep='\t')  # Note: Your data appears to be tab-separated

# --- Extract time and selected iEMG channels ---
time = df['Timestamp'].values
channel_map = [
    'iEMG0',   # 0
    'iEMG1',   # 1
    'iEMG2',   # 2
    'iEMG4',   # 3
    'iEMG12',  # 4
    'iEMG13',  # 5
    'iEMG14',  # 6
    'iEMG15',  # 7
]
emg = df[channel_map].values

# --- Plot WITHOUT additional normalization ---
plt.figure(figsize=(13, 6))
for i, ch in enumerate(channel_map):
    plt.plot(time, emg[:, i], label=f'{ch} (mapped {i})', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Normalized iEMG (0-1)')
plt.title('Normalized iEMG Channels vs Time')
plt.ylim(0, 1.1)  # Set y-axis limits to show the normalized range
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()

# Optional: Print statistics to verify
print("EMG Statistics:")
for i, ch in enumerate(channel_map):
    print(f"{ch}: min={emg[:, i].min():.3f}, max={emg[:, i].max():.3f}, mean={emg[:, i].mean():.3f}")