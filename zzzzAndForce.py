import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
movement = 'pinch_interaction'
root = f'data/Ema/recordings/{movement}/experiments/2'
emg = np.load(f'{root}/aligned_filtered_emg.npy')        # [N, 16]
ts  = np.load(f'{root}/aligned_timestamps.npy')          # [N]
angles = pd.read_parquet(f'{root}/aligned_angles.parquet')

# --- Normalize/flatten columns so selection is easy ---
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        # Case 1: actual tuple / MultiIndex level values
        if isinstance(c, tuple):
            new_cols.append('_'.join(map(str, c)))
            continue
        # Case 2: string that looks like a tuple: "('Right', 'index_Pos')"
        s = str(c)
        if s.startswith("(") and s.endswith(")") and "'" in s:
            try:
                t = ast.literal_eval(s)
                if isinstance(t, (tuple, list)):
                    new_cols.append('_'.join(map(str, t)))
                    continue
            except Exception:
                pass
        # Fallback
        new_cols.append(s)
    out = df.copy()
    out.columns = new_cols
    return out

angles = normalize_columns(angles)

# Quick sanity peek:
# print(angles.columns.tolist()[:10])

# --- Pick out index finger position & force (Right hand) ---
# Will match things like "Right_index_Pos" / "Right_index_Force"
pos_col = next((c for c in angles.columns if 'index_Pos' in c), None)
force_col = next((c for c in angles.columns if 'index_Force' in c), None)

if pos_col is None:
    raise RuntimeError("No column containing 'index_Pos' found in angles Parquet.")
if force_col is None:
    raise RuntimeError("No column containing 'index_Force' found in angles Parquet.")

angle_series = angles[pos_col].to_numpy()
force_series = angles[force_col].to_numpy()

# --- Your EMG selection as before ---
mapped_channels = [0, 1, 2, 3, 4, 5, 6]
labels = [
    'EMG 0 (mapped 0)', 'EMG 1 (mapped 1)', 'EMG 2 (mapped 2)', 'EMG 4 (mapped 3)',
    'EMG 12 (mapped 4)', 'EMG 13 (mapped 5)', 'EMG 14 (mapped 6)', 'EMG 15 (mapped 7)'
][:len(mapped_channels)]

# --- Plot EMG only (unchanged) ---
plt.figure(figsize=(15, 6))
for ch, lab in zip(mapped_channels, labels):
    plt.plot(ts, emg[:, ch], label=lab, alpha=0.8)
plt.xlabel('Time (s)'); plt.ylabel('EMG (a.u.)')
plt.title('Mapped EMG Channels vs Time'); plt.legend(ncol=2, fontsize=9)
plt.tight_layout(); plt.show()

# --- Plot angle + EMG + Force ---
fig, ax1 = plt.subplots(figsize=(15, 6))

# Angle on left axis
ax1.plot(ts, angle_series, linewidth=2, label=f'{pos_col}')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Finger angle (deg)')
ax1.tick_params(axis='y')

# EMG on right axis
ax2 = ax1.twinx()
for ch, lab in zip(mapped_channels, labels):
    ax2.plot(ts, emg[:, ch], label=lab, alpha=0.6)
ax2.set_ylabel('EMG (a.u.)')

# Optional: third axis for force (so scales don’t fight)
ax3 = ax1.twinx()
ax3.spines.right.set_position(("axes", 1.08))  # offset the 3rd axis
ax3.plot(ts, force_series, linewidth=2, linestyle='--', label=f'{force_col}')
ax3.set_ylabel('Force (a.u.)')

# Build a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

plt.title('Index Finger Position + EMG + Force')
plt.tight_layout(); plt.show()
