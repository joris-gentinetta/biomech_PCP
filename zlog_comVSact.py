import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
filename = "C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/data/Emanuel9.12/logs/xx.csv"  # <-- Update with your actual filename

# --- Load CSV ---
df = pd.read_csv(filename, sep=None, engine='python')

# --- Extract time and position columns ---
time = df['Timestamp'].values
pos_com = df['index_PosCom'].values  # Commanded position
pos_act = df['index_Pos'].values     # Actual position

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(time, pos_com, label='Commanded Position', linewidth=2)
plt.plot(time, pos_act, label='Actual Position', linewidth=2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Index Position')
plt.title('Actual vs. Commanded Index Finger Position')
plt.legend()
plt.tight_layout()
plt.show()
