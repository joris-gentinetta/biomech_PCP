import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions and ground truth
df = pd.read_parquet('data/Emanuel9.12-2/recordings/thumbFlEx/experiments/1/pred_angles-Emanuel.parquet')
dfo = pd.read_parquet('data/Emanuel9.12-2/recordings/thumbFlEx/experiments/1/aligned_angles.parquet')

col = "('Right', 'thumbFlex_Pos')"

# Calculate split index (for 4/5 train, 1/5 test)
n = len(dfo)
split_idx = int(n / 5 * 4)

# Get predictions and ground truth for test set
pred_test = df[col].iloc[:].reset_index(drop=True)
gt_test = dfo[col].iloc[split_idx:].reset_index(drop=True)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(gt_test, label='Ground Truth (Thumb)')
plt.plot(pred_test, label='Predicted (Thumb)')
plt.xlabel('Test Sample index')
plt.ylabel('Thumb Position (deg)')
plt.title('Predicted vs. Ground Truth Thumb Position (Test Set)')
plt.legend()
plt.tight_layout()
plt.show()