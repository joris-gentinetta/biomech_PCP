import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
data = scipy.io.loadmat(r"C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/pretraining/CyberGlove2/s_1_angles/s_1_angles/S1_E2_A1.mat")

# Check what's inside
print(data.keys())

# Extract angles and stimulus
angles = data['angles']
stimulus = data['stimulus'].flatten()  # make it a 1D array

# Relevant joints (indices)
selected_indices = [1, 2, 4, 7, 9, 13]
# selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
labels = ['Thumb Rotation', 'Thumb Flexion', 'Index MCP Flexion', 'Middle MCP Flexion', 'Ring MCP Flexion', 'Pinky MCP Flexion']

# Plot
plt.figure(figsize=(14, 7))


line_styles = ['-', '--', '-.', ':', '-', '--']
markers = ['o', '^', 's', 'x', 'd', '*']


# Plot only the selected joints
for idx, i in enumerate(selected_indices):
    plt.plot(angles[:, i], label=labels[idx])
    # plt.plot(angles[:, i])

# Find the points where the task changes
change_indices = np.where(np.diff(stimulus) != 0)[0] + 1  # +1 because diff reduces array size by 1

for idx in change_indices:
    plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.6)
    plt.text(idx, plt.ylim()[1]*0.95, f"Task {stimulus[idx]}", rotation=90, verticalalignment='top', fontsize=8)

# Plot settings
plt.xlabel('Time (samples)')
plt.ylabel('Angle (degrees)')
plt.title('CyberGlove-II Relevant Joint Angles with Task Changes')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

