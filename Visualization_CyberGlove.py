# ==========================================
# Script: cyberglove_to_psyonic.py
# Purpose: Control the Psyonic Ability Hand
#          using CyberGlove-II angles from NinaPro
#          with optional thumb lock,
#          low-pass filtering,
#          and correct mapping to the hand's range.
# ==========================================

import scipy.io
import numpy as np
import time
import sys
import os
from scipy.signal import butter, filtfilt

# Add path for custom libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from psyonicHand import psyonicArm

# ------------------------------------------
# 1) Low-pass filter function
# ------------------------------------------
def lowpass_filter(data, cutoff=5, fs=100, order=4):
    """
    data: 2D array (samples, channels)
    cutoff: cutoff frequency in Hz
    fs: sampling rate in Hz
    order: filter order
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

# ------------------------------------------
# 2) Load angles + stimulus, clip up to stimulus == 2
# ------------------------------------------
mat_file_path = r"C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/pretraining/CyberGlove2/s_1_angles/s_1_angles/S1_E3_A1.mat"
data = scipy.io.loadmat(mat_file_path)

angles = data['angles']             # shape (samples, 22)
stimulus = data['stimulus'].flatten()

# We stop at the first time stimulus == 2
if 2 in stimulus:
    task_change_idx = np.where(stimulus == 2)[0][0]
else:
    task_change_idx = len(stimulus)

angles = angles[:task_change_idx]
stimulus = stimulus[:task_change_idx]

print(f"Clipped dataset to {angles.shape[0]} samples (up to first appearance of task 2).")

# ------------------------------------------
# 3) Choose the correct sensor indices
#    in the order: [Thumb Flex, Index, Middle, Ring, Pinky, Thumb Rot]
#
# Example from your mention: 
#   - Thumb flex = sensor 1 (0-based index?), 
#   - Index flex = sensor 6 => 5 in 0-based,
#   - Middle flex = sensor 9 => 8 in 0-based,
#   - Ring flex = sensor 13 => 12 in 0-based,
#   - Pinky flex = sensor 17 => 16 in 0-based,
#   - Thumb rotation = sensor 0 => (?), 
#
# Adjust these carefully!
# ------------------------------------------
# Suppose for example:
#   sensor  1 =>  0 in code => Thumb flex
#   sensor  6 =>  5 => Index MCP
#   sensor  9 =>  8 => Middle MCP
#   sensor 13 => 12 => Ring MCP
#   sensor 17 => 16 => Pinky MCP
#   sensor  0 =>  -1 => ??? for rotation
#
# Check your glove map carefully!
# 
# For demonstration, let's do this:
selected_indices = [
    1,   # (col 0) -> Thumb Flex
    5,   # (col 1) -> Index MCP
    8,   # (col 2) -> Middle MCP
    12,  # (col 3) -> Ring MCP
    16,  # (col 4) -> Pinky MCP
    0    # (col 5) -> Thumb Rotation
]

selected_angles = angles[:, selected_indices]

print("Selected Indices => Final columns order:")
print("0 -> Thumb Flex\n1 -> Index\n2 -> Middle\n3 -> Ring\n4 -> Pinky\n5 -> Thumb Rotation")

# ------------------------------------------
# 4) Apply a low-pass filter (5 Hz cut) to smooth angles
# ------------------------------------------
fs = 100   # assumed sampling freq
cutoff = 3 # adjust as needed
order = 4
selected_angles = lowpass_filter(selected_angles, cutoff=cutoff, fs=fs, order=order)

# ------------------------------------------
# 5) Analyze min/max for normalization 
#    (only from clipped & filtered data)
# ------------------------------------------
mins = np.min(selected_angles, axis=0)
maxs = np.max(selected_angles, axis=0)

print("\nMin/Max per final angle (after clipping & filtering):")
for col_i, (mn, mx) in enumerate(zip(mins, maxs)):
    print(f"  Col {col_i}: min={mn:.2f}, max={mx:.2f}")

# Now normalize each column
normalized = (selected_angles - mins) / (maxs - mins)
normalized = np.clip(normalized, 0, 1)

# ------------------------------------------
# 6) Map to Psyonic range
#    Here the function expects:
#        col 0 -> Thumb Flex
#        col 1 -> Index
#        col 2 -> Middle
#        col 3 -> Ring
#        col 4 -> Pinky
#        col 5 -> Thumb Rotation
# ------------------------------------------
def map_to_psyonic_range(norm_data, control_thumb=True):
    # create output (samples, 6 motors)
    out = np.zeros_like(norm_data)  

    # Thumb flex => row[:,0], range 0..120
    # Index => row[:,1], middle => row[:,2], ring => row[:,3], pinky => row[:,4], each 0..120
    # Thumb rot => row[:,5], range -120..0

    # copy thumb flex
    if control_thumb:
        out[:,0] = norm_data[:,0] * 120.0        # thumb flex
        out[:,5] = -120.0 + norm_data[:,5]*120.0 # thumb rotation
    else:
        # fix thumb to half open
        out[:,0] = 10.0   # flex
        out[:,5] = -60.0  # rotation

    # index, middle, ring, pinky
    out[:,1] = norm_data[:,1] * 119.0 + 1
    out[:,2] = norm_data[:,2] * 119.0 + 1
    out[:,3] = norm_data[:,3] * 119.0 + 1
    out[:,4] = norm_data[:,4] * 119.0 + 1

    return out

# Ask user if they want to control thumb
thumb_choice = input("\nControl thumb via CyberGlove? (y/n): ").strip().lower()
if thumb_choice == 'y':
    control_thumb = True
    print("Thumb is controlled by glove angles.")
else:
    control_thumb = False
    print("Thumb is locked to semi-open position.")

motor_commands = map_to_psyonic_range(normalized, control_thumb=control_thumb)

# ------------------------------------------
# 7) Connect to the actual Ability Hand
# ------------------------------------------
arm = psyonicArm(hand='left')
arm.initSensors()
arm.startComms()

# Move to neutral pose first
neutral_pose = np.array([5, 5, 5, 5, 5, -5])
arm.mainControlLoop(posDes=neutral_pose, period=0.08)
time.sleep(1)

try:
    print("\nSending smoothed + normalized CyberGlove-based movements ...")
    # Subsample to avoid flooding
    motor_commands_downsampled = motor_commands[::5]
    arm.mainControlLoop(posDes=motor_commands_downsampled, period=0.06)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    arm.close()
    print("Closed connection.")

# ==========================================
# Notes:
#  - We do a low-pass filter on the chosen angles.
#  - We clip data up to the first time stimulus == 2
#  - We reorder columns so col0=thumb flex, col1=index, etc.
#  - Then map to [0..120], [ -120..0 ] for thumb rotation
#  - Subsample frames for less dense motion
# ==========================================
