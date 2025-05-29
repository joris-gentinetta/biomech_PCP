import scipy.io
import numpy as np
import time
import sys
from scipy.signal import savgol_filter
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from psyonicHand import psyonicArm

def load_and_preprocess(mat_path):
    data       = scipy.io.loadmat(mat_path)
    angles_raw = data['angles']  # (n_frames, n_channels)
    # normalize each channel into 0–multipliers range
    mins   = angles_raw.min(axis=0)
    maxs   = angles_raw.max(axis=0)
    ranges = maxs - mins + 1e-8
    multipliers = np.full(angles_raw.shape[1], 110.0)
    for i in [4,7,9,13,6,8,11,15]:
        multipliers[i] = 110.0
    for i in [16,17,18,19]:
        multipliers[i] = 80.0
    for i in [1]:
        multipliers[i] = 160.0
    angles_deg = (angles_raw - mins) / ranges * multipliers
    # smooth
    angles_smooth = savgol_filter(angles_deg, window_length=5, polyorder=2, axis=0)
    return angles_smooth

def extract_normalized(mat_path):
    """
    Loads, preprocesses, selects the 6 glove channels in the order
      [thumb_flex, index, middle, ring, pinky, thumb_rot]
    and returns an (n_samples, 6) array normalized 0–1.
    """
    angles = load_and_preprocess(mat_path)

    # channel indices in the CyberGlove
    thumb_idx  = [0,1,2,3]
    index_idx  = [4,6,16]
    middle_idx = [7,8,17]
    ring_idx   = [9,11,18]
    pinky_idx  = [13,15,19]

    # pick out in the exact order map_to_psyonic_range expects:
    #   col 0 = thumb flex,   col 1 = index, col 2 = middle,
    #   col 3 = ring,         col 4 = pinky, col 5 = thumb rotation
    sel_idxs = [
        index_idx[0],   # index MCP
        middle_idx[0],  # middle MCP
        ring_idx[0],    # ring MCP
        pinky_idx[0],   # pinky MCP
        thumb_idx[0],   # thumb flex
        thumb_idx[1]    # thumb rotation
    ]
    data6 = angles[:, sel_idxs]

    # normalize each column to [0,1]
    mins = data6.min(axis=0)
    maxs = data6.max(axis=0)
    normalized = (data6 - mins) / (maxs - mins + 1e-8)
    return normalized

def map_to_psyonic_range(norm_data, control_thumb=True):
    """
    norm_data: (n_samples,6) in [0,1]
    Returns (n_samples,6) in Ability Hand ROM:
      thumb flex  0..120
      index−pinky 0..120
      thumb rot  -120..0
    """
    out = np.zeros_like(norm_data)
    if control_thumb:
        out[:,4] = norm_data[:,0] * 120.0        # thumb flex
        out[:,5] = -120.0 + norm_data[:,5]*120.0 # thumb rot
    else:
        out[:,4] = 10.0   # fixed semi-open flex
        out[:,5] = -60.0  # fixed semi-open rot

    # index, middle, ring, pinky
    out[:,1] = norm_data[:,1] * 120.0
    out[:,2] = norm_data[:,2] * 120.0
    out[:,3] = norm_data[:,3] * 120.0
    out[:,4] = norm_data[:,4] * 120.0

    return out

if __name__ == '__main__':
    # 1) load & normalize
    mat_path   = r"C:/Users/Emanuel Wicki/Documents/MIT/biomech_PCP/pretraining/CyberGlove2/s_1_angles/s_1_angles/S64_E1_A1.mat"
    normalized = extract_normalized(mat_path)
    print(f"Extracted and normalized {normalized.shape[0]} frames of glove data.")

    # 2) ask thumb control
    ctrl = input("Control thumb via glove? (y/n): ").strip().lower() == 'y'
    motor_commands = map_to_psyonic_range(normalized, control_thumb=ctrl)

    # 3) connect to Ability Hand
    arm = psyonicArm(hand='left')
    arm.initSensors()
    arm.startComms()

    # 4) neutral pose
    neutral = np.array([5, 5, 5, 5, 5, -5])  # semi-open
    arm.mainControlLoop(posDes=neutral, period=0.08)
    time.sleep(1.0)

    # 5) stream commands (downsample every 5th to avoid flooding)
    try:
        print("Streaming to Ability Hand. Ctrl-C to stop.")
        ds = motor_commands[::70]
        arm.mainControlLoop(posDes=ds, period=0.06)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        arm.close()
        print("Connection closed.")
