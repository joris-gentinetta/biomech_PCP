import os
import numpy as np
import yaml
import argparse
import pandas as pd
from scipy import signal
from helpers.EMGClass import EMG

SKIP_MOVEMENTS = {"calibration", "Calibration", "calib", "Calib"}

def get_used_channels_from_snr(scaling_yaml_path, snr_threshold=5.0):
    """Returns indices of channels with SNR above threshold."""
    with open(scaling_yaml_path, 'r') as file:
        scalers = yaml.safe_load(file)
    maxVals = np.array(scalers['maxVals'])
    noiseLevel = np.array(scalers['noiseLevels'])
    print("maxVals:", maxVals)
    print("noiseLevel:", noiseLevel)
    snr = (maxVals - noiseLevel) / (noiseLevel + 1e-10)
    used_channels = [i for i, val in enumerate(snr) if val > snr_threshold]
    print(f"Calculated SNR per channel: {snr}")
    print(f"Selected used channels (SNR > {snr_threshold}): {used_channels}")
    return used_channels

def is_interaction_experiment(movement_name):
    """Check if this is a force interaction experiment."""
    return "_interaction" in movement_name.lower()

def update_yaml_configs(person_id, hand_side, used_channels, out_root='data'):
    config_dir = os.path.join(out_root, person_id, 'configs')
    recordings_dir = os.path.join(out_root, person_id, 'recordings')
    hand_side_cap = hand_side.capitalize()

    # Get all movements
    movement_names = sorted([
        d for d in os.listdir(recordings_dir)
        if os.path.isdir(os.path.join(recordings_dir, d)) and d not in SKIP_MOVEMENTS
    ])

    # Split movements by type
    free_space_movements = [m for m in movement_names if not is_interaction_experiment(m)]
    interaction_movements = [m for m in movement_names if is_interaction_experiment(m)]

    print(f"Free-space movements: {free_space_movements}")
    print(f"Interaction movements: {interaction_movements}")

    # Build feature and target lists
    emg_features = [[f"emg", str(ch)] for ch in used_channels]
    force_features = [[hand_side_cap, f"{finger}_Force"] for finger in ["index", "middle", "ring", "pinky", "thumb"]]
    position_targets = [[hand_side_cap, f"{joint}_Pos"] for joint in ["index", "middle", "ring", "pinky", "thumbFlex", "thumbRot"]]
    
    test_recordings = [m for m in ['indexFlEx', 'mrpFlEx', 'fingersFlEx'] if m in free_space_movements]

    # Update free-space config (modular_fs.yaml)
    if free_space_movements:
        fs_config_path = os.path.join(config_dir, 'modular_fs.yaml')
        if os.path.exists(fs_config_path):
            update_config_sections(
                fs_config_path,
                features=emg_features,  # EMG only
                targets=position_targets,
                recordings=free_space_movements,
                test_recordings=test_recordings
            )
        else:
            print(f"Warning: {fs_config_path} not found")

    # Update interaction config (modular_inter.yaml)
    if interaction_movements:
        inter_config_path = os.path.join(config_dir, 'modular_inter.yaml')
        if os.path.exists(inter_config_path):
            update_config_sections(
                inter_config_path,
                features=emg_features + force_features,  # EMG + Force
                targets=position_targets,
                recordings=interaction_movements,
                test_recordings=interaction_movements[:3]
            )
        else:
            print(f"Warning: {inter_config_path} not found")

def update_config_sections(config_path, features, targets, recordings, test_recordings):
    """Update specific sections of an existing config file"""
    with open(config_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Update features section
        if line.strip().startswith("features:"):
            new_lines.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "value:":
                new_lines.append(lines[i])
                i += 1
            new_lines.append("    value:\n")
            for feature in features:
                new_lines.append(f"    - {feature}\n")
            # Skip existing feature lines
            while i < len(lines) and (lines[i].strip().startswith("- [") or lines[i].strip() == "value:" or lines[i].strip() == ""):
                i += 1
            continue
            
        # Update targets section
        elif line.strip().startswith("targets:"):
            new_lines.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "value:":
                new_lines.append(lines[i])
                i += 1
            new_lines.append("    value:\n")
            for target in targets:
                new_lines.append(f"    - {target}\n")
            # Skip existing target lines
            while i < len(lines) and (lines[i].strip().startswith("- [") or lines[i].strip() == "value:" or lines[i].strip() == ""):
                i += 1
            continue
            
        # Update recordings section
        elif line.strip().startswith("recordings:") and "parameters:" in "".join(lines[max(0, i-2):i+1]):
            new_lines.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "value:":
                new_lines.append(lines[i])
                i += 1
            new_lines.append("    value:\n")
            for recording in recordings:
                new_lines.append(f"    - {recording}\n")
            # Skip existing recording lines
            while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip() == "value:" or lines[i].strip() == ""):
                i += 1
            continue
            
        # Update test_recordings section
        elif line.strip().startswith("test_recordings:"):
            new_lines.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "value:":
                new_lines.append(lines[i])
                i += 1
            new_lines.append("    value:\n")
            for recording in test_recordings:
                new_lines.append(f"    - {recording}\n")
            # Skip existing test recording lines
            while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip() == "value:" or lines[i].strip() == ""):
                i += 1
            continue
            
        else:
            new_lines.append(line)
            i += 1

    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    print(f" Updated {config_path}")

def extract_finger_forces_by_indices(angles_data):
    """
    Extract and sum force sensor data for each finger using known column indices.
    Based on the header: force sensors start at column 48 and follow pattern:
    index0_Force through index5_Force (cols 48-53)
    middle0_Force through middle5_Force (cols 54-59) 
    ring0_Force through ring5_Force (cols 60-65)
    pinky0_Force through pinky5_Force (cols 66-71)
    thumb0_Force through thumb5_Force (cols 72-77)
    
    Args:
        angles_data: numpy array of shape (n_samples, n_features)
    
    Returns:
        force_data: numpy array of shape (n_samples, 5) for [index, middle, ring, pinky, thumb]
        finger_names: list of finger names
    """
    finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb']
    force_data = np.zeros((angles_data.shape[0], len(finger_names)))
    
    # Define the starting column index for each finger's force sensors
    force_start_indices = {
        'index': 48,   # index0_Force through index5_Force
        'middle': 54,  # middle0_Force through middle5_Force  
        'ring': 60,    # ring0_Force through ring5_Force
        'pinky': 66,   # pinky0_Force through pinky5_Force
        'thumb': 72    # thumb0_Force through thumb5_Force
    }
    
    print(f"Extracting force data using known column indices")
    print(f"Angles data shape: {angles_data.shape}")
    
    for finger_idx, finger in enumerate(finger_names):
        start_col = force_start_indices[finger]
        end_col = start_col + 6  # 6 sensors per finger
        
        if end_col <= angles_data.shape[1]:
            # Sum the 6 force sensors for this finger
            finger_force_sum = np.sum(angles_data[:, start_col:end_col], axis=1)
            force_data[:, finger_idx] = finger_force_sum
            
            non_zero_count = np.sum(finger_force_sum > 0.001)
            print(f"  {finger} (cols {start_col}-{end_col-1}): mean={np.mean(finger_force_sum):.3f}, max={np.max(finger_force_sum):.3f}, non-zero={non_zero_count}/{len(finger_force_sum)}")
        else:
            print(f"  {finger}: columns {start_col}-{end_col-1} exceed data width ({angles_data.shape[1]})")
    
    return force_data, finger_names

def extract_finger_forces(angles_data, headers):
    """
    Extract and sum force sensor data for each finger from the angles array.
    Try header-based extraction first, fall back to index-based if needed.
    
    Args:
        angles_data: numpy array of shape (n_samples, n_features)
        headers: list of column names from angles_header.txt
    
    Returns:
        force_data: numpy array of shape (n_samples, 5) for [index, middle, ring, pinky, thumb]
        finger_names: list of finger names
    """
    # First try index-based extraction (more reliable)
    if angles_data.shape[1] >= 78:  # Need at least 78 columns for force data
        print("Using index-based force extraction (more reliable)")
        return extract_finger_forces_by_indices(angles_data)
    
    # Fall back to header-based extraction
    print("Using header-based force extraction")
    finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb']
    force_data = np.zeros((angles_data.shape[0], len(finger_names)))
    
    print(f"Extracting force data from angles array with shape {angles_data.shape}")
    print(f"Headers available: {len(headers)} columns")
    
    for finger_idx, finger in enumerate(finger_names):
        finger_force_sum = np.zeros(angles_data.shape[0])
        sensors_found = 0
        
        # Sum all 6 force sensors for this finger
        for sensor_idx in range(6):
            force_sensor_name = f"{finger}{sensor_idx}_Force"
            try:
                col_idx = headers.index(force_sensor_name)
                finger_force_sum += angles_data[:, col_idx]
                sensors_found += 1
            except ValueError:
                print(f"Warning: Force sensor {force_sensor_name} not found in headers")
        
        if sensors_found > 0:
            force_data[:, finger_idx] = finger_force_sum
            print(f"  {finger}: found {sensors_found}/6 sensors, sum range [{np.min(finger_force_sum):.3f}, {np.max(finger_force_sum):.3f}]")
        else:
            print(f"  {finger}: no force sensors found!")
    
    print(f"Extracted force data for fingers: {finger_names}")
    print(f"Force data shape: {force_data.shape}")
    
    return force_data, finger_names


def process_force_realtime_style(force_data, sampling_freq, static_offsets=None):
    """
    Process force data to match the real-time filtering pipeline in s5_inference.py
    BUT using zero-phase filtering for superior offline processing.
    
    This matches the real-time filter characteristics (Butterworth, 3Hz, 2nd order)
    while providing zero-lag filtering for training data.
    
    Args:
        force_data: numpy array of shape (n_samples, 5) for per-finger forces
        sampling_freq: sampling frequency in Hz
        static_offsets: numpy array of shape (5,) with static baseline offsets
                       If None, uses mean of first 10% of data as baseline
    
    Returns:
        filtered_force: processed force data with same characteristics as real-time
                       but with zero-phase filtering for better training data
    """
    from scipy import signal
    
    # 1. Static baseline correction (like zeroJoints calibration in real-time)
    if static_offsets is None:
        # Estimate static baseline from first 10% of data (assuming start is baseline)
        baseline_samples = int(0.1 * len(force_data))
        static_offsets = np.mean(force_data[:baseline_samples, :], axis=0)
        print(f"Estimated static offsets: {static_offsets}")
    
    static_corrected = np.maximum(force_data - static_offsets[None, :], 0)
    
    # 2. Butterworth filter - SAME CHARACTERISTICS as real-time but zero-phase
    # Real-time uses: 3Hz cutoff, 2nd order Butterworth
    # We use: 3Hz cutoff, 2nd order Butterworth + filtfilt (zero-phase)
    nyquist = 0.5 * sampling_freq
    normal_cutoff = min(3.0 / nyquist, 0.99)
    b, a = signal.butter(2, normal_cutoff, btype='low')
    
    # Apply ZERO-PHASE filtering to each finger (this is the key advantage for training)
    hf_filtered = np.zeros_like(static_corrected)
    for finger_idx in range(static_corrected.shape[1]):
        hf_filtered[:, finger_idx] = signal.filtfilt(b, a, static_corrected[:, finger_idx])
    
    # 3. Since all your data is "during contact", skip adaptive baseline correction
    # Just ensure non-negative values (matching real-time clipping)
    final_filtered = np.maximum(hf_filtered, 0)
    
    print(f"Applied zero-phase Butterworth filter (3Hz, 2nd order) - matches real-time characteristics")
    return final_filtered

def extract_and_filter_force_data_realtime_style(angles_data, headers, sampling_freq):
    """
    Extract force data and apply real-time style filtering for training data.
    """
    # Extract per-finger force sums (same as before)
    force_data, finger_names = extract_finger_forces(angles_data, headers)
    
    # Apply real-time style filtering instead of the Bessel filter
    filtered_force = process_force_realtime_style(force_data, sampling_freq)
    
    # Normalize (same as before)
    normalized_force = normalize_force_data(filtered_force, finger_names)
    
    print(f"Applied real-time style filtering to force data")
    print(f"Filtered force data summary:")
    for i, finger in enumerate(finger_names):
        print(f"  {finger}: mean={np.mean(filtered_force[:, i]):.3f}, max={np.max(filtered_force[:, i]):.3f}")
    
    return normalized_force, finger_names

def process_force_zero_phase(force_data, sampling_freq, cutoff_freq=2.0):
    """
    Smooth force data using zero-phase low-pass filtering optimized for force sensors.
    
    Args:
        force_data: numpy array of shape (n_samples, n_fingers)
        sampling_freq: sampling frequency in Hz
        cutoff_freq: low-pass filter cutoff frequency in Hz (default: 2.0 Hz for force)
    
    Returns:
        filtered_force: smoothed force data with same shape as input
    """
    
    # Create low-pass filter for force smoothing
    # Using a 4th order Bessel filter with lower cutoff for force (more aggressive smoothing)
    lowpass_sos = signal.bessel(N=4, Wn=cutoff_freq, btype='lowpass', 
                               output='sos', fs=sampling_freq, analog=False)
    
    # Apply zero-phase filtering to each force channel
    filtered_force = np.copy(force_data)
    n_fingers = force_data.shape[1] if force_data.ndim == 2 else 1
    
    if force_data.ndim == 1:
        # Single force channel
        filtered_force = signal.sosfiltfilt(lowpass_sos, filtered_force)
    else:
        # Multiple force channels
        for finger_idx in range(n_fingers):
            filtered_force[:, finger_idx] = signal.sosfiltfilt(lowpass_sos, 
                                                             filtered_force[:, finger_idx])
    
    # Ensure non-negative force values
    filtered_force = np.clip(filtered_force, 0, None)
    
    print(f"Applied zero-phase low-pass filter ({cutoff_freq} Hz) to force data")
    return filtered_force

def normalize_force_data(force_data, finger_names, max_forces_per_finger=None):
    """
    Normalize force data to [0, 1] range based on maximum achievable force per finger
    
    Args:
        force_data: numpy array of shape (n_samples, 5) 
        finger_names: list of finger names ['index', 'middle', 'ring', 'pinky', 'thumb']
        max_forces_per_finger: dict of max forces per finger (in Newtons)
                              If None, uses reasonable defaults
    
    Returns:
        normalized_force: force data normalized to [0, 1]
    """
    if max_forces_per_finger is None:
        # Default maximum forces based on prosthetic hand capabilities
        max_forces_per_finger = {
            'index': 12.0,   
            'middle': 12.0,    
            'ring': 12.0,    
            'pinky': 12.0,   
            'thumb': 12.0    
        }
    
    normalized_force = np.copy(force_data)
    
    for finger_idx, finger_name in enumerate(finger_names):
        max_force = max_forces_per_finger[finger_name]
        normalized_force[:, finger_idx] = np.clip(force_data[:, finger_idx] / max_force, 0, 1)
        
        print(f"  {finger_name}: max_force={max_force}N, "
              f"raw_range=[{np.min(force_data[:, finger_idx]):.2f}, {np.max(force_data[:, finger_idx]):.2f}], "
              f"normalized_range=[{np.min(normalized_force[:, finger_idx]):.3f}, {np.max(normalized_force[:, finger_idx]):.3f}]")
    
    return normalized_force

def scale_emg_timestamps_to_match_angles(emg_timestamps, angle_timestamps):
    """
    Scale EMG timestamps to match the duration of angle timestamps.
    This corrects for clock drift between EMG and angle recording systems.
    
    Args:
        emg_timestamps: Array of EMG timestamps
        angle_timestamps: Array of angle timestamps (reference duration)
    
    Returns:
        scaled_emg_timestamps: EMG timestamps scaled to match angle duration
    """
    emg_duration = emg_timestamps[-1] - emg_timestamps[0]
    angle_duration = angle_timestamps[-1] - angle_timestamps[0]
    
    print(f"Original EMG duration: {emg_duration:.3f}s")
    print(f"Angle duration: {angle_duration:.3f}s")
    print(f"Duration difference: {emg_duration - angle_duration:.3f}s")
    
    # Calculate scaling factor
    scale_factor = angle_duration / emg_duration
    print(f"Scaling factor: {scale_factor:.6f}")
    
    # Scale the EMG timestamps
    # First normalize to start at 0, then scale, then offset back
    emg_start_time = emg_timestamps[0]
    scaled_emg_timestamps = (emg_timestamps - emg_start_time) * scale_factor + emg_start_time
    
    # Verify the scaling
    scaled_duration = scaled_emg_timestamps[-1] - scaled_emg_timestamps[0]
    print(f"Scaled EMG duration: {scaled_duration:.3f}s")
    print(f"Final duration difference: {scaled_duration - angle_duration:.6f}s")
    
    return scaled_emg_timestamps

def save_angles_as_parquet(data_dir, angles_array, timestamps=None, hand_side='left', force_data=None, finger_names=None):
    header_path = os.path.join(data_dir, 'angles_header.txt')
    if not os.path.isfile(header_path):
        print(f"No angles_header.txt found in {data_dir}. Skipping parquet export.")
        return

    # read your raw headers
    with open(header_path, 'r') as f:
        headers = [h.strip() for h in f.read().split(',')]

    # define exactly the 6 angle names you care about:
    SIDE_TUPLE_COLS = [
        'index_Pos',
        'middle_Pos',
        'ring_Pos',
        'pinky_Pos',
        'thumbFlex_Pos',
        'thumbRot_Pos'
    ]

    # locate their column indices in the raw angles_array
    col_indices = [headers.index(h) for h in SIDE_TUPLE_COLS]

    # build a new DataFrame with only timestamp + those six columns
    hand_side_cap = hand_side.capitalize()
    # start with timestamp
    df = pd.DataFrame({'timestamp': timestamps}) if timestamps is not None else pd.DataFrame()

    # add only the six angle columns, as MultiIndex
    for h, idx in zip(SIDE_TUPLE_COLS, col_indices):
        df[(hand_side_cap, h)] = angles_array[:, idx]

    # Add force data if available (for interaction experiments)
    if force_data is not None and finger_names is not None:
        print(f"Adding force data to parquet file...")
        for finger_idx, finger_name in enumerate(finger_names):
            force_col_name = f"{finger_name}_Force"
            df[(hand_side_cap, force_col_name)] = force_data[:, finger_idx]
        print(f"Added force columns: {[f'{hand_side_cap}_{fn}_Force' for fn in finger_names]}")

    # finally, write parquet
    parquet_path = os.path.join(data_dir, 'aligned_angles.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {parquet_path} with shape {df.shape}")
    print("Columns:", df.columns.tolist())

def warp_emg_timestamps(emg_ts, ang_ts):
    # Fit EMG→angle time mapping: ang_ts ≈ slope * emg_ts + intercept
    slope, intercept = np.polyfit(emg_ts, ang_ts, 1)
    emg_ts_warped = emg_ts * slope + intercept
    return emg_ts_warped

def end_align_data(emg_data, emg_timestamps, angles_data, angles_timestamps):
    """
    End-align EMG to angles by shifting EMG timestamps so both datasets end at the same time.
    Returns the aligned and cropped data.
    """
    # Calculate shift needed to align end times
    shift = emg_timestamps[-1] - angles_timestamps[-1]
    emg_timestamps_shifted = emg_timestamps - shift
    
    # Mask for EMG samples that overlap with angles timespan
    mask = (emg_timestamps_shifted >= angles_timestamps[0]) & (emg_timestamps_shifted <= angles_timestamps[-1])
    
    # Apply mask to get overlapping data
    emg_aligned = emg_data[:, mask] if emg_data.ndim == 2 else emg_data[mask]
    emg_timestamps_aligned = emg_timestamps_shifted[mask]

    # emg_ts_warped = warp_emg_timestamps(emg_timestamps_aligned, angles_timestamps)
    
    print(f"End-alignment: shifted EMG by {shift:.6f}s, kept {np.sum(mask)}/{len(mask)} EMG samples")
    
    return emg_aligned, emg_timestamps_aligned, angles_data, angles_timestamps

def process_emg_zero_phase(emg_data, sampling_freq, maxVals, noiseLevel):
    """
    Process EMG data with zero-phase filtering (no delay) for training data generation.
    Replicates the EMGClass filtering pipeline but uses filtfilt instead of causal filters.
    """
    from scipy import signal
    import numpy as np
    
    # Create the same filters as in EMGClass but get their coefficients for filtfilt
    numElectrodes = emg_data.shape[0]
    
    # 1. Power line filter (bandstop 58-62 Hz)
    powerline_sos = signal.bessel(N=8, Wn=[58, 62], btype='bandstop', output='sos', fs=sampling_freq, analog=False)
    
    # 2. High pass filter (20 Hz) - removes motion artifacts and drift
    highpass_sos = signal.bessel(N=4, Wn=20, btype='highpass', output='sos', fs=sampling_freq, analog=False)
    
    # 3. Low pass filter (3 Hz) - smooths the envelope
    lowpass_sos = signal.bessel(N=4, Wn=3, btype='lowpass', output='sos', fs=sampling_freq, analog=False)
    
    # Apply zero-phase filtering
    filtered_emg = np.copy(emg_data)
    
    # Apply bandstop filter (power line noise removal)
    for ch in range(numElectrodes):
        filtered_emg[ch, :] = signal.sosfiltfilt(powerline_sos, filtered_emg[ch, :])
    
    # Apply high pass filter (motion artifact removal)
    for ch in range(numElectrodes):
        filtered_emg[ch, :] = signal.sosfiltfilt(highpass_sos, filtered_emg[ch, :])
    
    # Take absolute value and clip noise (same as EMGClass)
    filtered_emg = np.abs(filtered_emg)
    filtered_emg = np.clip(filtered_emg - noiseLevel[:, None], 0, None)
    
    # Apply low pass filter for envelope smoothing
    for ch in range(numElectrodes):
        filtered_emg[ch, :] = signal.sosfiltfilt(lowpass_sos, filtered_emg[ch, :])
    
    # Clip to ensure non-negative values
    filtered_emg = np.clip(filtered_emg, 0, None)
    
    # Normalize (same as EMGClass)
    normalized_emg = filtered_emg / maxVals[:, None]
    normalized_emg = np.clip(normalized_emg, 0, 1)
    
    return normalized_emg

def process_angles_zero_phase(angles_data, sampling_freq, cutoff_freq=5.0):
    """
    Smooth angle data using zero-phase low-pass filtering to remove high-frequency noise.
    
    Args:
        angles_data: numpy array of shape (n_samples, n_angles)
        sampling_freq: sampling frequency in Hz
        cutoff_freq: low-pass filter cutoff frequency in Hz (default: 5.0 Hz)
    
    Returns:
        filtered_angles: smoothed angle data with same shape as input
    """
    
    # Create low-pass filter for smoothing
    # Using a 4th order Bessel filter for smooth response
    lowpass_sos = signal.bessel(N=4, Wn=cutoff_freq, btype='lowpass', 
                               output='sos', fs=sampling_freq, analog=False)
    
    # Apply zero-phase filtering to each angle channel
    filtered_angles = np.copy(angles_data)
    n_angles = angles_data.shape[1] if angles_data.ndim == 2 else 1
    
    if angles_data.ndim == 1:
        # Single angle channel
        filtered_angles = signal.sosfiltfilt(lowpass_sos, filtered_angles)
    else:
        # Multiple angle channels
        for angle_idx in range(n_angles):
            filtered_angles[:, angle_idx] = signal.sosfiltfilt(lowpass_sos, 
                                                             filtered_angles[:, angle_idx])
    
    print(f"Applied zero-phase low-pass filter ({cutoff_freq} Hz) to angle data")
    return filtered_angles

def process_emg_and_angles(
        data_dir, 
        person_id,
        out_root,
        used_channels,
        hand_side='left',
        emg_file='raw_emg.npy', 
        emg_timestamps_file='raw_timestamps.npy',
        angles_file='angles.npy',
        angles_timestamps_file='angle_timestamps.npy',
        angle_filter_cutoff=5.0,
        force_filter_cutoff=2.0,
        use_timestamp_scaling=True,  # New parameter to enable/disable scaling
        angle_shift_seconds=0.0,
    ):
    # Load the data
    emg = np.load(os.path.join(data_dir, emg_file)).T
    emg_timestamps = np.load(os.path.join(data_dir, emg_timestamps_file))
    angles = np.load(os.path.join(data_dir, angles_file))
    angles_timestamps = np.load(os.path.join(data_dir, angles_timestamps_file))
    
    print(f"Loaded data from {data_dir}")

    # Load experiment config and calculate contact detection offset
    config_path = os.path.join(data_dir, 'experiment_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        duration = config.get('duration', 0)
        total_duration = config.get('total_duration', 0)
        contact_offset = total_duration - duration
        print(f"Trimming pre-contact data: Removing first {contact_offset:.3f}s")

        # Trim EMG to only include data after contact detection
        emg_mask = emg_timestamps >= (emg_timestamps[0] + contact_offset)
        emg = emg[:, emg_mask]
        emg_timestamps = emg_timestamps[emg_mask]

        # Trim angles to only include data after contact detection
        angles_mask = angles_timestamps >= (angles_timestamps[0] + contact_offset)
        angles = angles[angles_mask, :]
        angles_timestamps = angles_timestamps[angles_mask]
    else:
        print(f"Warning: experiment_config.yaml not found in {data_dir}. No trimming applied.")

    print(f"EMG shape: {emg.shape}, timestamps: {len(emg_timestamps)}")
    print(f"Angles shape: {angles.shape}, timestamps: {len(angles_timestamps)}")

    # Check if this is an interaction experiment (has force data)
    movement_name = os.path.basename(os.path.dirname(os.path.dirname(data_dir)))
    is_interaction = is_interaction_experiment(movement_name)
    print(f"Is interaction experiment: {is_interaction}")

    # Load headers to identify force columns
    header_path = os.path.join(data_dir, 'angles_header.txt')
    headers = []
    if os.path.exists(header_path):
        with open(header_path, 'r') as f:
            header_content = f.read().strip()
            headers = [h.strip() for h in header_content.split(',')]
        print(f"Loaded {len(headers)} headers from {header_path}")
        
        # Debug: Show force-related headers
        force_headers = [h for h in headers if 'Force' in h]
        print(f"Found {len(force_headers)} force-related headers")
        if len(force_headers) > 0:
            print(f"First few force headers: {force_headers[:10]}")
    else:
        print(f"Warning: No angles_header.txt found at {header_path}")

    # Extract force data if this is an interaction experiment
    force_data = None
    finger_names = None
    if is_interaction and headers and len(headers) == angles.shape[1]:
        print("\n=== Processing Force Data ===")
        try:
            force_data, finger_names = extract_finger_forces(angles, headers)
            print(f"Force data summary:")
            for i, finger in enumerate(finger_names):
                force_values = force_data[:, i]
                non_zero_count = np.sum(force_values > 0.001)  # Count values > 1mN
                print(f"  {finger}: mean={np.mean(force_values):.3f}, max={np.max(force_values):.3f}, non-zero samples={non_zero_count}/{len(force_values)}")
        except Exception as e:
            print(f"Error extracting force data: {e}")
            force_data = None
            finger_names = None
    elif is_interaction:
        if not headers:
            print("Cannot process force data: no headers loaded")
        elif len(headers) != angles.shape[1]:
            print(f"Cannot process force data: header count ({len(headers)}) doesn't match data columns ({angles.shape[1]})")
        else:
            print("Force data processing skipped for unknown reason")

    # TIMESTAMP SCALING - Apply before end-alignment
    if use_timestamp_scaling:
        print("\n=== Applying Timestamp Scaling ===")
        # Save original timestamps for reference
        original_emg_timestamps = emg_timestamps.copy()
        
        # Scale EMG timestamps to match angle duration
        emg_timestamps = scale_emg_timestamps_to_match_angles(emg_timestamps, angles_timestamps)
        
        # Save both original and scaled timestamps for debugging
        np.save(os.path.join(data_dir, 'emg_timestamps_original.npy'), original_emg_timestamps)
        np.save(os.path.join(data_dir, 'emg_timestamps_scaled.npy'), emg_timestamps)
        
        print("✓ Timestamp scaling applied and saved")
    else:
        print("Timestamp scaling disabled - using original timestamps")

    # END-ALIGN the data (now using scaled timestamps if enabled)
    print("\n=== Applying End-Alignment ===")
    emg, emg_timestamps, angles, angles_timestamps = end_align_data(
        emg, emg_timestamps, angles, angles_timestamps
    )

    # Load calibration data
    scaling_yaml_path = os.path.join(
        out_root, person_id, "recordings", "Calibration", "experiments", "1", "scaling.yaml"
    )
    with open(scaling_yaml_path, 'r') as file:
        scalers = yaml.safe_load(file)
    maxVals = np.array(scalers['maxVals'])
    noiseLevel = np.array(scalers['noiseLevels'])

    # Select only the used channels
    emg = emg[used_channels, :]
    maxVals = maxVals[used_channels]
    noiseLevel = noiseLevel[used_channels]

    # Calculate sampling frequencies
    sf = (len(emg_timestamps) - 1) / (emg_timestamps[-1] - emg_timestamps[0])
    print(f"EMG sampling frequency: {sf:.1f} Hz")
    
    # Use zero-phase filtering for EMG
    filtered_emg = process_emg_zero_phase(emg, sf, maxVals, noiseLevel)

    # Process angles
    angle_sf = (len(angles_timestamps) - 1) / (angles_timestamps[-1] - angles_timestamps[0])
    print(f"Angle sampling frequency: {angle_sf:.1f} Hz")
    filtered_angles = process_angles_zero_phase(angles, angle_sf, angle_filter_cutoff)

    # Process force data if available
    filtered_force = None
    if force_data is not None:
        print(f"\n=== Filtering Force Data ===")
        # filtered_force = process_force_zero_phase(force_data, angle_sf, force_filter_cutoff)
        filtered_force = process_force_realtime_style(force_data, angle_sf)
        
        # Normalize force data
        print(f"\n=== Normalizing Force Data ===")
        filtered_force = normalize_force_data(filtered_force, finger_names)
        
        print(f"Normalized force data summary:")
        for i, finger in enumerate(finger_names):
            print(f"  {finger}: mean={np.mean(filtered_force[:, i]):.3f}, max={np.max(filtered_force[:, i]):.3f}")

    # Continue with the rest of existing processing...
    # Use angle timestamps as reference (EMG interpolated onto angle time base)
    ref_t = angles_timestamps
    aligned_angles = filtered_angles
    aligned_emg = np.zeros((len(ref_t), filtered_emg.shape[0]))
    for ch in range(filtered_emg.shape[0]):
        aligned_emg[:, ch] = np.interp(ref_t, emg_timestamps, filtered_emg[ch, :])

    # Align force data if available
    aligned_force = None
    if filtered_force is not None:
        aligned_force = filtered_force  # Force data is already on angle timebase

    # Downsample to 60Hz using windowed integration (like EMGClass)
    current_sf = (len(ref_t) - 1) / (ref_t[-1] - ref_t[0])  # Current sampling frequency
    target_sf = 60.0  # Target 60Hz
    downsample_ratio = current_sf / target_sf # Hz
    downsample_step = int(round(downsample_ratio))
    
    emg_60Hz = aligned_emg[::downsample_step, :]
    angles_60Hz = aligned_angles[::downsample_step, :]
    downsampled_t = ref_t[::downsample_step]

    # Downsample force data if available
    force_60Hz = None
    if aligned_force is not None:
        force_60Hz = aligned_force[::downsample_step, :]

    downsampled_t = downsampled_t - downsampled_t[0]

    final_sf = (len(downsampled_t) - 1) / (downsampled_t[-1] - downsampled_t[0]) if len(downsampled_t) > 1 else 0
    print(f"Final sampling frequency: {final_sf:.1f} Hz")
    print(f"Final data length: {len(downsampled_t)} samples")

    if angle_shift_seconds > 0:
        print(f"\n=== Applying Time Shift: {angle_shift_seconds}s ===")
        shift_samples = int(angle_shift_seconds * final_sf)
        if shift_samples >= len(emg_60Hz):
            print(f"Warning: Shift ({angle_shift_seconds}s = {shift_samples} samples) is larger than data length ({len(emg_60Hz)})")
            shift_samples = len(emg_60Hz) - 1

        print(f"Shifting EMG by {shift_samples} samples ({shift_samples/final_sf:.3f}s)")

        # Shift EMG to the left (remove first N samples)
        emg_60Hz_shifted = emg_60Hz[shift_samples:, :]
        # Trim angles from the end to match the EMG length
        angles_60Hz_trimmed = angles_60Hz[:len(emg_60Hz_shifted), :]

        # Also trim force data if available
        if force_60Hz is not None:
            force_60Hz_trimmed = force_60Hz[:len(emg_60Hz_shifted), :]
        else:
            force_60Hz_trimmed = None

        # Create new timestamp array
        downsampled_t_shifted = downsampled_t[:len(emg_60Hz_shifted)]

        # Update the data arrays
        emg_60Hz = emg_60Hz_shifted
        angles_60Hz = angles_60Hz_trimmed
        force_60Hz = force_60Hz_trimmed
        downsampled_t = downsampled_t_shifted
        
        print(f"After shift - EMG shape: {emg_60Hz.shape}, Angles shape: {angles_60Hz.shape}")
        if force_60Hz is not None:
            print(f"After shift - Force shape: {force_60Hz.shape}")
        print(f"New data length: {len(downsampled_t)} samples ({downsampled_t[-1]:.2f}s)")

    # Drop the first few seconds because of synchronization
    drop_secs = 0.0
    start_idx = np.searchsorted(downsampled_t, drop_secs)
    if drop_secs > 0:
        start_idx = np.searchsorted(downsampled_t, drop_secs)
        emg_60Hz = emg_60Hz[start_idx:]
        angles_60Hz = angles_60Hz[start_idx:]
        if force_60Hz is not None:
            force_60Hz = force_60Hz[start_idx:]
        downsampled_t = downsampled_t[start_idx:] - drop_secs

    # Save results
    np.save(os.path.join(data_dir, 'aligned_filtered_emg.npy'), emg_60Hz)
    np.save(os.path.join(data_dir, 'aligned_angles.npy'), angles_60Hz)
    np.save(os.path.join(data_dir, 'aligned_timestamps.npy'), downsampled_t)
    
    # Save force data if available
    if force_60Hz is not None:
        np.save(os.path.join(data_dir, 'aligned_force.npy'), force_60Hz)
        print(f"Saved aligned_force.npy with shape {force_60Hz.shape}")

    # Save parquet file with force data if available
    save_angles_as_parquet(data_dir, angles_60Hz, downsampled_t, hand_side=hand_side, 
                          force_data=force_60Hz, finger_names=finger_names)
    
    print(f"{os.path.basename(data_dir)}: Downsampled to 60Hz ({len(downsampled_t)} datapoints)")
    
    # Print final alignment quality
    print(f"\n=== Final Alignment Quality ===")
    print(f"Final EMG samples: {len(emg_60Hz)}")
    print(f"Final angle samples: {len(angles_60Hz)}")
    if force_60Hz is not None:
        print(f"Final force samples: {len(force_60Hz)}")
    print(f"Final duration: {downsampled_t[-1]:.2f}s")
    if angle_shift_seconds > 0:
        print(f"Time shift applied: {angle_shift_seconds}s (EMG predicts angles {angle_shift_seconds}s into the future)")
    

def process_all_experiments(person_id, out_root, movement=None, snr_threshold=3.0, hand_side='left', use_timestamp_scaling=True, angle_shift_seconds=0.0):
    recordings_dir = os.path.join(out_root, person_id, "recordings")
    if not os.path.exists(recordings_dir):
        print(f"No such directory: {recordings_dir}")
        return

    scaling_yaml_path = os.path.join(
        out_root, person_id, "recordings", "Calibration", "experiments", "1", "scaling.yaml"
    )
    used_channels = get_used_channels_from_snr(scaling_yaml_path, snr_threshold=snr_threshold)

    update_yaml_configs(person_id, hand_side, used_channels, out_root)

    movements_to_process = []
    if movement:
        movements_to_process = [movement]
    else:
        for m in sorted(os.listdir(recordings_dir)):
            if m in SKIP_MOVEMENTS:
                continue
            m_dir = os.path.join(recordings_dir, m)
            if os.path.isdir(m_dir):
                movements_to_process.append(m)

    for movement in movements_to_process:
        experiments_dir = os.path.join(recordings_dir, movement, "experiments")
        if not os.path.exists(experiments_dir):
            print(f"No experiments found for movement: {movement}")
            continue
        
        # Check if this is an interaction experiment
        is_interaction = is_interaction_experiment(movement)
        if is_interaction:
            print(f"\n=== Processing INTERACTION experiment: {movement} ===")
        else:
            print(f"\n=== Processing standard experiment: {movement} ===")
            
        for experiment in sorted(os.listdir(experiments_dir)):
            exp_dir = os.path.join(experiments_dir, experiment)
            if not os.path.isdir(exp_dir):
                continue
            required_files = ['raw_emg.npy', 'raw_timestamps.npy', 'angles.npy', 'angle_timestamps.npy']
            if not all(os.path.exists(os.path.join(exp_dir, f)) for f in required_files):
                print(f"Skipping {exp_dir} (missing files)")
                continue
            print(f"Processing {exp_dir} ...")
            try:
                process_emg_and_angles(
                    exp_dir, 
                    person_id, 
                    out_root, 
                    used_channels, 
                    hand_side=hand_side,
                    use_timestamp_scaling=use_timestamp_scaling,  # Pass the parameter
                    angle_shift_seconds=angle_shift_seconds
                )
            except Exception as e:
                print(f"Failed to process {exp_dir}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process all EMG/Angle experiments for a person/movement.')
    parser.add_argument('--person_id', required=True, help='Person ID (e.g. Emanuel_FirstTries)')
    parser.add_argument('--movement', required=False, help='Movement name (e.g. indexFlDigitsEx, optional)')
    parser.add_argument('--out_root', default='data', help='Root directory (default: data)')
    parser.add_argument('--snr_threshold', type=float, default=0.9, help='SNR threshold for channel selection')
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], default="left", help="Side of the prosthetic hand")
    parser.add_argument('--no_timestamp_scaling', action='store_true', help='Disable timestamp scaling (use original timestamps)')
    parser.add_argument('--angle_shift', type=float, default=0.5, help= 'Shift angles to the left by this many seconds')
    parser.add_argument('--force_filter_cutoff', type=float, default=2.0, help='Force filter cutoff frequency in Hz (default: 2.0)')
    args = parser.parse_args()

    use_timestamp_scaling = not args.no_timestamp_scaling  # Default is True unless --no_timestamp_scaling is specified
    
    process_all_experiments(
        args.person_id, 
        args.out_root, 
        args.movement, 
        args.snr_threshold, 
        args.hand_side,
        use_timestamp_scaling=use_timestamp_scaling,
        angle_shift_seconds=args.angle_shift 
    )