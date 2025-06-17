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

def update_yaml_configs(person_id, hand_side, used_channels, out_root='data'):
    config_dir = os.path.join(out_root, person_id, 'configs')
    recordings_dir = os.path.join(out_root, person_id, 'recordings')
    hand_side_cap = hand_side.capitalize()

    SKIP_MOVEMENTS = {"calibration", "Calibration", "calib", "Calib"}

    # Movements for recordings:value
    movement_names = sorted([
        d for d in os.listdir(recordings_dir)
        if os.path.isdir(os.path.join(recordings_dir, d)) and d not in SKIP_MOVEMENTS
    ])

    # Fixed list for test_recordings:value
    test_recordings = ['indexFlEx', 'mrpFlEx', 'fingersFlEx']

    # Standard targets, switch side as needed
    TARGETS = [
        "index_Pos",
        "middle_Pos",
        "ring_Pos",
        "pinky_Pos",
        "thumbFlex_Pos",
        "thumbRot_Pos"
    ]
    # Formatting for targets
    targets_block = ""
    for t in TARGETS:
        targets_block += f"    - [{hand_side_cap}, {t}]\n"

    # Formatting for features (with channel comments)
    features_block = ""
    for idx, ch in enumerate(used_channels):
        features_block += f"    - [emg, '{ch}'] # {idx}\n"

    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    for yaml_file in yaml_files:
        yaml_path = os.path.join(config_dir, yaml_file)
        with open(yaml_path, 'r') as f:
            lines = f.readlines()

        # Find key sections
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # parameters:recordings:value
            if line.strip().startswith("recordings:") and "parameters:" in "".join(lines[max(0, i-2):i+1]):
                new_lines.append(line)
                i += 1
                while i < len(lines) and lines[i].strip() != "value:":
                    new_lines.append(lines[i])
                    i += 1
                new_lines.append("    value:\n")
                for m in movement_names:
                    new_lines.append(f"    - {m}\n")
                new_lines.append('\n')  # Blank line after recordings:value block
                while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip() == "value:" or lines[i].strip() == "" or lines[i].strip().startswith("#")):
                    i += 1
                continue
            # parameters:test_recordings:value
            if line.strip().startswith("test_recordings:"):
                new_lines.append(line)
                i += 1
                while i < len(lines) and lines[i].strip() != "value:":
                    new_lines.append(lines[i])
                    i += 1
                new_lines.append("    value:\n")
                for m in test_recordings:
                    new_lines.append(f"    - {m}\n")
                # new_lines.append('\n')  # Blank line after test_recordings:value block
                while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip() == "value:" or lines[i].strip() == "" or lines[i].strip().startswith("#")):
                    i += 1
                continue
            # parameters:targets:value
            if line.strip().startswith("targets:"):
                new_lines.append(line)
                i += 1
                while i < len(lines) and lines[i].strip() != "value:":
                    new_lines.append(lines[i])
                    i += 1
                new_lines.append("    value:\n")
                new_lines.extend(targets_block)
                # No blank line here (matches your formatting)
                while i < len(lines) and (lines[i].strip().startswith("- [") or lines[i].strip() == "value:" or lines[i].strip().startswith("#") or lines[i].strip() == ""):
                    i += 1
                continue
            # parameters:features:value
            if line.strip().startswith("features:"):
                new_lines.append('\n')  # Blank line BEFORE features block
                new_lines.append(line)
                i += 1
                while i < len(lines) and lines[i].strip() != "value:":
                    new_lines.append(lines[i])
                    i += 1
                new_lines.append("    value:\n")
                new_lines.extend(features_block)
                new_lines.append('\n')  # Blank line AFTER features:value block
                while i < len(lines) and (lines[i].strip().startswith("- [emg,") or lines[i].strip() == "value:" or lines[i].strip().startswith("#") or lines[i].strip() == ""):
                    i += 1
                continue
            # Default: keep old line
            new_lines.append(line)
            i += 1

        with open(yaml_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Updated {yaml_path}")

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


def save_angles_as_parquet(data_dir, angles_array, timestamps=None, hand_side='left'):
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
        use_timestamp_scaling=True,  # New parameter to enable/disable scaling
    ):
    # Load the data
    emg = np.load(os.path.join(data_dir, emg_file)).T
    emg_timestamps = np.load(os.path.join(data_dir, emg_timestamps_file))
    angles = np.load(os.path.join(data_dir, angles_file))
    angles_timestamps = np.load(os.path.join(data_dir, angles_timestamps_file))
    
    print(f"Loaded data from {data_dir}")
    print(f"EMG shape: {emg.shape}, timestamps: {len(emg_timestamps)}")
    print(f"Angles shape: {angles.shape}, timestamps: {len(angles_timestamps)}")

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

    # Continue with the rest of your existing processing...
    # Use angle timestamps as reference (EMG interpolated onto angle time base)
    ref_t = angles_timestamps
    aligned_angles = filtered_angles
    aligned_emg = np.zeros((len(ref_t), filtered_emg.shape[0]))
    for ch in range(filtered_emg.shape[0]):
        aligned_emg[:, ch] = np.interp(ref_t, emg_timestamps, filtered_emg[ch, :])

    # Downsample to 60Hz using windowed integration (like EMGClass)
    current_sf = (len(ref_t) - 1) / (ref_t[-1] - ref_t[0])  # Current sampling frequency
    target_sf = 60.0  # Target 60Hz
    downsample_ratio = current_sf / target_sf # Hz
    downsample_step = int(round(downsample_ratio))
    
    emg_60Hz = aligned_emg[::downsample_step, :]
    angles_60Hz = aligned_angles[::downsample_step, :]
    downsampled_t = ref_t[::downsample_step]

    downsampled_t = downsampled_t - downsampled_t[0]

    final_sf = (len(downsampled_t) - 1) / (downsampled_t[-1] - downsampled_t[0]) if len(downsampled_t) > 1 else 0
    print(f"Final sampling frequency: {final_sf:.1f} Hz")
    print(f"Final data length: {len(downsampled_t)} samples")

    # Drop the first few seconds because of synchronization
    drop_secs = 0.0
    start_idx = np.searchsorted(downsampled_t, drop_secs)
    if drop_secs > 0:
        start_idx = np.searchsorted(downsampled_t, drop_secs)
        emg_60Hz = emg_60Hz[start_idx:]
        angles_60Hz = angles_60Hz[start_idx:]
        downsampled_t = downsampled_t[start_idx:] - drop_secs

    # Save results
    np.save(os.path.join(data_dir, 'aligned_filtered_emg.npy'), emg_60Hz)
    np.save(os.path.join(data_dir, 'aligned_angles.npy'), angles_60Hz)
    np.save(os.path.join(data_dir, 'aligned_timestamps.npy'), downsampled_t)
    save_angles_as_parquet(data_dir, angles_60Hz, downsampled_t, hand_side=hand_side)
    
    print(f"{os.path.basename(data_dir)}: Downsampled to 60Hz ({len(downsampled_t)} datapoints)")
    
    # Print final alignment quality
    print(f"\n=== Final Alignment Quality ===")
    print(f"Final EMG samples: {len(emg_60Hz)}")
    print(f"Final angle samples: {len(angles_60Hz)}")
    print(f"Final duration: {downsampled_t[-1]:.2f}s")
    

def process_all_experiments(person_id, out_root, movement=None, snr_threshold=3.0, hand_side='left', use_timestamp_scaling=True):
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
                    use_timestamp_scaling=use_timestamp_scaling  # Pass the parameter
                )
            except Exception as e:
                print(f"Failed to process {exp_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process all EMG/Angle experiments for a person/movement.')
    parser.add_argument('--person_id', required=True, help='Person ID (e.g. Emanuel_FirstTries)')
    parser.add_argument('--movement', required=False, help='Movement name (e.g. indexFlDigitsEx, optional)')
    parser.add_argument('--out_root', default='data', help='Root directory (default: data)')
    parser.add_argument('--snr_threshold', type=float, default=2, help='SNR threshold for channel selection')
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], default="left", help="Side of the prosthetic hand")
    parser.add_argument('--no_timestamp_scaling', action='store_true', help='Disable timestamp scaling (use original timestamps)')
    args = parser.parse_args()

    use_timestamp_scaling = not args.no_timestamp_scaling  # Default is True unless --no_timestamp_scaling is specified
    
    process_all_experiments(
        args.person_id, 
        args.out_root, 
        args.movement, 
        args.snr_threshold, 
        args.hand_side,
        use_timestamp_scaling=use_timestamp_scaling
    )

