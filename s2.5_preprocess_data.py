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


def end_align_data(emg_data, emg_timestamps, angles_data, angles_timestamps):
    """
    End-align EMG to angles by shifting EMG timestamps so both datasets end at the same time.
    Returns the aligned and cropped data.
    """
    # Calculate shift needed to align end times
    shift = emg_timestamps[-1] - angles_timestamps[-1]
    emg_timestamps_shifted = emg_timestamps - shift
    
    # Mask for EMG samples that overlap with angles timespan
    mask = (emg_timestamps_shifted >= 0) & (emg_timestamps_shifted <= angles_timestamps[-1])
    
    # Apply mask to get overlapping data
    emg_aligned = emg_data[:, mask] if emg_data.ndim == 2 else emg_data[mask]
    emg_timestamps_aligned = emg_timestamps_shifted[mask]
    
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
    ):
    emg = np.load(os.path.join(data_dir, emg_file)).T
    emg_timestamps = np.load(os.path.join(data_dir, emg_timestamps_file))
    angles = np.load(os.path.join(data_dir, angles_file))
    angles_timestamps = np.load(os.path.join(data_dir, angles_timestamps_file))

    # END-ALIGN the data first (moved from s1.5)
    emg, emg_timestamps, angles, angles_timestamps = end_align_data(
        emg, emg_timestamps, angles, angles_timestamps
    )

    scaling_yaml_path = os.path.join(
        out_root, person_id, "recordings", "Calibration", "experiments", "1", "scaling.yaml"
    )
    with open(scaling_yaml_path, 'r') as file:
        scalers = yaml.safe_load(file)
    maxVals = np.array(scalers['maxVals'])
    noiseLevel = np.array(scalers['noiseLevels'])

    emg = emg[used_channels, :]
    maxVals = maxVals[used_channels]
    noiseLevel = noiseLevel[used_channels]

    sf = (len(emg_timestamps) - 1) / (emg_timestamps[-1] - emg_timestamps[0])
    print(f"Sampling frequency sf: {sf}")
    
    # MODIFIED: Use zero-phase filtering instead of EMGClass pipeline
    filtered_emg = process_emg_zero_phase(emg, sf, maxVals, noiseLevel)

    # Use angle timestamps as reference (EMG interpolated onto angle time base)
    ref_t = angles_timestamps
    aligned_angles = angles
    aligned_emg = np.zeros((len(ref_t), filtered_emg.shape[0]))
    for ch in range(filtered_emg.shape[0]):
        aligned_emg[:, ch] = np.interp(ref_t, emg_timestamps, filtered_emg[ch, :])

    ds_freq = 60 # Hz
    t_start = ref_t[0]
    t_end = ref_t[-1]
    n_samples = int(np.floor((t_end - t_start) * ds_freq)) + 1
    downsampled_t = np.linspace(t_start, t_start + (n_samples - 1) / ds_freq, n_samples)

    # Downsample to 60Hz using windowed integration (like EMGClass)
    ds_freq = 60 # Hz
    window_size = int(sf / ds_freq)  # samples per 60Hz window (e.g., ~17 samples for 1000Hz->60Hz)
    
    # Calculate number of complete windows
    n_windows = len(ref_t) // window_size
    
    # Reshape and integrate over windows for EMG
    if aligned_emg.ndim == 1:
        aligned_emg = aligned_emg[:, None]
    
    emg_windowed = aligned_emg[:n_windows * window_size, :].reshape(n_windows, window_size, aligned_emg.shape[1])
    emg_60Hz = np.mean(emg_windowed, axis=1)  # or use np.sum for true integration
    
    # For angles, take the middle sample of each window (or mean)
    angles_windowed = aligned_angles[:n_windows * window_size, :].reshape(n_windows, window_size, aligned_angles.shape[1])
    angles_60Hz = angles_windowed[:, window_size//2, :]  # middle sample, or use np.mean(angles_windowed, axis=1)
    
    # Create corresponding timestamps (middle of each window)
    downsampled_t = ref_t[window_size//2::window_size][:n_windows]

    # Drop the first 3 s because of the synchronization
    drop_secs = 5.0
    start_idx = np.searchsorted(downsampled_t, drop_secs)

    # Slice off sync segments
    emg_60Hz = emg_60Hz[start_idx:]
    angles_60Hz = angles_60Hz[start_idx:]
    downsampled_t = downsampled_t[start_idx:] - drop_secs

    np.save(os.path.join(data_dir, 'aligned_filtered_emg.npy'), emg_60Hz)
    np.save(os.path.join(data_dir, 'aligned_angles.npy'), angles_60Hz)
    np.save(os.path.join(data_dir, 'aligned_timestamps.npy'), downsampled_t)
    save_angles_as_parquet(data_dir, angles_60Hz, downsampled_t, hand_side=hand_side)
    print(f"{os.path.basename(data_dir)}: Downsampled to 60Hz ({len(downsampled_t)} datapoints)")
    

def process_all_experiments(person_id, out_root, movement=None, snr_threshold=3.0, hand_side='left'):
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
                process_emg_and_angles(exp_dir, person_id, out_root, used_channels, hand_side=hand_side)
            except Exception as e:
                print(f"Failed to process {exp_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process all EMG/Angle experiments for a person/movement.')
    parser.add_argument('--person_id', required=True, help='Person ID (e.g. Emanuel_FirstTries)')
    parser.add_argument('--movement', required=False, help='Movement name (e.g. indexFlDigitsEx, optional)')
    parser.add_argument('--out_root', default='data', help='Root directory (default: data)')
    parser.add_argument('--snr_threshold', type=float, default=6, help='SNR threshold for channel selection')
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], default="left", help="Side of the prosthetic hand")
    args = parser.parse_args()
    process_all_experiments(args.person_id, args.out_root, args.movement, args.snr_threshold, args.hand_side)