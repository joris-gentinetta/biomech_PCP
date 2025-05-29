import os
import numpy as np
import yaml
import argparse
import pandas as pd
from helpers.EMGClass import EMG

SKIP_MOVEMENTS = {"calibration", "Calibration", "calib", "Calib"}

def get_used_channels_from_snr(scaling_yaml_path, snr_threshold=5.0):
    """Returns indices of channels with SNR above threshold."""
    with open(scaling_yaml_path, 'r') as file:
        scalers = yaml.safe_load(file)
    maxVals = np.array(scalers['maxVals'])
    noiseLevel = np.array(scalers['noiseLevels'])
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

    print("Going into EMG Class !!!!!!!!!!!!!!")

    sf = (len(emg_timestamps) - 1) * 1e6 / (emg_timestamps[-1] - emg_timestamps[0])
    print(f"Sampling frequency sf: {sf}")
    emg_proc = EMG(samplingFreq=sf, offlineData=emg, maxVals=maxVals, noiseLevel=noiseLevel, numElectrodes=len(used_channels))
    print("Going out of EMG Class !!!!!!!!!!!!!!")
    emg_proc.startCommunication()
    emg_proc.emgThread.join()

    filtered_emg = emg_proc.emgHistory[:, emg_proc.numPackets * 100 + 1:]
    filtered_emg = filtered_emg[:, :emg.shape[1]]

    emg_t = emg_timestamps / 1e6  # seconds
    angle_t = angles_timestamps

    if len(emg_t) >= len(angle_t):
        ref_t = emg_t
        aligned_emg = filtered_emg.T
        aligned_angles = np.zeros((len(ref_t), angles.shape[1]))
        for dof in range(angles.shape[1]):
            aligned_angles[:, dof] = np.interp(ref_t, angle_t, angles[:, dof])
    else:
        ref_t = angle_t
        aligned_angles = angles
        aligned_emg = np.zeros((len(ref_t), filtered_emg.shape[0]))
        for ch in range(filtered_emg.shape[0]):
            aligned_emg[:, ch] = np.interp(ref_t, emg_t, filtered_emg[ch, :])

    np.save(os.path.join(data_dir, 'aligned_filtered_emg.npy'), aligned_emg)
    np.save(os.path.join(data_dir, 'aligned_angles.npy'), aligned_angles)
    np.save(os.path.join(data_dir, 'aligned_timestamps.npy'), ref_t)
    save_angles_as_parquet(data_dir, aligned_angles, ref_t, hand_side=hand_side)
    print(f"{os.path.basename(data_dir)}: {aligned_emg.shape[0]} synchronized EMG and angle datapoints saved.")

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
    parser.add_argument('--snr_threshold', type=float, default=0.3, help='SNR threshold for channel selection')
    parser.add_argument("--hand_side", "-s", choices=["left", "right"], default="left", help="Side of the prosthetic hand")
    args = parser.parse_args()
    process_all_experiments(args.person_id, args.out_root, args.movement, args.snr_threshold, args.hand_side)