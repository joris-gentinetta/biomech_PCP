import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

midpoint = 900


def load_and_split_file(file_path, length=None):
    file_path = str(file_path)
    """Load and split a file into two halves."""
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        return data[:midpoint], data[midpoint:], len(data)
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
        return data.iloc[:midpoint].reset_index(drop=True), data.iloc[midpoint:length].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def concatenate_files(data_dir, folders, experiment):
    emg_data_first_half = []
    emg_data_second_half = []
    angles_data_first_half = []
    angles_data_second_half = []

    for folder in folders:
        folder_path = Path(data_dir) / folder / 'experiments/2'
        emg_file_path = folder_path / 'cropped_emg.npy'
        angles_file_path = folder_path / 'cropped_smooth_angles.parquet'

        if emg_file_path.exists() and angles_file_path.exists():
            emg_first_half, emg_second_half, length = load_and_split_file(emg_file_path)
            if length != 1800:
                print(f"Length of {folder} is only {length}")
            angles_first_half, angles_second_half = load_and_split_file(angles_file_path, length)

            emg_data_first_half.append(emg_first_half)
            emg_data_second_half.append(emg_second_half)

            angles_data_first_half.append(angles_first_half)
            angles_data_second_half.append(angles_second_half)
        else:
            print(f"Missing files in {folder}, skipping...")

    # Invert the order of the second halves before concatenation
    emg_data_second_half.reverse()
    angles_data_second_half.reverse()

    # Concatenate the halves
    final_emg_first_half = np.concatenate(emg_data_first_half, axis=0)
    final_emg_second_half = np.concatenate(emg_data_second_half, axis=0)
    final_angles_first_half = pd.concat(angles_data_first_half, ignore_index=True)
    final_angles_second_half = pd.concat(angles_data_second_half, ignore_index=True)

    # Save the concatenated results
    output_dir = Path(data_dir) / experiment / 'experiments' / '1'
    output_dir.mkdir(parents=True, exist_ok=True)

    final_angles_first_half = pd.concat([final_angles_first_half, final_angles_second_half], ignore_index=True)
    np.save(output_dir / 'cropped_emg.npy', np.concatenate([final_emg_first_half, final_emg_second_half], axis=0))
    final_angles_first_half.to_parquet(output_dir / 'cropped_smooth_angles.parquet')

    print(f"Combined files saved in {output_dir}")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--person_id", type=str, required=True)
    args = args.parse_args()

    data_dir = f"data/{args.person_id}/recordings"

    experiment = 'online_concat'
    folders = [
        "indexFlDigitsEx",
        "wristFlHandCl",
        "keyOpCl",
        "pointOpCl",
        "pinchOpCl",
        "handOpCl",
        "wristFlEx",
        "fingersFlEx",
        "mrpFlEx",
        "indexFlEx",
        "thumbAbAd",
        "thumbFlEx"
    ]
    concatenate_files(data_dir, folders, experiment)

    experiment = 'online_concat_inverse'
    folders.reverse()
    concatenate_files(data_dir, folders, experiment)

    experiment = 'online_concat_comp'
    folders = [
        'thumbFlEx',
        'thumbAbAd',
        'indexFlEx',
        'mrpFlEx',
        'fingersFlEx',
        'handOpCl',
        'pinchOpCl',
        'pointOpCl',
        'keyOpCl',
        'indexFlDigitsEx',
        'wristFlEx',
        'wristFlHandCl'
    ]
    concatenate_files(data_dir, folders, experiment)

    # create interpolated data:
    file_path = Path(f'data/{args.person_id}/recordings/online_concat_comp/experiments/1')
    output_file_path = Path(f'data/{args.person_id}/recordings/online_concat_comp_interp/experiments/1')

    if os.path.exists(file_path):
        df = pd.read_parquet(file_path / 'cropped_smooth_angles.parquet')
        df_sampled = df.iloc[::3].copy()
        df_interpolated = df_sampled.reindex(df.index)
        df_interpolated = df_interpolated.interpolate()
        emg = np.load(file_path / 'cropped_emg.npy')

        os.makedirs(output_file_path, exist_ok=True)
        df_interpolated.to_parquet(output_file_path / 'cropped_smooth_angles.parquet')
        np.save(output_file_path / 'cropped_emg.npy', emg)

        print(f'Data for {args.person_id} saved to {output_file_path}')
    else:
        print(f'File {file_path} does not exist.')
