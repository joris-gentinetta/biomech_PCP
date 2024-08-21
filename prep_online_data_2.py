import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# Define the list of folders in the desired order
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

def load_and_split_file(file_path):
    file_path = str(file_path)
    """Load and split a file into two halves."""
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        midpoint = len(data) // 2
        return data[:midpoint], data[midpoint:]
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
        midpoint = len(data) // 2
        return data.iloc[:midpoint].reset_index(drop=True), data.iloc[midpoint:].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def concatenate_files(data_dir, folders):
    emg_data_first_half = []
    emg_data_second_half = []
    angles_data_first_half = []
    angles_data_second_half = []

    for folder in folders:
        folder_path = Path(data_dir) / folder / 'experiments/2'
        emg_file_path = folder_path / 'cropped_emg.npy'
        angles_file_path = folder_path / 'cropped_smooth_angles.parquet'

        if emg_file_path.exists() and angles_file_path.exists():
            emg_first_half, emg_second_half = load_and_split_file(emg_file_path)
            angles_first_half, angles_second_half = load_and_split_file(angles_file_path)

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
    output_dir = Path(data_dir) / 'online_concat' / 'experiments' / '1'
    output_dir.mkdir(parents=True, exist_ok=True)

    final_angles_first_half = pd.concat([final_angles_first_half, final_angles_second_half], ignore_index=True)
    np.save(output_dir / 'cropped_emg.npy', np.concatenate([final_emg_first_half, final_emg_second_half], axis=0))
    final_angles_first_half.to_parquet(output_dir / 'cropped_smooth_angles.parquet')

    print(f"Combined files saved in {output_dir}")

    # final_angles_first_half.plot()
    # # not label:
    # plt.legend().remove()
    # plt.show()
    # plt.plot(np.concatenate([final_emg_first_half, final_emg_second_half], axis=0))
    # plt.show()



if __name__ == "__main__":
    for person_id in ["P_577", "P_407", "P_711", "P_238", "P_426", "P_950", "P_149", "P_668"]:
        data_dir = f"data/{person_id}/recordings"
        concatenate_files(data_dir, folders)


        # for recording in folders:
        #     angles_file = Path(data_dir) / recording / 'experiments/2' / 'cropped_smooth_angles.parquet'
        #     if not os.path.exists(angles_file):
        #         print(f"MISSING: person_id: {person_id}, recording: {recording}")
        #         continue
        #     angles = pd.read_parquet(angles_file)
        #     #check for nan:
        #     if angles.isnull().values.any():
        #         print(f"NaN: person_id: {person_id}, recording: {recording}")
        #         continue
        #     #check for inf:
        #     if angles.isin([np.inf, -np.inf]).values.any():
        #         print(f"Inf: person_id: {person_id}, recording: {recording}")
        #         continue
        #     # plot the data: (only left arm) multiindex: (Left/Right, Joint)
        #     left_arm_angles = angles.xs('Left', level=0, axis=1)
        #     left_arm_angles.plot()
        #     plt.title(f"{person_id} - {recording}")
        #     plt.show()
