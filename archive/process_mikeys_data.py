# import os
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from os.path import join
#
# def split_and_save(data_dir):
#     # Traverse the directories
#     for root, dirs, files in os.walk(data_dir):
#         # Check if the specific files exist in the current directory
#         emg_path = os.path.join(root, 'experiments/1/cropped_emg.npy')
#         angles_path = os.path.join(root, 'experiments/1/cropped_smooth_angles.parquet')
#
#         if os.path.exists(emg_path) and os.path.exists(angles_path):
#             # Load the files
#             emg_data = np.load(emg_path)
#             angles_data = pd.read_parquet(angles_path)
#
#             # Determine the midpoints
#             emg_midpoint = len(emg_data) // 2
#             angles_midpoint = len(angles_data) // 2
#
#             # Split the data
#             emg_data_1 = emg_data[:emg_midpoint]
#             emg_data_2 = emg_data[emg_midpoint:]
#
#             angles_data_1 = angles_data.iloc[:angles_midpoint].reset_index(drop=True)
#             angles_data_2 = angles_data.iloc[angles_midpoint:].reset_index(drop=True)
#
#             print(len(angles_data_1), len(emg_data_1), len(angles_data_2), len(emg_data_2))
#
#             # Create new directories
#             new_dir_1 = Path((str(Path(root)) + '_1'), 'experiments/1')
#             new_dir_2 = Path((str(Path(root)) + '_2'), 'experiments/1')
#             print(new_dir_1, new_dir_2)
#             new_dir_1.mkdir(parents=True, exist_ok=True)
#             new_dir_2.mkdir(parents=True, exist_ok=True)
#
#             # Save the split files
#             np.save(join(new_dir_1, 'cropped_emg.npy'), emg_data_1)
#             np.save(join(new_dir_2, 'cropped_emg.npy'), emg_data_2)
#
#             angles_data_1.to_parquet(join(new_dir_1, 'cropped_smooth_angles.parquet'))
#             angles_data_2.to_parquet(join(new_dir_2, 'cropped_smooth_angles.parquet'))
#
#             print(f"Processed and saved files for {root}")
#
#
# if __name__ == "__main__":
#     data_dir = "/home/haptix/haptix/biomech_PCP/data/mikey/recordings"
#     split_and_save(data_dir)

import os
import numpy as np
import pandas as pd
from pathlib import Path

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

    np.save(output_dir / 'cropped_emg.npy', np.concatenate([final_emg_first_half, final_emg_second_half], axis=0))
    final_angles_first_half = pd.concat([final_angles_first_half, final_angles_second_half], ignore_index=True)
    final_angles_first_half.to_parquet(output_dir / 'cropped_smooth_angles.parquet')

    print(f"Combined files saved in {output_dir}")

if __name__ == "__main__":
    for person_id in ["P_577", "P_407", "P_711", "P_238", "P_426", "P_950", "P_149", "P_668"]:
        data_dir = f"data/{person_id}/recordings"
        concatenate_files(data_dir, folders)

import os
import numpy as np
import pandas as pd
from pathlib import Path


# def load_and_concatenate_cycling(data_dir):
#     cycling_dir = Path(data_dir) / 'cycling' / 'experiments' / '1'
#     output_dir = Path(data_dir) / 'online_cycling' / 'experiments' / '1'
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # Initialize lists to store data
#     emg_data_list = []
#     angles_data_list = []
#
#     # Load all files in the cycling directory
#     for root, dirs, files in os.walk(cycling_dir):
#         for file in files:
#             if file == 'cropped_emg.npy':
#                 emg_data = np.load(Path(root) / file)
#                 emg_data_list.append(emg_data)
#             elif file == 'cropped_smooth_angles.parquet':
#                 angles_data = pd.read_parquet(Path(root) / file)
#                 angles_data_list.append(angles_data)
#
#     # Concatenate the data
#     if emg_data_list:
#         emg_combined = np.concatenate(emg_data_list * 2, axis=0)[:21600]
#         np.save(output_dir / 'cropped_emg.npy', emg_combined)
#
#     if angles_data_list:
#         angles_combined = pd.concat(angles_data_list * 2, ignore_index=True).iloc[:21600]
#         angles_combined.to_parquet(output_dir / 'cropped_smooth_angles.parquet')
#
#     print(f"Saved concatenated data to {output_dir}")
#
#
# if __name__ == "__main__":
#     data_dir = "/home/haptix/haptix/biomech_PCP/data/mikey/recordings"
#     load_and_concatenate_cycling(data_dir)