import numpy as np

person_dirs = ['P_577', 'P_407', 'P_711', 'P_238', 'P_426', 'P_950', 'P_149', 'P_668']


def evaluate_true_online():
    import subprocess

    # List of person directories

    # Define the base command
    base_command = "python s4_train.py --intact_hand Left --config_name modular_online --evaluate"

    # Loop through each person directory and run the command
    for person_dir in person_dirs:
        for perturb in ['', '--perturb']:
            # Create the full command by appending the person_dir
            command = f"{base_command} --person_dir {person_dir} {perturb}"

            # Print the command (for debugging)
            print(f"Running command: {command}")

            # Execute the command
            subprocess.run(command, shell=True, check=True)

def interpolate():
    import pandas as pd
    import os
    from pathlib import Path


    # Base output directory
    base_output_dir = ''

    # Process each person directory
    for person_dir in person_dirs:
        # Define the input file path
        file_path = Path(f'data/{person_dir}/recordings/online_concat/experiments/1')
        output_file_path = Path(f'data/{person_dir}/recordings/online_concat_interp/experiments/1')

        if os.path.exists(file_path):
            # Load the parquet file
            df = pd.read_parquet(file_path / 'cropped_smooth_angles.parquet')
            df_sampled = df.iloc[::3].copy()
            df_interpolated = df_sampled.reindex(df.index)
            df_interpolated = df_interpolated.interpolate()
            emg = np.load(file_path / 'cropped_emg.npy')

            os.makedirs(output_file_path , exist_ok=True)
            df_interpolated.to_parquet(output_file_path / 'cropped_smooth_angles.parquet')
            np.save(output_file_path / 'cropped_emg.npy', emg)


            print(f'Data for {person_dir} saved to {output_file_path}')
        else:
            print(f'File {file_path} does not exist.')

    print('Processing complete.')

if __name__ == "__main__":
    # evaluate_true_online()
    interpolate()
    # import pandas as pd
    # import numpy as np
    #
    # # Example DataFrame
    # df = pd.DataFrame({
    #     'x': np.arange(0, 100, 1),
    #     'y': np.sin(np.arange(0, 100, 1))
    # })
    #
    # # Keep every 3rd data point
    # df_sampled = df.iloc[::3].copy()
    #
    # # Reindex to match the original DataFrame
    # df_interpolated = df_sampled.reindex(df.index)
    #
    # # Interpolate the missing values
    # df_interpolated['y'] = df_interpolated['y'].interpolate()
    #
    # print(df_interpolated)
