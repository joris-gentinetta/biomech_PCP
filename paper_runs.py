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


if __name__ == "__main__":
    evaluate_true_online()