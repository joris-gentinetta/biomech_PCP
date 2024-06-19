import subprocess

data_dir = "/home/haptix/UE AMI Clinical Work/Patient Data/P7 - 453/P7_0325_2024/Data Recording/PROCESSED"
recordings_experiments = {'P7_0325_free': ['1', '2'],  'P7_0325_pickPlace': ['1'],  'P7_0325_test': ['1', '2', '3', '4'],  'P7_0325_test2': ['1', '2', '3', '4']}
# recordings_experiments = {'tt': ['test2']}


for recording, experiments in recordings_experiments.items():
    for experiment in experiments:
        cmd = [
            'python', 'process_video.py',
            '--data_dir', f'{data_dir}/{recording}',
            '--experiment_name', experiment,
            '--intact_hand', 'Right',
            '--visualize'
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Command '{' '.join(cmd)}' failed.")