import subprocess

data_dir = "/Users/jg/projects/biomech/DataGen/data/joris"
recordings_experiments = {
    'tt': ['test2', 'test3'],
}


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