import os
import subprocess
from multiprocessing import Process

def run_script(directory):
    subprocess.run(['python', 'preprocess_data.py', '--data_dir', directory, '--experiment_name', '1', '--trigger_channel', '15', '--process'])

if __name__ == '__main__':
    base_dir = '/Users/jg/projects/biomech/DataGen/data/linda'
    directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ['fingers-no_camera', 'minJerk', 'video_archive', 'wrist-no_end_trigger, keyboardTrain' ]]

    processes = [Process(target=run_script, args=(d,)) for d in directories]

    for p in processes:
        p.start()

    for p in processes:
        p.join()