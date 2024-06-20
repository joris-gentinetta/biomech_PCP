import os
import subprocess
from multiprocessing import Process
import shutil
def run_script(directory):
    subprocess.run(['python', '/Users/jg/projects/biomech/DataGen/2_preprocess_data.py', '--data_dir', directory, '--experiment_name', '1', '--trigger_channel', '15', '--process'])

if __name__ == '__main__':
    base_dir = '/Users/jg/projects/biomech/DataGen/data/linda/minJerk'
    directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) ] # and d not in ['fingers-no_camera', 'minJerk', 'video_archive', 'wrist-no_end_trigger, keyboardTrain' ]]

    # for d in directories:
    #     dir = os.path.join(base_dir, d)
    #     for f in os.listdir(dir):
    #         if not (f.endswith('.csv') or f.endswith('.XML') or f.endswith('.yaml') or f == 'video.mp4'):
    #             print(os.path.join(dir, f))
    #             try:
    #                 os.remove(os.path.join(dir, f))
    #             except:
    #                 shutil.rmtree(os.path.join(dir, f))

    processes = [Process(target=run_script, args=(d,)) for d in directories]

    for p in processes:
        p.start()

    for p in processes:
        p.join()