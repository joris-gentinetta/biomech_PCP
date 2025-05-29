import argparse
import os
import signal
import threading
import time
from os.path import join
import multiprocessing
import numpy as np
import sys
from helpers.EMGClass import EMG


stop_flag = multiprocessing.Value('b', False)

def signal_handler(sig, frame):
    global stop_flag
    stop_flag.value = True

signal.signal(signal.SIGINT, signal_handler)


def capture_EMG(stop_flag, output_dir, dummy_emg):

    emg_timestamps = []
    if dummy_emg:
        num_electrodes = 8
        emgHistory = np.empty((1, num_electrodes))

        while not stop_flag.value:
            emg_timestamps.append(time.time())
            thisEMG = np.random.rand(1, num_electrodes)
            emgHistory = np.concatenate((emgHistory, thisEMG), axis=0)
    else:
        emg = EMG()
        emg.startCommunication(raw=True)
        emgHistory = np.empty((1, emg.numElectrodes))

        last_time = emg.OS_time
        while not stop_flag.value:
            time_sample = emg.OS_time

            if (time_sample - last_time)/1e6 > 0.1:
                print(f'Read time: {time_sample}, expected time: {last_time}')
                raise ValueError('EMG alignment lost. Please restart the EMG board and the script.')

            elif time_sample > last_time:
                emg_sample = np.asarray(emg.rawEMG)
                emgHistory = np.concatenate((emgHistory, emg_sample[None, :]), axis=0)
                emg_timestamps.append(time_sample)
            last_time = time_sample

        emg.exitEvent.set()

    np.save(join(output_dir, 'emg.npy'), emgHistory)
    np.save(join(output_dir, 'emg_timestamps.npy'), emg_timestamps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dummy_emg', action='store_true', help='Use this flag for testing without the EMG board')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    capture_EMG(stop_flag, args.data_dir, args.dummy_emg)

    emg_timestamps = np.load(join(args.data_dir, 'emg_timestamps.npy'))
    emg = np.load(join(args.data_dir, 'emg.npy'))

    print('######################## STATS ########################')
    print(f"Data Dir: {args.data_dir}\n")
    print(f'EMG Shape: {emg.shape}\n')
    print(f'EMG Sampling Rate: {len(emg_timestamps) * 10**6 / (emg_timestamps[-1] - emg_timestamps[0])}')
    print('########################################################')