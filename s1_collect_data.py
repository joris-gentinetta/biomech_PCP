import argparse
import os
import signal
import threading
import time
from os.path import join
import multiprocessing
import numpy as np
import sys

from config import NUM_CHANNELS, SENSORS_PER_FINGER, NUM_FINGERS, BUFFER_LENGTH, ACQ_FREQUENCY
from helpers.EMGClass import EMG
from helpers.ForceClass import Force

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
        # Start communication; note that any parameters (like raw=True) should be adjusted as needed.
        emg.startCommunication()
        emgHistory = np.empty((1, emg.numElectrodes))
        # Wait until the EMG thread updates OS_time
        last_time = emg.OS_time if emg.OS_time is not None else time.time()
        while not stop_flag.value:
            time_sample = emg.OS_time if emg.OS_time is not None else time.time()
            # For simplicity, we assume a valid sample is received if time increases.
            if time_sample > last_time:
                emg_sample = np.asarray(emg.rawEMG)
                emgHistory = np.concatenate((emgHistory, emg_sample[None, :]), axis=0)
                emg_timestamps.append(time_sample)
            last_time = time_sample
        emg.exitEvent.set()
    np.save(join(output_dir, 'emg.npy'), emgHistory)
    np.save(join(output_dir, 'emg_timestamps.npy'), emg_timestamps)
    print("EMG data collection finished.")

def capture_Force(stop_flag, output_dir, dummy_force):
    force_timestamps = []
    if dummy_force:
        num_channels = 30
        forceHistory = np.empty((1, num_channels))
        start_time = time.time()
        while not stop_flag.value:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time < 5: # First 5 seconds no contact to simulate free space
                thisForce = np.zeros((1, num_channels))
            else: # After 5s simulate force to switch to interaction mode (0.7N with noise)
                thisForce = 0.7 * np.ones((1, num_channels)) + 0.05 * np.random.randn(1, num_channels)
            
            force_timestamps.append(time.time())
            thisForce = np.random.rand(1, num_channels)
            forceHistory = np.concatenate((forceHistory, thisForce), axis=0)
    else:
        # Instantiate the Force object from ForceClass (which now uses s0_forceInterface)
        force_sensor = Force(hand='left', frequency=ACQ_FREQUENCY, numChannels=NUM_CHANNELS, samplingFreq=ACQ_FREQUENCY)
        
        # Baseline is necc. for normalization, it is done in start communication if it fails here
        force_sensor.calibrateBaseline(num_samples=100)
        
        force_sensor.startCommunication()
        # While the background thread in force_sensor is processing, record timestamps.
        while not stop_flag.value:
            force_timestamps.append(time.time())
            time.sleep(0.01)
        force_sensor.shutdown()
        forceHistory = force_sensor.forceHistory
    np.save(join(output_dir, 'force.npy'), forceHistory)
    np.save(join(output_dir, 'force_timestamps.npy'), force_timestamps)
    print("Force data collection finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture EMG and Force Data.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dummy_emg', action='store_true', help='Use dummy EMG data for testing')
    parser.add_argument('--dummy_force', action='store_true', help='Use dummy force data for testing')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Create threads for both EMG and Force data collection.
    emg_thread = threading.Thread(target=capture_EMG, args=(stop_flag, args.data_dir, args.dummy_emg))
    force_thread = threading.Thread(target=capture_Force, args=(stop_flag, args.data_dir, args.dummy_force))

    emg_thread.start()
    force_thread.start()

    # Run data collection until CTRL+C is pressed.
    try:
        while not stop_flag.value:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_flag.value = True

    # Wait for both threads to finish.
    emg_thread.join()
    force_thread.join()

    # Load and print stats for EMG data.
    emg_timestamps = np.load(join(args.data_dir, 'emg_timestamps.npy'))
    emg_data = np.load(join(args.data_dir, 'emg.npy'))
    print('######################## EMG STATS ########################')
    print(f"Data Dir: {args.data_dir}")
    print(f'EMG Shape: {emg_data.shape}')
    if len(emg_timestamps) > 1:
        sampling_rate_emg = len(emg_timestamps) * 1e6 / (emg_timestamps[-1] - emg_timestamps[0])
        print(f'EMG Sampling Rate: {sampling_rate_emg} Hz')
    print('########################################################')

    # Load and print stats for Force data.
    force_timestamps = np.load(join(args.data_dir, 'force_timestamps.npy'))
    force_data = np.load(join(args.data_dir, 'force.npy'))
    print('######################## FORCE STATS ########################')
    print(f"Data Dir: {args.data_dir}")
    print(f'Force Shape: {force_data.shape}')
    if len(force_timestamps) > 1:
        sampling_rate_force = len(force_timestamps) * 1e6 / (force_timestamps[-1] - force_timestamps[0])
        print(f'Force Sampling Rate: {sampling_rate_force} Hz')
    print('########################################################')
