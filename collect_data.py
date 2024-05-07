import argparse
import os
import signal
import threading
import time
from os.path import join
import multiprocessing

import cv2
import numpy as np

import sys
from helpers.EMGClass import EMG

try:
    sys.path.append('/home/haptix/haptix/haptix_controller/handsim/src/')
except:
    print("EMGClass not found")

stop_flag = multiprocessing.Value('b', False)  # 'b' is a typecode for boolean


def signal_handler(sig, frame):
    global stop_flag
    stop_flag.value = True


signal.signal(signal.SIGINT, signal_handler)

def show_video(stop_flag):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video window
                break
        print(stop_flag.value)
        if stop_flag.value:
            break

    cap.release()
    cv2.destroyAllWindows()


def capture_video(stop_flag, output_dir, fps):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(join(output_dir, 'video.mp4'), fourcc, fps, (frame_width, frame_height))
    timestamps = []
    ret, frame = cap.read()  # skip the first frame

    while True:
        frame_time = time.time()
        ret, frame = cap.read()
        if ret:
            timestamps.append(frame_time)
            out.write(frame)

        if stop_flag.value:
            break

    cap.release()
    out.release()
    timestamps = np.array(timestamps)
    np.save(join(output_dir, 'video_timestamps.npy'), timestamps)


def capture_EMG(stop_flag, save_type, output_dir, dummy_emg):

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
        emg.startCommunication()
        emgHistory = np.empty((1, emg.numElectrodes))

        emgTime = emg.OS_time
        while not stop_flag.value:
            emg_timestamps.append(time.time())

            if save_type == 'raw':
                thisEMG = np.asarray(emg.rawEMG)
            elif save_type == 'normed':
                thisEMG = np.asarray(emg.normedEMG)
            elif save_type == 'iEMG':
                thisEMG = np.asarray(emg.iEMG)
            elif save_type == 'act':
                thisEMG = np.asarray(emg.muscleAct)
            else:
                raise ValueError(f'Improper EMG saving type {save_type}')

            if abs(emg.OS_time - emgTime)/1e6 > 0.1:
                # alignment lost
                print(f'Read time: {emg.OS_time}, expected time: {emgTime}')
                raise ValueError('EMG alignment lost. Please restart the EMG board and the script.')
            else:
                emgTime = emg.OS_time

            emgHistory = np.concatenate((emgHistory, thisEMG[None, :]), axis=0)

        emg.exitEvent.set()

    np.save(join(output_dir, 'emg.npy'), emgHistory)
    np.save(join(output_dir, 'emg_timestamps.npy'), emg_timestamps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dummy_emg', action='store_true', help='Use this flag for testing without the EMG board')
    parser.add_argument('--camera', type=int, help='Camera Index', default=-1)
    parser.add_argument('--fps', type=int, help='Frames per second', default=30)
    parser.add_argument('--record', action='store_true', help='Record video')
    args = parser.parse_args()
    args.save_type = 'raw'
    if not args.record:
        print('Preview started. No data is being saved! Press Ctrl+C to stop the video.')
        show_video(stop_flag)
        exit(0)
    if os.path.exists(args.data_dir):
        raise ValueError(f'Directory {args.data_dir} already exists. Please provide a new directory.')
    os.makedirs(args.data_dir, exist_ok=True)
    if args.camera != -1:
        video_process = multiprocessing.Process(target=capture_video, args=(stop_flag, args.data_dir, args.fps))
        video_process.start()

    emg_process = multiprocessing.Process(target=capture_EMG,
                                  args=(stop_flag, args.save_type, args.data_dir, args.dummy_emg))
    emg_process.start()

    if args.camera != -1:
        video_process.join()
    emg_process.join()



    emg_timestamps = np.load(join(args.data_dir, 'emg_timestamps.npy'))
    emg = np.load(join(args.data_dir, 'emg.npy'))

    print('######################## STATS ########################')
    print(f"Data Dir: {args.data_dir}\n")

    if args.camera != -1:
        video_timestamps = np.load(join(args.data_dir, 'video_timestamps.npy'))
        cap = cv2.VideoCapture(join(args.data_dir, 'video.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Length: {video_length}")
        print(f'Video Size: W:{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} H:{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
        print(f"Requested Frame Rate: {fps}")
        print(f"Actual Frame Rate: {video_length / (video_timestamps[-1] - video_timestamps[0])}\n")
        cap.release()

    print(f'EMG Shape: {emg.shape}\n')
    print(f'Actual EMG Sampling Rate: {len(emg_timestamps) / (emg_timestamps[-1] - emg_timestamps[0])}')
    print('########################################################')


