import argparse
import os
import signal
import threading
import time
from os.path import join

import cv2
import numpy as np

stop_video = False


def signal_handler(sig, frame):
    print('Recording stopped')
    global stop_video
    stop_video = True


signal.signal(signal.SIGINT, signal_handler)


def capture_video(output_dir):
    global stop_video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(join(output_dir, 'video.mp4'), fourcc, 24.0, (frame_width, frame_height))
    timestamps = []
    while True:
        timestamps.append(time.time())
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

        if stop_video:
            break

    cap.release()
    out.release()
    timestamps = np.array(timestamps)
    np.save(join(output_dir, 'video_timestamps.npy'), timestamps)

    cv2.destroyAllWindows()


def capture_EMG(save_type, output_dir, sampling_rate, dummy_emg=False):
    global stop_video
    delay_time = 1.0 / sampling_rate

    emg_timestamps = []
    if dummy_emg:
        num_electrodes = 8
        emgHistory = np.empty((1, num_electrodes))

        while not stop_video:
            emg_timestamps.append(time.time())
            thisEMG = np.random.rand(1, num_electrodes)
            emgHistory = np.concatenate((emgHistory, thisEMG), axis=0)
            time.sleep(delay_time)
    else:
        from EMGClass import EMG
        emg = EMG()
        emg.startCommunication()
        emgHistory = np.empty((1, emg.numElectrodes))

        while not stop_video:
            emg_timestamps.append(time.time())
            if save_type == 'normed':
                thisEMG = np.asarray(emg.normedEMG)
            elif save_type == 'iEMG':
                thisEMG = np.asarray(emg.iEMG)
            elif save_type == 'act':
                thisEMG = np.asarray(emg.muscleAct)
            else:
                raise ValueError(f'Improper EMG saving type {save_type}')

            emgHistory = np.concatenate((emgHistory, thisEMG[None, :]), axis=0)
            time.sleep(delay_time)

        emg.exitEvent.set()

    np.save(join(output_dir, 'emg.npy'), emgHistory)
    np.save(join(output_dir, 'emg_timestamps.npy'), emg_timestamps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--save_type', type=str, default='normed', help='EMG saving type')
    parser.add_argument('--emg_sampling_rate', type=int, default=1000, help='EMG sampling rate')
    parser.add_argument('--dummy_emg', action='store_true', help='Use this flag for testing without the EMG board')
    parser.add_argument('--camera', type=int, help='Camera Index', default=0)
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    video_thread = threading.Thread(target=capture_video, args=(args.data_dir,))
    emg_thread = threading.Thread(target=capture_EMG,
                                  args=(args.save_type, args.data_dir, args.emg_sampling_rate, args.dummy_emg))
    video_thread.start()
    emg_thread.start()

    video_thread.join()
    emg_thread.join()

    video_timestamps = np.load(join(args.data_dir, 'video_timestamps.npy'))
    cap = cv2.VideoCapture(join(args.data_dir, 'video.mp4'), args.camera)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame Rate: {fps}")
    print(f"Video Length: {video_length}")
    print(f'Video Size: W:{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} H:{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
    print(f'Video Timestamps Length: {len(video_timestamps)}')
    emg_timestamps = np.load(join(args.data_dir, 'emg_timestamps.npy'))
    emg = np.load(join(args.data_dir, 'emg.npy'))
    print(f'EMG Sampling Rate: {args.emg_sampling_rate}')
    print(f'EMG Timestamps Length: {len(emg_timestamps)}')
    print(f'EMG Shape: {emg.shape}')
    print()
