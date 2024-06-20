import argparse
import os
from os.path import join, exists
import yaml

import pandas as pd
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from helpers.EMGClass import EMG


def crop_data(cap, data_dir, out_dir, start_frame, end_frame, x_start, x_end, y_start, y_end):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = x_end - x_start
    height = y_end - y_start

    output_path = join(out_dir, f'cropped_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = -1
    print('Cropping video...')
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                break

            if frame_count < start_frame:
                continue
            if frame_count >= end_frame:
                break

            cropped_frame = frame[y_start:y_end, x_start:x_end]
            out.write(cropped_frame)

            pbar.update(1)

    cap.release()
    out.release()

    video_timestamps = np.load(os.path.join(data_dir, 'triggered_video_timestamps.npy'))
    np.save(join(out_dir, f'cropped_video_timestamps.npy'), video_timestamps[start_frame:end_frame])


def trigger_crop_video(data_dir):
    cap = cv2.VideoCapture(join(args.data_dir, 'video.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path = join(data_dir, f'triggered_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    found_trigger = False
    record_frame = False

    print("Finding LED trigger...")
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame[0, frame.shape[1]//2, 0] >= 200 and frame[0, frame.shape[1]//2, 1] >= 200 and not found_trigger:
                found_trigger = True

                plt.imshow(frame)
                if record_frame:
                    title = f'End_trigger_frame-{frame_count}'
                else:
                    title = f'Start_trigger_frame-{frame_count}'
                plt.title(title)
                # plt.show()
                plt.savefig(join(data_dir, f'trigger_frame_{frame_count}.png'))
                if record_frame:
                    break

            if found_trigger and frame[0, frame.shape[1]//2, 0] < 200 and frame[0, frame.shape[1]//2, 1] < 200:
                record_frame = True
                found_trigger = False

            if record_frame:
                out.write(frame)

            frame_count += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    out.release()


def trigger_crop_emg(cap, data_dir, trigger_channel, trigger_value):
    emg = np.load(join(data_dir, 'emg.npy'))
    emg_timestamps = np.load(join(data_dir, 'emg_timestamps.npy'))
    video_timestamps = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))


    found_trigger = False
    record_frame = False
    start_frame = -1
    end_frame = -1
    for i, signal in enumerate(emg[:, trigger_channel]):
        if abs(signal) >= trigger_value:
            found_trigger = True
            if record_frame:
                end_frame = i
                print(f'end_frame: {i}')
                break

        if found_trigger and abs(signal) < trigger_value:
            found_trigger = False
            record_frame = True
            start_frame = i
            print(f'start_frame: {i}')

    assert start_frame != -1 and end_frame != -1, 'Could not find trigger in EMG data.'

    triggered_emg = emg[start_frame:end_frame]
    triggered_emg_timestamps = emg_timestamps[start_frame:end_frame]
    np.save(join(data_dir, 'triggered_emg.npy'), triggered_emg)
    np.save(join(data_dir, 'triggered_emg_timestamps.npy'), triggered_emg_timestamps)

    for i in range(len(video_timestamps)):
        video_timestamps[i] = emg_timestamps[start_frame] + (emg_timestamps[end_frame] - emg_timestamps[start_frame]) * i / len(video_timestamps)
        np.save(join(data_dir, 'triggered_video_timestamps.npy'), video_timestamps)
    return start_frame, end_frame


def filter_emg(data_dir):
    emg_data = np.load(join(data_dir, 'triggered_emg.npy')).T
    emg_timestamps = np.load(join(data_dir, 'triggered_emg_timestamps.npy'))
    with open(join(args.data_dir, 'scaling.yaml'), 'r') as file:
        scalers = yaml.safe_load(file)

    maxVals = np.array([scalers['maxVals'][i] for i in range(16)])
    noiseLevel = np.array([scalers['noiseLevels'][i] for i in range(16)])
    sf = (len(emg_timestamps) - 1) * 10**6 / (emg_timestamps[-1] - emg_timestamps[0])

    emg = EMG(samplingFreq=sf, offlineData=emg_data, maxVals=maxVals, noiseLevel=noiseLevel)
    emg.startCommunication()
    emg.emgThread.join()
    normalized_emg = emg.emgHistory[:, emg.numPackets * 100 + 1:]
    normalized_emg = normalized_emg[:, :emg_data.shape[1]]

    np.save(join(data_dir, 'filtered_emg.npy'), normalized_emg.T)


def downsample_video(cap, out_dir):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = join(out_dir, f'cropped_downsampled_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 60, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    old_video_timestamps = np.load(join(out_dir, 'cropped_video_timestamps.npy'))
    new_video_timestamps = []

    frame_count = 0
    print('Downsampling video...')
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 2 == 0:
                out.write(frame)
                new_video_timestamps.append(old_video_timestamps[frame_count])

            frame_count += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        out.release()

    np.save(join(out_dir, 'cropped_video_timestamps.npy'), np.array(new_video_timestamps))
    os.remove(join(out_dir, 'cropped_video.mp4'))
    os.rename(output_path, join(out_dir, 'cropped_video.mp4'))


def align_emg(data_dir, out_dir):
    emg = np.load(join(data_dir, 'filtered_emg.npy'))
    video_timestamps = np.load(join(out_dir, 'cropped_video_timestamps.npy'))
    emg_timestamps = np.load(join(data_dir, 'triggered_emg_timestamps.npy'))

    # Align EMG data with video timestamps
    aligned_emg = np.zeros((len(video_timestamps), emg.shape[1]))
    for i, timestamp in enumerate(video_timestamps):

        idx = np.argmin(np.abs(emg_timestamps - timestamp))
        aligned_emg[i] = emg[idx]

    np.save(join(out_dir, 'aligned_emg.npy'), aligned_emg)


def check_cropping_args(args):
    x_start = int(args.x_start)
    x_end = int(args.x_end)
    y_start = int(args.y_start)
    y_end = int(args.y_end)
    start_frame = int(args.start_frame)
    end_frame = int(args.end_frame)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if x_end == -1:
        x_end = frame_width
    if y_end == -1:
        y_end = frame_height
    if end_frame == -1:
        end_frame = n_frames

    if x_start < 0 or x_start >= frame_width:
        raise ValueError(f'Invalid x_start value: {x_start}')
    if x_end < 0 or x_end > frame_width or x_end <= x_start:
        raise ValueError(f'Invalid x_end value: {x_end}')
    if y_start < 0 or y_start > frame_height:
        raise ValueError(f'Invalid y_start value: {y_start}')
    if y_end < 0 or y_end > frame_height or y_end <= y_start:
        raise ValueError(f'Invalid y_end value: {y_end}')
    if start_frame < 0 or start_frame >= n_frames:
        raise ValueError(f'Invalid start_frame value: {start_frame}')
    if end_frame < 0 or end_frame > n_frames or end_frame <= start_frame:
        raise ValueError(f'Invalid end_frame value: {end_frame}')

    return x_start, x_end, y_start, y_end, start_frame, end_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--x_start', type=int, default=0, help='Start x coordinate for cropping')
    parser.add_argument('--x_end', type=int, default=-1, help='End x coordinate for cropping')
    parser.add_argument('--y_start', type=int, default=0, help='Start y coordinate for cropping')
    parser.add_argument('--y_end', type=int, default=-1, help='End y coordinate for cropping')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame for cropping')
    parser.add_argument('--end_frame', type=int, default=-1, help='End frame for cropping')
    parser.add_argument('--trigger_channel', type=int, required=True, help='Trigger channel')
    parser.add_argument('--trigger_value', type=int, default=550, help='Trigger value')
    parser.add_argument('--process', action='store_true', help='Process video, otherwise show video and EMG trigger channel')
    args = parser.parse_args()

    if not exists(join(args.data_dir, 'video.mp4')):
        for file in os.listdir(args.data_dir):
            if file.endswith('.MP4'):
                os.rename(join(args.data_dir, file), join(args.data_dir, 'video.mp4'))
                break

    if not exists(join(args.data_dir, 'emg.npy')):
        for file in os.listdir(args.data_dir):
            if file.endswith('.csv'):
                emg = pd.read_csv(join(args.data_dir, file), delimiter='\t')
                emg = emg.to_numpy()
                np.save(join(args.data_dir, 'emg.npy'), emg[:, 0:16])
                np.save(join(args.data_dir, 'emg_timestamps.npy'), emg[:, 19])
                break

    if not exists(join(args.data_dir, 'triggered_video.mp4')):
        trigger_crop_video(args.data_dir)
    cap = cv2.VideoCapture(join(args.data_dir, 'triggered_video.mp4'))

    start_frame, end_frame = trigger_crop_emg(cap, args.data_dir, trigger_channel=args.trigger_channel, trigger_value=abs(args.trigger_value))
    emg = np.load(join(args.data_dir, 'emg.npy'))
    plt.figure(figsize=(10, 5))
    plt.plot(emg[:, args.trigger_channel])
    plt.axvline(x=start_frame, color='r', linestyle='--')
    plt.axvline(x=end_frame, color='g', linestyle='--')
    # horizontal line at trigger value
    if np.mean(emg[:, args.trigger_channel]) > 0:
        plt.axhline(y=args.trigger_value, color='k', linestyle='--')
    else:
        plt.axhline(y=-args.trigger_value, color='k', linestyle='--')
    plt.title(f'Trigger value: {args.trigger_value}')

    if not args.process:
        plt.show()
        subprocess.run(['python', 'video_gui.py', '--file', join(args.data_dir, 'triggered_video.mp4')])

    else:
        plt.savefig(join(args.data_dir, 'trigger_channel.png'))

        filter_emg(args.data_dir)

        out_dir = join(args.data_dir, 'experiments', args.experiment_name)
        if exists(out_dir):
            raise ValueError(f'Output directory {out_dir} already exists.')
        os.makedirs(out_dir, exist_ok=True)

        x_start, x_end, y_start, y_end, start_frame, end_frame = check_cropping_args(args)
        crop_data(cap, args.data_dir, out_dir, start_frame=start_frame, end_frame=end_frame, x_start=x_start,
                   x_end=x_end, y_start=y_start, y_end=y_end)

        cap = cv2.VideoCapture(join(out_dir, 'cropped_video.mp4'))
        if cap.get(cv2.CAP_PROP_FPS) > 115 and cap.get(cv2.CAP_PROP_FPS) < 125:
            downsample_video(cap, out_dir)

        align_emg(args.data_dir, out_dir)

