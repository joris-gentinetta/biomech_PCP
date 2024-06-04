import argparse
import os
from os.path import join

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
    timestamps = np.load(join(data_dir, 'video_timestamps.npy')) if os.path.exists(join(data_dir, 'video_timestamps.npy')) else np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    cropped_timestamps = []
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
                cropped_timestamps.append(timestamps[frame_count])

            frame_count += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    out.release()
    if os.path.exists(join(data_dir, 'video_timestamps.npy')):  # if webcam video
        cropped_timestamps = np.array(cropped_timestamps)
        np.save(join(data_dir, 'triggered_video_timestamps.npy'), cropped_timestamps)


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

    plt.figure(figsize=(10, 5))
    plt.plot(emg[:, args.trigger_channel])
    plt.title(f'start: {start_frame}, end: {end_frame}')
    # plt.show()
    plt.savefig(join(args.data_dir, 'trigger_channel.png'))


    assert start_frame != -1 and end_frame != -1, 'Could not find trigger in EMG data.'

    triggered_emg = emg[start_frame:end_frame]
    triggered_emg_timestamps = emg_timestamps[start_frame:end_frame]
    np.save(join(data_dir, 'triggered_emg.npy'), triggered_emg)
    np.save(join(data_dir, 'triggered_emg_timestamps.npy'), triggered_emg_timestamps)

    if not os.path.exists(join(args.data_dir, 'triggered_video_timestamps.npy')):  # if not webcam video
        for i in range(len(video_timestamps)):
            video_timestamps[i] = emg_timestamps[start_frame] + (emg_timestamps[end_frame] - emg_timestamps[start_frame]) * i / len(video_timestamps)
            np.save(join(data_dir, 'triggered_video_timestamps.npy'), video_timestamps)


def filter_emg(data_dir):
    emg_data = np.load(join(data_dir, 'triggered_emg.npy')).T
    emg_timestamps = np.load(join(data_dir, 'triggered_emg_timestamps.npy'))
    sf = (len(emg_timestamps) - 1) * 10**6 / (emg_timestamps[-1] - emg_timestamps[0])
    emg = EMG(samplingFreq=sf, offlineData=emg_data)
    emg.startCommunication()
    emg.emgThread.join()
    filtered_emg = emg.emgHistory[:, emg.numPackets * 100 + 1:]
    filtered_emg = filtered_emg[:, :emg_data.shape[1]]
    # filtered_emg_timestamps = [emg_timestamps[i * emg.numPackets + emg.numPackets // 2] for i in
    #                            range(filtered_emg.shape[1])]
    filtered_emg_timestamps = emg_timestamps

    # min_vals = np.percentile(filtered_emg, 1, axis=1)
    # max_vals = np.percentile(filtered_emg, 99, axis=1)
    min_vals = filtered_emg.min(axis=1)
    max_vals = filtered_emg.max(axis=1)
    min_vals[:] = 10

    max_vals[0] = 400
    max_vals[1] = 40
    max_vals[2] = 150
    max_vals[4] = 200
    max_vals[5] = 250
    max_vals[8] = 150
    max_vals[10] = 400
    max_vals[11] = 300
    normalized_emg = np.clip((filtered_emg - min_vals[:, None]) / (max_vals - min_vals)[:, None], 0, 1)
    np.save(join(data_dir, 'min_norm_vals.npy'), min_vals)
    np.save(join(data_dir, 'max_norm_vals.npy'), max_vals)
    np.save(join(data_dir, 'filtered_emg.npy'), normalized_emg.T)
    np.save(join(data_dir, 'filtered_emg_timestamps.npy'), filtered_emg_timestamps)

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
    emg_timestamps = np.load(join(data_dir, 'filtered_emg_timestamps.npy'))

    # Align EMG data with video timestamps
    aligned_emg = np.zeros((len(video_timestamps), emg.shape[1]))
    for i, timestamp in enumerate(video_timestamps):

        idx = np.argmin(np.abs(emg_timestamps - timestamp))
        aligned_emg[i] = emg[idx]

    np.save(join(out_dir, 'aligned_emg.npy'), aligned_emg)


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
    parser.add_argument('--trigger_value', type=int, default=600, help='Trigger value')
    parser.add_argument('--process', action='store_true', help='Process video, otherwise show video and EMG trigger channel')

    args = parser.parse_args()

    if not os.path.exists(join(args.data_dir, 'video.mp4')):
        # find MP4 file in data_dir and rename to video.mp4
        for file in os.listdir(args.data_dir):
            if file.endswith('.MP4'):
                os.rename(join(args.data_dir, file), join(args.data_dir, 'video.mp4'))
                break
    if not os.path.exists(join(args.data_dir, 'emg.npy')):
        for file in os.listdir(args.data_dir):
            if file.endswith('.csv'):
                emg = pd.read_csv(join(args.data_dir, file), delimiter='\t')
                emg = emg.to_numpy()
                np.save(join(args.data_dir, 'emg.npy'), emg[:, 0:16])
                np.save(join(args.data_dir, 'emg_timestamps.npy'), emg[:, 19])
                break

    if not os.path.exists(join(args.data_dir, 'triggered_video.mp4')):
        trigger_crop_video(args.data_dir)  # produces triggered_video_timestamps.npy for webcam case
    cap = cv2.VideoCapture(join(args.data_dir, 'triggered_video.mp4'))

    if args.process:
        if not os.path.exists(join(args.data_dir, 'triggered_emg.npy')):  # produces triggered_video_timestamps.npy for prof cam case
            trigger_crop_emg(cap, args.data_dir, trigger_channel=args.trigger_channel, trigger_value=abs(args.trigger_value))

        if not os.path.exists(join(args.data_dir, 'filtered_emg.npy')):
            filter_emg(args.data_dir)

        x_start = int(args.x_start)
        x_end = int(args.x_end)
        y_start = int(args.y_start)
        y_end = int(args.y_end)
        start_frame = int(args.start_frame)
        end_frame = int(args.end_frame)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if x_end == -1:
            x_end = frame_width
        if y_end == -1:
            y_end = frame_height
        if end_frame == -1:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if x_start < 0 or x_start >= frame_width:
            raise ValueError(f'Invalid x_start value: {x_start}')
        if x_end < 0 or x_end > frame_width or x_end <= x_start:
            raise ValueError(f'Invalid x_end value: {x_end}')
        if y_start < 0 or y_start > frame_height:
            raise ValueError(f'Invalid y_start value: {y_start}')
        if y_end < 0 or y_end > frame_height or y_end <= y_start:
            raise ValueError(f'Invalid y_end value: {y_end}')

        out_dir = join(args.data_dir, 'experiments', args.experiment_name)
        if os.path.exists(out_dir):
            raise ValueError(f'Output directory {out_dir} already exists.')
        os.makedirs(out_dir, exist_ok=True)

        crop_data(cap, args.data_dir, out_dir, start_frame=start_frame, end_frame=end_frame, x_start=x_start,
                   x_end=x_end, y_start=y_start, y_end=y_end)

        cap = cv2.VideoCapture(join(out_dir, 'cropped_video.mp4'))
        if cap.get(cv2.CAP_PROP_FPS) > 115 and cap.get(cv2.CAP_PROP_FPS) < 125:
            downsample_video(cap, out_dir)
        if not os.path.exists(join(out_dir, 'aligned_emg.npy')):
            align_emg(args.data_dir, out_dir)

    else:
        emg = np.load(join(args.data_dir, 'emg.npy'))
        # make the plot higher
        plt.figure(figsize=(10, 5))
        plt.plot(emg[:, args.trigger_channel])
        plt.title(f'Trigger value: {args.trigger_value}')
        plt.show()
        # plt.savefig(join(args.data_dir, 'trigger_channel.png'))

        subprocess.run(['python', 'video_gui.py', '--file', join(args.data_dir, 'triggered_video.mp4')])

