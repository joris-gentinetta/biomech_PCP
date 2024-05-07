import argparse
import os
from os.path import join
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_video(cap, data_dir, out_dir, start_frame, end_frame, x_start, x_end, y_start, y_end):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = x_end - x_start
    height = y_end - y_start

    output_path = join(out_dir, f'cropped_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print('Cropping video...')
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if start_frame <= frame_count:
                if end_frame == -1 or frame_count <= end_frame:
                    cropped_frame = frame[y_start:y_end, x_start:x_end]
                    out.write(cropped_frame)

            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()

    timestamps = np.load(os.path.join(data_dir, 'cropped_video_timestamps.npy'))

    if end_frame == -1:
        end_frame = frame_count - 1
    timestamps = timestamps[start_frame:end_frame + 1]
    np.save(join(out_dir, f'cropped_timestamps.npy'), timestamps)

def trigger_crop(data_dir):
    cap = cv2.VideoCapture(join(args.data_dir, 'video.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path = join(data_dir, f'video_cropped.mp4')
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
                plt.title(f'Trigger, frame {frame_count}')
                plt.show()
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

    cap.release()
    out.release()
    if os.path.exists(join(data_dir, 'video_timestamps.npy')):
        cropped_timestamps = np.array(cropped_timestamps)
        np.save(join(data_dir, 'cropped_video_timestamps.npy'), cropped_timestamps)


def create_video_timestamps(cap, data_dir, trigger_channel, trigger_value):
    emg = np.load(join(data_dir, 'emg.npy'))
    emg_timestamps = np.load(join(data_dir, 'emg_timestamps.npy'))
    video_timestamps = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    plt.plot(emg[:, trigger_channel])
    print(emg.shape)
    plt.show()
    print('Trigger value: ', trigger_value)
    continue_ = input('Continue? (y/n): ')
    if continue_ == 'n':
        print('Exiting.')
        exit(0)

    found_trigger = False
    record_frame = False
    start_time = -1
    end_time = -1
    for i, signal in enumerate(emg[:, trigger_channel]):
        if abs(signal) >= trigger_value:
            found_trigger = True
            if record_frame:
                end_time = emg_timestamps[i]
                print(f'end_frame: {i}')
                break

        if found_trigger and abs(signal) < trigger_value:
            found_trigger = False
            record_frame = True
            start_time = emg_timestamps[i]
            print(f'start_frame: {i}')

    assert start_time != -1 and end_time != -1, 'Could not find trigger in EMG data.'

    for i in range(len(video_timestamps)):
        video_timestamps[i] = start_time + (end_time - start_time) * i / len(video_timestamps)

    np.save(join(data_dir, 'cropped_video_timestamps.npy'), video_timestamps)




def show_frame(cap, frame_number):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == frame_number:
            plt.imshow(frame)
            # label axes:
            plt.xlabel('x')
            plt.ylabel('y')

            print(frame.shape)
            plt.show()
            break

        frame_count += 1

    cap.release()


def align_emg(data_dir, out_dir):
    emg = np.load(join(data_dir, 'emg.npy'))
    video_timestamps = np.load(join(out_dir, 'cropped_timestamps.npy'))
    emg_timestamps = np.load(join(data_dir, 'emg_timestamps.npy'))

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
    parser.add_argument('--crop', action='store_true', help='Crop video, otherwise show frame.')
    parser.add_argument('--frame_number', type=int, default=2, help='Frame number to visualize')
    parser.add_argument('--x_start', type=int, default=0, help='Start x coordinate for cropping')
    parser.add_argument('--x_end', type=int, default=-1, help='End x coordinate for cropping')
    parser.add_argument('--y_start', type=int, default=0, help='Start y coordinate for cropping')
    parser.add_argument('--y_end', type=int, default=-1, help='End y coordinate for cropping')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame for cropping')
    parser.add_argument('--end_frame', type=int, default=-1, help='End frame for cropping')
    parser.add_argument('--trigger_channel', type=int, default=0, help='Trigger channel')
    parser.add_argument('--trigger_value', type=int, default=200, help='Trigger value')
    args = parser.parse_args()



    if not os.path.exists(join(args.data_dir, 'video_cropped.mp4')):
        trigger_crop(args.data_dir)



    cap = cv2.VideoCapture(join(args.data_dir, 'video_cropped.mp4'))
    if not os.path.exists(join(args.data_dir, 'cropped_video_timestamps.npy')):
        create_video_timestamps(cap, args.data_dir, trigger_channel=args.trigger_channel, trigger_value=abs(args.trigger_value))

    if args.crop:
        x_start = int(args.x_start)
        x_end = int(args.x_end)
        y_start = int(args.y_start)
        y_end = int(args.y_end)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if x_end == -1:
            x_end = frame_width - 1
        if y_end == -1:
            y_end = frame_height - 1

        if x_start < 0 or x_start >= frame_width:
            raise ValueError(f'Invalid x_start value: {x_start}')
        if x_end < 0 or x_end >= frame_width or x_end <= x_start:
            raise ValueError(f'Invalid x_end value: {x_end}')
        if y_start < 0 or y_start >= frame_height:
            raise ValueError(f'Invalid y_start value: {y_start}')
        if y_end < 0 or y_end >= frame_height or y_end <= y_start:
            raise ValueError(f'Invalid y_end value: {y_end}')

        out_dir = join(args.data_dir, 'experiments', args.experiment_name)
        if os.path.exists(out_dir):
            raise ValueError(f'Output directory {out_dir} already exists.')
        os.makedirs(out_dir, exist_ok=True)

        crop_video(cap, args.data_dir, out_dir, start_frame=args.start_frame, end_frame=args.end_frame, x_start=x_start,
                   x_end=x_end, y_start=y_start,
                   y_end=y_end)
        align_emg(args.data_dir, out_dir)

    else:
        show_frame(cap, frame_number=args.frame_number)
        # run video_gui.py:
        import subprocess
        subprocess.run(['python', 'video_gui.py', '--file', join(args.data_dir, 'video_cropped.mp4')])

